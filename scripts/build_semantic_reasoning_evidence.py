#!/usr/bin/env python3
"""Build the audited semantic-label evidence package.

The analysis keeps the original nine-way labels for transparency, but treats a
four-way grouping as primary because that grouping is more reliable on the
independently adjudicated pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from reveng.experiments.action_distribution_change_points import (
    jensen_shannon_bits,
)
from reveng.experiments.semantic_reasoning_classification import (
    SEMANTIC_LABELS,
    validate_annotation_rows,
)


ROOT = Path("outputs/hypothesis_tests/semantic_reasoning_classification_v1")
MATCHED = ROOT.parent / "action_distribution_cpd_beast_v1"
ACTIVATION = Path(
    "outputs/experiment1_activation_monitor/" "gpt_oss_local_sentence_matched46_v1"
)
ATTENTION = ROOT.parent / "prefix_action_attention_v1"

BLUE = "#1769aa"
LIGHT_BLUE = "#9ecae1"
DARK = "#17324d"
GRAY = "#66788a"
GRID = "#d9e3ec"

BROAD_LABELS = (
    "state_readout",
    "route_deliberation",
    "summary_or_restatement",
    "other",
)

BROAD_LABEL_MAP = {
    "state_reconstruction": "state_readout",
    "route_planning": "route_deliberation",
    "verification": "route_deliberation",
    "correction": "route_deliberation",
    "new_inference": "route_deliberation",
    "consolidation": "summary_or_restatement",
    "restatement": "summary_or_restatement",
    "procedural_continuation": "other",
    "unclear": "other",
}

SEMANTIC_ONLY_COLUMNS = (
    "sentence_id",
    "environment_id",
    "environment_step",
    "sentence_number",
    "context_before",
    "target_sentence",
    "target_sentence_characters",
    "primary_label",
    "broad_label",
    "confidence",
    "rationale",
    "explicitly_revises_prior_reasoning",
    "evaluates_prior_route_or_claim",
    "repeats_prior_content",
    "introduces_new_information_or_plan",
    "adjudication_applied",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT / "gpt_oss_local_annotations_v2.csv",
    )
    parser.add_argument(
        "--annotation-key",
        type=Path,
        default=ROOT / "annotation_key.csv",
    )
    parser.add_argument(
        "--pilot-annotations",
        type=Path,
        default=ROOT / "gpt_oss_local_pilot_v2.csv",
    )
    parser.add_argument(
        "--pilot-adjudication",
        type=Path,
        default=ROOT / "pilot_adjudicated_labels.csv",
    )
    parser.add_argument(
        "--pilot-duplicate-key",
        type=Path,
        default=ROOT / "pilot_duplicate_key.csv",
    )
    parser.add_argument(
        "--adjudication-overrides",
        type=Path,
        default=ROOT / "low_confidence_adjudication_overrides.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "final_analysis",
    )
    parser.add_argument(
        "--positions",
        type=Path,
        default=MATCHED / "position_change_probabilities.csv",
    )
    parser.add_argument(
        "--sentences",
        type=Path,
        default=Path(
            "data/behavioral_probes/doorkey_chunking_validation/sentences.csv"
        ),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--permutation-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def configure_style() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update(
        {
            "font.family": "Arial" if "Arial" in available else "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": "#9aabb8",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def normalize_strings(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include="object"):
        result[column] = result[column].fillna("").astype(str).str.strip()
    return result


def semantic_only_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Return the compact annotation product without experimental measurements."""
    rows = add_readable_sentence_identifiers(rows)
    missing = [column for column in SEMANTIC_ONLY_COLUMNS if column not in rows]
    if missing:
        raise ValueError(
            f"Cannot build compact semantic label table; missing columns: {missing}"
        )
    compact = rows.loc[:, SEMANTIC_ONLY_COLUMNS].copy()
    if compact["sentence_id"].duplicated().any():
        raise ValueError(
            "Compact semantic label table contains duplicate sentence IDs"
        )
    return compact


def add_readable_sentence_identifiers(rows: pd.DataFrame) -> pd.DataFrame:
    """Add a readable ID and its components without replacing internal join keys."""
    required = {"trajectory_id", "step_index", "position_index"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(
            f"Cannot build readable sentence identifiers; missing columns: {missing}"
        )
    result = rows.copy()
    result["environment_id"] = result["trajectory_id"].astype(str).str.extract(
        r"(keepdoor_\d+)$", expand=False
    )
    if result["environment_id"].isna().any():
        examples = result.loc[
            result["environment_id"].isna(), "trajectory_id"
        ].astype(str)
        raise ValueError(
            "Cannot extract environment ID from trajectory IDs: "
            + ", ".join(examples.head(3))
        )
    result["environment_step"] = result["step_index"].astype(int)
    result["sentence_number"] = result["position_index"].astype(int)
    result["sentence_id"] = [
        f"{environment_id}__step_{step:03d}__sentence_{sentence:03d}"
        for environment_id, step, sentence in zip(
            result["environment_id"],
            result["environment_step"],
            result["sentence_number"],
            strict=True,
        )
    ]
    return result


def build_all_sentence_inventory(
    positions: pd.DataFrame,
    sentences: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Inventory every reasoning sentence in the matched cohort.

    Semantic fields are populated only for the experimentally selected and
    classified subset.
    """
    position_columns = {
        "example_id",
        "trajectory_id",
        "step_index",
        "position_index",
    }
    sentence_columns = {"trace_id", "sentence_id", "kind", "text"}
    missing_positions = sorted(position_columns - set(positions.columns))
    missing_sentences = sorted(sentence_columns - set(sentences.columns))
    if missing_positions or missing_sentences:
        raise ValueError(
            "Cannot build all-sentence inventory; "
            f"missing position columns={missing_positions}, "
            f"missing sentence columns={missing_sentences}"
        )
    cohort = positions[
        positions["position_index"].astype(int).gt(0)
    ][list(position_columns)].copy()
    reasoning = sentences[sentences["kind"].eq("reasoning")][
        ["trace_id", "sentence_id", "text"]
    ].copy()
    reasoning["position_index"] = reasoning["sentence_id"].astype(int) + 1
    inventory = cohort.merge(
        reasoning,
        left_on=["example_id", "position_index"],
        right_on=["trace_id", "position_index"],
        how="left",
        validate="one_to_one",
    )
    if inventory["text"].isna().any():
        raise ValueError("Some matched-cohort positions lack canonical sentence text")
    inventory = add_readable_sentence_identifiers(inventory)
    label_columns = [
        "sentence_id",
        "primary_label",
        "broad_label",
        "confidence",
    ]
    inventory = inventory.merge(
        labels[label_columns],
        on="sentence_id",
        how="left",
        validate="one_to_one",
    )
    inventory["semantic_label_available"] = inventory["primary_label"].notna()
    inventory = inventory.rename(columns={"text": "target_sentence"})
    output_columns = [
        "sentence_id",
        "environment_id",
        "environment_step",
        "sentence_number",
        "target_sentence",
        "semantic_label_available",
        "primary_label",
        "broad_label",
        "confidence",
    ]
    result = inventory[output_columns].sort_values(
        ["environment_id", "environment_step", "sentence_number"]
    )
    if result["sentence_id"].duplicated().any():
        raise ValueError("All-sentence inventory contains duplicate sentence IDs")
    return result.reset_index(drop=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cohen_kappa(left: pd.Series, right: pd.Series) -> float:
    labels = sorted(set(left.astype(str)) | set(right.astype(str)))
    matrix = pd.crosstab(left, right).reindex(
        index=labels, columns=labels, fill_value=0
    )
    total = float(matrix.to_numpy().sum())
    if total == 0:
        return float("nan")
    observed = float(np.trace(matrix.to_numpy()) / total)
    row = matrix.sum(axis=1).to_numpy(dtype=float) / total
    column = matrix.sum(axis=0).to_numpy(dtype=float) / total
    expected = float(np.dot(row, column))
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def reliability_outputs(
    pilot: pd.DataFrame,
    adjudicated: pd.DataFrame,
    duplicate_key: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    unique = pilot.drop_duplicates("annotation_id").merge(
        adjudicated,
        on="annotation_id",
        how="inner",
        validate="one_to_one",
    )
    unique["judge_broad_label"] = unique["primary_label"].map(BROAD_LABEL_MAP)
    unique["adjudicated_broad_label"] = unique["adjudicated_label"].map(BROAD_LABEL_MAP)
    unique["fine_agrees"] = unique["primary_label"] == unique["adjudicated_label"]
    unique["broad_agrees"] = (
        unique["judge_broad_label"] == unique["adjudicated_broad_label"]
    )

    lookup = pilot.set_index("annotation_id")
    duplicate_rows: list[dict[str, object]] = []
    for row in duplicate_key.itertuples():
        original = str(row.duplicate_of_annotation_id)
        duplicate = str(row.duplicate_annotation_id)
        if original not in lookup.index or duplicate not in lookup.index:
            continue
        left = lookup.loc[original]
        right = lookup.loc[duplicate]
        duplicate_rows.append(
            {
                "original_annotation_id": original,
                "duplicate_annotation_id": duplicate,
                "original_fine_label": left["primary_label"],
                "duplicate_fine_label": right["primary_label"],
                "fine_agrees": left["primary_label"] == right["primary_label"],
                "original_broad_label": BROAD_LABEL_MAP[left["primary_label"]],
                "duplicate_broad_label": BROAD_LABEL_MAP[right["primary_label"]],
                "broad_agrees": (
                    BROAD_LABEL_MAP[left["primary_label"]]
                    == BROAD_LABEL_MAP[right["primary_label"]]
                ),
            }
        )
    duplicates = pd.DataFrame(duplicate_rows)
    summary: dict[str, float | int] = {
        "adjudicated_items": len(unique),
        "fine_exact_agreement": float(unique["fine_agrees"].mean()),
        "fine_cohen_kappa": cohen_kappa(
            unique["primary_label"], unique["adjudicated_label"]
        ),
        "broad_exact_agreement": float(unique["broad_agrees"].mean()),
        "broad_cohen_kappa": cohen_kappa(
            unique["judge_broad_label"], unique["adjudicated_broad_label"]
        ),
        "duplicate_pairs": len(duplicates),
        "duplicate_fine_agreement": (
            float(duplicates["fine_agrees"].mean()) if len(duplicates) else np.nan
        ),
        "duplicate_broad_agreement": (
            float(duplicates["broad_agrees"].mean()) if len(duplicates) else np.nan
        ),
    }
    return unique, duplicates, summary


def add_position_features(rows: pd.DataFrame) -> pd.DataFrame:
    positions = pd.read_csv(MATCHED / "position_change_probabilities.csv")
    positions = positions.sort_values(["example_id", "position_index"]).copy()
    probability_columns = ["prob_up", "prob_down", "prob_left", "prob_right"]
    previous = positions.groupby("example_id")[probability_columns].shift(1)
    positions["adjacent_js_divergence_bits"] = [
        (
            jensen_shannon_bits(
                previous_row.to_numpy(dtype=float),
                current_row.to_numpy(dtype=float),
            )
            if previous_row.notna().all()
            else np.nan
        )
        for (_, current_row), (_, previous_row) in zip(
            positions[probability_columns].iterrows(),
            previous.iterrows(),
            strict=True,
        )
    ]
    positions["previous_action_label"] = positions.groupby("example_id")[
        "action_label"
    ].shift(1)
    positions["previous_action_is_optimal"] = positions.groupby("example_id")[
        "action_is_optimal"
    ].shift(1)
    positions["recommendation_changes_here"] = positions[
        "previous_action_label"
    ].notna() & (positions["action_label"] != positions["previous_action_label"])
    positions["optimality_loss_here"] = positions["previous_action_is_optimal"].eq(
        True
    ) & positions["action_is_optimal"].eq(False)
    positions["recovery_here"] = positions["previous_action_is_optimal"].eq(
        False
    ) & positions["action_is_optimal"].eq(True)
    positions = positions.rename(
        columns={
            "posterior_change_probability": (
                "beast_posterior_change_probability_at_position"
            )
        }
    )
    keep = [
        "example_id",
        "position_index",
        "action_label",
        "action_is_optimal",
        "previous_action_label",
        "previous_action_is_optimal",
        "recommendation_changes_here",
        "optimality_loss_here",
        "recovery_here",
        "adjacent_js_divergence_bits",
        "l2_distance_from_initial_action_distribution",
        "beast_posterior_change_probability_at_position",
        *probability_columns,
    ]
    merged = rows.merge(
        positions[keep],
        on=["example_id", "position_index"],
        how="left",
        validate="one_to_one",
    )
    events = pd.read_csv(ACTIVATION / "event_rows.csv")
    event_sets = (
        events.groupby(["example_id", "event_reasoning_step_idx"])["event_type"]
        .agg(lambda values: set(values.astype(str)))
        .reset_index()
        .rename(columns={"event_reasoning_step_idx": "position_index"})
    )
    event_sets["commitment_onset_here"] = event_sets["event_type"].apply(
        lambda values: "commitment_onset" in values
    )
    event_sets["event_types_here"] = event_sets["event_type"].apply(
        lambda values: ";".join(sorted(values))
    )
    merged = merged.merge(
        event_sets[
            [
                "example_id",
                "position_index",
                "commitment_onset_here",
                "event_types_here",
            ]
        ],
        on=["example_id", "position_index"],
        how="left",
        validate="one_to_one",
    )
    merged["commitment_onset_here"] = merged["commitment_onset_here"].eq(True)
    merged["event_types_here"] = merged["event_types_here"].fillna("")

    geometry = pd.read_csv(ACTIVATION / "geometry_rows.csv")
    geometry = geometry[
        geometry["layer"].astype(int).eq(15)
        & geometry["representation"].eq("sentence_mean")
    ][
        [
            "example_id",
            "reasoning_step_idx",
            "update_norm",
            "adjacent_cosine",
            "previous_mean_cosine",
        ]
    ].rename(
        columns={"reasoning_step_idx": "position_index"}
    )
    merged = merged.merge(
        geometry,
        on=["example_id", "position_index"],
        how="left",
        validate="one_to_one",
    )
    return merged


def add_belief_features(rows: pd.DataFrame) -> pd.DataFrame:
    beliefs = pd.read_csv(ACTIVATION / "belief_rows.csv")
    beliefs = beliefs.sort_values(
        ["example_id", "question_id", "position_index"]
    ).copy()
    beliefs["answer_changes_here"] = beliefs.groupby(["example_id", "question_id"])[
        "answer_key"
    ].shift(1).notna() & (
        beliefs["answer_key"]
        != beliefs.groupby(["example_id", "question_id"])["answer_key"].shift(1)
    )
    beliefs["error_onset_here"] = beliefs["belief_is_error"].eq(True) & beliefs.groupby(
        ["example_id", "question_id"]
    )["belief_is_error"].shift(1).eq(False)
    beliefs["error_recovery_here"] = beliefs["belief_is_error"].eq(
        False
    ) & beliefs.groupby(["example_id", "question_id"])["belief_is_error"].shift(1).eq(
        True
    )
    summary = beliefs.groupby(["example_id", "position_index"], as_index=False).agg(
        mean_belief_error_rate=("belief_is_error", "mean"),
        mean_belief_entropy_bits=("entropy_bits", "mean"),
        belief_answer_changes=("answer_changes_here", "sum"),
        belief_error_onsets=("error_onset_here", "sum"),
        belief_error_recoveries=("error_recovery_here", "sum"),
    )
    return rows.merge(
        summary,
        on=["example_id", "position_index"],
        how="left",
        validate="one_to_one",
    )


def clustered_bootstrap_difference(
    rows: pd.DataFrame,
    feature: str,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    summary = (
        rows.groupby(["trajectory_id", "item_role"])["has_feature"]
        .agg(["sum", "count"])
        .unstack("item_role", fill_value=0)
        .sort_index()
    )
    event_sum = (
        summary["sum"]
        .get(
            "detected_change_point",
            pd.Series(0, index=summary.index),
        )
        .to_numpy(dtype=float)
    )
    event_count = (
        summary["count"]
        .get(
            "detected_change_point",
            pd.Series(0, index=summary.index),
        )
        .to_numpy(dtype=float)
    )
    control_sum = (
        summary["sum"]
        .get(
            "matched_non_change_sentence",
            pd.Series(0, index=summary.index),
        )
        .to_numpy(dtype=float)
    )
    control_count = (
        summary["count"]
        .get(
            "matched_non_change_sentence",
            pd.Series(0, index=summary.index),
        )
        .to_numpy(dtype=float)
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0,
        len(summary),
        size=(repeats, len(summary)),
    )
    event_rates = event_sum[draws].sum(axis=1) / event_count[draws].sum(axis=1)
    control_rates = control_sum[draws].sum(axis=1) / control_count[draws].sum(axis=1)
    values = event_rates - control_rates
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def benjamini_hochberg(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(ranked)
    result[order] = np.minimum(ranked, 1.0)
    return result


def matched_feature_summary(
    rows: pd.DataFrame,
    *,
    analysis_set: str,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    subset = (
        rows[rows["match_quality"].eq("primary")].copy()
        if analysis_set == "primary_matches"
        else rows.copy()
    )
    features: list[tuple[str, str, set[str]]] = []
    features.extend(("fine", label, {label}) for label in SEMANTIC_LABELS)
    features.extend(("broad", label, {label}) for label in BROAD_LABELS)
    features.append(
        (
            "prespecified_composite",
            "verification_or_consolidation_or_restatement",
            {"verification", "consolidation", "restatement"},
        )
    )
    output: list[dict[str, object]] = []
    for family, name, accepted in features:
        values = subset.copy()
        source = values["broad_label"] if family == "broad" else values["primary_label"]
        values["has_feature"] = source.isin(accepted)
        paired = values.pivot(
            index="pair_id", columns="item_role", values="has_feature"
        ).dropna()
        event = paired["detected_change_point"].astype(bool)
        control = paired["matched_non_change_sentence"].astype(bool)
        event_only = int((event & ~control).sum())
        control_only = int((~event & control).sum())
        discordant = event_only + control_only
        low, high = clustered_bootstrap_difference(
            values,
            "has_feature",
            repeats=repeats,
            seed=seed + len(output),
        )
        output.append(
            {
                "analysis_set": analysis_set,
                "label_family": family,
                "semantic_feature": name,
                "n_pairs": len(paired),
                "rate_at_detected_change_points": float(event.mean()),
                "rate_at_matched_non_change_sentences": float(control.mean()),
                "paired_rate_difference": float(event.mean() - control.mean()),
                "trajectory_bootstrap_ci_low": low,
                "trajectory_bootstrap_ci_high": high,
                "pairs_feature_only_at_change_point": event_only,
                "pairs_feature_only_at_comparison": control_only,
                "mcnemar_exact_p": (
                    float(binomtest(event_only, discordant, 0.5).pvalue)
                    if discordant
                    else 1.0
                ),
            }
        )
    result = pd.DataFrame(output)
    result["mcnemar_bh_q_within_set"] = benjamini_hochberg(result["mcnemar_exact_p"])
    return result


def omnibus_paired_permutation(
    rows: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> dict[str, float | int]:
    paired = rows.pivot(
        index="pair_id", columns="item_role", values="broad_label"
    ).dropna()

    def statistic(event: pd.Series, control: pd.Series) -> float:
        event_rates = event.value_counts(normalize=True).reindex(
            BROAD_LABELS, fill_value=0
        )
        control_rates = control.value_counts(normalize=True).reindex(
            BROAD_LABELS, fill_value=0
        )
        return float(0.5 * np.abs(event_rates - control_rates).sum())

    event = paired["detected_change_point"].copy()
    control = paired["matched_non_change_sentence"].copy()
    observed = statistic(event, control)
    event_codes = pd.Categorical(event, categories=BROAD_LABELS).codes
    control_codes = pd.Categorical(control, categories=BROAD_LABELS).codes
    pair_differences = np.zeros((len(paired), len(BROAD_LABELS)), dtype=float)
    pair_differences[np.arange(len(paired)), event_codes] += 1
    pair_differences[np.arange(len(paired)), control_codes] -= 1
    rng = np.random.default_rng(seed)
    signs = np.where(
        rng.random((repeats, len(paired))) < 0.5,
        -1.0,
        1.0,
    )
    rate_differences = signs @ pair_differences / len(paired)
    null = 0.5 * np.abs(rate_differences).sum(axis=1)
    return {
        "n_pairs": len(paired),
        "total_variation_distance": observed,
        "paired_permutation_p": float(
            (1 + np.count_nonzero(null >= observed)) / (1 + repeats)
        ),
        "permutation_repeats": repeats,
    }


def permuted_between_label_variance(
    frame: pd.DataFrame,
    outcome: str,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    """Weighted between-label variance and a within-trajectory permutation null."""

    labels = pd.Categorical(frame["broad_label"], categories=BROAD_LABELS).codes
    values = frame[outcome].to_numpy(dtype=float)
    overall = float(values.mean())

    def variances(code_matrix: np.ndarray) -> np.ndarray:
        result = np.zeros(len(code_matrix), dtype=float)
        for code in range(len(BROAD_LABELS)):
            members = code_matrix == code
            counts = members.sum(axis=1)
            sums = members @ values
            means = np.divide(
                sums,
                counts,
                out=np.full(len(code_matrix), overall, dtype=float),
                where=counts > 0,
            )
            result += counts * np.square(means - overall)
        return result / len(values)

    observed = float(variances(labels[None, :])[0])
    rng = np.random.default_rng(seed)
    permutations = np.broadcast_to(
        labels,
        (repeats, len(labels)),
    ).copy()
    for indices in frame.groupby("trajectory_id").indices.values():
        indices = np.asarray(indices, dtype=int)
        orders = np.argsort(
            rng.random((repeats, len(indices))),
            axis=1,
        )
        permutations[:, indices] = labels[indices][orders]
    return observed, variances(permutations)


def make_model(
    numeric: list[str],
    categorical: list[str],
) -> Pipeline:
    transform = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("features", transform),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def grouped_model_comparison(
    rows: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = rows[rows["match_quality"].eq("primary")].copy()
    frame["target"] = frame["item_role"].eq("detected_change_point").astype(int)
    numeric_baseline = ["reasoning_progress", "target_sentence_characters"]
    numeric_activation = ["adjacent_cosine", "previous_mean_cosine", "update_norm"]
    for column in [*numeric_baseline, *numeric_activation]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].fillna(frame[column].median())
    specifications = {
        "matched_design_baseline": (numeric_baseline, []),
        "baseline_plus_semantics": (numeric_baseline, ["broad_label"]),
        "baseline_plus_activation": (
            [*numeric_baseline, *numeric_activation],
            [],
        ),
        "baseline_plus_semantics_and_activation": (
            [*numeric_baseline, *numeric_activation],
            ["broad_label"],
        ),
    }
    groups = frame["trajectory_id"].astype(str)
    folds = min(5, groups.nunique())
    splitter = GroupKFold(n_splits=folds)
    prediction_rows: list[pd.DataFrame] = []
    for name, (numeric, categorical) in specifications.items():
        predictions = np.full(len(frame), np.nan)
        fold_ids = np.full(len(frame), -1, dtype=int)
        for fold, (train, test) in enumerate(
            splitter.split(frame, frame["target"], groups)
        ):
            model = make_model(numeric, categorical)
            model.fit(frame.iloc[train], frame.iloc[train]["target"])
            predictions[test] = model.predict_proba(frame.iloc[test])[:, 1]
            fold_ids[test] = fold
        part = frame[
            [
                "annotation_id",
                "pair_id",
                "trajectory_id",
                "target",
                "broad_label",
            ]
        ].copy()
        part["model"] = name
        part["fold"] = fold_ids
        part["predicted_probability"] = predictions
        prediction_rows.append(part)
    predictions = pd.concat(prediction_rows, ignore_index=True)

    summary_rows: list[dict[str, object]] = []
    baseline = predictions[
        predictions["model"].eq("matched_design_baseline")
    ].set_index("annotation_id")
    trajectory_ids = np.asarray(
        sorted(frame["trajectory_id"].unique()),
        dtype=object,
    )
    trajectory_code = pd.Categorical(
        frame["trajectory_id"],
        categories=trajectory_ids,
    ).codes
    rng = np.random.default_rng(seed)
    bootstrap_counts = rng.multinomial(
        len(trajectory_ids),
        np.repeat(1 / len(trajectory_ids), len(trajectory_ids)),
        size=repeats,
    )
    bootstrap_row_weights = bootstrap_counts[:, trajectory_code]
    for name, part in predictions.groupby("model", sort=False):
        auc = float(roc_auc_score(part["target"], part["predicted_probability"]))
        loss = float(log_loss(part["target"], part["predicted_probability"]))
        current = part.set_index("annotation_id")
        auc_deltas: list[float] = []
        loss_deltas: list[float] = []
        if name != "matched_design_baseline":
            y = frame["target"].to_numpy(dtype=int)
            current_probability = current.loc[
                frame["annotation_id"], "predicted_probability"
            ].to_numpy(dtype=float)
            baseline_probability = baseline.loc[
                frame["annotation_id"], "predicted_probability"
            ].to_numpy(dtype=float)
            clipped_current = np.clip(current_probability, 1e-15, 1 - 1e-15)
            clipped_baseline = np.clip(baseline_probability, 1e-15, 1 - 1e-15)
            current_row_loss = -(
                y * np.log(clipped_current) + (1 - y) * np.log(1 - clipped_current)
            )
            baseline_row_loss = -(
                y * np.log(clipped_baseline) + (1 - y) * np.log(1 - clipped_baseline)
            )
            for weights in bootstrap_row_weights:
                if weights[y == 1].sum() == 0 or weights[y == 0].sum() == 0:
                    continue
                auc_deltas.append(
                    float(
                        roc_auc_score(
                            y,
                            current_probability,
                            sample_weight=weights,
                        )
                        - roc_auc_score(
                            y,
                            baseline_probability,
                            sample_weight=weights,
                        )
                    )
                )
                loss_deltas.append(
                    float(
                        np.average(
                            baseline_row_loss - current_row_loss,
                            weights=weights,
                        )
                    )
                )
        summary_rows.append(
            {
                "model": name,
                "n_rows": len(part),
                "trajectories": frame["trajectory_id"].nunique(),
                "grouped_folds": folds,
                "held_out_auroc": auc,
                "held_out_log_loss": loss,
                "delta_auroc_vs_baseline": (
                    auc
                    - float(
                        roc_auc_score(
                            baseline["target"],
                            baseline["predicted_probability"],
                        )
                    )
                ),
                "log_loss_improvement_vs_baseline": (
                    float(
                        log_loss(
                            baseline["target"],
                            baseline["predicted_probability"],
                        )
                    )
                    - loss
                ),
                "delta_auroc_ci_low": (
                    float(np.quantile(auc_deltas, 0.025)) if auc_deltas else np.nan
                ),
                "delta_auroc_ci_high": (
                    float(np.quantile(auc_deltas, 0.975)) if auc_deltas else np.nan
                ),
                "log_loss_improvement_ci_low": (
                    float(np.quantile(loss_deltas, 0.025)) if loss_deltas else np.nan
                ),
                "log_loss_improvement_ci_high": (
                    float(np.quantile(loss_deltas, 0.975)) if loss_deltas else np.nan
                ),
            }
        )
    return pd.DataFrame(summary_rows), predictions


SEMANTIC_OUTCOMES = {
    "recommendation_changes_here": "binary",
    "optimality_loss_here": "binary",
    "recovery_here": "binary",
    "commitment_onset_here": "binary",
    "adjacent_js_divergence_bits": "continuous",
    "posterior_change_probability": "continuous",
    "update_norm": "continuous",
    "adjacent_cosine": "continuous",
    "previous_mean_cosine": "continuous",
    "mean_belief_error_rate": "continuous",
    "mean_belief_entropy_bits": "continuous",
    "belief_answer_changes": "continuous",
    "belief_error_onsets": "continuous",
    "belief_error_recoveries": "continuous",
}


def trajectory_bootstrap_mean_interval(
    rows: pd.DataFrame,
    outcome: str,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    frame = rows[["trajectory_id", outcome]].dropna()
    summary = frame.groupby("trajectory_id")[outcome].agg(["sum", "count"]).sort_index()
    if not len(summary):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0,
        len(summary),
        size=(repeats, len(summary)),
    )
    sums = summary["sum"].to_numpy(dtype=float)
    counts = summary["count"].to_numpy(dtype=float)
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def semantic_event_summary(
    rows: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    change_points = rows[
        rows["item_role"].eq("detected_change_point")
        & rows["match_quality"].eq("primary")
    ].copy()
    output: list[dict[str, object]] = []
    for label_index, label in enumerate(BROAD_LABELS):
        group = change_points[change_points["broad_label"].eq(label)]
        for outcome_index, (outcome, outcome_type) in enumerate(
            SEMANTIC_OUTCOMES.items()
        ):
            values = pd.to_numeric(group[outcome], errors="coerce").dropna()
            low, high = trajectory_bootstrap_mean_interval(
                group,
                outcome,
                repeats=repeats,
                seed=seed + label_index * len(SEMANTIC_OUTCOMES) + outcome_index,
            )
            output.append(
                {
                    "broad_label": label,
                    "outcome": outcome,
                    "outcome_type": outcome_type,
                    "sentences": len(group),
                    "observed_values": len(values),
                    "trajectories": group.loc[
                        pd.to_numeric(group[outcome], errors="coerce").notna(),
                        "trajectory_id",
                    ].nunique(),
                    "mean_or_event_rate": (
                        float(values.mean()) if len(values) else np.nan
                    ),
                    "trajectory_bootstrap_ci_low": low,
                    "trajectory_bootstrap_ci_high": high,
                    "event_count": (
                        int(values.sum()) if outcome_type == "binary" else np.nan
                    ),
                }
            )
    return pd.DataFrame(output)


def semantic_outcome_permutation_tests(
    rows: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Test whether an outcome varies by semantic class at change points.

    Labels are shuffled only within trajectories, preserving trajectory-level
    label composition and all outcome autocorrelation. The statistic is the
    label-frequency-weighted between-class variance of outcome means.
    """

    change_points = rows[
        rows["item_role"].eq("detected_change_point")
        & rows["match_quality"].eq("primary")
    ].copy()
    output: list[dict[str, object]] = []
    for outcome, outcome_type in SEMANTIC_OUTCOMES.items():
        frame = change_points[["trajectory_id", "broad_label", outcome]].copy()
        frame[outcome] = pd.to_numeric(frame[outcome], errors="coerce")
        frame = frame.dropna(subset=[outcome]).reset_index(drop=True)

        observed, null = permuted_between_label_variance(
            frame,
            outcome,
            repeats=repeats,
            seed=seed + len(output),
        )
        output.append(
            {
                "outcome": outcome,
                "outcome_type": outcome_type,
                "sentences": len(frame),
                "trajectories": frame["trajectory_id"].nunique(),
                "semantic_groups_observed": frame["broad_label"].nunique(),
                "weighted_between_group_variance": observed,
                "within_trajectory_permutation_p": float(
                    (1 + np.count_nonzero(null >= observed)) / (1 + repeats)
                ),
                "permutation_repeats": repeats,
            }
        )
    result = pd.DataFrame(output)
    result["bh_q_across_outcomes"] = benjamini_hochberg(
        result["within_trajectory_permutation_p"]
    )
    return result


def attention_by_semantics(
    rows: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    attention = pd.read_csv(ATTENTION / "paired_attention_differences.csv")
    attention = attention[
        attention["layer"].astype(int).eq(15)
        & attention["region"].eq("newest_reasoning_sentence")
        & attention["metric"].eq("mean_attention_per_token")
        & attention["match_quality"].eq("primary")
        & attention["kernel_valid_isolated_replayed_recommendation_change"].eq(True)
    ].copy()
    cp_labels = rows[rows["item_role"].eq("detected_change_point")][
        ["pair_id", "broad_label", "primary_label"]
    ]
    attention = attention.merge(
        cp_labels,
        on="pair_id",
        how="inner",
        validate="many_to_one",
    )
    output: list[dict[str, object]] = []
    for label in BROAD_LABELS:
        group = attention[attention["broad_label"].eq(label)]
        trajectory_means = group.groupby("trajectory_id")[
            "difference_in_differences"
        ].mean()
        output.append(
            {
                "broad_label": label,
                "pairs": len(group),
                "trajectories": group["trajectory_id"].nunique(),
                "mean_attention_difference_in_differences": group[
                    "difference_in_differences"
                ].mean(),
                "trajectory_mean_standard_error": (
                    trajectory_means.std(ddof=1) / np.sqrt(len(trajectory_means))
                    if len(trajectory_means) > 1
                    else np.nan
                ),
            }
        )
    summary = pd.DataFrame(output)
    frame = (
        attention[["trajectory_id", "broad_label", "difference_in_differences"]]
        .dropna()
        .reset_index(drop=True)
    )

    observed, null = permuted_between_label_variance(
        frame.rename(columns={"difference_in_differences": "_attention_outcome"}),
        "_attention_outcome",
        repeats=repeats,
        seed=seed,
    )
    test = {
        "attention_rows": len(frame),
        "trajectories": int(frame["trajectory_id"].nunique()),
        "semantic_groups_observed": int(frame["broad_label"].nunique()),
        "weighted_between_group_variance": observed,
        "within_trajectory_permutation_p": float(
            (1 + np.count_nonzero(null >= observed)) / (1 + repeats)
        ),
        "permutation_repeats": repeats,
    }
    return summary, test


def representative_examples(rows: pd.DataFrame) -> pd.DataFrame:
    change_points = rows[
        rows["item_role"].eq("detected_change_point")
        & rows["match_quality"].eq("primary")
    ].copy()
    change_points["confidence_rank"] = change_points["confidence"].map(
        {"low": 0, "medium": 1, "high": 2}
    )
    change_points["event_priority"] = (
        3 * change_points["optimality_loss_here"].fillna(False).astype(int)
        + 2 * change_points["recovery_here"].fillna(False).astype(int)
        + change_points["recommendation_changes_here"].fillna(False).astype(int)
    )
    selected: list[pd.DataFrame] = []
    for label in BROAD_LABELS:
        group = change_points[change_points["broad_label"].eq(label)].sort_values(
            [
                "event_priority",
                "confidence_rank",
                "posterior_change_probability",
                "target_sentence_characters",
            ],
            ascending=[False, False, False, True],
        )
        selected.append(group.head(3))
    examples = pd.concat(selected, ignore_index=True)
    controls = rows[rows["item_role"].eq("matched_non_change_sentence")][
        [
            "pair_id",
            "target_sentence",
            "primary_label",
            "broad_label",
            "confidence",
        ]
    ].rename(
        columns={
            "target_sentence": "matched_control_sentence",
            "primary_label": "matched_control_primary_label",
            "broad_label": "matched_control_broad_label",
            "confidence": "matched_control_confidence",
        }
    )
    examples = examples.merge(
        controls,
        on="pair_id",
        how="left",
        validate="one_to_one",
    )
    examples["context_excerpt"] = examples["context_before"].apply(
        lambda value: " / ".join(str(value).splitlines()[-3:])
    )
    return examples[
        [
            "annotation_id",
            "pair_id",
            "example_id",
            "position_index",
            "context_excerpt",
            "target_sentence",
            "primary_label",
            "broad_label",
            "confidence",
            "rationale",
            "matched_control_sentence",
            "matched_control_primary_label",
            "matched_control_broad_label",
            "matched_control_confidence",
            "previous_action_label",
            "action_label",
            "previous_action_is_optimal",
            "action_is_optimal",
            "recommendation_changes_here",
            "optimality_loss_here",
            "recovery_here",
            "commitment_onset_here",
            "posterior_change_probability",
            "adjacent_js_divergence_bits",
        ]
    ]


def plot_matched_rates(summary: pd.DataFrame, path: Path) -> None:
    frame = summary[
        summary["analysis_set"].eq("primary_matches")
        & summary["label_family"].eq("broad")
    ].copy()
    frame["semantic_feature"] = pd.Categorical(
        frame["semantic_feature"], categories=BROAD_LABELS, ordered=True
    )
    frame = frame.sort_values("semantic_feature")
    y = np.arange(len(frame))
    offset = 0.12
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.scatter(
        frame["rate_at_detected_change_points"],
        y - offset,
        color=BLUE,
        s=48,
        label="Detected change-point sentence",
        zorder=3,
    )
    ax.scatter(
        frame["rate_at_matched_non_change_sentences"],
        y + offset,
        color=GRAY,
        s=42,
        label="Same-state matched sentence",
        zorder=3,
    )
    for index, row in enumerate(frame.itertuples()):
        ax.plot(
            [
                row.rate_at_matched_non_change_sentences,
                row.rate_at_detected_change_points,
            ],
            [index + offset, index - offset],
            color=LIGHT_BLUE,
            linewidth=1.4,
            zorder=1,
        )
        ax.text(
            max(
                row.rate_at_detected_change_points,
                row.rate_at_matched_non_change_sentences,
            )
            + 0.015,
            index,
            f"{row.paired_rate_difference:+.1%}",
            va="center",
            fontsize=9,
            color=DARK,
        )
    labels = [
        str(value).replace("_", " ").title() for value in frame["semantic_feature"]
    ]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(
        0,
        max(
            0.65,
            frame[
                [
                    "rate_at_detected_change_points",
                    "rate_at_matched_non_change_sentences",
                ]
            ]
            .to_numpy()
            .max()
            + 0.12,
        ),
    )
    ax.set_xlabel("Fraction of sentences")
    ax.set_ylabel("Calibrated model-judge semantic function")
    ax.set_title(
        "Reasoning Functions at Retrospective Action-Distribution Change Points\n"
        "152 same-state pairs; matched on state, reasoning progress, and sentence length"
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(summary: pd.DataFrame, path: Path) -> None:
    frame = summary[~summary["model"].eq("matched_design_baseline")].copy()
    order = [
        "baseline_plus_semantics",
        "baseline_plus_activation",
        "baseline_plus_semantics_and_activation",
    ]
    labels = [
        "Add semantic function",
        "Add activation geometry",
        "Add semantics and geometry",
    ]
    frame = frame.set_index("model").loc[order].reset_index()
    y = np.arange(len(frame))
    values = frame["delta_auroc_vs_baseline"].to_numpy()
    low = frame["delta_auroc_ci_low"].to_numpy()
    high = frame["delta_auroc_ci_high"].to_numpy()
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    colors = [BLUE, LIGHT_BLUE, DARK]
    for index, (value, lower, upper) in enumerate(zip(values, low, high, strict=True)):
        ax.errorbar(
            value,
            index,
            xerr=[[value - lower], [upper - value]],
            fmt="o",
            color=colors[index],
            ecolor=colors[index],
            elinewidth=2.2,
            capsize=5,
            markersize=7,
        )
        ax.text(
            upper + 0.004,
            index,
            f"{value:+.3f} [{lower:+.3f}, {upper:+.3f}]",
            va="center",
            color=DARK,
            fontsize=9,
        )
    ax.axvline(0, color=GRAY, linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(min(-0.03, low.min() - 0.015), max(0.15, high.max() + 0.07))
    ax.set_xlabel("Change in trajectory-held-out AUROC vs progress + length")
    ax.set_ylabel("Features added to matched-design baseline")
    ax.set_title(
        "Incremental Association With Retrospective Change-Point Status\n"
        "Points are estimates; bars are trajectory-bootstrap 95% intervals"
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output: Path,
    rows: pd.DataFrame,
    reliability: dict[str, float | int],
    matched: pd.DataFrame,
    omnibus: dict[str, float | int],
    models: pd.DataFrame,
    events: pd.DataFrame,
    outcome_tests: pd.DataFrame,
    attention: pd.DataFrame,
    attention_test: dict[str, float | int],
    adjudication_sensitivity: dict[str, float | int],
) -> None:
    primary_broad = matched[
        matched["analysis_set"].eq("primary_matches")
        & matched["label_family"].eq("broad")
    ].sort_values("paired_rate_difference", ascending=False)
    semantic_model = models[models["model"].eq("baseline_plus_semantics")].iloc[0]
    combined_model = models[
        models["model"].eq("baseline_plus_semantics_and_activation")
    ].iloc[0]
    entropy_test = outcome_tests[
        outcome_tests["outcome"].eq("mean_belief_entropy_bits")
    ].iloc[0]
    entropy_profiles = events[
        events["outcome"].eq("mean_belief_entropy_bits")
    ].set_index("broad_label")
    exact_event_tests = outcome_tests[outcome_tests["outcome_type"].eq("binary")]
    lines = [
        "# Audited Semantic Reasoning Classification",
        "",
        "## Scope and reliability",
        "",
        f"- Fully labeled blinded items: {len(rows)} ({rows['pair_id'].nunique()} matched pairs).",
        f"- Low-confidence labels explicitly adjudicated: {int(rows['adjudication_applied'].sum())}.",
        f"- Primary close matches: {rows[rows['match_quality'].eq('primary')]['pair_id'].nunique()} pairs.",
        f"- Apparent fine-label calibration agreement: {reliability['fine_exact_agreement']:.1%}; Cohen's kappa {reliability['fine_cohen_kappa']:.3f}.",
        f"- Apparent four-way calibration agreement: {reliability['broad_exact_agreement']:.1%}; Cohen's kappa {reliability['broad_cohen_kappa']:.3f}.",
        f"- Hidden-duplicate agreement: fine {reliability['duplicate_fine_agreement']:.1%}; four-way {reliability['duplicate_broad_agreement']:.1%}.",
        "",
        "The nine-way labels are retained for transparency and examples, but the",
        "four-way grouping is the primary inferential taxonomy. The reliability",
        "sample is small and AI-adjudicated, and the prompt was calibrated after",
        "pilot inspection. These are reproducible judge labels, not an independent",
        "human reliability estimate or human ground truth.",
        "",
        "The primary conclusion is unchanged without the 18 low-confidence",
        "overrides: the raw-judge four-way omnibus test gives "
        f"p={adjudication_sensitivity['raw_judge_omnibus_p']:.3f} versus "
        f"p={adjudication_sensitivity['adjudicated_omnibus_p']:.3f} after "
        "adjudication, and the semantics-only AUROC change is "
        f"{adjudication_sensitivity['raw_judge_semantic_delta_auroc']:+.3f} "
        "versus "
        f"{adjudication_sensitivity['adjudicated_semantic_delta_auroc']:+.3f}.",
        "",
        "## Matched change-point comparison",
        "",
        f"The omnibus paired permutation test gives total-variation distance "
        f"{omnibus['total_variation_distance']:.3f} across the four semantic "
        f"categories (p={omnibus['paired_permutation_p']:.4f}; "
        f"{omnibus['n_pairs']} pairs).",
        "",
        "| Semantic function | Change points | Matched sentences | Paired difference | 95% trajectory-bootstrap interval | McNemar q |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary_broad.itertuples():
        lines.append(
            f"| {row.semantic_feature.replace('_', ' ')} | "
            f"{row.rate_at_detected_change_points:.1%} | "
            f"{row.rate_at_matched_non_change_sentences:.1%} | "
            f"{row.paired_rate_difference:+.1%} | "
            f"[{row.trajectory_bootstrap_ci_low:+.1%}, "
            f"{row.trajectory_bootstrap_ci_high:+.1%}] | "
            f"{row.mcnemar_bh_q_within_set:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Incremental association",
            "",
            f"Adding four-way semantic labels to progress and sentence length changes "
            f"trajectory-held-out AUROC by {semantic_model['delta_auroc_vs_baseline']:+.3f} "
            f"(95% trajectory-bootstrap interval "
            f"[{semantic_model['delta_auroc_ci_low']:+.3f}, "
            f"{semantic_model['delta_auroc_ci_high']:+.3f}]).",
            f"Adding both semantics and activation geometry changes AUROC by "
            f"{combined_model['delta_auroc_vs_baseline']:+.3f} "
            f"[{combined_model['delta_auroc_ci_low']:+.3f}, "
            f"{combined_model['delta_auroc_ci_high']:+.3f}].",
            "",
            "This is contemporaneous classification of retrospectively selected",
            "change points. It is not real-time prediction and cannot establish that",
            "a semantic function caused the action distribution to change.",
            "",
            "## Connections to existing analyses",
            "",
            "- `semantic_event_composition.csv` relates sentence functions to exact",
            "  recommendation changes, optimality loss, recovery, adjacent action-",
            "  distribution divergence, activation geometry, and behavioral-belief",
            "  changes, with trajectory-bootstrap intervals.",
            "- `semantic_outcome_omnibus_tests.csv` gives exploratory, within-",
            "  trajectory permutation tests for those outcome profiles.",
            "- `attention_by_semantic_function.csv` tests whether the existing",
            "  immediate-action attention effect is concentrated in a semantic class;",
            "  `attention_semantic_omnibus_test.json` gives its trajectory-preserving",
            "  permutation test.",
            "- `matched_semantic_comparisons.csv` contains both the prespecified",
            "  verification/consolidation/restatement composite and all fine-label",
            "  sensitivity analyses.",
            "",
            "## Interpretation for the central claim",
            "",
            "The primary semantic result is null-to-modest, not a clean discourse-",
            "function signature. Route deliberation is numerically more common at",
            "change points, but its interval includes zero; the four-way omnibus test",
            "is not significant; and semantics alone neither reliably improves AUROC",
            "nor held-out log loss. None of the exact recommendation-change,",
            "optimality-loss, recovery, or commitment associations survives correction",
            f"(smallest q={exact_event_tests['bh_q_across_outcomes'].min():.3f}).",
            "",
            "The one corrected exploratory association is behavioral-belief entropy",
            f"(within-trajectory permutation q={entropy_test['bh_q_across_outcomes']:.3f}). "
            "Mean entropy is highest during route deliberation "
            f"({entropy_profiles.loc['route_deliberation', 'mean_or_event_rate']:.3f} bits), "
            "versus state readout "
            f"({entropy_profiles.loc['state_readout', 'mean_or_event_rate']:.3f}), "
            "summary or restatement "
            f"({entropy_profiles.loc['summary_or_restatement', 'mean_or_event_rate']:.3f}), "
            "and other functions "
            f"({entropy_profiles.loc['other', 'mean_or_event_rate']:.3f}). "
            "This is a secondary association and should be replicated.",
            "",
            "The immediate-action attention difference is numerically largest for",
            "summary or restatement sentences, but its semantic heterogeneity test is",
            f"not reliable (p={attention_test['within_trajectory_permutation_p']:.3f}).",
            "",
            "Together, these results sharpen rather than replace the central claim.",
            "Action-distribution changes are behaviorally meaningful routing/timing",
            "boundaries, but they are not tied to one stable kind of verbal sentence.",
            "The paper should therefore emphasize imperfect conversion of available",
            "state/value information into action and the timing of commitment—not claim",
            "that one named discourse operation causes the action change.",
            "",
            "See `representative_examples.csv` and `EXAMPLE_AUDIT.md` before quoting",
            "individual cases.",
        ]
    )
    (output / "SEMANTIC_EVIDENCE_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figs").mkdir(exist_ok=True)
    configure_style()

    annotations = normalize_strings(pd.read_csv(args.annotations))
    validation = validate_annotation_rows(annotations, allow_partial=False)
    if len(annotations) != 320 or annotations["annotation_id"].nunique() != 320:
        raise ValueError(
            f"Expected 320 unique full annotations, got {len(annotations)} rows "
            f"and {annotations['annotation_id'].nunique()} unique IDs"
        )
    if (
        "error" in annotations
        and annotations["error"].fillna("").astype(str).str.strip().ne("").any()
    ):
        raise ValueError("Full annotations contain non-empty error fields")
    annotations["model_judge_primary_label"] = annotations["primary_label"]
    annotations["model_judge_confidence"] = annotations["confidence"]
    annotations["model_judge_rationale"] = annotations["rationale"]
    annotations["adjudication_applied"] = False
    annotations["adjudication_rationale"] = ""
    overrides = normalize_strings(pd.read_csv(args.adjudication_overrides))
    if overrides["annotation_id"].duplicated().any():
        raise ValueError("Adjudication overrides contain duplicate IDs")
    unknown = set(overrides["annotation_id"]) - set(annotations["annotation_id"])
    if unknown:
        raise ValueError(f"Adjudication overrides contain unknown IDs: {unknown}")
    invalid = set(overrides["adjudicated_label"]) - set(SEMANTIC_LABELS)
    if invalid:
        raise ValueError(f"Adjudication overrides contain invalid labels: {invalid}")
    override_lookup = overrides.set_index("annotation_id")
    for row_index, row in annotations.iterrows():
        annotation_id = str(row["annotation_id"])
        if annotation_id not in override_lookup.index:
            continue
        override = override_lookup.loc[annotation_id]
        if str(row["primary_label"]) != str(override["original_label"]):
            raise ValueError(
                f"Override original label mismatch for {annotation_id}: "
                f"{row['primary_label']} != {override['original_label']}"
            )
        annotations.at[row_index, "primary_label"] = override["adjudicated_label"]
        annotations.at[row_index, "confidence"] = override["adjudicated_confidence"]
        annotations.at[row_index, "rationale"] = override["adjudication_rationale"]
        annotations.at[row_index, "adjudication_applied"] = True
        annotations.at[row_index, "adjudication_rationale"] = override[
            "adjudication_rationale"
        ]
    validate_annotation_rows(annotations, allow_partial=False)
    annotations["broad_label"] = annotations["primary_label"].map(BROAD_LABEL_MAP)
    if annotations["broad_label"].isna().any():
        raise ValueError("Some fine labels do not map to the audited broad taxonomy")

    key = pd.read_csv(args.annotation_key)
    rows = annotations.merge(
        key.drop(columns=["target_sentence_characters"], errors="ignore"),
        on="annotation_id",
        how="inner",
        validate="one_to_one",
    )
    if len(rows) != 320:
        raise ValueError("Some full annotation IDs do not match the hidden key")
    rows["reasoning_progress"] = np.where(
        rows["item_role"].eq("detected_change_point"),
        rows["change_point_progress"],
        rows["control_progress"],
    )
    rows = add_position_features(rows)
    rows = add_belief_features(rows)
    rows = add_readable_sentence_identifiers(rows)
    readable_identifier_columns = [
        "sentence_id",
        "environment_id",
        "environment_step",
        "sentence_number",
    ]
    rows = rows[
        readable_identifier_columns
        + [column for column in rows if column not in readable_identifier_columns]
    ]
    raw_judge_rows = rows.copy()
    raw_judge_rows["primary_label"] = raw_judge_rows["model_judge_primary_label"]
    raw_judge_rows["broad_label"] = raw_judge_rows["primary_label"].map(BROAD_LABEL_MAP)

    pilot = normalize_strings(pd.read_csv(args.pilot_annotations))
    adjudicated = pd.read_csv(args.pilot_adjudication)
    duplicate_key = pd.read_csv(args.pilot_duplicate_key)
    reliability_rows, duplicate_rows, reliability = reliability_outputs(
        pilot, adjudicated, duplicate_key
    )

    comparisons = pd.concat(
        [
            matched_feature_summary(
                rows,
                analysis_set=analysis_set,
                repeats=args.bootstrap_repeats,
                seed=args.seed,
            )
            for analysis_set in ("primary_matches", "all_matches")
        ],
        ignore_index=True,
    )
    omnibus = omnibus_paired_permutation(
        rows[rows["match_quality"].eq("primary")],
        repeats=args.permutation_repeats,
        seed=args.seed,
    )
    models, model_predictions = grouped_model_comparison(
        rows,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    raw_judge_comparisons = matched_feature_summary(
        raw_judge_rows,
        analysis_set="primary_matches",
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    raw_judge_omnibus = omnibus_paired_permutation(
        raw_judge_rows[raw_judge_rows["match_quality"].eq("primary")],
        repeats=args.permutation_repeats,
        seed=args.seed,
    )
    raw_judge_models, _ = grouped_model_comparison(
        raw_judge_rows,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    adjudicated_semantic_model = models[
        models["model"].eq("baseline_plus_semantics")
    ].iloc[0]
    raw_semantic_model = raw_judge_models[
        raw_judge_models["model"].eq("baseline_plus_semantics")
    ].iloc[0]
    adjudication_sensitivity = {
        "raw_judge_omnibus_p": raw_judge_omnibus["paired_permutation_p"],
        "adjudicated_omnibus_p": omnibus["paired_permutation_p"],
        "raw_judge_total_variation_distance": raw_judge_omnibus[
            "total_variation_distance"
        ],
        "adjudicated_total_variation_distance": omnibus["total_variation_distance"],
        "raw_judge_semantic_delta_auroc": float(
            raw_semantic_model["delta_auroc_vs_baseline"]
        ),
        "adjudicated_semantic_delta_auroc": float(
            adjudicated_semantic_model["delta_auroc_vs_baseline"]
        ),
        "raw_judge_semantic_log_loss_improvement": float(
            raw_semantic_model["log_loss_improvement_vs_baseline"]
        ),
        "adjudicated_semantic_log_loss_improvement": float(
            adjudicated_semantic_model["log_loss_improvement_vs_baseline"]
        ),
    }
    events = semantic_event_summary(
        rows,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    outcome_tests = semantic_outcome_permutation_tests(
        rows,
        repeats=args.permutation_repeats,
        seed=args.seed,
    )
    attention, attention_test = attention_by_semantics(
        rows,
        repeats=args.permutation_repeats,
        seed=args.seed,
    )
    examples = representative_examples(rows)

    compact_labels = semantic_only_table(rows)
    all_sentences = build_all_sentence_inventory(
        pd.read_csv(args.positions),
        pd.read_csv(args.sentences),
        compact_labels,
    )
    compact_labels.to_csv(args.output_dir / "sentence_label_table.csv", index=False)
    all_sentences.to_csv(
        args.output_dir / "all_reasoning_sentence_inventory.csv", index=False
    )
    rows.to_csv(
        args.output_dir / "sentence_label_cpd_analysis_table.csv",
        index=False,
    )
    table_guide = """# Sentence Label Table Guide

## Recommended table

`sentence_label_table.csv` is the compact, semantics-only product. Each row is one
reasoning sentence with its preceding context, semantic function, confidence,
rationale, and four observable semantic cues. It intentionally contains no
change-point role, matched-pair key, action outcome, BEAST score, activation feature,
attention feature, or behavioral-belief measurement.

Use this table to inspect or reuse the semantic annotations without depending on the
change-point experiment. Its readable `sentence_id` is composed from the environment,
environment step, and one-based sentence number. The opaque annotation hash is retained
only in the full experimental ledger for compatibility with the original annotation run.

## All sentences in the matched cohort

`all_reasoning_sentence_inventory.csv` contains all 7,038 reasoning sentences in the
46-state matched cohort. The 320 classified sentences have populated semantic fields;
the remaining 6,718 rows have `semantic_label_available = false` and blank label fields.
This inventory makes the sampling denominator explicit but does not claim that unreviewed
sentences have been classified.

## Full experimental ledger

`sentence_label_cpd_analysis_table.csv` preserves the original full joined table used
to reproduce the matched CPD, action-event, activation, attention, and behavioral-
belief analyses. Use it only when those experimental fields are needed.

The classifications were produced from `context_before` and `target_sentence` while
the judge was blinded to CPD status and all downstream measurements. However, the 320
sentences were originally sampled as 160 CPD/matched-control pairs; the compact table
is therefore a reusable annotation table, not a population-representative corpus of
all reasoning sentences.

## Compact columns

| Column | Meaning |
|---|---|
| `sentence_id` | Readable compound identifier, such as `keepdoor_33__step_002__sentence_038`. |
| `environment_id` | DoorKey environment trajectory identifier, such as `keepdoor_33`. |
| `environment_step` | Environment step represented by this reasoning trace. |
| `sentence_number` | One-based sentence number within the trace. |
| `context_before` | Reasoning text preceding the target sentence. |
| `target_sentence` | Sentence whose discourse function was classified. |
| `target_sentence_characters` | Character length of the target sentence. |
| `primary_label` | Fine-grained semantic-function label used for examples and sensitivity analyses. |
| `broad_label` | Calibrated four-way label used for primary inference. |
| `confidence` | Judge/adjudicator confidence: high, medium, or low. |
| `rationale` | Short explanation for the final label. |
| `explicitly_revises_prior_reasoning` | Whether the sentence rejects or changes earlier reasoning. |
| `evaluates_prior_route_or_claim` | Whether it checks an earlier route or claim. |
| `repeats_prior_content` | Whether substantive content already appears in the context. |
| `introduces_new_information_or_plan` | Whether it adds a new fact, inference, route, or decision. |
| `adjudication_applied` | Whether a low-confidence model judgment was explicitly reviewed and replaced. |

These are calibrated AI-assisted labels, not independent human ground truth. Prefer
`broad_label` for aggregate analysis and inspect the context and rationale before
quoting a fine-label example.
"""
    (args.output_dir / "SENTENCE_LABEL_TABLE_GUIDE.md").write_text(table_guide)
    reliability_rows.to_csv(args.output_dir / "pilot_reliability_rows.csv", index=False)
    duplicate_rows.to_csv(
        args.output_dir / "pilot_duplicate_reliability.csv", index=False
    )
    comparisons.to_csv(
        args.output_dir / "matched_semantic_comparisons.csv", index=False
    )
    raw_judge_comparisons.to_csv(
        args.output_dir / "raw_judge_matched_semantic_comparisons.csv",
        index=False,
    )
    models.to_csv(args.output_dir / "incremental_model_comparison.csv", index=False)
    raw_judge_models.to_csv(
        args.output_dir / "raw_judge_incremental_model_comparison.csv",
        index=False,
    )
    model_predictions.to_csv(
        args.output_dir / "incremental_model_predictions.csv", index=False
    )
    events.to_csv(args.output_dir / "semantic_event_composition.csv", index=False)
    outcome_tests.to_csv(
        args.output_dir / "semantic_outcome_omnibus_tests.csv", index=False
    )
    attention.to_csv(
        args.output_dir / "attention_by_semantic_function.csv", index=False
    )
    (args.output_dir / "attention_semantic_omnibus_test.json").write_text(
        json.dumps(attention_test, indent=2, sort_keys=True) + "\n"
    )
    examples.to_csv(args.output_dir / "representative_examples.csv", index=False)
    (args.output_dir / "reliability_summary.json").write_text(
        json.dumps(reliability, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "omnibus_paired_test.json").write_text(
        json.dumps(omnibus, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "adjudication_sensitivity_summary.json").write_text(
        json.dumps(adjudication_sensitivity, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "validation_report.txt").write_text("\n".join(validation) + "\n")
    input_paths = {
        "analysis_script": Path(__file__),
        "annotations": args.annotations,
        "adjudication_overrides": args.adjudication_overrides,
        "annotation_key": args.annotation_key,
        "pilot_annotations": args.pilot_annotations,
        "pilot_adjudication": args.pilot_adjudication,
        "pilot_duplicate_key": args.pilot_duplicate_key,
        "position_change_probabilities": (
            args.positions
        ),
        "canonical_sentences": args.sentences,
        "event_rows": ACTIVATION / "event_rows.csv",
        "geometry_rows": ACTIVATION / "geometry_rows.csv",
        "belief_rows": ACTIVATION / "belief_rows.csv",
        "attention_differences": ATTENTION / "paired_attention_differences.csv",
    }
    manifest = {
        "analysis": "audited_semantic_reasoning_evidence_v1",
        "annotation_rows": len(rows),
        "matched_pairs": int(rows["pair_id"].nunique()),
        "primary_matched_pairs": int(
            rows.loc[rows["match_quality"].eq("primary"), "pair_id"].nunique()
        ),
        "trajectories": int(rows["trajectory_id"].nunique()),
        "judge": {
            column: sorted(annotations[column].dropna().astype(str).unique().tolist())
            for column in [
                "judge_model",
                "judge_revision",
                "judge_prompt_version",
                "judge_prompt_sha256",
                "judge_reasoning_effort",
                "judge_temperature",
                "judge_seed",
            ]
            if column in annotations
        },
        "primary_taxonomy": BROAD_LABEL_MAP,
        "bootstrap_repeats": args.bootstrap_repeats,
        "permutation_repeats": args.permutation_repeats,
        "random_seed": args.seed,
        "classification_scope": (
            "contemporaneous classification of retrospectively detected "
            "change-point sentences; not an online forecast"
        ),
        "table_products": {
            "compact_semantic_labels": {
                "path": str(args.output_dir / "sentence_label_table.csv"),
                "rows": len(compact_labels),
                "columns": list(compact_labels.columns),
                "contains_cpd_fields": False,
            },
            "matched_cohort_sentence_inventory": {
                "path": str(
                    args.output_dir / "all_reasoning_sentence_inventory.csv"
                ),
                "rows": len(all_sentences),
                "columns": list(all_sentences.columns),
                "semantically_labeled_rows": int(
                    all_sentences["semantic_label_available"].sum()
                ),
                "contains_cpd_fields": False,
            },
            "full_cpd_analysis_ledger": {
                "path": str(args.output_dir / "sentence_label_cpd_analysis_table.csv"),
                "rows": len(rows),
                "columns": list(rows.columns),
                "contains_cpd_fields": True,
            },
        },
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in input_paths.items()
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    plot_matched_rates(
        comparisons,
        args.output_dir / "figs" / "matched_semantic_function_rates.png",
    )
    plot_model_comparison(
        models,
        args.output_dir / "figs" / "semantic_incremental_classification.png",
    )
    write_report(
        args.output_dir,
        rows,
        reliability,
        comparisons,
        omnibus,
        models,
        events,
        outcome_tests,
        attention,
        attention_test,
        adjudication_sensitivity,
    )
    print(
        f"output={args.output_dir} annotations={len(rows)} "
        f"pairs={rows['pair_id'].nunique()} "
        f"broad_agreement={reliability['broad_exact_agreement']:.3f}"
    )


if __name__ == "__main__":
    main()
