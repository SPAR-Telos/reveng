"""Preparation and evaluation for analysis-independent sentence semantics."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from reveng.experiments.semantic_reasoning_classification import (
    ANNOTATION_COLUMNS,
    SEMANTIC_LABELS,
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
BROAD_LABELS = (
    "state_readout",
    "route_deliberation",
    "summary_or_restatement",
    "other",
)


def stable_hash(value: str, seed: int = 42, length: int = 16) -> str:
    return hashlib.sha256(f"{seed}\x1f{value}".encode()).hexdigest()[:length]


def build_general_inventory(
    inventory: pd.DataFrame,
    canonical_sentences: pd.DataFrame,
    *,
    context_sentences: int = 8,
    context_character_limit: int = 1800,
) -> pd.DataFrame:
    """Attach preceding-only context to every sentence in the cohort inventory."""
    required_inventory = {
        "sentence_id",
        "environment_id",
        "environment_step",
        "sentence_number",
        "target_sentence",
    }
    required_canonical = {"trace_id", "sentence_id", "kind", "text"}
    if missing := required_inventory - set(inventory.columns):
        raise ValueError(f"inventory is missing columns: {sorted(missing)}")
    if missing := required_canonical - set(canonical_sentences.columns):
        raise ValueError(f"canonical sentence table is missing columns: {sorted(missing)}")

    canonical = canonical_sentences.loc[
        canonical_sentences["kind"].eq("reasoning")
    ].copy()
    canonical["sentence_number"] = canonical["sentence_id"].astype(int) + 1
    canonical["environment_step"] = canonical["trace_id"].str.extract(
        r"_step_(\d+)$", expand=False
    ).astype(int)
    canonical["environment_id"] = canonical["trace_id"].str.extract(
        r"_doorkey_(.+)_step_\d+$", expand=False
    )
    cohort_pairs = set(
        zip(
            inventory["environment_id"].astype(str),
            inventory["environment_step"].astype(int),
            strict=True,
        )
    )
    canonical = canonical.loc[
        [
            (str(environment_id), int(environment_step)) in cohort_pairs
            for environment_id, environment_step in zip(
                canonical["environment_id"], canonical["environment_step"], strict=True
            )
        ]
    ].copy()
    if canonical["environment_id"].isna().any():
        raise ValueError("could not derive environment identifiers from canonical traces")

    context_lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    for trace_id, group in canonical.groupby("trace_id", sort=False):
        ordered = group.sort_values("sentence_number")
        n_sentences = len(ordered)
        history: list[tuple[int, str]] = []
        for row in ordered.itertuples(index=False):
            preceding = history[-context_sentences:]
            context = "\n".join(f"[{number}] {text}" for number, text in preceding)
            if len(context) > context_character_limit:
                context = context[-context_character_limit:]
                first_newline = context.find("\n")
                if first_newline >= 0:
                    context = "[Earlier context omitted]\n" + context[first_newline + 1 :]
            key = (
                str(row.environment_id),
                int(row.environment_step),
                int(row.sentence_number),
            )
            context_lookup[key] = {
                "example_id": str(trace_id),
                "context_before": context,
                "canonical_target_sentence": str(row.text),
                "trace_sentence_count": n_sentences,
            }
            history.append((int(row.sentence_number), str(row.text)))

    output: list[dict[str, Any]] = []
    for row in inventory.to_dict("records"):
        key = (
            str(row["environment_id"]),
            int(row["environment_step"]),
            int(row["sentence_number"]),
        )
        context = context_lookup.get(key)
        if context is None:
            raise ValueError(f"no canonical sentence for inventory key {key}")
        target = str(row["target_sentence"])
        if target != context["canonical_target_sentence"]:
            raise ValueError(f"target mismatch for {row['sentence_id']}")
        n_sentences = int(context["trace_sentence_count"])
        output.append(
            {
                "sentence_id": str(row["sentence_id"]),
                "environment_id": str(row["environment_id"]),
                "environment_step": int(row["environment_step"]),
                "sentence_number": int(row["sentence_number"]),
                "example_id": context["example_id"],
                "reasoning_progress": (int(row["sentence_number"]) - 0.5) / n_sentences,
                "context_before": context["context_before"],
                "target_sentence": target,
                "target_sentence_characters": len(target),
                "previously_labelled": bool(row.get("semantic_label_available", False)),
            }
        )
    result = pd.DataFrame(output)
    if result["sentence_id"].duplicated().any():
        raise ValueError("general inventory contains duplicate sentence IDs")
    return result


def _add_sampling_strata(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["progress_stratum"] = pd.cut(
        result["reasoning_progress"],
        [-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=("early", "early_middle", "late_middle", "late"),
    ).astype(str)
    ranked_length = result["target_sentence_characters"].rank(
        method="first", pct=True
    )
    result["length_stratum"] = pd.cut(
        ranked_length,
        [-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=("short", "medium", "long"),
    ).astype(str)
    result["sampling_stratum"] = (
        result["progress_stratum"] + "__" + result["length_stratum"]
    )
    return result


def stratified_general_sample(
    inventory: pd.DataFrame, *, sample_size: int = 240, seed: int = 42
) -> pd.DataFrame:
    """Sample unlabelled sentences across trajectories, progress, and length."""
    candidates = inventory.loc[~inventory["previously_labelled"]].copy()
    if sample_size > len(candidates):
        raise ValueError("sample size exceeds the number of unlabelled sentences")
    candidates = _add_sampling_strata(candidates)
    candidates["random_rank"] = candidates["sentence_id"].map(
        lambda value: stable_hash(str(value), seed, length=32)
    )
    selected_indices: set[int] = set()

    # Guarantee representation from every trajectory before balancing strata.
    for _, group in candidates.groupby("environment_id", sort=True):
        selected_indices.add(int(group.sort_values("random_rank").index[0]))
    if len(selected_indices) > sample_size:
        raise ValueError("sample size is too small to cover every trajectory")

    strata = sorted(candidates["sampling_stratum"].unique())
    while len(selected_indices) < sample_size:
        added = False
        for stratum in strata:
            group = candidates.loc[
                candidates["sampling_stratum"].eq(stratum)
                & ~candidates.index.isin(selected_indices)
            ].sort_values("random_rank")
            if len(group):
                selected_indices.add(int(group.index[0]))
                added = True
                if len(selected_indices) == sample_size:
                    break
        if not added:
            break
    if len(selected_indices) != sample_size:
        raise ValueError("could not construct the requested stratified sample")
    return candidates.loc[sorted(selected_indices)].drop(columns="random_rank")


def select_few_shot_examples(
    labels: pd.DataFrame, *, n_examples: int = 12, seed: int = 42
) -> pd.DataFrame:
    required = {
        "sentence_id",
        "context_before",
        "target_sentence",
        "primary_label",
        "confidence",
        *ANNOTATION_COLUMNS[:4],
        "rationale",
    }
    if missing := required - set(labels.columns):
        raise ValueError(f"label table is missing few-shot fields: {sorted(missing)}")
    candidates = labels.loc[labels["confidence"].eq("high")].copy()
    candidates["random_rank"] = candidates["sentence_id"].map(
        lambda value: stable_hash(str(value), seed, length=32)
    )
    selected: list[int] = []
    ordered_groups: dict[str, list[int]] = {}
    for label in SEMANTIC_LABELS:
        group = candidates.loc[candidates["primary_label"].eq(label)].sort_values(
            "random_rank"
        )
        ordered_groups[label] = [int(index) for index in group.index]
        if len(group):
            selected.append(int(group.index[0]))
    depth = 1
    while len(selected) < n_examples:
        added = False
        for label in SEMANTIC_LABELS:
            group_indices = ordered_groups[label]
            if depth < len(group_indices):
                selected.append(group_indices[depth])
                added = True
                if len(selected) == n_examples:
                    break
        if not added:
            break
        depth += 1
    result = candidates.loc[selected[:n_examples]].drop(columns="random_rank")
    if len(result) != n_examples:
        raise ValueError("not enough high-confidence labels for few-shot examples")
    return result.reset_index(drop=True)


def prepare_calibration_items(
    inventory: pd.DataFrame,
    existing_labels: pd.DataFrame,
    few_shots: pd.DataFrame,
    *,
    general_sample_size: int = 240,
    reference_sample_size: int = 80,
    duplicate_items: int = 12,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    general = stratified_general_sample(
        inventory, sample_size=general_sample_size, seed=seed
    ).copy()
    general["calibration_source"] = "general_random"
    general["reference_primary_label"] = ""
    general["reference_broad_label"] = ""

    references = existing_labels.loc[
        ~existing_labels["sentence_id"].isin(few_shots["sentence_id"])
    ].copy()
    references["random_rank"] = references["sentence_id"].map(
        lambda value: stable_hash(str(value), seed + 1, length=32)
    )
    chosen: list[int] = []
    for _, group in references.groupby("broad_label", sort=True):
        quota = max(1, reference_sample_size // references["broad_label"].nunique())
        chosen.extend(int(index) for index in group.sort_values("random_rank").index[:quota])
    remaining = references.loc[~references.index.isin(chosen)].sort_values("random_rank")
    chosen.extend(int(index) for index in remaining.index[: reference_sample_size - len(chosen)])
    reference = references.loc[chosen[:reference_sample_size]].copy()
    if len(reference) != reference_sample_size:
        raise ValueError("not enough reference labels for the calibration stress set")
    reference = reference.merge(
        inventory.drop(columns=["context_before", "target_sentence"], errors="ignore"),
        on="sentence_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_inventory"),
    )
    reference["calibration_source"] = "matched_reference_stress"
    reference["reference_primary_label"] = reference["primary_label"]
    reference["reference_broad_label"] = reference["broad_label"]

    common_columns = [
        "sentence_id",
        "environment_id",
        "environment_step",
        "sentence_number",
        "example_id",
        "reasoning_progress",
        "context_before",
        "target_sentence",
        "target_sentence_characters",
        "calibration_source",
        "reference_primary_label",
        "reference_broad_label",
    ]
    combined = pd.concat(
        [general[common_columns], reference[common_columns]], ignore_index=True
    )
    combined["annotation_id"] = combined["sentence_id"].map(
        lambda value: "cal_" + stable_hash(str(value), seed)
    )
    duplicate_source = combined.loc[
        combined["calibration_source"].eq("general_random")
    ].sample(n=duplicate_items, random_state=seed + 2)
    duplicates = duplicate_source.copy()
    duplicates["annotation_id"] = duplicates["annotation_id"].map(
        lambda value: "cal_" + stable_hash("repeat\x1f" + str(value), seed)
    )
    duplicates["duplicate_of_annotation_id"] = duplicate_source["annotation_id"].to_numpy()
    combined["duplicate_of_annotation_id"] = ""
    combined = pd.concat([combined, duplicates], ignore_index=True).sample(
        frac=1.0, random_state=seed + 3
    )
    combined = combined.reset_index(drop=True)

    key_columns = [
        "annotation_id",
        "sentence_id",
        "environment_id",
        "environment_step",
        "sentence_number",
        "example_id",
        "reasoning_progress",
        "calibration_source",
        "reference_primary_label",
        "reference_broad_label",
        "duplicate_of_annotation_id",
    ]
    key = combined[key_columns].copy()
    blinded = combined[
        [
            "annotation_id",
            "context_before",
            "target_sentence",
            "target_sentence_characters",
        ]
    ].copy()
    if blinded["annotation_id"].duplicated().any():
        raise ValueError("calibration annotation IDs are not unique")
    return blinded, key


def few_shots_to_json(few_shots: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "context_before",
        "target_sentence",
        *ANNOTATION_COLUMNS,
    ]
    return few_shots[columns].to_dict("records")


def select_broad_few_shot_examples(
    labels: pd.DataFrame, *, examples_per_broad_label: int = 4, seed: int = 42
) -> pd.DataFrame:
    """Choose balanced broad-label demonstrations with fine-label diversity."""
    required = {
        "sentence_id",
        "context_before",
        "target_sentence",
        "primary_label",
        "broad_label",
        "confidence",
        "rationale",
    }
    if missing := required - set(labels.columns):
        raise ValueError(f"label table is missing broad few-shot fields: {sorted(missing)}")
    candidates = labels.loc[labels["confidence"].eq("high")].copy()
    candidates["random_rank"] = candidates["sentence_id"].map(
        lambda value: stable_hash(str(value), seed + 10, length=32)
    )
    selected: list[int] = []
    for broad_label in BROAD_LABELS:
        group = candidates.loc[candidates["broad_label"].eq(broad_label)].copy()
        broad_selected: list[int] = []
        for _, fine_group in group.groupby("primary_label", sort=True):
            broad_selected.append(int(fine_group.sort_values("random_rank").index[0]))
            if len(broad_selected) == examples_per_broad_label:
                break
        remaining = group.loc[~group.index.isin(broad_selected)].sort_values("random_rank")
        broad_selected.extend(
            int(index)
            for index in remaining.index[
                : examples_per_broad_label - len(broad_selected)
            ]
        )
        if len(broad_selected) != examples_per_broad_label:
            raise ValueError(f"not enough examples for broad label {broad_label}")
        selected.extend(broad_selected)
    return candidates.loc[selected].drop(columns="random_rank").reset_index(drop=True)


def broad_few_shots_to_json(few_shots: pd.DataFrame) -> list[dict[str, Any]]:
    return few_shots[
        [
            "context_before",
            "target_sentence",
            "broad_label",
            "confidence",
            "rationale",
        ]
    ].to_dict("records")


def _macro_f1(reference: pd.Series, prediction: pd.Series, labels: list[str]) -> float:
    return float(
        f1_score(reference, prediction, labels=labels, average="macro", zero_division=0)
    )


def evaluate_candidate(
    annotations: pd.DataFrame, key: pd.DataFrame, *, model_name: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    merged = key.merge(
        annotations,
        on="annotation_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_prediction"),
    )
    predicted = merged["primary_label"].fillna("").astype(str)
    merged["predicted_broad_label"] = predicted.map(BROAD_LABEL_MAP).fillna("")
    parse_success = predicted.isin(SEMANTIC_LABELS)
    reference = merged.loc[
        merged["calibration_source"].eq("matched_reference_stress")
        & parse_success
    ].copy()
    non_duplicate = merged[merged["duplicate_of_annotation_id"].fillna("").eq("")]
    general = non_duplicate.loc[non_duplicate["calibration_source"].eq("general_random")]
    duplicate_rows = merged.loc[merged["duplicate_of_annotation_id"].fillna("").ne("")]
    original_predictions = merged.set_index("annotation_id")["primary_label"]
    duplicate_fine = [
        str(row.primary_label)
        == str(original_predictions.get(row.duplicate_of_annotation_id, ""))
        for row in duplicate_rows.itertuples()
    ]
    duplicate_broad = [
        BROAD_LABEL_MAP.get(str(row.primary_label), "")
        == BROAD_LABEL_MAP.get(
            str(original_predictions.get(row.duplicate_of_annotation_id, "")), ""
        )
        for row in duplicate_rows.itertuples()
    ]
    record: dict[str, Any] = {
        "model_name": model_name,
        "n_expected": len(key),
        "n_parsed": int(parse_success.sum()),
        "parse_success_rate": float(parse_success.mean()),
        "reference_rows_scored": len(reference),
        "reference_fine_accuracy": math.nan,
        "reference_fine_macro_f1": math.nan,
        "reference_broad_accuracy": math.nan,
        "reference_broad_macro_f1": math.nan,
        "duplicate_fine_agreement": float(np.mean(duplicate_fine)) if duplicate_fine else math.nan,
        "duplicate_broad_agreement": float(np.mean(duplicate_broad)) if duplicate_broad else math.nan,
        "general_fine_categories_predicted": int(general["primary_label"].nunique()),
        "general_broad_categories_predicted": int(general["predicted_broad_label"].nunique()),
    }
    if len(reference):
        record.update(
            {
                "reference_fine_accuracy": float(
                    accuracy_score(reference["reference_primary_label"], reference["primary_label"])
                ),
                "reference_fine_macro_f1": _macro_f1(
                    reference["reference_primary_label"],
                    reference["primary_label"],
                    list(SEMANTIC_LABELS),
                ),
                "reference_broad_accuracy": float(
                    accuracy_score(
                        reference["reference_broad_label"],
                        reference["predicted_broad_label"],
                    )
                ),
                "reference_broad_macro_f1": _macro_f1(
                    reference["reference_broad_label"],
                    reference["predicted_broad_label"],
                    list(BROAD_LABELS),
                ),
            }
        )
    return record, merged


def compare_candidates(
    evaluations: Mapping[str, pd.DataFrame], key: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    merged_by_name: dict[str, pd.DataFrame] = {}
    for name, annotations in evaluations.items():
        summary, merged = evaluate_candidate(annotations, key, model_name=name)
        summaries.append(summary)
        merged_by_name[name] = merged
    summary_frame = pd.DataFrame(summaries)
    disagreements = pd.DataFrame()
    gate: dict[str, Any] = {"status": "incomplete", "smaller_model_passes": False}
    if "llama_3_2_1b" in merged_by_name:
        small = summary_frame.set_index("model_name").loc["llama_3_2_1b"]
        absolute_checks = {
            "parse_success_at_least_99_percent": small["parse_success_rate"] >= 0.99,
            "reference_broad_accuracy_at_least_75_percent": small[
                "reference_broad_accuracy"
            ]
            >= 0.75,
            "reference_fine_accuracy_at_least_50_percent": small[
                "reference_fine_accuracy"
            ]
            >= 0.50,
            "duplicate_broad_agreement_at_least_95_percent": small[
                "duplicate_broad_agreement"
            ]
            >= 0.95,
            "all_four_broad_categories_predicted": small[
                "general_broad_categories_predicted"
            ]
            == 4,
        }
        gate = {
            "status": (
                "incomplete"
                if all(absolute_checks.values())
                else "smaller_model_rejected"
            ),
            "smaller_model_passes": False,
            "checks": {key: bool(value) for key, value in absolute_checks.items()},
            "pending_checks": [
                "within_three_points_of_teacher_broad_accuracy",
                "general_broad_agreement_at_least_80_percent",
            ],
            "manual_disagreement_review_required": True,
        }
    if {"llama_3_2_1b", "gpt_oss_20b"} <= set(merged_by_name):
        columns = ["annotation_id", "primary_label", "predicted_broad_label", "confidence"]
        left = merged_by_name["llama_3_2_1b"][columns].rename(
            columns={column: f"llama_{column}" for column in columns if column != "annotation_id"}
        )
        right = merged_by_name["gpt_oss_20b"][columns].rename(
            columns={column: f"gpt_oss_{column}" for column in columns if column != "annotation_id"}
        )
        disagreement_base = key.merge(left, on="annotation_id").merge(right, on="annotation_id")
        general = disagreement_base.loc[
            disagreement_base["calibration_source"].eq("general_random")
            & disagreement_base["duplicate_of_annotation_id"].fillna("").eq("")
        ]
        fine_agreement = float(
            (general["llama_primary_label"] == general["gpt_oss_primary_label"]).mean()
        )
        broad_agreement = float(
            (
                general["llama_predicted_broad_label"]
                == general["gpt_oss_predicted_broad_label"]
            ).mean()
        )
        disagreements = disagreement_base.loc[
            disagreement_base["llama_primary_label"]
            != disagreement_base["gpt_oss_primary_label"]
        ].copy()
        small = summary_frame.set_index("model_name").loc["llama_3_2_1b"]
        teacher = summary_frame.set_index("model_name").loc["gpt_oss_20b"]
        checks = {
            "parse_success_at_least_99_percent": small["parse_success_rate"] >= 0.99,
            "reference_broad_accuracy_at_least_75_percent": small[
                "reference_broad_accuracy"
            ]
            >= 0.75,
            "reference_fine_accuracy_at_least_50_percent": small[
                "reference_fine_accuracy"
            ]
            >= 0.50,
            "within_three_points_of_teacher_broad_accuracy": small[
                "reference_broad_accuracy"
            ]
            >= teacher["reference_broad_accuracy"] - 0.03,
            "general_broad_agreement_at_least_80_percent": broad_agreement >= 0.80,
            "duplicate_broad_agreement_at_least_95_percent": small[
                "duplicate_broad_agreement"
            ]
            >= 0.95,
            "all_four_broad_categories_predicted": small[
                "general_broad_categories_predicted"
            ]
            == 4,
        }
        gate = {
            "status": "complete",
            "smaller_model_passes": bool(all(checks.values())),
            "general_fine_agreement": fine_agreement,
            "general_broad_agreement": broad_agreement,
            "checks": {key: bool(value) for key, value in checks.items()},
            "manual_disagreement_review_required": True,
        }
    return summary_frame, disagreements, gate


def evaluate_broad_candidate(
    annotations: pd.DataFrame,
    key: pd.DataFrame,
    *,
    teacher_annotations: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Evaluate a direct four-way candidate without pretending it has fine labels."""
    merged = key.merge(
        annotations[["annotation_id", "broad_label", "confidence"]],
        on="annotation_id",
        how="left",
        validate="one_to_one",
    )
    parsed = merged["broad_label"].fillna("").isin(BROAD_LABELS)
    reference = merged.loc[
        merged["calibration_source"].eq("matched_reference_stress") & parsed
    ].copy()
    non_duplicate = merged["duplicate_of_annotation_id"].fillna("").eq("")
    general = merged.loc[
        merged["calibration_source"].eq("general_random") & non_duplicate
    ].copy()
    originals = merged.set_index("annotation_id")["broad_label"]
    duplicate_rows = merged.loc[~non_duplicate]
    duplicate_agreement = [
        str(row.broad_label)
        == str(originals.get(row.duplicate_of_annotation_id, ""))
        for row in duplicate_rows.itertuples()
    ]
    summary: dict[str, Any] = {
        "model_name": "llama_3_2_1b_broad",
        "n_expected": len(key),
        "n_parsed": int(parsed.sum()),
        "parse_success_rate": float(parsed.mean()),
        "reference_rows_scored": len(reference),
        "reference_broad_accuracy": math.nan,
        "reference_broad_macro_f1": math.nan,
        "duplicate_broad_agreement": (
            float(np.mean(duplicate_agreement)) if duplicate_agreement else math.nan
        ),
        "general_broad_categories_predicted": int(
            general.loc[general["broad_label"].isin(BROAD_LABELS), "broad_label"].nunique()
        ),
        "general_broad_agreement_with_teacher": math.nan,
    }
    if len(reference):
        summary["reference_broad_accuracy"] = float(
            accuracy_score(reference["reference_broad_label"], reference["broad_label"])
        )
        summary["reference_broad_macro_f1"] = _macro_f1(
            reference["reference_broad_label"],
            reference["broad_label"],
            list(BROAD_LABELS),
        )
    disagreements = pd.DataFrame()
    teacher_reference_accuracy = math.nan
    if teacher_annotations is not None:
        teacher = teacher_annotations[["annotation_id", "primary_label"]].copy()
        teacher["teacher_broad_label"] = teacher["primary_label"].map(BROAD_LABEL_MAP)
        paired = merged.merge(
            teacher[["annotation_id", "teacher_broad_label"]],
            on="annotation_id",
            how="inner",
            validate="one_to_one",
        )
        paired_general = paired.loc[
            paired["calibration_source"].eq("general_random")
            & paired["duplicate_of_annotation_id"].fillna("").eq("")
        ]
        summary["general_broad_agreement_with_teacher"] = float(
            (paired_general["broad_label"] == paired_general["teacher_broad_label"]).mean()
        )
        teacher_reference = paired.loc[
            paired["calibration_source"].eq("matched_reference_stress")
        ]
        teacher_reference_accuracy = float(
            accuracy_score(
                teacher_reference["reference_broad_label"],
                teacher_reference["teacher_broad_label"],
            )
        )
        disagreements = paired.loc[
            paired["broad_label"] != paired["teacher_broad_label"]
        ].copy()
    checks = {
        "parse_success_at_least_99_percent": summary["parse_success_rate"] >= 0.99,
        "reference_broad_accuracy_at_least_75_percent": summary[
            "reference_broad_accuracy"
        ]
        >= 0.75,
        "duplicate_broad_agreement_at_least_95_percent": summary[
            "duplicate_broad_agreement"
        ]
        >= 0.95,
        "all_four_broad_categories_predicted": summary[
            "general_broad_categories_predicted"
        ]
        == 4,
        "within_three_points_of_teacher_broad_accuracy": (
            summary["reference_broad_accuracy"] >= teacher_reference_accuracy - 0.03
            if not math.isnan(teacher_reference_accuracy)
            else False
        ),
        "general_broad_agreement_at_least_80_percent": summary[
            "general_broad_agreement_with_teacher"
        ]
        >= 0.80,
    }
    gate = {
        "status": "complete" if teacher_annotations is not None else "incomplete",
        "broad_candidate_passes": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "manual_disagreement_review_required": True,
    }
    return summary, disagreements, gate


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
