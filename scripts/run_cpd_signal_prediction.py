#!/usr/bin/env python3
"""Run prospective belief-to-CPD and combined-signal action-event analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, precision_recall_curve

from reveng.experiments.cpd_signal_prediction import (
    add_belief_change_summaries,
    add_future_commitment_targets,
    add_future_point_targets,
    attach_frozen_change_points,
    paired_model_comparison,
)
from reveng.experiments.practical_action_event_monitor import (
    ModelSpec,
    nested_group_predictions,
    prediction_metrics,
)


BLUE = "#1769AA"
LIGHT_BLUE = "#8CC8E8"
DARK = "#17324D"
GRAY = "#66737C"
GRID = "#DCEAF3"

BASELINE = (
    "reasoning_step_idx",
    "action_confidence",
    "recent_action_js_bits",
)
ACTIVATION = (
    "activation_similarity_to_preceding_mean",
    "activation_adjacent_cosine_distance",
    "activation_update_norm",
    "rolling_representation_dispersion",
    "sparse_cross_layer_change",
)
BELIEF_CHANGE = (
    "belief_mean_probability_shift",
    "belief_max_probability_shift",
    "belief_mean_absolute_entropy_shift",
    "belief_answer_changes",
)
OFFLINE_BEAST = (
    "offline_beast_change_probability",
    "offline_beast_change_point_here",
)

MODEL_LABELS = {
    "behavioral_baseline": "Behavioral baseline",
    "activation": "Baseline + activation",
    "belief_changes": "Baseline + belief changes",
    "offline_beast": "Baseline + offline BEAST",
    "activation_and_belief": "Baseline + activation + belief changes",
    "activation_and_offline_beast": "Baseline + activation + offline BEAST",
    "belief_and_offline_beast": "Baseline + belief changes + offline BEAST",
    "all_signals": "All signals, including offline BEAST",
    "timing_baseline": "Reasoning stage and confidence",
    "action_history_baseline": "Plus recent action-distribution change",
    "belief_change_only": "Reasoning stage + belief changes",
    "action_history_and_belief": "Action history + belief changes",
}
TARGET_LABELS = {
    "recommendation_change": "Recommendation change",
    "optimality_loss": "Loss of optimality",
    "recovery": "Recovery of optimality",
    "commitment_onset": "Commitment onset",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-rows",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/practical_action_event_monitor_v1/"
            "prospective_feature_rows.parquet"
        ),
    )
    parser.add_argument(
        "--belief-features",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/practical_action_event_monitor_v1/"
            "sentence_belief_features.parquet"
        ),
    )
    parser.add_argument(
        "--beast-positions",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/action_distribution_cpd_beast_v1/"
            "position_change_probabilities.csv"
        ),
    )
    parser.add_argument(
        "--detected-points",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/action_distribution_cpd_beast_v1/"
            "detected_change_points.csv"
        ),
    )
    parser.add_argument(
        "--point-stability",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/action_distribution_cpd_robustness_v1/"
            "original_point_stability.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hypothesis_tests/cpd_signal_prediction_v1"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_matplotlib() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update(
        {
            "font.family": "Arial" if "Arial" in available else "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_analysis_frame(args: argparse.Namespace) -> pd.DataFrame:
    frame = pd.read_parquet(args.feature_rows)
    belief_rows = add_belief_change_summaries(pd.read_parquet(args.belief_features))
    frame = frame.merge(
        belief_rows[["example_id", "reasoning_step_idx", *BELIEF_CHANGE]],
        on=["example_id", "reasoning_step_idx"],
        how="left",
        validate="one_to_one",
    )
    detected = pd.read_csv(args.detected_points)
    stability = pd.read_csv(args.point_stability)
    frame = attach_frozen_change_points(
        frame,
        pd.read_csv(args.beast_positions),
        detected,
        stability,
    )
    frame = add_future_point_targets(frame, detected, stability, horizons=(1, 3))
    frame = add_future_commitment_targets(frame, horizons=(1, 3))
    if frame[list(BELIEF_CHANGE)].isna().any().any():
        raise ValueError("Belief-change summaries are missing from analysis rows")
    return frame


def cpd_specs() -> list[ModelSpec]:
    timing = ("reasoning_step_idx", "action_confidence")
    return [
        ModelSpec("timing_baseline", timing),
        ModelSpec("action_history_baseline", BASELINE),
        ModelSpec("belief_change_only", timing + BELIEF_CHANGE),
        ModelSpec("action_history_and_belief", BASELINE + BELIEF_CHANGE),
    ]


def action_specs() -> list[ModelSpec]:
    return [
        ModelSpec("behavioral_baseline", BASELINE),
        ModelSpec("activation", BASELINE + ACTIVATION),
        ModelSpec("belief_changes", BASELINE + BELIEF_CHANGE),
        ModelSpec("offline_beast", BASELINE + OFFLINE_BEAST),
        ModelSpec("activation_and_belief", BASELINE + ACTIVATION + BELIEF_CHANGE),
        ModelSpec(
            "activation_and_offline_beast", BASELINE + ACTIVATION + OFFLINE_BEAST
        ),
        ModelSpec("belief_and_offline_beast", BASELINE + BELIEF_CHANGE + OFFLINE_BEAST),
        ModelSpec("all_signals", BASELINE + ACTIVATION + BELIEF_CHANGE + OFFLINE_BEAST),
    ]


def fit_models(frame: pd.DataFrame, *, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []

    for horizon in (1, 3):
        complete = frame[
            frame[f"complete_change_point_horizon_{horizon}"].astype(int).eq(1)
        ]
        for target in (
            f"change_point_in_next_{horizon}",
            f"robust_change_point_in_next_{horizon}",
        ):
            for spec in cpd_specs():
                print(f"Fitting {target}: {spec.name}", flush=True)
                predictions, coefficients = nested_group_predictions(
                    complete,
                    spec=spec,
                    target_column=target,
                    seed=seed + horizon,
                )
                prediction_frames.append(predictions)
                coefficient_frames.append(coefficients)

    for horizon in (1, 3):
        complete = frame[frame[f"complete_horizon_{horizon}"].astype(int).eq(1)]
        target_risks = (
            (f"recommendation_change_h{horizon}", complete),
            (
                f"optimality_loss_h{horizon}",
                complete[complete["current_action_is_optimal"].astype(int).eq(1)],
            ),
            (
                f"recovery_h{horizon}",
                complete[complete["current_action_is_optimal"].astype(int).eq(0)],
            ),
            (
                f"commitment_onset_h{horizon}",
                complete[complete["currently_uncommitted"].astype(int).eq(1)],
            ),
        )
        for target, risk in target_risks:
            for spec in action_specs():
                print(f"Fitting {target}: {spec.name}", flush=True)
                predictions, coefficients = nested_group_predictions(
                    risk,
                    spec=spec,
                    target_column=target,
                    seed=seed + horizon,
                )
                prediction_frames.append(predictions)
                coefficient_frames.append(coefficients)
    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(coefficient_frames, ignore_index=True),
    )


def reference_for_target(target: str) -> str:
    if target.startswith(("change_point_", "robust_change_point_")):
        return "action_history_baseline"
    return "behavioral_baseline"


def fast_group_bootstrap_difference(
    paired: pd.DataFrame,
    *,
    metric: str,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    """Resample trajectories without rebuilding pandas frames each time."""
    groups = []
    for _, rows in paired.groupby("trajectory_id", sort=True):
        groups.append(
            (
                rows["outcome"].to_numpy(int),
                np.clip(rows["baseline_probability"].to_numpy(float), 1e-6, 1 - 1e-6),
                np.clip(rows["predicted_probability"].to_numpy(float), 1e-6, 1 - 1e-6),
            )
        )
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(repeats):
        selected = rng.integers(0, len(groups), size=len(groups))
        labels = np.concatenate([groups[index][0] for index in selected])
        reference = np.concatenate([groups[index][1] for index in selected])
        candidate = np.concatenate([groups[index][2] for index in selected])
        if metric == "auprc":
            if np.unique(labels).size < 2:
                continue
            difference = average_precision_score(
                labels, candidate
            ) - average_precision_score(labels, reference)
        elif metric == "log_loss":
            reference_loss = -np.mean(
                labels * np.log(reference) + (1 - labels) * np.log(1 - reference)
            )
            candidate_loss = -np.mean(
                labels * np.log(candidate) + (1 - labels) * np.log(1 - candidate)
            )
            difference = reference_loss - candidate_loss
        else:
            raise ValueError(f"Unsupported bootstrap metric: {metric}")
        differences.append(float(difference))
    if not differences:
        return np.nan, np.nan
    low, high = np.quantile(differences, [0.025, 0.975])
    return float(low), float(high)


def summarize_predictions(
    predictions: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    del repeats, seed  # Uncertainty is computed only for the planned contrasts below.
    rows: list[dict[str, Any]] = []
    for target, target_rows in predictions.groupby("target", sort=True):
        reference_name = reference_for_target(str(target))
        reference = target_rows[target_rows["model"].eq(reference_name)][
            ["row_id", "trajectory_id", "outcome", "predicted_probability"]
        ].rename(columns={"predicted_probability": "baseline_probability"})
        for model, model_rows in target_rows.groupby("model", sort=True):
            metrics = prediction_metrics(
                model_rows["outcome"].to_numpy(int),
                model_rows["predicted_probability"].to_numpy(float),
            )
            row: dict[str, Any] = {
                "target": target,
                "model": model,
                "model_label": MODEL_LABELS[model],
                "reference_model": reference_name,
                "n_rows": len(model_rows),
                "n_trajectories": model_rows["trajectory_id"].nunique(),
                "n_positive_windows": int(model_rows["outcome"].sum()),
                "prevalence": float(model_rows["outcome"].mean()),
                **metrics,
            }
            paired = model_rows[
                ["row_id", "trajectory_id", "outcome", "predicted_probability"]
            ].merge(
                reference,
                on=["row_id", "trajectory_id", "outcome"],
                how="inner",
                validate="one_to_one",
            )
            reference_metrics = prediction_metrics(
                paired["outcome"].to_numpy(int),
                paired["baseline_probability"].to_numpy(float),
            )
            row["auprc_change_from_reference"] = (
                metrics["auprc"] - reference_metrics["auprc"]
            )
            row["log_loss_improvement_from_reference"] = (
                reference_metrics["log_loss"] - metrics["log_loss"]
            )
            rows.append(row)
    return pd.DataFrame(rows)


def planned_comparisons(
    predictions: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    comparisons = [
        (
            "action_history_and_belief",
            "action_history_baseline",
            "Do belief changes add information beyond recent action history?",
        ),
        (
            "activation",
            "behavioral_baseline",
            "Do activations add information beyond behavior?",
        ),
        (
            "belief_changes",
            "behavioral_baseline",
            "Do belief changes add information beyond behavior?",
        ),
        (
            "activation_and_belief",
            "behavioral_baseline",
            "Do activation and belief together add information beyond behavior?",
        ),
        (
            "offline_beast",
            "behavioral_baseline",
            "Does offline BEAST add retrospective information beyond behavior?",
        ),
        (
            "all_signals",
            "activation_and_belief",
            "Does offline BEAST add information beyond the current-and-past signals?",
        ),
        (
            "all_signals",
            "behavioral_baseline",
            "Do all signals together add information beyond behavior?",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for target, target_rows in predictions.groupby("target", sort=True):
        available = set(target_rows["model"])
        for candidate, reference, question in comparisons:
            if candidate not in available or reference not in available:
                continue
            paired = paired_model_comparison(
                target_rows, candidate=candidate, reference=reference
            )
            labels = paired["outcome"].to_numpy(int)
            candidate_metrics = prediction_metrics(
                labels, paired["candidate_probability"].to_numpy(float)
            )
            reference_metrics = prediction_metrics(
                labels, paired["reference_probability"].to_numpy(float)
            )
            bootstrap = paired.rename(
                columns={
                    "candidate_probability": "predicted_probability",
                    "reference_probability": "baseline_probability",
                }
            )
            auprc_low, auprc_high = fast_group_bootstrap_difference(
                bootstrap, metric="auprc", repeats=repeats, seed=seed
            )
            loss_low, loss_high = fast_group_bootstrap_difference(
                bootstrap, metric="log_loss", repeats=repeats, seed=seed
            )
            rows.append(
                {
                    "target": target,
                    "question": question,
                    "candidate_model": candidate,
                    "candidate_label": MODEL_LABELS[candidate],
                    "reference_model": reference,
                    "reference_label": MODEL_LABELS[reference],
                    "candidate_auprc": candidate_metrics["auprc"],
                    "reference_auprc": reference_metrics["auprc"],
                    "auprc_change": candidate_metrics["auprc"]
                    - reference_metrics["auprc"],
                    "auprc_change_ci_low": auprc_low,
                    "auprc_change_ci_high": auprc_high,
                    "log_loss_improvement": reference_metrics["log_loss"]
                    - candidate_metrics["log_loss"],
                    "log_loss_improvement_ci_low": loss_low,
                    "log_loss_improvement_ci_high": loss_high,
                }
            )
    return pd.DataFrame(rows)


def fmt(value: float, digits: int = 3, signed: bool = False) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def target_parts(target: str) -> tuple[str, int]:
    horizon = int(target.rsplit("_", 1)[1].removeprefix("h"))
    name = target.rsplit("_", 1)[0]
    return name, horizon


def write_report(
    output_dir: Path,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    figure_metrics: dict[str, float],
    figure_cpd_summary: pd.DataFrame,
) -> None:
    cp_rows = comparisons[
        comparisons["candidate_model"].eq("action_history_and_belief")
    ].copy()
    action_rows = comparisons[
        comparisons["candidate_model"].eq("activation_and_belief")
        & comparisons["reference_model"].eq("behavioral_baseline")
    ].copy()
    offline_rows = comparisons[
        comparisons["candidate_model"].eq("all_signals")
        & comparisons["reference_model"].eq("activation_and_belief")
    ].copy()

    lines = [
        "# Can Belief Changes Anticipate BEAST Change Points?",
        "",
        "## Natural-language summary",
        "",
        "This analysis asks whether changes in the model's explicit state beliefs at the current sentence help identify a change point that offline BEAST later places in the next sentence or next three sentences. It then asks a separate question: whether activation dynamics, belief changes, and offline BEAST output—alone and in combination—help predict future recommendation changes, commitment, loss of optimality, or recovery.",
        "",
        "Every model feature is taken from the current or an earlier sentence. Models without BEAST can therefore compute a score at the current sentence if the action, activation, and belief readouts are available. BEAST itself is retrospective because it uses the rest of the time series to decide whether the current location is a change point, so models containing BEAST are retrospective comparisons only. Commitment is also a retrospective evaluation target, as explained below.",
        "",
        "## Data and validation",
        "",
        f"- Environment states: {frame['example_id'].nunique()}",
        f"- Trajectories: {frame['trajectory_id'].nunique()}",
        f"- Sentence positions: {len(frame):,}",
        "- Held-out evaluation: five folds, keeping every sentence from a trajectory in the same fold.",
        "- Primary metric: area under the precision-recall curve (AUPRC); positive log-loss improvement means better probability estimates.",
        "- Frozen BEAST outcomes: 160 original change points; 127 repeatedly found in at least 12 of 16 reruns.",
        "",
        "## Question 1: do belief changes anticipate offline BEAST locations?",
        "",
        "The reference model uses the current sentence number, current action confidence, and the change in the action distribution at the preceding sentence boundary. The candidate adds four plain summaries of belief change: average and largest probability shift, average entropy shift, and the number of belief answers that changed.",
        "",
        "| BEAST outcome | Future window | Positive windows | Reference AUPRC | With belief changes | AUPRC change (95% interval) | Log-loss improvement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cp_rows.sort_values("target").itertuples():
        robust = str(row.target).startswith("robust_")
        horizon = int(str(row.target).rsplit("_", 1)[1])
        summary_row = summary[
            summary["target"].eq(row.target)
            & summary["model"].eq("action_history_and_belief")
        ].iloc[0]
        lines.append(
            f"| {'Repeatedly found subset' if robust else 'All frozen points'} | {horizon} sentence{'s' if horizon > 1 else ''} | {int(summary_row.n_positive_windows)} | {fmt(row.reference_auprc)} | {fmt(row.candidate_auprc)} | {fmt(row.auprc_change, signed=True)} ({fmt(row.auprc_change_ci_low, signed=True)} to {fmt(row.auprc_change_ci_high, signed=True)}) | {fmt(row.log_loss_improvement, signed=True)} |"
        )

    primary_cp = cp_rows[cp_rows["target"].eq("change_point_in_next_3")]
    if not primary_cp.empty:
        row = primary_cp.iloc[0]
        supported = row["auprc_change_ci_low"] > 0 and row["log_loss_improvement"] > 0
        lines.extend(
            [
                "",
                "### Result",
                "",
                (
                    "At the three-sentence horizon, belief changes improve both held-out ranking and probability quality beyond recent action history."
                    if supported
                    else "At the three-sentence horizon, the evidence does not show a clear improvement from belief changes beyond recent action history."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Question 2: which current signals predict future action events?",
            "",
            "This table first compares the current-and-past activation-plus-belief model with the behavioral baseline. It then measures what offline BEAST adds to that same model. The BEAST comparison is diagnostic, not a deployable predictor.",
            "",
            "| Event | Window | Activation + belief: AUPRC change | Log-loss improvement | Additional AUPRC change from offline BEAST | Additional log-loss improvement |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    action_lookup = action_rows.set_index("target")
    offline_lookup = offline_rows.set_index("target")
    for target in sorted(set(action_lookup.index) & set(offline_lookup.index)):
        name, horizon = target_parts(target)
        a = action_lookup.loc[target]
        o = offline_lookup.loc[target]
        lines.append(
            f"| {TARGET_LABELS[name]} | {horizon} sentence{'s' if horizon > 1 else ''} | {fmt(a.auprc_change, signed=True)} ({fmt(a.auprc_change_ci_low, signed=True)} to {fmt(a.auprc_change_ci_high, signed=True)}) | {fmt(a.log_loss_improvement, signed=True)} | {fmt(o.auprc_change, signed=True)} ({fmt(o.auprc_change_ci_low, signed=True)} to {fmt(o.auprc_change_ci_high, signed=True)}) | {fmt(o.log_loss_improvement, signed=True)} |"
        )

    lines.extend(
        [
            "",
            "### All signal combinations at the three-sentence horizon",
            "",
            "Each cell is held-out AUPRC, followed in parentheses by its change from the behavioral baseline. The belief signal here means the four belief-change summaries defined above; it is narrower than the full belief-state model in the earlier practical-monitor analysis.",
            "",
            "| Current-sentence inputs | Recommendation change | Commitment onset | Loss of optimality | Recovery |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    h3_targets = [
        "recommendation_change_h3",
        "commitment_onset_h3",
        "optimality_loss_h3",
        "recovery_h3",
    ]
    for model in [spec.name for spec in action_specs()]:
        cells = []
        for target in h3_targets:
            row = summary[
                summary["target"].eq(target) & summary["model"].eq(model)
            ].iloc[0]
            cells.append(
                f"{fmt(row.auprc)} ({fmt(row.auprc_change_from_reference, signed=True)})"
            )
        lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "### Result",
            "",
            "The preplanned activation-plus-belief comparison does not clearly improve any action event over the behavioral baseline: every AUPRC interval crosses zero, and most log-loss changes are negative or effectively zero. Adding offline BEAST to those current-and-past signals also does not produce a clear incremental improvement. Individual combinations are retained in the table and CSV as exploratory comparisons, not selected as new headline results.",
            "",
            "Commitment onset is also a retrospective evaluation label: its location is defined by checking that the recommendation remains stable through the rest of the trace. Predictors use no future features, but the target cannot be confirmed at the moment it occurs.",
            "",
        ]
    )

    lines.extend(
        [
            "",
            "## Report-ready figures",
            "",
            "In these figures, action distribution means the probability distribution over `RIGHT`, `LEFT`, `UP`, and `DOWN`.",
            "",
            f"Across {int(figure_metrics['n_sentence_boundaries']):,} sentence boundaries, the recommended action changed {int(figure_metrics['n_action_changes']):,} times. Euclidean distance from the previous sentence's distribution identified these changes with AUPRC {figure_metrics['adjacent_l2_auprc']:.3f}; distance from the pre-reasoning distribution had AUPRC {figure_metrics['initial_l2_auprc']:.3f}.",
            "",
            "![Distribution distance and action changes](figs/distribution_distance_action_changes.png)",
            "",
            f"Adjacent changes in the six behavioural-probe readouts had only a weak association with adjacent changes in the action distribution (Spearman r = {figure_metrics['belief_distribution_spearman_r']:.3f}). Adding these belief-change features changed held-out AUPRC for identifying offline BEAST locations from {figure_cpd_summary.iloc[0].reference_auprc:.3f} to {figure_cpd_summary.iloc[0].with_belief_changes_auprc:.3f}.",
            "",
            "![Belief changes and distribution outcomes](figs/belief_changes_distribution_outcomes.png)",
            "",
            "The first association is partly structural: both distribution distance and the recommended-action label are derived from the same four probabilities. The BEAST outcome is also derived from the full action-distribution series. These are correlational identification results, not independent causal predictions.",
            "",
            "## Interpretation rules",
            "",
            "- AUPRC change measures whether the added signals rank true future events above non-events more effectively.",
            "- Log-loss improvement measures whether the predicted probabilities become more accurate. A positive value is better.",
            "- We call an addition useful only when AUPRC improves, its trajectory-resampling interval supports the same direction, and log loss also improves.",
            "- Correlated signals may share information. A weak incremental result does not imply that a signal has no association on its own.",
            "- Offline BEAST results cannot be described as real-time prediction.",
            "",
            "## Files",
            "",
            "- `prediction_summary.csv`: held-out metrics for every target and model.",
            "- `planned_comparisons.csv`: the direct signal-addition questions and uncertainty intervals.",
            "- `out_of_fold_predictions.parquet`: one held-out probability per row, target, and model.",
            "- `analysis_rows.parquet`: aligned features and future labels.",
            "- `figs/distribution_distance_action_changes.png`: how distribution distances identify recommendation changes.",
            "- `figs/belief_changes_distribution_outcomes.png`: belief changes versus distribution changes and offline BEAST locations.",
            "- `../action_distribution_cpd_robustness_v1/MEASUREMENT_FREEZE.md`: frozen BEAST outcome definition.",
            "",
        ]
    )
    (output_dir / "run_report.md").write_text("\n".join(lines))


def add_report_distribution_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add same-boundary action changes and L2 distance from pre-reasoning."""
    result = frame.sort_values(["example_id", "position_index"]).copy()
    probability_columns = [
        "action_probability_up",
        "action_probability_down",
        "action_probability_left",
        "action_probability_right",
    ]
    result["action_changed_from_previous_sentence"] = (
        result.groupby("example_id")["action_label"]
        .transform(lambda values: values.ne(values.shift()))
        .astype(float)
    )
    first_indices = result.groupby("example_id", sort=False).head(1).index
    result.loc[first_indices, "action_changed_from_previous_sentence"] = np.nan
    result["l2_distance_from_pre_reasoning_distribution"] = np.nan
    result["l2_distance_from_previous_sentence_distribution"] = np.nan
    for _, indices in result.groupby("example_id", sort=False).groups.items():
        ordered = result.loc[list(indices)].sort_values("position_index").index
        distributions = result.loc[ordered, probability_columns].to_numpy(float)
        result.loc[ordered, "l2_distance_from_pre_reasoning_distribution"] = (
            np.linalg.norm(distributions - distributions[0], axis=1)
        )
        result.loc[ordered[1:], "l2_distance_from_previous_sentence_distribution"] = (
            np.linalg.norm(distributions[1:] - distributions[:-1], axis=1)
        )
    return result


def precision_at_minimum_recall(
    outcome: pd.Series, score: pd.Series, minimum_recall: float
) -> tuple[float, float, float]:
    precision, recall, thresholds = precision_recall_curve(outcome, score)
    eligible = np.flatnonzero(recall >= minimum_recall)
    index = int(eligible[np.argmax(precision[eligible])])
    threshold = float(thresholds[index]) if index < len(thresholds) else np.nan
    return float(precision[index]), float(recall[index]), threshold


def report_metrics(frame: pd.DataFrame) -> dict[str, float]:
    adjacent = frame.dropna(
        subset=[
            "action_changed_from_previous_sentence",
            "l2_distance_from_previous_sentence_distribution",
        ]
    )
    outcome = adjacent["action_changed_from_previous_sentence"].astype(int)
    metrics: dict[str, float] = {
        "n_sentence_boundaries": len(adjacent),
        "n_action_changes": int(outcome.sum()),
        "action_change_prevalence": float(outcome.mean()),
        "adjacent_l2_auprc": float(
            average_precision_score(
                outcome,
                adjacent["l2_distance_from_previous_sentence_distribution"],
            )
        ),
        "initial_l2_auprc": float(
            average_precision_score(
                outcome, adjacent["l2_distance_from_pre_reasoning_distribution"]
            )
        ),
    }
    for recall in (0.50, 0.80):
        precision, achieved_recall, threshold = precision_at_minimum_recall(
            outcome,
            adjacent["l2_distance_from_previous_sentence_distribution"],
            recall,
        )
        suffix = int(recall * 100)
        metrics[f"adjacent_l2_precision_at_{suffix}_recall"] = precision
        metrics[f"adjacent_l2_achieved_recall_{suffix}"] = achieved_recall
        metrics[f"adjacent_l2_threshold_at_{suffix}_recall"] = threshold
    belief = frame.dropna(
        subset=[
            "belief_mean_probability_shift",
            "l2_distance_from_previous_sentence_distribution",
        ]
    )
    metrics["belief_distribution_spearman_r"] = float(
        spearmanr(
            belief["belief_mean_probability_shift"],
            belief["l2_distance_from_previous_sentence_distribution"],
        ).statistic
    )
    metrics["n_offline_beast_change_points"] = int(
        frame["offline_beast_change_point_here"].sum()
    )
    return metrics


def _fit_offline_change_point_models(
    frame: pd.DataFrame, *, repeats: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = ModelSpec(
        "distribution_history",
        (
            "reasoning_step_idx",
            "action_confidence",
            "l2_distance_from_previous_sentence_distribution",
        ),
    )
    with_beliefs = ModelSpec(
        "distribution_history_and_belief_changes",
        baseline.feature_columns + BELIEF_CHANGE,
    )
    predictions = []
    for specification in (baseline, with_beliefs):
        result, _ = nested_group_predictions(
            frame,
            spec=specification,
            target_column="offline_beast_change_point_here",
            seed=seed,
        )
        predictions.append(result)
    combined = pd.concat(predictions, ignore_index=True)
    comparison = paired_model_comparison(
        combined,
        candidate=with_beliefs.name,
        reference=baseline.name,
    )
    labels = comparison["outcome"].to_numpy(int)
    candidate_auprc = average_precision_score(
        labels, comparison["candidate_probability"]
    )
    reference_auprc = average_precision_score(
        labels, comparison["reference_probability"]
    )
    bootstrap = comparison.rename(
        columns={
            "candidate_probability": "predicted_probability",
            "reference_probability": "baseline_probability",
        }
    )
    difference_low, difference_high = fast_group_bootstrap_difference(
        bootstrap,
        metric="auprc",
        repeats=repeats,
        seed=seed,
    )
    summary = pd.DataFrame(
        [
            {
                "comparison": "belief_changes_added_to_distribution_history",
                "n_rows": len(comparison),
                "n_change_points": int(labels.sum()),
                "reference_auprc": reference_auprc,
                "with_belief_changes_auprc": candidate_auprc,
                "auprc_change": candidate_auprc - reference_auprc,
                "auprc_change_ci_low": difference_low,
                "auprc_change_ci_high": difference_high,
            }
        ]
    )
    return combined, summary


def plot_distribution_distance_action_changes(
    output_dir: Path, frame: pd.DataFrame, metrics: dict[str, float]
) -> None:
    setup_matplotlib()
    usable = frame.dropna(
        subset=[
            "action_changed_from_previous_sentence",
            "l2_distance_from_previous_sentence_distribution",
        ]
    )
    outcome = usable["action_changed_from_previous_sentence"].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
    precision, recall, _ = precision_recall_curve(
        outcome, usable["l2_distance_from_previous_sentence_distribution"]
    )
    axes[0].plot(recall, precision, color=BLUE, linewidth=2.2)
    axes[0].axhline(
        outcome.mean(),
        color=GRAY,
        linestyle="--",
        linewidth=1.2,
        label="Random ranking",
    )
    axes[0].set(xlim=(0, 1), ylim=(0, 1.02))
    axes[0].set_xlabel("Recall: proportion of action changes identified")
    axes[0].set_ylabel(
        "Precision: proportion of flagged boundaries\nthat are action changes"
    )
    axes[0].set_title("A. Current vs. previous sentence", loc="left", color=DARK)
    axes[0].text(
        0.97,
        0.96,
        f"AUPRC = {metrics['adjacent_l2_auprc']:.3f}\n"
        f"At 50% recall: {metrics['adjacent_l2_precision_at_50_recall']:.0%} precision\n"
        f"At 80% recall: {metrics['adjacent_l2_precision_at_80_recall']:.0%} precision",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        color=DARK,
    )
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].grid(color=GRID, linewidth=0.8)

    labels = ["Current vs.\nprevious sentence", "Current vs.\npre-reasoning"]
    values = [metrics["adjacent_l2_auprc"], metrics["initial_l2_auprc"]]
    bars = axes[1].bar(labels, values, color=[BLUE, LIGHT_BLUE], width=0.62)
    axes[1].axhline(outcome.mean(), color=GRAY, linestyle="--", linewidth=1.2)
    for bar, value in zip(bars, values, strict=True):
        offset = 0.015 if value > 0.5 else 0.030
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            color=DARK,
        )
    axes[1].set_ylim(0, 0.88)
    axes[1].set_ylabel("Area under precision–recall curve (AUPRC)")
    axes[1].set_title(
        "B. Previous-sentence vs. pre-reasoning reference",
        loc="left",
        color=DARK,
    )
    axes[1].grid(axis="y", color=GRID, linewidth=0.8)
    fig.suptitle(
        "Sentence-to-sentence distribution distance identifies action changes",
        fontsize=14,
        color=DARK,
    )
    fig.text(
        0.06,
        0.015,
        "Action distribution = probabilities over RIGHT, LEFT, UP, and DOWN. Higher AUPRC means more precise identification.",
        color=GRAY,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.07, 0.99, 0.93), w_pad=2.8)
    fig.savefig(
        output_dir / "figs" / "distribution_distance_action_changes.png", dpi=220
    )
    # Replace the previous confusing summary at its established path.
    fig.savefig(
        output_dir / "figs" / "belief_and_signal_prediction_summary.png", dpi=220
    )
    plt.close(fig)


def plot_belief_changes_and_distribution_outcomes(
    output_dir: Path,
    frame: pd.DataFrame,
    cpd_summary: pd.DataFrame,
    metrics: dict[str, float],
) -> None:
    setup_matplotlib()
    usable = frame.dropna(
        subset=[
            "belief_mean_probability_shift",
            "l2_distance_from_previous_sentence_distribution",
        ]
    ).copy()
    usable["belief_change_bin"] = pd.qcut(
        usable["belief_mean_probability_shift"].rank(method="first"), 10, labels=False
    )
    binned = usable.groupby("belief_change_bin", as_index=False).agg(
        belief_change=("belief_mean_probability_shift", "mean"),
        distribution_change=(
            "l2_distance_from_previous_sentence_distribution",
            "mean",
        ),
        n=("row_id", "size"),
    )
    row = cpd_summary.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
    axes[0].plot(
        binned["belief_change"],
        binned["distribution_change"],
        marker="o",
        color=BLUE,
        linewidth=1.8,
    )
    axes[0].set_xlabel(
        "Mean change in six belief-probe readouts\n(total-variation distance)"
    )
    axes[0].set_ylabel(
        "Mean distance from previous sentence's\naction distribution (Euclidean distance)"
    )
    axes[0].set_title(
        "A. Belief changes and distribution changes", loc="left", color=DARK
    )
    axes[0].text(
        0.97,
        0.95,
        f"Spearman r = {metrics['belief_distribution_spearman_r']:.3f}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        color=DARK,
    )
    axes[0].grid(color=GRID, linewidth=0.8)

    labels = [
        "Reasoning step + confidence\n+ previous-sentence distance",
        "Same inputs\n+ belief changes",
    ]
    values = [row.reference_auprc, row.with_belief_changes_auprc]
    bars = axes[1].bar(labels, values, color=[LIGHT_BLUE, BLUE], width=0.62)
    prevalence = row.n_change_points / row.n_rows
    axes[1].axhline(prevalence, color=GRAY, linestyle="--", linewidth=1.2)
    for bar, value in zip(bars, values, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.009,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            color=DARK,
        )
    axes[1].set_ylim(0, max(values) + 0.09)
    axes[1].set_ylabel("Held-out AUPRC")
    axes[1].set_title(
        "B. Identifying offline BEAST change points", loc="left", color=DARK
    )
    axes[1].text(
        0.5,
        0.94,
        f"Change from adding beliefs: {row.auprc_change:+.3f}\n"
        f"95% interval: {row.auprc_change_ci_low:+.3f} to {row.auprc_change_ci_high:+.3f}",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        color=DARK,
    )
    axes[1].grid(axis="y", color=GRID, linewidth=0.8)
    fig.suptitle(
        "Belief changes provide little information about action-distribution changes",
        fontsize=14,
        color=DARK,
    )
    fig.text(
        0.06,
        0.015,
        "Action distribution = probabilities over RIGHT, LEFT, UP, and DOWN.\nBEAST change points are offline labels computed from each full-CoT distribution series.",
        color=GRAY,
        fontsize=8.7,
    )
    fig.tight_layout(rect=(0.02, 0.07, 0.99, 0.93), w_pad=2.8)
    fig.savefig(
        output_dir / "figs" / "belief_changes_distribution_outcomes.png", dpi=220
    )
    plt.close(fig)


def write_report_figure_captions(output_dir: Path) -> None:
    (output_dir / "FIGURE_CAPTIONS.md").write_text(
        "# Figure captions\n\n"
        "## distribution_distance_action_changes.png\n\n"
        "Identification of recommendation changes from distances between action distributions over RIGHT, LEFT, UP, and DOWN. The left panel shows the precision–recall curve for Euclidean distance between consecutive sentence readouts. The right compares that measure with Euclidean distance from the pre-reasoning distribution.\n\n"
        "## belief_changes_distribution_outcomes.png\n\n"
        "Association between adjacent changes in the six behavioural-probe readouts and changes in the four-action distribution. The left panel shows mean action-distribution change across ten equally sized groups ordered by belief change. The right shows held-out identification of offline BEAST change points before and after adding belief-change features to distribution history.\n"
    )


def write_report_ready_summary(
    output_dir: Path,
    metrics: dict[str, float],
    cpd_summary: pd.DataFrame,
) -> None:
    row = cpd_summary.iloc[0]
    (output_dir / "REPORT_READY_SUMMARY.md").write_text(
        "# Distribution changes, action changes, and belief changes\n\n"
        "In this report, **action distribution** means the probability distribution over `RIGHT`, `LEFT`, `UP`, and `DOWN`. An **action change** occurs when the highest-probability action differs between consecutive sentence readouts.\n\n"
        "## A. Does distribution distance identify action changes?\n\n"
        "### Motivation\n\n"
        "Distance from the pre-reasoning distribution is the scalar series used in the offline change-point analysis. Distance from the immediately previous sentence may instead provide a direct signal of action changes during a reasoning rollout.\n\n"
        "### Result\n\n"
        f"Across {int(metrics['n_sentence_boundaries']):,} sentence boundaries, the recommended action changed {int(metrics['n_action_changes']):,} times ({metrics['action_change_prevalence']:.1%}). Euclidean distance from the previous sentence's action distribution identified these boundaries with AUPRC {metrics['adjacent_l2_auprc']:.3f}, compared with {metrics['initial_l2_auprc']:.3f} for distance from the pre-reasoning distribution. At approximately 50% recall, the previous-sentence measure achieved {metrics['adjacent_l2_precision_at_50_recall']:.0%} precision; at approximately 80% recall, it achieved {metrics['adjacent_l2_precision_at_80_recall']:.0%} precision.\n\n"
        "![Distribution distance and action changes](figs/distribution_distance_action_changes.png)\n\n"
        "## B. Do belief changes identify distribution changes or offline change points?\n\n"
        "### Motivation\n\n"
        "If belief changes anticipate changes in the action distribution, behavioural probes may help locate cases where the model's represented state and selected action diverge. Offline BEAST locations provide a separate full-trace outcome, although they cannot be detected prospectively by BEAST itself.\n\n"
        "### Result\n\n"
        f"Adjacent changes in the six behavioural-probe readouts had only a weak association with adjacent action-distribution distance (Spearman r = {metrics['belief_distribution_spearman_r']:.3f}). Adding belief-change features to reasoning step, action confidence, and previous-sentence distribution distance changed held-out AUPRC for identifying offline BEAST locations from {row.reference_auprc:.3f} to {row.with_belief_changes_auprc:.3f} (change {row.auprc_change:+.3f}; 95% trajectory-resampling interval {row.auprc_change_ci_low:+.3f} to {row.auprc_change_ci_high:+.3f}).\n\n"
        "![Belief changes and distribution outcomes](figs/belief_changes_distribution_outcomes.png)\n\n"
        "## Interpretation\n\n"
        "The previous-sentence distance is useful for identifying action changes in this dataset; distance from the pre-reasoning distribution is not. The current belief-change summaries do not meaningfully identify either adjacent distribution changes or offline BEAST locations. These results are correlational. In particular, distribution distance and the action-change label are both derived from the same four probabilities.\n"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    (output_dir / "figs").mkdir(parents=True, exist_ok=True)
    frame = load_analysis_frame(args)
    frame.to_parquet(output_dir / "analysis_rows.parquet", index=False)
    prediction_path = output_dir / "out_of_fold_predictions.parquet"
    coefficient_path = output_dir / "fold_feature_coefficients.parquet"
    if args.report_only:
        predictions = pd.read_parquet(prediction_path)
    else:
        predictions, coefficients = fit_models(frame, seed=args.seed)
        predictions.to_parquet(prediction_path, index=False)
        coefficients.to_parquet(coefficient_path, index=False)
    summary = summarize_predictions(
        predictions, repeats=args.bootstrap_repeats, seed=args.seed
    )
    comparisons = planned_comparisons(
        predictions, repeats=args.bootstrap_repeats, seed=args.seed
    )
    summary.to_csv(output_dir / "prediction_summary.csv", index=False)
    comparisons.to_csv(output_dir / "planned_comparisons.csv", index=False)
    report_frame = add_report_distribution_features(frame)
    metrics = report_metrics(report_frame)
    cpd_predictions, cpd_summary = _fit_offline_change_point_models(
        report_frame,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    pd.DataFrame([metrics]).to_csv(
        output_dir / "report_figure_metrics.csv", index=False
    )
    cpd_predictions.to_parquet(
        output_dir / "report_offline_change_point_predictions.parquet", index=False
    )
    cpd_summary.to_csv(
        output_dir / "report_offline_change_point_summary.csv", index=False
    )
    write_report(
        output_dir,
        frame,
        summary,
        comparisons,
        metrics,
        cpd_summary,
    )
    plot_distribution_distance_action_changes(output_dir, report_frame, metrics)
    plot_belief_changes_and_distribution_outcomes(
        output_dir, report_frame, cpd_summary, metrics
    )
    write_report_figure_captions(output_dir)
    write_report_ready_summary(output_dir, metrics, cpd_summary)
    manifest = {
        "status": "complete",
        "seed": args.seed,
        "bootstrap_repeats": args.bootstrap_repeats,
        "input_files": {
            str(path): sha256(path)
            for path in (
                args.feature_rows,
                args.belief_features,
                args.beast_positions,
                args.detected_points,
                args.point_stability,
            )
        },
        "environment_states": int(frame["example_id"].nunique()),
        "trajectories": int(frame["trajectory_id"].nunique()),
        "sentence_positions": len(frame),
        "future_windows_sentences": [1, 3],
        "held_out_group": "trajectory_id",
        "outer_folds": 5,
        "inner_folds": 3,
        "beast_is_offline": True,
        "primary_change_points": 160,
        "repeatedly_found_change_points": 127,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
