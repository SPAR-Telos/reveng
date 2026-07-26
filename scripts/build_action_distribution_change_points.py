#!/usr/bin/env python3
"""Build the offline sentence-level action-distribution change-point analysis."""

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

from reveng.experiments.action_distribution_change_points import (
    ACTIONS,
    build_event_aligned_rows,
    build_progress_matched_controls,
    compute_boundary_offsets,
    compute_position_metrics,
    grouped_bootstrap_ci,
    random_position_alignment_null,
    select_change_points,
    trajectory_bootstrap_ci,
)
from reveng.experiments.attention_event_windows import (
    compare_attention_at_boundaries,
    summarize_attention_rows,
)
from reveng.experiments.grouped_event_prediction import (
    binary_metrics,
    grouped_logistic_predictions,
    permutation_p_value,
)


PRIMARY_BELIEFS = (
    "wall_left",
    "wall_right",
    "wall_up",
    "wall_down",
    "has_key",
    "door_open",
)
BLUE = "#1769AA"
DARK = "#0B3C5D"
LIGHT = "#8CC8E8"
FAILURE = "#1769AA"
CONTROL = "#94C9EA"
GRID = "#E3EEF5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def setup_matplotlib() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update(
        {
            "font.family": "Arial" if "Arial" in available_fonts else "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def as_bool(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    value = str(value).strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def bootstrap_prediction_auc_delta(
    paired: pd.DataFrame,
    *,
    target_column: str,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    grouped = {
        str(trajectory_id): (
            group[target_column].to_numpy(dtype=int),
            group["predicted_probability_baseline"].to_numpy(dtype=float),
            group["predicted_probability_extended"].to_numpy(dtype=float),
        )
        for trajectory_id, group in paired.groupby("trajectory_id")
    }
    trajectories = np.asarray(sorted(grouped), dtype=object)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(repeats):
        sampled = rng.choice(trajectories, size=len(trajectories), replace=True)
        labels = np.concatenate([grouped[trajectory][0] for trajectory in sampled])
        baseline = np.concatenate(
            [grouped[trajectory][1] for trajectory in sampled]
        )
        extended = np.concatenate(
            [grouped[trajectory][2] for trajectory in sampled]
        )
        baseline_metrics = binary_metrics(labels, baseline)
        extended_metrics = binary_metrics(labels, extended)
        deltas.append(
            extended_metrics["auroc"] - baseline_metrics["auroc"]
        )
    return tuple(float(value) for value in np.quantile(deltas, [0.025, 0.975]))


def aggregate_post_boundary_stability(
    positions: pd.DataFrame,
    change_points: pd.DataFrame,
    *,
    local_window: int = 5,
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for boundary in change_points.itertuples():
        state = positions[positions["example_id"] == boundary.example_id].sort_values(
            "position_index"
        )
        center = int(boundary.boundary_position)
        before = state[state["position_index"].astype(int) < center]
        after = state[state["position_index"].astype(int) >= center]
        local_before = state[
            state["position_index"].astype(int).between(
                max(0, center - local_window), center - 1
            )
        ]
        local_after = state[
            state["position_index"].astype(int).between(
                center, center + local_window - 1
            )
        ]
        output.append(
            {
                "example_id": boundary.example_id,
                "trajectory_id": boundary.trajectory_id,
                "matched_pair_id": boundary.matched_pair_id,
                "matched_role": boundary.matched_role,
                "boundary_type": boundary.boundary_type,
                "boundary_position": center,
                "n_pre_positions": len(before),
                "n_post_positions": len(after),
                "mean_pre_js_to_full": before["js_to_full_trace"].mean(),
                "mean_post_js_to_full": after["js_to_full_trace"].mean(),
                "post_minus_pre_js_to_full": after["js_to_full_trace"].mean()
                - before["js_to_full_trace"].mean(),
                "mean_pre_tv_to_full": before["tv_to_full_trace"].mean(),
                "mean_post_tv_to_full": after["tv_to_full_trace"].mean(),
                "post_minus_pre_tv_to_full": after["tv_to_full_trace"].mean()
                - before["tv_to_full_trace"].mean(),
                "local_pre_js_to_full": local_before["js_to_full_trace"].mean(),
                "local_post_js_to_full": local_after["js_to_full_trace"].mean(),
                "local_post_minus_pre_js": local_after["js_to_full_trace"].mean()
                - local_before["js_to_full_trace"].mean(),
                "local_pre_tv_to_full": local_before["tv_to_full_trace"].mean(),
                "local_post_tv_to_full": local_after["tv_to_full_trace"].mean(),
                "local_post_minus_pre_tv": local_after["tv_to_full_trace"].mean()
                - local_before["tv_to_full_trace"].mean(),
            }
        )
    return pd.DataFrame(output)


def summarize_change_points(
    change_points: pd.DataFrame,
    alignments: pd.DataFrame,
    stability: pd.DataFrame,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows: list[dict[str, Any]] = []
    for boundary_type in change_points["boundary_type"].unique():
        boundary_frame = change_points[change_points["boundary_type"] == boundary_type]
        stability_frame = stability[stability["boundary_type"] == boundary_type]
        for role in ("all", "failure", "control"):
            subset = (
                boundary_frame
                if role == "all"
                else boundary_frame[boundary_frame["matched_role"] == role]
            )
            stable_subset = (
                stability_frame
                if role == "all"
                else stability_frame[stability_frame["matched_role"] == role]
            )
            if subset.empty:
                continue
            progress_low, progress_high = trajectory_bootstrap_ci(
                subset,
                value_column="boundary_reasoning_progress",
                repeats=bootstrap_repeats,
                seed=seed,
            )
            post_low, post_high = trajectory_bootstrap_ci(
                stable_subset,
                value_column="local_post_minus_pre_js",
                repeats=bootstrap_repeats,
                seed=seed + 1,
            )
            group_rows.append(
                {
                    "boundary_type": boundary_type,
                    "state_group": role,
                    "n_states": subset["example_id"].nunique(),
                    "n_trajectories": subset["trajectory_id"].nunique(),
                    "mean_boundary_sentence": subset["boundary_position"].mean(),
                    "median_boundary_sentence": subset["boundary_position"].median(),
                    "mean_boundary_character_progress": subset[
                        "boundary_reasoning_progress"
                    ].mean(),
                    "boundary_progress_ci_low": progress_low,
                    "boundary_progress_ci_high": progress_high,
                    "optimal_action_rate_at_boundary": subset[
                        "boundary_action_is_optimal"
                    ].map(as_bool).mean(),
                    "mean_local_post_minus_pre_js": stable_subset[
                        "local_post_minus_pre_js"
                    ].mean(),
                    "local_post_minus_pre_js_ci_low": post_low,
                    "local_post_minus_pre_js_ci_high": post_high,
                }
            )

    alignment_rows: list[dict[str, Any]] = []
    for role in ("all", "failure", "control"):
        subset = (
            alignments
            if role == "all"
            else alignments[alignments["matched_role"] == role]
        )
        if subset.empty:
            continue
        abs_offset = subset["jump_minus_stable_sentences"].abs()
        ratio = subset.dropna(subset=["final_action_jump_abruptness_ratio"]).copy()
        ratio_low, ratio_high = trajectory_bootstrap_ci(
            ratio,
            value_column="final_action_jump_abruptness_ratio",
            statistic="median",
            repeats=bootstrap_repeats,
            seed=seed + 2,
        )
        alignment_rows.append(
            {
                "state_group": role,
                "n_states": subset["example_id"].nunique(),
                "median_abs_jump_stable_offset_sentences": abs_offset.median(),
                "mean_abs_jump_stable_offset_sentences": abs_offset.mean(),
                "fraction_jump_within_1_sentence": subset[
                    "jump_within_1_of_stable"
                ].mean(),
                "fraction_jump_within_3_sentences": subset[
                    "jump_within_3_of_stable"
                ].mean(),
                "fraction_jump_within_5_sentences": subset[
                    "jump_within_5_of_stable"
                ].mean(),
                "fraction_distribution_change_within_3_sentences": subset[
                    "distribution_within_3_of_stable"
                ].mean(),
                "median_final_action_jump_abruptness_ratio": ratio[
                    "final_action_jump_abruptness_ratio"
                ].median(),
                "abruptness_ratio_ci_low": ratio_low,
                "abruptness_ratio_ci_high": ratio_high,
                "stable_boundaries_reproduced": subset[
                    "stable_boundary_reproduced"
                ].sum(),
            }
        )
    return pd.DataFrame(group_rows), pd.DataFrame(alignment_rows)


def summarize_event_control_rows(
    rows: pd.DataFrame,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    if rows.empty:
        return pd.DataFrame()
    for boundary_type in sorted(rows["boundary_type"].unique()):
        boundary_rows = rows[rows["boundary_type"] == boundary_type]
        for metric in sorted(boundary_rows["metric"].unique()):
            metric_rows = boundary_rows[boundary_rows["metric"] == metric]
            for role in ("all", "failure", "control"):
                group = (
                    metric_rows
                    if role == "all"
                    else metric_rows[metric_rows["matched_role"] == role]
                )
                if group.empty:
                    continue
                low, high = grouped_bootstrap_ci(
                    group,
                    value_column="event_minus_control",
                    group_column="trajectory_id",
                    repeats=bootstrap_repeats,
                    seed=seed,
                )
                output.append(
                    {
                        "boundary_type": boundary_type,
                        "metric": metric,
                        "matched_role": role,
                        "n_states": group["example_id"].nunique(),
                        "n_trajectories": group["trajectory_id"].nunique(),
                        "mean_event_value": group["event_value"].mean(),
                        "mean_control_value": group["control_value"].mean(),
                        "mean_event_minus_control": group[
                            "event_minus_control"
                        ].mean(),
                        "event_minus_control_ci_low": low,
                        "event_minus_control_ci_high": high,
                        "fraction_event_greater": (
                            group["event_minus_control"] > 0
                        ).mean(),
                    }
                )
    return pd.DataFrame(output)


def build_activation_boundary_rows(
    geometry: pd.DataFrame,
    change_points: pd.DataFrame,
    controls: pd.DataFrame,
) -> pd.DataFrame:
    primary = geometry[
        (geometry["layer"].astype(int) == 15)
        & (geometry["representation"] == "sentence_mean")
    ].copy()
    metrics = ("update_norm", "adjacent_cosine", "previous_mean_cosine")
    control_lookup = {
        (row.example_id, row.boundary_type): int(row.control_position)
        for row in controls.itertuples()
    }
    output: list[dict[str, Any]] = []
    for boundary in change_points.itertuples():
        key = (boundary.example_id, boundary.boundary_type)
        if key not in control_lookup:
            continue
        event = primary[
            (primary["example_id"] == boundary.example_id)
            & (
                primary["reasoning_step_idx"].astype(int)
                == int(boundary.boundary_reasoning_step_idx)
            )
        ]
        control = primary[
            (primary["example_id"] == boundary.example_id)
            & (
                primary["reasoning_step_idx"].astype(int)
                == control_lookup[key]
            )
        ]
        if event.empty or control.empty:
            continue
        for metric in metrics:
            event_value = float(event.iloc[0][metric])
            control_value = float(control.iloc[0][metric])
            if not np.isfinite(event_value) or not np.isfinite(control_value):
                continue
            output.append(
                {
                    "example_id": boundary.example_id,
                    "trajectory_id": boundary.trajectory_id,
                    "matched_role": boundary.matched_role,
                    "boundary_type": boundary.boundary_type,
                    "metric": metric,
                    "event_value": event_value,
                    "control_value": control_value,
                    "event_minus_control": event_value - control_value,
                }
            )
    rows = pd.DataFrame(output)
    return rows


def build_belief_boundary_rows(
    beliefs: pd.DataFrame,
    change_points: pd.DataFrame,
    controls: pd.DataFrame,
) -> pd.DataFrame:
    primary = beliefs[beliefs["question_id"].isin(PRIMARY_BELIEFS)].copy()
    primary["belief_error"] = primary["belief_is_error"].map(as_bool)
    aggregated = (
        primary.groupby(["example_id", "position_index"], as_index=False)
        .agg(
            mean_belief_error=("belief_error", "mean"),
            mean_belief_entropy_bits=("entropy_bits", "mean"),
        )
    )
    control_lookup = {
        (row.example_id, row.boundary_type): int(row.control_position)
        for row in controls.itertuples()
    }
    output: list[dict[str, Any]] = []
    for boundary in change_points.itertuples():
        key = (boundary.example_id, boundary.boundary_type)
        if key not in control_lookup:
            continue
        event = aggregated[
            (aggregated["example_id"] == boundary.example_id)
            & (
                aggregated["position_index"].astype(int)
                == int(boundary.boundary_position)
            )
        ]
        control = aggregated[
            (aggregated["example_id"] == boundary.example_id)
            & (
                aggregated["position_index"].astype(int)
                == control_lookup[key]
            )
        ]
        if event.empty or control.empty:
            continue
        for metric in ("mean_belief_error", "mean_belief_entropy_bits"):
            event_value = float(event.iloc[0][metric])
            control_value = float(control.iloc[0][metric])
            output.append(
                {
                    "example_id": boundary.example_id,
                    "trajectory_id": boundary.trajectory_id,
                    "matched_role": boundary.matched_role,
                    "boundary_type": boundary.boundary_type,
                    "metric": metric,
                    "event_value": event_value,
                    "control_value": control_value,
                    "event_minus_control": event_value - control_value,
                }
            )
    rows = pd.DataFrame(output)
    return rows


def build_distribution_boundary_control_rows(
    positions: pd.DataFrame,
    change_points: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    local_window: int = 5,
) -> pd.DataFrame:
    boundary_lookup = {
        (row.example_id, row.boundary_type): row
        for row in change_points.itertuples()
    }
    output: list[dict[str, Any]] = []
    for control in controls.itertuples():
        key = (control.example_id, control.boundary_type)
        boundary = boundary_lookup[key]
        state = positions[positions["example_id"] == control.example_id]

        def local_delta(center: int, metric: str) -> float:
            before = state[
                state["position_index"].astype(int).between(
                    max(0, center - local_window), center - 1
                )
            ][metric]
            after = state[
                state["position_index"].astype(int).between(
                    center, center + local_window - 1
                )
            ][metric]
            if before.empty or after.empty:
                return float("nan")
            return float(after.mean() - before.mean())

        for metric in ("js_to_full_trace", "tv_to_full_trace"):
            event_delta = local_delta(int(boundary.boundary_position), metric)
            control_delta = local_delta(int(control.control_position), metric)
            output.append(
                {
                    "example_id": control.example_id,
                    "trajectory_id": control.trajectory_id,
                    "matched_role": boundary.matched_role,
                    "boundary_type": control.boundary_type,
                    "metric": f"local_change_in_{metric}",
                    "event_value": event_delta,
                    "control_value": control_delta,
                    "event_minus_control": event_delta - control_delta,
                }
            )
    return pd.DataFrame(output)


def build_matched_group_difference_summary(
    change_points: pd.DataFrame,
    stability: pd.DataFrame,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> pd.DataFrame:
    merged = change_points.merge(
        stability[
            [
                "example_id",
                "boundary_type",
                "local_post_minus_pre_js",
            ]
        ],
        on=["example_id", "boundary_type"],
        how="left",
    )
    metrics = (
        "boundary_reasoning_progress",
        "final_action_jump_abruptness_ratio",
        "local_post_minus_pre_js",
    )
    output: list[dict[str, Any]] = []
    for boundary_type, boundary_rows in merged.groupby("boundary_type"):
        for metric in metrics:
            pivot = boundary_rows.pivot_table(
                index="matched_pair_id",
                columns="matched_role",
                values=metric,
                aggfunc="first",
            ).dropna(subset=["failure", "control"])
            if pivot.empty:
                continue
            differences = pd.DataFrame(
                {
                    "matched_pair_id": pivot.index,
                    "failure_minus_control": (
                        pivot["failure"] - pivot["control"]
                    ).to_numpy(),
                }
            )
            low, high = grouped_bootstrap_ci(
                differences,
                value_column="failure_minus_control",
                group_column="matched_pair_id",
                repeats=bootstrap_repeats,
                seed=seed,
            )
            output.append(
                {
                    "boundary_type": boundary_type,
                    "metric": metric,
                    "n_matched_pairs": len(differences),
                    "mean_failure_minus_control": differences[
                        "failure_minus_control"
                    ].mean(),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(output)


def build_next_sentence_prediction(
    positions: pd.DataFrame,
    geometry: pd.DataFrame,
    change_points: pd.DataFrame,
    *,
    permutation_repeats: int,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = geometry[
        (geometry["layer"].astype(int) == 15)
        & (geometry["representation"] == "sentence_mean")
    ].copy()
    position_columns = positions[
        [
            "example_id",
            "trajectory_id",
            "reasoning_step_idx",
            "action_confidence",
        ]
    ].copy()
    frame = primary.merge(
        position_columns,
        on=["example_id", "trajectory_id", "reasoning_step_idx"],
        how="inner",
    )
    max_position = positions.groupby("example_id")["position_index"].max().to_dict()
    frame = frame[
        [
            int(row.reasoning_step_idx) < int(max_position[row.example_id])
            for row in frame.itertuples()
        ]
    ].copy()
    baseline_features = ("reasoning_progress", "action_confidence")
    activation_features = ("previous_mean_cosine",)
    prediction_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for target_index, boundary_type in enumerate(
        (
            "final_action_jump",
            "largest_distribution_change",
            "stable_action_boundary",
        )
    ):
        next_positions = {
            (row.example_id, int(row.boundary_position) - 1)
            for row in change_points[
                change_points["boundary_type"] == boundary_type
            ].itertuples()
            if int(row.boundary_position) > 1
        }
        target = f"next_{boundary_type}"
        target_frame = frame.copy()
        target_frame[target] = [
            int((row.example_id, int(row.reasoning_step_idx)) in next_positions)
            for row in target_frame.itertuples()
        ]
        baseline = grouped_logistic_predictions(
            target_frame,
            feature_columns=baseline_features,
            target_column=target,
            seed=seed + target_index,
        )
        baseline["model"] = "reasoning_progress_and_action_confidence"
        extended = grouped_logistic_predictions(
            target_frame,
            feature_columns=baseline_features + activation_features,
            target_column=target,
            seed=seed + target_index,
        )
        extended["model"] = "baseline_plus_activation_similarity"
        predictions = pd.concat([baseline, extended], ignore_index=True)
        predictions["boundary_type"] = boundary_type
        prediction_rows.append(predictions)

        baseline_metrics = binary_metrics(
            baseline[target].to_numpy(),
            baseline["predicted_probability"].to_numpy(),
        )
        extended_metrics = binary_metrics(
            extended[target].to_numpy(),
            extended["predicted_probability"].to_numpy(),
        )
        observed_delta, permutation_p, null_delta = permutation_p_value(
            target_frame,
            baseline_features=baseline_features,
            added_features=activation_features,
            target_column=target,
            repeats=permutation_repeats,
            seed=seed + target_index,
        )
        paired = baseline[
            [
                "example_id",
                "trajectory_id",
                "reasoning_step_idx",
                target,
                "predicted_probability",
            ]
        ].merge(
            extended[
                [
                    "example_id",
                    "trajectory_id",
                    "reasoning_step_idx",
                    "predicted_probability",
                ]
            ],
            on=["example_id", "trajectory_id", "reasoning_step_idx"],
            suffixes=("_baseline", "_extended"),
        )
        ci_low, ci_high = bootstrap_prediction_auc_delta(
            paired,
            target_column=target,
            repeats=bootstrap_repeats,
            seed=seed + 100 + target_index,
        )
        summary_rows.append(
            {
                "boundary_type": boundary_type,
                "prediction_target": "boundary occurs at the next sentence",
                "n_rows": len(target_frame),
                "n_events": int(target_frame[target].sum()),
                "n_trajectories": target_frame["trajectory_id"].nunique(),
                "baseline_features": "reasoning_progress; action_confidence",
                "added_feature": "cosine_similarity_to_mean_preceding_activations",
                "baseline_auroc": baseline_metrics["auroc"],
                "extended_auroc": extended_metrics["auroc"],
                "auroc_improvement": observed_delta,
                "auroc_improvement_ci_low": float(ci_low),
                "auroc_improvement_ci_high": float(ci_high),
                "within_state_circular_shift_p": permutation_p,
                "bonferroni_p_across_three_boundaries": min(
                    1.0, permutation_p * 3
                ),
                "permutation_null_mean_improvement": null_delta,
                "baseline_log_loss": baseline_metrics["log_loss"],
                "extended_log_loss": extended_metrics["log_loss"],
                "log_loss_improvement": (
                    baseline_metrics["log_loss"] - extended_metrics["log_loss"]
                ),
                "baseline_brier_score": baseline_metrics["brier_score"],
                "extended_brier_score": extended_metrics["brier_score"],
            }
        )
    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(summary_rows)


def build_activation_representation_sensitivity(
    positions: pd.DataFrame,
    geometry: pd.DataFrame,
    change_points: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> pd.DataFrame:
    boundary_type = "largest_distribution_change"
    boundaries = change_points[
        change_points["boundary_type"] == boundary_type
    ].copy()
    next_positions = {
        (row.example_id, int(row.boundary_position) - 1)
        for row in boundaries.itertuples()
        if int(row.boundary_position) > 1
    }
    control_positions = {
        row.example_id: int(row.control_position) - 1
        for row in controls[controls["boundary_type"] == boundary_type].itertuples()
    }
    boundary_positions = {
        row.example_id: int(row.boundary_position) - 1
        for row in boundaries.itertuples()
    }
    max_position = positions.groupby("example_id")["position_index"].max().to_dict()
    position_columns = positions[
        [
            "example_id",
            "trajectory_id",
            "reasoning_step_idx",
            "action_confidence",
        ]
    ]
    output: list[dict[str, Any]] = []
    for layer in (8, 15, 23):
        for representation in ("sentence_mean", "sentence_final"):
            primary = geometry[
                (geometry["layer"].astype(int) == layer)
                & (geometry["representation"] == representation)
            ].copy()
            frame = primary.merge(
                position_columns,
                on=["example_id", "trajectory_id", "reasoning_step_idx"],
                how="inner",
            )
            frame = frame[
                [
                    int(row.reasoning_step_idx) < int(max_position[row.example_id])
                    for row in frame.itertuples()
                ]
            ].copy()
            target = "next_largest_distribution_change"
            frame[target] = [
                int((row.example_id, int(row.reasoning_step_idx)) in next_positions)
                for row in frame.itertuples()
            ]
            baseline = grouped_logistic_predictions(
                frame,
                feature_columns=("reasoning_progress", "action_confidence"),
                target_column=target,
                seed=seed,
            )
            extended = grouped_logistic_predictions(
                frame,
                feature_columns=(
                    "reasoning_progress",
                    "action_confidence",
                    "previous_mean_cosine",
                ),
                target_column=target,
                seed=seed,
            )
            baseline_metrics = binary_metrics(
                baseline[target].to_numpy(),
                baseline["predicted_probability"].to_numpy(),
            )
            extended_metrics = binary_metrics(
                extended[target].to_numpy(),
                extended["predicted_probability"].to_numpy(),
            )
            paired = baseline[
                [
                    "example_id",
                    "trajectory_id",
                    "reasoning_step_idx",
                    target,
                    "predicted_probability",
                ]
            ].merge(
                extended[
                    [
                        "example_id",
                        "trajectory_id",
                        "reasoning_step_idx",
                        "predicted_probability",
                    ]
                ],
                on=["example_id", "trajectory_id", "reasoning_step_idx"],
                suffixes=("_baseline", "_extended"),
            )
            ci_low, ci_high = bootstrap_prediction_auc_delta(
                paired,
                target_column=target,
                repeats=bootstrap_repeats,
                seed=seed + layer,
            )

            predecessor_differences = []
            event_transition_differences = []
            for example_id in sorted(boundary_positions):
                event = frame[
                    (frame["example_id"] == example_id)
                    & (
                        frame["reasoning_step_idx"].astype(int)
                        == boundary_positions[example_id]
                    )
                ]
                control = frame[
                    (frame["example_id"] == example_id)
                    & (
                        frame["reasoning_step_idx"].astype(int)
                        == control_positions[example_id]
                    )
                ]
                if event.empty or control.empty:
                    continue
                predecessor_differences.append(
                    {
                        "trajectory_id": event.iloc[0]["trajectory_id"],
                        "event_minus_control": float(
                            event.iloc[0]["previous_mean_cosine"]
                            - control.iloc[0]["previous_mean_cosine"]
                        ),
                    }
                )
                boundary_position = boundary_positions[example_id] + 1
                event_sentence = primary[
                    (primary["example_id"] == example_id)
                    & (
                        primary["reasoning_step_idx"].astype(int)
                        == boundary_position
                    )
                ]
                previous_sentence = primary[
                    (primary["example_id"] == example_id)
                    & (
                        primary["reasoning_step_idx"].astype(int)
                        == boundary_position - 1
                    )
                ]
                if not event_sentence.empty and not previous_sentence.empty:
                    event_transition_differences.append(
                        {
                            "trajectory_id": event_sentence.iloc[0][
                                "trajectory_id"
                            ],
                            "event_minus_previous": float(
                                event_sentence.iloc[0]["previous_mean_cosine"]
                                - previous_sentence.iloc[0][
                                    "previous_mean_cosine"
                                ]
                            ),
                        }
                    )
            predecessor_frame = pd.DataFrame(predecessor_differences)
            predecessor_low, predecessor_high = grouped_bootstrap_ci(
                predecessor_frame,
                value_column="event_minus_control",
                group_column="trajectory_id",
                repeats=bootstrap_repeats,
                seed=seed + layer + 100,
            )
            transition_frame = pd.DataFrame(event_transition_differences)
            transition_low, transition_high = grouped_bootstrap_ci(
                transition_frame,
                value_column="event_minus_previous",
                group_column="trajectory_id",
                repeats=bootstrap_repeats,
                seed=seed + layer + 200,
            )
            output.append(
                {
                    "layer": layer,
                    "representation": representation,
                    "n_rows": len(frame),
                    "n_events": int(frame[target].sum()),
                    "baseline_auroc": baseline_metrics["auroc"],
                    "extended_auroc": extended_metrics["auroc"],
                    "auroc_improvement": (
                        extended_metrics["auroc"] - baseline_metrics["auroc"]
                    ),
                    "auroc_improvement_ci_low": float(ci_low),
                    "auroc_improvement_ci_high": float(ci_high),
                    "predecessor_similarity_minus_control": predecessor_frame[
                        "event_minus_control"
                    ].mean(),
                    "predecessor_similarity_ci_low": predecessor_low,
                    "predecessor_similarity_ci_high": predecessor_high,
                    "event_minus_previous_similarity": transition_frame[
                        "event_minus_previous"
                    ].mean(),
                    "event_minus_previous_ci_low": transition_low,
                    "event_minus_previous_ci_high": transition_high,
                }
            )
    return pd.DataFrame(output)


def build_semantic_candidates(
    change_points: pd.DataFrame,
    controls: pd.DataFrame,
    sentences: pd.DataFrame,
    positions: pd.DataFrame,
) -> list[dict[str, Any]]:
    text_lookup = {
        (str(row.trace_id), int(row.sentence_id)): str(row.text)
        for row in sentences.itertuples()
    }
    control_lookup = {
        (row.example_id, row.boundary_type): int(row.control_position)
        for row in controls.itertuples()
    }
    position_lookup = {
        (str(row.example_id), int(row.position_index)): row
        for row in positions.itertuples()
    }
    output: list[dict[str, Any]] = []
    for boundary in change_points.itertuples():
        position = int(boundary.boundary_position)
        if position <= 0:
            continue
        control_position = control_lookup.get(
            (boundary.example_id, boundary.boundary_type)
        )
        if control_position is None or control_position <= 0:
            continue
        sentence_id = position - 1
        control_sentence_id = control_position - 1
        boundary_row = position_lookup[(boundary.example_id, position)]
        previous_row = position_lookup.get((boundary.example_id, position - 1))
        control_row = position_lookup[(boundary.example_id, control_position)]
        control_previous_row = position_lookup.get(
            (boundary.example_id, control_position - 1)
        )
        output.append(
            {
                "candidate_id": (
                    f"{boundary.example_id}:{boundary.boundary_type}:{position}"
                ),
                "judge_input": {
                    "previous_sentence": text_lookup.get(
                        (boundary.example_id, sentence_id - 1), ""
                    ),
                    "current_sentence": text_lookup.get(
                        (boundary.example_id, sentence_id), ""
                    ),
                    "comparison_previous_sentence": text_lookup.get(
                        (boundary.example_id, control_sentence_id - 1), ""
                    ),
                    "comparison_current_sentence": text_lookup.get(
                        (boundary.example_id, control_sentence_id), ""
                    ),
                },
                "hidden_metadata": {
                    "example_id": boundary.example_id,
                    "trajectory_id": boundary.trajectory_id,
                    "boundary_type": boundary.boundary_type,
                    "boundary_position": position,
                    "boundary_sentence_id": sentence_id,
                    "boundary_score": boundary.boundary_score,
                    "boundary_action": boundary.boundary_action,
                    "boundary_action_is_optimal": boundary.boundary_action_is_optimal,
                    "action_distribution_before_json": (
                        previous_row.action_probabilities_json
                        if previous_row is not None
                        else ""
                    ),
                    "action_distribution_after_json": (
                        boundary_row.action_probabilities_json
                    ),
                    "recommended_action_changed": bool(
                        boundary_row.argmax_action_changed
                    ),
                    "optimality_changed": (
                        previous_row is not None
                        and as_bool(previous_row.action_is_optimal)
                        != as_bool(boundary_row.action_is_optimal)
                    ),
                    "matched_role": boundary.matched_role,
                    "primary_step_failure_mode": boundary.primary_step_failure_mode,
                    "control_position": control_position,
                    "control_sentence_id": control_sentence_id,
                    "control_action_distribution_before_json": (
                        control_previous_row.action_probabilities_json
                        if control_previous_row is not None
                        else ""
                    ),
                    "control_action_distribution_after_json": (
                        control_row.action_probabilities_json
                    ),
                },
            }
        )
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def plot_final_action_probability(
    aligned: pd.DataFrame,
    output: Path,
) -> None:
    data = aligned[aligned["boundary_type"] == "final_action_jump"].copy()
    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), sharex=True, sharey=True)
    for ax, role in zip(axes, ("control", "failure"), strict=True):
        subset = data[data["matched_role"] == role]
        summary = (
            subset.groupby("relative_sentence_offset")["prob_final_action"]
            .agg(
                median="median",
                q25=lambda values: values.quantile(0.25),
                q75=lambda values: values.quantile(0.75),
                n="count",
            )
            .reset_index()
        )
        x = summary["relative_sentence_offset"].to_numpy(dtype=float)
        ax.plot(x, summary["median"], color=BLUE, marker="o", markersize=3)
        ax.fill_between(
            x,
            summary["q25"].to_numpy(dtype=float),
            summary["q75"].to_numpy(dtype=float),
            color=LIGHT,
            alpha=0.35,
        )
        ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
        ax.set_title(f"{role.title()} states (n={subset['example_id'].nunique()})")
        ax.set_xlabel(
            "Sentence position relative to largest increase\n"
            "in full-trace action probability"
        )
        ax.set_ylim(-0.02, 1.02)
        ax.grid(color=GRID, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Probability assigned to the full-trace action")
    fig.suptitle("Final-Action Probability Around Its Largest Increase")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_boundary_agreement(alignments: pd.DataFrame, output: Path) -> None:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.2, 5.3))
    for role, color in (("control", CONTROL), ("failure", FAILURE)):
        subset = alignments[alignments["matched_role"] == role]
        ax.scatter(
            subset["final_action_jump_progress"],
            subset["stable_action_progress"],
            label=f"{role.title()} (n={len(subset)})",
            color=color,
            edgecolor=DARK,
            linewidth=0.4,
            alpha=0.85,
        )
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(
        "Largest final-action probability increase\n"
        "(fraction of reasoning characters)"
    )
    ax.set_ylabel(
        "First position where the recommended action remains stable\n"
        "(fraction of reasoning characters)"
    )
    ax.set_title("Do Probability Jumps and Stable Action Selection Agree?")
    ax.legend(frameon=False)
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_activation_similarity_transition(
    geometry: pd.DataFrame,
    change_points: pd.DataFrame,
    output: Path,
    *,
    window: int = 5,
) -> None:
    primary = geometry[
        (geometry["layer"].astype(int) == 15)
        & (geometry["representation"] == "sentence_mean")
    ].copy()
    boundaries = change_points[
        change_points["boundary_type"] == "largest_distribution_change"
    ]
    rows: list[dict[str, Any]] = []
    for boundary in boundaries.itertuples():
        state = primary[primary["example_id"] == boundary.example_id]
        for offset in range(-window, window + 1):
            match = state[
                state["reasoning_step_idx"].astype(int)
                == int(boundary.boundary_reasoning_step_idx) + offset
            ]
            if match.empty:
                continue
            rows.append(
                {
                    "relative_sentence_offset": offset,
                    "previous_mean_cosine": float(
                        match.iloc[0]["previous_mean_cosine"]
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("relative_sentence_offset")["previous_mean_cosine"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    x = summary["relative_sentence_offset"].to_numpy(dtype=float)
    ax.plot(
        x,
        summary["median"],
        color=BLUE,
        marker="o",
        linewidth=2,
        markersize=4,
    )
    ax.fill_between(
        x,
        summary["q25"].to_numpy(dtype=float),
        summary["q75"].to_numpy(dtype=float),
        color=LIGHT,
        alpha=0.35,
    )
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel(
        "Sentence position relative to largest change\n"
        "in the four-action probability distribution"
    )
    ax.set_ylabel(
        "Cosine similarity to the mean activation\n"
        "of preceding reasoning sentences"
    )
    ax.set_title(
        "Activation Similarity Around the Largest Action-Distribution Change\n"
        f"Layer 15 sentence means, {boundaries['example_id'].nunique()} states"
    )
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def validate_inputs(
    positions: pd.DataFrame,
    events: pd.DataFrame,
    change_points: pd.DataFrame,
    alignments: pd.DataFrame,
    *,
    expected_states: int,
    expected_trajectories: int,
    expected_positions: int,
) -> list[str]:
    checks = [
        (
            positions["example_id"].nunique() == expected_states,
            f"states={positions['example_id'].nunique()} expected={expected_states}",
        ),
        (
            positions["trajectory_id"].nunique() == expected_trajectories,
            f"trajectories={positions['trajectory_id'].nunique()} expected={expected_trajectories}",
        ),
        (
            len(positions) == expected_positions,
            f"positions={len(positions)} expected={expected_positions}",
        ),
        (
            len(change_points) == expected_states * 3,
            f"change_points={len(change_points)} expected={expected_states * 3}",
        ),
        (
            bool(alignments["stable_boundary_reproduced"].all()),
            "all stored stable-action boundaries reproduced",
        ),
    ]
    expected_events = {
        "action_change": 682,
        "sustained_optimal_to_suboptimal": 115,
        "transient_optimal_to_suboptimal": 46,
        "suboptimal_to_optimal": 182,
        "commitment_onset": 46,
    }
    counts = events["event_type"].value_counts().to_dict()
    for event_type, expected in expected_events.items():
        checks.append(
            (
                counts.get(event_type, 0) == expected,
                f"{event_type}={counts.get(event_type, 0)} expected={expected}",
            )
        )
    failures = [message for passed, message in checks if not passed]
    if failures:
        raise ValueError("Validation failed: " + "; ".join(failures))
    return [f"PASS: {message}" for _, message in checks]


def validate_robust_outputs(
    position_metrics: pd.DataFrame,
    change_points: pd.DataFrame,
    predictions: pd.DataFrame,
    activation_sensitivity: pd.DataFrame,
    semantic_candidates: list[dict[str, Any]],
) -> list[str]:
    action_columns = [f"prob_{action.lower()}" for action in ACTIONS]
    action_probabilities = position_metrics[action_columns].to_numpy(dtype=float)
    checks = [
        (
            bool(
                np.isfinite(action_probabilities).all()
                and np.allclose(action_probabilities.sum(axis=1), 1.0)
            ),
            "all four-action probability vectors are finite and sum to one",
        ),
        (
            bool(
                (
                    change_points.groupby("example_id")[
                        "boundary_type"
                    ].nunique()
                    == 3
                ).all()
            ),
            "every state has exactly three boundary definitions",
        ),
        (
            predictions.groupby(
                ["boundary_type", "model", "trajectory_id"]
            )["fold"].nunique().max()
            == 1,
            "no trajectory appears in multiple held-out folds",
        ),
        (
            not predictions.duplicated(
                [
                    "boundary_type",
                    "model",
                    "example_id",
                    "reasoning_step_idx",
                ]
            ).any(),
            "held-out prediction rows are unique",
        ),
        (
            len(activation_sensitivity) == 6,
            "activation sensitivity covers three layers and two representations",
        ),
        (
            all(
                all(str(value).strip() for value in row["judge_input"].values())
                for row in semantic_candidates
            ),
            "all semantic-judge sentence pairs contain complete text",
        ),
        (
            all(
                set(row["judge_input"])
                == {
                    "previous_sentence",
                    "current_sentence",
                    "comparison_previous_sentence",
                    "comparison_current_sentence",
                }
                for row in semantic_candidates
            ),
            "semantic judge inputs do not expose action or outcome labels",
        ),
    ]
    failures = [message for passed, message in checks if not passed]
    if failures:
        raise ValueError("Robust-output validation failed: " + "; ".join(failures))
    return [f"PASS: {message}" for _, message in checks]


def write_reports(
    output: Path,
    *,
    alignments: pd.DataFrame,
    group_summary: pd.DataFrame,
    alignment_summary: pd.DataFrame,
    alignment_null: pd.DataFrame,
    stability: pd.DataFrame,
    matched_group_summary: pd.DataFrame,
    distribution_control_summary: pd.DataFrame,
    activation_summary: pd.DataFrame,
    belief_summary: pd.DataFrame,
    attention_summary: pd.DataFrame,
    next_sentence_summary: pd.DataFrame,
    activation_sensitivity: pd.DataFrame,
    validation_lines: list[str],
) -> None:
    overall = alignment_summary[alignment_summary["state_group"] == "all"].iloc[0]
    jump_stability = stability[
        stability["boundary_type"] == "final_action_jump"
    ]
    local_js = jump_stability["local_post_minus_pre_js"].dropna()
    failure = group_summary[
        (group_summary["boundary_type"] == "final_action_jump")
        & (group_summary["state_group"] == "failure")
    ].iloc[0]
    control = group_summary[
        (group_summary["boundary_type"] == "final_action_jump")
        & (group_summary["state_group"] == "control")
    ].iloc[0]
    alignment_null_all = alignment_null[
        alignment_null["state_group"] == "all"
    ].iloc[0]
    boundary_order = alignments["jump_minus_stable_sentences"]
    jump_before = int((boundary_order < 0).sum())
    jump_same = int((boundary_order == 0).sum())
    jump_after = int((boundary_order > 0).sum())
    jump_distribution = distribution_control_summary[
        (distribution_control_summary["boundary_type"] == "final_action_jump")
        & (distribution_control_summary["metric"] == "local_change_in_js_to_full_trace")
        & (distribution_control_summary["matched_role"] == "all")
    ].iloc[0]
    jump_activation = activation_summary[
        (activation_summary["boundary_type"] == "final_action_jump")
        & (activation_summary["metric"] == "previous_mean_cosine")
        & (activation_summary["matched_role"] == "all")
    ].iloc[0]
    jump_belief = belief_summary[
        (belief_summary["boundary_type"] == "final_action_jump")
        & (belief_summary["metric"] == "mean_belief_error")
        & (belief_summary["matched_role"] == "all")
    ].iloc[0]
    stable_attention = attention_summary[
        (attention_summary["boundary_type"] == "stable_action_boundary")
        & (attention_summary["layer"].astype(int) == 23)
        & (attention_summary["matched_role"] == "all")
    ].iloc[0]
    next_jump = next_sentence_summary[
        next_sentence_summary["boundary_type"] == "final_action_jump"
    ].iloc[0]
    next_distribution = next_sentence_summary[
        next_sentence_summary["boundary_type"] == "largest_distribution_change"
    ].iloc[0]
    sensitivity_positive = int(
        (activation_sensitivity["auroc_improvement"] > 0).sum()
    )
    lines = [
        "# Sentence-Level Action-Distribution Change Points",
        "",
        "## Scope",
        "",
        "- Model: GPT-OSS-20B",
        "- States: 46 matched DoorKey states, 23 failure and 23 control",
        "- Trajectories: 31",
        "- Sentence positions: 7,084",
        "- Action distribution: candidate-token logprobs at temperature 0.7, normalized over UP, DOWN, LEFT, and RIGHT",
        "- New model inference: none",
        "- Claim audit: `MAIN_FINDING_ASSESSMENT.md`",
        "",
        "## Main Results",
        "",
        f"- Median final-action jump abruptness ratio: {overall['median_final_action_jump_abruptness_ratio']:.3f} "
        f"(trajectory-bootstrap interval [{overall['abruptness_ratio_ci_low']:.3f}, "
        f"{overall['abruptness_ratio_ci_high']:.3f}]).",
        f"- Median absolute offset between the largest probability jump and stable-action boundary: "
        f"{overall['median_abs_jump_stable_offset_sentences']:.1f} sentences.",
        f"- Fraction of states whose boundaries are within three sentences: "
        f"{overall['fraction_jump_within_3_sentences']:.3f}.",
        f"- A random valid sentence is within three sentences of stable action selection in "
        f"{alignment_null_all['null_fraction_within_three_mean']:.3f} of states on average "
        f"(permutation p={alignment_null_all['permutation_p_fraction_at_least_as_high']:.4f}).",
        f"- The largest probability jump occurs before stable action selection in {jump_before}/46 states, "
        f"at the same sentence in {jump_same}/46, and after it in {jump_after}/46.",
        f"- Mean local post-minus-pre Jensen-Shannon distance to the full-trace distribution: "
        f"{local_js.mean():.4f} bits. Negative values mean the action distribution becomes more like "
        "the full-trace distribution after the jump.",
        f"- Mean jump position by reasoning-character progress: failure={failure['mean_boundary_character_progress']:.3f}, "
        f"control={control['mean_boundary_character_progress']:.3f}.",
        "",
        "## Conclusion",
        "",
        "The largest rise in final-action probability is temporally related to stable action selection beyond a random-position baseline, "
        "and the distribution becomes substantially closer to its full-trace value. However, the largest jump is only modestly larger "
        "than competing jumps and is not interchangeable with stable commitment. The two boundaries are more than three sentences apart "
        "in most states, and the largest jump usually comes first. The supported interpretation is a two-stage descriptive account: an "
        "action becomes much more probable, then later becomes the recommendation that remains stable.",
        "",
        "This does not yet identify what semantic operation causes either transition. The analysis uses truncated-prefix action elicitation "
        "and retrospective full-trace information, so causal validation requires sentence truncation or replacement around the selected boundaries.",
        "",
        "## Internal Correlates",
        "",
        f"- Jensen-Shannon distance to the full-trace distribution decreases by "
        f"{-jump_distribution['mean_event_minus_control']:.3f} bits more than at same-state progress-matched positions over the five-sentence window "
        f"(trajectory-bootstrap interval [{-jump_distribution['event_minus_control_ci_high']:.3f}, "
        f"{-jump_distribution['event_minus_control_ci_low']:.3f}]).",
        f"- Cosine similarity to the mean of preceding sentence activations is higher at the probability jump by "
        f"{jump_activation['mean_event_minus_control']:.4f} "
        f"(interval [{jump_activation['event_minus_control_ci_low']:.4f}, "
        f"{jump_activation['event_minus_control_ci_high']:.4f}]).",
        f"- Mean wall, key, and door belief error differs from matched positions by "
        f"{jump_belief['mean_event_minus_control']:.4f} "
        f"(interval [{jump_belief['event_minus_control_ci_low']:.4f}, "
        f"{jump_belief['event_minus_control_ci_high']:.4f}]).",
        f"- At layer 23, final-action attention to the seven-sentence window around stable action selection exceeds its matched window by "
        f"{stable_attention['mean_event_minus_control_attention']:.4f} "
        f"(interval [{stable_attention['event_minus_control_ci_low']:.4f}, "
        f"{stable_attention['event_minus_control_ci_high']:.4f}]).",
        f"- Adding current-sentence activation similarity to reasoning progress and action confidence changes held-out AUROC for predicting "
        f"a probability jump at the next sentence by {next_jump['auroc_improvement']:.3f} "
        f"(trajectory-bootstrap interval [{next_jump['auroc_improvement_ci_low']:.3f}, "
        f"{next_jump['auroc_improvement_ci_high']:.3f}]; within-state circular-shift p="
        f"{next_jump['within_state_circular_shift_p']:.4f}).",
        f"- For the next sentence's largest change in the four-action distribution, the same added feature improves AUROC by "
        f"{next_distribution['auroc_improvement']:.3f} "
        f"(interval [{next_distribution['auroc_improvement_ci_low']:.3f}, "
        f"{next_distribution['auroc_improvement_ci_high']:.3f}]; corrected p="
        f"{next_distribution['bonferroni_p_across_three_boundaries']:.4f}). "
        f"The AUROC direction is positive in {sensitivity_positive}/{len(activation_sensitivity)} layer-representation checks, "
        "but log-loss improvement is approximately zero.",
        "",
        "These comparisons use same-state progress-matched controls. They are supporting associations, not causal evidence.",
        "",
        "## Failure Versus Control States",
        "",
        f"- Mean probability-jump position is {failure['mean_boundary_character_progress']:.3f} for failure states and "
        f"{control['mean_boundary_character_progress']:.3f} for control states.",
        "- The paired failure-control estimates and bootstrap intervals are in `matched_group_difference_summary.csv`. "
        "The pilot should not be read as establishing a group difference unless the corresponding interval excludes zero.",
        "",
        "## Limitation",
        "",
        "The stored probabilities are normalized over the four action candidates. They do not report probability mass assigned to non-action text.",
        "The jump is selected from the same probability series used to measure its magnitude. The random-position comparison validates timing "
        "agreement, but the local probability change around the selected maximum is descriptive by construction.",
    ]
    (output / "run_report.md").write_text("\n".join(lines) + "\n")
    captions = """# Figure Captions

## final_action_probability_around_jump.png

Probability assigned to the action recommended after the complete reasoning trace, aligned to the sentence producing its largest increase. Position 0 is that sentence. Lines show medians and bands show interquartile ranges across states. Probabilities come from temperature-0.7 candidate-token logprobs normalized over UP, DOWN, LEFT, and RIGHT.

## probability_jump_vs_stable_commitment.png

Comparison of two retrospective action boundaries. The horizontal axis is the sentence producing the largest increase in probability assigned to the full-trace action. The vertical axis is the first sentence after which the highest-probability action remains equal to the full-trace action. Both positions are measured as the fraction of reasoning characters revealed. Points on the diagonal indicate agreement.

## activation_similarity_around_distribution_change.png

Cosine similarity between each layer-15 sentence-mean activation and the mean activation of all preceding reasoning sentences, aligned to the sentence with the largest adjacent Jensen-Shannon divergence in the probability distribution over UP, DOWN, LEFT, and RIGHT. Position 0 is the distribution-change sentence. The line is the median and the band is the interquartile range across 46 states. This association does not establish that the activation change causes the action-distribution change.
"""
    (output / "FIGURE_CAPTIONS.md").write_text(captions)
    (output / "VALIDATION_REPORT.md").write_text(
        "# Validation Report\n\n"
        + "\n".join(f"- {line}" for line in validation_lines)
        + "\n"
    )


def parse_args() -> argparse.Namespace:
    base = Path(
        "outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--position-rows", type=Path, default=base / "position_rows.csv")
    parser.add_argument("--event-rows", type=Path, default=base / "event_rows.csv")
    parser.add_argument("--geometry-rows", type=Path, default=base / "geometry_rows.csv")
    parser.add_argument("--belief-rows", type=Path, default=base / "belief_rows.csv")
    parser.add_argument(
        "--sentences",
        type=Path,
        default=Path(
            "data/behavioral_probes/doorkey_chunking_validation/sentences.csv"
        ),
    )
    parser.add_argument(
        "--attention-index",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/attention_decision_relevance_v1/attention_index.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/action_distribution_change_points_v1"
        ),
    )
    parser.add_argument("--event-window", type=int, default=10)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--prediction-permutation-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-attention", action="store_true")
    parser.add_argument("--expected-states", type=int, default=46)
    parser.add_argument("--expected-trajectories", type=int, default=31)
    parser.add_argument("--expected-positions", type=int, default=7084)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figs").mkdir(exist_ok=True)

    positions_source = pd.read_csv(args.position_rows)
    events = pd.read_csv(args.event_rows)
    geometry = pd.read_csv(args.geometry_rows)
    beliefs = pd.read_csv(args.belief_rows)
    sentences = pd.read_csv(args.sentences)

    position_metrics = compute_position_metrics(positions_source)
    change_points = select_change_points(position_metrics)
    alignments = compute_boundary_offsets(change_points)
    aligned = build_event_aligned_rows(
        position_metrics,
        change_points,
        window=args.event_window,
    )
    controls = build_progress_matched_controls(position_metrics, change_points)
    stability = aggregate_post_boundary_stability(position_metrics, change_points)
    group_summary, alignment_summary = summarize_change_points(
        change_points,
        alignments,
        stability,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    alignment_null = random_position_alignment_null(
        alignments,
        repeats=max(args.bootstrap_repeats, 5000),
        seed=args.seed,
    )
    matched_group_summary = build_matched_group_difference_summary(
        change_points,
        stability,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    distribution_control_rows = build_distribution_boundary_control_rows(
        position_metrics,
        change_points,
        controls,
    )
    distribution_control_summary = summarize_event_control_rows(
        distribution_control_rows,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    activation_rows = build_activation_boundary_rows(
        geometry, change_points, controls
    )
    activation_summary = summarize_event_control_rows(
        activation_rows,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed + 10,
    )
    belief_rows = build_belief_boundary_rows(
        beliefs, change_points, controls
    )
    belief_summary = summarize_event_control_rows(
        belief_rows,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed + 20,
    )
    next_sentence_predictions, next_sentence_summary = (
        build_next_sentence_prediction(
            position_metrics,
            geometry,
            change_points,
            permutation_repeats=args.prediction_permutation_repeats,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=args.seed,
        )
    )
    activation_sensitivity = build_activation_representation_sensitivity(
        position_metrics,
        geometry,
        change_points,
        controls,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    attention_rows = pd.DataFrame()
    attention_summary = pd.DataFrame()
    if not args.skip_attention and args.attention_index.exists():
        attention_rows = compare_attention_at_boundaries(
            change_points,
            attention_index_path=args.attention_index,
        )
        attention_summary = summarize_attention_rows(
            attention_rows,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=args.seed + 30,
        )
    semantic_candidates = build_semantic_candidates(
        change_points, controls, sentences, position_metrics
    )

    validation_lines = validate_inputs(
        positions_source,
        events,
        change_points,
        alignments,
        expected_states=args.expected_states,
        expected_trajectories=args.expected_trajectories,
        expected_positions=args.expected_positions,
    )
    validation_lines.extend(
        validate_robust_outputs(
            position_metrics,
            change_points,
            next_sentence_predictions,
            activation_sensitivity,
            semantic_candidates,
        )
    )

    position_metrics.to_csv(
        args.output_dir / "position_distribution_metrics.csv", index=False
    )
    change_points.to_csv(args.output_dir / "state_change_points.csv", index=False)
    alignments.to_csv(args.output_dir / "boundary_alignment_rows.csv", index=False)
    aligned.to_csv(
        args.output_dir / "event_aligned_action_distribution.csv", index=False
    )
    controls.to_csv(args.output_dir / "progress_matched_controls.csv", index=False)
    stability.to_csv(args.output_dir / "post_boundary_stability.csv", index=False)
    group_summary.to_csv(args.output_dir / "state_group_summary.csv", index=False)
    alignment_summary.to_csv(
        args.output_dir / "boundary_alignment_summary.csv", index=False
    )
    alignment_null.to_csv(
        args.output_dir / "boundary_alignment_random_null.csv", index=False
    )
    matched_group_summary.to_csv(
        args.output_dir / "matched_group_difference_summary.csv", index=False
    )
    distribution_control_rows.to_csv(
        args.output_dir / "action_distribution_boundary_control_rows.csv",
        index=False,
    )
    distribution_control_summary.to_csv(
        args.output_dir / "action_distribution_boundary_control_summary.csv",
        index=False,
    )
    activation_rows.to_csv(
        args.output_dir / "activation_boundary_rows.csv", index=False
    )
    activation_summary.to_csv(
        args.output_dir / "activation_boundary_summary.csv", index=False
    )
    belief_rows.to_csv(
        args.output_dir / "belief_boundary_rows.csv", index=False
    )
    belief_summary.to_csv(
        args.output_dir / "belief_boundary_summary.csv", index=False
    )
    attention_rows.to_csv(
        args.output_dir / "attention_boundary_rows.csv", index=False
    )
    attention_summary.to_csv(
        args.output_dir / "attention_boundary_summary.csv", index=False
    )
    next_sentence_predictions.to_csv(
        args.output_dir / "next_sentence_boundary_predictions.csv", index=False
    )
    next_sentence_summary.to_csv(
        args.output_dir / "next_sentence_boundary_prediction_summary.csv",
        index=False,
    )
    activation_sensitivity.to_csv(
        args.output_dir / "activation_leading_indicator_sensitivity.csv",
        index=False,
    )
    write_jsonl(
        args.output_dir / "semantic_judge_candidates.jsonl",
        semantic_candidates,
    )

    plot_final_action_probability(
        aligned,
        args.output_dir / "figs" / "final_action_probability_around_jump.png",
    )
    plot_boundary_agreement(
        alignments,
        args.output_dir / "figs" / "probability_jump_vs_stable_commitment.png",
    )
    plot_activation_similarity_transition(
        geometry,
        change_points,
        args.output_dir
        / "figs"
        / "activation_similarity_around_distribution_change.png",
    )
    write_reports(
        args.output_dir,
        alignments=alignments,
        group_summary=group_summary,
        alignment_summary=alignment_summary,
        alignment_null=alignment_null,
        stability=stability,
        matched_group_summary=matched_group_summary,
        distribution_control_summary=distribution_control_summary,
        activation_summary=activation_summary,
        belief_summary=belief_summary,
        attention_summary=attention_summary,
        next_sentence_summary=next_sentence_summary,
        activation_sensitivity=activation_sensitivity,
        validation_lines=validation_lines,
    )
    manifest = {
        "status": "completed",
        "analysis": "sentence_level_action_distribution_change_points",
        "model": "openai/gpt-oss-20b",
        "temperature": 0.7,
        "top_p": 0.95,
        "seed": args.seed,
        "new_model_inference": False,
        "actions": list(ACTIONS),
        "states": positions_source["example_id"].nunique(),
        "trajectories": positions_source["trajectory_id"].nunique(),
        "positions": len(positions_source),
        "change_point_rows": len(change_points),
        "semantic_candidates": len(semantic_candidates),
        "attention_rows": len(attention_rows),
        "alignment_random_repeats": max(args.bootstrap_repeats, 5000),
        "prediction_permutation_repeats": args.prediction_permutation_repeats,
        "activation_sensitivity_rows": len(activation_sensitivity),
        "source_files": {
            str(path): sha256(path)
            for path in (
                args.position_rows,
                args.event_rows,
                args.geometry_rows,
                args.belief_rows,
                args.sentences,
            )
        },
    }
    write_json(args.output_dir / "run_manifest.json", manifest)
    print(f"Wrote change-point analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
