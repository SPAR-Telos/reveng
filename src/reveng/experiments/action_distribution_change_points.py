"""Sentence-level change-point metrics for categorical action distributions."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


def parse_action_distribution(
    value: str | dict[str, float],
    actions: Sequence[str] = ACTIONS,
) -> np.ndarray:
    """Parse one normalized action distribution in a fixed action order."""
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError("Action distribution must be a JSON object")
    missing = [action for action in actions if action not in parsed]
    extra = sorted(set(parsed) - set(actions))
    if missing or extra:
        raise ValueError(f"Action distribution keys differ: missing={missing}, extra={extra}")
    vector = np.asarray([float(parsed[action]) for action in actions], dtype=float)
    validate_action_distribution(vector)
    return vector


def validate_action_distribution(
    distribution: Sequence[float] | np.ndarray,
    tolerance: float = 1e-6,
) -> None:
    vector = np.asarray(distribution, dtype=float)
    if vector.ndim != 1 or vector.size != len(ACTIONS):
        raise ValueError(f"Expected {len(ACTIONS)} action probabilities")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Action distribution contains non-finite values")
    if np.any(vector < 0):
        raise ValueError("Action distribution contains negative values")
    if not math.isclose(float(vector.sum()), 1.0, abs_tol=tolerance):
        raise ValueError(f"Action distribution sums to {vector.sum():.9f}, not 1")


def jensen_shannon_bits(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> float:
    """Jensen-Shannon divergence with base-2 logarithms, bounded by one."""
    p = np.asarray(left, dtype=float)
    q = np.asarray(right, dtype=float)
    validate_action_distribution(p)
    validate_action_distribution(q)
    midpoint = 0.5 * (p + q)

    def kl_bits(source: np.ndarray, target: np.ndarray) -> float:
        mask = source > 0
        return float(np.sum(source[mask] * np.log2(source[mask] / target[mask])))

    return 0.5 * kl_bits(p, midpoint) + 0.5 * kl_bits(q, midpoint)


def total_variation(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> float:
    p = np.asarray(left, dtype=float)
    q = np.asarray(right, dtype=float)
    validate_action_distribution(p)
    validate_action_distribution(q)
    return float(0.5 * np.abs(p - q).sum())


def entropy_bits(distribution: Sequence[float] | np.ndarray) -> float:
    vector = np.asarray(distribution, dtype=float)
    validate_action_distribution(vector)
    positive = vector[vector > 0]
    return float(-np.sum(positive * np.log2(positive)))


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _ratio_largest_to_second(values: Iterable[float], *, positive_only: bool) -> float:
    usable = np.asarray(
        [
            float(value)
            for value in values
            if np.isfinite(float(value)) and (not positive_only or float(value) > 0)
        ],
        dtype=float,
    )
    if usable.size < 2:
        return float("nan")
    ordered = np.sort(usable)[::-1]
    if ordered[1] <= 0:
        return float("nan")
    return float(ordered[0] / ordered[1])


def compute_position_metrics(
    rows: pd.DataFrame,
    actions: Sequence[str] = ACTIONS,
) -> pd.DataFrame:
    """Compute adjacent and full-trace distribution metrics for every state."""
    required = {
        "example_id",
        "trajectory_id",
        "position_index",
        "reasoning_step_idx",
        "reasoning_progress",
        "action_probabilities_json",
        "action_label",
        "action_is_optimal",
        "final_full_trace_action",
        "commitment_onset",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Position rows missing required columns: {missing}")

    output: list[dict[str, Any]] = []
    for example_id, group in rows.groupby("example_id", sort=True):
        ordered = group.sort_values("position_index").copy()
        positions = ordered["position_index"].astype(int).to_numpy()
        if not np.array_equal(positions, np.arange(len(positions))):
            raise ValueError(f"{example_id}: position_index must be contiguous from zero")
        vectors = [
            parse_action_distribution(value, actions)
            for value in ordered["action_probabilities_json"]
        ]
        final_vector = vectors[-1]
        final_action = str(ordered.iloc[-1]["final_full_trace_action"]).upper()
        if final_action not in actions:
            raise ValueError(f"{example_id}: invalid full-trace action {final_action!r}")
        final_index = actions.index(final_action)
        previous: np.ndarray | None = None
        for source, vector in zip(ordered.to_dict("records"), vectors, strict=True):
            record = dict(source)
            for action, probability in zip(actions, vector, strict=True):
                record[f"prob_{action.lower()}"] = float(probability)
            record["final_action"] = final_action
            record["prob_final_action"] = float(vector[final_index])
            record["argmax_action"] = actions[int(np.argmax(vector))]
            record["computed_action_entropy_bits"] = entropy_bits(vector)
            record["computed_action_confidence"] = float(np.max(vector))
            record["js_to_full_trace"] = jensen_shannon_bits(vector, final_vector)
            record["tv_to_full_trace"] = total_variation(vector, final_vector)
            if previous is None:
                record.update(
                    {
                        "delta_prob_final_action": float("nan"),
                        "adjacent_js_bits": float("nan"),
                        "adjacent_total_variation": float("nan"),
                        "delta_action_entropy_bits": float("nan"),
                        "delta_action_confidence": float("nan"),
                        "argmax_action_changed": False,
                    }
                )
            else:
                previous_entropy = entropy_bits(previous)
                record.update(
                    {
                        "delta_prob_final_action": float(vector[final_index] - previous[final_index]),
                        "adjacent_js_bits": jensen_shannon_bits(previous, vector),
                        "adjacent_total_variation": total_variation(previous, vector),
                        "delta_action_entropy_bits": entropy_bits(vector) - previous_entropy,
                        "delta_action_confidence": float(np.max(vector) - np.max(previous)),
                        "argmax_action_changed": actions[int(np.argmax(previous))]
                        != actions[int(np.argmax(vector))],
                    }
                )
            output.append(record)
            previous = vector
    return pd.DataFrame(output)


def _earliest_max(frame: pd.DataFrame, column: str) -> pd.Series:
    usable = frame[frame[column].notna()]
    if usable.empty:
        raise ValueError(f"No valid values for {column}")
    maximum = float(usable[column].max())
    tied = usable[np.isclose(usable[column].astype(float), maximum, rtol=1e-12, atol=1e-12)]
    return tied.sort_values("position_index").iloc[0]


def _stable_action_position(frame: pd.DataFrame) -> int:
    ordered = frame.sort_values("position_index")
    final_action = str(ordered.iloc[-1]["argmax_action"])
    actions = ordered["argmax_action"].tolist()
    onset = len(actions) - 1
    while onset > 0 and actions[onset - 1] == final_action:
        onset -= 1
    return int(ordered.iloc[onset]["position_index"])


def select_change_points(position_metrics: pd.DataFrame) -> pd.DataFrame:
    """Select threshold-free change points and reproduce stable commitment."""
    output: list[dict[str, Any]] = []
    for example_id, group in position_metrics.groupby("example_id", sort=True):
        ordered = group.sort_values("position_index")
        if len(ordered) < 2:
            continue
        jump = _earliest_max(ordered, "delta_prob_final_action")
        divergence = _earliest_max(ordered, "adjacent_js_bits")
        stable_position = _stable_action_position(ordered)
        stable = ordered[ordered["position_index"].astype(int) == stable_position].iloc[0]
        stored = ordered[ordered["commitment_onset"].map(_as_bool).eq(True)]
        if len(stored) != 1:
            raise ValueError(f"{example_id}: expected one stored commitment, found {len(stored)}")
        stored_position = int(stored.iloc[0]["position_index"])
        common = {
            "example_id": example_id,
            "trajectory_id": ordered.iloc[0]["trajectory_id"],
            "step_index": int(ordered.iloc[0]["step_index"]),
            "matched_pair_id": ordered.iloc[0].get("matched_pair_id", ""),
            "matched_role": ordered.iloc[0].get("matched_role", ""),
            "trajectory_class": ordered.iloc[0].get("trajectory_class", ""),
            "primary_step_failure_mode": ordered.iloc[0].get("primary_step_failure_mode", ""),
            "n_positions": len(ordered),
            "n_reasoning_sentences": len(ordered) - 1,
            "final_action": ordered.iloc[-1]["final_action"],
            "final_action_is_optimal": _as_bool(ordered.iloc[-1]["action_is_optimal"]),
            "stored_stable_action_position": stored_position,
            "stable_boundary_reproduced": stable_position == stored_position,
            "final_action_jump_abruptness_ratio": _ratio_largest_to_second(
                ordered["delta_prob_final_action"], positive_only=True
            ),
            "distribution_change_abruptness_ratio": _ratio_largest_to_second(
                ordered["adjacent_js_bits"], positive_only=False
            ),
        }
        for boundary_type, row, score_name, score in (
            (
                "final_action_jump",
                jump,
                "delta_prob_final_action",
                jump["delta_prob_final_action"],
            ),
            (
                "largest_distribution_change",
                divergence,
                "adjacent_js_bits",
                divergence["adjacent_js_bits"],
            ),
            (
                "stable_action_boundary",
                stable,
                "stable_suffix",
                1.0,
            ),
        ):
            output.append(
                {
                    **common,
                    "boundary_type": boundary_type,
                    "boundary_position": int(row["position_index"]),
                    "boundary_reasoning_step_idx": int(row["reasoning_step_idx"]),
                    "boundary_reasoning_progress": float(row["reasoning_progress"]),
                    "boundary_score_name": score_name,
                    "boundary_score": float(score),
                    "boundary_action": row["argmax_action"],
                    "boundary_action_is_optimal": _as_bool(row["action_is_optimal"]),
                    "boundary_prob_final_action": float(row["prob_final_action"]),
                    "boundary_adjacent_js_bits": (
                        float(row["adjacent_js_bits"])
                        if pd.notna(row["adjacent_js_bits"])
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(output)


def compute_boundary_offsets(change_points: pd.DataFrame) -> pd.DataFrame:
    """Create one row per state with pairwise boundary offsets."""
    rows: list[dict[str, Any]] = []
    for example_id, group in change_points.groupby("example_id", sort=True):
        by_type = group.set_index("boundary_type")
        required = {
            "final_action_jump",
            "largest_distribution_change",
            "stable_action_boundary",
        }
        if set(by_type.index) != required:
            raise ValueError(f"{example_id}: missing change-point definitions")
        jump = by_type.loc["final_action_jump"]
        divergence = by_type.loc["largest_distribution_change"]
        stable = by_type.loc["stable_action_boundary"]
        rows.append(
            {
                "example_id": example_id,
                "trajectory_id": jump["trajectory_id"],
                "matched_pair_id": jump["matched_pair_id"],
                "matched_role": jump["matched_role"],
                "trajectory_class": jump["trajectory_class"],
                "primary_step_failure_mode": jump["primary_step_failure_mode"],
                "final_action": jump["final_action"],
                "final_action_is_optimal": jump["final_action_is_optimal"],
                "n_reasoning_sentences": int(jump["n_reasoning_sentences"]),
                "final_action_jump_position": int(jump["boundary_position"]),
                "distribution_change_position": int(divergence["boundary_position"]),
                "stable_action_position": int(stable["boundary_position"]),
                "final_action_jump_progress": float(jump["boundary_reasoning_progress"]),
                "distribution_change_progress": float(divergence["boundary_reasoning_progress"]),
                "stable_action_progress": float(stable["boundary_reasoning_progress"]),
                "jump_minus_stable_sentences": int(jump["boundary_position"])
                - int(stable["boundary_position"]),
                "distribution_change_minus_stable_sentences": int(divergence["boundary_position"])
                - int(stable["boundary_position"]),
                "jump_minus_stable_progress": float(jump["boundary_reasoning_progress"])
                - float(stable["boundary_reasoning_progress"]),
                "distribution_change_minus_stable_progress": float(
                    divergence["boundary_reasoning_progress"]
                )
                - float(stable["boundary_reasoning_progress"]),
                "jump_within_1_of_stable": abs(
                    int(jump["boundary_position"]) - int(stable["boundary_position"])
                )
                <= 1,
                "jump_within_3_of_stable": abs(
                    int(jump["boundary_position"]) - int(stable["boundary_position"])
                )
                <= 3,
                "jump_within_5_of_stable": abs(
                    int(jump["boundary_position"]) - int(stable["boundary_position"])
                )
                <= 5,
                "distribution_within_3_of_stable": abs(
                    int(divergence["boundary_position"]) - int(stable["boundary_position"])
                )
                <= 3,
                "final_action_jump_abruptness_ratio": jump[
                    "final_action_jump_abruptness_ratio"
                ],
                "distribution_change_abruptness_ratio": jump[
                    "distribution_change_abruptness_ratio"
                ],
                "stable_boundary_reproduced": bool(jump["stable_boundary_reproduced"]),
            }
        )
    return pd.DataFrame(rows)


def build_event_aligned_rows(
    position_metrics: pd.DataFrame,
    change_points: pd.DataFrame,
    *,
    window: int = 10,
) -> pd.DataFrame:
    by_example = {
        example_id: group.set_index(group["position_index"].astype(int))
        for example_id, group in position_metrics.groupby("example_id", sort=False)
    }
    output: list[dict[str, Any]] = []
    metric_columns = (
        "prob_final_action",
        "js_to_full_trace",
        "tv_to_full_trace",
        "adjacent_js_bits",
        "adjacent_total_variation",
        "computed_action_entropy_bits",
        "computed_action_confidence",
    )
    for boundary in change_points.itertuples():
        positions = by_example[boundary.example_id]
        center = int(boundary.boundary_position)
        for offset in range(-window, window + 1):
            position = center + offset
            if position not in positions.index:
                continue
            row = positions.loc[position]
            output.append(
                {
                    "example_id": boundary.example_id,
                    "trajectory_id": boundary.trajectory_id,
                    "matched_pair_id": boundary.matched_pair_id,
                    "matched_role": boundary.matched_role,
                    "boundary_type": boundary.boundary_type,
                    "boundary_position": center,
                    "relative_sentence_offset": offset,
                    "position_index": position,
                    "reasoning_progress": float(row["reasoning_progress"]),
                    **{column: row[column] for column in metric_columns},
                }
            )
    return pd.DataFrame(output)


def build_progress_matched_controls(
    position_metrics: pd.DataFrame,
    change_points: pd.DataFrame,
    *,
    exclusion_radius: int = 3,
) -> pd.DataFrame:
    """Select a deterministic same-state control outside all boundary windows."""
    output: list[dict[str, Any]] = []
    for example_id, boundaries in change_points.groupby("example_id", sort=True):
        positions = position_metrics[position_metrics["example_id"] == example_id].copy()
        forbidden = set(boundaries["boundary_position"].astype(int))
        for boundary in boundaries.itertuples():
            center = int(boundary.boundary_position)
            target_progress = float(boundary.boundary_reasoning_progress)
            candidates = positions[
                positions["position_index"].astype(int).map(
                    lambda position: all(
                        abs(position - other) > exclusion_radius for other in forbidden
                    )
                    and abs(position - center) > 2 * exclusion_radius
                )
            ].copy()
            if candidates.empty:
                continue
            candidates["progress_distance"] = (
                candidates["reasoning_progress"].astype(float) - target_progress
            ).abs()
            control = candidates.sort_values(
                ["progress_distance", "position_index"]
            ).iloc[0]
            output.append(
                {
                    "example_id": example_id,
                    "trajectory_id": boundary.trajectory_id,
                    "boundary_type": boundary.boundary_type,
                    "boundary_position": center,
                    "boundary_reasoning_progress": target_progress,
                    "control_position": int(control["position_index"]),
                    "control_reasoning_progress": float(control["reasoning_progress"]),
                    "control_progress_distance": float(control["progress_distance"]),
                }
            )
    return pd.DataFrame(output)


def trajectory_bootstrap_ci(
    frame: pd.DataFrame,
    *,
    value_column: str,
    statistic: str = "mean",
    repeats: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    valid = frame.dropna(subset=[value_column])
    grouped = {
        trajectory: group[value_column].astype(float).to_numpy()
        for trajectory, group in valid.groupby("trajectory_id")
    }
    if not grouped:
        return float("nan"), float("nan")
    trajectories = np.asarray(sorted(grouped), dtype=object)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(repeats):
        sampled = rng.choice(trajectories, size=len(trajectories), replace=True)
        values = np.concatenate([grouped[trajectory] for trajectory in sampled])
        estimates.append(
            float(np.median(values) if statistic == "median" else np.mean(values))
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def grouped_bootstrap_ci(
    frame: pd.DataFrame,
    *,
    value_column: str,
    group_column: str,
    statistic: str = "mean",
    repeats: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap a scalar statistic while retaining all rows in sampled groups."""
    valid = frame.dropna(subset=[value_column, group_column])
    grouped = {
        group_id: group[value_column].astype(float).to_numpy()
        for group_id, group in valid.groupby(group_column)
    }
    if not grouped:
        return float("nan"), float("nan")
    group_ids = np.asarray(sorted(grouped), dtype=object)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(repeats):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        values = np.concatenate([grouped[group_id] for group_id in sampled])
        estimates.append(
            float(np.median(values) if statistic == "median" else np.mean(values))
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def random_position_alignment_null(
    alignments: pd.DataFrame,
    *,
    repeats: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare jump-stable agreement with random valid positions in each state."""
    required = {
        "example_id",
        "matched_role",
        "n_reasoning_sentences",
        "stable_action_position",
        "jump_minus_stable_sentences",
    }
    missing = sorted(required - set(alignments.columns))
    if missing:
        raise ValueError(f"Alignment rows missing required columns: {missing}")

    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for role in ("all", "failure", "control"):
        subset = (
            alignments
            if role == "all"
            else alignments[alignments["matched_role"] == role]
        )
        if subset.empty:
            continue
        observed_abs = subset["jump_minus_stable_sentences"].abs().to_numpy()
        observed_median = float(np.median(observed_abs))
        observed_within_three = float(np.mean(observed_abs <= 3))
        null_medians: list[float] = []
        null_within_three: list[float] = []
        for _ in range(repeats):
            random_offsets: list[int] = []
            for row in subset.itertuples():
                # Position zero contains no reasoning sentence, so draw from 1..N.
                sampled_position = int(
                    rng.integers(1, int(row.n_reasoning_sentences) + 1)
                )
                random_offsets.append(
                    abs(sampled_position - int(row.stable_action_position))
                )
            null_medians.append(float(np.median(random_offsets)))
            null_within_three.append(float(np.mean(np.asarray(random_offsets) <= 3)))
        output.append(
            {
                "state_group": role,
                "n_states": subset["example_id"].nunique(),
                "random_repeats": repeats,
                "observed_median_absolute_offset_sentences": observed_median,
                "null_median_absolute_offset_mean": float(np.mean(null_medians)),
                "null_median_absolute_offset_ci_low": float(
                    np.quantile(null_medians, 0.025)
                ),
                "null_median_absolute_offset_ci_high": float(
                    np.quantile(null_medians, 0.975)
                ),
                "permutation_p_offset_at_least_as_close": float(
                    (1 + np.sum(np.asarray(null_medians) <= observed_median))
                    / (repeats + 1)
                ),
                "observed_fraction_within_three_sentences": observed_within_three,
                "null_fraction_within_three_mean": float(
                    np.mean(null_within_three)
                ),
                "null_fraction_within_three_ci_low": float(
                    np.quantile(null_within_three, 0.025)
                ),
                "null_fraction_within_three_ci_high": float(
                    np.quantile(null_within_three, 0.975)
                ),
                "permutation_p_fraction_at_least_as_high": float(
                    (
                        1 + np.sum(
                            np.asarray(null_within_three)
                            >= observed_within_three
                        )
                    )
                    / (repeats + 1)
                ),
            }
        )
    return pd.DataFrame(output)
