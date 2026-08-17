"""Signed movement of action probability toward and away from optimal actions."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
PROBABILITY_COLUMNS = {action: f"prob_{action.lower()}" for action in ACTIONS}


def parse_optimal_actions(value: str | Sequence[str]) -> tuple[str, ...]:
    parsed = json.loads(value) if isinstance(value, str) else list(value)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("optimal_actions_json must contain a non-empty JSON list")
    actions = tuple(str(action).strip().upper() for action in parsed)
    invalid = sorted(set(actions) - set(ACTIONS))
    if invalid:
        raise ValueError(f"Invalid optimal actions: {invalid}")
    if len(set(actions)) != len(actions):
        raise ValueError("Optimal actions must not contain duplicates")
    return tuple(sorted(actions))


def add_probability_flow_columns(
    positions: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Add optimal-action probability and signed adjacent changes."""
    required_positions = {
        "example_id",
        "trajectory_id",
        "matched_pair_id",
        "matched_role",
        "position_index",
        "reasoning_progress",
        "action_is_optimal",
        *PROBABILITY_COLUMNS.values(),
    }
    missing = sorted(required_positions - set(positions.columns))
    if missing:
        raise ValueError(f"Position table missing columns: {missing}")
    if "example_id" not in metadata or "optimal_actions_json" not in metadata:
        raise ValueError("Metadata must contain example_id and optimal_actions_json")

    meta = metadata[["example_id", "optimal_actions_json"]].drop_duplicates()
    if meta["example_id"].duplicated().any():
        raise ValueError("Metadata contains conflicting rows for an example_id")
    output = positions.merge(meta, on="example_id", how="left", validate="many_to_one")
    if output["optimal_actions_json"].isna().any():
        missing_ids = sorted(
            output.loc[output["optimal_actions_json"].isna(), "example_id"].unique()
        )
        raise ValueError(f"Missing optimal-action metadata for: {missing_ids[:5]}")

    output = output.sort_values(["example_id", "position_index"]).copy()
    optimal_probabilities: list[float] = []
    normalized_actions: list[str] = []
    for row in output.itertuples(index=False):
        actions = parse_optimal_actions(row.optimal_actions_json)
        probability = sum(
            float(getattr(row, PROBABILITY_COLUMNS[action])) for action in actions
        )
        if (
            not np.isfinite(probability)
            or probability < -1e-6
            or probability > 1 + 1e-6
        ):
            raise ValueError(
                f"Invalid optimal-action probability for {row.example_id}: {probability}"
            )
        optimal_probabilities.append(float(np.clip(probability, 0.0, 1.0)))
        normalized_actions.append(json.dumps(actions))

    output["optimal_actions_json"] = normalized_actions
    output["optimal_action_probability"] = optimal_probabilities
    output["suboptimal_action_probability"] = 1.0 - output["optimal_action_probability"]
    output["change_in_optimal_action_probability"] = output.groupby(
        "example_id", sort=False
    )["optimal_action_probability"].diff()
    change = output["change_in_optimal_action_probability"]
    output["probability_moved_toward_optimal"] = change.clip(lower=0)
    output["probability_moved_away_from_optimal"] = (-change).clip(lower=0)
    output["probability_flow_direction"] = np.select(
        [change > 0, change < 0, change.notna()],
        ["toward optimal actions", "away from optimal actions", "no change"],
        default="not applicable",
    )
    return output


def summarize_states(flow_rows: pd.DataFrame) -> pd.DataFrame:
    """Return one auditable summary row per environment state."""
    summaries: list[dict[str, Any]] = []
    for example_id, group in flow_rows.groupby("example_id", sort=True):
        ordered = group.sort_values("position_index")
        probability = ordered["optimal_action_probability"].to_numpy(float)
        changes = (
            ordered["change_in_optimal_action_probability"].dropna().to_numpy(float)
        )
        if not len(changes):
            raise ValueError(f"{example_id} has no adjacent transition")
        total_absolute_change = float(np.abs(changes).sum())
        quarter = max(1, int(np.ceil(len(probability) / 4)))
        summaries.append(
            {
                "example_id": example_id,
                "trajectory_id": ordered.iloc[0]["trajectory_id"],
                "matched_pair_id": ordered.iloc[0]["matched_pair_id"],
                "matched_role": ordered.iloc[0]["matched_role"],
                "n_positions": len(ordered),
                "n_transitions": len(changes),
                "initial_optimal_action_probability": probability[0],
                "final_optimal_action_probability": probability[-1],
                "mean_optimal_action_probability": float(probability.mean()),
                "last_quarter_optimal_action_probability": float(
                    probability[-quarter:].mean()
                ),
                "net_change_in_optimal_action_probability": float(
                    probability[-1] - probability[0]
                ),
                "mean_probability_moved_toward_optimal": float(
                    np.maximum(changes, 0).mean()
                ),
                "mean_probability_moved_away_from_optimal": float(
                    np.maximum(-changes, 0).mean()
                ),
                "mean_absolute_change_in_optimal_action_probability": float(
                    np.abs(changes).mean()
                ),
                "fraction_transitions_toward_optimal": float(np.mean(changes > 0)),
                "fraction_transitions_away_from_optimal": float(np.mean(changes < 0)),
                "directional_efficiency": (
                    float((probability[-1] - probability[0]) / total_absolute_change)
                    if total_absolute_change > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(summaries)


def paired_bootstrap_summary(
    state_summary: pd.DataFrame,
    metrics: Iterable[str],
    *,
    repeats: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare failure and control states by resampling complete matched pairs."""
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for metric in metrics:
        pivot = state_summary.pivot(
            index="matched_pair_id", columns="matched_role", values=metric
        )
        if set(pivot.columns) != {"failure", "control"} or pivot.isna().any().any():
            raise ValueError(
                f"Metric {metric} does not have complete failure-control pairs"
            )
        failure = pivot["failure"].to_numpy(float)
        control = pivot["control"].to_numpy(float)
        differences = failure - control
        samples = rng.integers(0, len(pivot), size=(repeats, len(pivot)))
        boot_failure = failure[samples].mean(axis=1)
        boot_control = control[samples].mean(axis=1)
        boot_difference = differences[samples].mean(axis=1)
        output.append(
            {
                "metric": metric,
                "n_matched_pairs": len(pivot),
                "failure_mean": float(failure.mean()),
                "failure_ci_low": float(np.quantile(boot_failure, 0.025)),
                "failure_ci_high": float(np.quantile(boot_failure, 0.975)),
                "control_mean": float(control.mean()),
                "control_ci_low": float(np.quantile(boot_control, 0.025)),
                "control_ci_high": float(np.quantile(boot_control, 0.975)),
                "failure_minus_control": float(differences.mean()),
                "difference_ci_low": float(np.quantile(boot_difference, 0.025)),
                "difference_ci_high": float(np.quantile(boot_difference, 0.975)),
            }
        )
    return pd.DataFrame(output)


def threshold_sensitivity(
    flow_rows: pd.DataFrame,
    thresholds: Sequence[float] = (0.0, 0.01, 0.05, 0.10),
    *,
    repeats: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarize how often adjacent changes exceed prespecified magnitudes."""
    state_rows: list[dict[str, Any]] = []
    for (example_id, pair_id, role), group in flow_rows.groupby(
        ["example_id", "matched_pair_id", "matched_role"], sort=True
    ):
        changes = group["change_in_optimal_action_probability"].dropna().to_numpy(float)
        for threshold in thresholds:
            state_rows.append(
                {
                    "example_id": example_id,
                    "matched_pair_id": pair_id,
                    "matched_role": role,
                    "minimum_change": float(threshold),
                    "fraction_away": float(np.mean(changes < -threshold)),
                    "fraction_toward": float(np.mean(changes > threshold)),
                }
            )
    states = pd.DataFrame(state_rows)
    output: list[pd.DataFrame] = []
    for threshold, group in states.groupby("minimum_change", sort=True):
        summary = paired_bootstrap_summary(
            group,
            ["fraction_away", "fraction_toward"],
            repeats=repeats,
            seed=seed + int(round(threshold * 1000)),
        )
        summary.insert(0, "minimum_change", threshold)
        output.append(summary)
    return pd.concat(output, ignore_index=True)


def progress_bin_summary(
    flow_rows: pd.DataFrame,
    *,
    n_bins: int = 10,
    repeats: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate optimal-action probability over normalized reasoning progress."""
    rows = flow_rows.copy()
    rows["progress_bin"] = np.minimum(
        (rows["reasoning_progress"].astype(float) * n_bins).astype(int), n_bins - 1
    )
    states = rows.groupby(
        ["example_id", "matched_pair_id", "matched_role", "progress_bin"],
        as_index=False,
    )["optimal_action_probability"].mean()
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for progress_bin, group in states.groupby("progress_bin", sort=True):
        pivot = group.pivot(
            index="matched_pair_id",
            columns="matched_role",
            values="optimal_action_probability",
        ).dropna()
        for role in ("control", "failure"):
            values = pivot[role].to_numpy(float)
            samples = rng.integers(0, len(values), size=(repeats, len(values)))
            boot = values[samples].mean(axis=1)
            output.append(
                {
                    "progress_bin": int(progress_bin),
                    "progress_midpoint": (progress_bin + 0.5) / n_bins,
                    "matched_role": role,
                    "n_matched_pairs": len(values),
                    "mean_optimal_action_probability": float(values.mean()),
                    "ci_low": float(np.quantile(boot, 0.025)),
                    "ci_high": float(np.quantile(boot, 0.975)),
                }
            )
    return pd.DataFrame(output)


def build_window_changes(
    flow_rows: pd.DataFrame,
    *,
    window: int = 3,
) -> pd.DataFrame:
    """Compute post-minus-pre optimal-action probability around every valid position."""
    if window < 1:
        raise ValueError("window must be positive")
    output: list[dict[str, Any]] = []
    for example_id, group in flow_rows.groupby("example_id", sort=True):
        ordered = group.sort_values("position_index").reset_index(drop=True)
        positions = ordered["position_index"].astype(int).to_numpy()
        if not np.array_equal(positions, np.arange(len(ordered))):
            raise ValueError(
                f"{example_id}: position_index must be contiguous from zero"
            )
        probability = ordered["optimal_action_probability"].to_numpy(float)
        for position in range(window, len(ordered) - window + 1):
            pre = float(probability[position - window : position].mean())
            post = float(probability[position : position + window].mean())
            source = ordered.iloc[position]
            output.append(
                {
                    "example_id": example_id,
                    "trajectory_id": source["trajectory_id"],
                    "matched_pair_id": source["matched_pair_id"],
                    "matched_role": source["matched_role"],
                    "position_index": position,
                    "reasoning_progress": float(source["reasoning_progress"]),
                    "pre_window_optimal_action_probability": pre,
                    "post_window_optimal_action_probability": post,
                    "signed_window_change": post - pre,
                    "absolute_window_change": abs(post - pre),
                }
            )
    return pd.DataFrame(output)


def match_change_points_to_controls(
    window_rows: pd.DataFrame,
    change_points: pd.DataFrame,
    *,
    exclusion_radius: int = 3,
) -> pd.DataFrame:
    """Pair each valid change point with a unique same-state position of similar progress."""
    required = {"example_id", "position_index", "reasoning_progress"}
    missing = sorted(required - set(change_points.columns))
    if missing:
        raise ValueError(f"Change-point table missing columns: {missing}")
    cp_lookup = {
        example_id: set(group["position_index"].astype(int))
        for example_id, group in change_points.groupby("example_id", sort=True)
    }
    output: list[dict[str, Any]] = []
    for example_id, positions in cp_lookup.items():
        available = window_rows[window_rows["example_id"] == example_id].copy()
        events = available[available["position_index"].isin(positions)].sort_values(
            "reasoning_progress"
        )
        controls = available[
            ~available["position_index"].map(
                lambda value: any(
                    abs(int(value) - point) <= exclusion_radius for point in positions
                )
            )
        ].copy()
        used: set[int] = set()
        for event_number, event in enumerate(events.itertuples(index=False)):
            candidates = controls[~controls["position_index"].isin(used)].copy()
            if candidates.empty:
                raise ValueError(f"{example_id}: insufficient control positions")
            candidates["progress_difference"] = (
                candidates["reasoning_progress"] - event.reasoning_progress
            ).abs()
            control = candidates.sort_values(
                ["progress_difference", "position_index"]
            ).iloc[0]
            used.add(int(control["position_index"]))
            event_id = f"{example_id}__cp_{int(event.position_index):04d}"
            for row_type, row in (
                ("change point", event),
                ("matched position", control),
            ):
                record = row._asdict() if hasattr(row, "_asdict") else row.to_dict()
                record.update(
                    {
                        "event_id": event_id,
                        "row_type": row_type,
                        "change_point_position": int(event.position_index),
                        "progress_match_difference": abs(
                            float(
                                row.reasoning_progress
                                if hasattr(row, "reasoning_progress")
                                else row["reasoning_progress"]
                            )
                            - float(event.reasoning_progress)
                        ),
                    }
                )
                record.pop("progress_difference", None)
                output.append(record)
    return pd.DataFrame(output)


def clustered_paired_event_summary(
    pairs: pd.DataFrame,
    metrics: Sequence[str] = ("absolute_window_change", "signed_window_change"),
    *,
    repeats: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare paired event and control values, resampling complete trajectories."""
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    trajectories = pairs["trajectory_id"].drop_duplicates().to_numpy()
    for metric in metrics:
        pivot = pairs.pivot(index="event_id", columns="row_type", values=metric)
        event_meta = pairs.drop_duplicates("event_id").set_index("event_id")
        paired = pivot.join(event_meta[["trajectory_id"]], how="inner")
        if paired[["change point", "matched position"]].isna().any().any():
            raise ValueError(f"Incomplete change-point/control pairs for {metric}")
        grouped = paired.groupby("trajectory_id", sort=True).agg(
            event_sum=("change point", "sum"),
            control_sum=("matched position", "sum"),
            n_events=("change point", "size"),
        )
        samples = rng.integers(0, len(grouped), size=(repeats, len(grouped)))
        counts = grouped["n_events"].to_numpy(float)[samples].sum(axis=1)
        bootstrap_event = (
            grouped["event_sum"].to_numpy(float)[samples].sum(axis=1) / counts
        )
        bootstrap_control = (
            grouped["control_sum"].to_numpy(float)[samples].sum(axis=1) / counts
        )
        bootstrap_difference = bootstrap_event - bootstrap_control
        event_values = paired["change point"].to_numpy(float)
        control_values = paired["matched position"].to_numpy(float)
        output.append(
            {
                "metric": metric,
                "n_change_points": len(paired),
                "n_trajectories": len(trajectories),
                "change_point_mean": float(event_values.mean()),
                "change_point_ci_low": float(np.quantile(bootstrap_event, 0.025)),
                "change_point_ci_high": float(np.quantile(bootstrap_event, 0.975)),
                "matched_position_mean": float(control_values.mean()),
                "matched_position_ci_low": float(np.quantile(bootstrap_control, 0.025)),
                "matched_position_ci_high": float(
                    np.quantile(bootstrap_control, 0.975)
                ),
                "change_point_minus_matched": float(
                    (event_values - control_values).mean()
                ),
                "difference_ci_low": float(np.quantile(bootstrap_difference, 0.025)),
                "difference_ci_high": float(np.quantile(bootstrap_difference, 0.975)),
            }
        )
    return pd.DataFrame(output)


def matched_role_event_summary(
    pairs: pd.DataFrame,
    metrics: Sequence[str] = ("absolute_window_change", "signed_window_change"),
    *,
    repeats: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare state-group means at paired change-point or control positions."""
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for row_type, type_rows in pairs.groupby("row_type", sort=True):
        for metric in metrics:
            states = type_rows.groupby(
                ["matched_pair_id", "matched_role"], as_index=False
            )[metric].mean()
            pivot = states.pivot(
                index="matched_pair_id", columns="matched_role", values=metric
            ).dropna()
            control = pivot["control"].to_numpy(float)
            failure = pivot["failure"].to_numpy(float)
            differences = control - failure
            samples = rng.integers(0, len(pivot), size=(repeats, len(pivot)))
            boot_difference = differences[samples].mean(axis=1)
            output.append(
                {
                    "row_type": row_type,
                    "metric": metric,
                    "n_complete_matched_pairs": len(pivot),
                    "control_mean": float(control.mean()),
                    "failure_mean": float(failure.mean()),
                    "control_minus_failure": float(differences.mean()),
                    "difference_ci_low": float(np.quantile(boot_difference, 0.025)),
                    "difference_ci_high": float(np.quantile(boot_difference, 0.975)),
                }
            )
    return pd.DataFrame(output)
