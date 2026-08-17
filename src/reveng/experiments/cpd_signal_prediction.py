"""Utilities for forecasting frozen BEAST labels and future action events."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


PRIMARY_BELIEFS = (
    "wall_left",
    "wall_right",
    "wall_up",
    "wall_down",
    "has_key",
    "door_open",
)


def _as_bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin(("true", "false", "1", "0"))
    if bool(invalid.any()):
        raise ValueError(
            f"Cannot parse Boolean values: {sorted(normalized[invalid].unique())}"
        )
    return normalized.isin(("true", "1"))


def add_belief_change_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize adjacent changes in the six current-state belief distributions.

    All features at row t compare the belief readout at t with t-1. They do not use
    the next sentence. Chosen-action consequence beliefs are excluded because their
    question changes when the recommendation changes.
    """
    required = {"example_id", "reasoning_step_idx"}
    for belief in PRIMARY_BELIEFS:
        required.update(
            {
                f"belief_{belief}_prob_yes",
                f"belief_{belief}_prob_no",
                f"belief_{belief}_prob_unknown",
                f"belief_{belief}_entropy",
                f"belief_{belief}_changed",
            }
        )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Belief feature frame missing columns: {missing}")

    result = frame.copy()
    result["belief_mean_probability_shift"] = np.nan
    result["belief_max_probability_shift"] = np.nan
    result["belief_mean_absolute_entropy_shift"] = np.nan
    result["belief_answer_changes"] = result[
        [f"belief_{belief}_changed" for belief in PRIMARY_BELIEFS]
    ].sum(axis=1)

    for _, indices in result.groupby("example_id", sort=False).groups.items():
        ordered = (
            result.loc[list(indices)].sort_values("reasoning_step_idx").index.to_numpy()
        )
        if len(ordered) < 2:
            continue
        probability_shifts = []
        entropy_shifts = []
        for belief in PRIMARY_BELIEFS:
            probability_columns = [
                f"belief_{belief}_prob_yes",
                f"belief_{belief}_prob_no",
                f"belief_{belief}_prob_unknown",
            ]
            probabilities = result.loc[ordered, probability_columns].to_numpy(float)
            # Total variation is half the L1 distance between two categorical beliefs.
            probability_shifts.append(
                0.5 * np.abs(np.diff(probabilities, axis=0)).sum(axis=1)
            )
            entropy = result.loc[ordered, f"belief_{belief}_entropy"].to_numpy(float)
            entropy_shifts.append(np.abs(np.diff(entropy)))
        probability_matrix = np.column_stack(probability_shifts)
        entropy_matrix = np.column_stack(entropy_shifts)
        result.loc[ordered[1:], "belief_mean_probability_shift"] = np.nanmean(
            probability_matrix, axis=1
        )
        result.loc[ordered[1:], "belief_max_probability_shift"] = np.nanmax(
            probability_matrix, axis=1
        )
        result.loc[ordered[1:], "belief_mean_absolute_entropy_shift"] = np.nanmean(
            entropy_matrix, axis=1
        )
    return result


def attach_frozen_change_points(
    frame: pd.DataFrame,
    position_probabilities: pd.DataFrame,
    detected_points: pd.DataFrame,
    point_stability: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the frozen offline BEAST probability and point indicators."""
    keys = ["example_id", "position_index"]
    required = set(keys + ["posterior_change_probability"])
    missing = sorted(required - set(position_probabilities.columns))
    if missing:
        raise ValueError(f"BEAST position table missing columns: {missing}")
    result = frame.merge(
        position_probabilities[keys + ["posterior_change_probability"]].rename(
            columns={"posterior_change_probability": "offline_beast_change_probability"}
        ),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if result["offline_beast_change_probability"].isna().any():
        raise ValueError("Some analysis rows have no BEAST position probability")

    point_keys = set(
        zip(
            detected_points["example_id"].astype(str),
            detected_points["position_index"].astype(int),
            strict=True,
        )
    )
    stable = _as_bool_series(point_stability["found_by_at_least_12_of_16_setups"])
    robust_keys = set(
        zip(
            point_stability.loc[stable, "example_id"].astype(str),
            point_stability.loc[stable, "position_index"].astype(int),
            strict=True,
        )
    )
    row_keys = list(
        zip(
            result["example_id"].astype(str),
            result["position_index"].astype(int),
            strict=True,
        )
    )
    result["offline_beast_change_point_here"] = [
        int(key in point_keys) for key in row_keys
    ]
    result["offline_robust_change_point_here"] = [
        int(key in robust_keys) for key in row_keys
    ]
    return result


def add_future_point_targets(
    frame: pd.DataFrame,
    detected_points: pd.DataFrame,
    point_stability: pd.DataFrame,
    *,
    horizons: Sequence[int] = (1, 3),
) -> pd.DataFrame:
    """Label whether a frozen BEAST point occurs strictly after the current row."""
    result = frame.copy()
    all_keys = set(
        zip(
            detected_points["example_id"].astype(str),
            detected_points["position_index"].astype(int),
            strict=True,
        )
    )
    stable = _as_bool_series(point_stability["found_by_at_least_12_of_16_setups"])
    robust_keys = set(
        zip(
            point_stability.loc[stable, "example_id"].astype(str),
            point_stability.loc[stable, "position_index"].astype(int),
            strict=True,
        )
    )
    maximum_by_example = result.groupby("example_id")["position_index"].max().to_dict()
    for horizon in horizons:
        complete = result["position_index"].astype(int) + horizon <= result[
            "example_id"
        ].map(maximum_by_example).astype(int)
        result[f"complete_change_point_horizon_{horizon}"] = complete.astype(int)
        primary: list[float] = []
        robust: list[float] = []
        for is_complete, example_id, position in zip(
            complete,
            result["example_id"].astype(str),
            result["position_index"].astype(int),
            strict=True,
        ):
            if not is_complete:
                primary.append(np.nan)
                robust.append(np.nan)
                continue
            future = [
                (example_id, position + offset) for offset in range(1, horizon + 1)
            ]
            primary.append(int(any(key in all_keys for key in future)))
            robust.append(int(any(key in robust_keys for key in future)))
        result[f"change_point_in_next_{horizon}"] = primary
        result[f"robust_change_point_in_next_{horizon}"] = robust
    return result


def add_future_commitment_targets(
    frame: pd.DataFrame,
    *,
    horizons: Sequence[int] = (1, 3),
) -> pd.DataFrame:
    """Add future commitment-onset outcomes and an at-risk indicator."""
    required = {"example_id", "position_index", "action_committed", "commitment_onset"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Commitment frame missing columns: {missing}")
    result = frame.copy()
    result["currently_uncommitted"] = (
        ~_as_bool_series(result["action_committed"])
    ).astype(int)
    onset_keys = set(
        zip(
            result.loc[
                _as_bool_series(result["commitment_onset"]), "example_id"
            ].astype(str),
            result.loc[
                _as_bool_series(result["commitment_onset"]), "position_index"
            ].astype(int),
            strict=True,
        )
    )
    maximum_by_example = result.groupby("example_id")["position_index"].max().to_dict()
    for horizon in horizons:
        values: list[float] = []
        for example_id, position in zip(
            result["example_id"].astype(str),
            result["position_index"].astype(int),
            strict=True,
        ):
            if position + horizon > int(maximum_by_example[example_id]):
                values.append(np.nan)
            else:
                values.append(
                    int(
                        any(
                            (example_id, position + offset) in onset_keys
                            for offset in range(1, horizon + 1)
                        )
                    )
                )
        result[f"commitment_onset_h{horizon}"] = values
    return result


def paired_model_comparison(
    predictions: pd.DataFrame,
    *,
    candidate: str,
    reference: str,
) -> pd.DataFrame:
    """Return row-aligned candidate and reference predictions."""
    keys = ["row_id", "example_id", "trajectory_id", "target", "outcome"]
    candidate_rows = predictions[predictions["model"].eq(candidate)][
        keys + ["predicted_probability"]
    ].rename(columns={"predicted_probability": "candidate_probability"})
    reference_rows = predictions[predictions["model"].eq(reference)][
        keys + ["predicted_probability"]
    ].rename(columns={"predicted_probability": "reference_probability"})
    return candidate_rows.merge(
        reference_rows, on=keys, how="inner", validate="one_to_one"
    )
