"""Persistence-aware regime-transition prediction with semantic features."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


SEMANTIC_LABELS = (
    "state_readout",
    "route_planning",
    "verification",
    "correction",
    "new_inference",
    "consolidation",
    "restatement",
    "procedural_continuation",
    "action_commitment",
)
CUE_COLUMNS = (
    "explicitly_revises_prior_reasoning",
    "evaluates_prior_route_or_claim",
    "repeats_prior_content",
    "introduces_new_information_or_plan",
)
REGIMES = (
    "unsettled_optimal",
    "unsettled_suboptimal",
    "stable_optimal",
    "stable_suboptimal",
)
DESTINATIONS = (
    "switch_unsettled",
    "enter_stable_optimal",
    "enter_stable_suboptimal",
)


def parse_semantic_labels(value: Any) -> set[str]:
    """Parse and validate one JSON multi-label set."""
    if pd.isna(value):
        return set()
    labels = set(json.loads(str(value)))
    unknown = labels - set(SEMANTIC_LABELS)
    if unknown:
        raise ValueError(f"unknown semantic labels: {sorted(unknown)}")
    return labels


def semantic_feature_table(annotations: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Convert annotation label sets and cue fields to multi-hot columns."""
    required = {"example_id", "sentence_number", "semantic_labels", *CUE_COLUMNS}
    missing = required - set(annotations.columns)
    if missing:
        raise ValueError(f"missing semantic columns: {sorted(missing)}")
    output = annotations[["example_id", "sentence_number"]].copy()
    valid = annotations["semantic_labels"].notna()
    output[f"{prefix}__valid"] = valid.astype(float)
    parsed = annotations["semantic_labels"].map(parse_semantic_labels)
    for label in SEMANTIC_LABELS:
        output[f"{prefix}__label__{label}"] = parsed.map(
            lambda values, label=label: float(label in values)
        )
    for cue in CUE_COLUMNS:
        normalized = annotations[cue].astype(str).str.strip().str.lower()
        output[f"{prefix}__cue__{cue}"] = normalized.eq("yes").astype(float)
    return output


def combine_semantic_runs(
    original: pd.DataFrame,
    replicate: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    """Combine two multi-hot annotation runs using a prespecified rule."""
    keys = ["example_id", "sentence_number"]
    merged = original.merge(
        replicate,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_original", "_replicate"),
    )
    original_columns = [
        column
        for column in original.columns
        if column not in keys and not column.endswith("__valid")
    ]
    output = merged[keys].copy()
    valid_original = merged["original__valid"]
    valid_replicate = merged["replicate__valid"]
    output["semantic_valid"] = (valid_original * valid_replicate).astype(float)
    for original_column in original_columns:
        base = original_column.removeprefix("original__")
        left = merged[original_column].astype(float)
        right = merged[original_column.replace("original__", "replicate__")].astype(
            float
        )
        if mode == "original":
            value = left
        elif mode == "replicate":
            value = right
        elif mode == "intersection":
            value = np.minimum(left, right)
        elif mode == "union":
            value = np.maximum(left, right)
        elif mode == "mean":
            value = (left + right) / 2.0
        else:
            raise ValueError(f"unknown semantic combination: {mode}")
        output[f"semantic__{base}"] = value
    return output


def add_history_features(frame: pd.DataFrame, maximum_history: int) -> pd.DataFrame:
    """Add regime lags and time since the latest regime change within each trace."""
    if maximum_history < 1:
        raise ValueError("maximum_history must be positive")
    ordered = frame.sort_values(["example_id", "position_index"]).copy()
    grouped = ordered.groupby("example_id", sort=False)
    ordered["regime_lag_0"] = ordered["current_regime"].astype(str)
    for lag in range(1, maximum_history):
        ordered[f"regime_lag_{lag}"] = (
            grouped["current_regime"].shift(lag).fillna("trace_start")
        )
    durations = pd.Series(index=ordered.index, dtype=float)
    for _, rows in ordered.groupby("example_id", sort=False):
        duration = 0
        previous: str | None = None
        for index, regime in zip(rows.index, rows.current_regime, strict=True):
            duration = duration + 1 if regime == previous else 1
            durations.loc[index] = duration
            previous = str(regime)
    ordered["regime_duration"] = durations.astype(float)
    return ordered


def destination_type(current: str, following: str) -> str | None:
    """Map a non-self transition to a destination class."""
    if current == following:
        return None
    if following == "stable_optimal":
        return "enter_stable_optimal"
    if following == "stable_suboptimal":
        return "enter_stable_suboptimal"
    if current.startswith("unsettled_") and following.startswith("unsettled_"):
        return "switch_unsettled"
    raise ValueError(f"unsupported transition: {current} -> {following}")


def destination_to_regime(current: str, destination: str) -> str:
    """Convert a destination class back to the exact next regime."""
    if destination == "enter_stable_optimal":
        return "stable_optimal"
    if destination == "enter_stable_suboptimal":
        return "stable_suboptimal"
    if destination == "switch_unsettled":
        if current == "unsettled_optimal":
            return "unsettled_suboptimal"
        if current == "unsettled_suboptimal":
            return "unsettled_optimal"
    raise ValueError(f"cannot map {current!r} and {destination!r}")


def combine_hurdle_probabilities(
    current_regimes: list[str] | np.ndarray,
    change_probabilities: np.ndarray,
    destination_probabilities: np.ndarray,
) -> np.ndarray:
    """Combine change hazard and conditional destination into four-regime mass."""
    current = np.asarray(current_regimes, dtype=object)
    hazard = np.asarray(change_probabilities, dtype=float)
    destination = np.asarray(destination_probabilities, dtype=float)
    if destination.shape != (len(current), len(DESTINATIONS)):
        raise ValueError("destination probabilities have incompatible shape")
    output = np.zeros((len(current), len(REGIMES)), dtype=float)
    regime_index = {value: index for index, value in enumerate(REGIMES)}
    for row, regime in enumerate(current):
        output[row, regime_index[str(regime)]] = 1.0 - hazard[row]
        for column, destination_name in enumerate(DESTINATIONS):
            following = destination_to_regime(str(regime), destination_name)
            output[row, regime_index[following]] += (
                hazard[row] * destination[row, column]
            )
    return output
