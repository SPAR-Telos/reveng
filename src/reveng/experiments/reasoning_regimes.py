"""Deterministic behavioral regimes for sentence-level reasoning trajectories."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


REGIMES = (
    "unsettled_optimal",
    "unsettled_suboptimal",
    "stable_optimal",
    "stable_suboptimal",
)
ACTION_COLUMNS = ("prob_up", "prob_down", "prob_left", "prob_right")


def as_bool(value: Any) -> bool:
    """Parse the boolean encodings used by repository CSV artifacts."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"not a boolean value: {value!r}")


def stable_action_position(frame: pd.DataFrame) -> int:
    """Return the first position in the final constant argmax-action suffix."""
    ordered = frame.sort_values("position_index")
    actions = ordered["argmax_action"].astype(str).tolist()
    if not actions:
        raise ValueError("cannot find a stable suffix in an empty trace")
    onset = len(actions) - 1
    while onset > 0 and actions[onset - 1] == actions[-1]:
        onset -= 1
    return int(ordered.iloc[onset]["position_index"])


def assign_regimes(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign one of four exclusive regimes to every valid trace position."""
    required = {"position_index", "argmax_action", "action_is_optimal"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing regime columns: {sorted(missing)}")
    ordered = frame.sort_values("position_index").copy()
    if ordered["position_index"].duplicated().any():
        raise ValueError("position_index must be unique within a trace")
    onset = stable_action_position(ordered)
    ordered["is_optimal"] = ordered["action_is_optimal"].map(as_bool)
    ordered["is_stable"] = ordered["position_index"].astype(int).ge(onset)
    ordered["regime"] = np.select(
        [
            ~ordered["is_stable"] & ordered["is_optimal"],
            ~ordered["is_stable"] & ~ordered["is_optimal"],
            ordered["is_stable"] & ordered["is_optimal"],
            ordered["is_stable"] & ~ordered["is_optimal"],
        ],
        REGIMES,
        default="",
    )
    if not ordered["regime"].isin(REGIMES).all():
        raise ValueError("every position must receive exactly one regime")
    ordered["stable_action_position"] = onset
    if set(ACTION_COLUMNS).issubset(ordered.columns):
        probabilities = ordered.loc[:, ACTION_COLUMNS].to_numpy(dtype=float)
        sorted_probabilities = np.sort(probabilities, axis=1)
        ordered["action_margin"] = (
            sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
        )
    return ordered


def collapsed_regimes(regimes: Iterable[str]) -> list[str]:
    """Collapse consecutive repeats while preserving transition order."""
    collapsed: list[str] = []
    for regime in regimes:
        if not collapsed or regime != collapsed[-1]:
            collapsed.append(regime)
    return collapsed


def recurrence_returns(regimes: Iterable[str]) -> int:
    """Count returns to a previously left regime in a collapsed state path."""
    collapsed = collapsed_regimes(regimes)
    seen: set[str] = set()
    returns = 0
    for regime in collapsed:
        if regime in seen:
            returns += 1
        seen.add(regime)
    return returns


def summarize_trace(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize stability, recurrence, and optimality transitions."""
    ordered = frame.sort_values("position_index")
    regimes = ordered["regime"].astype(str).tolist()
    optimal = ordered["is_optimal"].astype(bool).to_numpy()
    stable = ordered["is_stable"].astype(bool).to_numpy()
    losses = int(np.sum(optimal[:-1] & ~optimal[1:]))
    gains = int(np.sum(~optimal[:-1] & optimal[1:]))
    stable_positions = ordered.loc[stable]
    margins = (
        stable_positions["action_margin"].astype(float)
        if "action_margin" in stable_positions
        else pd.Series(dtype=float)
    )
    return {
        "n_positions": int(len(ordered)),
        "n_reasoning_sentences": int(max(len(ordered) - 1, 0)),
        "stable_action_position": int(ordered["stable_action_position"].iloc[0]),
        "stable_suffix_positions": int(stable.sum()),
        "stable_suffix_fraction": float(stable.mean()),
        "stable_suffix_at_least_3": bool(stable.sum() >= 3),
        "stable_suffix_at_least_5": bool(stable.sum() >= 5),
        "minimum_stable_action_margin": (
            float(margins.min()) if not margins.empty else float("nan")
        ),
        "ambiguous_stable_margin_fraction": (
            float((margins < 0.05).mean()) if not margins.empty else float("nan")
        ),
        "final_regime": regimes[-1],
        "final_action_optimal": bool(optimal[-1]),
        "regime_changes": int(len(collapsed_regimes(regimes)) - 1),
        "recurrence_returns": recurrence_returns(regimes),
        "recurrence_returns_per_100_positions": float(
            100.0 * recurrence_returns(regimes) / max(len(ordered) - 1, 1)
        ),
        "optimality_losses": losses,
        "optimality_losses_per_100_positions": float(
            100.0 * losses / max(len(ordered) - 1, 1)
        ),
        "optimality_gains": gains,
        "optimality_recovered": bool(losses > 0 and gains > 0),
    }


def transition_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Create one row per adjacent transition in a regime-labelled trace."""
    ordered = frame.sort_values("position_index").reset_index(drop=True)
    if len(ordered) < 2:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for index in range(len(ordered) - 1):
        current = ordered.iloc[index]
        following = ordered.iloc[index + 1]
        previous_regime = (
            str(ordered.iloc[index - 1]["regime"]) if index > 0 else "trace_start"
        )
        rows.append(
            {
                "position_index": int(current["position_index"]),
                "next_position_index": int(following["position_index"]),
                "reasoning_progress": float(current["reasoning_progress"]),
                "previous_regime": previous_regime,
                "current_regime": str(current["regime"]),
                "next_regime": str(following["regime"]),
                "is_self_transition": bool(current["regime"] == following["regime"]),
                "optimality_loss": bool(
                    current["is_optimal"] and not following["is_optimal"]
                ),
                "optimality_gain": bool(
                    not current["is_optimal"] and following["is_optimal"]
                ),
                "enters_stable_selection": bool(
                    not current["is_stable"] and following["is_stable"]
                ),
                **{
                    column: float(current[column])
                    for column in (*ACTION_COLUMNS, "computed_action_entropy_bits")
                    if column in ordered.columns and pd.notna(current[column])
                },
            }
        )
    return pd.DataFrame(rows)


def smoothed_transition_probabilities(
    transitions: pd.DataFrame,
    *,
    alpha: float = 0.5,
) -> dict[tuple[str, str], float]:
    """Estimate additive-smoothed first-order regime transition probabilities."""
    counts = transitions.groupby(["current_regime", "next_regime"]).size()
    totals = transitions.groupby("current_regime").size()
    probabilities: dict[tuple[str, str], float] = {}
    for current in REGIMES:
        denominator = float(totals.get(current, 0)) + alpha * len(REGIMES)
        for following in REGIMES:
            numerator = float(counts.get((current, following), 0)) + alpha
            probabilities[(current, following)] = numerator / denominator
    return probabilities
