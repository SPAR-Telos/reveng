"""Analysis-only pilot for action-relevant behavioral belief states."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


REGIME_ORDER = ("confidently_correct", "unresolved", "confidently_wrong")
REGIME_LABELS = {
    "confidently_correct": "Confidently correct",
    "unresolved": "Unresolved",
    "confidently_wrong": "Confidently wrong",
    "invalid": "Invalid",
}
ROLE_ORDER = ("chosen_wall", "hit_wall", "has_key_after")
METRICS = {
    "current_suboptimal": "current_suboptimal",
    "next3_optimality_loss": "next3_optimality_loss",
    "next3_recovery": "next3_recovery",
}

BLUE = "#1769aa"
DARK_BLUE = "#0b3c5d"
LIGHT_BLUE = "#9bd0f5"
PALE_BLUE = "#dbeafe"
GRID = "#e5eef5"
TEXT = "#172033"


@dataclass(frozen=True)
class PilotConfig:
    belief_rows_path: Path
    position_rows_path: Path
    output_dir: Path
    confidence_threshold: float = 0.80
    sensitivity_thresholds: tuple[float, ...] = (0.70, 0.80, 0.90)
    bootstrap_repeats: int = 5_000
    seed: int = 42


def parse_probabilities(value: Any) -> dict[str, float]:
    """Parse and normalize a yes/no/unknown probability distribution."""
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("probabilities must be a JSON object or mapping")
    required = ("yes", "no", "unknown")
    try:
        probabilities = {key: float(value[key]) for key in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("probabilities must contain numeric yes, no, and unknown") from exc
    if any(not math.isfinite(number) or number < 0 for number in probabilities.values()):
        raise ValueError("probabilities must be finite and non-negative")
    total = sum(probabilities.values())
    if total <= 0:
        raise ValueError("probabilities must have positive total mass")
    return {key: number / total for key, number in probabilities.items()}


def relevant_question_ids(action_label: str) -> dict[str, str]:
    direction = str(action_label).strip().lower()
    if direction not in {"left", "right", "up", "down"}:
        raise ValueError(f"unsupported action label: {action_label!r}")
    return {
        "chosen_wall": f"wall_{direction}",
        "hit_wall": f"hit_wall_after_{direction}",
        "has_key_after": f"has_key_after_{direction}",
    }


def classify_belief(
    probabilities: Mapping[str, float], ground_truth: str, threshold: float
) -> dict[str, Any]:
    """Classify one factual belief while retaining its component diagnostics."""
    truth = str(ground_truth).strip().lower()
    if truth not in {"yes", "no"}:
        raise ValueError(f"ground truth must be yes or no, got {ground_truth!r}")
    probs = parse_probabilities(probabilities)
    factual_choice = "yes" if probs["yes"] > probs["no"] else "no"
    factual_tie = math.isclose(probs["yes"], probs["no"], abs_tol=1e-12)
    factual_confidence = max(probs["yes"], probs["no"])
    prefers_unknown = probs["unknown"] >= factual_confidence
    confidently_factual = factual_confidence >= threshold and not prefers_unknown and not factual_tie
    if confidently_factual and factual_choice != truth:
        state = "confidently_wrong"
    elif confidently_factual and factual_choice == truth:
        state = "confidently_correct"
    else:
        state = "unresolved"
    return {
        "state": state,
        "p_yes": probs["yes"],
        "p_no": probs["no"],
        "p_unknown": probs["unknown"],
        "truth": truth,
        "factual_choice": "tie" if factual_tie else factual_choice,
        "factual_confidence": factual_confidence,
        "prefers_unknown": prefers_unknown,
        "confidently_factual": confidently_factual,
        "is_correct": factual_choice == truth and not factual_tie,
    }


def aggregate_regime(states: Iterable[str]) -> str:
    states = tuple(states)
    if len(states) != 3 or any(state == "invalid" for state in states):
        return "invalid"
    if "confidently_wrong" in states:
        return "confidently_wrong"
    if "unresolved" in states:
        return "unresolved"
    if all(state == "confidently_correct" for state in states):
        return "confidently_correct"
    return "invalid"


def _parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def add_next_three_outcomes(positions: pd.DataFrame) -> pd.DataFrame:
    """Add outcomes only where three later sentences exist in the same example."""
    result = positions.copy()
    result["current_suboptimal"] = ~result["action_is_optimal"].map(_parse_bool)
    result["has_complete_next3"] = False
    result["next3_optimality_loss"] = np.nan
    result["next3_recovery"] = np.nan
    for _, indices in result.groupby("example_id", sort=False).groups.items():
        ordered = result.loc[list(indices)].sort_values("position_index").index.tolist()
        optimal = result.loc[ordered, "action_is_optimal"].map(_parse_bool).tolist()
        for offset, index in enumerate(ordered):
            future = optimal[offset + 1 : offset + 4]
            if len(future) != 3:
                continue
            result.at[index, "has_complete_next3"] = True
            if optimal[offset]:
                result.at[index, "next3_optimality_loss"] = float(any(not item for item in future))
            else:
                result.at[index, "next3_recovery"] = float(any(future))
    return result


def _belief_lookup(beliefs: pd.DataFrame) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    keys = ["example_id", "position_index", "question_id"]
    duplicates = beliefs.duplicated(keys, keep=False)
    if duplicates.any():
        sample = beliefs.loc[duplicates, keys].head(3).to_dict("records")
        raise ValueError(f"duplicate belief rows for position/question keys: {sample}")
    return {
        (str(row.example_id), int(row.position_index), str(row.question_id)): row._asdict()
        for row in beliefs.itertuples(index=False)
    }


def classify_positions(
    positions: pd.DataFrame, beliefs: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    required_positions = {
        "example_id",
        "trajectory_id",
        "position_index",
        "action_label",
        "action_is_optimal",
    }
    required_beliefs = {
        "example_id",
        "position_index",
        "question_id",
        "probabilities_json",
        "ground_truth_key",
    }
    if missing := required_positions - set(positions.columns):
        raise ValueError(f"position table is missing columns: {sorted(missing)}")
    if missing := required_beliefs - set(beliefs.columns):
        raise ValueError(f"belief table is missing columns: {sorted(missing)}")
    lookup = _belief_lookup(beliefs)
    classified: list[dict[str, Any]] = []
    for source in positions.to_dict("records"):
        row = dict(source)
        role_states: list[str] = []
        for role, question_id in relevant_question_ids(row["action_label"]).items():
            prefix = f"{role}_belief"
            row[f"{prefix}_question_id"] = question_id
            belief = lookup.get((str(row["example_id"]), int(row["position_index"]), question_id))
            if belief is None:
                diagnostic = {"state": "invalid"}
                row[f"{prefix}_invalid_reason"] = "missing_row"
            else:
                try:
                    diagnostic = classify_belief(
                        belief["probabilities_json"], belief["ground_truth_key"], threshold
                    )
                    row[f"{prefix}_invalid_reason"] = ""
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    diagnostic = {"state": "invalid"}
                    row[f"{prefix}_invalid_reason"] = str(exc)
            role_states.append(str(diagnostic["state"]))
            for key in (
                "state",
                "p_yes",
                "p_no",
                "p_unknown",
                "truth",
                "factual_choice",
                "factual_confidence",
                "prefers_unknown",
                "confidently_factual",
                "is_correct",
            ):
                row[f"{prefix}_{key}"] = diagnostic.get(key, np.nan)
        row["confidence_threshold"] = threshold
        row["epistemic_regime"] = aggregate_regime(role_states)
        row["n_confidently_wrong_beliefs"] = role_states.count("confidently_wrong")
        row["n_unresolved_beliefs"] = role_states.count("unresolved")
        row["n_confidently_correct_beliefs"] = role_states.count("confidently_correct")
        row["has_confidently_wrong_belief"] = "confidently_wrong" in role_states
        row["has_unresolved_belief"] = "unresolved" in role_states
        classified.append(row)
    return add_next_three_outcomes(pd.DataFrame(classified))


def _group_arrays(
    frame: pd.DataFrame, metric: str, trajectory_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    usable = frame.dropna(subset=[metric])
    grouped = usable.groupby("trajectory_id")[metric].agg(["sum", "count"])
    sums = grouped["sum"].reindex(trajectory_ids, fill_value=0).to_numpy(float)
    counts = grouped["count"].reindex(trajectory_ids, fill_value=0).to_numpy(float)
    return sums, counts


def trajectory_bootstrap_rate(
    frame: pd.DataFrame, metric: str, *, repeats: int, seed: int
) -> tuple[float, float, float]:
    usable = frame.dropna(subset=[metric])
    if usable.empty:
        return math.nan, math.nan, math.nan
    point = float(usable[metric].mean())
    trajectory_ids = sorted(frame["trajectory_id"].astype(str).unique())
    sums, counts = _group_arrays(frame, metric, trajectory_ids)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(trajectory_ids), [1 / len(trajectory_ids)] * len(trajectory_ids), size=repeats)
    denominators = weights @ counts
    valid = denominators > 0
    estimates = (weights[valid] @ sums) / denominators[valid]
    if estimates.size == 0:
        return point, math.nan, math.nan
    low, high = np.quantile(estimates, [0.025, 0.975])
    return point, float(low), float(high)


def trajectory_bootstrap_contrast(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    metric: str,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float, float, float, float]:
    usable_a = frame_a.dropna(subset=[metric])
    usable_b = frame_b.dropna(subset=[metric])
    if usable_a.empty or usable_b.empty:
        return (math.nan,) * 5
    rate_a = float(usable_a[metric].mean())
    rate_b = float(usable_b[metric].mean())
    trajectory_ids = sorted(
        set(frame_a["trajectory_id"].astype(str)) | set(frame_b["trajectory_id"].astype(str))
    )
    sums_a, counts_a = _group_arrays(frame_a, metric, trajectory_ids)
    sums_b, counts_b = _group_arrays(frame_b, metric, trajectory_ids)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(trajectory_ids), [1 / len(trajectory_ids)] * len(trajectory_ids), size=repeats)
    denom_a, denom_b = weights @ counts_a, weights @ counts_b
    valid = (denom_a > 0) & (denom_b > 0)
    estimates = (weights[valid] @ sums_a) / denom_a[valid] - (weights[valid] @ sums_b) / denom_b[valid]
    if estimates.size == 0:
        return rate_a, rate_b, rate_a - rate_b, math.nan, math.nan
    low, high = np.quantile(estimates, [0.025, 0.975])
    return rate_a, rate_b, rate_a - rate_b, float(low), float(high)


def summarize_states(
    rows: pd.DataFrame, *, repeats: int, seed: int
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for regime_index, regime in enumerate(REGIME_ORDER):
        subset = rows.loc[rows["epistemic_regime"] == regime]
        record: dict[str, Any] = {
            "confidence_threshold": float(rows["confidence_threshold"].iloc[0]),
            "epistemic_regime": regime,
            "n_positions": len(subset),
            "n_trajectories": subset["trajectory_id"].nunique(),
        }
        record["is_sufficiently_populated"] = bool(
            record["n_positions"] >= 20 and record["n_trajectories"] >= 5
        )
        for metric_index, metric in enumerate(METRICS):
            usable = subset.dropna(subset=[metric])
            point, low, high = trajectory_bootstrap_rate(
                subset,
                metric,
                repeats=repeats,
                seed=seed + 100 * regime_index + metric_index,
            )
            record[f"{metric}_n_eligible"] = len(usable)
            record[f"{metric}_n_events"] = int(usable[metric].sum()) if len(usable) else 0
            record[f"{metric}_rate"] = point
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        output.append(record)
    return pd.DataFrame(output)


def build_contrasts(
    rows: pd.DataFrame, summary: pd.DataFrame, *, repeats: int, seed: int
) -> pd.DataFrame:
    populated = summary.set_index("epistemic_regime")["is_sufficiently_populated"].to_dict()
    output: list[dict[str, Any]] = []
    for comparison_index, regime in enumerate(("confidently_wrong", "unresolved")):
        for metric_index, metric in enumerate(METRICS):
            values = trajectory_bootstrap_contrast(
                rows.loc[rows["epistemic_regime"] == regime],
                rows.loc[rows["epistemic_regime"] == "confidently_correct"],
                metric,
                repeats=repeats,
                seed=seed + 1_000 + comparison_index * 100 + metric_index,
            )
            rate, reference_rate, difference, low, high = values
            output.append(
                {
                    "confidence_threshold": float(rows["confidence_threshold"].iloc[0]),
                    "comparison_regime": regime,
                    "reference_regime": "confidently_correct",
                    "outcome": metric,
                    "comparison_rate": rate,
                    "reference_rate": reference_rate,
                    "rate_difference": difference,
                    "rate_difference_ci_low": low,
                    "rate_difference_ci_high": high,
                    "comparative_claim_allowed": bool(
                        populated.get(regime, False) and populated.get("confidently_correct", False)
                    ),
                    "is_primary_comparison": bool(
                        regime == "confidently_wrong" and metric == "next3_optimality_loss"
                    ),
                }
            )
    return pd.DataFrame(output)


def build_measurement_audit(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for scope in ("overall", *ROLE_ORDER):
        if scope == "overall":
            state_columns = [f"{role}_belief_state" for role in ROLE_ORDER]
            unknown_columns = [f"{role}_belief_prefers_unknown" for role in ROLE_ORDER]
            reason_columns = [f"{role}_belief_invalid_reason" for role in ROLE_ORDER]
            states = rows[state_columns].to_numpy().ravel()
            unknown = pd.to_numeric(rows[unknown_columns].stack(), errors="coerce")
            reasons = rows[reason_columns].fillna("").to_numpy().ravel()
            n_measurements = len(states)
            n_invalid = int(sum(value == "invalid" for value in states))
        else:
            states = rows[f"{scope}_belief_state"].to_numpy()
            unknown = pd.to_numeric(rows[f"{scope}_belief_prefers_unknown"], errors="coerce")
            reasons = rows[f"{scope}_belief_invalid_reason"].fillna("").to_numpy()
            n_measurements = len(states)
            n_invalid = int(sum(value == "invalid" for value in states))
        output.append(
            {
                "audit_scope": scope,
                "n_positions": len(rows),
                "n_belief_measurements": n_measurements,
                "n_missing_or_invalid_beliefs": n_invalid,
                "n_missing_belief_rows": int(sum(reason == "missing_row" for reason in reasons)),
                "n_invalid_probability_rows": int(
                    sum(bool(reason) and reason != "missing_row" for reason in reasons)
                ),
                "n_explicit_unknown_top": int(unknown.fillna(False).astype(bool).sum()),
                "explicit_unknown_top_rate": float(unknown.mean()) if unknown.notna().any() else math.nan,
                "n_valid_positions": int((rows["epistemic_regime"] != "invalid").sum()),
                "n_invalid_positions": int((rows["epistemic_regime"] == "invalid").sum()),
                **{
                    f"n_{regime}_positions": int((rows["epistemic_regime"] == regime).sum())
                    for regime in (*REGIME_ORDER, "invalid")
                },
            }
        )
    return pd.DataFrame(output)


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
        }
    )


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot the primary threshold with each panel's risk set made explicit."""
    _set_plot_style()
    colors = [LIGHT_BLUE, BLUE, DARK_BLUE]
    panels = (
        (
            "current_suboptimal",
            "A  Is the current action suboptimal?",
            "All valid positions",
        ),
        (
            "next3_optimality_loss",
            "B  Is optimality lost within 3 sentences?",
            "Only positions currently optimal",
        ),
        (
            "next3_recovery",
            "C  Is optimality recovered within 3 sentences?",
            "Only positions currently suboptimal",
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4), sharex=True, sharey=True)
    ordered = summary.set_index("epistemic_regime").loc[list(REGIME_ORDER)]
    y = np.arange(len(REGIME_ORDER))
    for axis, (metric, question, risk_set) in zip(axes, panels, strict=True):
        estimates = ordered[f"{metric}_rate"].to_numpy(float)
        lows = ordered[f"{metric}_ci_low"].to_numpy(float)
        highs = ordered[f"{metric}_ci_high"].to_numpy(float)
        events = ordered[f"{metric}_n_events"].to_numpy(int)
        eligible = ordered[f"{metric}_n_eligible"].to_numpy(int)
        for index in range(len(y)):
            if np.isfinite(estimates[index]):
                axis.errorbar(
                    estimates[index],
                    y[index],
                    xerr=[
                        [estimates[index] - lows[index]],
                        [highs[index] - estimates[index]],
                    ],
                    fmt="o",
                    color=colors[index],
                    ecolor=colors[index],
                    capsize=4,
                    markersize=7,
                    linewidth=1.6,
                )
                label_x = min(max(highs[index] + 0.018, estimates[index] + 0.018), 0.61)
                axis.text(
                    label_x,
                    y[index],
                    f"{estimates[index]:.1%}  ({events[index]:,}/{eligible[index]:,})",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    color=TEXT,
                )
        axis.set_title(f"{question}\n{risk_set}", loc="left", pad=12, linespacing=1.5)
        axis.set_xlim(0, 0.70)
        axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.set_xlabel("Conditional event rate")
        axis.grid(axis="x", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [REGIME_LABELS[regime] for regime in REGIME_ORDER])
    axes[0].invert_yaxis()
    fig.suptitle("What happens to the recommended action in each belief state?", fontsize=13, y=1.03)
    fig.text(
        0.5,
        -0.045,
        "Label = event percentage (events / eligible positions). Compare belief states within a panel: "
        "B and C use opposite starting conditions and are not direct counterparts. Bars are 95% "
        "trajectory-bootstrap intervals.",
        ha="center",
        fontsize=9,
        color=TEXT,
    )
    fig.tight_layout(w_pad=2.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sensitivity(sensitivity: pd.DataFrame, output_path: Path) -> None:
    """Show rate and category-membership sensitivity as an exact-value heatmap."""
    _set_plot_style()
    thresholds = sorted(sensitivity["confidence_threshold"].unique())
    panels = (
        ("current_suboptimal_rate", "Current action\nis suboptimal", "All positions"),
        (
            "next3_optimality_loss_rate",
            "Optimality lost\nwithin 3 sentences",
            "Currently optimal only",
        ),
        (
            "next3_recovery_rate",
            "Optimality recovered\nwithin 3 sentences",
            "Currently suboptimal only",
        ),
        ("regime_share", "Positions assigned\nto each belief state", "All positions"),
    )
    total_positions = int(
        sensitivity.loc[
            np.isclose(sensitivity["confidence_threshold"], thresholds[0]), "n_positions"
        ].sum()
    )
    frame = sensitivity.copy()
    frame["regime_share"] = frame["n_positions"] / total_positions
    color_map = LinearSegmentedColormap.from_list(
        "repository_blues", ["#ffffff", PALE_BLUE, LIGHT_BLUE, BLUE, DARK_BLUE]
    )
    fig, axes = plt.subplots(1, 4, figsize=(14.8, 3.7), sharey=True)
    for axis, (column, title, risk_set) in zip(axes, panels, strict=True):
        matrix = np.array(
            [
                [
                    frame.loc[
                        np.isclose(frame["confidence_threshold"], threshold)
                        & (frame["epistemic_regime"] == regime),
                        column,
                    ].iloc[0]
                    for threshold in thresholds
                ]
                for regime in REGIME_ORDER
            ]
        )
        axis.imshow(matrix, cmap=color_map, vmin=0, vmax=0.75, aspect="auto")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.1%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="semibold",
                    color="white" if value >= 0.38 else TEXT,
                )
        axis.set_title(f"{title}\n{risk_set}", pad=10, linespacing=1.35)
        axis.set_xticks(
            np.arange(len(thresholds)),
            [
                f"{threshold:.0%}\n{label}"
                for threshold, label in zip(
                    thresholds, ("lenient", "primary", "strict"), strict=True
                )
            ],
        )
        axis.tick_params(axis="both", length=0)
        axis.spines[:].set_visible(False)
    axes[0].set_yticks(
        np.arange(len(REGIME_ORDER)),
        [REGIME_LABELS[regime] for regime in REGIME_ORDER],
    )
    fig.suptitle(
        "Sensitivity to what counts as a confident factual belief",
        fontsize=13,
        y=1.06,
    )
    fig.text(
        0.5,
        -0.045,
        "Only the confidence boundary changes; the data and outcome definitions stay fixed. "
        "Cells are observed percentages. The primary figure gives uncertainty intervals at 80%.",
        ha="center",
        fontsize=9,
        color=TEXT,
    )
    fig.tight_layout(w_pad=1.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _format_rate(value: float) -> str:
    return "not estimable" if pd.isna(value) else f"{value:.1%}"


def _format_difference(value: float) -> str:
    return "not estimable" if pd.isna(value) else f"{100 * value:+.1f} percentage points"


def write_report(
    output_path: Path,
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    sensitivity: pd.DataFrame,
    audit: pd.DataFrame,
    config: PilotConfig,
) -> None:
    indexed = summary.set_index("epistemic_regime")
    primary = contrasts.loc[contrasts["is_primary_comparison"]].iloc[0]
    unresolved_loss = contrasts.loc[
        (contrasts["comparison_regime"] == "unresolved")
        & (contrasts["outcome"] == "next3_optimality_loss")
    ].iloc[0]
    current_rates = indexed["current_suboptimal_rate"].to_dict()
    recovery_rates = indexed["next3_recovery_rate"].to_dict()
    sufficiently_populated = [
        REGIME_LABELS[regime]
        for regime in REGIME_ORDER
        if bool(indexed.loc[regime, "is_sufficiently_populated"])
    ]
    sparse = [REGIME_LABELS[regime] for regime in REGIME_ORDER if regime not in {
        name for name in REGIME_ORDER if bool(indexed.loc[name, "is_sufficiently_populated"])
    }]
    audit_overall = audit.loc[audit["audit_scope"] == "overall"].iloc[0]
    threshold_lines = []
    threshold_results: dict[float, pd.DataFrame] = {}
    for threshold in sorted(sensitivity["confidence_threshold"].unique()):
        threshold_rows = sensitivity.loc[sensitivity["confidence_threshold"] == threshold]
        threshold_results[float(threshold)] = threshold_rows.set_index("epistemic_regime")
        counts = threshold_rows.set_index("epistemic_regime")["n_positions"]
        threshold_lines.append(
            f"At {threshold:.2f}, there were {int(counts['confidently_correct']):,} confidently correct, "
            f"{int(counts['unresolved']):,} unresolved, and {int(counts['confidently_wrong']):,} confidently wrong positions."
        )
    loss_differences = [
        threshold_results[threshold].loc[
            "confidently_wrong", "next3_optimality_loss_rate"
        ]
        - threshold_results[threshold].loc[
            "confidently_correct", "next3_optimality_loss_rate"
        ]
        for threshold in sorted(threshold_results)
    ]
    limitation = (
        "These are descriptive associations, not causal effects. The categories use simulator truth and a chosen "
        "confidence cutoff; the threshold table shows how much the result depends on that cutoff. Repeated positions "
        "are correlated, so intervals resample whole trajectories. End-of-example positions without three later "
        "sentences remain in the state table but are not used for loss or recovery. Explicit `unknown` is only a "
        "measurement diagnostic here; the analysis makes no claim about honest or dishonest reporting."
    )
    report = f"""# Behavioral epistemic-state pilot

## Question and method

This analysis asks whether three beliefs relevant to the model's currently recommended action form useful behavioral states: confidently correct, unresolved, or confidently wrong. At every reasoning sentence, it checks beliefs about a wall in the chosen direction, whether that action will hit a wall, and whether the model will have the key afterward. A factual answer counts as confident at probability {config.confidence_threshold:.2f}. A confident error takes priority over unresolved evidence, so a position that contains both remains classified as confidently wrong while both component flags remain in the row-level table.

The analysis uses {len(rows):,} existing reasoning positions from {rows['trajectory_id'].nunique():,} trajectories. It makes no new model calls and does not use change-point detection, semantic labels, or activation probes. Loss means that an action that is currently optimal becomes suboptimal at least once in the next three sentences. Recovery is the reverse. Both outcomes require three later sentences in the same example.

## Counts and result

At the primary threshold, {int(indexed.loc['confidently_correct', 'n_positions']):,} positions were confidently correct, {int(indexed.loc['unresolved', 'n_positions']):,} were unresolved, and {int(indexed.loc['confidently_wrong', 'n_positions']):,} were confidently wrong. {', '.join(sufficiently_populated) if sufficiently_populated else 'No regime'} met the planned minimum of 20 positions across five trajectories.{(' Sparse regimes are shown but not used for comparative claims: ' + ', '.join(sparse) + '.') if sparse else ''}

The primary comparison is optimality loss after confidently wrong versus confidently correct beliefs. The observed rates were {_format_rate(primary['comparison_rate'])} and {_format_rate(primary['reference_rate'])}, a difference of {_format_difference(primary['rate_difference'])} (95% trajectory-bootstrap interval {_format_difference(primary['rate_difference_ci_low'])} to {_format_difference(primary['rate_difference_ci_high'])}). This comparison {'passes' if primary['comparative_claim_allowed'] else 'does not pass'} the planned population gate.

The primary interval includes zero, so this pilot does not provide clear evidence that confidently wrong beliefs have a different near-term loss rate. For unresolved versus confidently correct positions, the loss-rate difference was {_format_difference(unresolved_loss['rate_difference'])} (95% interval {_format_difference(unresolved_loss['rate_difference_ci_low'])} to {_format_difference(unresolved_loss['rate_difference_ci_high'])}). The strongest descriptive pattern was instead current suboptimality: {_format_rate(current_rates['confidently_correct'])} for confidently correct, {_format_rate(current_rates['unresolved'])} for unresolved, and {_format_rate(current_rates['confidently_wrong'])} for confidently wrong positions. Recovery rates were {_format_rate(recovery_rates['confidently_correct'])}, {_format_rate(recovery_rates['unresolved'])}, and {_format_rate(recovery_rates['confidently_wrong'])}, respectively. The high suboptimal-action rate in the confidently correct group identifies belief-utilization failure candidates, but it could also mean these three local beliefs omit other information needed to choose the globally optimal action.

## Why 0.80, and how sensitive is the result?

The 0.80 boundary is a prespecified operational definition of high factual confidence, not a uniquely correct or literature-derived cutoff. It requires at least 80% probability on one factual answer, leaving at most 20% for the other factual answer and `unknown` together. It was fixed before examining these action outcomes. Because any hard boundary can move near-threshold positions between categories, the analysis repeats everything at 0.70 (lenient) and 0.90 (strict), changing no data or outcome definition.

{' '.join(threshold_lines)} The qualitative ordering was stable: confidently correct positions had the highest current-suboptimality rate, and confidently wrong positions had the highest subsequent-loss rate, at all three boundaries. The wrong-minus-correct loss difference was {_format_difference(loss_differences[0])}, {_format_difference(loss_differences[1])}, and {_format_difference(loss_differences[2])} at 0.70, 0.80, and 0.90. However, category membership changed substantially, so the exact rates and labels remain threshold-dependent. `figures/threshold_sensitivity.png` shows both the outcome rates and this reclassification directly.

Across all three required beliefs, explicit `unknown` was the top-probability answer in {int(audit_overall['n_explicit_unknown_top']):,} of {int(audit_overall['n_belief_measurements']):,} valid-or-attempted measurements ({_format_rate(audit_overall['explicit_unknown_top_rate'])}). There were {int(audit_overall['n_invalid_positions']):,} invalid positions. A position is marked invalid when at least one required belief row is missing or has malformed probabilities.

## Limitations

{limitation}

## Files

- `epistemic_state_rows.csv`: auditable sentence-level states, belief probabilities, and outcomes.
- `epistemic_state_summary.csv`: primary-threshold rates and trajectory-bootstrap intervals.
- `state_rate_contrasts.csv`: state-versus-confidently-correct rate differences.
- `threshold_sensitivity.csv`: the same summaries at 0.70, 0.80, and 0.90.
- `measurement_audit.csv`: missingness, explicit-unknown frequency, and state coverage.
- `figures/epistemic_state_action_outcomes.png`: primary-threshold conditional outcome rates, with each panel's denominator stated explicitly.
- `figures/threshold_sensitivity.png`: outcome rates and category membership at the 0.70, 0.80, and 0.90 boundaries.
"""
    output_path.write_text(report)


def run_pilot(config: PilotConfig) -> dict[str, Path]:
    if not 0 < config.confidence_threshold < 1:
        raise ValueError("confidence threshold must lie between zero and one")
    if config.bootstrap_repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    positions = pd.read_csv(config.position_rows_path)
    beliefs = pd.read_csv(config.belief_rows_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_threshold: dict[float, pd.DataFrame] = {}
    summaries: list[pd.DataFrame] = []
    thresholds = tuple(dict.fromkeys((*config.sensitivity_thresholds, config.confidence_threshold)))
    for threshold in sorted(thresholds):
        classified = classify_positions(positions, beliefs, threshold)
        rows_by_threshold[threshold] = classified
        summaries.append(
            summarize_states(classified, repeats=config.bootstrap_repeats, seed=config.seed + int(threshold * 1000))
        )
    rows = rows_by_threshold[config.confidence_threshold]
    sensitivity = pd.concat(summaries, ignore_index=True)
    summary = sensitivity.loc[
        np.isclose(sensitivity["confidence_threshold"], config.confidence_threshold)
    ].reset_index(drop=True)
    contrasts = build_contrasts(rows, summary, repeats=config.bootstrap_repeats, seed=config.seed)
    audit = build_measurement_audit(rows)

    paths = {
        "rows": config.output_dir / "epistemic_state_rows.csv",
        "summary": config.output_dir / "epistemic_state_summary.csv",
        "contrasts": config.output_dir / "state_rate_contrasts.csv",
        "sensitivity": config.output_dir / "threshold_sensitivity.csv",
        "audit": config.output_dir / "measurement_audit.csv",
        "report": config.output_dir / "run_report.md",
        "figure": config.output_dir / "figures" / "epistemic_state_action_outcomes.png",
        "sensitivity_figure": config.output_dir / "figures" / "threshold_sensitivity.png",
        "manifest": config.output_dir / "run_manifest.json",
    }
    rows.to_csv(paths["rows"], index=False)
    summary.to_csv(paths["summary"], index=False)
    contrasts.to_csv(paths["contrasts"], index=False)
    sensitivity.to_csv(paths["sensitivity"], index=False)
    audit.to_csv(paths["audit"], index=False)
    plot_summary(summary, paths["figure"])
    plot_threshold_sensitivity(sensitivity, paths["sensitivity_figure"])
    write_report(paths["report"], rows, summary, contrasts, sensitivity, audit, config)
    paths["manifest"].write_text(
        json.dumps(
            {
                "analysis": "behavioral_epistemic_state_pilot",
                "belief_rows_path": str(config.belief_rows_path),
                "position_rows_path": str(config.position_rows_path),
                "confidence_threshold": config.confidence_threshold,
                "sensitivity_thresholds": list(config.sensitivity_thresholds),
                "bootstrap_repeats": config.bootstrap_repeats,
                "seed": config.seed,
                "n_positions": len(rows),
                "n_trajectories": int(rows["trajectory_id"].nunique()),
                "used_gpu": False,
                "made_model_calls": False,
            },
            indent=2,
        )
        + "\n"
    )
    return paths
