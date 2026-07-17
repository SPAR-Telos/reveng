#!/usr/bin/env python3
"""Build reader-facing figures and tables for Experiments 1 and 2."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


DEFAULT_SENTENCE_EXP1 = Path(
    "outputs/experiment1_activation_monitor/weisheng_8_state_sentence_logprob_experiment1"
)
DEFAULT_PACKED_EXP1 = Path(
    "outputs/experiment1_activation_monitor/weisheng_8_state_packed_logprob_experiment1"
)
DEFAULT_SENTENCE_EXP2 = Path(
    "outputs/experiment2_behavioral_beliefs/weisheng_8_state_sentence_logprob_v1"
)
DEFAULT_PACKED_EXP2 = Path(
    "outputs/experiment2_behavioral_beliefs/weisheng_8_state_packed_logprob_v1"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/reader_facing/experiment1_2_weisheng_8_state_logprob_v1"
)

BLUE = "#1f5f9f"
MID_BLUE = "#5b9bd5"
LIGHT_BLUE = "#dbeafe"
LIGHTER_BLUE = "#edf6ff"
TEXT = "#172033"
GRID = "#d7e3f0"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def as_bool(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def fmt(value: Any, digits: int = 3) -> str:
    if value in {None, ""}:
        return ""
    try:
        out = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(out):
        return "nan"
    if abs(out) < 10 ** -digits and out != 0:
        return f"{out:.2e}"
    return f"{out:.{digits}f}"


def action_confidence(row: dict[str, str]) -> float | None:
    try:
        probs = json.loads(row.get("action_logprob_probs_json", "{}"))
    except Exception:
        return None
    values = [as_float(value) for value in probs.values()]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def setup_matplotlib() -> Any:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
        }
    )
    return plt


def summarize_run(
    *,
    label: str,
    exp1_dir: Path,
    exp2_dir: Path,
    activation_backed_prefixes: int | str,
) -> dict[str, Any]:
    actions = read_csv(exp1_dir / "prefix_action_rows.csv")
    events = read_csv(exp2_dir / "action_transition_rows.csv")
    commitment_rows = read_csv(exp1_dir / "trajectory_commitment_summary.csv")
    beliefs_path = exp2_dir / "belief_rows.csv"
    beliefs = read_csv(beliefs_path) if beliefs_path.exists() else []
    event_counts = Counter(row["event_type"] for row in events)
    states = {row["example_id"] for row in actions}
    commitments = sum(1 for row in commitment_rows if row.get("commitment_step") not in {"", None})
    return {
        "analysis unit": label,
        "states": len(states),
        "prefixes": len(actions),
        "action changes": sum(event_counts.values()),
        "optimal to suboptimal": (
            event_counts["sustained_optimal_to_suboptimal"]
            + event_counts["transient_optimal_to_suboptimal"]
        ),
        "sustained optimal to suboptimal": event_counts["sustained_optimal_to_suboptimal"],
        "suboptimal to optimal": event_counts["suboptimal_to_optimal_recovery"],
        "commitments": commitments,
        "belief probes": len(beliefs),
        "activation-backed prefixes": activation_backed_prefixes,
    }


def build_coverage_table(
    *,
    sentence_exp1: Path,
    packed_exp1: Path,
    sentence_exp2: Path,
    packed_exp2: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    activation_backed = ""
    report_path = sentence_exp1 / "sentence_activation_join_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        activation_backed = report.get("n_activation_backed_sentences", "")
    rows = [
        summarize_run(
            label="Sentences",
            exp1_dir=sentence_exp1,
            exp2_dir=sentence_exp2,
            activation_backed_prefixes=activation_backed,
        ),
        summarize_run(
            label="Packed prefixes",
            exp1_dir=packed_exp1,
            exp2_dir=packed_exp2,
            activation_backed_prefixes="not used",
        ),
    ]
    write_csv(output_dir / "table1_run_coverage_event_counts.csv", rows)
    lines = [
        "# Table 1. Run Coverage and Event Counts",
        "",
        "Summary of the eight DoorKey environment states used in the pilot. Sentence prefixes use canonical sentence boundaries; packed prefixes group nearby sentences. Action labels come from temperature 0.7 logprob queries over UP, DOWN, LEFT, and RIGHT. Commitments count states where the recommended action eventually matches the final full-trace action and remains stable. Belief probes count all behavioral belief-query rows.",
        "",
        "| analysis unit | states | prefixes | action changes | optimal to suboptimal | sustained optimal to suboptimal | suboptimal to optimal | commitments | belief probes | activation-backed prefixes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(row[key]) for key in rows[0])
            + " |"
        )
    (output_dir / "table1_run_coverage_event_counts.md").write_text("\n".join(lines) + "\n")
    return rows


def binned_state_summary(
    rows: list[dict[str, str]],
    *,
    value_fn: Any,
    progress_key: str = "reasoning_character_progress",
) -> list[dict[str, float]]:
    by_state_bin: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        progress = as_float(row.get(progress_key) or row.get("reasoning_progress"))
        value = value_fn(row)
        if progress is None or value is None:
            continue
        bin_index = min(9, max(0, int(progress * 10)))
        by_state_bin[(row["example_id"], bin_index)].append(value)
    by_bin: dict[int, list[float]] = defaultdict(list)
    for (_, bin_index), values in by_state_bin.items():
        by_bin[bin_index].append(sum(values) / len(values))
    out: list[dict[str, float]] = []
    for bin_index in sorted(by_bin):
        values = sorted(by_bin[bin_index])
        out.append(
            {
                "progress": (bin_index + 0.5) / 10,
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "mean": float(np.mean(values)),
                "n_states": len(values),
            }
        )
    return out


def plot_action_confidence(
    *,
    sentence_exp1: Path,
    packed_exp1: Path,
    output_dir: Path,
) -> None:
    plt = setup_matplotlib()
    fig_dir = output_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        ("Sentence prefixes", read_csv(sentence_exp1 / "prefix_action_rows.csv")),
        ("Packed prefixes", read_csv(packed_exp1 / "prefix_action_rows.csv")),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=True)
    summary_rows: list[dict[str, Any]] = []
    for ax, (title, rows) in zip(axes, panels):
        summary = binned_state_summary(rows, value_fn=action_confidence)
        for item in summary:
            summary_rows.append({"analysis_unit": title, **item})
        xs = [item["progress"] for item in summary]
        med = [item["median"] for item in summary]
        q25 = [item["q25"] for item in summary]
        q75 = [item["q75"] for item in summary]
        ax.plot(xs, med, marker="o", color=BLUE, linewidth=1.8)
        ax.fill_between(xs, q25, q75, color=LIGHT_BLUE, alpha=0.9)
        ax.set_title(f"{title}\n8 states, {len(rows)} prefixes")
        ax.set_xlabel("Fraction of reasoning characters revealed")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.99998, 1.000001)
        ticks = [0.99998, 0.999985, 0.99999, 0.999995, 1.0]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:.6f}" for tick in ticks])
    axes[0].set_ylabel("Probability assigned to the recommended action")
    fig.suptitle("Action Confidence Over Reasoning Progress", fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure1_action_confidence_over_reasoning_progress.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    write_csv(output_dir / "figure1_action_confidence_over_reasoning_progress.csv", summary_rows)


def plot_belief_errors_around_events(*, sentence_exp2: Path, output_dir: Path) -> None:
    plt = setup_matplotlib()
    fig_dir = output_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    positions = read_csv(sentence_exp2 / "state_uncertainty_position_rows.csv")
    events = read_csv(sentence_exp2 / "action_transition_rows.csv")
    position_by_key = {
        (row["example_id"], int(row["reasoning_step_idx"])): row for row in positions
    }
    event_types = [
        ("sustained_optimal_to_suboptimal", "Sustained optimal to suboptimal", BLUE),
        ("transient_optimal_to_suboptimal", "Transient optimal to suboptimal", MID_BLUE),
        ("suboptimal_to_optimal_recovery", "Suboptimal to optimal", "#b7d8f0"),
    ]
    offsets = list(range(-3, 4))
    summary_rows: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for event_type, label, color in event_types:
        selected = [row for row in events if row["event_type"] == event_type]
        means: list[float | None] = []
        for offset in offsets:
            values: list[float] = []
            for event in selected:
                idx = int(event["reasoning_step_idx"]) + offset
                row = position_by_key.get((event["example_id"], idx))
                value = as_float(row.get("primary_belief_error_rate") if row else None)
                if value is not None:
                    values.append(value)
            mean_value = sum(values) / len(values) if values else None
            means.append(mean_value)
            summary_rows.append(
                {
                    "event_type": event_type,
                    "sentence_offset": offset,
                    "n_events": len(selected),
                    "n_values": len(values),
                    "mean_belief_error_rate": mean_value if mean_value is not None else "",
                }
            )
        xs = [offset for offset, value in zip(offsets, means) if value is not None]
        ys = [value for value in means if value is not None]
        ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=f"{label} (n={len(selected)})")
    ax.axvline(0, color=TEXT, linestyle="--", linewidth=1)
    ax.set_title("Belief Errors Around Action Optimality Changes")
    ax.set_xlabel("Sentence offset from action optimality change")
    ax.set_ylabel("Mean error rate across wall, key, and door beliefs")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=True, facecolor="white")
    fig.tight_layout()
    fig.savefig(fig_dir / "figure2_belief_errors_around_action_optimality_changes.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    write_csv(output_dir / "figure2_belief_errors_around_action_optimality_changes.csv", summary_rows)


def plot_state_belief_uncertainty(*, sentence_exp2: Path, output_dir: Path) -> None:
    plt = setup_matplotlib()
    fig_dir = output_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(sentence_exp2 / "state_uncertainty_position_rows.csv")
    summary = binned_state_summary(
        rows,
        value_fn=lambda row: as_float(row.get("state_belief_entropy_mean_bits")),
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    xs = [item["progress"] for item in summary]
    means = [item["mean"] for item in summary]
    q25 = [item["q25"] for item in summary]
    q75 = [item["q75"] for item in summary]
    ax.plot(xs, means, marker="o", color=BLUE, linewidth=1.8)
    ax.fill_between(xs, q25, q75, color=LIGHT_BLUE, alpha=0.9)
    ax.set_title("State-Belief Uncertainty Over Reasoning Progress")
    ax.set_xlabel("Fraction of reasoning characters revealed")
    ax.set_ylabel("Mean state-belief entropy (bits)")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure3_state_belief_uncertainty_over_reasoning_progress.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    write_csv(output_dir / "figure3_state_belief_uncertainty_over_reasoning_progress.csv", summary)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40, 40)))


def fit_logistic(
    rows: list[dict[str, Any]],
    *,
    outcome: str,
    predictors: list[str],
    model_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x_values: list[list[float]] = []
    y_values: list[float] = []
    for row in rows:
        y_bool = as_bool(row.get(outcome))
        if y_bool is None:
            continue
        xs: list[float] = []
        missing = False
        for predictor in predictors:
            if predictor == "recommended_action_conflicts_with_reported_state":
                value_bool = as_bool(row.get(predictor))
                if value_bool is None:
                    missing = True
                    break
                xs.append(1.0 if value_bool else 0.0)
            else:
                value = as_float(row.get(predictor))
                if value is None:
                    missing = True
                    break
                xs.append(value)
        if missing:
            continue
        x_values.append(xs)
        y_values.append(1.0 if y_bool else 0.0)
    n = len(y_values)
    events = int(sum(y_values))
    non_events = n - events
    status = {
        "model_name": model_name,
        "outcome": outcome,
        "n": n,
        "events": events,
        "non_events": non_events,
        "converged": False,
        "warning": "",
    }
    if 0 < min(events, non_events) < 10:
        status["warning"] = (
            f"rare event model: {min(events, non_events)} rows in the smaller class"
        )
    if n == 0 or events == 0 or non_events == 0:
        status["warning"] = "not fitted: outcome has no variation"
        return [], status
    x = np.column_stack([np.ones(n), np.asarray(x_values, dtype=float)])
    y = np.asarray(y_values, dtype=float)

    def objective(beta: np.ndarray) -> float:
        p = sigmoid(x @ beta)
        eps = 1e-9
        return float(-np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    def gradient(beta: np.ndarray) -> np.ndarray:
        return x.T @ (sigmoid(x @ beta) - y)

    result = minimize(
        objective,
        np.zeros(x.shape[1]),
        jac=gradient,
        method="BFGS",
        options={"maxiter": 1000, "gtol": 1e-6},
    )
    beta = np.asarray(result.x, dtype=float)
    probabilities = sigmoid(x @ beta)
    weights = probabilities * (1.0 - probabilities)
    hessian = x.T @ (x * weights[:, None])
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(hessian)
        status["warning"] = (
            status["warning"] + "; " if status["warning"] else ""
        ) + "used pseudo-inverse covariance"
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    if not result.success:
        status["warning"] = (
            status["warning"] + "; " if status["warning"] else ""
        ) + str(result.message)
    if min(probabilities) < 1e-5 or max(probabilities) > 1 - 1e-5:
        status["warning"] = (
            status["warning"] + "; " if status["warning"] else ""
        ) + "near-perfect fitted probabilities"
    status["converged"] = bool(result.success)
    out: list[dict[str, Any]] = []
    for term, coef, se_value in zip(["Intercept", *predictors], beta, se):
        z_value = coef / se_value if se_value > 0 else float("nan")
        p_value = 2 * norm.sf(abs(z_value)) if math.isfinite(z_value) else float("nan")
        ci_low = coef - 1.96 * se_value
        ci_high = coef + 1.96 * se_value
        out.append(
            {
                "model_name": model_name,
                "outcome": outcome,
                "term": term,
                "coefficient": coef,
                "std_error": se_value,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "p_value": p_value,
                "events": events,
                "n": n,
                "warning": status["warning"],
            }
        )
    return out, status


def build_regression_table(*, sentence_exp2: Path, output_dir: Path) -> list[dict[str, Any]]:
    positions = read_csv(sentence_exp2 / "state_uncertainty_position_rows.csv")
    factors = read_csv(sentence_exp2 / "factorization_position_rows.csv")
    position_by_key = {
        (row["example_id"], int(row["reasoning_step_idx"])): row for row in positions
    }
    confidences = [action_confidence(row) for row in positions]
    confidences = [value for value in confidences if value is not None]
    mean_confidence = float(np.mean(confidences)) if confidences else 0.0
    rows: list[dict[str, Any]] = []
    for factor in factors:
        key = (factor["example_id"], int(factor["reasoning_step_idx"]))
        position = position_by_key.get(key, {})
        confidence = action_confidence(position) if position else None
        rows.append(
            {
                **factor,
                "reasoning_progress": factor.get("reasoning_progress", ""),
                "action_confidence_per_0p00001": (
                    (confidence - mean_confidence) / 0.00001
                    if confidence is not None
                    else ""
                ),
                "state_belief_uncertainty_0p01_bits": (
                    (as_float(position.get("state_belief_entropy_mean_bits")) or 0.0) / 0.01
                    if as_float(position.get("state_belief_entropy_mean_bits")) is not None
                    else ""
                ),
                "belief_error_rate": position.get("primary_belief_error_rate", ""),
                "recommended_action_conflicts_with_reported_state": factor.get(
                    "chosen_action_conflicts_with_reported_state", ""
                ),
            }
        )
    predictors = [
        "reasoning_progress",
        "action_confidence_per_0p00001",
        "state_belief_uncertainty_0p01_bits",
        "belief_error_rate",
        "recommended_action_conflicts_with_reported_state",
    ]
    specs = [
        ("action_change_next", "Action changes at next prefix"),
        ("optimality_loss_next", "Optimal action becomes suboptimal"),
        ("optimality_recovery_next", "Suboptimal action becomes optimal"),
    ]
    predictor_labels = {
        "reasoning_progress": "reasoning progress",
        "action_confidence_per_0p00001": "action confidence (+0.00001 probability)",
        "state_belief_uncertainty_0p01_bits": "state-belief uncertainty (0.01 bits)",
        "belief_error_rate": "belief error rate",
        "recommended_action_conflicts_with_reported_state": (
            "recommended action conflicts with reported state"
        ),
    }
    table_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for outcome, outcome_label in specs:
        coefficients, status = fit_logistic(
            rows,
            outcome=outcome,
            predictors=predictors,
            model_name=outcome,
        )
        status_rows.append(status)
        for row in coefficients:
            if row["term"] == "Intercept":
                continue
            table_rows.append(
                {
                    "outcome": outcome_label,
                    "predictor": predictor_labels[row["term"]],
                    "coefficient": row["coefficient"],
                    "SE": row["std_error"],
                    "95% CI": f"[{fmt(row['ci_95_low'])}, {fmt(row['ci_95_high'])}]",
                    "p": row["p_value"],
                    "events": row["events"],
                    "warning": row["warning"],
                }
            )
    write_csv(output_dir / "table2_exploratory_regression_results.csv", table_rows)
    write_csv(output_dir / "table2_exploratory_regression_status.csv", status_rows)
    lines = [
        "# Table 2. Exploratory Regression Results",
        "",
        "Exploratory logistic regressions predicting whether the next reasoning prefix changes the recommended action or its optimality. Rows are prefixes from eight states, so p-values are diagnostics rather than population-level inference. Action confidence is scaled per +0.00001 probability because the recommended-action probability is almost always close to one.",
        "",
        "| outcome | predictor | coefficient | SE | 95% CI | p | events | warning |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in table_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["outcome"]),
                    str(row["predictor"]),
                    fmt(row["coefficient"]),
                    fmt(row["SE"]),
                    str(row["95% CI"]),
                    fmt(row["p"], 4),
                    str(row["events"]),
                    str(row["warning"]),
                ]
            )
            + " |"
        )
    (output_dir / "table2_exploratory_regression_results.md").write_text(
        "\n".join(lines) + "\n"
    )
    return table_rows


def write_captions_and_index(output_dir: Path) -> None:
    captions = [
        "# Figure Captions",
        "",
        "## `figure1_action_confidence_over_reasoning_progress.png`",
        "At each reasoning prefix, the model assigns probabilities to UP, DOWN, LEFT, and RIGHT from a temperature 0.7 logprob query. The recommended action is the highest-probability action. The line shows the median across eight states, and the band shows the interquartile range. The plot shows that action selection is usually highly confident, explaining why action entropy is close to zero.",
        "",
        "## `figure2_belief_errors_around_action_optimality_changes.png`",
        "Belief error rate before and after changes in whether the recommended action is optimal. Offset 0 is the first prefix where action optimality changes. Belief error is averaged over wall-left, wall-right, wall-up, wall-down, key-held, and door-open probes.",
        "",
        "## `figure3_state_belief_uncertainty_over_reasoning_progress.png`",
        "State-belief uncertainty is Shannon entropy over yes, no, and unknown probabilities from temperature 0.7 logprob queries. The curve shows the mean across eight state-level bin averages, and the band shows the interquartile range. Each prefix averages wall, key, and door beliefs.",
        "",
    ]
    (output_dir / "FIGURE_CAPTIONS.md").write_text("\n".join(captions))
    index = [
        "# Reader-Facing Figures and Tables",
        "",
        "Use these outputs for the pilot write-up. They are selected because each answers a specific question with field-standard terms.",
        "",
        "## Recommended Tables",
        "",
        "- `table1_run_coverage_event_counts.md`: what was measured and how many events were observed.",
        "- `table2_exploratory_regression_results.md`: exploratory regression coefficients for action-change outcomes.",
        "",
        "## Recommended Figures",
        "",
        "- `figs/figure1_action_confidence_over_reasoning_progress.png`: whether recommended actions are high-confidence across reasoning.",
        "- `figs/figure2_belief_errors_around_action_optimality_changes.png`: whether belief errors change near optimality changes.",
        "- `figs/figure3_state_belief_uncertainty_over_reasoning_progress.png`: how state-belief uncertainty changes as more reasoning is revealed.",
        "",
        "## Diagnostics Not Recommended for Main Text",
        "",
        "- `action_entropy_events.png`: action entropy is near zero throughout, so this is a diagnostic rather than a main result.",
        "- the previous activation-metric diagnostic figure: the current activation metrics are hard to interpret and should remain exploratory.",
        "- `sentence_vs_packed_boundaries.png`: useful as a methods appendix figure, but Table 1 is clearer for the main result.",
        "- action-transition heatmaps: useful diagnostics, but too dependent on this specific grid path for a reader-facing result.",
        "",
    ]
    (output_dir / "READER_FACING_FIGURES.md").write_text("\n".join(index))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence-exp1", type=Path, default=DEFAULT_SENTENCE_EXP1)
    parser.add_argument("--packed-exp1", type=Path, default=DEFAULT_PACKED_EXP1)
    parser.add_argument("--sentence-exp2", type=Path, default=DEFAULT_SENTENCE_EXP2)
    parser.add_argument("--packed-exp2", type=Path, default=DEFAULT_PACKED_EXP2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_coverage_table(
        sentence_exp1=args.sentence_exp1,
        packed_exp1=args.packed_exp1,
        sentence_exp2=args.sentence_exp2,
        packed_exp2=args.packed_exp2,
        output_dir=args.output_dir,
    )
    plot_action_confidence(
        sentence_exp1=args.sentence_exp1,
        packed_exp1=args.packed_exp1,
        output_dir=args.output_dir,
    )
    plot_belief_errors_around_events(sentence_exp2=args.sentence_exp2, output_dir=args.output_dir)
    plot_state_belief_uncertainty(sentence_exp2=args.sentence_exp2, output_dir=args.output_dir)
    build_regression_table(sentence_exp2=args.sentence_exp2, output_dir=args.output_dir)
    write_captions_and_index(args.output_dir)
    print(json.dumps({"status": "completed", "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
