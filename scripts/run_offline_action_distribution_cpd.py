#!/usr/bin/env python3
"""Run offline BEAST change-point detection on action-probability time series."""

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

from reveng.experiments.beast_action_change_points import (
    IMPLEMENTATION_VERSION,
    BeastConfig,
    fit_action_distribution_states,
)


ACTION_COLORS = {
    "UP": "#0B3C5D",
    "DOWN": "#1769AA",
    "LEFT": "#5FA8D3",
    "RIGHT": "#A7D5ED",
}
FAILURE = "#1769AA"
CONTROL = "#8CC8E8"
DARK = "#17324D"
GRID = "#DCEAF3"
EVENT_LABELS = {
    "action_change": "Recommendation change",
    "optimal_to_suboptimal": "Optimal to suboptimal",
    "suboptimal_to_optimal": "Suboptimal to optimal",
    "commitment_onset": "Stable final recommendation begins",
}


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trajectory_bootstrap_fraction(
    rows: pd.DataFrame,
    value_column: str,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    grouped = {
        str(key): group[value_column].astype(float).to_numpy()
        for key, group in rows.groupby("trajectory_id")
    }
    keys = np.asarray(sorted(grouped), dtype=object)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repeats):
        sampled = rng.choice(keys, len(keys), replace=True)
        combined = np.concatenate([grouped[key] for key in sampled])
        values.append(float(np.mean(combined)))
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def build_event_positions(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    frame["position_index"] = frame["event_reasoning_step_idx"].astype(int)
    loss = frame["event_type"].isin(
        ["sustained_optimal_to_suboptimal", "transient_optimal_to_suboptimal"]
    )
    frame.loc[loss, "event_type"] = "optimal_to_suboptimal"
    return frame[frame["event_type"].isin(EVENT_LABELS)].copy()


def event_proximity_analysis(
    positions: pd.DataFrame,
    detected: pd.DataFrame,
    events: pd.DataFrame,
    *,
    window: int,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_lookup = {
        (str(example_id), event_type): group["position_index"].astype(int).to_numpy()
        for (example_id, event_type), group in events.groupby(
            ["example_id", "event_type"]
        )
    }
    observed_rows: list[dict[str, Any]] = []
    for point in detected.itertuples():
        for event_type in EVENT_LABELS:
            event_positions = event_lookup.get((str(point.example_id), event_type), [])
            nearest = (
                min(abs(int(point.position_index) - int(value)) for value in event_positions)
                if len(event_positions)
                else np.nan
            )
            observed_rows.append(
                {
                    "example_id": point.example_id,
                    "trajectory_id": point.trajectory_id,
                    "matched_role": point.matched_role,
                    "change_point_position": int(point.position_index),
                    "event_type": event_type,
                    "nearest_event_distance_sentences": nearest,
                    "event_within_window": bool(
                        np.isfinite(nearest) and nearest <= window
                    ),
                }
            )
    observed = pd.DataFrame(observed_rows)

    position_lookup = {
        str(example_id): group.sort_values("position_index")
        for example_id, group in positions.groupby("example_id")
    }
    rng = np.random.default_rng(seed)
    null_rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        counts = {event_type: [0, 0] for event_type in EVENT_LABELS}
        for point in detected.itertuples():
            state = position_lookup[str(point.example_id)]
            progress = float(point.reasoning_progress)
            decile = min(9, int(progress * 10))
            lower, upper = decile / 10, (decile + 1) / 10
            candidates = state[
                state["reasoning_progress"].astype(float).between(
                    lower, upper, inclusive="left"
                )
            ]["position_index"].astype(int)
            candidates = candidates[candidates != int(point.position_index)]
            if candidates.empty:
                candidates = state["position_index"].astype(int)
            random_position = int(rng.choice(candidates.to_numpy()))
            for event_type in EVENT_LABELS:
                event_positions = event_lookup.get(
                    (str(point.example_id), event_type), []
                )
                close = any(
                    abs(random_position - int(value)) <= window
                    for value in event_positions
                )
                counts[event_type][0] += int(close)
                counts[event_type][1] += 1
        for event_type, (near, total) in counts.items():
            null_rows.append(
                {
                    "repeat": repeat,
                    "event_type": event_type,
                    "fraction_within_window": near / total if total else np.nan,
                }
            )
    null = pd.DataFrame(null_rows)
    return observed, null


def summarize_event_proximity(
    observed: pd.DataFrame,
    null: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for event_type in EVENT_LABELS:
        rows = observed[observed["event_type"] == event_type]
        low, high = _trajectory_bootstrap_fraction(
            rows,
            "event_within_window",
            repeats=repeats,
            seed=seed,
        )
        random_values = null[null["event_type"] == event_type][
            "fraction_within_window"
        ]
        observed_fraction = rows["event_within_window"].mean()
        output.append(
            {
                "event_type": event_type,
                "event_label": EVENT_LABELS[event_type],
                "n_detected_change_points": len(rows),
                "observed_fraction": observed_fraction,
                "observed_ci_low": low,
                "observed_ci_high": high,
                "progress_matched_random_fraction": random_values.mean(),
                "observed_minus_random_fraction": (
                    observed_fraction - random_values.mean()
                ),
                "random_ci_low": random_values.quantile(0.025),
                "random_ci_high": random_values.quantile(0.975),
                "randomization_p_value": (
                    1 + int((random_values >= observed_fraction).sum())
                )
                / (1 + len(random_values)),
            }
        )
    return pd.DataFrame(output)


def timing_summary(
    states: pd.DataFrame,
    detected: pd.DataFrame,
    *,
    bins: int,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    rows: list[dict[str, Any]] = []
    state_base = states[["example_id", "trajectory_id", "matched_role"]].copy()
    for role in ("failure", "control"):
        base = state_base[state_base["matched_role"] == role]
        points = detected[detected["matched_role"] == role].copy()
        points["bin"] = np.minimum(
            bins - 1,
            np.floor(points["reasoning_progress"].astype(float) * bins).astype(int),
        )
        for index in range(bins):
            indicator = base.copy()
            active = set(points.loc[points["bin"] == index, "example_id"])
            indicator["has_change_point"] = indicator["example_id"].isin(active)
            low, high = _trajectory_bootstrap_fraction(
                indicator,
                "has_change_point",
                repeats=repeats,
                seed=seed + index,
            )
            rows.append(
                {
                    "state_group": role,
                    "progress_bin_start": edges[index],
                    "progress_bin_end": edges[index + 1],
                    "progress_bin_midpoint": (edges[index] + edges[index + 1]) / 2,
                    "n_states": len(indicator),
                    "fraction_states_with_change_point": indicator[
                        "has_change_point"
                    ].mean(),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def choose_examples(
    states: pd.DataFrame,
    modes: pd.DataFrame,
) -> list[str]:
    detected = modes[modes["detected_change_point"]].copy()
    detected["strength"] = (
        detected["posterior_change_probability_at_position"]
        * detected["abrupt_change_in_distance"].abs()
    )
    selected: list[str] = []
    for role in ("failure", "control"):
        candidates = detected[detected["matched_role"] == role]
        scores = (
            candidates.groupby("example_id", as_index=False)["strength"]
            .max()
            .merge(states[["example_id", "n_positions"]], on="example_id")
        )
        if scores.empty:
            continue
        median_length = float(scores["n_positions"].median())
        scores["length_distance"] = abs(scores["n_positions"] - median_length)
        selected.append(
            str(
                scores.sort_values(
                    ["strength", "length_distance"], ascending=[False, True]
                ).iloc[0]["example_id"]
            )
        )
    return selected


def plot_examples(
    positions: pd.DataFrame,
    modes: pd.DataFrame,
    examples: list[str],
    path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(examples),
        2,
        figsize=(12, 3.5 * len(examples)),
        constrained_layout=True,
    )
    if len(examples) == 1:
        axes = np.asarray([axes])
    for row_index, example_id in enumerate(examples):
        state = positions[positions["example_id"] == example_id].sort_values(
            "position_index"
        )
        points = modes[
            (modes["example_id"] == example_id) & modes["detected_change_point"]
        ]
        role = str(state.iloc[0]["matched_role"]).capitalize()
        short_id = (
            str(example_id)
            .replace("together_ai_openai_gpt-oss-20b_rooms2_doorkey_", "")
            .replace("_step_", ", environment step ")
        )
        ax = axes[row_index, 0]
        for action, color in ACTION_COLORS.items():
            ax.plot(
                state["position_index"],
                state[f"prob_{action.lower()}"],
                label=action,
                color=color,
                linewidth=1.4,
            )
        for point in points.itertuples():
            ax.axvline(point.position_index, color=DARK, linewidth=0.8, alpha=0.45)
        ax.set_title(f"{role} state: {short_id}")
        ax.set_xlabel("Number of reasoning sentences revealed")
        ax.set_ylabel("Probability assigned to action")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.legend(title="Action", ncol=4, frameon=False, loc="upper right")

        ax = axes[row_index, 1]
        ax.plot(
            state["position_index"],
            state["posterior_change_probability"],
            color=FAILURE,
            linewidth=1.5,
        )
        ax.axhline(
            0.70,
            color=DARK,
            linestyle="--",
            linewidth=1,
            label="Detection threshold: 0.70",
        )
        for point in points.itertuples():
            ax.axvline(point.position_index, color=DARK, linewidth=0.8, alpha=0.45)
        ax.set_title("Posterior probability of a change point")
        ax.set_xlabel("Number of reasoning sentences revealed")
        ax.set_ylabel("Probability of a change at this sentence")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.legend(frameon=False, loc="upper right")
    figure.suptitle(
        "Action Probabilities and Detected Change Points in Example States",
        fontsize=14,
    )
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_timing(summary: pd.DataFrame, path: Path) -> None:
    figure, ax = plt.subplots(figsize=(8.3, 4.8), constrained_layout=True)
    for role, color, marker in (
        ("failure", FAILURE, "o"),
        ("control", CONTROL, "s"),
    ):
        rows = summary[summary["state_group"] == role]
        x = 100 * rows["progress_bin_midpoint"].to_numpy()
        y = rows["fraction_states_with_change_point"].to_numpy()
        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            linewidth=1.8,
            label=f"{role.capitalize()} states (n={int(rows.n_states.iloc[0])})",
        )
        ax.fill_between(
            x,
            rows["ci_low"].to_numpy(),
            rows["ci_high"].to_numpy(),
            color=color,
            alpha=0.18,
        )
    ax.set_title("When Action-Distribution Change Points Occur During Reasoning")
    ax.set_xlabel("Percentage of reasoning sentences completed")
    ax.set_ylabel("Fraction of states with a detected change point")
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.legend(frameon=False)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_counts(
    states: pd.DataFrame,
    path: Path,
    *,
    repeats: int,
    seed: int,
) -> None:
    figure, ax = plt.subplots(figsize=(8.3, 4.8), constrained_layout=True)
    rng = np.random.default_rng(seed)
    for index, (role, color) in enumerate(
        (("control", CONTROL), ("failure", FAILURE))
    ):
        subset = states[states["matched_role"] == role]
        values = subset["n_detected_change_points"].astype(float).to_numpy()
        jitter = rng.uniform(-0.10, 0.10, len(values))
        ax.scatter(
            index + jitter,
            values,
            s=34,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
            label=f"Individual {role} states (n={len(subset)})",
        )
        low, high = _trajectory_bootstrap_fraction(
            subset,
            "n_detected_change_points",
            repeats=repeats,
            seed=seed + index,
        )
        mean = float(np.mean(values))
        ax.errorbar(
            index,
            mean,
            yerr=[[max(0.0, mean - low)], [max(0.0, high - mean)]],
            fmt="D",
            color=DARK,
            markersize=7,
            capsize=5,
            linewidth=1.8,
        )
        ax.text(index + 0.13, mean, f"Mean {mean:.2f}", va="center", fontsize=9)
    ax.set_title("Detected Action-Distribution Change Points by State Group")
    ax.set_xlabel("State group")
    ax.set_ylabel("Number of detected change points per state")
    ax.set_xticks([0, 1], ["Control", "Failure"])
    ax.set_ylim(bottom=-0.4)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_event_proximity(summary: pd.DataFrame, path: Path, window: int) -> None:
    order = ["action_change", "optimal_to_suboptimal", "suboptimal_to_optimal"]
    selected = summary.set_index("event_type").loc[order].reset_index()
    labels = [
        "Recommended action changes",
        "Recommendation becomes suboptimal",
        "Recommendation becomes optimal",
    ]
    y = np.arange(len(selected))
    offset = 0.13
    figure, ax = plt.subplots(figsize=(9.2, 4.8))
    observed = selected["observed_fraction"].to_numpy()
    random = selected["progress_matched_random_fraction"].to_numpy()
    observed_err = np.vstack(
        [
            np.maximum(0.0, observed - selected["observed_ci_low"].to_numpy()),
            np.maximum(
                0.0, selected["observed_ci_high"].to_numpy() - observed
            ),
        ]
    )
    random_err = np.vstack(
        [
            np.maximum(0.0, random - selected["random_ci_low"].to_numpy()),
            np.maximum(0.0, selected["random_ci_high"].to_numpy() - random),
        ]
    )
    ax.errorbar(
        observed,
        y - offset,
        xerr=observed_err,
        fmt="o",
        color=FAILURE,
        capsize=4,
        markersize=7,
        label="Detected action-distribution change points",
    )
    ax.errorbar(
        random,
        y + offset,
        xerr=random_err,
        fmt="o",
        color=CONTROL,
        capsize=4,
        markersize=6,
        label="Progress-matched sentences",
    )
    for y_pos, value in zip(y - offset, observed, strict=True):
        ax.text(value + 0.018, y_pos, f"{value:.1%}", va="center", fontsize=9)
    for y_pos, value in zip(y + offset, random, strict=True):
        ax.text(value + 0.018, y_pos, f"{value:.1%}", va="center", fontsize=9)
    ax.set_title(
        "Action-Distribution Changes Cluster Near Recommendation Changes\n"
        f"{int(selected.n_detected_change_points.max())} detected change points"
    )
    ax.set_xlabel(
        f"Fraction within {window} sentences before or after the event"
    )
    ax.set_ylabel("Reasoning event")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=2,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_example_text(
    examples: list[str],
    modes: pd.DataFrame,
    sentences: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Example Detected Change Points",
        "",
        "Examples are selected by the largest posterior-weighted change in the "
        "distance from the initial action distribution within each state group.",
        "",
    ]
    for example_id in examples:
        lines.extend([f"## {example_id}", ""])
        points = modes[
            (modes["example_id"] == example_id) & modes["detected_change_point"]
        ].sort_values("position_index")
        trace_sentences = sentences[
            (sentences["trace_id"] == example_id)
            & (sentences["kind"] == "reasoning")
        ].set_index("sentence_id")
        for point in points.itertuples():
            sentence_id = int(point.position_index) - 1
            text = (
                str(trace_sentences.loc[sentence_id, "text"])
                if sentence_id in trace_sentences.index
                else "[No reasoning sentence: initial prompt position]"
            )
            lines.extend(
                [
                    f"- Sentence position {int(point.position_index)}; "
                    "posterior change probability "
                    f"{point.posterior_change_probability_at_position:.3f}; "
                    f"recommended action {point.action_label}; "
                    f"optimal action: {point.action_is_optimal}.",
                    f'  Text: "{text}"',
                ]
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positions",
        type=Path,
        default=Path(
            "outputs/experiment1_activation_monitor/"
            "gpt_oss_local_sentence_matched46_v1/position_rows.csv"
        ),
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=Path(
            "outputs/experiment1_activation_monitor/"
            "gpt_oss_local_sentence_matched46_v1/event_rows.csv"
        ),
    )
    parser.add_argument(
        "--sentences",
        type=Path,
        default=Path(
            "data/behavioral_probes/doorkey_chunking_validation/sentences.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/action_distribution_cpd_beast_v1"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--null-repeats", type=int, default=2000)
    parser.add_argument("--event-window", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figs"
    figure_dir.mkdir(exist_ok=True)
    setup_matplotlib()

    source = pd.read_csv(args.positions)
    events = build_event_positions(pd.read_csv(args.events))
    sentences = pd.read_csv(args.sentences)
    config = BeastConfig(seed=args.seed, samples=args.samples)
    positions, states, modes = fit_action_distribution_states(source, config)
    detected = modes[modes["detected_change_point"]].copy()

    observed, random_null = event_proximity_analysis(
        positions,
        detected,
        events,
        window=args.event_window,
        repeats=args.null_repeats,
        seed=args.seed,
    )
    event_summary = summarize_event_proximity(
        observed,
        random_null,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    timing = timing_summary(
        states,
        detected,
        bins=10,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    examples = choose_examples(states, modes)

    positions.to_csv(args.output_dir / "position_change_probabilities.csv", index=False)
    states.to_csv(args.output_dir / "state_change_point_summary.csv", index=False)
    modes.to_csv(args.output_dir / "change_point_modes.csv", index=False)
    detected.to_csv(args.output_dir / "detected_change_points.csv", index=False)
    observed.to_csv(args.output_dir / "change_point_event_proximity.csv", index=False)
    event_summary.to_csv(args.output_dir / "event_proximity_summary.csv", index=False)
    timing.to_csv(args.output_dir / "change_point_timing_summary.csv", index=False)

    plot_examples(
        positions,
        modes,
        examples,
        figure_dir / "example_action_probabilities_and_change_points.png",
    )
    plot_timing(timing, figure_dir / "change_point_timing_by_state_group.png")
    plot_counts(
        states,
        figure_dir / "change_point_count_by_state_group.png",
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    plot_event_proximity(
        event_summary,
        figure_dir / "change_points_near_decision_events.png",
        args.event_window,
    )
    write_example_text(
        examples,
        modes,
        sentences,
        args.output_dir / "EXAMPLE_CHANGE_POINTS.md",
    )

    n_states = states["example_id"].nunique()
    n_detected_states = int((states["n_detected_change_points"] > 0).sum())
    n_points = len(detected)
    failure_mean = states.loc[
        states["matched_role"] == "failure", "n_detected_change_points"
    ].mean()
    control_mean = states.loc[
        states["matched_role"] == "control", "n_detected_change_points"
    ].mean()
    event_result = event_summary.set_index("event_type")
    recommendation = event_result.loc["action_change"]
    optimality_loss = event_result.loc["optimal_to_suboptimal"]
    recovery = event_result.loc["suboptimal_to_optimal"]
    report = f"""# Offline Action-Distribution Change-Point Detection

## Question

Where does the model's elicited next-action distribution change abruptly as more
reasoning sentences are revealed, and do those changes coincide with recommendation
or action-optimality events?

## Method

For each of {n_states} fixed DoorKey environment states, the input is the sequence of
temperature 0.7 logprob distributions over `UP`, `DOWN`, `LEFT`, and `RIGHT` measured
after each reasoning sentence. Following the scalar reduction used in Forking Paths,
the analyzed series is the Euclidean distance between the action distribution at each
sentence and the distribution before reasoning.

A trend-only BEAST model fits piecewise-linear segments. A state is considered to
contain a change point when the posterior odds of one or more changes versus no change
exceed {config.trace_bayes_factor_threshold:g}:1. A location is reported when its
posterior change probability is at least {config.location_probability_threshold:.2f}.
The minimum segment length is {config.min_segment_length} sentences, the maximum
number of changes is {config.max_change_points}, and the MCMC seed is {config.seed}.
No continuation resampling and no added Gaussian noise are used.

## Results

- {n_detected_states} of {n_states} states contain at least one detected change point.
- {n_points} change points are detected in total.
- Failure states have {failure_mean:.2f} detected points on average; control states
  have {control_mean:.2f}. This pilot does not show a substantial group difference.
- {recommendation.observed_fraction:.1%} of detected points are within three sentences
  of a recommendation change, compared with {recommendation.progress_matched_random_fraction:.1%}
  for progress-matched random sentences (randomization p={recommendation.randomization_p_value:.4f}).
- Proximity to optimality loss is weaker: {optimality_loss.observed_fraction:.1%}
  observed versus {optimality_loss.progress_matched_random_fraction:.1%} at random
  (p={optimality_loss.randomization_p_value:.4f}).
- {recovery.observed_fraction:.1%} are near recovery to an optimal recommendation,
  compared with {recovery.progress_matched_random_fraction:.1%} at random
  (p={recovery.randomization_p_value:.4f}).

These are change points in immediate action readouts conditioned on observed reasoning
text. They are not causal forks in sampled continuations. The analysis can identify
candidate sentences for later semantic review or intervention, but it cannot show that
the sentence caused the later action.
"""
    (args.output_dir / "run_report.md").write_text(report)

    captions = f"""# Figure Captions

## `example_action_probabilities_and_change_points.png`

Action probabilities and Bayesian change-point probabilities for one failure state and
one control state. The left panels show temperature 0.7 candidate-token probabilities
over `UP`, `DOWN`, `LEFT`, and `RIGHT` after each reasoning sentence. Vertical lines
mark detected changes. The right panels show the BEAST posterior probability that each
sentence is a change point; the dashed line is the prespecified {config.location_probability_threshold:.2f}
location threshold. Examples are selected by the largest posterior-weighted change in
distance from the initial action distribution within each group.

## `change_point_timing_by_state_group.png`

Timing of detected action-distribution change points across reasoning. The horizontal
axis divides each state trace into ten equal bins by the fraction of reasoning sentences
completed. The vertical axis is the fraction of states with at least one detected point
in each bin. Bands are 95% trajectory-bootstrap intervals. Each group contains 23 states.

## `change_point_count_by_state_group.png`

Number of detected action-distribution change points in each fixed environment state.
Dots are states; diamonds are group means; error bars are 95% trajectory-bootstrap
intervals. A trace must have posterior odds greater than 9:1 for one or more changes,
and each reported location must have posterior probability at least 0.70. The pilot
contains 23 failure states and 23 matched control states.

## `change_points_near_decision_events.png`

Fraction of detected action-distribution change points occurring within {args.event_window}
sentences before or after each decision event. Error bars for detected points are 95%
trajectory-bootstrap intervals. The comparison samples random sentences from the same
state and same tenth of reasoning progress; its error bars show the 2.5th and 97.5th
percentiles over {args.null_repeats} randomizations. Recommendation changes include all
changes in the highest-probability action. Optimality is evaluated by the DoorKey planner.
"""
    (args.output_dir / "FIGURE_CAPTIONS.md").write_text(captions)

    reader_index = """# Reader-Facing Figures

Use these figures in this order:

1. `figs/example_action_probabilities_and_change_points.png` shows what the
   detector receives and how a detected sentence is identified.
2. `figs/change_points_near_decision_events.png` is the main result. It tests
   whether detected points occur near recommendation and optimality events more
   often than progress-matched random sentences.
3. `figs/change_point_count_by_state_group.png` shows that failure and control
   states have similar numbers of detected points.
4. `figs/change_point_timing_by_state_group.png` shows where detected points
   occur over reasoning progress; it is secondary because the confidence
   intervals are wide in this 46-state pilot.

The historical figures in
`outputs/hypothesis_tests/action_distribution_change_points_v1/` select the
largest observed change in every state and are not change-point detection
results.
"""
    (args.output_dir / "READER_FACING_FIGURES.md").write_text(reader_index)

    manifest = {
        "analysis": "offline_beast_action_distribution_change_points",
        "implementation_version": IMPLEMENTATION_VERSION,
        "input_positions": str(args.positions),
        "input_positions_sha256": sha256(args.positions),
        "input_events": str(args.events),
        "input_events_sha256": sha256(args.events),
        "n_states": n_states,
        "n_trajectories": int(states["trajectory_id"].nunique()),
        "n_positions": len(positions),
        "n_detected_states": n_detected_states,
        "n_detected_change_points": n_points,
        "config": config.__dict__,
        "event_window_sentences": args.event_window,
        "bootstrap_repeats": args.bootstrap_repeats,
        "null_repeats": args.null_repeats,
        "examples": examples,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
