#!/usr/bin/env python3
"""Analyse whether action probability moves toward or away from optimal actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from reveng.experiments.action_probability_flow import (
    add_probability_flow_columns,
    build_window_changes,
    clustered_paired_event_summary,
    match_change_points_to_controls,
    matched_role_event_summary,
    paired_bootstrap_summary,
    progress_bin_summary,
    summarize_states,
    threshold_sensitivity,
)


CONTROL_COLOR = "#8FC9E8"
FAILURE_COLOR = "#1764A0"
EVENT_COLOR = "#1764A0"
MATCHED_COLOR = "#75889A"
TEXT_COLOR = "#203040"
GRID_COLOR = "#D9E6EF"


METRICS = (
    "initial_optimal_action_probability",
    "final_optimal_action_probability",
    "mean_optimal_action_probability",
    "last_quarter_optimal_action_probability",
    "net_change_in_optimal_action_probability",
    "mean_probability_moved_toward_optimal",
    "mean_probability_moved_away_from_optimal",
    "mean_absolute_change_in_optimal_action_probability",
    "fraction_transitions_toward_optimal",
    "fraction_transitions_away_from_optimal",
    "directional_efficiency",
)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def plot_probability_flow(
    progress: pd.DataFrame,
    sensitivity: pd.DataFrame,
    output_path: Path,
) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    role_style = {
        "control": (CONTROL_COLOR, "Control states"),
        "failure": (FAILURE_COLOR, "Failure states"),
    }

    ax = axes[0]
    for role in ("control", "failure"):
        group = progress[progress["matched_role"] == role].sort_values(
            "progress_midpoint"
        )
        color, label = role_style[role]
        x = 100 * group["progress_midpoint"].to_numpy(float)
        y = group["mean_optimal_action_probability"].to_numpy(float)
        low = group["ci_low"].to_numpy(float)
        high = group["ci_high"].to_numpy(float)
        ax.plot(x, y, marker="o", linewidth=2, markersize=4, color=color, label=label)
        ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
    ax.set_title("A. Probability assigned to optimal actions")
    ax.set_xlabel("Reasoning progress (% of sentences revealed)")
    ax.set_ylabel("Probability assigned to optimal actions")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    away = sensitivity[sensitivity["metric"] == "fraction_away"].sort_values(
        "minimum_change"
    )
    for role in ("control", "failure"):
        color, label = role_style[role]
        x = 100 * away["minimum_change"].to_numpy(float)
        y = away[f"{role}_mean"].to_numpy(float)
        low = away[f"{role}_ci_low"].to_numpy(float)
        high = away[f"{role}_ci_high"].to_numpy(float)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([y - low, high - y]),
            marker="o",
            linewidth=2,
            markersize=5,
            capsize=3,
            color=color,
            label=label,
        )
    ax.set_title("B. Sentences that reduce optimal-action probability")
    ax.set_xlabel("Minimum reduction in one sentence (percentage points)")
    ax.set_ylabel("Fraction of sentence transitions")
    ax.set_xticks(100 * away["minimum_change"].to_numpy(float))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")

    fig.suptitle(
        "Where action probability moves during reasoning",
        fontsize=14,
        fontweight="semibold",
        y=1.01,
    )
    fig.text(
        0.5,
        -0.02,
        "Lines and points are means across 23 matched failure–control pairs; bands and bars are 95% matched-pair bootstrap intervals.",
        ha="center",
        va="top",
        fontsize=9,
        color="#5D6D7E",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_change_point_flow(
    summary: pd.DataFrame,
    role_comparison: pd.DataFrame,
    output_path: Path,
    window: int,
) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    labels = ["BEAST\nchange points", "Progress-matched\npositions"]
    colors = [EVENT_COLOR, MATCHED_COLOR]
    primary = summary[
        (summary["point_set"] == "all detected points")
        & (summary["state_group"] == "all states")
    ].set_index("metric")

    ax = axes[0]
    row = primary.loc["absolute_window_change"]
    values = np.asarray([row["change_point_mean"], row["matched_position_mean"]])
    low = np.asarray([row["change_point_ci_low"], row["matched_position_ci_low"]])
    high = np.asarray([row["change_point_ci_high"], row["matched_position_ci_high"]])
    for index in range(2):
        ax.errorbar(
            index,
            100 * values[index],
            yerr=np.asarray(
                [
                    [100 * (values[index] - low[index])],
                    [100 * (high[index] - values[index])],
                ]
            ),
            fmt="o",
            markersize=8,
            capsize=4,
            linewidth=2,
            color=colors[index],
        )
    ax.set_xticks(np.arange(2), labels)
    ax.set_title("A. Magnitude of probability movement")
    ax.set_ylabel("Absolute change (percentage points)")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.text(
        0.5,
        0.04,
        f"Difference: {100 * row['change_point_minus_matched']:+.1f} pp\n"
        f"95% interval [{100 * row['difference_ci_low']:+.1f}, {100 * row['difference_ci_high']:+.1f}]",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": GRID_COLOR,
        },
    )

    ax = axes[1]
    signed = summary[
        (summary["point_set"] == "all detected points")
        & (summary["metric"] == "signed_window_change")
        & (summary["state_group"].isin(["control states", "failure states"]))
    ].set_index("state_group")
    x = np.arange(2)
    for offset, row_type in ((-0.11, "change point"), (0.11, "matched position")):
        values = []
        lows = []
        highs = []
        for state_group in ("control states", "failure states"):
            row = signed.loc[state_group]
            prefix = (
                "change_point" if row_type == "change point" else "matched_position"
            )
            values.append(float(row[f"{prefix}_mean"]))
            lows.append(float(row[f"{prefix}_ci_low"]))
            highs.append(float(row[f"{prefix}_ci_high"]))
        values_array = np.asarray(values)
        low_array = np.asarray(lows)
        high_array = np.asarray(highs)
        color = EVENT_COLOR if row_type == "change point" else MATCHED_COLOR
        label = (
            "BEAST change points"
            if row_type == "change point"
            else "Progress-matched positions"
        )
        ax.errorbar(
            x + offset,
            100 * values_array,
            yerr=np.vstack(
                [100 * (values_array - low_array), 100 * (high_array - values_array)]
            ),
            fmt="o",
            markersize=7,
            capsize=4,
            linewidth=2,
            color=color,
            label=label,
        )
    ax.axhline(0, color=TEXT_COLOR, linewidth=1)
    ax.set_xticks(x, ["Control states", "Failure states"])
    ax.set_title("B. Direction of probability movement")
    ax.set_ylabel("Signed change (percentage points)")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.02,
        0.03,
        "Positive: toward optimal actions\nNegative: away from optimal actions",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#5D6D7E",
    )
    group_row = role_comparison[
        (role_comparison["point_set"] == "all detected points")
        & (role_comparison["row_type"] == "change point")
        & (role_comparison["metric"] == "signed_window_change")
    ].iloc[0]
    ax.text(
        0.48,
        0.82,
        f"Control − failure: {100 * group_row['control_minus_failure']:+.1f} pp\n"
        f"95% interval [{100 * group_row['difference_ci_low']:+.1f}, {100 * group_row['difference_ci_high']:+.1f}]",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#5D6D7E",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": GRID_COLOR,
        },
    )

    fig.suptitle(
        "Probability movement around retrospective change points",
        fontsize=14,
        fontweight="semibold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        f"Change = mean optimal-action probability in {window} sentences after minus {window} before.\n"
        "Error bars are 95% trajectory-bootstrap intervals; the state-group contrast resamples matched pairs.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#5D6D7E",
    )
    fig.tight_layout(rect=(0.02, 0.09, 0.98, 0.90))
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def format_interval(
    value: float, low: float, high: float, *, percentage: bool = True
) -> str:
    scale = 100 if percentage else 1
    suffix = " percentage points" if percentage else ""
    return f"{scale * value:+.1f}{suffix} (95% interval {scale * low:+.1f} to {scale * high:+.1f})"


def write_report(
    output_dir: Path,
    state_summary: pd.DataFrame,
    matched_summary: pd.DataFrame,
    cp_summary: pd.DataFrame,
    role_comparison: pd.DataFrame,
    cp_pairs: pd.DataFrame,
    window: int,
) -> None:
    metrics = matched_summary.set_index("metric")
    mean_probability = metrics.loc["mean_optimal_action_probability"]
    initial = metrics.loc["initial_optimal_action_probability"]
    final = metrics.loc["final_optimal_action_probability"]
    away = metrics.loc["fraction_transitions_away_from_optimal"]
    cp_metrics = cp_summary[
        (cp_summary["point_set"] == "all detected points")
        & (cp_summary["state_group"] == "all states")
    ].set_index("metric")
    cp_abs = cp_metrics.loc["absolute_window_change"]
    cp_signed = cp_metrics.loc["signed_window_change"]
    cp_signed_groups = cp_summary[
        (cp_summary["point_set"] == "all detected points")
        & (cp_summary["metric"] == "signed_window_change")
    ].set_index("state_group")
    net = metrics.loc["net_change_in_optimal_action_probability"]
    group_all = role_comparison[
        (role_comparison["point_set"] == "all detected points")
        & (role_comparison["row_type"] == "change point")
        & (role_comparison["metric"] == "signed_window_change")
    ].iloc[0]
    group_stable = role_comparison[
        (role_comparison["point_set"] == "repeatedly detected points")
        & (role_comparison["row_type"] == "change point")
        & (role_comparison["metric"] == "signed_window_change")
    ].iloc[0]

    lines = [
        "# Where Action Probability Moves During Reasoning",
        "",
        "## Question",
        "",
        "Do sentence-to-sentence changes move probability toward or away from optimal actions, and do failure states differ from matched control states? We also ask whether retrospective BEAST change points identify more directional movement than ordinary positions at the same stage of reasoning.",
        "",
        "## Method",
        "",
        f"The analysis uses {len(state_summary)} states in {state_summary['matched_pair_id'].nunique()} matched failure–control pairs. At each sentence, **optimal-action probability** is the sum of the readout probabilities assigned to all actions that are optimal in the current environment state. The primary transition measure is its signed sentence-to-sentence change: positive values move probability toward optimal actions and negative values move it away. No threshold is used for the primary continuous comparison.",
        "",
        f"For the change-point comparison, each valid BEAST point is paired with a non-change-point position from the same state at similar reasoning progress. Windowed change is the mean optimal-action probability in the {window} sentences after the position minus the mean in the {window} sentences before it. BEAST remains retrospective.",
        "",
        "## Results",
        "",
        f"Failure and control states began with similar optimal-action probability: {initial['failure_mean']:.3f} versus {initial['control_mean']:.3f}; the paired difference was {format_interval(initial['failure_minus_control'], initial['difference_ci_low'], initial['difference_ci_high'])}. Across the full reasoning trace, failure states assigned less probability to optimal actions: {mean_probability['failure_mean']:.3f} versus {mean_probability['control_mean']:.3f}; difference {format_interval(mean_probability['failure_minus_control'], mean_probability['difference_ci_low'], mean_probability['difference_ci_high'])}. At the final readout, the corresponding probabilities were {final['failure_mean']:.3f} and {final['control_mean']:.3f}; difference {format_interval(final['failure_minus_control'], final['difference_ci_low'], final['difference_ci_high'])}.",
        "",
        f"From the initial to the final readout, optimal-action probability increased by {100 * net['failure_mean']:.1f} percentage points in failure states and {100 * net['control_mean']:.1f} in controls; paired difference {format_interval(net['failure_minus_control'], net['difference_ci_low'], net['difference_ci_high'])}. A sentence reduced optimal-action probability in {100 * away['failure_mean']:.1f}% of failure-state transitions and {100 * away['control_mean']:.1f}% of control-state transitions; difference {format_interval(away['failure_minus_control'], away['difference_ci_low'], away['difference_ci_high'])}. Thus, the groups differ clearly in net accumulation, not in the frequency of individual decreases. Threshold sensitivity gives the same qualitative result.",
        "",
        f"The change-point comparison retained {int(cp_abs['n_change_points'])} BEAST points with complete windows, drawn from {int(cp_abs['n_trajectories'])} trajectories. Their absolute windowed change was {100 * cp_abs['change_point_mean']:.1f} percentage points, compared with {100 * cp_abs['matched_position_mean']:.1f} at progress-matched positions; difference {format_interval(cp_abs['change_point_minus_matched'], cp_abs['difference_ci_low'], cp_abs['difference_ci_high'])}. Across all states, signed change was {100 * cp_signed['change_point_mean']:+.1f} percentage points at BEAST points and {100 * cp_signed['matched_position_mean']:+.1f} at matched positions; difference {format_interval(cp_signed['change_point_minus_matched'], cp_signed['difference_ci_low'], cp_signed['difference_ci_high'])}.",
        "",
        f"At BEAST points, optimal-action probability changed by {100 * cp_signed_groups.loc['control states', 'change_point_mean']:+.1f} percentage points in controls and {100 * cp_signed_groups.loc['failure states', 'change_point_mean']:+.1f} in failure states. Across {int(group_all['n_complete_matched_pairs'])} complete state pairs, the control-minus-failure contrast was {format_interval(group_all['control_minus_failure'], group_all['difference_ci_low'], group_all['difference_ci_high'])}; its interval includes zero. For the points recovered in at least 12 of 16 robustness reruns, the contrast was {format_interval(group_stable['control_minus_failure'], group_stable['difference_ci_low'], group_stable['difference_ci_high'])}. This subset result is exploratory.",
        "",
        "## Interpretation",
        "",
        "Failure and control states start similarly, but controls accumulate substantially more probability on optimal actions. Failure states do not show a clearly higher frequency of sentence-level decreases, so the result should not be described simply as failures taking more wrong-way steps. The analysis does not establish that a particular sentence causes a later action.",
        "",
        "BEAST points identify substantially larger probability movements than matched positions. Their average direction is toward optimal actions in control states but close to zero in failure states. The direct state-group contrast is uncertain for all detected points and positive for the repeatedly detected subset. The supported conclusion is therefore that CPD identifies large revisions; differential consolidation on optimal actions is promising but requires replication. BEAST is retrospective, and these associations are not causal.",
        "",
        "## Figures",
        "",
        "- `figs/action_probability_flow_by_progress.png`",
        "- `figs/change_point_probability_flow.png`",
        "",
        "## Output tables",
        "",
        "- `probability_flow_rows.csv`: one row per reasoning position.",
        "- `state_probability_flow_summary.csv`: one row per state.",
        "- `matched_failure_control_summary.csv`: paired state comparisons.",
        "- `threshold_sensitivity.csv`: minimum-change sensitivity analysis.",
        "- `progress_summary.csv`: optimal-action probability over normalized reasoning progress.",
        "- `change_point_control_pairs.csv`: BEAST points and their progress-matched positions.",
        "- `change_point_flow_summary.csv`: change-point comparisons with trajectory-bootstrap intervals.",
        "- `change_point_state_group_comparison.csv`: matched failure–control contrasts at change points.",
    ]
    (output_dir / "run_report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positions",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/action_distribution_change_points_v1/position_distribution_metrics.csv"
        ),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "data/behavioral_probes/reasoning_belief_action_cohorts/matched_46_candidates.csv"
        ),
    )
    parser.add_argument(
        "--change-points",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/action_distribution_cpd_beast_v1/detected_change_points.csv"
        ),
    )
    parser.add_argument(
        "--change-point-stability",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/action_distribution_cpd_robustness_v1/original_point_stability.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hypothesis_tests/action_probability_flow_v1"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_repeats < 100:
        raise ValueError("bootstrap-repeats must be at least 100")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figs"
    figure_dir.mkdir(exist_ok=True)

    positions = pd.read_csv(args.positions)
    metadata = pd.read_csv(args.metadata)
    change_points = pd.read_csv(args.change_points)
    stability = pd.read_csv(args.change_point_stability)
    flow = add_probability_flow_columns(positions, metadata)
    states = summarize_states(flow)
    matched = paired_bootstrap_summary(
        states, METRICS, repeats=args.bootstrap_repeats, seed=args.seed
    )
    sensitivity = threshold_sensitivity(
        flow, repeats=args.bootstrap_repeats, seed=args.seed
    )
    progress = progress_bin_summary(
        flow, repeats=args.bootstrap_repeats, seed=args.seed
    )
    windows = build_window_changes(flow, window=args.window)
    pairs = match_change_points_to_controls(
        windows, change_points, exclusion_radius=args.window
    )
    stable_lookup = stability.set_index(["example_id", "position_index"])[
        "found_by_at_least_12_of_16_setups"
    ].to_dict()
    pairs["repeatedly_detected"] = pairs.apply(
        lambda row: bool(
            stable_lookup.get(
                (row["example_id"], int(row["change_point_position"])), False
            )
        ),
        axis=1,
    )
    cp_summaries: list[pd.DataFrame] = []
    role_summaries: list[pd.DataFrame] = []
    for point_set, point_rows in (
        ("all detected points", pairs),
        ("repeatedly detected points", pairs[pairs["repeatedly_detected"]]),
    ):
        for state_group, group_rows in (
            ("all states", point_rows),
            ("control states", point_rows[point_rows["matched_role"] == "control"]),
            ("failure states", point_rows[point_rows["matched_role"] == "failure"]),
        ):
            summary = clustered_paired_event_summary(
                group_rows, repeats=args.bootstrap_repeats, seed=args.seed
            )
            summary.insert(0, "state_group", state_group)
            summary.insert(0, "point_set", point_set)
            cp_summaries.append(summary)
        role_summary = matched_role_event_summary(
            point_rows, repeats=args.bootstrap_repeats, seed=args.seed
        )
        role_summary.insert(0, "point_set", point_set)
        role_summaries.append(role_summary)
    cp_summary = pd.concat(cp_summaries, ignore_index=True)
    role_comparison = pd.concat(role_summaries, ignore_index=True)

    flow.to_csv(args.output_dir / "probability_flow_rows.csv", index=False)
    states.to_csv(args.output_dir / "state_probability_flow_summary.csv", index=False)
    matched.to_csv(args.output_dir / "matched_failure_control_summary.csv", index=False)
    sensitivity.to_csv(args.output_dir / "threshold_sensitivity.csv", index=False)
    progress.to_csv(args.output_dir / "progress_summary.csv", index=False)
    pairs.to_csv(args.output_dir / "change_point_control_pairs.csv", index=False)
    cp_summary.to_csv(args.output_dir / "change_point_flow_summary.csv", index=False)
    role_comparison.to_csv(
        args.output_dir / "change_point_state_group_comparison.csv", index=False
    )

    plot_probability_flow(
        progress, sensitivity, figure_dir / "action_probability_flow_by_progress.png"
    )
    plot_change_point_flow(
        cp_summary,
        role_comparison,
        figure_dir / "change_point_probability_flow.png",
        args.window,
    )
    write_report(
        args.output_dir,
        states,
        matched,
        cp_summary,
        role_comparison,
        pairs,
        args.window,
    )
    manifest = {
        "positions": str(args.positions),
        "metadata": str(args.metadata),
        "change_points": str(args.change_points),
        "change_point_stability": str(args.change_point_stability),
        "output_dir": str(args.output_dir),
        "bootstrap_repeats": args.bootstrap_repeats,
        "window": args.window,
        "seed": args.seed,
        "n_positions": len(flow),
        "n_states": flow["example_id"].nunique(),
        "n_trajectories": flow["trajectory_id"].nunique(),
        "n_matched_pairs": flow["matched_pair_id"].nunique(),
        "n_change_point_pairs": pairs["event_id"].nunique(),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
