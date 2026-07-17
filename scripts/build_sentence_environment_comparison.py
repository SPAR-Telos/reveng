#!/usr/bin/env python3
"""Compare sentence-prefix and environment-step Experiment 1/2 outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def binned(frame: pd.DataFrame, field: str) -> tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(0.0, 1.0, 11)
    groups = pd.cut(frame.reasoning_progress, bins=bins, include_lowest=True, labels=False)
    values = frame.assign(progress_bin=groups).groupby("progress_bin")[field].mean()
    centers = (bins[:-1] + bins[1:]) / 2
    indices = values.index.to_numpy(dtype=int)
    return centers[indices], values.to_numpy(dtype=float)


def event_count(events: pd.DataFrame, names: set[str]) -> int:
    return int(events.event_type.isin(names).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sentence-dir",
        type=Path,
        default=Path("outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1"),
    )
    parser.add_argument(
        "--environment-dir",
        type=Path,
        default=Path("outputs/experiment1_activation_monitor/gpt_oss_local_environment_step_all1276_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiment1_activation_monitor/gpt_oss_local_sentence_vs_environment_v1"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figs"
    figure_dir.mkdir(exist_ok=True)

    inputs = {
        "Sentence prefixes": args.sentence_dir,
        "Environment steps": args.environment_dir,
    }
    frames = {label: pd.read_csv(path / "position_rows.csv") for label, path in inputs.items()}
    events = {label: pd.read_csv(path / "event_rows.csv") for label, path in inputs.items()}
    rows = []
    for label in inputs:
        frame = frames[label]
        event_frame = events[label]
        rows.append(
            {
                "analysis_unit": label,
                "trajectory_states": frame.example_id.nunique(),
                "source_trajectories": frame.trajectory_id.nunique(),
                "measurement_positions": len(frame),
                "mean_action_confidence": frame.action_confidence.mean(),
                "mean_action_entropy_bits": frame.action_entropy_bits.mean(),
                "mean_state_belief_entropy_bits": frame.state_belief_entropy_bits.mean(),
                "mean_primary_belief_error_rate": frame.primary_belief_error_rate.mean(),
                "recommendation_changes": event_count(event_frame, {"action_change"}),
                "optimal_to_suboptimal": event_count(
                    event_frame,
                    {"sustained_optimal_to_suboptimal", "transient_optimal_to_suboptimal"},
                ),
                "suboptimal_to_optimal": event_count(event_frame, {"suboptimal_to_optimal"}),
                "commitment_onsets": event_count(event_frame, {"commitment_onset"}),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "sentence_environment_summary.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), sharey="row")
    colors = {"Sentence prefixes": "#0B3C5D", "Environment steps": "#328CC1"}
    for column, (label, frame) in enumerate(frames.items()):
        x, y = binned(frame, "action_confidence")
        axes[0, column].plot(x, y, marker="o", color=colors[label])
        axes[0, column].set_title(f"{label} (n={len(frame):,})")
        axes[0, column].set_xlabel(
            "Fraction of reasoning revealed"
            if label == "Sentence prefixes"
            else "Fraction of environment steps completed"
        )
        axes[0, column].grid(color="#E5EEF5", linewidth=0.8)
        x, y = binned(frame, "state_belief_entropy_bits")
        axes[1, column].plot(x, y, marker="o", color=colors[label])
        axes[1, column].set_xlabel(
            "Fraction of reasoning revealed"
            if label == "Sentence prefixes"
            else "Fraction of environment steps completed"
        )
        axes[1, column].grid(color="#E5EEF5", linewidth=0.8)
    axes[0, 0].set_ylabel("Probability of recommended action")
    axes[1, 0].set_ylabel("Mean wall, key, and door belief entropy (bits)")
    fig.suptitle("Action Confidence and State-Belief Uncertainty")
    fig.tight_layout()
    fig.savefig(figure_dir / "sentence_environment_uncertainty.png", dpi=220)
    plt.close(fig)

    report = [
        "# Sentence and Environment-Step Comparison",
        "",
        "Sentence-prefix analysis measures changes while the grid state remains fixed. Environment-step analysis "
        "measures changes after actions update the grid state. The populations differ: the sentence analysis uses "
        "46 matched states, whereas the environment analysis uses all 1,276 states from 95 trajectories. Values "
        "should therefore be compared as robustness descriptions, not as a controlled segmentation effect.",
        "",
        "| Analysis unit | States | Trajectories | Positions | Mean action confidence | Mean belief entropy | Action changes | Optimality losses | Recoveries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        report.append(
            f"| {row.analysis_unit} | {row.trajectory_states} | {row.source_trajectories} | "
            f"{row.measurement_positions:,} | {row.mean_action_confidence:.3f} | "
            f"{row.mean_state_belief_entropy_bits:.3f} | {row.recommendation_changes} | "
            f"{row.optimal_to_suboptimal} | {row.suboptimal_to_optimal} |"
        )
    report.extend(
        [
            "",
            "Commitment onset is defined only within a fixed-state reasoning trace, so it is not reported for "
            "environment-step transitions.",
        ]
    )
    (args.output_dir / "comparison_report.md").write_text("\n".join(report) + "\n")
    caption = (
        "# Figure Captions\n\n"
        "## sentence_environment_uncertainty.png\n\n"
        "Action confidence and state-belief uncertainty measured within fixed-state reasoning traces and across "
        "executed environment steps. Sentence-prefix curves use 7,084 prefixes from 46 matched states. "
        "Environment-step curves use 1,276 consecutive states from 95 trajectories. State-belief uncertainty is "
        "mean Shannon entropy over wall-left, wall-right, wall-up, wall-down, key-held, and door-open candidate "
        "probabilities at temperature 0.7. The panels use different populations and progress definitions.\n"
    )
    (args.output_dir / "FIGURE_CAPTIONS.md").write_text(caption)


if __name__ == "__main__":
    main()
