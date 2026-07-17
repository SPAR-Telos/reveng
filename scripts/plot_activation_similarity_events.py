#!/usr/bin/env python3
"""Summarize increases in activation similarity at action-event positions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


EVENT_TYPES = {
    "Recommendation change": {"action_change"},
    "Commitment onset": {"commitment_onset"},
    "Optimal to suboptimal": {
        "sustained_optimal_to_suboptimal",
        "transient_optimal_to_suboptimal",
    },
    "Suboptimal to optimal": {"suboptimal_to_optimal"},
}
TRANSITION_CLASSES = (
    "Other recommendation change",
    "Optimal to suboptimal",
    "Suboptimal to optimal",
)
BLUE = "#1769AA"
LIGHT_BLUE = "#8CC8E8"
PALE_BLUE = "#DCEFF8"
GRAY = "#6B7280"


def _bootstrap_rate(
    frame: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    grouped = {key: group for key, group in frame.groupby("trajectory_id")}
    keys = np.array(sorted(grouped), dtype=object)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        sample = pd.concat(
            [grouped[key] for key in rng.choice(keys, len(keys), replace=True)],
            ignore_index=True,
        )
        estimates.append(float(sample["similarity_increased"].mean()))
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def build_outputs(
    experiment_dir: Path,
    *,
    bootstrap_repeats: int,
) -> None:
    geometry = pd.read_csv(experiment_dir / "geometry_rows.csv")
    events = pd.read_csv(experiment_dir / "event_rows.csv")
    primary = geometry[
        (geometry["layer"] == 15)
        & (geometry["representation"] == "sentence_mean")
    ][
        [
            "example_id",
            "trajectory_id",
            "reasoning_step_idx",
            "reasoning_progress",
            "previous_mean_cosine",
        ]
    ].copy()
    primary = primary.dropna(subset=["previous_mean_cosine"]).sort_values(
        ["example_id", "reasoning_step_idx"]
    )
    primary["similarity_change"] = primary.groupby("example_id")[
        "previous_mean_cosine"
    ].diff()
    primary = primary.dropna(subset=["similarity_change"]).copy()
    primary["similarity_increased"] = primary["similarity_change"] > 0

    event_sets = {
        label: set(
            zip(
                events.loc[events["event_type"].isin(types), "example_id"],
                events.loc[
                    events["event_type"].isin(types), "reasoning_step_idx"
                ].astype(int),
            )
        )
        for label, types in EVENT_TYPES.items()
    }
    keys = list(
        zip(primary["example_id"], primary["reasoning_step_idx"].astype(int))
    )
    for label, positions in event_sets.items():
        primary[label] = [key in positions for key in keys]
    tracked_positions = set().union(*event_sets.values())
    primary["No tracked action event"] = [
        key not in tracked_positions for key in keys
    ]

    rate_rows = []
    rate_order = ["No tracked action event", *EVENT_TYPES]
    for index, label in enumerate(rate_order):
        subset = primary[primary[label]].copy()
        low, high = _bootstrap_rate(
            subset,
            repeats=bootstrap_repeats,
            seed=42 + index,
        )
        rate_rows.append(
            {
                "event": label,
                "n_eligible_positions": len(subset),
                "n_similarity_increased": int(subset["similarity_increased"].sum()),
                "rate_similarity_increased": float(
                    subset["similarity_increased"].mean()
                ),
                "ci_low": low,
                "ci_high": high,
            }
        )
    rates = pd.DataFrame(rate_rows)

    increased_changes = primary[
        primary["similarity_increased"] & primary["Recommendation change"]
    ].copy()
    composition = []
    for row in increased_changes.itertuples():
        key = (row.example_id, int(row.reasoning_step_idx))
        if key in event_sets["Optimal to suboptimal"]:
            event_class = "Optimal to suboptimal"
        elif key in event_sets["Suboptimal to optimal"]:
            event_class = "Suboptimal to optimal"
        else:
            event_class = "Other recommendation change"
        composition.append(event_class)
    increased_changes["recommendation_change_class"] = composition
    composition_counts = (
        increased_changes["recommendation_change_class"]
        .value_counts()
        .reindex(TRANSITION_CLASSES, fill_value=0)
    )
    composition_rows = pd.DataFrame(
        {
            "recommendation_change_class": composition_counts.index,
            "n_positions": composition_counts.values,
            "proportion": composition_counts.values / composition_counts.sum(),
        }
    )
    commitment_overlap = int(
        increased_changes["Commitment onset"].sum()
    )
    composition_rows["commitment_overlap_n"] = commitment_overlap
    composition_rows["commitment_overlap_note"] = (
        "Commitment is reported separately because it can coincide with any class."
    )

    primary.to_csv(
        experiment_dir / "activation_similarity_change_position_rows.csv",
        index=False,
    )
    rates.to_csv(
        experiment_dir / "activation_similarity_increase_event_rates.csv",
        index=False,
    )
    composition_rows.to_csv(
        experiment_dir / "activation_similarity_event_composition.csv",
        index=False,
    )

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    figure_font = "Arial" if "Arial" in available_fonts else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": figure_font,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    plot_rates = rates.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot_rates))
    values = plot_rates["rate_similarity_increased"].to_numpy()
    errors = np.vstack(
        [
            values - plot_rates["ci_low"].to_numpy(),
            plot_rates["ci_high"].to_numpy() - values,
        ]
    )
    colors = [
        GRAY if label == "No tracked action event" else BLUE
        for label in plot_rates["event"]
    ]
    axes[0].barh(y, values, color=colors, alpha=0.9)
    axes[0].errorbar(
        values,
        y,
        xerr=errors,
        fmt="none",
        ecolor="#243746",
        capsize=2,
        linewidth=1,
    )
    axes[0].set_yticks(
        y,
        labels=[
            f"{row.event} (n={row.n_eligible_positions})"
            for row in plot_rates.itertuples()
        ],
    )
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Positions with increased activation similarity")
    axes[0].set_ylabel("Action event at the current sentence")
    axes[0].set_title("How Often Does Similarity Increase?")
    axes[0].grid(axis="x", color="#E5EEF5", linewidth=0.8)

    x = np.arange(len(composition_rows))
    axes[1].bar(
        x,
        composition_rows["proportion"],
        color=[LIGHT_BLUE, BLUE, "#0B3C5D"],
    )
    axes[1].set_xticks(
        x,
        labels=[
            "Other action\nchange",
            "Optimal to\nsuboptimal",
            "Suboptimal to\noptimal",
        ],
    )
    axes[1].set_ylim(0, 0.65)
    axes[1].set_ylabel("Proportion of recommendation changes")
    axes[1].set_xlabel("Recommendation-change category")
    axes[1].set_title(
        "Events With Increased Similarity\n"
        f"(denominator: {len(increased_changes)} recommendation changes with increased similarity; "
        f"{commitment_overlap} also mark commitment)"
    )
    axes[1].grid(axis="y", color="#E5EEF5", linewidth=0.8)
    for index, row in composition_rows.iterrows():
        axes[1].text(
            index,
            row["proportion"] + 0.015,
            f"{int(row['n_positions'])}\n({row['proportion']:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(
        "Activation Similarity Changes at Action Events",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    figure_dir = experiment_dir / "figs"
    figure_dir.mkdir(exist_ok=True)
    fig.savefig(
        figure_dir / "activation_similarity_event_composition.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path(
            "outputs/experiment1_activation_monitor/"
            "gpt_oss_local_sentence_matched46_v1"
        ),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()
    build_outputs(
        args.experiment_dir,
        bootstrap_repeats=args.bootstrap_repeats,
    )


if __name__ == "__main__":
    main()
