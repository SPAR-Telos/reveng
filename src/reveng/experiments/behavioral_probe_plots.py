"""Plots for behavioral-probe smoke test outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


LIGHT_BLUE = "#7DB7E8"
ORANGE = "#F4A261"
LIGHT_GRID = "#D9E2EF"
TEXT_COLOR = "#243447"


def plot_behavioral_probe_summary(
    summary_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Render a compact summary figure for behavioral-probe smoke-test metrics."""
    if not summary_rows:
        raise ValueError("Cannot plot behavioral-probe summary: no rows provided.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    question_ids = [str(row["question_id"]) for row in summary_rows]
    greedy = [float(row["greedy_accuracy"]) for row in summary_rows]
    modal = [float(row["modal_sampled_accuracy"]) for row in summary_rows]
    entropy = [float(row["mean_entropy"]) for row in summary_rows]

    x = list(range(len(question_ids)))
    width = 0.36

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.8), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.bar(
        [idx - width / 2 for idx in x],
        greedy,
        width=width,
        color=LIGHT_BLUE,
        edgecolor="white",
        linewidth=0.8,
        label="Greedy accuracy",
    )
    ax.bar(
        [idx + width / 2 for idx in x],
        modal,
        width=width,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.8,
        label="Modal sampled accuracy",
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy", color=TEXT_COLOR)
    ax.set_title("Behavioral-Probe Smoke Test Summary", color=TEXT_COLOR, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(question_ids, rotation=30, ha="right")
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    ax.bar(
        x,
        entropy,
        width=0.55,
        color=LIGHT_BLUE,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_ylabel("Mean entropy (bits)", color=TEXT_COLOR)
    ax.set_xlabel("Question", color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(question_ids, rotation=30, ha="right")
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(LIGHT_GRID)
        axis.spines["bottom"].set_color(LIGHT_GRID)
        axis.tick_params(colors=TEXT_COLOR)

    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


__all__ = ["plot_behavioral_probe_summary"]
