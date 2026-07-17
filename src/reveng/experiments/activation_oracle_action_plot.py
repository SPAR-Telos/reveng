from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


TEXT_COLOR = "#203040"
GRID_COLOR = "#D7E3F1"
MODEL_ACTION_BLUE = "#1D4E89"
MODEL_VS_OPTIMAL_BLUE = "#5B8CC0"
OPTIMAL_ACTION_BLUE = "#A9C8E8"


def _parse_percent(cell: str) -> float:
    match = re.search(r"\(([\d.]+)%\)", cell)
    if not match:
        raise ValueError(f"Could not parse percentage from cell: {cell!r}")
    return float(match.group(1))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_activation_oracle_action_summary(
    summary_csv: str = "data/activation_oracle/ao_action_summary_table.csv",
    output_png: str = "figs/activation_oracle_action_clean_failure.png",
    output_pdf: str = "figs/activation_oracle_action_clean_failure.pdf",
) -> None:
    rows = _read_csv(summary_csv)
    panel_order = ["clean public slice", "failure-focused revealed-CoT slice"]
    panel_titles = {
        "clean public slice": "Clean Slice",
        "failure-focused revealed-CoT slice": "Failure Slice",
    }
    metric_specs = [
        ("ask_next_action_vs_model_action", "Ask next action -> model action", MODEL_ACTION_BLUE),
        ("ask_next_action_vs_optimal_action", "Ask next action -> optimal action set", MODEL_VS_OPTIMAL_BLUE),
        ("ask_optimal_action_vs_optimal_action", "Ask optimal action -> optimal action set", OPTIMAL_ACTION_BLUE),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True, constrained_layout=True)
    fig.patch.set_facecolor("white")

    for ax, panel_key in zip(axes, panel_order, strict=True):
        panel_rows = [
            row for row in rows
            if row["slice"] == panel_key and row["revealed_reasoning_fraction_pct"] != "Overall"
        ]
        panel_rows.sort(key=lambda row: int(row["revealed_reasoning_fraction_pct"]))
        x = [int(row["revealed_reasoning_fraction_pct"]) for row in panel_rows]

        for metric_key, label, color in metric_specs:
            y = [_parse_percent(row[metric_key]) for row in panel_rows]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.2,
                markersize=5.5,
                color=color,
                label=label,
            )

        ax.set_title(panel_titles[panel_key], color=TEXT_COLOR, fontsize=13, pad=10)
        ax.set_xlim(-2, 102)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Revealed reasoning fraction (%)", color=TEXT_COLOR)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.9)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.5, alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID_COLOR)
        ax.spines["bottom"].set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR)

    axes[0].set_ylabel("Accuracy (%)", color=TEXT_COLOR)
    axes[0].set_ylim(0, 105)
    axes[0].set_yticks([0, 20, 40, 60, 80, 100])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        frameon=False,
        ncols=3,
        fontsize=9.5,
    )

    for output_path in (output_png, output_pdf):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")

    plt.close(fig)


__all__ = ["plot_activation_oracle_action_summary"]
