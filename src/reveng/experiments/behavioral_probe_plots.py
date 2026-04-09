"""Plots for behavioral-probe experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


GREEDY_BLUE = "#7DB7E8"
LOGPROB_T0_BLUE = "#3C78A8"
LOGPROB_T07_AMBER = "#D98B2B"
LOGPROB_T1_ORANGE = "#F4A261"
MC_GREY = "#B8BDC7"
LIGHT_GRID = "#D9E2EF"
TEXT_COLOR = "#243447"


def plot_behavioral_probe_summary(
    summary_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    label_rows = [row for row in summary_rows if row.get("answer_space") == "label3"]
    if not label_rows:
        raise ValueError("Cannot plot behavioral-probe summary: no label3 rows provided.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    question_ids = [str(row["question_id"]) for row in label_rows]
    greedy = [float(row["greedy_accuracy"]) for row in label_rows]
    logprob_t0 = [float(row.get("logprob_yes_no_accuracy_t0", row["logprob_t0_accuracy"])) for row in label_rows]
    logprob_t07 = [float(row.get("logprob_yes_no_accuracy_t07", row["logprob_t07_accuracy"])) for row in label_rows]
    logprob_t1 = [float(row.get("logprob_yes_no_accuracy_t1", row["logprob_t1_accuracy"])) for row in label_rows]
    mc = [float(row.get("mc_yes_no_accuracy", row["mc_accuracy"])) for row in label_rows]
    entropy_t0 = [float(row["mean_entropy_t0"]) for row in label_rows]
    entropy_t07 = [float(row["mean_entropy_t07"]) for row in label_rows]
    entropy_t1 = [float(row["mean_entropy_t1"]) for row in label_rows]
    mc_entropy = [float(row["mean_mc_entropy"]) for row in label_rows]

    x = list(range(len(question_ids)))
    width = 0.16

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.bar([idx - 2 * width for idx in x], greedy, width, color=GREEDY_BLUE, label="Greedy")
    ax.bar([idx - width for idx in x], logprob_t0, width, color=LOGPROB_T0_BLUE, label="Logprob yes>no (T=0.0)")
    ax.bar([idx for idx in x], logprob_t07, width, color=LOGPROB_T07_AMBER, label="Logprob yes>no (T=0.7)")
    ax.bar([idx + width for idx in x], logprob_t1, width, color=LOGPROB_T1_ORANGE, label="Logprob yes>no (T=1.0)")
    ax.bar([idx + 2 * width for idx in x], mc, width, color=MC_GREY, label="MC sampled yes/no")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy", color=TEXT_COLOR)
    ax.set_title("Behavioral Probe Accuracy by Readout", color=TEXT_COLOR, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(question_ids, rotation=30, ha="right")
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.8, alpha=0.9)
    for boundary in range(1, len(question_ids)):
        ax.axvline(boundary - 0.5, color=LIGHT_GRID, linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", ncols=2)

    ax = axes[1]
    ax.bar([idx - 1.5 * width for idx in x], entropy_t0, width, color=LOGPROB_T0_BLUE, label="Logprob entropy (T=0.0)")
    ax.bar([idx - 0.5 * width for idx in x], entropy_t07, width, color=LOGPROB_T07_AMBER, label="Logprob entropy (T=0.7)")
    ax.bar([idx + 0.5 * width for idx in x], entropy_t1, width, color=LOGPROB_T1_ORANGE, label="Logprob entropy (T=1.0)")
    ax.bar([idx + 1.5 * width for idx in x], mc_entropy, width, color=MC_GREY, label="MC sampled entropy")
    ax.set_ylabel("Mean entropy (bits)", color=TEXT_COLOR)
    ax.set_xlabel("Question", color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(question_ids, rotation=30, ha="right")
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.8, alpha=0.9)
    for boundary in range(1, len(question_ids)):
        ax.axvline(boundary - 0.5, color=LIGHT_GRID, linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", ncols=2)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(LIGHT_GRID)
        axis.spines["bottom"].set_color(LIGHT_GRID)
        axis.tick_params(colors=TEXT_COLOR)

    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_directional_probe_heatmap(
    summary_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    directional_rows = [
        row
        for row in summary_rows
        if row.get("answer_space") == "label3"
        and row.get("question_family") in {"wall_directional", "object_directional", "action_effects"}
    ]
    if not directional_rows:
        raise ValueError("Cannot plot directional heatmap: no directional label3 rows provided.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    action_order = ["LEFT", "RIGHT", "UP", "DOWN"]
    family_order = ["hit_wall_after", "has_key_after", "door_open_after", "wall", "is_goal", "is_key", "is_door"]
    family_labels = {
        "hit_wall_after": "Hit wall after",
        "has_key_after": "Has key after",
        "door_open_after": "Door open after",
        "wall": "Wall",
        "is_goal": "Goal adjacent",
        "is_key": "Key adjacent",
        "is_door": "Door adjacent",
    }

    panels = [
        ("greedy_accuracy", "Greedy", GREEDY_BLUE),
        ("mc_yes_no_accuracy", "MC sampled yes/no", MC_GREY),
        ("logprob_yes_no_accuracy_t0", "Logprob yes>no (T=0.0)", LOGPROB_T0_BLUE),
        ("logprob_yes_no_accuracy_t1", "Logprob yes>no (T=1.0)", LOGPROB_T1_ORANGE),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for ax, (metric_key, title, _) in zip(axes, panels, strict=True):
        matrix = []
        row_labels = []
        for family_prefix in family_order:
            row = []
            include = False
            for action in action_order:
                action_lower = action.lower()
                if family_prefix in {"wall", "is_goal", "is_key", "is_door"}:
                    question_id = f"{family_prefix}_{action_lower}"
                else:
                    question_id = f"{family_prefix}_{action_lower}"
                match = next((item for item in directional_rows if item["question_id"] == question_id), None)
                row.append(float(match[metric_key]) if match and metric_key in match else 0.0)
                include = include or match is not None
            if include:
                matrix.append(row)
                row_labels.append(family_labels[family_prefix])
        image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="Blues")
        ax.set_title(title, color=TEXT_COLOR)
        ax.set_xticks(range(len(action_order)))
        ax.set_xticklabels(action_order)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        for y in range(1, len(row_labels)):
            ax.axhline(y - 0.5, color=LIGHT_GRID, linestyle="--", linewidth=0.9, alpha=0.9)
        for x in range(1, len(action_order)):
            ax.axvline(x - 0.5, color=LIGHT_GRID, linestyle="--", linewidth=0.7, alpha=0.6)
        for y, row in enumerate(matrix):
            for x, value in enumerate(row):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", color=TEXT_COLOR, fontsize=8)
        ax.tick_params(colors=TEXT_COLOR)
    fig.colorbar(image, ax=axes, shrink=0.8, label="Accuracy")
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_coordinate_probe_summary(
    summary_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    coord_rows = [row for row in summary_rows if row.get("answer_space") == "coord_json"]
    if not coord_rows:
        raise ValueError("Cannot plot coordinate summary: no coord_json rows provided.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    question_ids = [str(row["question_id"]) for row in coord_rows]
    greedy_acc = [float(row["greedy_exact_match_accuracy"]) for row in coord_rows]
    mc_acc = [float(row["mc_modal_exact_match_accuracy"]) for row in coord_rows]
    greedy_dist = [float(row["mean_greedy_manhattan_distance"]) for row in coord_rows]
    mc_dist = [float(row["mean_mc_modal_manhattan_distance"]) for row in coord_rows]
    greedy_invalid = [float(row["greedy_invalid_rate"]) for row in coord_rows]
    mc_invalid = [float(row["mc_invalid_rate"]) for row in coord_rows]

    x = list(range(len(question_ids)))
    width = 0.35
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9), constrained_layout=True)
    fig.patch.set_facecolor("white")

    axes[0].bar([idx - width / 2 for idx in x], greedy_acc, width, color=GREEDY_BLUE, label="Greedy")
    axes[0].bar([idx + width / 2 for idx in x], mc_acc, width, color=MC_GREY, label="MC modal")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Exact-match")
    axes[0].legend(frameon=False)

    axes[1].bar([idx - width / 2 for idx in x], greedy_dist, width, color=GREEDY_BLUE, label="Greedy")
    axes[1].bar([idx + width / 2 for idx in x], mc_dist, width, color=MC_GREY, label="MC modal")
    axes[1].set_ylabel("Mean Manhattan distance")

    axes[2].bar([idx - width / 2 for idx in x], greedy_invalid, width, color=GREEDY_BLUE, label="Greedy")
    axes[2].bar([idx + width / 2 for idx in x], mc_invalid, width, color=MC_GREY, label="MC")
    axes[2].set_ylabel("Invalid parse rate")
    axes[2].set_xlabel("Question")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(question_ids, rotation=30, ha="right")
        ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(LIGHT_GRID)
        ax.spines["bottom"].set_color(LIGHT_GRID)
        ax.tick_params(colors=TEXT_COLOR)

    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


__all__ = [
    "plot_behavioral_probe_summary",
    "plot_coordinate_probe_summary",
    "plot_directional_probe_heatmap",
]
