"""Plots for behavioral-probe experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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
    *,
    mc_sample_repeats: int | None = None,
    mc_temperature: float | None = None,
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
    mc_label = "MC sampled yes/no"
    mc_entropy_label = "MC sampled entropy"
    if mc_sample_repeats is not None and mc_temperature is not None:
        mc_label = f"MC sampled yes/no (n={mc_sample_repeats}, T={mc_temperature})"
        mc_entropy_label = f"MC sampled entropy (n={mc_sample_repeats}, T={mc_temperature})"

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.bar([idx - 2 * width for idx in x], greedy, width, color=GREEDY_BLUE, label="Greedy")
    ax.bar([idx - width for idx in x], logprob_t0, width, color=LOGPROB_T0_BLUE, label="Logprob yes>no (T=0.0)")
    ax.bar([idx for idx in x], logprob_t07, width, color=LOGPROB_T07_AMBER, label="Logprob yes>no (T=0.7)")
    ax.bar([idx + width for idx in x], logprob_t1, width, color=LOGPROB_T1_ORANGE, label="Logprob yes>no (T=1.0)")
    ax.bar([idx + 2 * width for idx in x], mc, width, color=MC_GREY, label=mc_label)
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
    ax.bar([idx + 1.5 * width for idx in x], mc_entropy, width, color=MC_GREY, label=mc_entropy_label)
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
    *,
    mc_sample_repeats: int | None = None,
    mc_temperature: float | None = None,
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

    mc_title = "MC sampled yes/no"
    if mc_sample_repeats is not None and mc_temperature is not None:
        mc_title = f"MC sampled yes/no (n={mc_sample_repeats}, T={mc_temperature})"
    panels = [
        ("greedy_accuracy", "Greedy", GREEDY_BLUE),
        ("mc_yes_no_accuracy", mc_title, MC_GREY),
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


def plot_belief_action_gap_summary(
    gap_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    if not gap_rows:
        raise ValueError("Cannot plot belief-action gap summary: no gap rows provided.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(row["direction"]) for row in gap_rows]
    greedy_accuracy = [float(row["greedy_accuracy"]) for row in gap_rows]
    mc_accuracy = [float(row["mc_yes_no_accuracy"]) for row in gap_rows]
    local_gap = [float(row["local_belief_action_gap_rate"]) for row in gap_rows]
    mc_local_gap = [float(row["mc_local_belief_action_gap_rate"]) for row in gap_rows]
    astar_gap = [float(row["greedy_astar_gap_rate"]) for row in gap_rows]
    mc_astar_gap = [float(row["mc_astar_gap_rate"]) for row in gap_rows]

    x = list(range(len(labels)))
    width = 0.18
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.bar([idx - width / 2 for idx in x], greedy_accuracy, width, color=GREEDY_BLUE, label="Greedy probe accuracy")
    ax.bar([idx + width / 2 for idx in x], mc_accuracy, width, color=MC_GREY, label="MC probe accuracy")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Accuracy", color=TEXT_COLOR)
    ax.set_title("Can the Model Correctly Report Walls in Each Direction?", color=TEXT_COLOR, fontsize=13)
    for xpos, value in zip([idx - width / 2 for idx in x], greedy_accuracy, strict=True):
        ax.text(xpos, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=8.5, color=TEXT_COLOR)
    for xpos, value in zip([idx + width / 2 for idx in x], mc_accuracy, strict=True):
        ax.text(xpos, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=8.5, color=TEXT_COLOR)
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    positions = {
        "local": [idx - 1.5 * width for idx in x],
        "mc_local": [idx - 0.5 * width for idx in x],
        "astar": [idx + 0.5 * width for idx in x],
        "mc_astar": [idx + 1.5 * width for idx in x],
    }
    bars = [
        (positions["local"], local_gap, GREEDY_BLUE, "Moves into a wall it reports, greedy"),
        (positions["mc_local"], mc_local_gap, MC_GREY, "Moves into a wall it reports, MC"),
        (positions["astar"], astar_gap, LOGPROB_T0_BLUE, "Non-A* move despite correct wall report, greedy"),
        (positions["mc_astar"], mc_astar_gap, "#6F747D", "Non-A* move despite correct wall report, MC"),
    ]
    for xpos, values, color, label in bars:
        ax.bar(xpos, values, width, color=color, label=label)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Gap rate", color=TEXT_COLOR)
    ax.set_xlabel("Observed-action direction", color=TEXT_COLOR)
    ax.set_title("Does the Model Act Against Correctly Reported Wall Information?", color=TEXT_COLOR, fontsize=13)
    ax.text(
        0.0,
        1.03,
        "Fractions above bars = gap cases / eligible single-step states in this subset, not trajectories.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=TEXT_COLOR,
        va="bottom",
    )

    annotations = [
        ("local", "local_gap_count", "local_gap_denominator", "local_belief_action_gap_rate"),
        ("mc_local", "mc_local_gap_count", "mc_local_gap_denominator", "mc_local_belief_action_gap_rate"),
        ("astar", "greedy_astar_gap_count", "greedy_astar_gap_denominator", "greedy_astar_gap_rate"),
        ("mc_astar", "mc_astar_gap_count", "mc_astar_gap_denominator", "mc_astar_gap_rate"),
    ]
    for key, count_key, denom_key, rate_key in annotations:
        for xpos, row in zip(positions[key], gap_rows, strict=True):
            value = float(row[rate_key])
            label = f"{int(float(row[count_key]))}/{int(float(row[denom_key]))}"
            ax.text(xpos, value + 0.025, label, ha="center", va="bottom", fontsize=8.0, color=TEXT_COLOR)
    ax.legend(frameon=False, loc="upper left", ncols=2)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.8, alpha=0.9)
        for boundary in range(1, len(labels)):
            ax.axvline(boundary - 0.5, color=LIGHT_GRID, linestyle="--", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(LIGHT_GRID)
        ax.spines["bottom"].set_color(LIGHT_GRID)
        ax.tick_params(colors=TEXT_COLOR)

    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_failure_mode_gap_summary(
    gap_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    if not gap_rows:
        raise ValueError("Cannot plot failure-mode gap summary: no gap rows provided.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(row["failure_mode_label"]) for row in gap_rows]
    counts = [int(float(row["n_action_relevant_states"])) for row in gap_rows]
    greedy_acc = [float(row["greedy_observed_action_wall_accuracy"]) for row in gap_rows]
    mc_acc = [float(row["mc_observed_action_wall_accuracy"]) for row in gap_rows]
    local_gap = [float(row["local_belief_action_gap_rate"]) for row in gap_rows]
    mc_local_gap = [float(row["mc_local_belief_action_gap_rate"]) for row in gap_rows]
    astar_gap = [float(row["greedy_optimality_conditioned_gap_rate"]) for row in gap_rows]
    mc_astar_gap = [float(row["mc_optimality_conditioned_gap_rate"]) for row in gap_rows]

    x = list(range(len(labels)))
    width = 0.18
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.6), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.bar([idx - width / 2 for idx in x], greedy_acc, width, color=GREEDY_BLUE, label="Greedy observed-action wall accuracy")
    ax.bar([idx + width / 2 for idx in x], mc_acc, width, color=MC_GREY, label="MC observed-action wall accuracy")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Accuracy", color=TEXT_COLOR)
    ax.set_title("How Accurate Is the Observed-Action Wall Probe Within Each Failure Mode?", color=TEXT_COLOR, fontsize=13)
    for idx, count in enumerate(counts):
        ax.text(idx, 1.04, f"n={count}", ha="center", va="bottom", fontsize=8.5, color=TEXT_COLOR)
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    positions = {
        "local": [idx - 1.5 * width for idx in x],
        "mc_local": [idx - 0.5 * width for idx in x],
        "astar": [idx + 0.5 * width for idx in x],
        "mc_astar": [idx + 1.5 * width for idx in x],
    }
    bars = [
        (positions["local"], local_gap, GREEDY_BLUE, "Moves into a wall it reports, greedy"),
        (positions["mc_local"], mc_local_gap, MC_GREY, "Moves into a wall it reports, MC"),
        (positions["astar"], astar_gap, LOGPROB_T0_BLUE, "Non-A* move despite correct wall report, greedy"),
        (positions["mc_astar"], mc_astar_gap, "#6F747D", "Non-A* move despite correct wall report, MC"),
    ]
    for xpos, values, color, label in bars:
        ax.bar(xpos, values, width, color=color, label=label)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Gap rate", color=TEXT_COLOR)
    ax.set_xlabel("Failure mode", color=TEXT_COLOR)
    ax.set_title("Which Failure Modes Contain Belief-Action Mismatches?", color=TEXT_COLOR, fontsize=13)
    ax.text(
        0.0,
        1.03,
        "Fractions above bars = gap cases / eligible tagged failure states in this subset, not trajectories.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=TEXT_COLOR,
        va="bottom",
    )

    annotations = [
        ("local", "local_belief_action_gap_count", "local_belief_action_gap_denominator", "local_belief_action_gap_rate"),
        ("mc_local", "mc_local_belief_action_gap_count", "mc_local_belief_action_gap_denominator", "mc_local_belief_action_gap_rate"),
        ("astar", "greedy_optimality_conditioned_gap_count", "greedy_optimality_conditioned_gap_denominator", "greedy_optimality_conditioned_gap_rate"),
        ("mc_astar", "mc_optimality_conditioned_gap_count", "mc_optimality_conditioned_gap_denominator", "mc_optimality_conditioned_gap_rate"),
    ]
    for key, count_key, denom_key, rate_key in annotations:
        for xpos, row in zip(positions[key], gap_rows, strict=True):
            value = float(row[rate_key])
            label = f"{int(float(row[count_key]))}/{int(float(row[denom_key]))}"
            ax.text(xpos, value + 0.025, label, ha="center", va="bottom", fontsize=8.0, color=TEXT_COLOR)
    ax.legend(frameon=False, loc="upper left", ncols=2)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.8, alpha=0.9)
        for boundary in range(1, len(labels)):
            ax.axvline(boundary - 0.5, color=LIGHT_GRID, linestyle="--", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(LIGHT_GRID)
        ax.spines["bottom"].set_color(LIGHT_GRID)
        ax.tick_params(colors=TEXT_COLOR)

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
    "plot_belief_action_gap_summary",
    "plot_behavioral_probe_summary",
    "plot_coordinate_probe_summary",
    "plot_directional_probe_heatmap",
    "plot_failure_mode_gap_summary",
]


def plot_wall_hit_suboptimality(
    case_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    if not case_rows:
        raise ValueError("Cannot plot wall-hit sub-optimality: no case rows provided.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    row_spacing = 2.25
    top_margin = 2.45
    fig, ax = plt.subplots(figsize=(15.5, max(5.6, 2.1 * len(case_rows) + 2.4)), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(case_rows) * row_spacing + top_margin)
    ax.axis("off")

    include_source = any("source" in row for row in case_rows)
    headers = [("Case", 0.03)]
    if include_source:
        headers.append(("Source", 0.23))
        observed_x = 0.34
        optimal_x = 0.45
        subopt_x = 0.58
        directional_x = 0.72
        action_x = 0.90
        grid_x = 1.13
        x_max = 1.29
    else:
        observed_x = 0.28
        optimal_x = 0.39
        subopt_x = 0.53
        directional_x = 0.66
        action_x = 0.83
        grid_x = 1.04
        x_max = 1.21
    headers.extend(
        [
            ("Observed", observed_x),
            ("Optimal", optimal_x),
            ("Non-opt", subopt_x),
            ("Wall in\nobserved\ndir.?", directional_x),
            ("Observed\naction\nhits wall?", action_x),
            ("Full grid", grid_x),
        ]
    )
    for label, xpos in headers:
        ax.text(
            xpos,
            len(case_rows) * row_spacing + 1.02,
            label,
            fontsize=10,
            fontweight="bold",
            color=TEXT_COLOR,
            ha="center" if xpos > 0.1 else "left",
            va="center",
        )
    ax.hlines(len(case_rows) * row_spacing + 0.82, 0.02, x_max - 0.02, color=LIGHT_GRID, linewidth=1.2)
    ax.set_xlim(0, x_max)

    def _status_color(value: str) -> str:
        return {"yes": "#D8F0DA", "no": "#F7D6D6"}.get(value.lower(), "#E9EDF2")

    def _draw_status_cell(x_center: float, y_center: float, text: str) -> None:
        width = 0.085
        height = 0.58
        ax.add_patch(
            Rectangle(
                (x_center - width / 2, y_center - height / 2),
                width,
                height,
                facecolor=_status_color(text),
                edgecolor=LIGHT_GRID,
                linewidth=1.0,
            )
        )
        ax.text(x_center, y_center, text, ha="center", va="center", fontsize=10, color=TEXT_COLOR)

    for idx, row in enumerate(case_rows):
        y = len(case_rows) * row_spacing - idx * row_spacing
        ax.hlines(y - 1.12, 0.02, x_max - 0.02, color=LIGHT_GRID, linestyle="--", linewidth=0.8, alpha=0.9)
        ax.text(0.03, y, str(row["example_id"]).replace("together_ai_openai_gpt-oss-20b_", ""), fontsize=9, color=TEXT_COLOR, va="center")
        if include_source:
            ax.text(0.23, y, str(row["source"]), fontsize=9, color=TEXT_COLOR, va="center", ha="center")
        ax.text(observed_x, y, str(row["observed_action"]), fontsize=10, color=TEXT_COLOR, va="center", ha="center")
        ax.text(optimal_x, y, str(row["optimal_actions"]), fontsize=10, color=TEXT_COLOR, va="center", ha="center")
        ax.text(subopt_x, y, "yes" if row["is_suboptimal"] else "no", fontsize=10, color=TEXT_COLOR, va="center", ha="center")
        _draw_status_cell(directional_x, y, str(row["directional_probe"]))
        _draw_status_cell(action_x, y, str(row["action_probe"]))
        grid_text = str(row["grid_text"])
        ax.text(
            grid_x,
            y,
            grid_text,
            fontsize=7.0,
            family="monospace",
            color=TEXT_COLOR,
            va="center",
            ha="center",
        )

    directional_hits = sum(1 for row in case_rows if str(row["directional_probe"]).lower() == "yes")
    action_hits = sum(1 for row in case_rows if str(row["action_probe"]).lower() == "yes")
    subtitle = (
        f"Wall-hit steps: {len(case_rows)} | non-optimal: "
        f"{sum(1 for row in case_rows if row['is_suboptimal'])}/{len(case_rows)} | "
        f"wall-in-direction probe: {directional_hits}/{len(case_rows)} | "
        f"action-hits-wall probe: {action_hits}/{len(case_rows)}"
    )
    ax.text(
        0.03,
        len(case_rows) * row_spacing + 1.55,
        "Wall-Hit Sub-Optimality Cases",
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
        va="center",
    )
    ax.text(0.03, len(case_rows) * row_spacing + 1.22, subtitle, fontsize=10, color=TEXT_COLOR, va="center")

    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


__all__.append("plot_wall_hit_suboptimality")


def plot_failure_mode_summary(
    manifest_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    if not manifest_rows:
        raise ValueError("Cannot plot failure-mode summary: no manifest rows provided.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    mode_order = [
        "wall_hit",
        "backtrack",
        "oscillation_2cycle",
        "short_loop",
        "freeze_repeat",
        "avoidable_detour",
    ]
    mode_labels = {
        "wall_hit": "Wall hit",
        "backtrack": "Backtrack",
        "oscillation_2cycle": "2-cycle",
        "short_loop": "Short loop",
        "freeze_repeat": "Freeze repeat",
        "avoidable_detour": "Avoidable detour",
    }

    suboptimal_rows = [row for row in manifest_rows if row.get("trajectory_class") == "suboptimal_success"]
    failed_rows = [row for row in manifest_rows if row.get("trajectory_class") == "failed"]
    suboptimal_counts = [sum(1 for row in suboptimal_rows if row.get(f"contains_{mode}")) for mode in mode_order]
    failed_counts = [sum(1 for row in failed_rows if row.get(f"contains_{mode}")) for mode in mode_order]

    x = list(range(len(mode_order)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.bar([idx - width / 2 for idx in x], suboptimal_counts, width, color=LOGPROB_T0_BLUE, label="Suboptimal success")
    ax.bar([idx + width / 2 for idx in x], failed_counts, width, color=MC_GREY, label="Failed")
    ax.set_xticks(x)
    ax.set_xticklabels([mode_labels[mode] for mode in mode_order], rotation=20, ha="right")
    ax.set_ylabel("Trajectory count", color=TEXT_COLOR)
    ax.set_title("Failure Modes by Trajectory Outcome", color=TEXT_COLOR, fontsize=14)
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LIGHT_GRID)
    ax.spines["bottom"].set_color(LIGHT_GRID)
    ax.tick_params(colors=TEXT_COLOR)

    for idx, value in enumerate(suboptimal_counts):
        ax.text(idx - width / 2, value + 0.05, str(value), ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)
    for idx, value in enumerate(failed_counts):
        ax.text(idx + width / 2, value + 0.05, str(value), ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)

    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_failure_mode_case_selection(
    case_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    if not case_rows:
        raise ValueError("Cannot plot failure-mode case selection: no case rows provided.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    row_spacing = 2.35
    top_margin = 1.9
    fig, ax = plt.subplots(figsize=(16.5, max(5.6, 2.15 * len(case_rows) + 2.4)), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1.42)
    ax.set_ylim(0, len(case_rows) * row_spacing + top_margin)
    ax.axis("off")

    headers = [
        ("Trajectory", 0.03),
        ("Step", 0.30),
        ("What this\nrow is", 0.39),
        ("Failure mode", 0.49),
        ("Observed", 0.63),
        ("Optimal", 0.76),
        ("Linked\nfailure\nstep", 0.91),
        ("Selected", 1.02),
        ("Reason", 1.14),
        ("Full grid", 1.33),
    ]
    for label, xpos in headers:
        ax.text(
            xpos,
            len(case_rows) * row_spacing + 1.28,
            label,
            fontsize=10,
            fontweight="bold",
            color=TEXT_COLOR,
            ha="center" if xpos > 0.1 else "left",
            va="center",
        )
    ax.hlines(len(case_rows) * row_spacing + 0.58, 0.02, 1.39, color=LIGHT_GRID, linewidth=1.2)

    for idx, row in enumerate(case_rows):
        y = len(case_rows) * row_spacing - idx * row_spacing
        ax.hlines(y - 1.12, 0.02, 1.39, color=LIGHT_GRID, linestyle="--", linewidth=0.8, alpha=0.9)
        ax.text(0.03, y, str(row["trajectory_id"]).replace("together_ai_openai_gpt-oss-20b_", ""), fontsize=8.5, color=TEXT_COLOR, va="center")
        ax.text(0.30, y, str(row["step_index"]), fontsize=10, color=TEXT_COLOR, va="center", ha="center")
        role = "state just\nbefore failure" if row.get("is_pre_failure_context") else "tagged\nfailure state"
        ax.text(0.39, y, role, fontsize=8.5, color=TEXT_COLOR, va="center", ha="center")
        ax.text(0.49, y, str(row.get("primary_step_failure_mode", "none")), fontsize=9.5, color=TEXT_COLOR, va="center", ha="center")
        ax.text(0.63, y, str(row.get("observed_action", "")), fontsize=10, color=TEXT_COLOR, va="center", ha="center")
        ax.text(0.76, y, str(row.get("optimal_actions", "")), fontsize=9.5, color=TEXT_COLOR, va="center", ha="center")
        pair_value = ""
        if row.get("is_pre_failure_context"):
            pair_value = f"failure\nstep {row.get('pre_failure_for_step_index', '')}"
        elif row.get("selection_stage") == "failure":
            pair_value = ""
        ax.text(0.91, y, pair_value, fontsize=8.0, color=TEXT_COLOR, va="center", ha="center")
        ax.text(1.02, y, "yes" if row.get("selected_for_probe") else "no", fontsize=10, color=TEXT_COLOR, va="center", ha="center")
        reason = str(row.get("selection_reason", "")).replace("pre_failure_context", "previous_state")
        ax.text(1.14, y, reason, fontsize=8.0, color=TEXT_COLOR, va="center", ha="center")
        ax.text(
            1.33,
            y,
            str(row["grid_text"]),
            fontsize=7.0,
            family="monospace",
            color=TEXT_COLOR,
            va="center",
            ha="center",
        )

    ax.text(
        0.03,
        len(case_rows) * row_spacing + 1.95,
        "Trajectory States Selected for Behavioral-Probe Case Studies",
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
        va="center",
    )
    ax.text(
        0.03,
        len(case_rows) * row_spacing + 1.62,
        "Rows labeled 'state just before failure' are comparison states immediately before the linked tagged failure state; probes still see one state at a time.",
        fontsize=9.5,
        color=TEXT_COLOR,
        va="center",
    )
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


__all__.extend(["plot_failure_mode_summary", "plot_failure_mode_case_selection"])
