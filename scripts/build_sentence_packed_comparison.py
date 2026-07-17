#!/usr/bin/env python3
"""Compare the 8-state sentence and packed-boundary behavioral runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    value_float = as_float(value)
    return int(value_float) if value_float is not None else None


def character_progress(rows: list[dict[str, str]]) -> dict[tuple[str, int], float]:
    maximum = {
        example_id: max(int(row.get("revealed_analysis_chars", 0)) for row in example_rows)
        for example_id, example_rows in group_by_example(rows).items()
    }
    return {
        (row["example_id"], int(row["reasoning_step_idx"])): (
            int(row.get("revealed_analysis_chars", 0)) / maximum[row["example_id"]]
            if maximum[row["example_id"]]
            else 0.0
        )
        for row in rows
    }


def group_by_example(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["example_id"]].append(row)
    return grouped


def build_position_rows(
    *,
    label: str,
    exp1_dir: Path,
    exp2_dir: Path,
) -> list[dict[str, Any]]:
    prefix_rows = read_csv(exp1_dir / "prefix_action_rows.csv")
    exp2_positions = {
        (row["example_id"], int(row["reasoning_step_idx"])): row
        for row in read_csv(exp2_dir / "position_rows.csv")
    }
    uncertainty = {
        (row["example_id"], int(row["reasoning_step_idx"])): row
        for row in read_csv(exp2_dir / "state_uncertainty_position_rows.csv")
    }
    categorical_by_position: dict[tuple[str, int], list[float]] = defaultdict(list)
    primary_ids = {
        "wall_left", "wall_right", "wall_up", "wall_down", "has_key", "door_open"
    }
    for belief_row in read_csv(exp2_dir / "belief_rows.csv"):
        if belief_row.get("question_id") not in primary_ids:
            continue
        value = as_float(belief_row.get("categorical_entropy_bits"))
        if value is not None:
            categorical_by_position[
                (belief_row["example_id"], int(belief_row["reasoning_step_idx"]))
            ].append(value)
    progress = character_progress(prefix_rows)
    output: list[dict[str, Any]] = []
    for row in prefix_rows:
        key = (row["example_id"], int(row["reasoning_step_idx"]))
        belief = exp2_positions.get(key, {})
        state_uncertainty = uncertainty.get(key, {})
        output.append(
            {
                "boundary_type": label,
                "example_id": row["example_id"],
                "trajectory_id": row["trajectory_id"],
                "step_index": row["step_index"],
                "boundary_index": row["reasoning_step_idx"],
                "reasoning_character_progress": progress[key],
                "revealed_analysis_chars": row.get("revealed_analysis_chars", ""),
                "action_label": row.get("action_label", ""),
                "action_is_optimal": row.get("action_is_optimal", ""),
                "action_entropy_bits": row.get(
                    "action_logprob_entropy_bits", row.get("action_mc_entropy", "")
                ),
                "primary_belief_error_rate": belief.get("primary_belief_error_rate", ""),
                "state_belief_entropy_mean_bits": state_uncertainty.get(
                    "state_belief_entropy_mean_bits",
                    mean(categorical_by_position[key])
                    if categorical_by_position.get(key)
                    else "",
                ),
            }
        )
    return output


def summary_row(
    *,
    label: str,
    exp1_dir: Path,
    exp2_dir: Path,
    position_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    prefix_rows = read_csv(exp1_dir / "prefix_action_rows.csv")
    trajectory_rows = read_csv(exp1_dir / "trajectory_wrong_turn_summary.csv")
    event_rows = read_csv(exp2_dir / "action_transition_rows.csv")
    event_counts = Counter(row["event_type"] for row in event_rows)
    progress = character_progress(prefix_rows)
    commitment_progress: list[float] = []
    for row in trajectory_rows:
        commitment = as_int(row.get("commitment_step"))
        if commitment is not None:
            value = progress.get((row["example_id"], commitment))
            if value is not None:
                commitment_progress.append(value)
    action_changes = 0
    for rows in group_by_example(prefix_rows).values():
        ordered = sorted(rows, key=lambda row: int(row["reasoning_step_idx"]))
        action_changes += sum(
            current.get("action_label") != previous.get("action_label")
            for previous, current in zip(ordered, ordered[1:])
            if previous.get("action_label") != "INVALID" and current.get("action_label") != "INVALID"
        )
    action_entropy = [
        value for row in position_rows
        if (value := as_float(row.get("action_entropy_bits"))) is not None
    ]
    belief_error = [
        value for row in position_rows
        if (value := as_float(row.get("primary_belief_error_rate"))) is not None
    ]
    belief_entropy = [
        value for row in position_rows
        if (value := as_float(row.get("state_belief_entropy_mean_bits"))) is not None
    ]
    return {
        "boundary_type": label,
        "n_states": len({row["example_id"] for row in prefix_rows}),
        "n_prefix_positions": len(prefix_rows),
        "n_action_identity_changes": action_changes,
        "n_sustained_optimality_losses": event_counts["sustained_optimal_to_suboptimal"],
        "n_transient_optimality_losses": event_counts["transient_optimal_to_suboptimal"],
        "n_optimality_recoveries": event_counts["suboptimal_to_optimal_recovery"],
        "n_commitment_onsets": len(commitment_progress),
        "mean_commitment_character_progress": (
            mean(commitment_progress) if commitment_progress else ""
        ),
        "mean_action_entropy_bits": mean(action_entropy) if action_entropy else "",
        "mean_primary_belief_error_rate": mean(belief_error) if belief_error else "",
        "mean_state_belief_entropy_bits": mean(belief_entropy) if belief_entropy else "",
    }


def binned_state_means(
    rows: list[dict[str, Any]], metric: str
) -> dict[str, dict[int, list[float]]]:
    state_bin: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        value = as_float(row.get(metric))
        progress = as_float(row.get("reasoning_character_progress"))
        if value is None or progress is None:
            continue
        bin_index = min(9, max(0, int(progress * 10)))
        state_bin[(row["boundary_type"], row["example_id"], bin_index)].append(value)
    output: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (label, _example_id, bin_index), values in state_bin.items():
        output[label][bin_index].append(mean(values))
    return output


def plot_comparison(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    run_dirs: dict[str, Path],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    labels = ("Packed units", "Sentences")
    colors = {"Packed units": "#8bb8df", "Sentences": "#1f5f9f"}
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    for ax, metric, title, ylabel in (
        (
            axes[0],
            "action_entropy_bits",
            "Recommended-action uncertainty",
            "Entropy from T=0.7 action-token logprobs (bits)",
        ),
        (axes[1], "primary_belief_error_rate", "State-belief errors", "Mean error rate: wall, key, and door"),
    ):
        grouped = binned_state_means(rows, metric)
        for label in labels:
            bins = sorted(grouped.get(label, {}))
            if bins:
                ax.plot(
                    [(index + 0.5) / 10 for index in bins],
                    [mean(grouped[label][index]) for index in bins],
                    marker="o",
                    linewidth=1.8,
                    color=colors[label],
                    label=label,
                )
        ax.set_title(title)
        ax.set_xlabel("Fraction of reasoning characters revealed")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", color="#d7e3f0", linewidth=0.8)
    commitment_by_label: dict[str, dict[str, float]] = {}
    for label, run_dir in run_dirs.items():
        prefixes = read_csv(run_dir / "prefix_action_rows.csv")
        progress = character_progress(prefixes)
        commitment_by_label[label] = {}
        for row in read_csv(run_dir / "trajectory_wrong_turn_summary.csv"):
            step = as_int(row.get("commitment_step"))
            if step is not None and (row["example_id"], step) in progress:
                commitment_by_label[label][row["example_id"]] = progress[(row["example_id"], step)]
    shared = sorted(set(commitment_by_label.get(labels[0], {})) & set(commitment_by_label.get(labels[1], {})))
    for idx, example_id in enumerate(shared):
        xs = [commitment_by_label[label][example_id] for label in labels]
        axes[2].plot(xs, [idx, idx], color="#cbdbea", linewidth=1)
        for label, x in zip(labels, xs):
            axes[2].scatter(x, idx, color=colors[label], s=30, label=label if idx == 0 else None)
    axes[2].set_title("First stable recommendation")
    axes[2].set_xlabel("Fraction of reasoning characters revealed")
    axes[2].set_ylabel("Environment state")
    axes[2].set_yticks(range(len(shared)))
    axes[2].set_yticklabels([f"step {example_id.rsplit('_', 1)[-1]}" for example_id in shared])
    axes[2].set_xlim(0, 1)
    axes[2].grid(axis="x", color="#d7e3f0", linewidth=0.8)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Effect of Reasoning Boundary Choice", fontsize=11, y=1.03)
    fig.tight_layout()
    figs = output_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / "sentence_vs_packed_boundaries.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packed-exp1-dir",
        type=Path,
        default=Path("outputs/experiment1_activation_monitor/weisheng_8_state_canonical_prefix_actions"),
    )
    parser.add_argument(
        "--sentence-exp1-dir",
        type=Path,
        default=Path("outputs/experiment1_activation_monitor/weisheng_8_state_sentence_experiment1"),
    )
    parser.add_argument(
        "--packed-exp2-dir",
        type=Path,
        default=Path("outputs/experiment2_behavioral_beliefs/weisheng_8_state_v1"),
    )
    parser.add_argument(
        "--sentence-exp2-dir",
        type=Path,
        default=Path("outputs/experiment2_behavioral_beliefs/weisheng_8_state_sentence_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiment1_activation_monitor/weisheng_8_state_sentence_vs_packed"),
    )
    args = parser.parse_args()
    run_specs = (
        ("Packed units", args.packed_exp1_dir, args.packed_exp2_dir),
        ("Sentences", args.sentence_exp1_dir, args.sentence_exp2_dir),
    )
    position_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for label, exp1_dir, exp2_dir in run_specs:
        run_positions = build_position_rows(label=label, exp1_dir=exp1_dir, exp2_dir=exp2_dir)
        position_rows.extend(run_positions)
        summary_rows.append(
            summary_row(
                label=label,
                exp1_dir=exp1_dir,
                exp2_dir=exp2_dir,
                position_rows=run_positions,
            )
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "comparison_position_rows.csv", position_rows)
    write_csv(args.output_dir / "comparison_summary.csv", summary_rows)
    lines = [
        "# Sentence and Packed-Boundary Comparison",
        "",
        "All timing values use the fraction of reasoning characters revealed.",
        "",
        "| Boundary | States | Prefixes | Action changes | Sustained losses | Transient losses | Recoveries | Commitments | Mean commitment progress | Mean action entropy | Mean belief error | Mean belief entropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {boundary_type} | {n_states} | {n_prefix_positions} | {n_action_identity_changes} | "
            "{n_sustained_optimality_losses} | {n_transient_optimality_losses} | "
            "{n_optimality_recoveries} | {n_commitment_onsets} | "
            "{mean_commitment_character_progress:.3f} | {mean_action_entropy_bits:.3f} | "
            "{mean_primary_belief_error_rate:.3f} | {mean_state_belief_entropy_bits:.3f} |".format(**row)
        )
    (args.output_dir / "comparison_summary.md").write_text("\n".join(lines) + "\n")
    plot_comparison(
        output_dir=args.output_dir,
        rows=position_rows,
        run_dirs={label: exp1_dir for label, exp1_dir, _ in run_specs},
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "comparison_axis": "fraction_of_reasoning_characters_revealed",
                "packed_exp1_dir": str(args.packed_exp1_dir),
                "sentence_exp1_dir": str(args.sentence_exp1_dir),
                "packed_exp2_dir": str(args.packed_exp2_dir),
                "sentence_exp2_dir": str(args.sentence_exp2_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
