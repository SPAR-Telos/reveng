"""Build a clarified summary table for scaled gradual-CoT black-box runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

DEFAULT_SLICE_DIRS: dict[str, str] = {
    "non_failure_controls": "data/behavioral_probes/gradual_cot_alignment_non_failure_controls",
    "balanced_failure_modes": "data/behavioral_probes/gradual_cot_alignment_balanced_failure_modes",
    "short_loop": "data/behavioral_probes/gradual_cot_alignment_short_loop",
}

DEFAULT_OUTPUT_CSV = "data/behavioral_probes/gradual_cot_scaling_clarified_summary.csv"
DEFAULT_OUTPUT_MD = "data/behavioral_probes/gradual_cot_scaling_clarified_summary.md"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: str | float) -> str:
    return f"{100 * float(value):.1f}%"


def build_gradual_cot_scaling_summary(
    *,
    slice_dirs: dict[str, str] = DEFAULT_SLICE_DIRS,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    output_md: str = DEFAULT_OUTPUT_MD,
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for slice_name, slice_dir in slice_dirs.items():
        table_path = Path(slice_dir) / "gradual_blackbox_reasoning_table.csv"
        summary_rows = _read_csv(table_path)
        for row in summary_rows:
            rows_out.append(
                {
                    "slice": slice_name,
                    "reasoning_reveal_pct": int(row["reasoning_reveal_pct"]),
                    "n_wall_question_rows": int(row["n_rows"]),
                    "adjacent_wall_belief_accuracy": float(row["blackbox_wall_accuracy"]),
                    "probed_next_action_in_optimal_action_set_rate": float(row["action_greedy_modal_optimality_rate"]),
                    "chosen_direction_wall_belief_contradiction_rate": float(row["local_belief_action_gap_rate"]),
                    "chosen_direction_wall_belief_correct_but_action_suboptimal_rate": float(row["astar_conditioned_gap_rate"]),
                    "wall_belief_entropy": float(row["mean_blackbox_entropy"]),
                }
            )

    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_csv_path, rows_out)

    md_lines = [
        "# Clarified Gradual-CoT Scaling Summary",
        "",
        "Columns",
        "- `adjacent_wall_belief_accuracy`: accuracy of the black-box wall-state probe against the ground-truth adjacent-wall label.",
        "- `probed_next_action_in_optimal_action_set_rate`: the revealed-CoT next-action query asks the model what action it would take; this column reports whether that probed next action is in the state's optimal action set.",
        "- `chosen_direction_wall_belief_contradiction_rate`: among rows where the wall question targets the same direction as the model's chosen action and the wall answer is a valid `yes/no`, the rate at which the probe says that chosen direction is blocked (`yes`).",
        "- `chosen_direction_wall_belief_correct_but_action_suboptimal_rate`: among rows where the wall question targets the model's chosen direction and the wall answer is both valid and correct, the rate at which the model's chosen action is still suboptimal.",
        "",
        "Not represented in this table",
        "- `probed optimal action, evaluated against optimal action`",
        "- `probed optimal action, evaluated against model action`",
        "- `probed next/model action, evaluated against an external model-action label`",
        "",
        "| Slice | Reveal % | Wall belief accuracy | Probed next action in optimal set | Probe says chosen direction is blocked | Probe is correct about chosen direction but action is suboptimal |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows_out:
        md_lines.append(
            "| {slice} | {reasoning_reveal_pct} | {adj} | {opt} | {contr} | {subopt} |".format(
                slice=row["slice"],
                reasoning_reveal_pct=row["reasoning_reveal_pct"],
                adj=_pct(row["adjacent_wall_belief_accuracy"]),
                opt=_pct(row["probed_next_action_in_optimal_action_set_rate"]),
                contr=_pct(row["chosen_direction_wall_belief_contradiction_rate"]),
                subopt=_pct(row["chosen_direction_wall_belief_correct_but_action_suboptimal_rate"]),
            )
        )
    Path(output_md).write_text("\n".join(md_lines) + "\n")
    return rows_out
