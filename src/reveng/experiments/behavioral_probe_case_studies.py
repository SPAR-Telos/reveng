"""Failure-case analysis for behavioral probes on trajectory-derived states."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from reveng.experiments.behavioral_probe_questions import get_behavioral_probe_questions
from reveng.experiments.behavioral_probe_plots import plot_failure_mode_case_selection
from reveng.experiments.behavioral_probe_runner import run_behavioral_probe_on_instances
from reveng.experiments.behavioral_probe_trajectory_data import mine_behavioral_probe_instances


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _display_stage(instance: dict[str, Any]) -> str:
    if instance.get("is_pre_failure_context"):
        return f"state just before tagged failure step {instance.get('pre_failure_for_step_index')}"
    return "tagged failure state"


def _display_reason(reason: str) -> str:
    return reason.replace("pre_failure_context", "previous_state_before_failure")


def _failure_only(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [instance for instance in instances if not instance.get("is_pre_failure_context")]


def run_behavioral_probe_case_studies(
    trajectory_dir: str,
    model_name: str = "together_ai/openai/gpt-oss-20b",
    output_dir: str = "data/behavioral_probes/case_studies",
    slice_type: str = "selection_candidates",
    max_instances: int = 12,
    prompt_preset: str = "cardinal_action_explicit",
    verbose: bool = True,
) -> None:
    """Run probe families on trajectory-derived failure cases and write a compact markdown report."""
    instances = mine_behavioral_probe_instances(
        trajectory_dir,
        slice_type=slice_type,
        max_instances=max_instances,
    )
    instances = sorted(instances, key=lambda row: (row["trajectory_id"], int(row["step_index"])))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = get_behavioral_probe_questions(question_family="wall_directional", answer_space="label3") + get_behavioral_probe_questions(question_family="object_directional", answer_space="label3")
    run_behavioral_probe_on_instances(
        model_name=model_name,
        instances=instances,
        questions=questions,
        output_dir=str(out_dir),
        prompt_preset=prompt_preset,
        verbose=verbose,
    )

    case_rows = []
    for instance in instances:
        case_rows.append(
            {
                "example_id": instance["example_id"],
                "trajectory_id": instance["trajectory_id"],
                "step_index": instance["step_index"],
                "observed_action": instance["observed_action"],
                "optimal_actions_json": json.dumps(instance["optimal_actions"]),
                "is_optimal_action": instance["is_optimal_action"],
                "wall_hit": instance["wall_hit"],
                "failure_modes_json": json.dumps(instance["failure_modes"]),
                "primary_step_failure_mode": instance["primary_step_failure_mode"],
                "is_failure_onset": instance["is_failure_onset"],
                "is_terminal_failure_tail": instance["is_terminal_failure_tail"],
                "is_pre_failure_context": instance["is_pre_failure_context"],
                "pre_failure_for_step_index": instance["pre_failure_for_step_index"],
                "selection_stage": instance["selection_stage"],
                "selected_for_probe": instance["selected_for_probe"],
                "selection_reason": instance["selection_reason"],
                "grid_text": instance["grid_text"],
            }
        )
    _write_csv(out_dir / "case_study_rows.csv", case_rows)

    summary = {
        "n_instances": len(instances),
        "slice_type": slice_type,
        "wall_hit_count": sum(1 for row in instances if row["wall_hit"]),
        "non_optimal_count": sum(1 for row in instances if not row["is_optimal_action"]),
        "selected_for_probe_count": sum(1 for row in instances if row["selected_for_probe"]),
        "failure_onset_count": sum(1 for row in instances if row["is_failure_onset"]),
        "pre_failure_context_count": sum(1 for row in instances if row["is_pre_failure_context"]),
    }
    _write_csv(out_dir / "case_study_summary.csv", [summary])

    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    failure_instances = _failure_only(instances)
    plot_failure_mode_case_selection(failure_instances, figs_dir / "failure_mode_case_selection.png")
    if len(failure_instances) != len(instances):
        plot_failure_mode_case_selection(instances, figs_dir / "failure_mode_case_selection_with_context.png")

    markdown_lines = [
        "# Behavioral Probe Case Studies",
        "",
        f"- Slice type: `{slice_type}`",
        f"- Instances: `{len(instances)}`",
        "- `state just before tagged failure`: the immediately preceding distinct state, included only as comparison context; it is not itself a failure-mode label.",
        "- `tagged failure state`: the single trajectory step selected because it matches a deterministic failure-mode rule such as wall hit, backtrack, loop, freeze, or avoidable detour.",
        "- Probes remain Markovian: each row is queried independently as one rendered state, with no trajectory history in the prompt.",
        "- Main figure: tagged failure states only. Appendix figure: tagged failure states plus comparison states immediately before them.",
        "",
        "| Example | Stage | Mode | Observed action | Optimal actions | Selected | Reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for instance in failure_instances:
        markdown_lines.append(
            f"| {instance['example_id']} | {_display_stage(instance)} | {instance['primary_step_failure_mode']} | "
            f"{instance['observed_action']} | {', '.join(instance['optimal_actions']) or '[]'} | "
            f"{instance['selected_for_probe']} | {_display_reason(instance['selection_reason']) or 'n/a'} |"
        )
    (out_dir / "case_study.md").write_text("\n".join(markdown_lines) + "\n")


__all__ = ["run_behavioral_probe_case_studies"]
