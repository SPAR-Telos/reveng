"""Failure-case analysis for behavioral probes on trajectory-derived states."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from reveng.experiments.behavioral_probe_questions import get_behavioral_probe_questions
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


def run_behavioral_probe_case_studies(
    trajectory_dir: str,
    model_name: str = "together_ai/openai/gpt-oss-20b",
    output_dir: str = "data/behavioral_probes/case_studies",
    slice_type: str = "non_optimal_action",
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
                "grid_text": instance["grid_text"],
            }
        )
    _write_csv(out_dir / "case_study_rows.csv", case_rows)

    summary = {
        "n_instances": len(instances),
        "slice_type": slice_type,
        "wall_hit_count": sum(1 for row in instances if row["wall_hit"]),
        "non_optimal_count": sum(1 for row in instances if not row["is_optimal_action"]),
    }
    _write_csv(out_dir / "case_study_summary.csv", [summary])

    markdown_lines = [
        "# Behavioral Probe Case Studies",
        "",
        f"- Slice type: `{slice_type}`",
        f"- Instances: `{len(instances)}`",
        "",
        "| Example | Observed action | Optimal actions | Wall hit |",
        "|---|---|---|---|",
    ]
    for instance in instances:
        markdown_lines.append(
            f"| {instance['example_id']} | {instance['observed_action']} | "
            f"{', '.join(instance['optimal_actions']) or '[]'} | {instance['wall_hit']} |"
        )
    (out_dir / "case_study.md").write_text("\n".join(markdown_lines) + "\n")


__all__ = ["run_behavioral_probe_case_studies"]
