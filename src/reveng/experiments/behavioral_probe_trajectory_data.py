"""Build trajectory-derived behavioral-probe datasets from trace-viewer JSON files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from reveng.analysis.analysis_utils import ACTION_ID_TO_NAME, compute_optimal_actions_from_text_grid
from reveng.experiments.behavioral_probe_smoke_data import (
    derive_probe_truths_from_state,
    grid_text_to_layout,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _grid_lines_to_text(grid_state: list[str] | str) -> str:
    if isinstance(grid_state, str):
        return grid_state
    return "\n".join(grid_state)


def _optimal_actions_from_state(grid_text: str) -> list[str]:
    truths = derive_probe_truths_from_state(grid_text, carrying_key=False)
    goal = truths["goal_location"]
    if goal["row"] < 0 or goal["col"] < 0:
        return []
    layout = grid_text_to_layout(grid_text)
    optimal_actions, _ = compute_optimal_actions_from_text_grid(layout, (goal["col"], goal["row"]))
    agent = truths["agent_location"]
    action_ids = optimal_actions.get((agent["col"], agent["row"]), set())
    return sorted(ACTION_ID_TO_NAME[action_id] for action_id in action_ids)


def mine_behavioral_probe_instances(
    trajectory_dir: str,
    *,
    slice_type: str = "all_steps",
    max_instances: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(trajectory_dir).rglob("*.json")):
        data = json.loads(path.read_text())
        steps = data.get("steps", [])
        if not steps:
            continue
        for step_index, step in enumerate(steps):
            grid_text = _grid_lines_to_text(step.get("grid_state", []))
            carrying_key = bool(step.get("carrying_key", False))
            truths = derive_probe_truths_from_state(grid_text, carrying_key)
            observed_action = step.get("agent_action")
            optimal_actions = _optimal_actions_from_state(grid_text)
            is_optimal_action = observed_action in optimal_actions if observed_action else False
            wall_hit = False
            if observed_action:
                hit_key = f"hit_wall_after_{observed_action.lower()}"
                wall_hit = truths.get(hit_key) == "yes"

            row = {
                "example_id": f"{path.stem}_step_{step_index:03d}",
                "trajectory_id": path.stem,
                "source_file": str(path),
                "step_index": step_index,
                "grid_text": grid_text,
                "carrying_key": carrying_key,
                "observed_action": observed_action,
                "probe_truths": truths,
                "optimal_actions": optimal_actions,
                "is_optimal_action": is_optimal_action,
                "wall_hit": wall_hit,
            }
            rows.append(row)

    if slice_type == "wall_hit":
        rows = [row for row in rows if row["wall_hit"]]
    elif slice_type == "non_optimal_action":
        rows = [row for row in rows if row["observed_action"] and not row["is_optimal_action"]]
    elif slice_type != "all_steps":
        raise ValueError(f"Unknown slice_type: {slice_type}")

    if max_instances is not None:
        rows = rows[:max_instances]
    return rows


def run_behavioral_probe_trajectory_eval(
    trajectory_dir: str,
    output_dir: str = "data/behavioral_probes/trajectory_instances",
    slice_type: str = "all_steps",
    max_instances: int | None = None,
) -> None:
    """Mine a reusable single-step probe dataset from trace-viewer trajectories."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = mine_behavioral_probe_instances(
        trajectory_dir,
        slice_type=slice_type,
        max_instances=max_instances,
    )
    serializable_rows = []
    for row in instances:
        serializable_rows.append(
            {
                **row,
                "probe_truths_json": json.dumps(row["probe_truths"], sort_keys=True),
                "optimal_actions_json": json.dumps(row["optimal_actions"]),
            }
        )
        serializable_rows[-1].pop("probe_truths")
        serializable_rows[-1].pop("optimal_actions")
    _write_csv(out_dir / "trajectory_probe_instances.csv", serializable_rows)
    (out_dir / "trajectory_probe_instances.json").write_text(json.dumps(instances, indent=2))


__all__ = [
    "mine_behavioral_probe_instances",
    "run_behavioral_probe_trajectory_eval",
]
