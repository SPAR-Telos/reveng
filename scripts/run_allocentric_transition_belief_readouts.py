#!/usr/bin/env python3
"""Run allocentric counterfactual transition-belief readouts for DoorKey prefixes."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import torch

from reveng.experiments.behavioral_probe_smoke_data import grid_text_to_layout
from reveng.experiments.behavioral_probe_trajectory_data import DoorKeyStateSolver, StateKey
from reveng.experiments.local_immediate_behavioral import (
    LocalImmediateReadout,
    _analysis_text,
    _bool,
    _context_text,
    _entropy,
    _load_sentence_ends,
    _trajectory_step,
)
from reveng.experiments.gpt_oss_activation_pilot import DEFAULT_MODEL_SNAPSHOT

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
RAW_TO_SEMANTIC = {"A": "yes", "B": "no", "C": "unknown"}
QUESTIONS = (
    ("hit_wall", "will the hypothetical move hit a wall"),
    ("has_key", "will the hypothetical agent have the key after the move"),
    ("door_open", "will any door be open after the move"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_grid(layout: list[list[str]]) -> str:
    width = len(layout[0]) if layout else 0
    lines = ["  " + " ".join(str(i) for i in range(width))]
    for row_idx, row in enumerate(layout):
        lines.append(f"{row_idx} " + " ".join(row))
    return "\n".join(lines)


def normalize_grid(grid_text: str, *, agent_row: int, agent_col: int, has_key: bool, door_open: bool) -> str:
    layout = grid_text_to_layout(grid_text)
    for r, row in enumerate(layout):
        for c, value in enumerate(row):
            if value == "A":
                layout[r][c] = "_"
            elif value in {"D", "O"}:
                layout[r][c] = "O" if door_open else "D"
            elif value == "K" and has_key:
                layout[r][c] = "_"
    layout[agent_row][agent_col] = "A"
    return render_grid(layout)


def traversable_empty_cells(grid_text: str) -> list[tuple[int, int]]:
    layout = grid_text_to_layout(grid_text)
    cells: list[tuple[int, int]] = []
    for r, row in enumerate(layout):
        for c, value in enumerate(row):
            if value == "_":
                cells.append((r, c))
    return cells


def any_open_door(grid_text: str) -> bool:
    return any(token == "O" for line in grid_text.splitlines()[1:] for token in line.split()[1:])


def make_cases(candidate: dict[str, str], *, cases_per_state: int = 4) -> list[dict[str, Any]]:
    cells = traversable_empty_cells(candidate["grid_text"])
    if not cells:
        return []
    # Deterministic spread through available empty cells. Pair cases with actions
    # and key/door status toggles to avoid collapsing onto the actual agent state.
    step = max(1, len(cells) // max(1, cases_per_state))
    selected = [cells[(idx * step) % len(cells)] for idx in range(cases_per_state)]
    cases: list[dict[str, Any]] = []
    solver = DoorKeyStateSolver()
    for idx, (row, col) in enumerate(selected):
        action = ACTIONS[idx % len(ACTIONS)]
        has_key = bool(idx % 2)
        door_open = bool((idx // 2) % 2)
        hypo_grid = normalize_grid(
            candidate["grid_text"],
            agent_row=row,
            agent_col=col,
            has_key=has_key,
            door_open=door_open,
        )
        state: StateKey = (hypo_grid, has_key)
        transition = solver.step(state, action)
        next_grid, next_has_key = transition["next_state"]
        cases.append(
            {
                "case_index": idx,
                "source_row": row,
                "source_col": col,
                "has_key": has_key,
                "door_open": door_open,
                "action": action,
                "hypothetical_grid": hypo_grid,
                "truth_hit_wall": "yes" if transition["hit_wall"] else "no",
                "truth_has_key": "yes" if next_has_key else "no",
                "truth_door_open": "yes" if any_open_door(next_grid) else "no",
            }
        )
    return cases


def question_tuple(case: dict[str, Any], question_id: str, text: str) -> tuple[str, str, tuple[str, ...]]:
    prompt = (
        "# Question\n"
        "Ignore the current agent marker in the grid for this question. "
        f"Consider a hypothetical agent at row {case['source_row']}, column {case['source_col']}. "
        f"The hypothetical agent has_key={str(case['has_key']).lower()} and door_open={str(case['door_open']).lower()}. "
        f"If that hypothetical agent moves {case['action']}, {text}?\n"
        "Return one label only: A = yes, B = no, C = unknown."
    )
    return question_id, prompt, ("A", "B", "C")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-rows-path", default="data/behavioral_probes/reasoning_belief_action_cohorts/matched_46_candidates.csv")
    parser.add_argument("--prefix-rows-path", default="outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1/position_rows.csv")
    parser.add_argument("--output-dir", default="outputs/hypothesis_tests/transition_activation_commitment_v1/allocentric_readouts")
    parser.add_argument("--trajectory-dir", default="data/hf/trajectories_key_door_100/trajectories_key_door")
    parser.add_argument("--sentences-path", default="data/behavioral_probes/doorkey_chunking_validation/sentences.csv")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_SNAPSHOT))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cases-per-state", type=int, default=4)
    parser.add_argument("--limit-positions", type=int)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "allocentric_transition_readouts.jsonl"
    completed = set()
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add((row["example_id"], int(row["position_index"])))

    candidates = {row["example_id"]: row for row in read_csv(Path(args.candidate_rows_path))}
    prefix_rows = read_csv(Path(args.prefix_rows_path))
    if args.limit_positions is not None:
        prefix_rows = prefix_rows[: args.limit_positions]
    sentence_ends = _load_sentence_ends(Path(args.sentences_path))
    payload_cache: dict[str, dict[str, Any]] = {}
    reader = LocalImmediateReadout(args.model_path, temperature=args.temperature)
    torch.manual_seed(42)
    rows: list[dict[str, Any]] = []
    with checkpoint.open("a") as handle:
        for prefix in prefix_rows:
            key = (prefix["example_id"], int(prefix["position_index"]))
            if key in completed:
                continue
            candidate = candidates[prefix["example_id"]]
            trajectory_id = candidate["trajectory_id"]
            payload = payload_cache.get(trajectory_id)
            if payload is None:
                payload = json.loads((Path(args.trajectory_dir) / f"{trajectory_id}.json").read_text())
                payload_cache[trajectory_id] = payload
            step_index = int(candidate["step_index"])
            analysis = _analysis_text(_trajectory_step(payload, step_index)["output_text"])
            position_index = int(prefix["position_index"])
            if position_index == 0:
                reasoning = ""
            else:
                trace_id = f"{trajectory_id}_step_{step_index:03d}"
                boundaries = dict(sentence_ends[trace_id])
                reasoning = analysis[: boundaries[position_index - 1]]
            context = _context_text(
                grid_text=candidate["grid_text"],
                carrying_key=_bool(candidate["carrying_key"]),
                reasoning_prefix=reasoning,
            )
            questions = []
            truth_by_qid: dict[str, str] = {}
            for case in make_cases(candidate, cases_per_state=args.cases_per_state):
                for short, text in QUESTIONS:
                    qid = f"allocentric_{case['case_index']}_{short}"
                    questions.append(question_tuple(case, qid, text))
                    truth_by_qid[qid] = str(case[f"truth_{short}"])
            readouts = reader.score_questions(context=context, questions=questions)
            out_row = {
                "example_id": prefix["example_id"],
                "trajectory_id": trajectory_id,
                "step_index": step_index,
                "position_index": position_index,
                "reasoning_progress": prefix["reasoning_progress"],
                "readouts": [],
            }
            for qid, result in readouts.items():
                answer = RAW_TO_SEMANTIC[result["answer"]]
                truth = truth_by_qid[qid]
                out_row["readouts"].append(
                    {
                        "question_id": qid,
                        "answer_key": answer,
                        "ground_truth_key": truth,
                        "belief_is_error": answer != truth,
                        "entropy_bits": result["entropy_bits"],
                        "probabilities": {RAW_TO_SEMANTIC[k]: v for k, v in result["probabilities"].items()},
                    }
                )
            handle.write(json.dumps(out_row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    # Materialize flat CSV from all checkpoint rows.
    for line in checkpoint.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for item in row["readouts"]:
            rows.append({
                "example_id": row["example_id"],
                "trajectory_id": row["trajectory_id"],
                "step_index": row["step_index"],
                "position_index": row["position_index"],
                "reasoning_progress": row["reasoning_progress"],
                **{k: v for k, v in item.items() if k != "probabilities"},
                "probabilities_json": json.dumps(item["probabilities"], sort_keys=True),
            })
    write_csv(output / "allocentric_transition_belief_rows.csv", rows)
    positions_in_checkpoint = len({
        (r["example_id"], r["position_index"]) for r in rows
    })
    status = "completed" if positions_in_checkpoint == len(prefix_rows) else "partial"
    (output / "run_manifest.json").write_text(json.dumps({
        "status": status,
        "rows": len(rows),
        "expected_positions": len(prefix_rows),
        "positions_in_checkpoint": positions_in_checkpoint,
        "cases_per_state": args.cases_per_state,
        "temperature": args.temperature,
    }, indent=2, sort_keys=True) + "\n")
    parent_manifest_path = output.parent / "run_manifest.json"
    if parent_manifest_path.exists():
        parent_manifest = json.loads(parent_manifest_path.read_text())
        parent_manifest["allocentric_probe_readouts"] = {
            "status": status,
            "path": str(output / "allocentric_transition_belief_rows.csv"),
            "positions": positions_in_checkpoint,
            "rows": len(rows),
        }
        parent_manifest_path.write_text(
            json.dumps(parent_manifest, indent=2, sort_keys=True) + "\n"
        )
    print(output / "allocentric_transition_belief_rows.csv")


if __name__ == "__main__":
    main()
