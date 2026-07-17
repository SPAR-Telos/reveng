#!/usr/bin/env python3
"""Run recommended-action agent-location-after-move probes for matched DoorKey prefixes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reveng.experiments.behavioral_probe_runner import _behavioral_probe_preamble
from reveng.experiments.gpt_oss_activation_pilot import DEFAULT_MODEL_SNAPSHOT
from reveng.experiments.local_immediate_behavioral import (
    _analysis_text,
    _bool,
    _load_sentence_ends,
    _trajectory_step,
)

ACTIONS = {"UP", "DOWN", "LEFT", "RIGHT"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def context_text(*, grid_text: str, carrying_key: bool, reasoning_prefix: str) -> str:
    return (
        _behavioral_probe_preamble("cardinal_action_explicit")
        + "\n# Inputs\n\nCurrent grid state:\n\n"
        + grid_text
        + "\n\nAgent status:\n- Carrying key: "
        + str(carrying_key).lower()
        + "\n\nReasoning trace available so far:\n"
        + reasoning_prefix.strip()
        + "\n\n"
    )


def question_text(action: str) -> str:
    return (
        "# Question\n"
        f"Given the current grid only, if you move {action} now, what will the agent's location be after that move?\n"
        'Return exactly a JSON object of the form {"row": <int>, "col": <int>}. '
        "If the location cannot be determined, use -1 for both row and col."
    )


def extract_json_coord(text: str) -> tuple[dict[str, int] | None, str]:
    matches = re.findall(r"\{[^{}]*\}", text)
    for match in matches:
        try:
            payload = json.loads(match)
        except Exception:
            continue
        if "row" in payload and "col" in payload:
            try:
                return {"row": int(payload["row"]), "col": int(payload["col"])}, match
            except Exception:
                continue
    return None, ""


def coord_equal(left: dict[str, int] | None, right: dict[str, Any] | None) -> bool:
    if left is None or right is None:
        return False
    try:
        return int(left["row"]) == int(right["row"]) and int(left["col"]) == int(right["col"])
    except Exception:
        return False


class LocalCoordinateGenerator:
    def __init__(self, model_path: str | Path, *, max_new_tokens: int) -> None:
        self.model_path = Path(model_path)
        self.max_new_tokens = int(max_new_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        ).eval()
        self.device = self.model.model.embed_tokens.weight.device

    def ask(self, *, context: str, question: str) -> dict[str, Any]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": context + question}],
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="low",
        )
        rendered += "<|channel|>final<|message|>"
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0]
        new_ids = output_ids[inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        parsed, raw_json = extract_json_coord(text)
        return {
            "raw_response": text,
            "parsed_coord": parsed,
            "parsed_json": raw_json,
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "generated_tokens": int(new_ids.shape[0]),
        }


def build_planned_rows(
    *,
    candidate_rows_path: Path,
    position_rows_path: Path,
    trajectory_dir: Path,
    sentences_path: Path,
) -> list[dict[str, Any]]:
    candidates = {row["example_id"]: row for row in read_csv(candidate_rows_path)}
    positions = read_csv(position_rows_path)
    sentence_ends = _load_sentence_ends(sentences_path)
    payload_cache: dict[str, dict[str, Any]] = {}
    planned: list[dict[str, Any]] = []
    for position in positions:
        example_id = position["example_id"]
        candidate = candidates[example_id]
        action = position["action_label"]
        if action not in ACTIONS:
            continue
        trajectory_id = candidate["trajectory_id"]
        payload = payload_cache.get(trajectory_id)
        if payload is None:
            payload = json.loads((trajectory_dir / f"{trajectory_id}.json").read_text())
            payload_cache[trajectory_id] = payload
        step_index = int(candidate["step_index"])
        analysis = _analysis_text(_trajectory_step(payload, step_index)["output_text"])
        reasoning_step_idx = int(position["reasoning_step_idx"])
        if reasoning_step_idx == 0:
            reasoning_prefix = ""
        else:
            trace_id = f"{trajectory_id}_step_{step_index:03d}"
            boundaries = dict(sentence_ends[trace_id])
            reasoning_prefix = analysis[: boundaries[reasoning_step_idx - 1]]
        truths = json.loads(candidate["probe_truths_json"])
        truth_key = f"agent_location_after_{action.lower()}"
        planned.append(
            {
                "example_id": example_id,
                "trajectory_id": trajectory_id,
                "step_index": step_index,
                "position_index": int(position["position_index"]),
                "reasoning_step_idx": reasoning_step_idx,
                "reasoning_progress": position["reasoning_progress"],
                "action_label": action,
                "question_id": truth_key,
                "ground_truth_coord": truths.get(truth_key),
                "grid_text": candidate["grid_text"],
                "carrying_key": _bool(candidate["carrying_key"]),
                "reasoning_prefix": reasoning_prefix,
                "matched_pair_id": position.get("matched_pair_id", ""),
                "matched_role": position.get("matched_role", ""),
                "trajectory_class": position.get("trajectory_class", ""),
                "primary_step_failure_mode": position.get("primary_step_failure_mode", ""),
            }
        )
    return planned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-rows-path", type=Path, default=Path("data/behavioral_probes/reasoning_belief_action_cohorts/matched_46_candidates.csv"))
    parser.add_argument("--position-rows-path", type=Path, default=Path("outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1/position_rows.csv"))
    parser.add_argument("--trajectory-dir", type=Path, default=Path("data/hf/trajectories_key_door_100/trajectories_key_door"))
    parser.add_argument("--sentences-path", type=Path, default=Path("data/behavioral_probes/doorkey_chunking_validation/sentences.csv"))
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hypothesis_tests/transition_activation_commitment_v1/agent_location_after_recommended_action"))
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "agent_location_after_readouts.jsonl"
    completed: set[tuple[str, int]] = set()
    if args.resume and checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add((row["example_id"], int(row["position_index"])))

    planned = build_planned_rows(
        candidate_rows_path=args.candidate_rows_path,
        position_rows_path=args.position_rows_path,
        trajectory_dir=args.trajectory_dir,
        sentences_path=args.sentences_path,
    )
    if args.limit is not None:
        planned = planned[: int(args.limit)]
    pending = [row for row in planned if (row["example_id"], int(row["position_index"])) not in completed]
    manifest = {
        "status": "running" if pending else "completed",
        "planned_rows": len(planned),
        "completed_rows": len(completed),
        "pending_rows": len(pending),
        "model_path": str(args.model_path),
        "max_new_tokens": args.max_new_tokens,
        "probe_scope": "recommended_action_only",
        "probe": "agent_location_after_recommended_action",
    }
    atomic_json(args.output_dir / "run_manifest.json", manifest)

    if pending:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        reader = LocalCoordinateGenerator(args.model_path, max_new_tokens=args.max_new_tokens)
        with checkpoint.open("a") as handle:
            for row in pending:
                result = reader.ask(
                    context=context_text(
                        grid_text=row["grid_text"],
                        carrying_key=bool(row["carrying_key"]),
                        reasoning_prefix=row["reasoning_prefix"],
                    ),
                    question=question_text(row["action_label"]),
                )
                parsed = result["parsed_coord"]
                truth = row["ground_truth_coord"]
                out = {
                    "example_id": row["example_id"],
                    "trajectory_id": row["trajectory_id"],
                    "step_index": row["step_index"],
                    "position_index": row["position_index"],
                    "reasoning_step_idx": row["reasoning_step_idx"],
                    "reasoning_progress": row["reasoning_progress"],
                    "action_label": row["action_label"],
                    "question_id": row["question_id"],
                    "ground_truth_coord_json": json.dumps(truth, sort_keys=True),
                    "predicted_coord_json": json.dumps(parsed, sort_keys=True) if parsed is not None else "",
                    "answer_valid": parsed is not None,
                    "exact_match": coord_equal(parsed, truth),
                    "raw_response": result["raw_response"],
                    "parsed_json": result["parsed_json"],
                    "input_tokens": result["input_tokens"],
                    "generated_tokens": result["generated_tokens"],
                    "matched_pair_id": row["matched_pair_id"],
                    "matched_role": row["matched_role"],
                    "trajectory_class": row["trajectory_class"],
                    "primary_step_failure_mode": row["primary_step_failure_mode"],
                }
                handle.write(json.dumps(out, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    rows = [json.loads(line) for line in checkpoint.read_text().splitlines() if line.strip()] if checkpoint.exists() else []
    write_csv(args.output_dir / "agent_location_after_rows.csv", rows)
    valid = [row for row in rows if bool(row.get("answer_valid"))]
    summary_rows = []
    def add_summary(scope: str, group: list[dict[str, Any]]) -> None:
        valid_group = [row for row in group if bool(row.get("answer_valid"))]
        summary_rows.append({
            "scope": "all",
            "group": scope,
            "rows": len(group),
            "valid_rows": len(valid_group),
            "valid_rate": len(valid_group) / len(group) if group else "",
            "exact_match_accuracy_all_rows": sum(bool(row.get("exact_match")) for row in group) / len(group) if group else "",
            "exact_match_accuracy_valid_rows": sum(bool(row.get("exact_match")) for row in valid_group) / len(valid_group) if valid_group else "",
            "states": len({row["example_id"] for row in group}),
            "trajectories": len({row["trajectory_id"] for row in group}),
        })

    add_summary("all", rows)
    for field in ("matched_role", "trajectory_class", "action_label"):
        for value in sorted({str(row.get(field, "")) for row in rows}):
            group = [row for row in rows if str(row.get(field, "")) == value]
            add_summary(f"{field}={value}", group)
    write_csv(args.output_dir / "agent_location_after_summary.csv", summary_rows)
    manifest.update(
        {
            "status": "completed" if len(rows) == len(planned) else "partial",
            "planned_rows": len(planned),
            "completed_rows": len(rows),
            "pending_rows": max(0, len(planned) - len(rows)),
            "valid_rows": len(valid),
        }
    )
    atomic_json(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
