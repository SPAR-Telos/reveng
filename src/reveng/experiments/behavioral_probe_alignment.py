"""Matched-input alignment utilities for white-box and black-box probes."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from reveng.experiments.behavioral_probe_parse import parse_behavioral_probe_answer
from reveng.experiments.behavioral_probe_questions import (
    BehavioralProbeQuestion,
    get_behavioral_probe_questions,
)
from reveng.experiments.behavioral_probe_runner import run_behavioral_probe_on_instances


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return list(payload["rows"])
    raise ValueError(f"Unsupported matched-row format: {path}")


def _normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _normalize_question_id(row: dict[str, Any]) -> str:
    for key in ("question_id", "target_name", "target_variable"):
        value = row.get(key)
        if value:
            return str(value)
    raise ValueError("Matched probe row missing question_id/target_name/target_variable.")


def _parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def _all_question_by_id(answer_space: str) -> dict[str, BehavioralProbeQuestion]:
    return {
        question.question_id: question
        for question in get_behavioral_probe_questions(question_family="all", answer_space=answer_space)
    }


def load_matched_probe_rows(
    matched_rows_path: str,
    *,
    answer_space: str = "label3",
    reasoning_split: str | None = None,
    question_family: str | None = None,
) -> list[dict[str, Any]]:
    path = Path(matched_rows_path)
    rows = _load_rows(path)
    question_by_id = _all_question_by_id(answer_space)
    allowed_ids: set[str] | None = None
    if question_family is not None:
        allowed_ids = {
            question.question_id
            for question in get_behavioral_probe_questions(
                question_family=question_family,
                answer_space=answer_space,
            )
        }

    normalized_rows: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        question_id = _normalize_question_id(row)
        if question_id not in question_by_id:
            raise ValueError(
                f"Matched probe row references unsupported {answer_space} question_id: {question_id}"
            )
        if allowed_ids is not None and question_id not in allowed_ids:
            continue

        row["question_id"] = question_id
        row["ground_truth_label"] = str(
            row.get("ground_truth_label", row.get("ground_truth", ""))
        )
        row["example_id"] = str(row.get("example_id", ""))
        if not row["example_id"]:
            raise ValueError(f"Matched probe row for {question_id} is missing example_id.")

        split = str(row.get("reasoning_split", "pre") or "pre")
        row["reasoning_split"] = split
        if reasoning_split is not None and split != reasoning_split:
            continue

        state_description_text = row.get("state_description_text", row.get("state_text", ""))
        row["state_description_text"] = str(state_description_text or "")
        row["grid_text"] = str(row.get("grid_text", ""))
        if not row["state_description_text"] and not row["grid_text"]:
            raise ValueError(
                f"Matched probe row {row['example_id']}/{question_id} is missing both state_description_text and grid_text."
            )

        carrying_key = row.get("carrying_key", row.get("has_key_observed", False))
        carrying_key_bool = _normalize_bool(carrying_key)
        row["carrying_key"] = False if carrying_key_bool is None else carrying_key_bool
        row["trajectory_id"] = str(row.get("trajectory_id", ""))
        row["step_index"] = str(row.get("step_index", ""))
        row["observed_action"] = str(row.get("observed_action", ""))
        row["optimal_actions"] = _parse_json_list(row.get("optimal_actions_json", row.get("optimal_actions", [])))
        row["is_optimal_action"] = _normalize_bool(row.get("is_optimal_action"))
        row["wall_hit"] = _normalize_bool(row.get("wall_hit"))
        row["source_dataset"] = str(row.get("source_dataset", "matched_whitebox"))
        row["whitebox_prediction"] = str(
            row.get("whitebox_prediction", row.get("probe_prediction", ""))
        )
        row["whitebox_score"] = row.get("whitebox_score", row.get("probe_score", ""))
        if not row["ground_truth_label"]:
            raise ValueError(
                f"Matched probe row {row['example_id']}/{question_id} is missing ground_truth_label."
            )
        normalized_rows.append(row)
    return normalized_rows


def select_behavioral_questions_for_matched_rows(
    matched_rows: list[dict[str, Any]],
    *,
    answer_space: str = "label3",
) -> list[BehavioralProbeQuestion]:
    question_by_id = _all_question_by_id(answer_space)
    question_ids = sorted({str(row["question_id"]) for row in matched_rows})
    return [question_by_id[question_id] for question_id in question_ids]


def build_behavioral_instances_from_matched_rows(
    matched_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in matched_rows:
        grouped[(str(row["example_id"]), str(row["reasoning_split"]))].append(row)

    instances: list[dict[str, Any]] = []
    for (example_id, split), group_rows in sorted(grouped.items()):
        base = group_rows[0]
        allowed_question_ids = [str(row["question_id"]) for row in group_rows]
        probe_truths = {str(row["question_id"]): str(row["ground_truth_label"]) for row in group_rows}

        for row in group_rows[1:]:
            shared_keys = (
                "trajectory_id",
                "step_index",
                "grid_text",
                "state_description_text",
                "observed_action",
            )
            for key in shared_keys:
                if row.get(key) != base.get(key):
                    raise ValueError(
                        f"Inconsistent matched rows for {example_id}/{split}: field {key} differs across targets."
                    )

        instances.append(
            {
                "example_id": example_id,
                "trajectory_id": base.get("trajectory_id", ""),
                "step_index": base.get("step_index", ""),
                "grid_text": base.get("grid_text", ""),
                "state_description_text": base.get("state_description_text", ""),
                "carrying_key": bool(base.get("carrying_key", False)),
                "observed_action": base.get("observed_action") or None,
                "optimal_actions": list(base.get("optimal_actions", [])),
                "is_optimal_action": base.get("is_optimal_action", ""),
                "wall_hit": base.get("wall_hit", ""),
                "probe_truths": probe_truths,
                "allowed_question_ids": allowed_question_ids,
                "reasoning_split": split,
                "source_dataset": base.get("source_dataset", "matched_whitebox"),
            }
        )
    return instances


def build_behavioral_probe_instances_from_matched_rows(
    matched_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return build_behavioral_instances_from_matched_rows(matched_rows)


def build_probe_alignment_rows(
    matched_rows: list[dict[str, Any]],
    behavioral_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    matched_index = {
        (str(row["example_id"]), str(row["reasoning_split"]), str(row["question_id"])): row
        for row in matched_rows
    }
    alignment_rows: list[dict[str, Any]] = []
    for blackbox_row in behavioral_rows:
        key = (
            str(blackbox_row.get("example_id", "")),
            str(blackbox_row.get("reasoning_split", "pre")),
            str(blackbox_row.get("question_id", "")),
        )
        matched_row = matched_index.get(key)
        if matched_row is None:
            continue

        whitebox_prediction = parse_behavioral_probe_answer(matched_row.get("whitebox_prediction"))
        ground_truth = str(matched_row["ground_truth_label"])
        blackbox_greedy_answer = str(blackbox_row.get("greedy_answer", "invalid"))
        blackbox_mc_answer = str(blackbox_row.get("mc_yes_no_answer", blackbox_row.get("mc_answer", "invalid")))
        blackbox_greedy_correct = blackbox_greedy_answer == ground_truth
        blackbox_mc_correct = blackbox_mc_answer == ground_truth
        whitebox_correct = (
            whitebox_prediction == ground_truth if whitebox_prediction != "invalid" else None
        )
        action_non_optimal = None
        is_optimal_action = matched_row.get("is_optimal_action")
        if isinstance(is_optimal_action, bool):
            action_non_optimal = not is_optimal_action

        wall_hit = matched_row.get("wall_hit")
        wall_hit_bool = wall_hit if isinstance(wall_hit, bool) else None

        unified_row = {
            **matched_row,
            "blackbox_greedy_answer": blackbox_greedy_answer,
            "blackbox_greedy_modal_answer": str(
                blackbox_row.get("greedy_modal_answer", blackbox_greedy_answer)
            ),
            "blackbox_mc_answer": str(blackbox_row.get("mc_answer", blackbox_mc_answer)),
            "blackbox_mc_yes_no_answer": blackbox_mc_answer,
            "blackbox_logprob_t0_yes_no_answer": str(
                blackbox_row.get("logprob_yes_no_answer_t0", blackbox_row.get("logprob_answer_t0", "invalid"))
            ),
            "blackbox_greedy_correct": blackbox_greedy_correct,
            "blackbox_mc_correct": blackbox_mc_correct,
            "whitebox_parsed_prediction": whitebox_prediction,
            "whitebox_correct": whitebox_correct,
            "whitebox_blackbox_agree": (
                whitebox_prediction == blackbox_greedy_answer if whitebox_prediction != "invalid" else None
            ),
            "both_correct": (
                bool(whitebox_correct) and blackbox_greedy_correct if whitebox_correct is not None else None
            ),
            "whitebox_correct_blackbox_wrong": (
                bool(whitebox_correct) and not blackbox_greedy_correct if whitebox_correct is not None else None
            ),
            "blackbox_correct_whitebox_wrong": (
                blackbox_greedy_correct and whitebox_correct is False if whitebox_correct is not None else None
            ),
            "action_non_optimal": action_non_optimal,
            "both_correct_action_non_optimal": (
                bool(whitebox_correct) and blackbox_greedy_correct and action_non_optimal
                if whitebox_correct is not None and action_non_optimal is not None
                else None
            ),
            "wall_hit_bool": wall_hit_bool,
            "both_correct_wall_hit": (
                bool(whitebox_correct) and blackbox_greedy_correct and wall_hit_bool
                if whitebox_correct is not None and wall_hit_bool is not None
                else None
            ),
            "blackbox_belief_action_consistency": blackbox_row.get("belief_action_consistency", ""),
            "blackbox_mc_belief_action_consistency": blackbox_row.get("mc_belief_action_consistency", ""),
        }
        alignment_rows.append(unified_row)
    return alignment_rows


def summarize_probe_alignment_rows(alignment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in alignment_rows:
        grouped[(str(row.get("reasoning_split", "pre")), str(row["question_id"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (split, question_id), rows in sorted(grouped.items()):
        def _mean_bool(key: str) -> float:
            values = [row[key] for row in rows if isinstance(row.get(key), bool)]
            return sum(1.0 if value else 0.0 for value in values) / len(values) if values else 0.0

        whitebox_rows = [row for row in rows if row.get("whitebox_parsed_prediction") not in {"", "invalid", None}]
        both_correct_rows = [row for row in rows if row.get("both_correct") is True]
        both_correct_non_optimal = [row for row in both_correct_rows if row.get("action_non_optimal") is True]
        both_correct_wall_hit = [row for row in both_correct_rows if row.get("wall_hit_bool") is True]

        summary_rows.append(
            {
                "reasoning_split": split,
                "question_id": question_id,
                "n_rows": len(rows),
                "trajectory_rows_with_whitebox": len(whitebox_rows),
                "blackbox_greedy_accuracy": _mean_bool("blackbox_greedy_correct"),
                "blackbox_mc_accuracy": _mean_bool("blackbox_mc_correct"),
                "whitebox_accuracy": _mean_bool("whitebox_correct"),
                "whitebox_blackbox_agreement": _mean_bool("whitebox_blackbox_agree"),
                "both_correct_count": len(both_correct_rows),
                "both_correct_rate": len(both_correct_rows) / len(whitebox_rows) if whitebox_rows else 0.0,
                "both_correct_action_non_optimal_count": len(both_correct_non_optimal),
                "both_correct_action_non_optimal_rate": (
                    len(both_correct_non_optimal) / len(both_correct_rows) if both_correct_rows else 0.0
                ),
                "both_correct_wall_hit_count": len(both_correct_wall_hit),
                "both_correct_wall_hit_rate": (
                    len(both_correct_wall_hit) / len(both_correct_rows) if both_correct_rows else 0.0
                ),
                "whitebox_correct_blackbox_wrong_count": sum(
                    1 for row in rows if row.get("whitebox_correct_blackbox_wrong") is True
                ),
                "blackbox_correct_whitebox_wrong_count": sum(
                    1 for row in rows if row.get("blackbox_correct_whitebox_wrong") is True
                ),
            }
        )
    return summary_rows


def build_probe_alignment_gap_cases(alignment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in alignment_rows
        if row.get("both_correct_action_non_optimal") is True
        or row.get("both_correct_wall_hit") is True
    ]


def run_behavioral_probe_matched_eval(
    matched_rows_path: str,
    model_name: str = "together_ai/openai/gpt-oss-20b",
    output_dir: str = "data/behavioral_probes/matched_eval",
    *,
    reasoning_split: str = "pre",
    question_family: str = "wall_directional",
    answer_space: str = "label3",
    prompt_preset: str = "cardinal_action_explicit",
    min_valid_parse_rate: float = 0.90,
    mc_sample_repeats: int = 10,
    mc_temperature: float = 0.7,
    top_logprobs: int = 20,
    logprob_temperatures: tuple[float, ...] = (0.0, 0.7, 1.0),
    verbose: bool = True,
    checkpoint_dir: str | None = None,
    greedy_repeats: int = 10,
) -> None:
    matched_rows = load_matched_probe_rows(
        matched_rows_path,
        answer_space=answer_space,
        reasoning_split=reasoning_split,
        question_family=question_family,
    )
    if not matched_rows:
        raise ValueError("No matched probe rows remain after filtering by reasoning_split/question_family.")

    instances = build_behavioral_probe_instances_from_matched_rows(matched_rows)
    questions = select_behavioral_questions_for_matched_rows(matched_rows, answer_space=answer_space)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "matched_probe_input_rows.csv", matched_rows)
    _write_csv(out_dir / "matched_probe_instances.csv", instances)

    run_behavioral_probe_on_instances(
        model_name=model_name,
        instances=instances,
        questions=questions,
        output_dir=str(out_dir),
        prompt_preset=prompt_preset,
        min_valid_parse_rate=min_valid_parse_rate,
        mc_sample_repeats=mc_sample_repeats,
        mc_temperature=mc_temperature,
        top_logprobs=top_logprobs,
        logprob_temperatures=logprob_temperatures,
        verbose=verbose,
        checkpoint_dir=checkpoint_dir or str(out_dir / "checkpoints"),
        greedy_repeats=greedy_repeats,
    )

    with open(out_dir / "behavioral_probe_rows.csv", newline="") as handle:
        behavioral_rows = list(csv.DictReader(handle))
    alignment_rows = build_probe_alignment_rows(matched_rows, behavioral_rows)
    alignment_summary_rows = summarize_probe_alignment_rows(alignment_rows)
    gap_case_rows = build_probe_alignment_gap_cases(alignment_rows)
    _write_csv(out_dir / "probe_alignment_rows.csv", alignment_rows)
    _write_csv(out_dir / "probe_alignment_summary.csv", alignment_summary_rows)
    _write_csv(out_dir / "probe_alignment_gap_cases.csv", gap_case_rows)


__all__ = [
    "build_behavioral_instances_from_matched_rows",
    "build_behavioral_probe_instances_from_matched_rows",
    "build_probe_alignment_gap_cases",
    "build_probe_alignment_rows",
    "load_matched_probe_rows",
    "run_behavioral_probe_matched_eval",
    "select_behavioral_questions_for_matched_rows",
    "summarize_probe_alignment_rows",
]
