#!/usr/bin/env python3
"""Expose integrated categorical logprob rows to the existing regression builder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from reveng.experiments.reasoning_belief_action import PRIMARY_QUESTION_IDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    with (args.run_dir / "belief_rows.csv").open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = []
    for row in source_rows:
        if row.get("question_id") not in PRIMARY_QUESTION_IDS:
            continue
        probs = json.loads(row.get("categorical_probs_json") or "{}")
        entropy = row.get("categorical_entropy_bits", "")
        rows.append(
            {
                "example_id": row["example_id"],
                "trajectory_id": row.get("trajectory_id", ""),
                "step_index": row.get("step_index", ""),
                "failure_category": row.get("failure_category", ""),
                "reasoning_step_idx": int(row["reasoning_step_idx"]),
                "reasoning_progress": row.get("reasoning_progress", ""),
                "reasoning_character_progress": row.get(
                    "reasoning_character_progress", ""
                ),
                "analysis_unit": row.get("analysis_unit", ""),
                "analysis_unit_index": row.get("analysis_unit_index", ""),
                "analysis_unit_char_start": row.get(
                    "analysis_unit_char_start", ""
                ),
                "analysis_unit_char_end": row.get("analysis_unit_char_end", ""),
                "canonical_sentence_id": row.get("canonical_sentence_id", ""),
                "question_id": row["question_id"],
                "answer_key_greedy": row.get("answer_key", ""),
                "ground_truth_key": row.get("ground_truth_key", ""),
                "belief_is_error_greedy": row.get("belief_is_error", ""),
                "logprob_temperature": row.get(
                    "categorical_logprob_temperature", ""
                ),
                "top_logprobs": row.get("categorical_top_logprobs", ""),
                "seed": 0,
                "query_error": row.get("query_error", ""),
                "visible_answer": row.get("visible_answer", ""),
                "logprob_answer": row.get("categorical_argmax_answer", ""),
                "p_yes": probs.get("yes", 0.0),
                "p_no": probs.get("no", 0.0),
                "p_unknown": probs.get("unknown", 0.0),
                "p_invalid": probs.get("invalid", 0.0),
                "valid_label_probability_mass": row.get(
                    "categorical_candidate_probability_mass", ""
                ),
                "state_belief_entropy_bits": entropy,
                "state_belief_entropy_normalized": (
                    float(entropy) / 1.584962500721156
                    if entropy not in {None, ""}
                    else ""
                ),
                "semantic_probs_json": row.get("categorical_probs_json", ""),
                "raw_probs_json": "",
                "answer_token_logprobs_json": row.get(
                    "categorical_raw_top_logprobs_json", ""
                ),
                "raw_output": row.get("raw_belief_text", ""),
                "cost_usd": row.get("cost_usd", 0.0),
                "prompt_tokens": row.get("prompt_tokens", 0),
                "completion_tokens": row.get("completion_tokens", 0),
                "total_tokens": row.get("total_tokens", 0),
                "retry_count": row.get("retry_count", 0),
            }
        )
    path = args.run_dir / "state_belief_uncertainty_rows.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"status": "completed", "rows": len(rows), "path": str(path)}))


if __name__ == "__main__":
    main()
