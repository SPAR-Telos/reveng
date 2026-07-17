#!/usr/bin/env python3
"""Build GPT-OSS token spans for canonical DoorKey reasoning sentences."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import groupby
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from reveng.experiments.gradual_cot_blackbox_alignment import ANALYSIS_START, FINAL_START


DEFAULT_SENTENCES = Path("data/behavioral_probes/doorkey_chunking_validation/sentences.csv")
DEFAULT_OUTPUT = Path(
    "data/behavioral_probes/doorkey_chunking_validation/"
    "sentence_token_boundaries_gpt_oss_20b.csv"
)
DEFAULT_TOKENIZER = Path(
    "/root/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/"
    "6cee5e81ee83917806bbde320786a8fb61efebee"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def token_indices_for_span(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> list[int]:
    return [
        idx
        for idx, (start, end) in enumerate(offsets)
        if end > start and end > char_start and start < char_end
    ]


def build_boundaries(
    *,
    sentences_path: Path,
    output_path: Path,
    tokenizer_path: Path,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer_revision = tokenizer_path.name
    output_fields = [
        "trace_id",
        "path",
        "step_id",
        "sentence_id",
        "char_start",
        "char_end",
        "analysis_token_start",
        "analysis_token_end_exclusive",
        "output_token_start",
        "output_token_end_exclusive",
        "n_tokens",
        "tokenizer_name",
        "tokenizer_revision",
        "sentence_text_sha256",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_files: set[str] = set()
    seen_trace_keys: set[tuple[str, int]] = set()
    n_reasoning_traces = 0
    n_reasoning_sentences = 0
    traces_with_tokens_outside_sentences = 0
    n_tokens_outside_sentences = 0
    current_path = ""
    payload: dict[str, Any] = {}
    step_by_id: dict[int, dict[str, Any]] = {}
    with sentences_path.open(newline="") as input_handle, output_path.open(
        "w", newline=""
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        reasoning_rows = (row for row in reader if row.get("kind") == "reasoning")
        writer = csv.DictWriter(output_handle, fieldnames=output_fields)
        writer.writeheader()
        for (path_string, step_id_string), grouped_rows in groupby(
            reasoning_rows,
            key=lambda row: (row["path"], row["step_id"]),
        ):
            step_id = int(step_id_string)
            trace_key = (path_string, step_id)
            if trace_key in seen_trace_keys:
                raise ValueError(
                    "Sentence CSV is not grouped by path and step; "
                    f"trace repeated non-contiguously: {trace_key}"
                )
            seen_trace_keys.add(trace_key)
            n_reasoning_traces += 1
            trajectory_files.add(path_string)
            path = Path(path_string)
            if path_string != current_path:
                payload = json.loads(path.read_text())
                step_by_id = {
                    int(step.get("step_id", idx)): step
                    for idx, step in enumerate(payload.get("steps", []))
                }
                current_path = path_string
            rows = list(grouped_rows)
            step = step_by_id.get(step_id)
            if step is None:
                raise ValueError(f"Missing step {step_id} in {path}")
            output_text = str(step.get("output_text", ""))
            if ANALYSIS_START not in output_text or FINAL_START not in output_text:
                raise ValueError(f"Missing analysis channel markers in {path}, step {step_id}")
            raw_analysis = output_text.split(ANALYSIS_START, 1)[1].split(FINAL_START, 1)[0]
            stripped_analysis = raw_analysis.strip()
            leading_trim = len(raw_analysis) - len(raw_analysis.lstrip())
            encoded = tokenizer(
                raw_analysis,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            stored_analysis_tokens = [
                token
                for token in step.get("output_tokens", [])
                if "analysis" in token.get("token_groups", [])
            ]
            stored_ids = [int(token["token_id"]) for token in stored_analysis_tokens]
            if list(encoded["input_ids"]) != stored_ids:
                raise ValueError(
                    f"Tokenizer mismatch in {path}, step {step_id}: "
                    f"local={len(encoded['input_ids'])}, stored={len(stored_ids)}"
                )
            offsets = [
                (int(start) - leading_trim, int(end) - leading_trim)
                for start, end in encoded["offset_mapping"]
            ]
            ordered = sorted(rows, key=lambda row: int(row["sentence_id"]))
            previous_char_end = -1
            previous_token_end = -1
            assigned_tokens: set[int] = set()
            for row in ordered:
                char_start = int(row["char_start"])
                char_end = int(row["char_end"])
                if char_start < previous_char_end:
                    raise ValueError(f"Overlapping sentence characters in {row['trace_id']}")
                if stripped_analysis[char_start:char_end] != row["text"]:
                    raise ValueError(
                        f"Sentence text mismatch in {row['trace_id']}, sentence {row['sentence_id']}"
                    )
                token_indices = token_indices_for_span(offsets, char_start, char_end)
                if not token_indices:
                    raise ValueError(
                        f"No tokenizer token for {row['trace_id']}, sentence {row['sentence_id']}"
                    )
                duplicate = assigned_tokens.intersection(token_indices)
                if duplicate:
                    raise ValueError(
                        f"Tokens assigned to multiple sentences in {row['trace_id']}: {sorted(duplicate)}"
                    )
                analysis_start = min(token_indices)
                analysis_end = max(token_indices) + 1
                if analysis_start < previous_token_end:
                    raise ValueError(f"Overlapping sentence token spans in {row['trace_id']}")
                output_ids = [int(stored_analysis_tokens[idx]["id"]) for idx in token_indices]
                if output_ids != list(range(min(output_ids), max(output_ids) + 1)):
                    raise ValueError(f"Non-contiguous output token IDs in {row['trace_id']}")
                writer.writerow(
                    {
                        "trace_id": row["trace_id"],
                        "path": row["path"],
                        "step_id": step_id,
                        "sentence_id": int(row["sentence_id"]),
                        "char_start": char_start,
                        "char_end": char_end,
                        "analysis_token_start": analysis_start,
                        "analysis_token_end_exclusive": analysis_end,
                        "output_token_start": min(output_ids),
                        "output_token_end_exclusive": max(output_ids) + 1,
                        "n_tokens": len(token_indices),
                        "tokenizer_name": "openai/gpt-oss-20b",
                        "tokenizer_revision": tokenizer_revision,
                        "sentence_text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                    }
                )
                n_reasoning_sentences += 1
                assigned_tokens.update(token_indices)
                previous_char_end = char_end
                previous_token_end = analysis_end
            outside = {
                idx
                for idx, (start, end) in enumerate(offsets)
                if end > start and end > 0 and start < len(stripped_analysis)
            } - assigned_tokens
            if outside:
                traces_with_tokens_outside_sentences += 1
                n_tokens_outside_sentences += len(outside)
    manifest = {
        "status": "completed",
        "sentences_path": str(sentences_path),
        "output_path": str(output_path),
        "tokenizer_name": "openai/gpt-oss-20b",
        "tokenizer_revision": tokenizer_revision,
        "span_convention": "zero-based half-open",
        "n_trajectory_files": len(trajectory_files),
        "n_reasoning_traces": n_reasoning_traces,
        "n_reasoning_sentences": n_reasoning_sentences,
        "n_traces_with_tokens_outside_sentences": traces_with_tokens_outside_sentences,
        "n_tokens_outside_sentences": n_tokens_outside_sentences,
        "tokenizer_ids_match_stored_tokens": True,
        "duplicated_token_assignments": 0,
        "overlapping_token_spans": 0,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    report_path = output_path.with_suffix(".md")
    report_path.write_text(
        "\n".join(
            [
                "# GPT-OSS Sentence Token Boundaries",
                "",
                "Token positions are zero-based and end-exclusive. Analysis-token positions are relative to the reasoning channel; output-token positions include the output channel template tokens.",
                "",
                f"- Trajectory files: {manifest['n_trajectory_files']}",
                f"- Reasoning traces: {manifest['n_reasoning_traces']}",
                f"- Reasoning sentences: {manifest['n_reasoning_sentences']}",
                f"- Tokenizer revision: `{tokenizer_revision}`",
                f"- Exact tokenizer-ID matches: yes",
                f"- Duplicate token assignments: 0",
                f"- Overlapping token spans: 0",
                f"- Tokens outside reasoning sentences: {n_tokens_outside_sentences} across {traces_with_tokens_outside_sentences} traces",
            ]
        )
        + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences-path", type=Path, default=DEFAULT_SENTENCES)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    args = parser.parse_args()
    print(
        json.dumps(
            build_boundaries(
                sentences_path=args.sentences_path,
                output_path=args.output_path,
                tokenizer_path=args.tokenizer_path,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
