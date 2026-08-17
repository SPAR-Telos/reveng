#!/usr/bin/env python3
"""Run a resumable Together judge on the blinded semantic calibration set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from together import Together

from reveng.experiments.general_semantic_labeling import write_json
from reveng.experiments.semantic_reasoning_classification import ANNOTATION_COLUMNS
try:
    from scripts.run_general_semantic_labeler import (
        PROMPT_VERSION,
        SYSTEM_PROMPT,
        build_messages,
        normalize_payload,
    )
except ModuleNotFoundError:  # Direct execution puts scripts/ on sys.path.
    from run_general_semantic_labeler import (
        PROMPT_VERSION,
        SYSTEM_PROMPT,
        build_messages,
        normalize_payload,
    )


DEFAULT_ROOT = Path(
    "outputs/hypothesis_tests/semantic_reasoning_classification_v1/general_corpus_v1"
)
DEFAULT_MODEL = "Qwen/Qwen3.5-397B-A17B"

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "explicitly_revises_prior_reasoning": {
            "type": "string",
            "enum": ["yes", "no", "unsure"],
        },
        "evaluates_prior_route_or_claim": {
            "type": "string",
            "enum": ["yes", "no", "unsure"],
        },
        "repeats_prior_content": {
            "type": "string",
            "enum": ["yes", "no", "unsure"],
        },
        "introduces_new_information_or_plan": {
            "type": "string",
            "enum": ["yes", "no", "unsure"],
        },
        "primary_label": {
            "type": "string",
            "enum": [
                "correction",
                "verification",
                "state_reconstruction",
                "route_planning",
                "new_inference",
                "consolidation",
                "restatement",
                "procedural_continuation",
                "unclear",
            ],
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "rationale": {"type": "string"},
    },
    "required": list(ANNOTATION_COLUMNS),
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_ROOT / "calibration_items.csv"
    )
    parser.add_argument(
        "--few-shots", type=Path, default=DEFAULT_ROOT / "few_shot_examples.json"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_ROOT / "annotations_qwen_3_5_397b.csv"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def usage_value(usage: Any, name: str) -> int:
    value = getattr(usage, name, 0) if usage is not None else 0
    return int(value or 0)


def main() -> None:
    args = parse_args()
    load_dotenv(".env")
    api_key = os.getenv("TOGETHERAI_API_KEY") or os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHERAI_API_KEY or TOGETHER_API_KEY is required")

    source = pd.read_csv(args.input).fillna("")
    if args.limit is not None:
        source = source.head(args.limit)
    few_shots = json.loads(args.few_shots.read_text())
    config = {
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "few_shots": str(args.few_shots),
        "few_shots_sha256": sha256(args.few_shots),
        "model": args.model,
        "reasoning": args.reasoning,
        "stream": args.stream,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "limit": args.limit,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "output_schema_sha256": hashlib.sha256(
            json.dumps(OUTPUT_SCHEMA, sort_keys=True).encode()
        ).hexdigest(),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != config:
        raise ValueError("existing output manifest uses a different configuration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, config)

    completed: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        previous = pd.read_csv(args.output).fillna("")
        completed = {
            str(row["annotation_id"]): row
            for row in previous.to_dict("records")
            if str(row.get("primary_label", "")).strip()
        }
    output = list(completed.values())
    status_path = args.output.with_suffix(".status.json")
    status: dict[str, Any] = {
        "status": "running",
        "model_name": args.model,
        "input": str(args.input),
        "output": str(args.output),
        "expected_rows": len(source),
        "labeled_rows": len(completed),
    }
    write_json(status_path, status)

    client = Together(api_key=api_key)
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "semantic_annotation", "schema": OUTPUT_SCHEMA},
    }
    for index, row in enumerate(source.itertuples(index=False), start=1):
        annotation_id = str(row.annotation_id)
        if annotation_id in completed:
            continue
        raw_response = ""
        error = ""
        record: dict[str, Any] | None = None
        for attempt in range(1, args.max_attempts + 1):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=build_messages(row, few_shots),
                    temperature=args.temperature,
                    seed=args.seed,
                    max_tokens=args.max_tokens,
                    reasoning={"enabled": args.reasoning},
                    response_format=response_format,
                    stream=args.stream,
                )
                usage = None
                if args.stream:
                    parts: list[str] = []
                    for chunk in response:
                        usage = getattr(chunk, "usage", None) or usage
                        if not chunk.choices:
                            continue
                        content = getattr(chunk.choices[0].delta, "content", None)
                        if content:
                            parts.append(content)
                    raw_response = "".join(parts)
                else:
                    raw_response = response.choices[0].message.content or ""
                    usage = getattr(response, "usage", None)
                parsed = normalize_payload(json.loads(raw_response))
                record = {
                    "annotation_id": annotation_id,
                    "context_before": row.context_before,
                    "target_sentence": row.target_sentence,
                    "target_sentence_characters": row.target_sentence_characters,
                    **parsed,
                    "judge_model": args.model,
                    "judge_provider": "together_ai",
                    "judge_prompt_version": PROMPT_VERSION,
                    "judge_prompt_sha256": config["prompt_sha256"],
                    "judge_temperature": args.temperature,
                    "judge_seed": args.seed,
                    "reasoning_enabled": args.reasoning,
                    "prompt_tokens": usage_value(usage, "prompt_tokens"),
                    "completion_tokens": usage_value(usage, "completion_tokens"),
                    "total_tokens": usage_value(usage, "total_tokens"),
                    "raw_response": raw_response,
                    "attempts": attempt,
                    "error": "",
                }
                break
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                if attempt < args.max_attempts:
                    time.sleep(min(2**attempt, 8))
        if record is None:
            record = {
                "annotation_id": annotation_id,
                "context_before": row.context_before,
                "target_sentence": row.target_sentence,
                "target_sentence_characters": row.target_sentence_characters,
                **{column: "" for column in ANNOTATION_COLUMNS},
                "judge_model": args.model,
                "judge_provider": "together_ai",
                "judge_prompt_version": PROMPT_VERSION,
                "judge_prompt_sha256": config["prompt_sha256"],
                "judge_temperature": args.temperature,
                "judge_seed": args.seed,
                "reasoning_enabled": args.reasoning,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "raw_response": raw_response,
                "attempts": args.max_attempts,
                "error": error,
            }
        output.append(record)
        if record["primary_label"]:
            completed[annotation_id] = record
        pd.DataFrame(output).drop_duplicates("annotation_id", keep="last").to_csv(
            args.output, index=False
        )
        status["labeled_rows"] = len(completed)
        write_json(status_path, status)
        if index % 10 == 0 or error:
            print(
                f"processed={index}/{len(source)} completed={len(completed)} "
                f"error={error or 'none'}",
                flush=True,
            )
    status["status"] = "complete" if len(completed) == len(source) else "incomplete"
    status["labeled_rows"] = len(completed)
    write_json(status_path, status)
    print(f"output={args.output} labeled={len(completed)}/{len(source)} model={args.model}")


if __name__ == "__main__":
    main()
