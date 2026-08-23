#!/usr/bin/env python3
"""Run a resumable multi-label Together judge on a sentence inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from together import Together

from reveng.experiments.general_semantic_labeling import stable_hash, write_json


DEFAULT_ROOT = Path(
    "outputs/hypothesis_tests/semantic_reasoning_classification_v1/general_corpus_v1"
)
DEFAULT_MODEL = "openai/gpt-oss-20b"
PROMPT_VERSION = "general_multilabel_v3_human_calibrated"
SEMANTIC_LABELS = (
    "state_readout",
    "route_planning",
    "verification",
    "correction",
    "new_inference",
    "consolidation",
    "restatement",
    "procedural_continuation",
    "action_commitment",
)
PRIMARY_PRECEDENCE = (
    "correction",
    "verification",
    "action_commitment",
    "state_readout",
    "route_planning",
    "new_inference",
    "consolidation",
    "restatement",
    "procedural_continuation",
)
BOOLEAN_FIELDS = (
    "explicitly_revises_prior_reasoning",
    "evaluates_prior_route_or_claim",
    "repeats_prior_content",
    "introduces_new_information_or_plan",
)
MODEL_OUTPUT_FIELDS = (
    "semantic_labels",
    "annotation_status",
    *BOOLEAN_FIELDS,
    "confidence",
    "rationale",
)
BROAD_LABEL_MAP = {
    "state_readout": "state_readout",
    "route_planning": "route_deliberation",
    "verification": "route_deliberation",
    "correction": "route_deliberation",
    "new_inference": "route_deliberation",
    "consolidation": "summary_or_restatement",
    "restatement": "summary_or_restatement",
    "procedural_continuation": "other",
    "action_commitment": "route_deliberation",
}
SOURCE_METADATA_COLUMNS = (
    "sentence_id",
    "environment_id",
    "environment_step",
    "sentence_number",
    "example_id",
    "reasoning_progress",
    "duplicate_of_annotation_id",
)

SYSTEM_PROMPT = """Assign all semantic discourse functions performed by one
target sentence, using only the preceding reasoning and the target sentence.
Do not judge factual correctness, action optimality, task success, change
points, activations, or later text.

Labels are non-exclusive:
- state_readout: states environment or agent state, including an object/cell
  observation or the coordinate reached by a definite numbered route step.
- route_planning: explores or proposes a possible move, route, subgoal, or
  action sequence that has not yet been selected.
- verification: checks or evaluates an earlier fact, move, or route.
- correction: rejects, revises, or reverses earlier reasoning.
- new_inference: derives a new consequence, constraint, or conclusion.
- consolidation: combines several earlier findings into a summary or decision.
- restatement: repeats an earlier substantive proposition without changing it.
- procedural_continuation: setup, formatting, bookkeeping, or connective
  narration with no substantive state, plan, evaluation, or conclusion.
- action_commitment: records an already selected concrete move or explicitly
  chooses/recommends the next concrete move.

Select every applicable label, normally one to three and never more than four.
Use procedural_continuation only when no substantive label applies. Repeated
format or enumeration alone is procedural_continuation; restatement requires a
repeated proposition. A sentence may both repeat state information and verify
it, or correct a route and propose a replacement.

Do not add route_planning to a definite numbered route step merely because it
contains movement. Use action_commitment for a selected concrete step and
state_readout as well when the sentence records its resulting coordinate. Use
route_planning for alternatives, tentative routes, or construction of an
unselected sequence. A coordinate in a hypothetical proposal is route_planning,
not state_readout. Do not add restatement simply because the grid or topic was
mentioned earlier: the same substantive proposition must actually be repeated.

annotation_status is complete unless context is genuinely insufficient or the
function remains ambiguous; then use unclear while still selecting the most
plausible labels. The four cue fields must be yes, no, or unsure. confidence is
high, medium, or low. Keep rationale under 30 words. Return exactly one JSON
object matching the requested schema."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "semantic_labels": {
            "type": "array",
            "items": {"type": "string", "enum": list(SEMANTIC_LABELS)},
            "minItems": 1,
            "maxItems": 4,
            "uniqueItems": True,
        },
        "annotation_status": {
            "type": "string",
            "enum": ["complete", "unclear"],
        },
        **{
            field: {"type": "string", "enum": ["yes", "no", "unsure"]}
            for field in BOOLEAN_FIELDS
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "rationale": {"type": "string"},
    },
    "required": list(MODEL_OUTPUT_FIELDS),
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_ROOT / "sentence_inventory.csv"
    )
    parser.add_argument(
        "--few-shots",
        type=Path,
        default=DEFAULT_ROOT / "few_shot_examples_multilabel.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT / "annotations_gpt_oss_20b_multilabel.csv",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="low",
    )
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--sample-strategy", choices=("head", "stratified"), default="stratified"
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--duplicate-items", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def usage_value(usage: Any, name: str) -> int:
    value = getattr(usage, name, 0) if usage is not None else 0
    return int(value or 0)


def prepare_source(source: pd.DataFrame) -> pd.DataFrame:
    required = {"context_before", "target_sentence", "target_sentence_characters"}
    if missing := required - set(source.columns):
        raise ValueError(f"input is missing required columns: {sorted(missing)}")
    result = source.copy()
    if "annotation_id" not in result:
        if "sentence_id" not in result:
            raise ValueError("input must contain annotation_id or sentence_id")
        result["annotation_id"] = result["sentence_id"].astype(str)
    result["annotation_id"] = result["annotation_id"].astype(str)
    if result["annotation_id"].duplicated().any():
        raise ValueError("input contains duplicate annotation IDs")
    if "duplicate_of_annotation_id" not in result:
        result["duplicate_of_annotation_id"] = ""
    return result


def select_source(
    source: pd.DataFrame,
    *,
    limit: int | None,
    strategy: str,
    seed: int,
    duplicate_items: int,
) -> pd.DataFrame:
    """Select a deterministic stratified sample and add hidden duplicates."""
    if limit is None:
        if duplicate_items:
            raise ValueError("--duplicate-items requires --limit")
        return source.copy()
    if limit < 1 or limit > len(source):
        raise ValueError("--limit must be between 1 and the input row count")
    if duplicate_items < 0 or duplicate_items > limit:
        raise ValueError("--duplicate-items must be between 0 and --limit")
    if strategy == "head":
        selected = source.head(limit).copy()
    else:
        required = {"environment_id", "reasoning_progress"}
        if missing := required - set(source.columns):
            raise ValueError(f"stratified sampling requires: {sorted(missing)}")
        candidates = source.copy()
        candidates["_progress"] = pd.cut(
            candidates["reasoning_progress"].astype(float),
            [-float("inf"), 0.25, 0.5, 0.75, float("inf")],
            labels=("early", "early_middle", "late_middle", "late"),
        ).astype(str)
        length_rank = candidates["target_sentence_characters"].rank(
            method="first", pct=True
        )
        candidates["_length"] = pd.cut(
            length_rank,
            [-float("inf"), 1 / 3, 2 / 3, float("inf")],
            labels=("short", "medium", "long"),
        ).astype(str)
        candidates["_stratum"] = candidates["_progress"] + "__" + candidates["_length"]
        candidates["_rank"] = candidates["annotation_id"].map(
            lambda value: stable_hash(str(value), seed, length=32)
        )
        chosen: list[int] = []
        chosen_set: set[int] = set()

        def choose_first(groups: Any) -> None:
            for _, group in groups:
                index = int(group.sort_values("_rank").index[0])
                if index not in chosen_set and len(chosen) < limit:
                    chosen.append(index)
                    chosen_set.add(index)

        choose_first(candidates.groupby("environment_id", sort=True))
        choose_first(candidates.groupby("_stratum", sort=True))
        for index in candidates.sort_values("_rank").index:
            index = int(index)
            if index not in chosen_set and len(chosen) < limit:
                chosen.append(index)
                chosen_set.add(index)
        selected = candidates.loc[chosen].sort_values("_rank").copy()
        selected = selected.drop(
            columns=["_progress", "_length", "_stratum", "_rank"]
        )
    if duplicate_items:
        duplicates = selected.head(duplicate_items).copy()
        duplicates["duplicate_of_annotation_id"] = duplicates["annotation_id"]
        duplicates["annotation_id"] = [
            f"{value}__duplicate_{index:03d}"
            for index, value in enumerate(duplicates["annotation_id"], start=1)
        ]
        selected = pd.concat([selected, duplicates], ignore_index=True)
    return selected.reset_index(drop=True)


def derive_primary_label(labels: list[str]) -> str:
    label_set = set(labels)
    return next(label for label in PRIMARY_PRECEDENCE if label in label_set)


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    labels_value = payload.get("semantic_labels")
    if not isinstance(labels_value, list):
        raise ValueError("semantic_labels must be an array")
    labels = list(
        dict.fromkeys(str(value).strip().lower() for value in labels_value)
    )
    if not 1 <= len(labels) <= 4:
        raise ValueError("semantic_labels must contain one to four unique labels")
    invalid = set(labels) - set(SEMANTIC_LABELS)
    if invalid:
        raise ValueError(f"invalid semantic labels: {sorted(invalid)}")
    if "procedural_continuation" in labels and len(labels) > 1:
        raise ValueError("procedural_continuation cannot accompany substantive labels")
    labels = [label for label in PRIMARY_PRECEDENCE if label in labels]
    status = str(payload.get("annotation_status", "")).strip().lower()
    if status not in {"complete", "unclear"}:
        raise ValueError(f"invalid annotation_status: {status!r}")
    result: dict[str, Any] = {
        "semantic_labels": labels,
        "annotation_status": status,
    }
    for field in BOOLEAN_FIELDS:
        value = str(payload.get(field, "")).strip().lower()
        if value not in {"yes", "no", "unsure"}:
            raise ValueError(f"invalid {field}: {value!r}")
        result[field] = value
    confidence = str(payload.get("confidence", "")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        raise ValueError(f"invalid confidence: {confidence!r}")
    rationale = str(payload.get("rationale", "")).strip()
    if not rationale:
        raise ValueError("missing rationale")
    result["confidence"] = confidence
    result["rationale"] = rationale
    return result


def model_input(row: Any) -> str:
    context = str(row.context_before).strip() or "[No preceding reasoning.]"
    return f"Preceding reasoning:\n{context}\n\nTarget sentence:\n{row.target_sentence}"


def few_shot_payload(example: dict[str, Any]) -> dict[str, Any]:
    return {field: example[field] for field in MODEL_OUTPUT_FIELDS}


def build_messages(
    row: Any, few_shots: list[dict[str, Any]]
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in few_shots:
        normalize_payload(few_shot_payload(example))
        messages.extend(
            [
                {
                    "role": "user",
                    "content": model_input(type("Example", (), example)()),
                },
                {
                    "role": "assistant",
                    "content": json.dumps(few_shot_payload(example), ensure_ascii=False),
                },
            ]
        )
    messages.append({"role": "user", "content": model_input(row)})
    return messages


def request_reasoning_parameters(
    model: str, *, reasoning: bool, reasoning_effort: str
) -> dict[str, Any]:
    if model.startswith("openai/gpt-oss-"):
        return {"reasoning_effort": reasoning_effort}
    return {"reasoning": {"enabled": reasoning}}


def source_metadata(row: Any) -> dict[str, Any]:
    return {
        column: getattr(row, column)
        for column in SOURCE_METADATA_COLUMNS
        if hasattr(row, column)
    }


def is_completed_record(record: dict[str, Any]) -> bool:
    if str(record.get("error", "")).strip():
        return False
    try:
        labels = json.loads(str(record.get("semantic_labels", "")))
    except (TypeError, json.JSONDecodeError):
        return False
    return bool(labels and str(record.get("primary_label", "")).strip())


def main() -> None:
    args = parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be at least 1")
    load_dotenv(".env")
    api_key = os.getenv("TOGETHERAI_API_KEY") or os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHERAI_API_KEY or TOGETHER_API_KEY is required")

    source = prepare_source(pd.read_csv(args.input).fillna(""))
    source = select_source(
        source,
        limit=args.limit,
        strategy=args.sample_strategy,
        seed=args.sample_seed,
        duplicate_items=args.duplicate_items,
    )
    few_shots = json.loads(args.few_shots.read_text())
    for example in few_shots:
        normalize_payload(few_shot_payload(example))
    config = {
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "few_shots": str(args.few_shots),
        "few_shots_sha256": sha256(args.few_shots),
        "model": args.model,
        "reasoning": args.reasoning,
        "reasoning_effort": args.reasoning_effort,
        "stream": args.stream,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "max_attempts": args.max_attempts,
        "limit": args.limit,
        "sample_strategy": args.sample_strategy,
        "sample_seed": args.sample_seed,
        "duplicate_items": args.duplicate_items,
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
            if is_completed_record(row)
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
        "failed_rows": 0,
        "max_workers": args.max_workers,
        "checkpoint_every": args.checkpoint_every,
    }
    write_json(status_path, status)

    clients = threading.local()

    def get_client() -> Together:
        client = getattr(clients, "client", None)
        if client is None:
            client = Together(api_key=api_key)
            clients.client = client
        return client

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "semantic_multilabel_annotation",
            "schema": OUTPUT_SCHEMA,
        },
    }

    def label_row(row: Any) -> dict[str, Any]:
        annotation_id = str(row.annotation_id)
        raw_response = ""
        error = ""
        for attempt in range(1, args.max_attempts + 1):
            try:
                response = get_client().chat.completions.create(
                    model=args.model,
                    messages=build_messages(row, few_shots),
                    temperature=args.temperature,
                    seed=args.seed,
                    max_tokens=args.max_tokens,
                    **request_reasoning_parameters(
                        args.model,
                        reasoning=args.reasoning,
                        reasoning_effort=args.reasoning_effort,
                    ),
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
                primary = derive_primary_label(parsed["semantic_labels"])
                return {
                    "annotation_id": annotation_id,
                    **source_metadata(row),
                    "context_before": row.context_before,
                    "target_sentence": row.target_sentence,
                    "target_sentence_characters": row.target_sentence_characters,
                    "semantic_labels": json.dumps(parsed.pop("semantic_labels")),
                    "primary_label": primary,
                    **parsed,
                    "broad_label": BROAD_LABEL_MAP[primary],
                    "judge_model": args.model,
                    "judge_provider": "together_ai",
                    "judge_prompt_version": PROMPT_VERSION,
                    "judge_prompt_sha256": config["prompt_sha256"],
                    "judge_temperature": args.temperature,
                    "judge_seed": args.seed,
                    "reasoning_enabled": args.reasoning,
                    "judge_reasoning_effort": (
                        args.reasoning_effort
                        if args.model.startswith("openai/gpt-oss-")
                        else ""
                    ),
                    "prompt_tokens": usage_value(usage, "prompt_tokens"),
                    "completion_tokens": usage_value(usage, "completion_tokens"),
                    "total_tokens": usage_value(usage, "total_tokens"),
                    "raw_response": raw_response,
                    "attempts": attempt,
                    "error": "",
                }
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                if attempt < args.max_attempts:
                    time.sleep(min(2**attempt, 8))
        return {
            "annotation_id": annotation_id,
            **source_metadata(row),
            "context_before": row.context_before,
            "target_sentence": row.target_sentence,
            "target_sentence_characters": row.target_sentence_characters,
            "semantic_labels": "",
            "primary_label": "",
            "annotation_status": "",
            **{field: "" for field in BOOLEAN_FIELDS},
            "confidence": "",
            "rationale": "",
            "broad_label": "",
            "judge_model": args.model,
            "judge_provider": "together_ai",
            "judge_prompt_version": PROMPT_VERSION,
            "judge_prompt_sha256": config["prompt_sha256"],
            "judge_temperature": args.temperature,
            "judge_seed": args.seed,
            "reasoning_enabled": args.reasoning,
            "judge_reasoning_effort": (
                args.reasoning_effort
                if args.model.startswith("openai/gpt-oss-")
                else ""
            ),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "raw_response": raw_response,
            "attempts": args.max_attempts,
            "error": error,
        }

    source_rows = list(source.itertuples(index=False))
    pending = [row for row in source_rows if str(row.annotation_id) not in completed]

    def checkpoint() -> None:
        latest = {str(record["annotation_id"]): record for record in output}
        ordered = [
            latest[str(row.annotation_id)]
            for row in source_rows
            if str(row.annotation_id) in latest
        ]
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        pd.DataFrame(ordered).to_csv(temporary, index=False)
        temporary.replace(args.output)
        status["labeled_rows"] = len(completed)
        status["failed_rows"] = sum(
            bool(str(record.get("error", "")).strip()) for record in latest.values()
        )
        write_json(status_path, status)

    initial_pending = len(pending)
    processed = 0
    if pending:
        preflight_record = label_row(pending.pop(0))
        output.append(preflight_record)
        preflight_id = str(preflight_record["annotation_id"])
        if is_completed_record(preflight_record):
            completed[preflight_id] = preflight_record
        processed = 1
        checkpoint()
        permanent_markers = (
            "model_not_available",
            "Unable to access non-serverless model",
        )
        if any(marker in preflight_record["error"] for marker in permanent_markers):
            status["status"] = "incomplete"
            status["error"] = preflight_record["error"]
            write_json(status_path, status)
            raise RuntimeError(preflight_record["error"])

    if args.max_workers == 1:
        records = map(label_row, pending)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=args.max_workers)
        futures = [executor.submit(label_row, row) for row in pending]
        records = (future.result() for future in as_completed(futures))
    try:
        for record in records:
            processed += 1
            output.append(record)
            annotation_id = str(record["annotation_id"])
            if is_completed_record(record):
                completed[annotation_id] = record
            if processed % args.checkpoint_every == 0 or record["error"]:
                checkpoint()
            if processed % 10 == 0 or record["error"]:
                print(
                    f"processed={processed}/{initial_pending} completed={len(completed)} "
                    f"error={record['error'] or 'none'}",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    checkpoint()
    status["status"] = "complete" if len(completed) == len(source) else "incomplete"
    status["labeled_rows"] = len(completed)
    write_json(status_path, status)
    print(
        f"output={args.output} labeled={len(completed)}/{len(source)} "
        f"model={args.model} workers={args.max_workers}"
    )


if __name__ == "__main__":
    main()
