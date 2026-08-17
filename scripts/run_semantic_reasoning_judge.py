#!/usr/bin/env python3
"""Label blinded reasoning sentences with a reproducible external AI judge."""

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

from reveng.experiments.semantic_reasoning_classification import (
    ANNOTATION_COLUMNS,
    BOOLEAN_LABELS,
    CONFIDENCE_LABELS,
    SEMANTIC_LABELS,
    validate_annotation_rows,
)


DEFAULT_INPUT = Path(
    "outputs/hypothesis_tests/semantic_reasoning_classification_v1/"
    "full_annotation_template.csv"
)
DEFAULT_OUTPUT = Path(
    "outputs/hypothesis_tests/semantic_reasoning_classification_v1/"
    "judge_annotations.csv"
)

SYSTEM_PROMPT = """You label the discourse function of one sentence in a
reasoning trace. Use only the preceding reasoning and target sentence. Do not
judge factual correctness, action optimality, task success, or whether an
action distribution changed.

Choose exactly one primary label using this precedence:
1. correction: explicitly rejects, revises, or reverses an earlier claim,
route, or conclusion.
2. verification: checks whether an earlier fact or proposed route is valid,
traversable, or efficient without replacing it.
3. state_reconstruction: directly reads, records, or locates an environment
feature from the grid.
4. route_planning: proposes a new action, route, subgoal, or action sequence.
5. new_inference: derives a new consequence, constraint, comparison, or
conclusion that is not primarily a route proposal.
6. consolidation: combines multiple earlier facts or conclusions into a
summary, plan, or decision without adding a materially new premise.
7. restatement: repeats one earlier fact, route, or conclusion without
materially checking, combining, or changing it.
8. procedural_continuation: continues enumeration, arithmetic, bookkeeping,
or connective narration without fitting a more specific category.
9. unclear: context is insufficient or two labels remain equally plausible.

Also mark four observable cues as yes, no, or unsure:
- explicitly_revises_prior_reasoning
- evaluates_prior_route_or_claim
- repeats_prior_content
- introduces_new_information_or_plan

Return one JSON object only, with those four cue fields plus primary_label,
confidence (high, medium, or low), and a one-sentence rationale."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=4)
    return parser.parse_args()


def prompt_hash() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def normalize_payload(payload: dict[str, Any]) -> dict[str, str]:
    result = {column: str(payload.get(column, "")).strip() for column in ANNOTATION_COLUMNS}
    for column in ANNOTATION_COLUMNS[:4]:
        result[column] = result[column].lower()
        if result[column] not in BOOLEAN_LABELS:
            raise ValueError(f"Invalid {column}: {result[column]!r}")
    result["primary_label"] = result["primary_label"].lower()
    if result["primary_label"] not in SEMANTIC_LABELS:
        raise ValueError(f"Invalid primary_label: {result['primary_label']!r}")
    result["confidence"] = result["confidence"].lower()
    if result["confidence"] not in CONFIDENCE_LABELS:
        raise ValueError(f"Invalid confidence: {result['confidence']!r}")
    if not result["rationale"]:
        raise ValueError("Missing rationale")
    return result


def parse_json_response(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(value)


def user_prompt(row: Any) -> str:
    context = str(row.context_before).strip()
    if not context:
        context = "[No preceding reasoning sentence.]"
    return (
        "Preceding reasoning:\n"
        f"{context}\n\n"
        "Target sentence:\n"
        f"{row.target_sentence}\n"
    )


def main() -> None:
    args = parse_args()
    load_dotenv(".env")
    api_key = os.getenv("TOGETHERAI_API_KEY") or os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHERAI_API_KEY or TOGETHER_API_KEY is required")
    client = Together(api_key=api_key)

    source = pd.read_csv(args.input).fillna("")
    if args.limit is not None:
        source = source.head(args.limit)
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        previous = pd.read_csv(args.output).fillna("")
        completed = {
            str(row["annotation_id"]): row
            for row in previous.to_dict("records")
            if str(row.get("primary_label", "")).strip()
        }
    output = list(completed.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(source.itertuples(index=False), start=1):
        annotation_id = str(row.annotation_id)
        if annotation_id in completed:
            continue
        error = ""
        for attempt in range(1, args.max_attempts + 1):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt(row)},
                    ],
                    temperature=args.temperature,
                    seed=args.seed,
                    max_tokens=args.max_tokens,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content
                parsed = normalize_payload(parse_json_response(raw))
                record = {
                    "annotation_id": annotation_id,
                    "context_before": row.context_before,
                    "target_sentence": row.target_sentence,
                    "target_sentence_characters": row.target_sentence_characters,
                    **parsed,
                    "judge_model": args.model,
                    "judge_temperature": args.temperature,
                    "judge_seed": args.seed,
                    "judge_prompt_sha256": prompt_hash(),
                    "raw_response": raw,
                    "attempts": attempt,
                    "error": "",
                }
                output.append(record)
                completed[annotation_id] = record
                break
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                if attempt < args.max_attempts:
                    time.sleep(min(2**attempt, 8))
        else:
            output.append(
                {
                    "annotation_id": annotation_id,
                    "context_before": row.context_before,
                    "target_sentence": row.target_sentence,
                    "target_sentence_characters": row.target_sentence_characters,
                    **{column: "" for column in ANNOTATION_COLUMNS},
                    "judge_model": args.model,
                    "judge_temperature": args.temperature,
                    "judge_seed": args.seed,
                    "judge_prompt_sha256": prompt_hash(),
                    "raw_response": "",
                    "attempts": args.max_attempts,
                    "error": error,
                }
            )
        pd.DataFrame(output).drop_duplicates(
            "annotation_id", keep="last"
        ).to_csv(args.output, index=False)
        if index % 20 == 0:
            print(f"processed={index}/{len(source)} completed={len(completed)}")

    final = pd.read_csv(args.output).fillna("")
    labeled = final[final["primary_label"].astype(str).str.strip().ne("")]
    validate_annotation_rows(labeled, allow_partial=False)
    print(
        f"output={args.output} labeled={len(labeled)}/{len(source)} "
        f"model={args.model}"
    )


if __name__ == "__main__":
    main()
