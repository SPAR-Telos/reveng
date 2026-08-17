#!/usr/bin/env python3
"""Run a resumable local judge on blinded general semantic items."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from reveng.experiments.general_semantic_labeling import write_json
from reveng.experiments.semantic_reasoning_classification import (
    ANNOTATION_COLUMNS,
    BOOLEAN_LABELS,
    CONFIDENCE_LABELS,
    SEMANTIC_LABELS,
)


DEFAULT_ROOT = Path(
    "outputs/hypothesis_tests/semantic_reasoning_classification_v1/general_corpus_v1"
)
DEFAULT_LLAMA = Path(
    "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/"
    "snapshots/9213176726f574b556790deb65791e0c5aa438b6"
)

SYSTEM_PROMPT = """Classify the discourse function of one target sentence using
only the preceding reasoning and the target. Do not judge factual correctness,
action optimality, task success, change points, activations, or later text.

Apply this precedence:
1. correction: explicitly replaces, rejects, or reverses an earlier claim.
2. verification: evaluates an earlier fact, move, or route.
3. state_reconstruction: directly reads or records visible state information.
4. route_planning: proposes a move, route, subgoal, or action sequence.
5. new_inference: derives a new consequence or constraint not covered above.
6. consolidation: combines several earlier findings into a summary or decision.
7. restatement: repeats one earlier fact or conclusion without evaluating it.
8. procedural_continuation: enumeration, bookkeeping, or connective narration.
9. unclear: context is insufficient or two labels remain equally plausible.

Return exactly one JSON object. The four cue fields must be yes, no, or unsure.
`primary_label` must be one label above. `confidence` must be high, medium, or
low. Keep `rationale` under 24 words.

Required fields: explicitly_revises_prior_reasoning,
evaluates_prior_route_or_claim, repeats_prior_content,
introduces_new_information_or_plan, primary_label, confidence, rationale."""
PROMPT_VERSION = "general_fewshot_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_ROOT / "calibration_items.csv"
    )
    parser.add_argument(
        "--few-shots", type=Path, default=DEFAULT_ROOT / "few_shot_examples.json"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_ROOT / "annotations_llama_3_2_1b.csv"
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_LLAMA)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--minimum-free-gib", type=float, default=3.0)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_input(row: Any) -> str:
    context = str(row.context_before).strip() or "[No preceding reasoning.]"
    return f"Preceding reasoning:\n{context}\n\nTarget sentence:\n{row.target_sentence}"


def annotation_payload(example: dict[str, Any]) -> dict[str, str]:
    return {column: str(example[column]) for column in ANNOTATION_COLUMNS}


def build_messages(
    row: Any, few_shots: list[dict[str, Any]]
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in few_shots:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": model_input(type("Example", (), example)()),
                },
                {
                    "role": "assistant",
                    "content": json.dumps(annotation_payload(example), ensure_ascii=False),
                },
            ]
        )
    messages.append({"role": "user", "content": model_input(row)})
    return messages


def build_gpt_oss_prompt(row: Any, few_shots: list[dict[str, Any]]) -> str:
    demonstrations = []
    for index, example in enumerate(few_shots, start=1):
        demonstrations.append(
            f"Example {index}\nINPUT\n{model_input(type('Example', (), example)())}\n"
            f"OUTPUT\n{json.dumps(annotation_payload(example), ensure_ascii=False)}"
        )
    return (
        SYSTEM_PROMPT
        + "\n\nLabelled examples:\n\n"
        + "\n\n".join(demonstrations)
        + "\n\nNow classify this item.\n"
        + model_input(row)
    )


def extract_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    decoder = json.JSONDecoder()
    for position, character in enumerate(value):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("response does not contain a JSON object")


def normalize_payload(payload: dict[str, Any]) -> dict[str, str]:
    result = {column: str(payload.get(column, "")).strip() for column in ANNOTATION_COLUMNS}
    for column in ANNOTATION_COLUMNS[:4]:
        result[column] = result[column].lower()
        if result[column] not in BOOLEAN_LABELS:
            raise ValueError(f"invalid {column}: {result[column]!r}")
    result["primary_label"] = result["primary_label"].lower()
    if result["primary_label"] not in SEMANTIC_LABELS:
        raise ValueError(f"invalid primary_label: {result['primary_label']!r}")
    result["confidence"] = result["confidence"].lower()
    if result["confidence"] not in CONFIDENCE_LABELS:
        raise ValueError(f"invalid confidence: {result['confidence']!r}")
    if not result["rationale"]:
        raise ValueError("missing rationale")
    return result


def gpu_status(gpu_index: int, minimum_free_gib: float) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {
            "status": "blocked_gpu_unavailable",
            "gpu_index": gpu_index,
            "minimum_free_gib": minimum_free_gib,
            "message": "CUDA is unavailable; no CPU fallback is permitted.",
        }
    try:
        with torch.cuda.device(gpu_index):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            name = torch.cuda.get_device_name(gpu_index)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "blocked_gpu_unavailable",
            "gpu_index": gpu_index,
            "minimum_free_gib": minimum_free_gib,
            "message": f"{type(exc).__name__}: {exc}",
        }
    free_gib = free_bytes / 2**30
    status = "ready" if free_gib >= minimum_free_gib else "blocked_gpu_busy"
    return {
        "status": status,
        "gpu_index": gpu_index,
        "gpu_name": name,
        "free_gib": free_gib,
        "total_gib": total_bytes / 2**30,
        "minimum_free_gib": minimum_free_gib,
    }


def main() -> None:
    args = parse_args()
    status_path = args.output.with_suffix(".status.json")
    status = gpu_status(args.gpu_index, args.minimum_free_gib)
    status.update(
        {
            "model_name": args.model_name,
            "model_path": str(args.model_path),
            "input": str(args.input),
            "output": str(args.output),
        }
    )
    write_json(status_path, status)
    if status["status"] != "ready":
        print(f"{status['status']}: {status.get('message', status)}")
        raise SystemExit(2)
    status["status"] = "running"
    write_json(status_path, status)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = pd.read_csv(args.input).fillna("")
    if args.limit is not None:
        source = source.head(args.limit)
    few_shots = json.loads(args.few_shots.read_text())
    config = {
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "few_shots": str(args.few_shots),
        "few_shots_sha256": sha256(args.few_shots),
        "model_name": args.model_name,
        "model_path": str(args.model_path),
        "model_revision": args.model_path.name,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != config:
        raise ValueError("existing output manifest uses a different configuration")
    write_json(manifest_path, config)

    completed: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        previous = pd.read_csv(args.output).fillna("")
        completed = {
            str(row["annotation_id"]): row
            for row in previous.to_dict("records")
            if str(row.get("primary_label", "")) in SEMANTIC_LABELS
        }
    output = list(completed.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": args.gpu_index},
        low_cpu_mem_usage=True,
    ).eval()
    device = next(model.parameters()).device
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    is_gpt_oss = "gpt-oss" in args.model_name.lower()

    for index, row in enumerate(source.itertuples(index=False), start=1):
        annotation_id = str(row.annotation_id)
        if annotation_id in completed:
            continue
        if is_gpt_oss:
            rendered = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": build_gpt_oss_prompt(row, few_shots),
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="medium",
            )
            rendered += "<|channel|>final<|message|>"
        else:
            rendered = tokenizer.apply_chat_template(
                build_messages(row, few_shots),
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(
            device
        )
        raw_response = ""
        error = ""
        parsed: dict[str, str] = {}
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )[0]
            raw_response = tokenizer.decode(
                generated[inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            parsed = normalize_payload(extract_json_object(raw_response))
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        record = {
            "annotation_id": annotation_id,
            "context_before": row.context_before,
            "target_sentence": row.target_sentence,
            "target_sentence_characters": row.target_sentence_characters,
            **{column: parsed.get(column, "") for column in ANNOTATION_COLUMNS},
            "judge_model": args.model_name,
            "judge_revision": args.model_path.name,
            "judge_prompt_version": PROMPT_VERSION,
            "judge_prompt_sha256": config["prompt_sha256"],
            "judge_temperature": 0.0,
            "judge_seed": args.seed,
            "raw_response": raw_response,
            "error": error,
        }
        output.append(record)
        if parsed:
            completed[annotation_id] = record
        pd.DataFrame(output).drop_duplicates("annotation_id", keep="last").to_csv(
            args.output, index=False
        )
        if index % 10 == 0 or error:
            print(
                f"processed={index}/{len(source)} completed={len(completed)} "
                f"error={error or 'none'}",
                flush=True,
            )
    status["status"] = "complete"
    status["labeled_rows"] = len(completed)
    status["expected_rows"] = len(source)
    write_json(status_path, status)
    print(f"output={args.output} labeled={len(completed)}/{len(source)}")


if __name__ == "__main__":
    main()
