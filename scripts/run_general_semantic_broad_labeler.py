#!/usr/bin/env python3
"""Run the balanced four-way semantic candidate on blinded sentences."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from reveng.experiments.general_semantic_labeling import (
    BROAD_LABEL_MAP,
    BROAD_LABELS,
    write_json,
)
from reveng.experiments.semantic_reasoning_classification import CONFIDENCE_LABELS
try:
    from scripts.run_general_semantic_labeler import (
        DEFAULT_LLAMA,
        DEFAULT_ROOT,
        extract_json_object,
        gpu_status,
        model_input,
        sha256,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repo root.
    from run_general_semantic_labeler import (
        DEFAULT_LLAMA,
        DEFAULT_ROOT,
        extract_json_object,
        gpu_status,
        model_input,
        sha256,
    )


SYSTEM_PROMPT = """Classify the broad discourse function of one target sentence.
Use only the preceding reasoning and target sentence. Ignore correctness, action
quality, change points, activations, and later text.

Choose exactly one:
- state_readout: directly reads, records, or locates visible state information.
- route_deliberation: proposes, checks, rejects, corrects, compares, or derives a
  route, action, factual claim, consequence, or constraint.
- summary_or_restatement: summarizes several earlier findings or repeats an
  earlier fact, route, conclusion, or decision without materially changing it.
- other: bookkeeping, enumeration, connective narration, an incomplete fragment,
  or genuinely unclear function that does not fit the first three categories.

Precedence: state_readout applies only to direct state transcription, not when the
same fact evaluates a route. Route deliberation outranks summary when a repeated
claim is being checked or revised. Return exactly one JSON object with broad_label,
confidence (high, medium, or low), and a rationale under 20 words."""
PROMPT_VERSION = "general_broad_balanced_fewshot_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_ROOT / "calibration_items.csv"
    )
    parser.add_argument(
        "--few-shots",
        type=Path,
        default=DEFAULT_ROOT / "broad_few_shot_examples.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT / "annotations_llama_3_2_1b_broad.csv",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_LLAMA)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--minimum-free-gib", type=float, default=3.0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_broad_payload(payload: dict[str, Any]) -> dict[str, str]:
    reported_label = str(payload.get("broad_label", "")).strip().lower()
    normalized_label = BROAD_LABEL_MAP.get(reported_label, reported_label)
    result = {
        "broad_label": normalized_label,
        "reported_label": reported_label,
        "label_normalization_applied": str(normalized_label != reported_label).lower(),
        "confidence": str(payload.get("confidence", "")).strip().lower(),
        "rationale": str(payload.get("rationale", "")).strip(),
    }
    if result["broad_label"] not in BROAD_LABELS:
        raise ValueError(f"invalid broad_label: {result['broad_label']!r}")
    if result["confidence"] not in CONFIDENCE_LABELS:
        raise ValueError(f"invalid confidence: {result['confidence']!r}")
    if not result["rationale"]:
        raise ValueError("missing rationale")
    return result


def recover_truncated_broad_payload(text: str) -> dict[str, str]:
    """Recover explicit valid fields when only the JSON closing text was truncated."""
    label = re.search(r'"broad_label"\s*:\s*"([^"]+)"', text)
    confidence = re.search(r'"confidence"\s*:\s*"([^"]+)"', text)
    rationale = re.search(r'"rationale"\s*:\s*"(.+)', text, flags=re.DOTALL)
    if not label or not confidence or not rationale:
        raise ValueError("truncated response lacks explicit required fields")
    rationale_text = rationale.group(1).rstrip().removesuffix("}").rstrip().removesuffix('"')
    return normalize_broad_payload(
        {
            "broad_label": label.group(1),
            "confidence": confidence.group(1),
            "rationale": rationale_text,
        }
    )


def build_messages(
    row: Any, few_shots: list[dict[str, Any]]
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in few_shots:
        payload = {
            "broad_label": example["broad_label"],
            "confidence": example["confidence"],
            "rationale": example["rationale"],
        }
        messages.extend(
            [
                {
                    "role": "user",
                    "content": model_input(type("Example", (), example)()),
                },
                {
                    "role": "assistant",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ]
        )
    messages.append({"role": "user", "content": model_input(row)})
    return messages


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
            "taxonomy": "broad_four_way",
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
        "input_sha256": sha256(args.input),
        "few_shots_sha256": sha256(args.few_shots),
        "model_name": args.model_name,
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
            if str(row.get("broad_label", "")) in BROAD_LABELS
        }
    output = list(completed.values())
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

    for index, row in enumerate(source.itertuples(index=False), start=1):
        annotation_id = str(row.annotation_id)
        if annotation_id in completed:
            continue
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
        parse_repair_applied = False
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
            try:
                parsed = normalize_broad_payload(extract_json_object(raw_response))
            except ValueError:
                parsed = recover_truncated_broad_payload(raw_response)
                parse_repair_applied = True
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        record = {
            "annotation_id": annotation_id,
            "context_before": row.context_before,
            "target_sentence": row.target_sentence,
            "target_sentence_characters": row.target_sentence_characters,
            "broad_label": parsed.get("broad_label", ""),
            "reported_label": parsed.get("reported_label", ""),
            "label_normalization_applied": parsed.get(
                "label_normalization_applied", ""
            ),
            "confidence": parsed.get("confidence", ""),
            "rationale": parsed.get("rationale", ""),
            "judge_model": args.model_name,
            "judge_revision": args.model_path.name,
            "judge_prompt_version": PROMPT_VERSION,
            "judge_prompt_sha256": config["prompt_sha256"],
            "judge_temperature": 0.0,
            "judge_seed": args.seed,
            "raw_response": raw_response,
            "parse_repair_applied": parse_repair_applied,
            "error": error,
        }
        output.append(record)
        if parsed:
            completed[annotation_id] = record
        pd.DataFrame(output).drop_duplicates("annotation_id", keep="last").to_csv(
            args.output, index=False
        )
        if index % 20 == 0 or error:
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
