#!/usr/bin/env python3
"""Run the general multi-label semantic judge with a local causal LM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from reveng.experiments.general_semantic_labeling import write_json

try:
    from scripts.run_general_semantic_labeler import extract_json_object, gpu_status
    from scripts.run_general_semantic_together_multilabel_labeler import (
        BOOLEAN_FIELDS,
        BROAD_LABEL_MAP,
        PROMPT_VERSION,
        SYSTEM_PROMPT,
        derive_primary_label,
        few_shot_payload,
        is_completed_record,
        model_input,
        normalize_payload,
        prepare_source,
        select_source,
        source_metadata,
    )
except ModuleNotFoundError:  # Direct execution puts scripts/ on sys.path.
    from run_general_semantic_labeler import extract_json_object, gpu_status
    from run_general_semantic_together_multilabel_labeler import (
        BOOLEAN_FIELDS,
        BROAD_LABEL_MAP,
        PROMPT_VERSION,
        SYSTEM_PROMPT,
        derive_primary_label,
        few_shot_payload,
        is_completed_record,
        model_input,
        normalize_payload,
        prepare_source,
        select_source,
        source_metadata,
    )


DEFAULT_ROOT = Path(
    "outputs/hypothesis_tests/semantic_reasoning_classification_v1/general_corpus_v1"
)
DEFAULT_MODEL = Path(
    "/root/.cache/huggingface/hub/models--openai--gpt-oss-20b/"
    "snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
)
LOCAL_PROMPT_VERSION = PROMPT_VERSION + "_local"


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
        default=DEFAULT_ROOT / "annotations_gpt_oss_20b_multilabel_local.csv",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-name", default="openai/gpt-oss-20b")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--minimum-free-gib", type=float, default=16.0)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument(
        "--reasoning-effort", choices=("low", "medium", "high"), default="medium"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--sample-strategy", choices=("head", "stratified"), default="stratified"
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--duplicate-items", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_gpt_oss_prompt(row: Any, few_shots: list[dict[str, Any]]) -> str:
    demonstrations: list[str] = []
    for index, example in enumerate(few_shots, start=1):
        payload = few_shot_payload(example)
        normalize_payload(payload)
        demonstrations.append(
            f"Example {index}\nINPUT\n{model_input(type('Example', (), example)())}\n"
            f"OUTPUT\n{json.dumps(payload, ensure_ascii=False)}"
        )
    return (
        SYSTEM_PROMPT
        + "\n\nLabelled examples:\n\n"
        + "\n\n".join(demonstrations)
        + "\n\nNow classify this item.\n"
        + model_input(row)
    )


def empty_record(
    row: Any, *, config: dict[str, Any], error: str, raw: str
) -> dict[str, Any]:
    return {
        "annotation_id": str(row.annotation_id),
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
        "judge_model": config["model_name"],
        "judge_provider": "local_transformers",
        "judge_revision": config["model_revision"],
        "judge_prompt_version": LOCAL_PROMPT_VERSION,
        "judge_prompt_sha256": config["prompt_sha256"],
        "judge_temperature": 0.0,
        "judge_seed": config["seed"],
        "judge_reasoning_effort": config["reasoning_effort"],
        "raw_response": raw,
        "error": error,
    }


def main() -> None:
    args = parse_args()
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be at least 1")
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
    if args.validate_only:
        print(
            f"validated input_rows={len(source)} few_shots={len(few_shots)} "
            f"prompt_version={LOCAL_PROMPT_VERSION}"
        )
        return

    status_path = args.output.with_suffix(".status.json")
    status = gpu_status(args.gpu_index, args.minimum_free_gib)
    status.update(
        {
            "model_name": args.model_name,
            "model_path": str(args.model_path),
            "input": str(args.input),
            "output": str(args.output),
            "taxonomy": PROMPT_VERSION,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(status_path, status)
    if status["status"] != "ready":
        print(f"{status['status']}: {status.get('message', status)}")
        raise SystemExit(2)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = {
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "few_shots": str(args.few_shots),
        "few_shots_sha256": sha256(args.few_shots),
        "model_name": args.model_name,
        "model_path": str(args.model_path),
        "model_revision": args.model_path.name,
        "reasoning_effort": args.reasoning_effort,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "sample_strategy": args.sample_strategy,
        "sample_seed": args.sample_seed,
        "duplicate_items": args.duplicate_items,
        "prompt_version": LOCAL_PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
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
            if is_completed_record(row)
        }
    records_by_id = dict(completed)
    source_rows = list(source.itertuples(index=False))
    status.update(
        {
            "status": "running",
            "expected_rows": len(source),
            "labeled_rows": len(completed),
            "failed_rows": 0,
        }
    )
    write_json(status_path, status)

    try:
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
    except Exception as exc:
        status["status"] = "blocked_model_load"
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_json(status_path, status)
        raise
    device = next(model.parameters()).device
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    def checkpoint() -> None:
        ordered = [
            records_by_id[str(row.annotation_id)]
            for row in source_rows
            if str(row.annotation_id) in records_by_id
        ]
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        pd.DataFrame(ordered).to_csv(temporary, index=False)
        temporary.replace(args.output)
        status["labeled_rows"] = len(completed)
        status["failed_rows"] = sum(
            bool(str(record.get("error", "")).strip())
            for record in records_by_id.values()
        )
        write_json(status_path, status)

    processed = 0
    for row in source_rows:
        annotation_id = str(row.annotation_id)
        if annotation_id in completed:
            continue
        raw_response = ""
        try:
            rendered = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": build_gpt_oss_prompt(row, few_shots),
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort=args.reasoning_effort,
            )
            rendered += "<|channel|>final<|message|>"
            inputs = tokenizer(
                rendered, return_tensors="pt", add_special_tokens=False
            ).to(device)
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
            labels = parsed.pop("semantic_labels")
            primary = derive_primary_label(labels)
            record = {
                "annotation_id": annotation_id,
                **source_metadata(row),
                "context_before": row.context_before,
                "target_sentence": row.target_sentence,
                "target_sentence_characters": row.target_sentence_characters,
                "semantic_labels": json.dumps(labels),
                "primary_label": primary,
                **parsed,
                "broad_label": BROAD_LABEL_MAP[primary],
                "judge_model": args.model_name,
                "judge_provider": "local_transformers",
                "judge_revision": args.model_path.name,
                "judge_prompt_version": LOCAL_PROMPT_VERSION,
                "judge_prompt_sha256": config["prompt_sha256"],
                "judge_temperature": 0.0,
                "judge_seed": args.seed,
                "judge_reasoning_effort": args.reasoning_effort,
                "raw_response": raw_response,
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001
            record = empty_record(
                row,
                config=config,
                error=f"{type(exc).__name__}: {exc}",
                raw=raw_response,
            )
        records_by_id[annotation_id] = record
        if is_completed_record(record):
            completed[annotation_id] = record
        processed += 1
        if processed % args.checkpoint_every == 0 or record["error"]:
            checkpoint()
        if processed % 10 == 0 or record["error"]:
            print(
                f"processed={processed} completed={len(completed)}/{len(source)} "
                f"error={record['error'] or 'none'}",
                flush=True,
            )

    checkpoint()
    status["status"] = "complete" if len(completed) == len(source) else "incomplete"
    status["labeled_rows"] = len(completed)
    write_json(status_path, status)
    print(f"output={args.output} labeled={len(completed)}/{len(source)}")


if __name__ == "__main__":
    main()
