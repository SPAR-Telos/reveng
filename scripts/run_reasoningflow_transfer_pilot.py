#!/usr/bin/env python3
"""Prepare, label, and summarize a small ReasoningFlow transfer pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from reveng.experiments.reasoningflow_transfer import (
    NODE_COLUMNS,
    REASONINGFLOW_LABELS,
    analyze_pilot,
    extract_json_object,
    gpu_status,
    normalize_nodes,
    prepare_pilot_items,
    select_review_sentences,
    validate_output_columns,
    write_example_splits,
    write_json,
)


DEFAULT_ROOT = Path("outputs/hypothesis_tests/reasoningflow_transfer_pilot_v1")
DEFAULT_INVENTORY = Path(
    "outputs/hypothesis_tests/semantic_reasoning_classification_v1/"
    "general_corpus_v1/sentence_inventory.csv"
)
DEFAULT_MODEL = Path(
    "/root/.cache/huggingface/hub/models--openai--gpt-oss-20b/"
    "snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
)

SYSTEM_PROMPT = """Segment one target sentence into ReasoningFlow nodes and label
each node by its general discourse role. A node is a contiguous span expressing
one role. It can be a complete sentence or a sub-sentence clause.

Use only the target and preceding reasoning. Do not use or predict change points,
action probabilities, optimality, task success, activations, clustering, or
explicit-action-commitment timing.

Labels:
- planning: introduces a next step, subgoal, verification attempt, or alternative.
- fact: states external general knowledge independent of the prompt and prior nodes.
- reasoning: derives a result from the prompt or preceding nodes, including calculations.
- restatement: repeats prompt or preceding content without deriving something new.
- assumption: temporarily supposes a proposition or hypothetical case.
- example: introduces a concrete illustrative case.
- reflection: evaluates the reasoning state, expressing confidence, doubt, or error.
- conclusion: states the answer or a concluded result without further derivation.

Partition the target exactly: nodes must be ordered, non-overlapping, contiguous,
start at character 0, and end at the target's exact character length. Preserve
every character—including spaces and punctuation—in exactly one node. Offsets use
Python half-open indexing [start_character, end_character).

Return exactly one JSON object with key `nodes`. Each node requires:
start_character, end_character, node_text, reasoningflow_label, confidence, and
short_rationale. Confidence is high, medium, or low. Keep each rationale under 20
words. Do not add an action-commitment label or any DoorKey-specific label."""
PROMPT_VERSION = "reasoningflow_transfer_v1"

FEW_SHOTS = [
    {
        "context_before": "The left passage has already been ruled out.",
        "target_sentence": "Let me test whether the upper route is open.",
        "nodes": [
            {
                "start_character": 0,
                "end_character": 44,
                "node_text": "Let me test whether the upper route is open.",
                "reasoningflow_label": "planning",
                "confidence": "high",
                "short_rationale": "It initiates a verification step.",
            }
        ],
    },
    {
        "context_before": "The displayed cell to the right is open.",
        "target_sentence": "Because the right cell is open, moving right remains possible.",
        "nodes": [
            {
                "start_character": 0,
                "end_character": 62,
                "node_text": "Because the right cell is open, moving right remains possible.",
                "reasoningflow_label": "reasoning",
                "confidence": "high",
                "short_rationale": "It derives action feasibility from prior state information.",
            }
        ],
    },
    {
        "context_before": "The key is at coordinate (2, 3).",
        "target_sentence": "So the key is at (2, 3).",
        "nodes": [
            {
                "start_character": 0,
                "end_character": 24,
                "node_text": "So the key is at (2, 3).",
                "reasoningflow_label": "restatement",
                "confidence": "high",
                "short_rationale": "It repeats the preceding location.",
            }
        ],
    },
    {
        "context_before": "The route may pass through the upper cell.",
        "target_sentence": "Suppose that cell is blocked; then the route must turn left.",
        "nodes": [
            {
                "start_character": 0,
                "end_character": 30,
                "node_text": "Suppose that cell is blocked; ",
                "reasoningflow_label": "assumption",
                "confidence": "high",
                "short_rationale": "It introduces a hypothetical case.",
            },
            {
                "start_character": 30,
                "end_character": 60,
                "node_text": "then the route must turn left.",
                "reasoningflow_label": "reasoning",
                "confidence": "high",
                "short_rationale": "It derives a consequence of the assumption.",
            },
        ],
    },
    {
        "context_before": "I previously treated the door as open.",
        "target_sentence": "That may have been a mistake, so I should check the door again.",
        "nodes": [
            {
                "start_character": 0,
                "end_character": 30,
                "node_text": "That may have been a mistake, ",
                "reasoningflow_label": "reflection",
                "confidence": "high",
                "short_rationale": "It expresses doubt about prior reasoning.",
            },
            {
                "start_character": 30,
                "end_character": 63,
                "node_text": "so I should check the door again.",
                "reasoningflow_label": "planning",
                "confidence": "high",
                "short_rationale": "It initiates a new verification step.",
            },
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("prepare", "label", "analyze", "all"), default="all"
    )
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--review-size", type=int, default=40)
    parser.add_argument("--priority-review-size", type=int, default=20)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-name", default="openai/gpt-oss-20b")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--minimum-free-gib", type=float, default=16.0)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument(
        "--reasoning-effort", choices=("low", "medium", "high"), default="medium"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_manifest(path: Path, config: dict[str, Any]) -> None:
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError(f"existing manifest uses a different configuration: {path}")
    write_json(path, config)


def _prompt(row: Any) -> str:
    demonstrations = []
    for index, example in enumerate(FEW_SHOTS, start=1):
        demonstrations.append(
            f"Example {index}\nPreceding reasoning:\n{example['context_before']}\n\n"
            f"Target sentence:\n{example['target_sentence']}\n\nOutput:\n"
            f"{json.dumps({'nodes': example['nodes']}, ensure_ascii=False)}"
        )
    context = str(row.context_before).strip() or "[No preceding reasoning.]"
    return (
        SYSTEM_PROMPT
        + "\n\nLabelled examples:\n\n"
        + "\n\n".join(demonstrations)
        + f"\n\nNow label this item. The target has exactly {len(str(row.target_sentence))} characters."
        + f"\n\nPreceding reasoning:\n{context}\n\nTarget sentence:\n{row.target_sentence}"
    )


def prepare(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(args.inventory)
    items = prepare_pilot_items(inventory, sample_size=args.sample_size, seed=args.seed)
    output = args.output_dir / "pilot_items.csv"
    config = {
        "analysis": "reasoningflow_transfer_pilot_v1",
        "stage": "prepare",
        "inventory": str(args.inventory),
        "inventory_sha256": sha256(args.inventory),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "selection_version": "text_structure_v2",
        "selection_uses_only_text_and_identifiers": True,
    }
    _check_manifest(args.output_dir / "preparation_manifest.json", config)
    items.to_csv(output, index=False)
    counts = items["sample_stratum"].value_counts()
    (args.output_dir / "PREPARATION_REPORT.md").write_text(
        f"""# ReasoningFlow transfer-pilot preparation

This pilot asks whether fine-grained ReasoningFlow nodes transfer cleanly to DoorKey reasoning. It does not test change points, action outcomes, clustering, or explicit action commitment.

- Pilot sentences: {len(items):,}
- Random sentences: {int(counts.get('random', 0)):,}
- Structure-enriched sentences: {int(counts.get('structure_enriched', 0)):,}
- Trajectories represented: {items['environment_id'].nunique():,}
- Previously unlabelled: {len(items):,}

The structure-enriched half is selected from surface features such as length, conjunctions and punctuation. No experimental outcome was used.
"""
    )
    (args.output_dir / "RUNBOOK.md").write_text(
        f"""# ReasoningFlow transfer-pilot runbook

Run from the repository root.

```bash
.venv/bin/python scripts/run_reasoningflow_transfer_pilot.py --stage label --resume
.venv/bin/python scripts/run_reasoningflow_transfer_pilot.py --stage analyze
```

After labelling, complete the review fields in `{args.output_dir / 'manual_review.csv'}` and rerun the analysis stage. Use `yes` or `no` for `boundary_reasonable` and `labels_reasonable`. Record any missing DoorKey-specific role in `missing_doorkey_function`.
"""
    )
    print(f"prepared={len(items)} output={output}")


def label(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    input_path = args.output_dir / "pilot_items.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"prepare stage has not produced {input_path}")
    items = pd.read_csv(input_path).fillna("")
    if args.limit is not None:
        items = items.head(args.limit)

    nodes_path = args.output_dir / "model_node_annotations.csv"
    records_path = args.output_dir / "label_status.csv"
    status_path = args.output_dir / "label_run_status.json"
    manifest_path = args.output_dir / "label_manifest.json"
    config = {
        "analysis": "reasoningflow_transfer_pilot_v1",
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "model_name": args.model_name,
        "model_path": str(args.model_path),
        "model_revision": args.model_path.name,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "few_shots_sha256": hashlib.sha256(
            json.dumps(FEW_SHOTS, sort_keys=True).encode()
        ).hexdigest(),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "reasoning_effort": args.reasoning_effort,
        "limit": args.limit,
    }
    _check_manifest(manifest_path, config)

    gpu = gpu_status(args.gpu_index, args.minimum_free_gib)
    status = {
        **gpu,
        "model_name": args.model_name,
        "model_path": str(args.model_path),
        "input": str(input_path),
        "nodes_output": str(nodes_path),
        "sentence_status_output": str(records_path),
    }
    write_json(status_path, status)
    if gpu["status"] != "ready":
        print(f"{gpu['status']}: {gpu.get('message', gpu)}")
        raise SystemExit(2)
    if not args.model_path.exists():
        status["status"] = "blocked_model_missing"
        write_json(status_path, status)
        raise FileNotFoundError(
            f"local model snapshot does not exist: {args.model_path}"
        )

    sentence_records: list[dict[str, Any]] = []
    node_records: list[dict[str, Any]] = []
    if args.resume and records_path.exists():
        sentence_records = pd.read_csv(records_path).fillna("").to_dict("records")
    if args.resume and nodes_path.exists():
        node_records = pd.read_csv(nodes_path).fillna("").to_dict("records")
    completed = {
        str(record["pilot_id"])
        for record in sentence_records
        if str(record.get("status", "")) == "complete"
    }

    status["status"] = "running"
    write_json(status_path, status)
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

    for index, row in enumerate(items.itertuples(index=False), start=1):
        pilot_id = str(row.pilot_id)
        if pilot_id in completed:
            continue
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": _prompt(row)}],
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=args.reasoning_effort,
        )
        rendered += "<|channel|>final<|message|>"
        inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(
            device
        )
        raw_response = ""
        error = ""
        parsed_nodes: list[dict[str, Any]] = []
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
            parsed_nodes = normalize_nodes(
                extract_json_object(raw_response),
                target_sentence=str(row.target_sentence),
            )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

        sentence_records = [
            r for r in sentence_records if str(r["pilot_id"]) != pilot_id
        ]
        sentence_records.append(
            {
                "pilot_id": pilot_id,
                "sentence_id": row.sentence_id,
                "status": "complete" if parsed_nodes else "error",
                "node_count": len(parsed_nodes),
                "judge_model": args.model_name,
                "judge_revision": args.model_path.name,
                "judge_prompt_version": PROMPT_VERSION,
                "judge_prompt_sha256": config["prompt_sha256"],
                "judge_temperature": 0.0,
                "judge_seed": args.seed,
                "raw_response": raw_response,
                "error": error,
            }
        )
        node_records = [r for r in node_records if str(r["pilot_id"]) != pilot_id]
        for node in parsed_nodes:
            node_records.append(
                {
                    "pilot_id": pilot_id,
                    "sentence_id": row.sentence_id,
                    "environment_id": row.environment_id,
                    "environment_step": row.environment_step,
                    "sentence_number": row.sentence_number,
                    "sample_stratum": row.sample_stratum,
                    "context_before": row.context_before,
                    "target_sentence": row.target_sentence,
                    **node,
                    "judge_model": args.model_name,
                    "judge_revision": args.model_path.name,
                    "judge_prompt_version": PROMPT_VERSION,
                    "judge_prompt_sha256": config["prompt_sha256"],
                }
            )
        pd.DataFrame(sentence_records).sort_values("pilot_id").to_csv(
            records_path, index=False
        )
        pd.DataFrame(node_records).sort_values(["pilot_id", "node_number"]).to_csv(
            nodes_path, index=False
        )
        if parsed_nodes:
            completed.add(pilot_id)
        if index % 10 == 0 or error:
            print(
                f"processed={index}/{len(items)} completed={len(completed)} "
                f"error={error or 'none'}",
                flush=True,
            )

    status["status"] = (
        "complete" if len(completed) == len(items) else "completed_with_errors"
    )
    status["labelled_sentences"] = len(completed)
    status["expected_sentences"] = len(items)
    write_json(status_path, status)
    print(f"status={status['status']} labelled={len(completed)}/{len(items)}")


def analyze(args: argparse.Namespace) -> None:
    items_path = args.output_dir / "pilot_items.csv"
    nodes_path = args.output_dir / "model_node_annotations.csv"
    review_path = args.output_dir / "manual_review.csv"
    if not items_path.exists() or not nodes_path.exists():
        raise FileNotFoundError("prepare and label stages must finish before analysis")
    items = pd.read_csv(items_path).fillna("")
    nodes = pd.read_csv(nodes_path).fillna("")
    validate_output_columns(nodes, NODE_COLUMNS)

    if review_path.exists():
        review = pd.read_csv(review_path).fillna("")
    else:
        review = select_review_sentences(
            items,
            nodes,
            review_size=args.review_size,
            priority_size=args.priority_review_size,
            seed=args.seed,
        )
        review.to_csv(review_path, index=False)

    result = analyze_pilot(items, nodes, review)
    result["segmentation_summary"].to_csv(
        args.output_dir / "segmentation_summary.csv", index=False
    )
    result["label_summary"].to_csv(args.output_dir / "label_summary.csv", index=False)
    result["schema_gaps"].to_csv(args.output_dir / "schema_gaps.csv", index=False)
    write_example_splits(args.output_dir / "example_splits.md", items, nodes)

    reviewed = result["reviewed_metrics"]
    boundary = reviewed.get("boundary_reasonable", {})
    labels = reviewed.get("labels_reasonable", {})
    boundary_text = (
        f"{boundary['yes']}/{boundary['reviewed']} ({boundary['rate']:.1%})"
        if boundary.get("reviewed")
        else "pending manual review"
    )
    labels_text = (
        f"{labels['yes']}/{labels['reviewed']} ({labels['rate']:.1%})"
        if labels.get("reviewed")
        else "pending manual review"
    )
    multi = int(
        result["sentence_summary"]["segmentation_group"].eq("multiple_nodes").sum()
    )
    report = f"""# ReasoningFlow transfer pilot

## Question

Can a literature-derived, sub-sentence ReasoningFlow schema describe DoorKey reasoning clearly enough to serve as a shared segmentation layer?

## Method

The pilot contains {result['pilot_sentences']:,} previously unlabelled DoorKey sentences: half randomly sampled and half enriched for multi-clause surface structure. GPT-OSS-20B receives only preceding reasoning and the target sentence. It assigns the eight ReasoningFlow response-node roles. No edge labels, change points, action outcomes, clustering labels, or action-commitment labels are used.

## Current result

- Successfully labelled sentences: {result['labelled_sentences']:,}/{result['pilot_sentences']:,}
- Produced nodes: {result['nodes']:,}
- Sentences split into multiple nodes: {multi:,}
- Reviewed boundary acceptability: {boundary_text}
- Reviewed label acceptability: {labels_text}
- Reviewed examples reporting a missing DoorKey function: {len(result['schema_gaps']):,}

## Interpretation

The model output is not a validated annotation corpus until the 40-sentence review is completed. If the boundaries and general labels are consistently understandable, ReasoningFlow can be retained as a general discourse layer. Observation, route evaluation and action selection should remain a separate DoorKey-specific layer when reviewers report that the general schema loses those distinctions. Explicit commitment remains a separate annotation joined by the shared sentence and span identifiers.

## Scope relative to other work

This is the literature-supervised branch of reasoning-step classification, not a separate fifth classification project. Clustering should remain label-blind until comparison. The pilot does not operationalize explicit action commitment.
"""
    (args.output_dir / "run_report.md").write_text(report)
    write_json(
        args.output_dir / "analysis_summary.json",
        {
            "pilot_sentences": result["pilot_sentences"],
            "labelled_sentences": result["labelled_sentences"],
            "nodes": result["nodes"],
            "multiple_node_sentences": multi,
            "reviewed_metrics": reviewed,
            "schema_gap_rows": len(result["schema_gaps"]),
            "reasoningflow_labels": list(REASONINGFLOW_LABELS),
        },
    )
    print(f"report={args.output_dir / 'run_report.md'}")


def main() -> None:
    args = parse_args()
    if args.stage in {"prepare", "all"}:
        prepare(args)
    if args.stage in {"label", "all"}:
        label(args)
    if args.stage in {"analyze", "all"}:
        analyze(args)


if __name__ == "__main__":
    main()
