#!/usr/bin/env python3
"""Compact final-action attention analysis for DoorKey reasoning traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from reveng.experiments.gpt_oss_activation_full import _load_corpus_definition
from reveng.experiments.gpt_oss_activation_pilot import (
    DEFAULT_LAYERS,
    DEFAULT_MODEL_SNAPSHOT,
    DEFAULT_SENTENCE_BOUNDARIES,
    DEFAULT_TRAJECTORY_DIR,
    PeakResourceMonitor,
    PilotState,
    PreparedState,
    _filesystem_bytes,
    _payload_step,
    _prepare_state,
    _tensor_logical_bytes,
)


DEFAULT_OUTPUT_DIR = Path("outputs/hypothesis_tests/attention_decision_relevance_v1")
DEFAULT_POSITION_ROWS = Path(
    "outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1/position_rows.csv"
)
DEFAULT_EVENT_ROWS = Path(
    "outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1/event_rows.csv"
)
DEFAULT_BELIEF_SHIFT_ROWS = Path(
    "outputs/experiment2_behavioral_beliefs/gpt_oss_local_sentence_matched46_v1/belief_shift_rows.csv"
)


def log(message: str) -> None:
    print(f"[attention-decision] {message}", file=sys.stderr, flush=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mu = mean(values)
    var = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(var / len(values))


def ci95(values: list[float]) -> tuple[float, float]:
    mu = mean(values)
    se = standard_error(values)
    if math.isnan(se):
        return float("nan"), float("nan")
    return mu - 1.96 * se, mu + 1.96 * se


def trace_id(trajectory_id: str, step_index: int) -> str:
    return f"{trajectory_id}_step_{int(step_index):03d}"


def selected_trace_ids(scope: str, position_rows_path: Path) -> set[str] | None:
    if scope == "full":
        return None
    rows = read_csv(position_rows_path)
    return {str(row["example_id"]) for row in rows}


def expected_keys(layers: Sequence[int]) -> set[str]:
    keys = {"sentence_token_counts"}
    for layer in layers:
        keys.add(f"layer_{int(layer)}.final_action_to_sentence_mass")
        keys.add(f"layer_{int(layer)}.final_action_to_sentence_mean")
    return keys


def completed_record_is_valid(record_path: Path, layers: Sequence[int]) -> bool:
    if not record_path.exists():
        return False
    try:
        record = json.loads(record_path.read_text())
        shard_path = Path(record["shard_path"])
        if not shard_path.exists():
            return False
        if shard_path.stat().st_size != int(record["file_size_bytes"]):
            return False
        if sha256_file(shard_path) != record["shard_sha256"]:
            return False
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != expected_keys(layers):
                return False
            metadata = handle.metadata() or {}
            if metadata.get("trace_id") != record["trace_id"]:
                return False
        return True
    except Exception:
        return False


def extract_final_action_attention(
    model: Any,
    state: PreparedState,
    *,
    layers: Sequence[int],
    input_device: torch.device,
    forward_chunk_size: int,
    output_path: Path,
) -> dict[str, Any]:
    base_model = getattr(model, "model", model)
    decoder_layers = getattr(base_model, "layers", None)
    if decoder_layers is None:
        raise ValueError("Could not locate GPT-OSS decoder layers.")
    input_ids = torch.tensor([state.input_ids], dtype=torch.long, device=input_device)
    past_key_values = None
    prefix_seconds = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    with PeakResourceMonitor() as monitor, torch.inference_mode():
        for start in range(0, state.action_position, forward_chunk_size):
            end = min(state.action_position, start + forward_chunk_size)
            before = time.perf_counter()
            outputs = base_model(
                input_ids=input_ids[:, start:end],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prefix_seconds += time.perf_counter() - before
            past_key_values = outputs.past_key_values
            del outputs

        captured: dict[int, torch.Tensor] = {}
        handles = []

        def capture(layer: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                if not isinstance(output, tuple) or output[1] is None:
                    raise ValueError(f"Layer {layer} did not expose attention weights.")
                captured[layer] = output[1].detach().to(device="cpu", dtype=torch.float32)

            return hook

        for layer in layers:
            handles.append(decoder_layers[int(layer)].self_attn.register_forward_hook(capture(int(layer))))
        old_implementation = getattr(model.config, "_attn_implementation", None)
        model.config._attn_implementation = "eager"
        query_start = time.perf_counter()
        try:
            outputs = base_model(
                input_ids=input_ids[:, state.action_position : state.action_position + 1],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_attentions=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            query_seconds = time.perf_counter() - query_start
            del outputs
        finally:
            if old_implementation is not None:
                model.config._attn_implementation = old_implementation
            for handle in handles:
                handle.remove()

        aggregate_start = time.perf_counter()
        sentence_counts = torch.tensor(
            [end - start for start, end in state.sentence_spans], dtype=torch.int64
        )
        tensors: dict[str, torch.Tensor] = {"sentence_token_counts": sentence_counts}
        for layer in layers:
            weights = captured[int(layer)]
            action_weights = weights[0, :, 0, :]
            sentence_mass = torch.zeros(
                (action_weights.shape[0], len(state.sentence_spans)), dtype=torch.float32
            )
            sentence_mean = torch.zeros_like(sentence_mass)
            for sentence_index, (start, end) in enumerate(state.sentence_spans):
                clipped_end = min(end, action_weights.shape[-1])
                if start < clipped_end:
                    span_weights = action_weights[:, start:clipped_end]
                    sentence_mass[:, sentence_index] = span_weights.sum(dim=-1)
                    sentence_mean[:, sentence_index] = span_weights.mean(dim=-1)
            tensors[f"layer_{layer}.final_action_to_sentence_mass"] = sentence_mass
            tensors[f"layer_{layer}.final_action_to_sentence_mean"] = sentence_mean
        aggregate_seconds = time.perf_counter() - aggregate_start

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".safetensors.partial")
    serialization_start = time.perf_counter()
    save_file(
        tensors,
        str(temporary),
        metadata={
            "trace_id": state.spec.trace_id,
            "query_position": str(state.action_position),
            "attention_implementation": "eager",
            "scope": "final action token to preceding canonical reasoning sentences",
            "primary_metric": "final_action_to_sentence_mass",
        },
    )
    os.replace(temporary, output_path)
    serialization_seconds = time.perf_counter() - serialization_start
    file_size = output_path.stat().st_size
    metrics = {
        "trace_id": state.spec.trace_id,
        "trajectory_id": state.spec.trajectory_id,
        "step_index": state.spec.step_index,
        "layers": json.dumps([int(layer) for layer in layers]),
        "sentence_count": len(state.sentence_spans),
        "action_position": state.action_position,
        "final_action": state.final_action,
        "prefix_input_tokens": state.action_position,
        "source_tokens_at_action": state.action_position + 1,
        "prefix_forward_seconds": prefix_seconds,
        "attention_query_seconds": query_seconds,
        "attention_aggregate_seconds": aggregate_seconds,
        "serialization_seconds": serialization_seconds,
        "logical_tensor_bytes": _tensor_logical_bytes(tensors),
        "file_size_bytes": file_size,
        "filesystem_bytes": _filesystem_bytes(output_path),
        "peak_rss_bytes": monitor.peak_rss_bytes,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
        "shard_path": str(output_path),
        "shard_sha256": sha256_file(output_path),
    }
    del past_key_values, input_ids, tensors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def extract_attention(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir
    shard_dir = output / "shards"
    record_dir = output / "state_records"
    output.mkdir(parents=True, exist_ok=True)
    specs, boundary_rows, states_by_trajectory = _load_corpus_definition(
        trajectory_dir=args.trajectory_dir,
        sentence_boundaries_path=args.sentence_boundaries_path,
    )
    selected = selected_trace_ids(args.scope, args.position_rows)
    if selected is not None:
        specs = [spec for spec in specs if spec.trace_id in selected]
        states_by_trajectory = {
            trajectory_id: [spec for spec in trajectory_specs if spec.trace_id in selected]
            for trajectory_id, trajectory_specs in states_by_trajectory.items()
        }
        states_by_trajectory = {key: value for key, value in states_by_trajectory.items() if value}
    if args.limit is not None:
        specs = specs[: int(args.limit)]
        limited = {spec.trace_id for spec in specs}
        states_by_trajectory = {
            trajectory_id: [spec for spec in trajectory_specs if spec.trace_id in limited]
            for trajectory_id, trajectory_specs in states_by_trajectory.items()
        }
        states_by_trajectory = {key: value for key, value in states_by_trajectory.items() if value}

    layers = tuple(sorted({int(layer) for layer in args.layers}))
    completed = {
        spec.trace_id
        for spec in specs
        if args.resume and completed_record_is_valid(record_dir / f"{spec.trace_id}.json", layers)
    }
    pending = [spec for spec in specs if spec.trace_id not in completed]
    manifest = {
        "status": "running" if pending else "completed",
        "scope": args.scope,
        "model_name": "openai/gpt-oss-20b",
        "model_snapshot": str(args.model_path),
        "model_revision": args.model_path.name,
        "layers": list(layers),
        "forward_chunk_size": args.forward_chunk_size,
        "states": len(specs),
        "completed_states": len(completed),
        "pending_states": len(pending),
        "sentence_boundaries_path": str(args.sentence_boundaries_path),
        "attention_metric_primary": "final_action_to_sentence_mass",
        "attention_metric_secondary": "final_action_to_sentence_mean",
    }
    write_json(output / "run_manifest.json", manifest)
    if pending:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        log("loading openai/gpt-oss-20b")
        load_start = time.perf_counter()
        with PeakResourceMonitor() as load_monitor:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                local_files_only=True,
                dtype=torch.bfloat16,
                device_map={"": 0},
                low_cpu_mem_usage=True,
            ).eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        manifest["model_load_seconds"] = time.perf_counter() - load_start
        manifest["model_load_peak_rss_bytes"] = load_monitor.peak_rss_bytes
        input_device = model.model.embed_tokens.weight.device
        extracted = 0
        for trajectory_id in sorted(states_by_trajectory):
            trajectory_specs = [spec for spec in states_by_trajectory[trajectory_id] if spec.trace_id in {item.trace_id for item in pending}]
            if not trajectory_specs:
                continue
            payload = json.loads((args.trajectory_dir / f"{trajectory_id}.json").read_text())
            for spec in trajectory_specs:
                log(f"extracting {spec.trace_id} ({spec.expected_sentences} sentences)")
                prepared = _prepare_state(
                    spec,
                    trajectory_dir=args.trajectory_dir,
                    boundary_rows=boundary_rows[spec.trace_id],
                    tokenizer=tokenizer,
                    payload=payload,
                )
                shard_path = shard_dir / f"{spec.trace_id}.safetensors"
                metrics = extract_final_action_attention(
                    model,
                    prepared,
                    layers=layers,
                    input_device=input_device,
                    forward_chunk_size=args.forward_chunk_size,
                    output_path=shard_path,
                )
                metrics["token_ids_match"] = prepared.token_ids_match
                metrics["tokenizer_validation_seconds"] = prepared.tokenizer_validation_seconds
                write_json(record_dir / f"{spec.trace_id}.json", metrics)
                extracted += 1
                manifest.update(
                    {
                        "completed_states": len(completed) + extracted,
                        "pending_states": len(pending) - extracted,
                        "last_completed_trace_id": spec.trace_id,
                    }
                )
                write_json(output / "run_manifest.json", manifest)
            del payload
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    records = [json.loads(path.read_text()) for path in sorted(record_dir.glob("*.json"))]
    records = [record for record in records if record["trace_id"] in {spec.trace_id for spec in specs}]
    write_csv(output / "attention_index.csv", records)
    manifest.update(
        {
            "status": "completed" if len(records) == len(specs) else "partial",
            "completed_states": len(records),
            "pending_states": len(specs) - len(records),
            "completed_sentences": sum(int(record["sentence_count"]) for record in records),
            "file_size_bytes": sum(int(record["file_size_bytes"]) for record in records),
            "filesystem_bytes": sum(int(record["filesystem_bytes"]) for record in records),
        }
    )
    write_json(output / "run_manifest.json", manifest)
    return manifest


def load_attention_tensor(record: dict[str, Any], layer: int, metric: str) -> torch.Tensor:
    key = f"layer_{int(layer)}.final_action_to_sentence_{metric}"
    with safe_open(record["shard_path"], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).float()


def rows_by_trace(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        grouped[row["example_id"]].append(row)
    return dict(grouped)


def build_events(
    *,
    event_rows_path: Path,
    belief_shift_rows_path: Path,
    position_rows_path: Path,
    include_belief_families: bool = True,
) -> list[dict[str, Any]]:
    positions = rows_by_trace(position_rows_path)
    metadata = {
        trace_id: {
            "matched_role": rows[0].get("matched_role", ""),
            "trajectory_class": rows[0].get("trajectory_class", ""),
            "primary_step_failure_mode": rows[0].get("primary_step_failure_mode", ""),
            "final_full_trace_action": rows[0].get("final_full_trace_action", ""),
            "final_action_is_optimal": rows[-1].get("action_is_optimal", ""),
        }
        for trace_id, rows in positions.items()
        if rows
    }
    events: list[dict[str, Any]] = []
    for row in read_csv(event_rows_path):
        idx = int(float(row["reasoning_step_idx"]))
        center = idx - 1
        if center < 0:
            continue
        trace = row["example_id"]
        events.append(
            {
                "event_source": "action",
                "event_type": row["event_type"],
                "trace_id": trace,
                "trajectory_id": row["trajectory_id"],
                "step_index": int(row["step_index"]),
                "event_reasoning_step_idx": idx,
                "center_sentence_index": center,
                **metadata.get(trace, {}),
            }
        )
    for row in read_csv(belief_shift_rows_path):
        idx = int(float(row["reasoning_step_idx"]))
        center = idx - 1
        if center < 0:
            continue
        trace = row["example_id"]
        event_type = "belief_change"
        if include_belief_families:
            event_type = f"belief_change:{row.get('question_family', '') or 'unknown'}"
        events.append(
            {
                "event_source": "belief",
                "event_type": event_type,
                "trace_id": trace,
                "trajectory_id": row["trajectory_id"],
                "step_index": int(row["step_index"]),
                "event_reasoning_step_idx": idx,
                "center_sentence_index": center,
                "question_id": row.get("question_id", ""),
                "question_family": row.get("question_family", ""),
                "shift_type": row.get("shift_type", ""),
                **metadata.get(trace, {}),
            }
        )
    return events


def window_indices(center: int, sentence_count: int, width: int) -> list[int]:
    return [idx for idx in range(center - width, center + width + 1) if 0 <= idx < sentence_count]


def choose_control_center(center: int, sentence_count: int, width: int) -> int | None:
    candidates = [
        idx
        for idx in range(sentence_count)
        if abs(idx - center) > width and window_indices(idx, sentence_count, width)
    ]
    if not candidates:
        return None
    target_fraction = center / max(1, sentence_count - 1)
    return min(candidates, key=lambda idx: (abs(idx / max(1, sentence_count - 1) - target_fraction), abs(idx - center)))


def analyze_attention(args: argparse.Namespace) -> None:
    output = args.output_dir
    records = read_csv(output / "attention_index.csv")
    records_by_trace = {row["trace_id"]: row for row in records}
    events = build_events(
        event_rows_path=args.event_rows,
        belief_shift_rows_path=args.belief_shift_rows,
        position_rows_path=args.position_rows,
    )
    events = [event for event in events if event["trace_id"] in records_by_trace]
    layers = tuple(sorted({int(layer) for layer in args.layers}))
    event_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    tensor_cache: dict[tuple[str, int], torch.Tensor] = {}
    for event in events:
        record = records_by_trace[event["trace_id"]]
        sentence_count = int(record["sentence_count"])
        center = int(event["center_sentence_index"])
        if center >= sentence_count:
            continue
        control_center = choose_control_center(center, sentence_count, args.event_window)
        if control_center is None:
            continue
        event_window = window_indices(center, sentence_count, args.event_window)
        control_window = window_indices(control_center, sentence_count, args.event_window)
        for layer in layers:
            cache_key = (event["trace_id"], layer)
            if cache_key not in tensor_cache:
                tensor_cache[cache_key] = load_attention_tensor(record, layer, args.metric)
            attention = tensor_cache[cache_key]
            event_head_values = attention[:, event_window].sum(dim=1)
            control_head_values = attention[:, control_window].sum(dim=1)
            diff = event_head_values - control_head_values
            common = {
                **event,
                "layer": layer,
                "metric": args.metric,
                "event_window_radius": args.event_window,
                "event_window_n_sentences": len(event_window),
                "control_center_sentence_index": control_center,
                "control_window_n_sentences": len(control_window),
                "event_attention_mean_across_heads": float(event_head_values.mean().item()),
                "control_attention_mean_across_heads": float(control_head_values.mean().item()),
                "event_minus_control_attention": float(diff.mean().item()),
                "attention_heads": int(attention.shape[0]),
            }
            event_rows.append(common)
            for head_index in range(int(attention.shape[0])):
                head_rows.append(
                    {
                        **event,
                        "layer": layer,
                        "head_index": head_index,
                        "metric": args.metric,
                        "event_attention": float(event_head_values[head_index].item()),
                        "control_attention": float(control_head_values[head_index].item()),
                        "event_minus_control_attention": float(diff[head_index].item()),
                        "fold": int(hashlib.sha256(str(event["trajectory_id"]).encode()).hexdigest(), 16) % args.folds,
                    }
                )

    write_csv(output / "attention_event_window_rows.csv", event_rows)
    write_csv(output / "attention_head_event_rows.csv", head_rows)
    write_csv(output / "attention_event_rows_used.csv", events)
    write_csv(output / "attention_event_window_summary.csv", summarize_event_rows(event_rows))
    write_csv(output / "attention_head_summary.csv", summarize_head_rows(head_rows, args.folds))
    write_report(output, event_rows, head_rows, args)
    write_figures(output, event_rows, head_rows, args)


def summarize_event_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["event_type"]), int(row["layer"]), "all")].append(row)
        grouped[(str(row["event_type"]), int(row["layer"]), str(row.get("matched_role", "")))].append(row)
    out: list[dict[str, Any]] = []
    for (event_type, layer, matched_role), group in sorted(grouped.items()):
        diffs = [float(row["event_minus_control_attention"]) for row in group]
        lo, hi = ci95(diffs)
        out.append(
            {
                "event_type": event_type,
                "layer": layer,
                "matched_role": matched_role,
                "events": len(group),
                "states": len({row["trace_id"] for row in group}),
                "trajectories": len({row["trajectory_id"] for row in group}),
                "mean_event_attention": mean(float(row["event_attention_mean_across_heads"]) for row in group),
                "mean_control_attention": mean(float(row["control_attention_mean_across_heads"]) for row in group),
                "mean_event_minus_control_attention": mean(diffs),
                "se_event_minus_control_attention": standard_error(diffs),
                "ci95_low": lo,
                "ci95_high": hi,
                "fraction_positive": mean(1.0 if value > 0 else 0.0 for value in diffs),
            }
        )
    return out


def summarize_head_rows(rows: list[dict[str, Any]], folds: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["event_type"]), int(row["layer"]), int(row["head_index"]))].append(row)
    out: list[dict[str, Any]] = []
    for (event_type, layer, head), group in sorted(grouped.items()):
        diffs = [float(row["event_minus_control_attention"]) for row in group]
        fold_means = []
        for fold in range(folds):
            fold_values = [float(row["event_minus_control_attention"]) for row in group if int(row["fold"]) == fold]
            if fold_values:
                fold_means.append(mean(fold_values))
        out.append(
            {
                "event_type": event_type,
                "layer": layer,
                "head_index": head,
                "events": len(group),
                "states": len({row["trace_id"] for row in group}),
                "mean_event_minus_control_attention": mean(diffs),
                "fraction_positive_events": mean(1.0 if value > 0 else 0.0 for value in diffs),
                "folds_observed": len(fold_means),
                "folds_positive": sum(value > 0 for value in fold_means),
                "replicated_positive": len(fold_means) >= max(2, folds - 1) and sum(value > 0 for value in fold_means) >= max(2, folds - 1),
            }
        )
    out.sort(key=lambda row: (not bool(row["replicated_positive"]), -float(row["mean_event_minus_control_attention"])))
    return out


def write_report(output: Path, event_rows: list[dict[str, Any]], head_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    summary = summarize_event_rows(event_rows)
    top_heads = summarize_head_rows(head_rows, args.folds)[:10]
    lines = [
        "# Attention Decision Relevance",
        "",
        "This analysis stores compact attention from the final action token to canonical reasoning sentences. It does not store full token-by-token attention matrices.",
        "",
        f"- Scope: `{args.scope}`",
        f"- Metric: final-action attention `{args.metric}` aggregated over sentence windows",
        f"- Event window: +/-{args.event_window} sentences",
        f"- Event-window rows: {len(event_rows):,}",
        f"- Head-event rows: {len(head_rows):,}",
        "",
        "## Main event-window summary",
        "",
        "| Event type | Layer | Group | Events | Mean event minus control attention | Fraction positive |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in summary:
        if int(row["layer"]) != 15:
            continue
        lines.append(
            f"| {row['event_type']} | {row['layer']} | {row['matched_role'] or 'all'} | "
            f"{row['events']} | {float(row['mean_event_minus_control_attention']):.6f} | "
            f"{float(row['fraction_positive']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Top replicated heads",
            "",
            "| Event type | Layer | Head | Events | Mean event minus control attention | Positive folds |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top_heads:
        lines.append(
            f"| {row['event_type']} | {row['layer']} | {row['head_index']} | "
            f"{row['events']} | {float(row['mean_event_minus_control_attention']):.6f} | "
            f"{row['folds_positive']}/{row['folds_observed']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation note",
            "",
            "This is correlational. A positive event-minus-control value means the final action token attended more to sentences near that event than to a progress-matched non-event window in the same state. This supports an information-routing hypothesis only as auxiliary evidence; it does not prove causal use.",
        ]
    )
    (output / "attention_decision_relevance_report.md").write_text("\n".join(lines) + "\n")


def write_figures(output: Path, event_rows: list[dict[str, Any]], head_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    figs = output / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    summary = [
        row
        for row in summarize_event_rows(event_rows)
        if int(row["layer"]) == 15 and row["matched_role"] == "all"
    ]
    if not summary:
        summary = [row for row in summarize_event_rows(event_rows) if int(row["layer"]) == 15]
    summary = sorted(summary, key=lambda row: str(row["event_type"]))
    labels = [str(row["event_type"]).replace("belief_change:", "belief: ") for row in summary]
    values = [float(row["mean_event_minus_control_attention"]) for row in summary]
    lows = [float(row["ci95_low"]) for row in summary]
    highs = [float(row["ci95_high"]) for row in summary]
    lower = [abs(v - lo) if not math.isnan(lo) else 0.0 for v, lo in zip(values, lows)]
    upper = [abs(hi - v) if not math.isnan(hi) else 0.0 for v, hi in zip(values, highs)]
    if labels:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(range(len(labels)), values, color="#2f6fb3", yerr=[lower, upper], capsize=3)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Final-action attention to event window minus matched non-event window")
        ax.set_title("Final Action Attention Near Decision Events")
        fig.tight_layout()
        fig.savefig(figs / "attention_event_window_difference.png", dpi=200)
        plt.close(fig)

    heads = [
        row
        for row in summarize_head_rows(head_rows, args.folds)
        if bool(row["replicated_positive"])
    ][:15]
    if heads:
        labels = [f"L{row['layer']} H{row['head_index']} {row['event_type']}" for row in heads]
        values = [float(row["mean_event_minus_control_attention"]) for row in heads]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.barh(range(len(labels)), values, color="#5b9bd5")
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Final-action attention to event window minus matched non-event window")
        ax.set_title("Heads With Replicated Attention Increase Near Events")
        fig.tight_layout()
        fig.savefig(figs / "top_attention_heads.png", dpi=200)
        plt.close(fig)

    captions = [
        "# Figure Captions",
        "",
        "## attention_event_window_difference.png",
        "",
        "For each state, attention is measured from the final action token to reasoning sentences. The plotted value is attention to sentences within three sentences of a decision or belief event minus attention to a progress-matched non-event window in the same state. Positive values mean the final action token attended more to the event window.",
        "",
        "## top_attention_heads.png",
        "",
        "Heads are shown when their event-window attention increase is positive in most trajectory-grouped folds. This is a descriptive head-level screen, not causal evidence.",
    ]
    (output / "FIGURE_CAPTIONS.md").write_text("\n".join(captions) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["matched46", "full"], default="matched46")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_SNAPSHOT)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--sentence-boundaries-path", type=Path, default=DEFAULT_SENTENCE_BOUNDARIES)
    parser.add_argument("--position-rows", type=Path, default=DEFAULT_POSITION_ROWS)
    parser.add_argument("--event-rows", type=Path, default=DEFAULT_EVENT_ROWS)
    parser.add_argument("--belief-shift-rows", type=Path, default=DEFAULT_BELIEF_SHIFT_ROWS)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--forward-chunk-size", type=int, default=512)
    parser.add_argument("--event-window", type=int, default=3)
    parser.add_argument("--metric", choices=["mass", "mean"], default="mass")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_extraction:
        extract_attention(args)
    if not args.skip_analysis:
        analyze_attention(args)


if __name__ == "__main__":
    main()
