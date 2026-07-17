"""Resumable full-corpus GPT-OSS boundary-activation extraction."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from reveng.experiments.gpt_oss_activation_pilot import (
    DEFAULT_LAYERS,
    DEFAULT_MODEL_SNAPSHOT,
    DEFAULT_SENTENCE_BOUNDARIES,
    DEFAULT_TRAJECTORY_DIR,
    FULL_CORPUS_ENVIRONMENT_STATES,
    FULL_CORPUS_REASONING_TOKENS,
    FULL_CORPUS_SENTENCES,
    BoundaryAggregateCollector,
    PeakResourceMonitor,
    PilotState,
    PreparedState,
    _filesystem_bytes,
    _index_rows,
    _payload_step,
    _prepare_state,
    _tensor_logical_bytes,
    _write_csv,
    _write_parquet,
)


DEFAULT_FULL_OUTPUT_DIR = Path("outputs/activation_collection/gpt_oss_20b_boundary_v1")
FORMAT_VERSION = 1


def _log(message: str) -> None:
    print(f"[gpt-oss-full] {message}", file=sys.stderr, flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _config_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_corpus_definition(
    *,
    trajectory_dir: Path,
    sentence_boundaries_path: Path,
) -> tuple[list[PilotState], dict[str, list[dict[str, Any]]], dict[str, list[PilotState]]]:
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with sentence_boundaries_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped_rows[row["trace_id"]].append(row)

    states_by_trajectory: dict[str, list[PilotState]] = defaultdict(list)
    for trace_id, rows in grouped_rows.items():
        rows.sort(key=lambda row: int(row["sentence_id"]))
        if any(int(row["sentence_id"]) != index for index, row in enumerate(rows)):
            raise ValueError(f"Non-contiguous sentence IDs for {trace_id}.")
        step_index = int(rows[0]["step_id"])
        suffix = f"_step_{step_index:03d}"
        if not trace_id.endswith(suffix):
            raise ValueError(f"Trace ID does not agree with step_id: {trace_id}.")
        trajectory_id = trace_id[: -len(suffix)]
        if any(int(row["step_id"]) != step_index for row in rows):
            raise ValueError(f"Mixed environment steps in {trace_id}.")
        source_path = trajectory_dir / f"{trajectory_id}.json"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        states_by_trajectory[trajectory_id].append(
            PilotState(
                length_class="full_corpus",
                trajectory_id=trajectory_id,
                step_index=step_index,
                expected_reasoning_tokens=-1,
                expected_sentences=len(rows),
            )
        )

    specs: list[PilotState] = []
    for trajectory_id in sorted(states_by_trajectory):
        source_path = trajectory_dir / f"{trajectory_id}.json"
        payload = json.loads(source_path.read_text())
        resolved: list[PilotState] = []
        for state in sorted(states_by_trajectory[trajectory_id], key=lambda item: item.step_index):
            step = _payload_step(payload, state.step_index)
            reasoning_tokens = sum(
                "analysis" in token.get("token_groups", []) for token in step["output_tokens"]
            )
            resolved.append(
                PilotState(
                    length_class=state.length_class,
                    trajectory_id=state.trajectory_id,
                    step_index=state.step_index,
                    expected_reasoning_tokens=reasoning_tokens,
                    expected_sentences=state.expected_sentences,
                )
            )
        states_by_trajectory[trajectory_id] = resolved
        specs.extend(resolved)

    if len(specs) != FULL_CORPUS_ENVIRONMENT_STATES:
        raise ValueError(
            f"Expected {FULL_CORPUS_ENVIRONMENT_STATES} states, found {len(specs)}."
        )
    if sum(spec.expected_sentences for spec in specs) != FULL_CORPUS_SENTENCES:
        raise ValueError("Canonical sentence count differs from the calibrated corpus.")
    if sum(spec.expected_reasoning_tokens for spec in specs) != FULL_CORPUS_REASONING_TOKENS:
        raise ValueError("Reasoning-token count differs from the calibrated corpus.")
    return specs, dict(grouped_rows), dict(states_by_trajectory)


def _build_run_config(
    *,
    model_snapshot: Path,
    trajectory_dir: Path,
    sentence_boundaries_path: Path,
    layers: Sequence[int],
    forward_chunk_size: int,
) -> dict[str, Any]:
    trajectory_files = sorted(trajectory_dir.glob("*.json"))
    corpus_digest = hashlib.sha256()
    for path in trajectory_files:
        corpus_digest.update(path.name.encode())
        corpus_digest.update(bytes.fromhex(_sha256_file(path)))
    config = {
        "format_version": FORMAT_VERSION,
        "model_name": "openai/gpt-oss-20b",
        "model_snapshot": str(model_snapshot.resolve()),
        "model_revision": model_snapshot.name,
        "layers": sorted({int(layer) for layer in layers}),
        "forward_chunk_size": int(forward_chunk_size),
        "activation_dtype": "bfloat16",
        "prompt_mode": "original_trace_stored_token_ids",
        "representations": [
            "sentence_mean",
            "sentence_final",
            "reasoning_mean",
            "reasoning_final",
            "pre_window",
            "post_window",
            "action_token",
        ],
        "trajectory_files": len(trajectory_files),
        "trajectory_corpus_sha256": corpus_digest.hexdigest(),
        "sentence_boundaries_path": str(sentence_boundaries_path.resolve()),
        "sentence_boundaries_sha256": _sha256_file(sentence_boundaries_path),
        "expected_states": FULL_CORPUS_ENVIRONMENT_STATES,
        "expected_sentences": FULL_CORPUS_SENTENCES,
        "expected_reasoning_tokens": FULL_CORPUS_REASONING_TOKENS,
        "seed": 42,
        "attention_extracted": False,
    }
    config["config_fingerprint"] = _config_fingerprint(config)
    return config


def _expected_tensor_keys(layers: Sequence[int]) -> set[str]:
    groups = {
        "sentence_mean",
        "sentence_final",
        "reasoning_mean",
        "reasoning_final",
        "pre_window",
        "post_window",
        "action_token",
    }
    return {f"layer_{int(layer)}.{group}" for layer in layers for group in groups}


def _completed_record_is_valid(
    record_path: Path,
    *,
    spec: PilotState,
    layers: Sequence[int],
    config_fingerprint: str,
) -> bool:
    if not record_path.exists():
        return False
    try:
        record = json.loads(record_path.read_text())
        shard_path = Path(record["shard_path"])
        if record["config_fingerprint"] != config_fingerprint or not shard_path.exists():
            return False
        if shard_path.stat().st_size != int(record["file_size_bytes"]):
            return False
        if _sha256_file(shard_path) != record["shard_sha256"]:
            return False
        expected_keys = _expected_tensor_keys(layers)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != expected_keys:
                return False
            metadata = handle.metadata() or {}
            if metadata.get("trace_id") != spec.trace_id:
                return False
            for layer in layers:
                if tuple(handle.get_slice(f"layer_{layer}.sentence_mean").get_shape())[0] != spec.expected_sentences:
                    return False
                if tuple(handle.get_slice(f"layer_{layer}.sentence_final").get_shape())[0] != spec.expected_sentences:
                    return False
        return True
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def _load_records(record_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted(record_dir.glob("*.json"))]


def _prepared_from_record(
    spec: PilotState,
    rows: list[dict[str, Any]],
    record: dict[str, Any],
) -> PreparedState:
    prompt_length = int(record["prompt_length"])
    return PreparedState(
        spec=spec,
        source_path=Path(record["source_path"]),
        input_ids=[],
        sentence_rows=rows,
        sentence_spans=[
            (
                prompt_length + int(row["output_token_start"]),
                prompt_length + int(row["output_token_end_exclusive"]),
            )
            for row in rows
        ],
        reasoning_span=tuple(record["reasoning_span"]),
        pre_positions=list(record["pre_positions"]),
        post_positions=list(record["post_positions"]),
        action_position=int(record["action_position"]),
        tokenizer_validation_seconds=float(record["tokenizer_validation_seconds"]),
        token_ids_match=bool(record["token_ids_match"]),
        final_action=str(record["final_action"]),
    )


def _write_partial_outputs(
    *,
    output_dir: Path,
    records: list[dict[str, Any]],
) -> None:
    _write_csv(output_dir / "timing_rows.partial.csv", records)


def _build_report(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# GPT-OSS-20B Full Boundary-Activation Extraction",
            "",
            f"- Status: {manifest['status']}",
            f"- Model revision: `{manifest['model_revision']}`",
            f"- Trajectory files: {manifest['trajectory_files']:,}",
            f"- Environment states: {manifest['completed_states']:,}",
            f"- Reasoning sentences: {manifest['completed_sentences']:,}",
            f"- Reasoning tokens: {manifest['completed_reasoning_tokens']:,}",
            f"- Layers: {', '.join(map(str, manifest['layers']))}",
            f"- Forward chunk size: {manifest['forward_chunk_size']}",
            f"- Wall-clock time: {manifest['wall_seconds'] / 3600:.2f} hours",
            f"- Model loading time: {manifest['model_load_seconds']:.2f} seconds",
            f"- Forward time: {manifest['forward_seconds'] / 3600:.2f} hours",
            f"- Mean reasoning-token throughput: {manifest['reasoning_tokens_per_forward_second']:.1f} tokens/second",
            f"- Peak reserved VRAM: {manifest['peak_cuda_reserved_bytes'] / 2**30:.2f} GiB",
            f"- Peak process RAM: {manifest['peak_rss_bytes'] / 2**30:.2f} GiB",
            f"- Packed activation storage: {manifest['filesystem_bytes'] / 1e9:.2f} GB",
            "",
            "Each state shard contains BF16 sentence means, sentence-final activations, "
            "complete-reasoning means and finals, PRE and POST three-token windows, and the "
            "final-action-token activation at layers 8, 15, and 23.",
            "",
            "Attention is not included in this dataset. The separate pilot benchmark measured "
            "final-action attention to preceding sentences without adding it to the primary extraction.",
            "",
        ]
    )


def validate_gpt_oss_activation_full(
    output_dir: str | Path = DEFAULT_FULL_OUTPUT_DIR,
) -> dict[str, Any]:
    """Verify every shard, tensor shape, checksum, and activation-index row."""
    import pandas as pd

    output = Path(output_dir)
    manifest = json.loads((output / "run_manifest.json").read_text())
    config = json.loads((output / "run_config.json").read_text())
    layers = tuple(int(layer) for layer in config["layers"])
    records = _load_records(output / "state_records")
    if manifest["status"] != "completed":
        raise ValueError(f"Cannot validate incomplete run with status={manifest['status']}.")
    if len(records) != int(config["expected_states"]):
        raise ValueError("State-record count does not match run configuration.")
    shards = sorted((output / "shards").glob("*.safetensors"))
    if len(shards) != len(records):
        raise ValueError("Shard count does not match state-record count.")
    if list(output.rglob("*.partial")) or list(output.rglob("*.tmp")):
        raise ValueError("Temporary extraction files remain in the completed output.")

    checksum_lines: list[str] = []
    for record in records:
        shard_path = Path(record["shard_path"])
        if shard_path.stat().st_size != int(record["file_size_bytes"]):
            raise ValueError(f"Shard size mismatch: {shard_path}")
        digest = _sha256_file(shard_path)
        if digest != record["shard_sha256"]:
            raise ValueError(f"Shard checksum mismatch: {shard_path}")
        checksum_lines.append(f"{digest}  shards/{shard_path.name}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != _expected_tensor_keys(layers):
                raise ValueError(f"Unexpected tensor keys: {shard_path}")
            if (handle.metadata() or {}).get("trace_id") != record["trace_id"]:
                raise ValueError(f"Shard metadata mismatch: {shard_path}")
            hidden_sizes: set[int] = set()
            for layer in layers:
                for group in ("sentence_mean", "sentence_final"):
                    shape = tuple(handle.get_slice(f"layer_{layer}.{group}").get_shape())
                    if shape[0] != int(record["sentences"]):
                        raise ValueError(f"Sentence tensor shape mismatch: {shard_path}")
                    hidden_sizes.add(shape[1])
                for group in ("pre_window", "post_window"):
                    shape = tuple(handle.get_slice(f"layer_{layer}.{group}").get_shape())
                    if shape[0] != 3:
                        raise ValueError(f"Boundary-window shape mismatch: {shard_path}")
                    hidden_sizes.add(shape[1])
            if len(hidden_sizes) != 1:
                raise ValueError(f"Inconsistent hidden dimensions: {shard_path}")

    index = pd.read_parquet(output / "activation_index.parquet")
    expected_index_rows = (int(config["expected_sentences"]) + 5 * len(records)) * len(layers)
    if len(index) != expected_index_rows:
        raise ValueError("Activation-index row count is incorrect.")
    sentence_rows = index[index["record_kind"] == "sentence"]
    if len(sentence_rows) != int(config["expected_sentences"]) * len(layers):
        raise ValueError("Sentence index coverage is incorrect.")
    if sentence_rows.duplicated(["trace_id", "sentence_id", "layer"]).any():
        raise ValueError("Activation index contains duplicate sentence-layer rows.")
    if not (sentence_rows["n_tokens"] > 0).all():
        raise ValueError("Activation index contains an empty sentence token span.")

    (output / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")
    report = {
        "status": "passed",
        "validated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "states": len(records),
        "shards": len(shards),
        "verified_sha256_checksums": len(checksum_lines),
        "activation_index_rows": len(index),
        "sentence_layer_rows": len(sentence_rows),
        "reasoning_sentences": sum(int(record["sentences"]) for record in records),
        "reasoning_tokens": sum(int(record["reasoning_tokens"]) for record in records),
        "filesystem_bytes": sum(int(record["filesystem_bytes"]) for record in records),
        "layers": list(layers),
        "tokenizer_ids_all_match": all(record["token_ids_match"] for record in records),
        "temporary_files": 0,
    }
    _atomic_write_json(output / "validation_report.json", report)
    return report


def run_gpt_oss_activation_full(
    *,
    output_dir: str | Path = DEFAULT_FULL_OUTPUT_DIR,
    model_snapshot: str | Path = DEFAULT_MODEL_SNAPSHOT,
    trajectory_dir: str | Path = DEFAULT_TRAJECTORY_DIR,
    sentence_boundaries_path: str | Path = DEFAULT_SENTENCE_BOUNDARIES,
    layers: Sequence[int] = DEFAULT_LAYERS,
    forward_chunk_size: int = 512,
    resume: bool = True,
    minimum_free_bytes: int = 8_000_000_000,
    limit: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output_dir)
    shard_dir = output / "shards"
    record_dir = output / "state_records"
    output.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(exist_ok=True)
    record_dir.mkdir(exist_ok=True)
    model_path = Path(model_snapshot)
    trajectory_path = Path(trajectory_dir)
    boundary_path = Path(sentence_boundaries_path)

    _log("loading and validating the canonical corpus definition")
    specs, boundary_rows, states_by_trajectory = _load_corpus_definition(
        trajectory_dir=trajectory_path,
        sentence_boundaries_path=boundary_path,
    )
    run_config = _build_run_config(
        model_snapshot=model_path,
        trajectory_dir=trajectory_path,
        sentence_boundaries_path=boundary_path,
        layers=layers,
        forward_chunk_size=forward_chunk_size,
    )
    config_path = output / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing != run_config:
            raise ValueError(
                "Existing output configuration does not match this request. Use a new output directory."
            )
    else:
        _atomic_write_json(config_path, run_config)

    free_bytes = shutil.disk_usage(output).free
    if free_bytes < minimum_free_bytes:
        raise RuntimeError(
            f"Only {free_bytes / 1e9:.2f} GB free; require at least "
            f"{minimum_free_bytes / 1e9:.2f} GB before extraction."
        )

    specs_by_trace = {spec.trace_id: spec for spec in specs}
    completed: set[str] = set()
    if resume:
        _log("validating completed shards before resume")
        for spec in specs:
            record_path = record_dir / f"{spec.trace_id}.json"
            if _completed_record_is_valid(
                record_path,
                spec=spec,
                layers=layers,
                config_fingerprint=run_config["config_fingerprint"],
            ):
                completed.add(spec.trace_id)
    elif any(record_dir.glob("*.json")) or any(shard_dir.glob("*.safetensors")):
        raise FileExistsError("Output contains prior extraction artifacts and resume is disabled.")

    pending = [spec for spec in specs if spec.trace_id not in completed]
    if limit is not None:
        pending = pending[: int(limit)]
    _log(f"resume found {len(completed):,} valid states; {len(pending):,} states pending")

    manifest: dict[str, Any] = {
        **run_config,
        "status": "running",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "free_bytes_at_start": free_bytes,
        "completed_states": len(completed),
        "pending_states": len(pending),
        "resumed_states": len(completed),
        "limit": limit,
    }
    _atomic_write_json(output / "run_manifest.json", manifest)

    tokenizer_load_seconds = 0.0
    model_load_seconds = 0.0
    load_peak_rss = 0
    model = None
    try:
        if pending:
            before = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            tokenizer_load_seconds = time.perf_counter() - before
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            _log("loading openai/gpt-oss-20b on CUDA")
            before = time.perf_counter()
            with PeakResourceMonitor() as load_monitor:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    dtype=torch.bfloat16,
                    device_map={"": 0},
                    low_cpu_mem_usage=True,
                ).eval()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_load_seconds = time.perf_counter() - before
            load_peak_rss = load_monitor.peak_rss_bytes
            input_device = model.model.embed_tokens.weight.device
            collector = BoundaryAggregateCollector(
                model,
                layers=layers,
                input_device=input_device,
            )

            pending_set = {spec.trace_id for spec in pending}
            extracted_this_run = 0
            for trajectory_number, trajectory_id in enumerate(sorted(states_by_trajectory), start=1):
                trajectory_states = [
                    spec
                    for spec in states_by_trajectory[trajectory_id]
                    if spec.trace_id in pending_set
                ]
                if not trajectory_states:
                    continue
                payload = json.loads((trajectory_path / f"{trajectory_id}.json").read_text())
                for spec in trajectory_states:
                    state_number = len(completed) + extracted_this_run + 1
                    _log(
                        f"extracting state {state_number}/{len(specs)}: {spec.trace_id} "
                        f"({spec.expected_reasoning_tokens:,} reasoning tokens)"
                    )
                    prepared = _prepare_state(
                        spec,
                        trajectory_dir=trajectory_path,
                        boundary_rows=boundary_rows[spec.trace_id],
                        tokenizer=tokenizer,
                        payload=payload,
                    )
                    tensors, metrics = collector.run(
                        prepared,
                        forward_chunk_size=forward_chunk_size,
                    )
                    shard_path = shard_dir / f"{spec.trace_id}.safetensors"
                    temporary_shard = shard_path.with_suffix(".safetensors.partial")
                    serialization_start = time.perf_counter()
                    save_file(
                        tensors,
                        str(temporary_shard),
                        metadata={
                            "model_revision": model_path.name,
                            "trace_id": spec.trace_id,
                            "activation_dtype": "bfloat16",
                            "config_fingerprint": run_config["config_fingerprint"],
                        },
                    )
                    os.replace(temporary_shard, shard_path)
                    serialization_seconds = time.perf_counter() - serialization_start
                    record = {
                        "config_fingerprint": run_config["config_fingerprint"],
                        "trace_id": spec.trace_id,
                        "trajectory_id": spec.trajectory_id,
                        "step_index": spec.step_index,
                        "source_path": str(prepared.source_path),
                        "prompt_length": prepared.reasoning_span[0]
                        - min(int(row["output_token_start"]) for row in prepared.sentence_rows),
                        "reasoning_span": list(prepared.reasoning_span),
                        "pre_positions": prepared.pre_positions,
                        "post_positions": prepared.post_positions,
                        "action_position": prepared.action_position,
                        "final_action": prepared.final_action,
                        "tokenizer_validation_seconds": prepared.tokenizer_validation_seconds,
                        "token_ids_match": prepared.token_ids_match,
                        "reasoning_tokens": spec.expected_reasoning_tokens,
                        "sentences": spec.expected_sentences,
                        "serialization_seconds": serialization_seconds,
                        "logical_tensor_bytes": _tensor_logical_bytes(tensors),
                        "file_size_bytes": shard_path.stat().st_size,
                        "filesystem_bytes": _filesystem_bytes(shard_path),
                        "shard_path": str(shard_path),
                        "shard_sha256": _sha256_file(shard_path),
                        **metrics,
                    }
                    _atomic_write_json(record_dir / f"{spec.trace_id}.json", record)
                    del tensors
                    extracted_this_run += 1
                    manifest.update(
                        {
                            "completed_states": len(completed) + extracted_this_run,
                            "pending_states": len(pending) - extracted_this_run,
                            "last_completed_trace_id": spec.trace_id,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    )
                    _atomic_write_json(output / "run_manifest.json", manifest)
                    if extracted_this_run % 25 == 0:
                        _write_partial_outputs(
                            output_dir=output,
                            records=_load_records(record_dir),
                        )
                del payload

        records = _load_records(record_dir)
        completed_trace_ids = {record["trace_id"] for record in records}
        expected_for_completion = set(specs_by_trace)
        complete = completed_trace_ids == expected_for_completion
        if limit is None and not complete:
            missing = sorted(expected_for_completion - completed_trace_ids)
            raise RuntimeError(f"Full extraction ended with {len(missing)} missing states.")

        records.sort(key=lambda row: row["trace_id"])
        _write_csv(output / "timing_rows.csv", records)
        index_rows: list[dict[str, Any]] = []
        records_by_trace = {record["trace_id"]: record for record in records}
        for spec in specs:
            if spec.trace_id not in records_by_trace:
                continue
            prepared = _prepared_from_record(
                spec,
                boundary_rows[spec.trace_id],
                records_by_trace[spec.trace_id],
            )
            index_rows.extend(
                _index_rows(
                    prepared,
                    layers=layers,
                    shard_path=Path(records_by_trace[spec.trace_id]["shard_path"]),
                )
            )
        _write_parquet(output / "activation_index.parquet", index_rows)

        total_forward = sum(float(record["forward_seconds"]) for record in records)
        total_tokens = sum(int(record["reasoning_tokens"]) for record in records)
        manifest.update(
            {
                "status": "completed" if complete else "limited_completed",
                "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "completed_states": len(records),
                "completed_sentences": sum(int(record["sentences"]) for record in records),
                "completed_reasoning_tokens": total_tokens,
                "activation_index_rows": len(index_rows),
                "pending_states": len(specs) - len(records),
                "tokenizer_load_seconds": tokenizer_load_seconds,
                "model_load_seconds": model_load_seconds,
                "model_load_peak_rss_bytes": load_peak_rss,
                "forward_seconds": total_forward,
                "aggregate_seconds": sum(float(record["aggregate_seconds"]) for record in records),
                "serialization_seconds": sum(
                    float(record["serialization_seconds"]) for record in records
                ),
                "reasoning_tokens_per_forward_second": total_tokens / total_forward,
                "peak_cuda_allocated_bytes": max(
                    int(record["peak_cuda_allocated_bytes"]) for record in records
                ),
                "peak_cuda_reserved_bytes": max(
                    int(record["peak_cuda_reserved_bytes"]) for record in records
                ),
                "peak_rss_bytes": max(
                    [load_peak_rss, *(int(record["peak_rss_bytes"]) for record in records)]
                ),
                "logical_tensor_bytes": sum(
                    int(record["logical_tensor_bytes"]) for record in records
                ),
                "file_size_bytes": sum(int(record["file_size_bytes"]) for record in records),
                "filesystem_bytes": sum(int(record["filesystem_bytes"]) for record in records),
                "wall_seconds": time.perf_counter() - started,
                "all_tokenizer_ids_match": all(record["token_ids_match"] for record in records),
                "attention_extracted": False,
            }
        )
        _atomic_write_json(output / "run_manifest.json", manifest)
        (output / "run_report.md").write_text(_build_report(manifest))
        partial_path = output / "timing_rows.partial.csv"
        if partial_path.exists():
            partial_path.unlink()
        _log(f"full extraction completed: {len(records):,} states")
        return manifest
    except BaseException as exc:
        manifest.update(
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
                "completed_states": len(list(record_dir.glob("*.json"))),
            }
        )
        _atomic_write_json(output / "run_manifest.json", manifest)
        raise
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "DEFAULT_FULL_OUTPUT_DIR",
    "run_gpt_oss_activation_full",
    "validate_gpt_oss_activation_full",
    "_completed_record_is_valid",
    "_load_corpus_definition",
]
