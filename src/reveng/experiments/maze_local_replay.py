"""Deterministically teacher-force saved maze completions through a local model.

The Together API currently returns token pieces and bytes, but not necessarily token
IDs.  This module reconstructs IDs with the exact checkpoint tokenizer, validates
the reconstruction before loading model weights, and captures token-aligned decoder
activations while replaying the saved sequence (it never samples new tokens).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from reveng.experiments.maze_local_preflight import (
    _chat_inputs,
    _model_input_device,
    _version_tuple,
    locate_decoder_layers,
    reasoning_token_indices,
)
from reveng.experiments.maze_smoke_test import read_jsonl


@dataclass(frozen=True)
class PreparedReplay:
    record_id: str
    row: dict[str, Any]
    prompt_ids: list[int]
    completion_ids: list[int]
    reasoning_indices: list[int]
    reconstruction_source: str
    boundary_validation: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _record_id(row: dict[str, Any]) -> str:
    return f"{row['trajectory_id']}__step_{int(row['step_index']):03d}"


def _tokenizer_input_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, dict):
        values = encoded["input_ids"]
    else:
        values = encoded.input_ids
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _decode_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    kwargs = {
        "skip_special_tokens": False,
        "clean_up_tokenization_spaces": False,
    }
    try:
        return str(tokenizer.decode(list(token_ids), **kwargs))
    except TypeError:
        kwargs.pop("clean_up_tokenization_spaces")
        return str(tokenizer.decode(list(token_ids), **kwargs))


def _provider_bytes(row: dict[str, Any]) -> tuple[list[bytes], bytes]:
    pieces = list(row.get("generated_tokens", []) or [])
    raw_bytes = list(row.get("generated_token_bytes", []) or [])
    expected = int(row.get("output_tokens", 0))
    if expected <= 0:
        raise ValueError("output_tokens must be positive")
    if len(pieces) != expected:
        raise ValueError(
            f"provider token-piece count mismatch: expected={expected}, got={len(pieces)}"
        )
    if raw_bytes and len(raw_bytes) != expected:
        raise ValueError(
            f"provider token-byte count mismatch: expected={expected}, got={len(raw_bytes)}"
        )
    if raw_bytes:
        per_token = [bytes(int(value) for value in values) for values in raw_bytes]
    else:
        per_token = [str(piece).encode("utf-8") for piece in pieces]
    return per_token, b"".join(per_token)


def _prefix_boundary_validation(
    tokenizer: Any,
    token_ids: Sequence[int],
    provider_token_bytes: Sequence[bytes],
) -> str:
    """Validate provider boundaries when the fast tokenizer exposes offsets.

    Full byte round-trip and token count are always mandatory. Offset mappings are
    an additional boundary check; tokenizers that do not expose reliable offsets
    are reported explicitly rather than being treated as if IDs came from the API.
    """
    joined = b"".join(provider_token_bytes)
    try:
        text = joined.decode("utf-8", errors="strict")
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping") if hasattr(encoded, "get") else None
        if offsets and offsets and isinstance(offsets[0], list):
            offsets = offsets[0]
        encoded_ids = encoded.get("input_ids") if hasattr(encoded, "get") else None
        if encoded_ids and isinstance(encoded_ids[0], list):
            encoded_ids = encoded_ids[0]
        if not offsets or list(map(int, encoded_ids or [])) != list(
            map(int, token_ids)
        ):
            return "full_byte_roundtrip_and_count"
        provider_ends: list[int] = []
        cursor = 0
        for value in provider_token_bytes:
            cursor += len(value)
            provider_ends.append(cursor)
        char_to_byte = [
            len(text[:index].encode("utf-8")) for index in range(len(text) + 1)
        ]
        local_ends = [char_to_byte[int(end)] for _start, end in offsets]
        # Added/special tokens commonly report (0, 0). Their literal bytes are
        # already protected by the mandatory full round-trip, so compare every
        # meaningful offset and require the final boundary.
        comparable = [
            index for index, (_start, end) in enumerate(offsets) if int(end) > 0
        ]
        if comparable and all(
            local_ends[index] == provider_ends[index] for index in comparable
        ):
            return "provider_boundaries_match_local_offsets"
    except (KeyError, TypeError, UnicodeDecodeError, ValueError):
        pass
    return "full_byte_roundtrip_and_count"


def reconstruct_completion_ids(
    row: dict[str, Any], tokenizer: Any
) -> tuple[list[int], str, str]:
    """Return checkpoint IDs and the strength/source of their validation."""
    per_token_bytes, joined = _provider_bytes(row)
    stored_ids = [int(value) for value in row.get("generated_token_ids", []) or []]
    if stored_ids:
        ids = stored_ids
        source = "provider_token_ids"
    else:
        try:
            text = joined.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "Completion bytes are not valid UTF-8 and no provider token IDs were saved."
            ) from exc
        ids = _tokenizer_input_ids(tokenizer, text)
        source = "reconstructed_exact_checkpoint_tokenizer"

    expected = int(row["output_tokens"])
    if len(ids) != expected:
        raise ValueError(
            "Local tokenizer does not reproduce the provider token count: "
            f"provider={expected}, local={len(ids)}"
        )
    decoded = _decode_ids(tokenizer, ids).encode("utf-8")
    if decoded != joined:
        raise ValueError(
            "Local tokenizer IDs do not round-trip to the exact provider completion bytes."
        )
    boundary = _prefix_boundary_validation(tokenizer, ids, per_token_bytes)
    return ids, source, boundary


def validate_static_record(row: dict[str, Any], *, model_name: str) -> None:
    required = {
        "trajectory_id",
        "step_index",
        "model",
        "reasoning_setting",
        "prompt_text",
        "prompt_sha256",
        "prompt_tokens",
        "output_tokens",
        "generated_tokens",
        "token_sequence_complete",
    }
    missing = sorted(required.difference(row))
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")
    if str(row["model"]) != model_name:
        raise ValueError(f"record model {row['model']!r} does not match {model_name!r}")
    if not row.get("api_success"):
        raise ValueError("API call was not successful")
    if not row.get("token_sequence_complete"):
        raise ValueError("provider token sequence is incomplete")
    prompt = str(row["prompt_text"])
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != str(row["prompt_sha256"]):
        raise ValueError("prompt_sha256 does not match prompt_text")
    _provider_bytes(row)


def prepare_replay_record(
    row: dict[str, Any],
    *,
    processor: Any,
    tokenizer: Any,
    model_config: dict[str, Any],
) -> PreparedReplay:
    model_name = str(model_config["name"])
    validate_static_record(row, model_name=model_name)
    prompt_inputs = _chat_inputs(
        processor,
        tokenizer,
        str(row["prompt_text"]),
        str(row["reasoning_setting"]),
        model_config,
    )
    prompt_values = prompt_inputs["input_ids"]
    if hasattr(prompt_values, "detach"):
        prompt_values = prompt_values.detach().cpu().tolist()
    if prompt_values and isinstance(prompt_values[0], list):
        prompt_values = prompt_values[0]
    prompt_ids = [int(value) for value in prompt_values]
    provider_prompt_tokens = int(row["prompt_tokens"])
    if len(prompt_ids) != provider_prompt_tokens:
        raise ValueError(
            "Local chat template does not reproduce the provider prompt token count: "
            f"provider={provider_prompt_tokens}, local={len(prompt_ids)}"
        )
    completion_ids, source, boundary = reconstruct_completion_ids(row, tokenizer)
    reasoning = reasoning_token_indices(
        [str(value) for value in row["generated_tokens"]],
        str(model_config["reasoning_trace_format"]),
    )
    if not reasoning:
        raise ValueError(
            "No reasoning token positions were found in the saved completion."
        )
    return PreparedReplay(
        record_id=_record_id(row),
        row=row,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        reasoning_indices=reasoning,
        reconstruction_source=source,
        boundary_validation=boundary,
    )


def capture_replay_activations(
    model: Any,
    prepared: PreparedReplay,
    *,
    layers: Sequence[int],
    forward_chunk_size: int,
    capture_scope: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Teacher-force one saved sequence and return CPU activation tensors."""
    import torch

    if forward_chunk_size <= 0:
        raise ValueError("forward_chunk_size must be positive")
    decoder_layers = locate_decoder_layers(model)
    selected_layers = sorted({int(layer) for layer in layers})
    if not selected_layers:
        raise ValueError("At least one layer is required")
    for layer in selected_layers:
        if layer < 0 or layer >= len(decoder_layers):
            raise ValueError(
                f"Layer {layer} is outside model depth {len(decoder_layers)}"
            )

    if capture_scope == "completion":
        relative_positions = list(range(len(prepared.completion_ids)))
    elif capture_scope == "reasoning":
        relative_positions = list(prepared.reasoning_indices)
    else:
        raise ValueError("capture_scope must be 'reasoning' or 'completion'")
    prompt_length = len(prepared.prompt_ids)
    selected_global = [prompt_length + value for value in relative_positions]
    all_ids = prepared.prompt_ids + prepared.completion_ids
    input_device = _model_input_device(model)
    input_ids = torch.tensor([all_ids], dtype=torch.long, device=input_device)
    parts: dict[int, list[torch.Tensor]] = {layer: [] for layer in selected_layers}
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def hook_for(layer: int):
        def hook(_module: Any, _args: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden.detach()[0]

        return hook

    for layer in selected_layers:
        handles.append(decoder_layers[layer].register_forward_hook(hook_for(layer)))

    past_key_values = None
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for chunk_start in range(0, len(all_ids), int(forward_chunk_size)):
                chunk_end = min(len(all_ids), chunk_start + int(forward_chunk_size))
                captured.clear()
                outputs = model(
                    input_ids=input_ids[:, chunk_start:chunk_end],
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                if set(captured) != set(selected_layers):
                    raise ValueError(
                        "Forward hooks did not capture every requested layer"
                    )
                positions = [
                    value
                    for value in selected_global
                    if chunk_start <= value < chunk_end
                ]
                if positions:
                    local = torch.tensor(
                        [value - chunk_start for value in positions],
                        dtype=torch.long,
                        device=captured[selected_layers[0]].device,
                    )
                    for layer in selected_layers:
                        layer_local = local.to(captured[layer].device)
                        parts[layer].append(
                            captured[layer]
                            .index_select(0, layer_local)
                            .to(device="cpu", dtype=torch.bfloat16)
                        )
                del outputs
    finally:
        for handle in handles:
            handle.remove()
        del past_key_values
        del input_ids

    tensors: dict[str, Any] = {
        "generated_token_ids": torch.tensor(prepared.completion_ids, dtype=torch.int64),
        "captured_completion_positions": torch.tensor(
            relative_positions, dtype=torch.int64
        ),
        "reasoning_token_indices": torch.tensor(
            prepared.reasoning_indices, dtype=torch.int64
        ),
    }
    for layer in selected_layers:
        if not parts[layer]:
            raise ValueError(f"No selected activations were captured at layer {layer}")
        value = torch.cat(parts[layer], dim=0).contiguous()
        if int(value.shape[0]) != len(relative_positions):
            raise ValueError(f"Activation count mismatch at layer {layer}")
        if not bool(torch.isfinite(value.float()).all().item()):
            raise ValueError(f"NaN or Inf detected at layer {layer}")
        tensors[f"layer_{layer}"] = value
    metrics = {
        "input_tokens": len(all_ids),
        "prompt_tokens": prompt_length,
        "completion_tokens": len(prepared.completion_ids),
        "captured_tokens": len(relative_positions),
        "layers": selected_layers,
        "forward_chunk_size": int(forward_chunk_size),
        "forward_seconds": time.perf_counter() - started,
        "capture_scope": capture_scope,
    }
    return tensors, metrics


def _load_processor_tokenizer(
    model_config: dict[str, Any], *, local_files_only: bool
) -> tuple[Any, Any]:
    import transformers

    minimum = str(model_config.get("minimum_transformers_version", ""))
    if minimum and _version_tuple(transformers.__version__) < _version_tuple(minimum):
        raise RuntimeError(
            f"{model_config['name']} requires transformers>={minimum}; "
            f"installed={transformers.__version__}"
        )
    checkpoint = str(model_config["local_checkpoint"])
    common = {
        "revision": model_config.get("revision"),
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        "local_files_only": local_files_only,
    }
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint, **common)
    return tokenizer, tokenizer


def _load_model(model_config: dict[str, Any], *, local_files_only: bool) -> Any:
    import torch
    import transformers

    class_name = str(model_config.get("local_model_class", "AutoModelForCausalLM"))
    model_class = getattr(transformers, class_name, None)
    if model_class is None:
        raise RuntimeError(
            f"Installed transformers {transformers.__version__} lacks {class_name}"
        )
    checkpoint = str(model_config["local_checkpoint"])
    dtype = getattr(torch, str(model_config.get("local_dtype", "bfloat16")))
    kwargs = {
        "revision": model_config.get("revision"),
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        "local_files_only": local_files_only,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    try:
        model = model_class.from_pretrained(checkpoint, dtype=dtype, **kwargs)
    except TypeError:
        model = model_class.from_pretrained(checkpoint, torch_dtype=dtype, **kwargs)
    model.eval()
    return model


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _select_rows(
    rows: Iterable[dict[str, Any]],
    *,
    model_name: str,
    record_ids: set[str] | None,
    max_records: int | None,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row.get("model")) == model_name]
    if record_ids:
        selected = [row for row in selected if _record_id(row) in record_ids]
        missing = sorted(record_ids.difference(_record_id(row) for row in selected))
        if missing:
            raise ValueError(f"Requested record IDs not found: {', '.join(missing)}")
    selected.sort(key=lambda row: (str(row["trajectory_id"]), int(row["step_index"])))
    if max_records is not None:
        selected = selected[: int(max_records)]
    if not selected:
        raise ValueError(f"No raw API records selected for model {model_name!r}")
    return selected


def run_local_replay(
    *,
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    layers: Sequence[int] = (8, 15, 23),
    forward_chunk_size: int = 256,
    minimum_chunk_size: int = 32,
    capture_scope: str = "reasoning",
    max_records: int | None = None,
    record_ids: set[str] | None = None,
    local_files_only: bool = False,
    validate_only: bool = False,
) -> dict[str, Any]:
    config_path = input_dir / "config.lock.json"
    raw_path = input_dir / "raw_api_calls.jsonl"
    if not config_path.exists() or not raw_path.exists():
        raise FileNotFoundError(
            f"Expected config.lock.json and raw_api_calls.jsonl under {input_dir}"
        )
    config = json.loads(config_path.read_text())
    matches = [row for row in config.get("models", []) if row.get("name") == model_name]
    if len(matches) != 1:
        raise ValueError(
            f"Model {model_name!r} is not uniquely defined in config.lock.json"
        )
    model_config = dict(matches[0])
    rows = _select_rows(
        read_jsonl(raw_path),
        model_name=model_name,
        record_ids=record_ids,
        max_records=max_records,
    )
    processor, tokenizer = _load_processor_tokenizer(
        model_config, local_files_only=local_files_only
    )

    prepared_rows: list[PreparedReplay] = []
    validation_failures: list[dict[str, str]] = []
    for row in rows:
        try:
            prepared_rows.append(
                prepare_replay_record(
                    row,
                    processor=processor,
                    tokenizer=tokenizer,
                    model_config=model_config,
                )
            )
        except Exception as exc:
            validation_failures.append(
                {"record_id": _record_id(row), "error": f"{type(exc).__name__}: {exc}"}
            )
    validation = {
        "schema_version": 1,
        "model": model_name,
        "checkpoint": model_config["local_checkpoint"],
        "revision": model_config.get("revision"),
        "input_dir": str(input_dir),
        "raw_api_calls_sha256": _sha256_bytes(raw_path.read_bytes()),
        "selected_records": len(rows),
        "validated_records": len(prepared_rows),
        "reconstruction_sources": {
            value: sum(item.reconstruction_source == value for item in prepared_rows)
            for value in sorted({item.reconstruction_source for item in prepared_rows})
        },
        "boundary_validations": {
            value: sum(item.boundary_validation == value for item in prepared_rows)
            for value in sorted({item.boundary_validation for item in prepared_rows})
        },
        "failures": validation_failures,
        "status": "PASS" if not validation_failures else "FAIL",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    if validation_failures:
        first = validation_failures[0]
        raise ValueError(
            f"Replay validation failed for {len(validation_failures)} record(s); "
            f"first={first['record_id']}: {first['error']}"
        )
    if validate_only:
        return validation

    model = _load_model(model_config, local_files_only=local_files_only)
    manifest_path = output_dir / "replay_manifest.jsonl"
    existing = read_jsonl(manifest_path)
    completed = {
        str(row["record_id"])
        for row in existing
        if row.get("status") == "PASS"
        and Path(str(row.get("artifact_path", ""))).exists()
    }
    pass_count = len(completed.intersection(item.record_id for item in prepared_rows))
    failures: list[dict[str, Any]] = []
    artifact_dir = output_dir / "activations"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for index, prepared in enumerate(prepared_rows, start=1):
        if prepared.record_id in completed:
            print(
                f"[replay {index}/{len(prepared_rows)}] already complete: {prepared.record_id}"
            )
            continue
        print(f"[replay {index}/{len(prepared_rows)}] {prepared.record_id}", flush=True)
        chunk_size = int(forward_chunk_size)
        while True:
            try:
                tensors, metrics = capture_replay_activations(
                    model,
                    prepared,
                    layers=layers,
                    forward_chunk_size=chunk_size,
                    capture_scope=capture_scope,
                )
                break
            except RuntimeError as exc:
                import torch

                if (
                    "out of memory" not in str(exc).lower()
                    or chunk_size <= minimum_chunk_size
                ):
                    raise
                chunk_size = max(int(minimum_chunk_size), chunk_size // 2)
                torch.cuda.empty_cache()
                print(f"  CUDA OOM; retrying with chunk size {chunk_size}", flush=True)
        from safetensors.torch import save_file

        artifact_path = (
            artifact_dir / f"{_safe_filename(prepared.record_id)}.safetensors"
        )
        temporary_path = artifact_path.with_suffix(".safetensors.tmp")
        save_file(tensors, temporary_path)
        temporary_path.replace(artifact_path)
        row = {
            "schema_version": 1,
            "record_id": prepared.record_id,
            "trajectory_id": prepared.row["trajectory_id"],
            "step_index": int(prepared.row["step_index"]),
            "model": model_name,
            "checkpoint": model_config["local_checkpoint"],
            "revision": model_config.get("revision"),
            "prompt_sha256": prepared.row["prompt_sha256"],
            "completion_bytes_sha256": _sha256_bytes(_provider_bytes(prepared.row)[1]),
            "reconstruction_source": prepared.reconstruction_source,
            "boundary_validation": prepared.boundary_validation,
            "provider_logprobs_retained_in": str(raw_path),
            "artifact_path": str(artifact_path),
            "artifact_sha256": _sha256_bytes(artifact_path.read_bytes()),
            "status": "PASS",
            **metrics,
        }
        _append_jsonl(manifest_path, row)
        pass_count += 1

    summary = {
        **validation,
        "status": (
            "PASS" if not failures and pass_count == len(prepared_rows) else "FAIL"
        ),
        "replayed_records": pass_count,
        "replay_failures": failures,
        "layers": sorted({int(layer) for layer in layers}),
        "capture_scope": capture_scope,
        "manifest_path": str(manifest_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and teacher-force saved maze API completions locally."
    )
    parser.add_argument("stage", choices=("validate", "replay"))
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 15, 23])
    parser.add_argument("--forward-chunk-size", type=int, default=256)
    parser.add_argument("--minimum-chunk-size", type=int, default=32)
    parser.add_argument(
        "--capture-scope", choices=("reasoning", "completion"), default="reasoning"
    )
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--record-id", action="append")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    summary = run_local_replay(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        layers=args.layers,
        forward_chunk_size=args.forward_chunk_size,
        minimum_chunk_size=args.minimum_chunk_size,
        capture_scope=args.capture_scope,
        max_records=args.max_records,
        record_ids=set(args.record_id) if args.record_id else None,
        local_files_only=args.local_files_only,
        validate_only=args.stage == "validate",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
