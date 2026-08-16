"""Technical local-model preflight for the maze activation experiment.

This module is intentionally separate from the behavioral API smoke test.  It
loads one exact checkpoint at a time, generates one action, and verifies that
reasoning-token residual-stream activations can be captured at selected layers.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from reveng.experiments.maze_smoke_test import (
    PROMPT_TEMPLATE,
    parse_action,
    render_grid,
)


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"(\d+(?:\.\d+)*)", value)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def inspect_local_environment(
    configured_models: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Validate software/CUDA prerequisites without loading or downloading weights."""
    import torch
    import transformers

    model_checks = []
    for model in configured_models:
        class_name = str(model.get("local_model_class", "AutoModelForCausalLM"))
        minimum = str(model.get("minimum_transformers_version", ""))
        class_available = getattr(transformers, class_name, None) is not None
        version_ok = not minimum or _version_tuple(
            transformers.__version__
        ) >= _version_tuple(minimum)
        model_checks.append(
            {
                "model": str(model["name"]),
                "checkpoint": str(model["local_checkpoint"]),
                "model_class": class_name,
                "model_class_available": class_available,
                "minimum_transformers_version": minimum or "not specified",
                "transformers_version_ok": version_ok,
                "ready_without_loading_weights": class_available and version_ok,
            }
        )
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    return {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
        "models": model_checks,
        "ready_without_loading_weights": bool(cuda_devices)
        and all(row["ready_without_loading_weights"] for row in model_checks),
    }


def locate_decoder_layers(model: Any) -> Sequence[Any]:
    """Find the decoder block list without assuming one Transformers architecture."""
    candidates = (
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("layers",),
        ("model", "model", "layers"),
    )
    for path in candidates:
        value = model
        for name in path:
            value = getattr(value, name, None)
            if value is None:
                break
        if value is not None and hasattr(value, "__len__"):
            return value
    raise ValueError(
        "Could not locate decoder layers. Add this checkpoint's decoder path to "
        "locate_decoder_layers before collecting activations."
    )


def token_pieces(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    """Return per-token text increments consistent with full-prefix decoding."""
    pieces: list[str] = []
    previous = ""
    for index in range(len(token_ids)):
        current = tokenizer.decode(
            [int(token_id) for token_id in token_ids[: index + 1]],
            skip_special_tokens=False,
        )
        if not current.startswith(previous):
            # This can occur for an incomplete byte-fallback token. The preflight
            # should fail visibly rather than silently assign the wrong positions.
            raise ValueError(
                "Tokenizer prefix decoding changed earlier text; reasoning-token "
                "alignment is not reliable for this generated output."
            )
        pieces.append(current[len(previous) :])
        previous = current
    return pieces


def _indices_between(
    pieces: Sequence[str], start_marker: str, end_marker: str
) -> list[int]:
    text = "".join(pieces)
    start = text.find(start_marker)
    if start < 0:
        return []
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end < 0:
        return []
    indices: list[int] = []
    cursor = 0
    for index, piece in enumerate(pieces):
        piece_start, piece_end = cursor, cursor + len(piece)
        if piece_end > start and piece_start < end and piece.strip():
            indices.append(index)
        cursor = piece_end
    return indices


def reasoning_token_indices(pieces: Sequence[str], trace_format: str) -> list[int]:
    """Identify reasoning tokens using the checkpoint's declared output convention."""
    if trace_format == "gpt_oss_channels":
        return _indices_between(
            pieces,
            "<|channel|>analysis<|message|>",
            "<|end|><|start|>assistant<|channel|>final<|message|>",
        ) or _indices_between(pieces, "analysis<|message|>", "final<|message|>")
    if trace_format == "qwen_think_tags":
        return _indices_between(pieces, "<think>", "</think>")
    if trace_format == "before_action_json":
        text = "".join(pieces)
        match = re.search(r"\{\s*[\"']action[\"']\s*:", text, flags=re.IGNORECASE)
        if match is None:
            return []
        indices: list[int] = []
        cursor = 0
        for index, piece in enumerate(pieces):
            piece_end = cursor + len(piece)
            cleaned = re.sub(r"<[^>]+>", "", piece).strip()
            if cursor < match.start() and cleaned:
                indices.append(index)
            cursor = piece_end
        return indices
    raise ValueError(f"Unknown reasoning_trace_format: {trace_format}")


def validate_activation_tensor(tensor: Any, *, expected_tokens: int) -> dict[str, Any]:
    import torch

    if tensor.ndim != 2:
        raise ValueError(
            f"Expected [tokens, hidden] activation tensor, got {tuple(tensor.shape)}."
        )
    if int(tensor.shape[0]) != int(expected_tokens):
        raise ValueError(
            f"Expected {expected_tokens} reasoning-token activations, got {int(tensor.shape[0])}."
        )
    finite = bool(torch.isfinite(tensor.float()).all().item())
    return {
        "tokens": int(tensor.shape[0]),
        "hidden_size": int(tensor.shape[1]),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "finite": finite,
        "bytes": int(tensor.numel() * tensor.element_size()),
    }


def _load_processor_and_model(
    model_config: dict[str, Any], *, local_files_only: bool
) -> tuple[Any, Any, Any]:
    import torch
    import transformers

    checkpoint = str(model_config["local_checkpoint"])
    common = {
        "revision": model_config.get("revision"),
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        "local_files_only": local_files_only,
    }
    common = {key: value for key, value in common.items() if value is not None}
    class_name = str(model_config.get("local_model_class", "AutoModelForCausalLM"))
    model_class = getattr(transformers, class_name, None)
    if model_class is None:
        raise RuntimeError(
            f"Installed transformers {transformers.__version__} lacks {class_name}, required by {checkpoint}. "
            "Install a Transformers release supporting the checkpoint on the GPU machine."
        )
    if class_name in {"AutoModelForMultimodalLM", "AutoModelForImageTextToText"}:
        processor = transformers.AutoProcessor.from_pretrained(checkpoint, **common)
        tokenizer = getattr(processor, "tokenizer", processor)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint, **common)
        processor = tokenizer
    dtype_name = str(model_config.get("local_dtype", "bfloat16"))
    dtype = getattr(torch, dtype_name)
    load_kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        **common,
    }
    try:
        model = model_class.from_pretrained(checkpoint, dtype=dtype, **load_kwargs)
    except TypeError:
        model = model_class.from_pretrained(
            checkpoint, torch_dtype=dtype, **load_kwargs
        )
    model.eval()
    return processor, tokenizer, model


def _chat_inputs(
    processor: Any,
    tokenizer: Any,
    prompt: str,
    reasoning_setting: str,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    messages = [{"role": "user", "content": prompt}]
    template_kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    if reasoning_setting in {"low", "medium", "high"}:
        template_kwargs["reasoning_effort"] = reasoning_setting
    template_kwargs.update(model_config.get("chat_template_kwargs", {}))
    try:
        encoded = processor.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("reasoning_effort", None)
        encoded = processor.apply_chat_template(messages, **template_kwargs)
    if isinstance(encoded, torch.Tensor):
        return {"input_ids": encoded}
    if hasattr(encoded, "items"):
        return dict(encoded.items())
    # Some processors return rendered text unless tokenize=True is implemented.
    return dict(tokenizer(str(encoded), return_tensors="pt"))


def _model_input_device(model: Any) -> Any:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def run_one_local_preflight(
    *,
    model_config: dict[str, Any],
    grid: dict[str, Any],
    layers: Sequence[int],
    reasoning_setting: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    output_dir: Path,
    local_files_only: bool = False,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; local activation preflight requires a GPU."
        )
    processor, tokenizer, model = _load_processor_and_model(
        model_config, local_files_only=local_files_only
    )
    decoder_layers = locate_decoder_layers(model)
    selected_layers = tuple(int(layer) for layer in layers)
    for layer in selected_layers:
        if layer < 0 or layer >= len(decoder_layers):
            raise ValueError(
                f"Layer {layer} is outside checkpoint depth {len(decoder_layers)}."
            )

    prompt = PROMPT_TEMPLATE.format(grid_state=render_grid(grid["layout"]))
    inputs = _chat_inputs(processor, tokenizer, prompt, reasoning_setting, model_config)
    device = _model_input_device(model)
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    captured: dict[int, list[Any]] = {layer: [] for layer in selected_layers}
    handles = []

    def make_hook(layer: int):
        def hook(_module: Any, _args: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer].append(
                hidden.detach().to(device="cpu", dtype=torch.bfloat16)[0]
            )

        return hook

    for layer in selected_layers:
        handles.append(decoder_layers[layer].register_forward_hook(make_hook(layer)))

    torch.cuda.empty_cache()
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_index)
    started = time.perf_counter()
    generation_error: Exception | None = None
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                return_dict_in_generate=True,
                use_cache=True,
            )
        torch.cuda.synchronize()
    except Exception as exc:
        generation_error = exc
        raise
    finally:
        for handle in handles:
            handle.remove()
    generation_seconds = time.perf_counter() - started
    if generation_error is not None:  # pragma: no cover - retained for explicitness
        raise generation_error

    sequences = outputs.sequences[0].detach().cpu()
    generated_ids = sequences[prompt_tokens:].tolist()
    pieces = token_pieces(tokenizer, generated_ids)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    action = parse_action(generated_text)
    if action is None:
        raise ValueError(
            "Generated output did not contain a parsable maze action JSON object."
        )
    reasoning_indices = reasoning_token_indices(
        pieces, str(model_config["reasoning_trace_format"])
    )
    if not reasoning_indices:
        raise ValueError(
            "No reasoning-token span was identified under the declared "
            f"format {model_config['reasoning_trace_format']!r}."
        )

    saved: dict[str, Any] = {
        "generated_token_ids": torch.tensor(generated_ids, dtype=torch.int64),
        "reasoning_token_indices": torch.tensor(reasoning_indices, dtype=torch.int64),
    }
    activation_bytes = 0
    for layer in selected_layers:
        all_hidden = torch.cat(captured[layer], dim=0)
        if int(all_hidden.shape[0]) < prompt_tokens:
            raise ValueError(
                "Layer hook captured fewer positions than the prompt length."
            )
        generated_hidden = all_hidden[prompt_tokens:]
        usable_indices = [
            index for index in reasoning_indices if index < len(generated_hidden)
        ]
        if len(usable_indices) != len(reasoning_indices):
            # Generation hooks do not see the final generated token because it is never fed
            # back through the model; a final reasoning token would make the alignment invalid.
            raise ValueError("A reasoning token lacks a same-token hooked activation.")
        tensor = generated_hidden[usable_indices].contiguous()
        info = validate_activation_tensor(
            tensor, expected_tokens=len(reasoning_indices)
        )
        if not info["finite"]:
            raise ValueError(f"NaN or Inf detected at layer {layer}.")
        saved[f"layer_{layer}"] = tensor
        activation_bytes += int(info["bytes"])

    artifact_dir = output_dir / "local_preflight_activations"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-z0-9]+", "-", str(model_config["name"]).lower()).strip("-")
    artifact_path = artifact_dir / f"{safe_name}__{grid['grid_id']}.safetensors"
    from safetensors.torch import save_file

    save_file(saved, artifact_path)
    revision = (
        getattr(model.config, "_commit_hash", None)
        or model_config.get("revision")
        or "revision not reported by loader"
    )
    peak_vram_by_device = {
        f"cuda:{device_index}": int(torch.cuda.max_memory_allocated(device_index))
        for device_index in range(torch.cuda.device_count())
    }
    peak_vram = max(peak_vram_by_device.values(), default=0)
    peak_vram_total = sum(peak_vram_by_device.values())
    optimal_path_length = int(grid["optimal_path_length"])
    record = {
        "model": model_config["name"],
        "checkpoint": model_config["local_checkpoint"],
        "resolved_revision": revision,
        "grid_id": grid["grid_id"],
        "reasoning_setting": reasoning_setting,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "seed": int(seed),
        "hook_position": "decoder block output for each generated reasoning token",
        "layers": list(selected_layers),
        "decoder_depth": len(decoder_layers),
        "generated_tokens": len(generated_ids),
        "reasoning_tokens_captured": len(reasoning_indices),
        "reasoning_token_indices": reasoning_indices,
        "generated_text_sha256": hashlib.sha256(
            generated_text.encode("utf-8")
        ).hexdigest(),
        "selected_action": action,
        "generation_and_hook_seconds": generation_seconds,
        "peak_vram_bytes": peak_vram,
        "peak_vram_total_bytes": peak_vram_total,
        "peak_vram_bytes_by_device": peak_vram_by_device,
        "activation_bytes": activation_bytes,
        "projected_activation_bytes_at_optimal_path": activation_bytes
        * optimal_path_length,
        "optimal_path_length": optimal_path_length,
        "activation_artifact": str(artifact_path),
        "activation_shape_by_layer": {
            key: list(value.shape)
            for key, value in saved.items()
            if key.startswith("layer_")
        },
        "all_finite": True,
        "status": "PASS",
    }
    del outputs, model
    torch.cuda.empty_cache()
    return record


def write_local_preflight_report(
    output_dir: Path,
    configured_models: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
) -> None:
    by_model = {str(row["model"]): row for row in records}
    failure_by_model = {str(row["model"]): row for row in failures}
    exact = []
    for model in configured_models:
        name = str(model["name"])
        record = by_model.get(name)
        if record:
            exact.append(
                f"{name}: {record['checkpoint']} @ {record['resolved_revision']}"
            )
        else:
            exact.append(
                f"{name}: {model['local_checkpoint']} @ {model['revision']} (not loaded)"
            )
    all_pass = len(by_model) == len(configured_models) and not failures
    layers_pass = (
        all(set(map(int, row.get("layers", []))) == {8, 15, 23} for row in records)
        and all_pass
    )
    reasoning_pass = (
        all(int(row.get("reasoning_tokens_captured", 0)) > 0 for row in records)
        and all_pass
    )
    peak = max((int(row["peak_vram_bytes"]) for row in records), default=0)
    activation_per_action = (
        sum(int(row["activation_bytes"]) for row in records) / len(records)
        if records
        else 0
    )
    activation_per_trajectory = (
        sum(
            int(row.get("projected_activation_bytes_at_optimal_path", 0))
            for row in records
        )
        / len(records)
        if records
        else 0
    )
    issues = []
    for model in configured_models:
        name = str(model["name"])
        if name in failure_by_model:
            issues.append(f"{name}: {failure_by_model[name]['error']}")
        elif name not in by_model:
            issues.append(f"{name}: not run")
    lines = [
        "# Local activation preflight",
        "",
        f"Status: {'PASS' if all_pass else 'INCOMPLETE / FAIL'}",
        "",
        f"Exact local checkpoints: {'; '.join(exact)}",
        "",
        f"Activation capture at layers 8/15/23: {'PASS' if layers_pass else 'FAIL/NOT RUN'}",
        "",
        f"Reasoning-trace activation capture: {'PASS' if reasoning_pass else 'FAIL/NOT RUN'}",
        "",
        f"Peak VRAM: {peak / 2**30:.2f} GiB" if peak else "Peak VRAM: not measured",
        "",
        (
            f"Approx. activation storage/action: {activation_per_action / 2**20:.2f} MiB "
            "for the three captured layers"
            if records
            else "Approx. activation storage/action: not measured"
        ),
        "",
        (
            f"Approx. activation storage/trajectory: {activation_per_trajectory / 2**20:.2f} MiB "
            "when the one-action measurement is scaled to each grid's optimal path length"
            if records
            else "Approx. activation storage/trajectory: not measured"
        ),
        "",
        f"Local inference/hook issues: {'; '.join(issues) if issues else 'none observed'}",
        "",
        f"Ready for full activation run: {'YES' if all_pass else 'NO'}",
        "",
    ]
    if records:
        lines += [
            "## Per-model checks",
            "",
            "| Model | Grid | Action parsed | Reasoning tokens | Activation shape/layer | Time (s) | Peak VRAM/device |",
            "|---|---|---:|---:|---|---:|---|",
        ]
        for row in records:
            shapes = ", ".join(
                f"{key}: {value}"
                for key, value in row["activation_shape_by_layer"].items()
            )
            device_memory = row.get("peak_vram_bytes_by_device") or {
                "cuda:0": int(row["peak_vram_bytes"])
            }
            peak_text = ", ".join(
                f"{device}: {int(value) / 2**30:.2f} GiB"
                for device, value in device_memory.items()
            )
            lines.append(
                f"| {row['model']} | {row['grid_id']} | yes ({row['selected_action']}) | "
                f"{row['reasoning_tokens_captured']} | {shapes} | "
                f"{float(row['generation_and_hook_seconds']):.1f} | {peak_text} |"
            )
        lines.append("")
    (output_dir / "local_preflight.md").write_text("\n".join(lines))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
