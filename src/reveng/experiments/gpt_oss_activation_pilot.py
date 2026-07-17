"""Calibrate boundary-level GPT-OSS activation extraction on four real states."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import psutil
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from reveng.experiments.gradual_cot_blackbox_alignment import ANALYSIS_START, FINAL_START


DEFAULT_MODEL_SNAPSHOT = Path(
    "/root/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/"
    "6cee5e81ee83917806bbde320786a8fb61efebee"
)
DEFAULT_TRAJECTORY_DIR = Path("data/hf/trajectories_key_door_100/trajectories_key_door")
DEFAULT_SENTENCE_BOUNDARIES = Path(
    "data/behavioral_probes/doorkey_chunking_validation/"
    "sentence_token_boundaries_gpt_oss_20b.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/activation_collection_pilot/gpt_oss_20b_boundary_v1")
DEFAULT_LAYERS = (8, 15, 23)
FULL_CORPUS_REASONING_TOKENS = 2_370_850
FULL_CORPUS_INPUT_TOKENS = 3_378_242
FULL_CORPUS_SENTENCES = 153_622
FULL_CORPUS_ENVIRONMENT_STATES = 1_276


@dataclass(frozen=True)
class PilotState:
    length_class: str
    trajectory_id: str
    step_index: int
    expected_reasoning_tokens: int
    expected_sentences: int

    @property
    def trace_id(self) -> str:
        return f"{self.trajectory_id}_step_{self.step_index:03d}"


PILOT_STATES = (
    PilotState(
        "short",
        "together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_59",
        9,
        348,
        28,
    ),
    PilotState(
        "median",
        "together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_41",
        9,
        1_480,
        110,
    ),
    PilotState(
        "p95",
        "together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_45",
        5,
        4_952,
        320,
    ),
    PilotState(
        "maximum",
        "together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_36",
        1,
        9_733,
        631,
    ),
)


@dataclass
class PreparedState:
    spec: PilotState
    source_path: Path
    input_ids: list[int]
    sentence_rows: list[dict[str, Any]]
    sentence_spans: list[tuple[int, int]]
    reasoning_span: tuple[int, int]
    pre_positions: list[int]
    post_positions: list[int]
    action_position: int
    tokenizer_validation_seconds: float
    token_ids_match: bool
    final_action: str


class PeakResourceMonitor:
    def __init__(self, poll_seconds: float = 0.05) -> None:
        self.poll_seconds = poll_seconds
        self.process = psutil.Process(os.getpid())
        self.peak_rss_bytes = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakResourceMonitor":
        def poll() -> None:
            while not self._stop.wait(self.poll_seconds):
                self.peak_rss_bytes = max(
                    self.peak_rss_bytes,
                    self.process.memory_info().rss,
                )

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_rss_bytes = max(
            self.peak_rss_bytes,
            self.process.memory_info().rss,
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _log(message: str) -> None:
    print(f"[gpt-oss-pilot] {message}", file=sys.stderr, flush=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised by environment preflight
        raise ModuleNotFoundError(
            "Writing activation_index.parquet requires pyarrow. Run `uv sync`."
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def _payload_step(payload: dict[str, Any], step_index: int) -> dict[str, Any]:
    for index, step in enumerate(payload.get("steps", [])):
        if int(step.get("step_id", index)) == step_index:
            return step
    raise ValueError(f"Trajectory does not contain environment step {step_index}.")


def _load_boundary_rows(
    path: Path,
    trace_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    grouped = {trace_id: [] for trace_id in trace_ids}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            trace_id = row["trace_id"]
            if trace_id in grouped:
                grouped[trace_id].append(row)
    for trace_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["sentence_id"]))
        if not rows:
            raise ValueError(f"No canonical sentence boundaries found for {trace_id}.")
    return grouped


def _prepare_state(
    spec: PilotState,
    *,
    trajectory_dir: Path,
    boundary_rows: list[dict[str, Any]],
    tokenizer: Any,
    payload: dict[str, Any] | None = None,
) -> PreparedState:
    source_path = trajectory_dir / f"{spec.trajectory_id}.json"
    if payload is None:
        payload = json.loads(source_path.read_text())
    step = _payload_step(payload, spec.step_index)
    output_text = str(step["output_text"])
    if ANALYSIS_START not in output_text or FINAL_START not in output_text:
        raise ValueError(f"Missing GPT-OSS reasoning/final channel markers in {spec.trace_id}.")
    raw_analysis = output_text.split(ANALYSIS_START, 1)[1].split(FINAL_START, 1)[0]
    stored_analysis = [
        token for token in step["output_tokens"] if "analysis" in token.get("token_groups", [])
    ]
    validation_start = time.perf_counter()
    encoded_analysis_ids = tokenizer(
        raw_analysis,
        add_special_tokens=False,
    )["input_ids"]
    tokenizer_validation_seconds = time.perf_counter() - validation_start
    stored_analysis_ids = [int(token["token_id"]) for token in stored_analysis]
    token_ids_match = list(encoded_analysis_ids) == stored_analysis_ids
    if not token_ids_match:
        raise ValueError(f"Tokenizer IDs do not match stored analysis IDs for {spec.trace_id}.")

    prompt_ids = [int(token["token_id"]) for token in payload["prompt"]["prompt_prefix_tokens"]]
    prompt_ids += [int(token["token_id"]) for token in step["grid_state_tokens"]]
    prompt_ids += [int(token["token_id"]) for token in step["prompt_suffix_tokens"]]
    output_ids = [int(token["token_id"]) for token in step["output_tokens"]]
    input_ids = prompt_ids + output_ids
    prompt_length = len(prompt_ids)

    analysis_output_positions = [int(token["id"]) for token in stored_analysis]
    if len(analysis_output_positions) != spec.expected_reasoning_tokens:
        raise ValueError(
            f"Reasoning-token count changed for {spec.trace_id}: "
            f"expected={spec.expected_reasoning_tokens}, actual={len(analysis_output_positions)}"
        )
    analysis_start_output = min(analysis_output_positions)
    analysis_end_output = max(analysis_output_positions) + 1
    if analysis_output_positions != list(range(analysis_start_output, analysis_end_output)):
        raise ValueError(f"Analysis output-token positions are not contiguous for {spec.trace_id}.")

    sentence_spans: list[tuple[int, int]] = []
    previous_end = -1
    for row in boundary_rows:
        start = prompt_length + int(row["output_token_start"])
        end = prompt_length + int(row["output_token_end_exclusive"])
        if start < previous_end or end <= start:
            raise ValueError(f"Invalid sentence token spans for {spec.trace_id}.")
        sentence_spans.append((start, end))
        previous_end = end
    if len(sentence_spans) != spec.expected_sentences:
        raise ValueError(
            f"Sentence count changed for {spec.trace_id}: "
            f"expected={spec.expected_sentences}, actual={len(sentence_spans)}"
        )

    post_output_positions = [
        int(token["id"])
        for token in step["output_tokens"]
        if int(token["id"]) >= analysis_end_output
    ][:3]
    if len(post_output_positions) != 3:
        raise ValueError(f"Could not resolve three POST positions for {spec.trace_id}.")
    final_action_tokens = [
        token
        for token in step["output_tokens"]
        if "action" in token.get("token_groups", []) and "final" in token.get("token_groups", [])
    ]
    if not final_action_tokens:
        final_action_tokens = [
            token for token in step["output_tokens"] if "action" in token.get("token_groups", [])
        ]
    if not final_action_tokens:
        raise ValueError(f"Could not resolve final action token for {spec.trace_id}.")
    action_token = final_action_tokens[-1]

    return PreparedState(
        spec=spec,
        source_path=source_path,
        input_ids=input_ids,
        sentence_rows=boundary_rows,
        sentence_spans=sentence_spans,
        reasoning_span=(
            prompt_length + analysis_start_output,
            prompt_length + analysis_end_output,
        ),
        pre_positions=list(range(prompt_length - 3, prompt_length)),
        post_positions=[prompt_length + position for position in post_output_positions],
        action_position=prompt_length + int(action_token["id"]),
        tokenizer_validation_seconds=tokenizer_validation_seconds,
        token_ids_match=token_ids_match,
        final_action=str(action_token["token"]).replace("Ġ", "").strip(),
    )


def _filesystem_bytes(path: Path) -> int:
    stat = path.stat()
    return int(getattr(stat, "st_blocks", math.ceil(stat.st_size / 512)) * 512)


class BoundaryAggregateCollector:
    def __init__(self, model: Any, *, layers: Sequence[int], input_device: torch.device) -> None:
        self.model = model
        self.layers = tuple(sorted({int(layer) for layer in layers}))
        self.input_device = input_device
        base_model = getattr(model, "model", model)
        decoder_layers = getattr(base_model, "layers", None)
        if decoder_layers is None:
            raise ValueError("Could not locate GPT-OSS decoder layers.")
        for layer in self.layers:
            if layer < 0 or layer >= len(decoder_layers):
                raise ValueError(f"Layer {layer} is outside model depth {len(decoder_layers)}.")
        self.base_model = base_model
        self.decoder_layers = decoder_layers

    def run(
        self,
        state: PreparedState,
        *,
        forward_chunk_size: int,
        retain_full_for_validation: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        input_ids = torch.tensor([state.input_ids], dtype=torch.long, device=self.input_device)
        sentence_count = len(state.sentence_spans)
        hidden_size = int(self.model.config.hidden_size)
        sentence_counts = torch.tensor(
            [end - start for start, end in state.sentence_spans], dtype=torch.int64
        )
        sentence_sums = {
            layer: torch.zeros((sentence_count, hidden_size), dtype=torch.float32)
            for layer in self.layers
        }
        sentence_finals = {
            layer: torch.empty((sentence_count, hidden_size), dtype=torch.bfloat16)
            for layer in self.layers
        }
        reasoning_sums = {
            layer: torch.zeros(hidden_size, dtype=torch.float32) for layer in self.layers
        }
        reasoning_counts = {layer: 0 for layer in self.layers}
        pre_values = {
            layer: torch.empty((3, hidden_size), dtype=torch.bfloat16) for layer in self.layers
        }
        post_values = {
            layer: torch.empty((3, hidden_size), dtype=torch.bfloat16) for layer in self.layers
        }
        action_values = {
            layer: torch.empty(hidden_size, dtype=torch.bfloat16) for layer in self.layers
        }
        reasoning_final_values = {
            layer: torch.empty(hidden_size, dtype=torch.bfloat16) for layer in self.layers
        }
        pre_seen = {layer: set() for layer in self.layers}
        post_seen = {layer: set() for layer in self.layers}
        action_seen = {layer: False for layer in self.layers}
        reasoning_final_seen = {layer: False for layer in self.layers}
        full_validation: dict[int, list[torch.Tensor]] = {layer: [] for layer in self.layers}
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def capture(layer: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer] = hidden.detach().to(device="cpu", dtype=torch.bfloat16)[0]

            return hook

        for layer in self.layers:
            handles.append(self.decoder_layers[layer].register_forward_hook(capture(layer)))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        rss_before = psutil.Process(os.getpid()).memory_info().rss
        aggregate_seconds = 0.0
        forward_seconds = 0.0
        past_key_values = None
        monitor = PeakResourceMonitor()
        try:
            with monitor, torch.inference_mode():
                for chunk_start in range(0, input_ids.shape[1], forward_chunk_size):
                    chunk_end = min(input_ids.shape[1], chunk_start + forward_chunk_size)
                    captured.clear()
                    forward_start = time.perf_counter()
                    outputs = self.base_model(
                        input_ids=input_ids[:, chunk_start:chunk_end],
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    forward_seconds += time.perf_counter() - forward_start
                    past_key_values = outputs.past_key_values
                    del outputs
                    if set(captured) != set(self.layers):
                        raise ValueError("Forward hooks did not capture every requested layer.")

                    aggregate_start = time.perf_counter()
                    for layer, hidden in captured.items():
                        if retain_full_for_validation:
                            full_validation[layer].append(hidden.clone())
                        for sentence_index, (start, end) in enumerate(state.sentence_spans):
                            overlap_start = max(start, chunk_start)
                            overlap_end = min(end, chunk_end)
                            if overlap_start < overlap_end:
                                local_start = overlap_start - chunk_start
                                local_end = overlap_end - chunk_start
                                sentence_sums[layer][sentence_index] += hidden[
                                    local_start:local_end
                                ].float().sum(dim=0)
                            final_position = end - 1
                            if chunk_start <= final_position < chunk_end:
                                sentence_finals[layer][sentence_index] = hidden[
                                    final_position - chunk_start
                                ]

                        reasoning_start, reasoning_end = state.reasoning_span
                        overlap_start = max(reasoning_start, chunk_start)
                        overlap_end = min(reasoning_end, chunk_end)
                        if overlap_start < overlap_end:
                            local_start = overlap_start - chunk_start
                            local_end = overlap_end - chunk_start
                            reasoning_sums[layer] += hidden[local_start:local_end].float().sum(dim=0)
                            reasoning_counts[layer] += overlap_end - overlap_start
                        reasoning_final = reasoning_end - 1
                        if chunk_start <= reasoning_final < chunk_end:
                            reasoning_final_values[layer] = hidden[reasoning_final - chunk_start]
                            reasoning_final_seen[layer] = True
                        for output_index, position in enumerate(state.pre_positions):
                            if chunk_start <= position < chunk_end:
                                pre_values[layer][output_index] = hidden[position - chunk_start]
                                pre_seen[layer].add(output_index)
                        for output_index, position in enumerate(state.post_positions):
                            if chunk_start <= position < chunk_end:
                                post_values[layer][output_index] = hidden[position - chunk_start]
                                post_seen[layer].add(output_index)
                        if chunk_start <= state.action_position < chunk_end:
                            action_values[layer] = hidden[state.action_position - chunk_start]
                            action_seen[layer] = True
                    aggregate_seconds += time.perf_counter() - aggregate_start
        finally:
            for handle in handles:
                handle.remove()
            del past_key_values
            del input_ids

        tensors: dict[str, torch.Tensor] = {}
        validation_max_abs = 0.0
        for layer in self.layers:
            if pre_seen[layer] != {0, 1, 2} or post_seen[layer] != {0, 1, 2}:
                raise ValueError(f"PRE/POST token collection incomplete at layer {layer}.")
            if not action_seen[layer] or not reasoning_final_seen[layer]:
                raise ValueError(f"Boundary token collection incomplete at layer {layer}.")
            if reasoning_counts[layer] != state.reasoning_span[1] - state.reasoning_span[0]:
                raise ValueError(f"Reasoning aggregation count mismatch at layer {layer}.")
            sentence_mean = (sentence_sums[layer] / sentence_counts[:, None]).to(torch.bfloat16)
            reasoning_mean = (reasoning_sums[layer] / reasoning_counts[layer]).to(torch.bfloat16)
            prefix = f"layer_{layer}"
            tensors[f"{prefix}.sentence_mean"] = sentence_mean.contiguous()
            tensors[f"{prefix}.sentence_final"] = sentence_finals[layer].contiguous()
            tensors[f"{prefix}.reasoning_mean"] = reasoning_mean.contiguous()
            tensors[f"{prefix}.reasoning_final"] = reasoning_final_values[layer].contiguous()
            tensors[f"{prefix}.pre_window"] = pre_values[layer].contiguous()
            tensors[f"{prefix}.post_window"] = post_values[layer].contiguous()
            tensors[f"{prefix}.action_token"] = action_values[layer].contiguous()

            if retain_full_for_validation:
                full = torch.cat(full_validation[layer], dim=0)
                for sentence_index, (start, end) in enumerate(state.sentence_spans):
                    direct_mean = full[start:end].float().mean(dim=0).to(torch.bfloat16)
                    direct_final = full[end - 1]
                    validation_max_abs = max(
                        validation_max_abs,
                        float((direct_mean.float() - sentence_mean[sentence_index].float()).abs().max()),
                        float(
                            (direct_final.float() - sentence_finals[layer][sentence_index].float())
                            .abs()
                            .max()
                        ),
                    )
                del full
        del full_validation
        rss_after = psutil.Process(os.getpid()).memory_info().rss
        peak_allocated = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        peak_reserved = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        metrics = {
            "forward_chunk_size": forward_chunk_size,
            "input_tokens": len(state.input_ids),
            "reasoning_tokens": state.reasoning_span[1] - state.reasoning_span[0],
            "sentences": sentence_count,
            "forward_seconds": forward_seconds,
            "aggregate_seconds": aggregate_seconds,
            "reasoning_tokens_per_forward_second": (
                (state.reasoning_span[1] - state.reasoning_span[0]) / forward_seconds
                if forward_seconds
                else 0.0
            ),
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "peak_rss_bytes": monitor.peak_rss_bytes,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "validation_max_abs_difference": validation_max_abs,
        }
        return tensors, metrics


def benchmark_final_action_attention(
    model: Any,
    state: PreparedState,
    *,
    layers: Sequence[int],
    input_device: torch.device,
    forward_chunk_size: int,
    output_path: Path,
) -> dict[str, Any]:
    """Measure a compact final-action-to-sentence attention readout."""
    base_model = getattr(model, "model", model)
    decoder_layers = getattr(base_model, "layers", None)
    if decoder_layers is None:
        raise ValueError("Could not locate GPT-OSS decoder layers for attention benchmark.")
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
                    raise ValueError(f"Layer {layer} did not expose attention weights in eager mode.")
                captured[layer] = output[1].detach().to(device="cpu", dtype=torch.float32)

            return hook

        for layer in layers:
            handles.append(decoder_layers[int(layer)].self_attn.register_forward_hook(capture(int(layer))))
        old_implementation = model.config._attn_implementation
        model.config._attn_implementation = "eager"
        try:
            query_start = time.perf_counter()
            outputs = base_model(
                input_ids=input_ids[:, state.action_position : state.action_position + 1],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            query_seconds = time.perf_counter() - query_start
            del outputs
        finally:
            model.config._attn_implementation = old_implementation
            for handle in handles:
                handle.remove()

        aggregate_start = time.perf_counter()
        tensors: dict[str, torch.Tensor] = {}
        for layer in layers:
            weights = captured[int(layer)]
            # One query token: [batch, heads, query=1, source positions].
            action_weights = weights[0, :, 0, :]
            sentence_attention = torch.zeros(
                (action_weights.shape[0], len(state.sentence_spans)), dtype=torch.float32
            )
            for sentence_index, (start, end) in enumerate(state.sentence_spans):
                clipped_end = min(end, action_weights.shape[-1])
                if start < clipped_end:
                    sentence_attention[:, sentence_index] = action_weights[
                        :, start:clipped_end
                    ].mean(dim=-1)
            tensors[f"layer_{layer}.final_action_to_sentence_mean"] = sentence_attention
        aggregate_seconds = time.perf_counter() - aggregate_start
    del past_key_values
    del input_ids
    serialization_start = time.perf_counter()
    save_file(
        tensors,
        str(output_path),
        metadata={
            "trace_id": state.spec.trace_id,
            "query_position": str(state.action_position),
            "attention_implementation": "eager",
        },
    )
    serialization_seconds = time.perf_counter() - serialization_start
    metrics = {
        "status": "completed",
        "trace_id": state.spec.trace_id,
        "layers": list(layers),
        "forward_chunk_size": forward_chunk_size,
        "prefix_input_tokens": state.action_position,
        "source_tokens_at_action": state.action_position + 1,
        "sentences": len(state.sentence_spans),
        "prefix_forward_seconds": prefix_seconds,
        "attention_query_seconds": query_seconds,
        "attention_aggregate_seconds": aggregate_seconds,
        "serialization_seconds": serialization_seconds,
        "logical_tensor_bytes": _tensor_logical_bytes(tensors),
        "file_size_bytes": output_path.stat().st_size,
        "filesystem_bytes": _filesystem_bytes(output_path),
        "peak_rss_bytes": monitor.peak_rss_bytes,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
        ),
        "attention_backend_for_query": "eager",
        "scope": "final action token to preceding canonical sentences",
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def _tensor_logical_bytes(tensors: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())


def _index_rows(
    state: PreparedState,
    *,
    layers: Sequence[int],
    shard_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in layers:
        for index, (sentence, span) in enumerate(zip(state.sentence_rows, state.sentence_spans)):
            rows.append(
                {
                    "record_kind": "sentence",
                    "trace_id": state.spec.trace_id,
                    "trajectory_id": state.spec.trajectory_id,
                    "step_index": state.spec.step_index,
                    "length_class": state.spec.length_class,
                    "layer": int(layer),
                    "sentence_id": int(sentence["sentence_id"]),
                    "tensor_row": index,
                    "token_start": span[0],
                    "token_end_exclusive": span[1],
                    "n_tokens": span[1] - span[0],
                    "char_start": int(sentence["char_start"]),
                    "char_end": int(sentence["char_end"]),
                    "mean_tensor_key": f"layer_{layer}.sentence_mean",
                    "final_tensor_key": f"layer_{layer}.sentence_final",
                    "shard_path": str(shard_path),
                }
            )
        rows.extend(
            {
                "record_kind": kind,
                "trace_id": state.spec.trace_id,
                "trajectory_id": state.spec.trajectory_id,
                "step_index": state.spec.step_index,
                "length_class": state.spec.length_class,
                "layer": int(layer),
                "sentence_id": None,
                "tensor_row": None,
                "token_start": start,
                "token_end_exclusive": end,
                "n_tokens": end - start,
                "char_start": None,
                "char_end": None,
                "mean_tensor_key": key if "mean" in key else None,
                "final_tensor_key": key if "mean" not in key else None,
                "shard_path": str(shard_path),
            }
            for kind, start, end, key in (
                (
                    "reasoning_mean",
                    state.reasoning_span[0],
                    state.reasoning_span[1],
                    f"layer_{layer}.reasoning_mean",
                ),
                (
                    "reasoning_final",
                    state.reasoning_span[1] - 1,
                    state.reasoning_span[1],
                    f"layer_{layer}.reasoning_final",
                ),
                (
                    "pre_window",
                    state.pre_positions[0],
                    state.pre_positions[-1] + 1,
                    f"layer_{layer}.pre_window",
                ),
                (
                    "post_window",
                    state.post_positions[0],
                    state.post_positions[-1] + 1,
                    f"layer_{layer}.post_window",
                ),
                (
                    "action_token",
                    state.action_position,
                    state.action_position + 1,
                    f"layer_{layer}.action_token",
                ),
            )
        )
    return rows


def _fit_disk_projection(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    xs = [float(row["sentences"]) for row in rows]
    ys = [float(row["filesystem_bytes"]) for row in rows]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    intercept = y_mean - slope * x_mean
    projected = max(0.0, intercept) * FULL_CORPUS_ENVIRONMENT_STATES + slope * FULL_CORPUS_SENTENCES
    return {
        "fitted_bytes_per_sentence": slope,
        "fitted_bytes_per_environment_state": max(0.0, intercept),
        "projected_filesystem_bytes": projected,
    }


def _build_resource_projection(
    extraction_rows: Sequence[dict[str, Any]],
    *,
    model_load_seconds: float,
    projected_logical_bytes: int,
    attention_benchmark: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_reasoning_tokens = sum(int(row["reasoning_tokens"]) for row in extraction_rows)
    total_input_tokens = sum(int(row["input_tokens"]) for row in extraction_rows)
    total_forward_seconds = sum(float(row["forward_seconds"]) for row in extraction_rows)
    total_aggregate_seconds = sum(float(row["aggregate_seconds"]) for row in extraction_rows)
    total_serialization_seconds = sum(
        float(row["serialization_seconds"]) for row in extraction_rows
    )
    total_filesystem_bytes = sum(int(row["filesystem_bytes"]) for row in extraction_rows)
    total_sentences = sum(int(row["sentences"]) for row in extraction_rows)
    central_input_throughput = total_input_tokens / total_forward_seconds
    long_rows = [row for row in extraction_rows if row["length_class"] in {"p95", "maximum"}]
    conservative_input_throughput = min(
        float(row["input_tokens"]) / float(row["forward_seconds"]) for row in long_rows
    )
    disk_projection = _fit_disk_projection(extraction_rows)
    projected_aggregate_seconds = (
        total_aggregate_seconds / total_sentences * FULL_CORPUS_SENTENCES
    )
    projected_serialization_seconds = (
        total_serialization_seconds
        / total_filesystem_bytes
        * disk_projection["projected_filesystem_bytes"]
    )
    central_forward_seconds = FULL_CORPUS_INPUT_TOKENS / central_input_throughput
    conservative_forward_seconds = FULL_CORPUS_INPUT_TOKENS / conservative_input_throughput
    fixed_seconds = model_load_seconds + projected_aggregate_seconds + projected_serialization_seconds
    projection = {
        "full_corpus_reasoning_tokens": FULL_CORPUS_REASONING_TOKENS,
        "full_corpus_input_tokens": FULL_CORPUS_INPUT_TOKENS,
        "full_corpus_sentences": FULL_CORPUS_SENTENCES,
        "full_corpus_environment_states": FULL_CORPUS_ENVIRONMENT_STATES,
        "central_input_tokens_per_second": central_input_throughput,
        "conservative_input_tokens_per_second": conservative_input_throughput,
        "observed_reasoning_tokens_per_second": total_reasoning_tokens / total_forward_seconds,
        "central_forward_seconds": central_forward_seconds,
        "conservative_forward_seconds": conservative_forward_seconds,
        "projected_aggregate_seconds": projected_aggregate_seconds,
        "projected_serialization_seconds": projected_serialization_seconds,
        "model_load_seconds": model_load_seconds,
        "central_seconds": central_forward_seconds + fixed_seconds,
        "conservative_seconds": conservative_forward_seconds + fixed_seconds,
        "projected_logical_bytes": projected_logical_bytes,
        **disk_projection,
    }
    if attention_benchmark is not None:
        attention_prefix_throughput = (
            float(attention_benchmark["prefix_input_tokens"])
            / float(attention_benchmark["prefix_forward_seconds"])
        )
        attention_prefix_seconds = FULL_CORPUS_INPUT_TOKENS / attention_prefix_throughput
        attention_query_seconds = (
            float(attention_benchmark["attention_query_seconds"])
            * FULL_CORPUS_ENVIRONMENT_STATES
        )
        attention_aggregate_seconds = (
            float(attention_benchmark["attention_aggregate_seconds"])
            / int(attention_benchmark["sentences"])
            * FULL_CORPUS_SENTENCES
        )
        logical_per_sentence = (
            int(attention_benchmark["logical_tensor_bytes"])
            / int(attention_benchmark["sentences"])
        )
        metadata_per_state = max(
            0,
            int(attention_benchmark["file_size_bytes"])
            - int(attention_benchmark["logical_tensor_bytes"]),
        )
        attention_storage_bytes = (
            logical_per_sentence * FULL_CORPUS_SENTENCES
            + metadata_per_state * FULL_CORPUS_ENVIRONMENT_STATES
        )
        attention_serialization_seconds = (
            float(attention_benchmark["serialization_seconds"])
            / int(attention_benchmark["file_size_bytes"])
            * attention_storage_bytes
        )
        projection.update(
            {
                "attention_projection_basis": "single median state; separate pass",
                "attention_prefix_input_tokens_per_second": attention_prefix_throughput,
                "projected_attention_prefix_forward_seconds": attention_prefix_seconds,
                "projected_attention_query_seconds": attention_query_seconds,
                "projected_attention_aggregate_seconds": attention_aggregate_seconds,
                "projected_attention_serialization_seconds": attention_serialization_seconds,
                "projected_attention_incremental_seconds": (
                    attention_prefix_seconds
                    + attention_query_seconds
                    + attention_aggregate_seconds
                    + attention_serialization_seconds
                ),
                "projected_attention_filesystem_bytes": attention_storage_bytes,
            }
        )
    return projection


def _build_report(
    *,
    manifest: dict[str, Any],
    timing_rows: Sequence[dict[str, Any]],
    projection: dict[str, Any],
) -> str:
    extraction_rows = [row for row in timing_rows if row["phase"] == "pilot_extraction"]
    lines = [
        "# GPT-OSS-20B Boundary-Activation Pilot",
        "",
        f"- Status: {manifest['status']}",
        f"- Model revision: `{manifest['model_revision']}`",
        f"- Selected chunk size: {manifest['selected_forward_chunk_size']}",
        f"- Model load time: {manifest['model_load_seconds']:.2f} seconds",
        f"- Pilot states completed: {len(extraction_rows)}",
        f"- Pilot reasoning tokens: {sum(int(row['reasoning_tokens']) for row in extraction_rows):,}",
        f"- Pilot sentences: {sum(int(row['sentences']) for row in extraction_rows):,}",
        "",
        "## Measured states",
        "",
        "| Length | Reasoning tokens | Sentences | Forward seconds | Tokens/second | Peak VRAM | Shard size |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in extraction_rows:
        lines.append(
            f"| {row['length_class']} | {int(row['reasoning_tokens']):,} | "
            f"{int(row['sentences']):,} | {float(row['forward_seconds']):.2f} | "
            f"{float(row['reasoning_tokens_per_forward_second']):.1f} | "
            f"{float(row['peak_cuda_reserved_bytes']) / 2**30:.2f} GiB | "
            f"{int(row['filesystem_bytes']) / 2**20:.2f} MiB |"
        )
    lines.extend(
        [
            "",
            "## Full-corpus projection",
            "",
            f"- Central end-to-end extraction time: {projection['central_seconds'] / 3600:.2f} hours",
            f"- Conservative end-to-end extraction time: {projection['conservative_seconds'] / 3600:.2f} hours",
            f"- Exact teacher-forced input tokens: {projection['full_corpus_input_tokens']:,}",
            f"- Projected packed storage: {projection['projected_filesystem_bytes'] / 1e9:.2f} GB",
            f"- Projected logical tensor storage: {projection['projected_logical_bytes'] / 1e9:.2f} GB",
            "",
            "## Separate compact-attention benchmark",
            "",
            f"- Median-state attention pass: {manifest['compact_attention_total_seconds']:.2f} seconds",
            f"- Median-state attention output: {manifest['compact_attention_file_size_bytes'] / 2**10:.1f} KiB",
            f"- Rough full-corpus incremental time: {projection['projected_attention_incremental_seconds'] / 3600:.2f} hours",
            f"- Rough full-corpus attention storage: {projection['projected_attention_filesystem_bytes'] / 1e6:.2f} MB",
            "",
            (
                "The attention estimate comes from one median-length state and is less reliable "
                "than the four-state residual-stream estimate. It covers attention from the final "
                "action token to preceding sentences and is not included in the extraction times above."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def run_gpt_oss_activation_pilot(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_snapshot: str | Path = DEFAULT_MODEL_SNAPSHOT,
    trajectory_dir: str | Path = DEFAULT_TRAJECTORY_DIR,
    sentence_boundaries_path: str | Path = DEFAULT_SENTENCE_BOUNDARIES,
    layers: Sequence[int] = DEFAULT_LAYERS,
    benchmark_chunk_sizes: Sequence[int] = (128, 256, 512),
    vram_limit_gib: float = 22.0,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shard_dir = output / "shards"
    shard_dir.mkdir(exist_ok=True)
    model_path = Path(model_snapshot)
    model_revision = model_path.name
    manifest: dict[str, Any] = {
        "status": "running",
        "model_name": "openai/gpt-oss-20b",
        "model_path": str(model_path),
        "model_revision": model_revision,
        "layers": list(layers),
        "activation_dtype": "bfloat16",
        "prompt_mode": "original_trace_stored_token_ids",
        "attention_benchmarked": False,
        "pilot_states": [state.__dict__ for state in PILOT_STATES],
    }
    _write_json(output / "pilot_manifest.json", manifest)
    try:
        _log("loading tokenizer and validating four trajectory states")
        tokenizer_load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tokenizer_load_seconds = time.perf_counter() - tokenizer_load_start
        boundaries = _load_boundary_rows(
            Path(sentence_boundaries_path),
            {state.trace_id for state in PILOT_STATES},
        )
        prepared = [
            _prepare_state(
                state,
                trajectory_dir=Path(trajectory_dir),
                boundary_rows=boundaries[state.trace_id],
                tokenizer=tokenizer,
            )
            for state in PILOT_STATES
        ]
        prepared_by_class = {state.spec.length_class: state for state in prepared}

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        _log("loading openai/gpt-oss-20b on CUDA")
        load_start = time.perf_counter()
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
        model_load_seconds = time.perf_counter() - load_start
        _log(f"model loaded in {model_load_seconds:.2f} seconds")
        input_device = model.model.embed_tokens.weight.device
        collector = BoundaryAggregateCollector(model, layers=layers, input_device=input_device)

        # Warm kernels before comparing chunk sizes.
        collector.run(
            prepared_by_class["short"],
            forward_chunk_size=256,
        )
        benchmark_rows: list[dict[str, Any]] = []
        for chunk_size in benchmark_chunk_sizes:
            _log(f"benchmarking forward_chunk_size={chunk_size} on median state")
            tensors, metrics = collector.run(
                prepared_by_class["median"],
                forward_chunk_size=int(chunk_size),
            )
            benchmark_rows.append(
                {
                    "phase": "chunk_benchmark",
                    "length_class": "median",
                    "trace_id": prepared_by_class["median"].spec.trace_id,
                    **metrics,
                }
            )
            del tensors
        eligible = [
            row
            for row in benchmark_rows
            if float(row["peak_cuda_reserved_bytes"]) <= vram_limit_gib * 2**30
        ]
        if not eligible:
            raise RuntimeError(
                f"No benchmark chunk size stayed below {vram_limit_gib:.1f} GiB reserved VRAM."
            )
        selected_chunk_size = int(
            max(eligible, key=lambda row: float(row["reasoning_tokens_per_forward_second"]))[
                "forward_chunk_size"
            ]
        )
        _log(f"selected forward_chunk_size={selected_chunk_size}")

        timing_rows = list(benchmark_rows)
        index_rows: list[dict[str, Any]] = []
        first_short_tensors: dict[str, torch.Tensor] | None = None
        for state in prepared:
            _log(
                f"extracting {state.spec.length_class} state: "
                f"{state.spec.expected_reasoning_tokens} reasoning tokens"
            )
            tensors, metrics = collector.run(
                state,
                forward_chunk_size=selected_chunk_size,
                retain_full_for_validation=(state.spec.length_class == "short"),
            )
            serialization_start = time.perf_counter()
            shard_path = shard_dir / f"{state.spec.trace_id}.safetensors"
            save_file(
                tensors,
                str(shard_path),
                metadata={
                    "model_revision": model_revision,
                    "trace_id": state.spec.trace_id,
                    "activation_dtype": "bfloat16",
                },
            )
            serialization_seconds = time.perf_counter() - serialization_start
            logical_bytes = _tensor_logical_bytes(tensors)
            filesystem_bytes = _filesystem_bytes(shard_path)
            sha256 = hashlib.sha256(shard_path.read_bytes()).hexdigest()
            timing_rows.append(
                {
                    "phase": "pilot_extraction",
                    "length_class": state.spec.length_class,
                    "trace_id": state.spec.trace_id,
                    "tokenizer_validation_seconds": state.tokenizer_validation_seconds,
                    "token_ids_match": state.token_ids_match,
                    "serialization_seconds": serialization_seconds,
                    "logical_tensor_bytes": logical_bytes,
                    "file_size_bytes": shard_path.stat().st_size,
                    "filesystem_bytes": filesystem_bytes,
                    "bytes_per_sentence": filesystem_bytes / len(state.sentence_spans),
                    "shard_sha256": sha256,
                    **metrics,
                }
            )
            index_rows.extend(_index_rows(state, layers=layers, shard_path=shard_path))
            if state.spec.length_class == "short":
                first_short_tensors = {key: value.clone() for key, value in tensors.items()}
            del tensors

        if first_short_tensors is None:
            raise AssertionError("Short-state validation tensors were not retained.")
        _log("repeating short state for determinism validation")
        repeated_short, repeat_metrics = collector.run(
            prepared_by_class["short"],
            forward_chunk_size=selected_chunk_size,
        )
        repeat_max_abs = max(
            float((first_short_tensors[key].float() - repeated_short[key].float()).abs().max())
            for key in first_short_tensors
        )
        repeat_equal = all(
            torch.equal(first_short_tensors[key], repeated_short[key]) for key in first_short_tensors
        )
        del first_short_tensors
        del repeated_short

        _log("benchmarking compact final-action attention on median state")
        attention_path = output / "compact_attention_median.safetensors"
        attention_benchmark = benchmark_final_action_attention(
            model,
            prepared_by_class["median"],
            layers=layers,
            input_device=input_device,
            forward_chunk_size=selected_chunk_size,
            output_path=attention_path,
        )
        _write_json(output / "compact_attention_benchmark.json", attention_benchmark)

        extraction_rows = [row for row in timing_rows if row["phase"] == "pilot_extraction"]
        logical_bytes_per_sentence = len(layers) * 2 * int(model.config.hidden_size) * 2
        logical_bytes_per_state = len(layers) * 9 * int(model.config.hidden_size) * 2
        projected_logical_bytes = (
            FULL_CORPUS_SENTENCES * logical_bytes_per_sentence
            + FULL_CORPUS_ENVIRONMENT_STATES * logical_bytes_per_state
        )
        projection = _build_resource_projection(
            extraction_rows,
            model_load_seconds=model_load_seconds,
            projected_logical_bytes=projected_logical_bytes,
            attention_benchmark=attention_benchmark,
        )
        _write_csv(output / "timing_rows.csv", timing_rows)
        _write_parquet(output / "activation_index.parquet", index_rows)
        _write_json(output / "resource_projection.json", projection)
        manifest.update(
            {
                "status": "completed",
                "tokenizer_load_seconds": tokenizer_load_seconds,
                "model_load_seconds": model_load_seconds,
                "model_load_peak_rss_bytes": load_monitor.peak_rss_bytes,
                "selected_forward_chunk_size": selected_chunk_size,
                "vram_limit_gib": vram_limit_gib,
                "n_completed_states": len(extraction_rows),
                "n_activation_index_rows": len(index_rows),
                "short_repeat_exact_equal": repeat_equal,
                "short_repeat_max_abs_difference": repeat_max_abs,
                "short_repeat_metrics": repeat_metrics,
                "temporary_full_token_tensors_retained": False,
                "tokenizer_ids_match_stored_trajectory_tokens": True,
                "attention_benchmarked": True,
                "compact_attention_total_seconds": (
                    float(attention_benchmark["prefix_forward_seconds"])
                    + float(attention_benchmark["attention_query_seconds"])
                    + float(attention_benchmark["attention_aggregate_seconds"])
                    + float(attention_benchmark["serialization_seconds"])
                ),
                "compact_attention_file_size_bytes": int(
                    attention_benchmark["file_size_bytes"]
                ),
                "compact_attention_benchmark_path": str(
                    output / "compact_attention_benchmark.json"
                ),
            }
        )
        _write_json(output / "pilot_manifest.json", manifest)
        (output / "pilot_report.md").write_text(
            _build_report(manifest=manifest, timing_rows=timing_rows, projection=projection)
        )
        _log("pilot completed")
        return manifest
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(output / "pilot_manifest.json", manifest)
        raise


__all__ = [
    "BoundaryAggregateCollector",
    "PILOT_STATES",
    "PilotState",
    "PreparedState",
    "benchmark_final_action_attention",
    "_build_resource_projection",
    "run_gpt_oss_activation_pilot",
]
