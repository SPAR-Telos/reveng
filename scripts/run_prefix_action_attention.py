#!/usr/bin/env python3
"""Extract and analyze attention at immediate DoorKey action readouts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from reveng.experiments.gpt_oss_activation_pilot import (
    DEFAULT_MODEL_SNAPSHOT,
    PeakResourceMonitor,
)
from reveng.experiments.local_immediate_behavioral import ACTION_LABELS
from reveng.experiments.prefix_action_attention import (
    REGION_LABELS,
    REGION_NAMES,
    add_grid_action_regions,
    aggregate_attention,
    build_prompt_specs,
    paired_attention_differences,
    probability_error,
    read_csv,
    trajectory_bootstrap_ci,
    write_csv,
    write_json,
)


DEFAULT_OUTPUT = Path("outputs/hypothesis_tests/prefix_action_attention_v1")
DEFAULT_PAIRS = Path(
    "outputs/hypothesis_tests/semantic_reasoning_classification_v1/"
    "matched_sentence_pairs.csv"
)
DEFAULT_CANDIDATES = Path(
    "data/behavioral_probes/reasoning_belief_action_cohorts/matched_46_candidates.csv"
)
DEFAULT_POSITIONS = Path(
    "outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1/"
    "position_rows.csv"
)
DEFAULT_SENTENCES = Path(
    "data/behavioral_probes/doorkey_chunking_validation/sentences.csv"
)
DEFAULT_EVENTS = Path(
    "outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1/"
    "event_rows.csv"
)
DEFAULT_TRAJECTORIES = Path(
    "data/hf/trajectories_key_door_100/trajectories_key_door"
)


def log(message: str) -> None:
    print(f"[prefix-action-attention] {message}", file=sys.stderr, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_layers(model: Any) -> Any:
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise ValueError("Could not locate GPT-OSS decoder layers.")
    return base, layers


def _validate_full_attention_layers(model: Any, layers: Sequence[int]) -> None:
    layer_types = getattr(model.config, "layer_types", None)
    if layer_types is None:
        raise ValueError("Model configuration does not expose attention layer types.")
    invalid = [layer for layer in layers if layer_types[int(layer)] != "full_attention"]
    if invalid:
        raise ValueError(
            "Immediate-readout analysis requires full-attention layers; rejected: "
            + ", ".join(map(str, invalid))
        )


def _sentence_token_indices(spec: dict[str, Any]) -> list[list[int]]:
    position = int(spec["position_index"])
    if position == 0:
        return []
    spans = spec["region_spans"]
    newest = spans["newest_reasoning_sentence"]
    reasoning_spans = []
    if spans["earlier_reasoning"]:
        earlier_start, _ = spans["earlier_reasoning"][0]
    elif newest:
        earlier_start = newest[0][0]
    else:
        return []
    # Canonical sentence boundaries are represented by the individual prompt
    # sentence text spans. Reconstruct them by splitting the earlier/newest area
    # with the source sentence rows stored on the spec.
    for span in spec["sentence_character_spans"]:
        reasoning_spans.append(span)
    offsets = spec["offsets"]
    return [
        [
            token
            for token, (token_start, token_end) in enumerate(offsets)
            if token_start < token_end
            and token != spec["query_index"]
            and token_start < end
            and token_end > start
        ]
        for start, end in reasoning_spans
    ]


def add_sentence_character_spans(specs: list[dict[str, Any]], sentence_rows_path: Path) -> None:
    from reveng.experiments.prefix_action_attention import load_sentence_rows, trace_id

    rows = load_sentence_rows(sentence_rows_path)
    for spec in specs:
        position = int(spec["position_index"])
        if position == 0:
            spec["sentence_character_spans"] = []
            continue
        canonical = rows[trace_id(spec["trajectory_id"], int(spec["step_index"]))][:position]
        reasoning_start = int(spec["reasoning_start"])
        leading_trim = int(spec["reasoning_leading_trim"])
        reasoning_length = int(spec["reasoning_length"])
        spans = []
        for row in canonical:
            start = max(0, int(row["char_start"]) - leading_trim)
            end = min(reasoning_length, int(row["char_end"]) - leading_trim)
            if start < end:
                spans.append((reasoning_start + start, reasoning_start + end))
        spec["sentence_character_spans"] = spans


def extract_one_prompt(
    model: Any,
    spec: dict[str, Any],
    *,
    layers: Sequence[int],
    temperature: float,
    forward_chunk_size: int,
) -> dict[str, Any]:
    base, decoder_layers = _model_layers(model)
    device = base.embed_tokens.weight.device
    input_ids = torch.tensor([spec["input_ids"]], dtype=torch.long, device=device)
    query_index = int(spec["query_index"])
    scoring_common_length = int(spec["scoring_common_length"])
    cache = None
    with torch.inference_mode():
        for start in range(0, scoring_common_length, forward_chunk_size):
            end = min(scoring_common_length, start + forward_chunk_size)
            output = base(
                input_ids=input_ids[:, start:end],
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = output.past_key_values
            del output

        candidate_ids = [
            int(model._prefix_action_candidate_ids[label]) for label in ACTION_LABELS
        ]

        # Validate against the original readout protocol before switching the
        # one-token attention query to the eager implementation.
        score_cache = copy.copy(cache)
        score_cache.layers = [copy.copy(layer) for layer in cache.layers]
        score_output = model(
            input_ids=input_ids[:, scoring_common_length : query_index + 1],
            past_key_values=score_cache,
            use_cache=False,
            return_dict=True,
            output_attentions=False,
        )
        score_logits = score_output.logits[0, -1, candidate_ids].float() / temperature
        score_probabilities_tensor = torch.softmax(score_logits, dim=0).cpu()
        probabilities = {
            label: float(score_probabilities_tensor[index])
            for index, label in enumerate(ACTION_LABELS)
        }
        reproduced_action = max(probabilities, key=probabilities.get)
        add_grid_action_regions(
            spec,
            current_position=spec["current_position"],
            action=reproduced_action,
        )
        del score_output, score_cache

        attention_cache = copy.copy(cache)
        attention_cache.layers = [copy.copy(layer) for layer in cache.layers]
        for start in range(scoring_common_length, query_index, forward_chunk_size):
            end = min(query_index, start + forward_chunk_size)
            output = base(
                input_ids=input_ids[:, start:end],
                past_key_values=attention_cache,
                use_cache=True,
                return_dict=True,
            )
            attention_cache = output.past_key_values
            del output

        captured: dict[int, torch.Tensor] = {}
        handles = []

        def capture(layer: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                    raise ValueError(f"Layer {layer} did not expose attention weights.")
                captured[layer] = output[1].detach().cpu().float()

            return hook

        for layer in layers:
            handles.append(
                decoder_layers[int(layer)].self_attn.register_forward_hook(capture(int(layer)))
            )
        old_implementation = getattr(model.config, "_attn_implementation", None)
        model.config._attn_implementation = "eager"
        try:
            output = model(
                input_ids=input_ids[:, query_index : query_index + 1],
                past_key_values=attention_cache,
                use_cache=False,
                return_dict=True,
                output_attentions=True,
            )
            attention_logits = output.logits[0, -1, candidate_ids].float() / temperature
            attention_probabilities_tensor = torch.softmax(
                attention_logits, dim=0
            ).cpu()
            attention_probabilities = {
                label: float(attention_probabilities_tensor[index])
                for index, label in enumerate(ACTION_LABELS)
            }
            del output
        finally:
            if old_implementation is not None:
                model.config._attn_implementation = old_implementation
            for handle in handles:
                handle.remove()

    sentence_indices = _sentence_token_indices(spec)
    aggregates = {}
    for layer in layers:
        weights = captured[int(layer)]
        if weights.shape[2] != 1:
            raise ValueError(f"Expected one query token at layer {layer}.")
        attention = weights[0, :, 0, :]
        if attention.shape[-1] != query_index + 1:
            raise ValueError(
                f"Layer {layer} returned {attention.shape[-1]} source tokens for "
                f"query position {query_index}; full attention was expected."
            )
        aggregates[int(layer)] = aggregate_attention(
            attention,
            sentence_token_indices=sentence_indices,
            region_token_indices=spec["region_token_indices"],
        )
    del cache, attention_cache, input_ids
    return {
        "probabilities": probabilities,
        "attention_kernel_probabilities": attention_probabilities,
        "aggregates": aggregates,
    }


def _state_shard_valid(path: Path, record_path: Path, layers: Sequence[int]) -> bool:
    if not path.exists() or not record_path.exists():
        return False
    try:
        record = json.loads(record_path.read_text())
        if record["sha256"] != sha256(path):
            return False
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
        return all(f"layer_{layer}.region_mean" in keys for layer in layers)
    except Exception:
        return False


def extract(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir
    shard_dir = output / "shards"
    record_dir = output / "state_records"
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    specs, pairs = build_prompt_specs(
        pair_rows_path=args.pair_rows,
        candidate_rows_path=args.candidate_rows,
        position_rows_path=args.position_rows,
        sentence_rows_path=args.sentence_rows,
        trajectory_dir=args.trajectory_dir,
        tokenizer=tokenizer,
    )
    add_sentence_character_spans(specs, args.sentence_rows)
    if args.limit_states is not None:
        selected = sorted({spec["example_id"] for spec in specs})[: args.limit_states]
        specs = [spec for spec in specs if spec["example_id"] in selected]
    if args.limit_prompts is not None:
        specs = specs[: args.limit_prompts]
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        by_state[spec["example_id"]].append(spec)
    layers = tuple(sorted(set(args.layers)))
    pending_states = [
        state
        for state in sorted(by_state)
        if not (
            args.resume
            and _state_shard_valid(
                shard_dir / f"{state}.safetensors",
                record_dir / f"{state}.json",
                layers,
            )
        )
    ]
    manifest = {
        "status": "running" if pending_states else "completed",
        "protocol": "immediate_action_query_attention_v1",
        "model_path": str(args.model_path),
        "model_revision": args.model_path.name,
        "temperature": args.temperature,
        "top_p": 0.95,
        "seed": 42,
        "layers": list(layers),
        "attention_layer_types": ["full_attention" for _ in layers],
        "query_position": (
            "last prompt token immediately before the action label token; its logits "
            "score UP, DOWN, LEFT, and RIGHT"
        ),
        "states": len(by_state),
        "prompts": len(specs),
        "pending_states": len(pending_states),
        "pair_rows": str(args.pair_rows),
    }
    write_json(output / "run_manifest.json", manifest)
    if not pending_states:
        return manifest

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    log(f"loading GPT-OSS-20B for {len(pending_states)} states")
    started_load = time.perf_counter()
    with PeakResourceMonitor() as load_monitor:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        ).eval()
    model._prefix_action_candidate_ids = {
        label: tokenizer(label, add_special_tokens=False)["input_ids"][0]
        for label in ACTION_LABELS
    }
    if any(
        len(tokenizer(label, add_special_tokens=False)["input_ids"]) != 1
        for label in ACTION_LABELS
    ):
        raise ValueError("Action labels are not single tokens.")
    _validate_full_attention_layers(model, layers)
    manifest["model_load_seconds"] = time.perf_counter() - started_load
    manifest["model_load_peak_rss_bytes"] = load_monitor.peak_rss_bytes
    write_json(output / "run_manifest.json", manifest)

    completed_new = 0
    for state in pending_states:
        state_specs = sorted(by_state[state], key=lambda spec: spec["position_index"])
        max_sentences = max(int(spec["position_index"]) for spec in state_specs)
        n_prompts = len(state_specs)
        heads = int(model.config.num_attention_heads)
        tensors: dict[str, torch.Tensor] = {
            "position_indices": torch.tensor(
                [spec["position_index"] for spec in state_specs], dtype=torch.int64
            ),
            "sentence_counts": torch.tensor(
                [spec["position_index"] for spec in state_specs], dtype=torch.int64
            ),
        }
        for layer in layers:
            tensors[f"layer_{layer}.sentence_mass"] = torch.zeros(
                (n_prompts, heads, max_sentences), dtype=torch.float32
            )
            tensors[f"layer_{layer}.sentence_mean"] = torch.zeros(
                (n_prompts, heads, max_sentences), dtype=torch.float32
            )
            tensors[f"layer_{layer}.region_mass"] = torch.zeros(
                (n_prompts, heads, len(REGION_NAMES)), dtype=torch.float32
            )
            tensors[f"layer_{layer}.region_mean"] = torch.zeros(
                (n_prompts, heads, len(REGION_NAMES)), dtype=torch.float32
            )
            tensors[f"layer_{layer}.region_counts"] = torch.zeros(
                (n_prompts, len(REGION_NAMES)), dtype=torch.int64
            )
        validation_rows = []
        state_start = time.perf_counter()
        for prompt_index, spec in enumerate(state_specs):
            result = extract_one_prompt(
                model,
                spec,
                layers=layers,
                temperature=args.temperature,
                forward_chunk_size=args.forward_chunk_size,
            )
            observed = result["probabilities"]
            attention_kernel_probabilities = result["attention_kernel_probabilities"]
            expected = spec["expected_probabilities"]
            observed_action = max(observed, key=observed.get)
            attention_kernel_action = max(
                attention_kernel_probabilities,
                key=attention_kernel_probabilities.get,
            )
            error = probability_error(observed, expected)
            kernel_probability_error = probability_error(
                attention_kernel_probabilities, observed
            )
            validation_rows.append(
                {
                    "example_id": state,
                    "trajectory_id": spec["trajectory_id"],
                    "step_index": spec["step_index"],
                    "position_index": spec["position_index"],
                    "prompt_tokens": len(spec["input_ids"]),
                    "expected_action": spec["expected_action"],
                    "reproduced_action": observed_action,
                    "action_matches": observed_action == spec["expected_action"],
                    "max_probability_absolute_error": error,
                    "attention_kernel_action": attention_kernel_action,
                    "attention_kernel_argmax_matches": (
                        attention_kernel_action == observed_action
                    ),
                    "attention_kernel_max_probability_difference": (
                        kernel_probability_error
                    ),
                    "expected_probabilities_json": json.dumps(expected, sort_keys=True),
                    "reproduced_probabilities_json": json.dumps(observed, sort_keys=True),
                    "attention_kernel_probabilities_json": json.dumps(
                        attention_kernel_probabilities, sort_keys=True
                    ),
                }
            )
            if args.strict_reproduction and (
                observed_action != spec["expected_action"]
                or error > args.max_probability_error
            ):
                raise ValueError(
                    f"Immediate action readout did not reproduce {state}:"
                    f"{spec['position_index']} (argmax={observed_action}, max error={error:.6g})."
                )
            for layer in layers:
                aggregate = result["aggregates"][layer]
                count = int(spec["position_index"])
                if count:
                    tensors[f"layer_{layer}.sentence_mass"][
                        prompt_index, :, :count
                    ] = aggregate["sentence_mass"]
                    tensors[f"layer_{layer}.sentence_mean"][
                        prompt_index, :, :count
                    ] = aggregate["sentence_mean"]
                tensors[f"layer_{layer}.region_mass"][prompt_index] = aggregate["region_mass"]
                tensors[f"layer_{layer}.region_mean"][prompt_index] = aggregate["region_mean"]
                tensors[f"layer_{layer}.region_counts"][prompt_index] = aggregate["region_counts"]
        shard_path = shard_dir / f"{state}.safetensors"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = shard_path.with_suffix(".partial")
        save_file(
            tensors,
            str(temporary),
            metadata={
                "example_id": state,
                "region_names": json.dumps(REGION_NAMES),
                "query": "last immediate-action prompt token before action label",
            },
        )
        os.replace(temporary, shard_path)
        write_csv(record_dir / f"{state}.validation.csv", validation_rows)
        record = {
            "example_id": state,
            "trajectory_id": state_specs[0]["trajectory_id"],
            "step_index": state_specs[0]["step_index"],
            "prompts": n_prompts,
            "max_sentences": max_sentences,
            "runtime_seconds": time.perf_counter() - state_start,
            "shard_path": str(shard_path),
            "file_size_bytes": shard_path.stat().st_size,
            "sha256": sha256(shard_path),
        }
        write_json(record_dir / f"{state}.json", record)
        completed_new += 1
        manifest.update(
            {
                "completed_states": len(by_state) - len(pending_states) + completed_new,
                "pending_states": len(pending_states) - completed_new,
                "last_completed_state": state,
            }
        )
        write_json(output / "run_manifest.json", manifest)
        log(
            f"{completed_new}/{len(pending_states)} states; {n_prompts} prompts; "
            f"{record['runtime_seconds']:.1f}s"
        )
        del tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del model
    manifest["status"] = "completed"
    write_json(output / "run_manifest.json", manifest)
    return manifest


def materialize_region_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = [
        json.loads(path.read_text())
        for path in sorted((args.output_dir / "state_records").glob("*.json"))
    ]
    rows: list[dict[str, Any]] = []
    for record in records:
        validation = read_csv(
            args.output_dir / "state_records" / f"{record['example_id']}.validation.csv"
        )
        with safe_open(record["shard_path"], framework="pt", device="cpu") as handle:
            positions = handle.get_tensor("position_indices").tolist()
            for layer in args.layers:
                mass = handle.get_tensor(f"layer_{layer}.region_mass").float().mean(dim=1)
                mean = handle.get_tensor(f"layer_{layer}.region_mean").float().mean(dim=1)
                counts = handle.get_tensor(f"layer_{layer}.region_counts")
                for prompt_index, position in enumerate(positions):
                    metadata = validation[prompt_index]
                    for region_index, region in enumerate(REGION_NAMES):
                        count = int(counts[prompt_index, region_index].item())
                        rows.append(
                            {
                                "example_id": record["example_id"],
                                "trajectory_id": record["trajectory_id"],
                                "step_index": record["step_index"],
                                "position_index": int(position),
                                "layer": int(layer),
                                "region": region,
                                "region_label": REGION_LABELS[region],
                                "region_tokens": count,
                                "available": count > 0,
                                "attention_mass": float(mass[prompt_index, region_index].item())
                                if count
                                else "",
                                "mean_attention_per_token": float(
                                    mean[prompt_index, region_index].item()
                                )
                                if count
                                else "",
                                "expected_action": metadata["expected_action"],
                            }
                        )
    write_csv(args.output_dir / "attention_region_rows.csv", rows)
    return rows


def _event_labels(event_rows_path: Path) -> dict[tuple[str, int], set[str]]:
    labels: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in read_csv(event_rows_path):
        labels[(row["example_id"], int(row["reasoning_step_idx"]))].add(row["event_type"])
    return labels


def _js_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    left_values = np.asarray([float(left[label]) for label in ACTION_LABELS], dtype=float)
    right_values = np.asarray([float(right[label]) for label in ACTION_LABELS], dtype=float)
    left_values /= left_values.sum()
    right_values /= right_values.sum()
    midpoint = 0.5 * (left_values + right_values)

    def kl(values: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / midpoint[mask])))

    return 0.5 * (kl(left_values) + kl(right_values))


def add_replayed_action_changes(
    rows: list[dict[str, Any]],
    validation_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
) -> None:
    validation = {
        (row["example_id"], int(row["position_index"])): row
        for row in validation_rows
    }
    pairs: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in pair_rows:
        pairs[row["pair_id"]][row["item_role"]] = row
    pair_metadata: dict[str, dict[str, Any]] = {}
    for pair_id, roles in pairs.items():
        change = roles["detected_change_point"]
        control = roles["matched_non_change_sentence"]

        def read(
            item: dict[str, str], delta: int
        ) -> tuple[str, dict[str, float], bool]:
            key = (
                item["example_id"],
                max(0, int(item["position_index"]) + delta),
            )
            row = validation[key]
            return (
                row["reproduced_action"],
                json.loads(row["reproduced_probabilities_json"]),
                row["attention_kernel_argmax_matches"] == "True",
            )

        cp_before_action, cp_before_prob, cp_before_kernel_valid = read(change, -1)
        cp_action, cp_prob, cp_kernel_valid = read(change, 0)
        (
            control_before_action,
            control_before_prob,
            control_before_kernel_valid,
        ) = read(control, -1)
        control_action, control_prob, control_kernel_valid = read(control, 0)
        cp_changes = cp_before_action != cp_action
        control_changes = control_before_action != control_action
        kernel_valid = all(
            (
                cp_before_kernel_valid,
                cp_kernel_valid,
                control_before_kernel_valid,
                control_kernel_valid,
            )
        )
        cp_js = _js_divergence(cp_before_prob, cp_prob)
        control_js = _js_divergence(control_before_prob, control_prob)
        pair_metadata[pair_id] = {
            "replayed_change_point_previous_action": cp_before_action,
            "replayed_change_point_action": cp_action,
            "replayed_control_previous_action": control_before_action,
            "replayed_control_action": control_action,
            "replayed_recommendation_change": cp_changes,
            "replayed_control_recommendation_change": control_changes,
            "isolated_replayed_recommendation_change": cp_changes and not control_changes,
            "attention_kernel_valid_all_positions": kernel_valid,
            "kernel_valid_isolated_replayed_recommendation_change": (
                cp_changes and not control_changes and kernel_valid
            ),
            "replayed_change_point_js_divergence_bits": cp_js,
            "replayed_control_js_divergence_bits": control_js,
            "replayed_change_point_minus_control_js_divergence_bits": cp_js - control_js,
        }
    for row in rows:
        row.update(pair_metadata[row["pair_id"]])


def summarize_differences(
    rows: list[dict[str, Any]],
    *,
    events: dict[tuple[str, int], set[str]],
    repeats: int,
) -> list[dict[str, Any]]:
    for row in rows:
        event_types = events.get(
            (str(row["example_id"]), int(row["change_point_position"])), set()
        )
        row["exact_event_types"] = ";".join(sorted(event_types))
        row["is_recommendation_change"] = "action_change" in event_types
        row["is_optimality_loss"] = any(
            name in event_types
            for name in (
                "sustained_optimal_to_suboptimal",
                "transient_optimal_to_suboptimal",
            )
        )
        row["is_recovery"] = "suboptimal_to_optimal" in event_types
        row["is_commitment_onset"] = "commitment_onset" in event_types

    subsets = {
        "previously_detected_change_points": lambda row: True,
        "replayed_recommendation_change": lambda row: bool(
            row["replayed_recommendation_change"]
        ),
        "isolated_replayed_recommendation_change": lambda row: bool(
            row["isolated_replayed_recommendation_change"]
        ),
        "kernel_valid_isolated_replayed_recommendation_change": lambda row: bool(
            row["kernel_valid_isolated_replayed_recommendation_change"]
        ),
        "previously_labeled_recommendation_change": lambda row: bool(
            row["is_recommendation_change"]
        ),
        "optimality_loss": lambda row: bool(row["is_optimality_loss"]),
        "recovery": lambda row: bool(row["is_recovery"]),
        "commitment_onset": lambda row: bool(row["is_commitment_onset"]),
    }
    summaries = []
    for subset_name, include in subsets.items():
        selected = [row for row in rows if include(row)]
        grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            grouped[(int(row["layer"]), str(row["region"]), "all")].append(row)
            if row["match_quality"] == "primary":
                grouped[(int(row["layer"]), str(row["region"]), "primary")].append(row)
        for (layer, region, quality), group in sorted(grouped.items()):
            mean, low, high = trajectory_bootstrap_ci(
                group,
                value_key="difference_in_differences",
                repeats=repeats,
                seed=42 + layer + len(region) + len(subset_name),
            )
            summaries.append(
                {
                    "event_subset": subset_name,
                    "match_quality": quality,
                    "layer": layer,
                    "region": region,
                    "region_label": REGION_LABELS[region],
                    "pairs": len(group),
                    "states": len({row["example_id"] for row in group}),
                    "trajectories": len({row["trajectory_id"] for row in group}),
                    "mean_difference_in_differences": mean,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "fraction_pairs_positive": float(
                        np.mean([float(row["difference_in_differences"]) > 0 for row in group])
                    ),
                }
            )
    return summaries


def plot_primary(summary_rows: list[dict[str, Any]], output: Path) -> None:
    selected_regions = (
        "newest_reasoning_sentence",
        "earlier_reasoning",
        "grid",
        "agent_status",
        "chosen_move_cells",
        "task_objects",
        "action_question",
    )
    rows = [
        row
        for row in summary_rows
        if row["event_subset"]
        == "kernel_valid_isolated_replayed_recommendation_change"
        and row["match_quality"] == "primary"
        and int(row["layer"]) == 15
        and row["region"] in selected_regions
    ]
    order = {region: index for index, region in enumerate(selected_regions)}
    rows.sort(key=lambda row: order[row["region"]])
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update(
        {
            "font.family": "Arial" if "Arial" in available_fonts else "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    x = np.arange(len(rows))
    means = np.asarray([100 * float(row["mean_difference_in_differences"]) for row in rows])
    lows = np.asarray([100 * float(row["bootstrap_ci_low"]) for row in rows])
    highs = np.asarray([100 * float(row["bootstrap_ci_high"]) for row in rows])
    ax.errorbar(
        x,
        means,
        yerr=np.vstack([means - lows, highs - means]),
        fmt="o",
        color="#1f5a94",
        ecolor="#8bb7dc",
        capsize=4,
        markersize=7,
        linewidth=2,
    )
    ax.axhline(0, color="#64748b", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([row["region_label"] for row in rows], rotation=25, ha="right")
    ax.set_ylabel(
        "Additional change in mean attention per token\n"
        "relative to matched sentences (percentage points)"
    )
    ax.set_title("Attention Changes When the Recommended Action Changes")
    n_pairs = rows[0]["pairs"] if rows else 0
    n_states = rows[0]["states"] if rows else 0
    n_trajectories = rows[0]["trajectories"] if rows else 0
    ax.text(
        0.01,
        0.98,
        f"Layer 15; {n_pairs} matched pairs; {n_states} states; "
        f"{n_trajectories} trajectories",
        transform=ax.transAxes,
        va="top",
        color="#334155",
    )
    ax.grid(axis="y", color="#dbe7f1", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    (output / "figs").mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output / "figs" / "attention_changes_when_recommended_action_changes.png",
        dpi=220,
    )
    plt.close(fig)


def analyze(args: argparse.Namespace) -> None:
    region_rows = materialize_region_rows(args)
    pairs = read_csv(args.pair_rows)
    differences = paired_attention_differences(
        region_rows,
        pairs,
        metric="mean_attention_per_token",
    )
    event_labels = _event_labels(args.event_rows)
    validation_rows = []
    for path in sorted((args.output_dir / "state_records").glob("*.validation.csv")):
        validation_rows.extend(read_csv(path))
    add_replayed_action_changes(differences, validation_rows, pairs)
    summary = summarize_differences(
        differences,
        events=event_labels,
        repeats=args.bootstrap_repeats,
    )
    write_csv(args.output_dir / "paired_attention_differences.csv", differences)
    write_csv(args.output_dir / "attention_region_summary.csv", summary)
    write_csv(args.output_dir / "prompt_validation_rows.csv", validation_rows)
    plot_primary(summary, args.output_dir)

    primary = [
        row
        for row in summary
        if row["event_subset"]
        == "kernel_valid_isolated_replayed_recommendation_change"
        and row["match_quality"] == "primary"
        and int(row["layer"]) == 15
    ]
    report = [
        "# Immediate-Action Attention Analysis",
        "",
        "## Question",
        "",
        "Does attention at the immediate action readout change when the model's "
        "recommended action changes?",
        "",
        "## Method",
        "",
        "- Query position: the last prompt token immediately before the action label. "
        "The logits at this position score UP, DOWN, LEFT, and RIGHT.",
        "- Comparison: change in attention from sentence t-1 to t where the replayed "
        "recommended action changes, minus the same change at a progress-matched "
        "sentence from the same environment state where the replayed action is stable.",
        "- Primary layer: full-attention layer 15. Full-attention layer 23 is a "
        "prespecified sensitivity analysis.",
        "- Primary measure: attention per token, averaged across all attention heads.",
        "- Uncertainty: 95% bootstrap intervals over trajectories.",
        "- Interpretation: positive values mean attention increased more when the "
        "replayed recommended action changed than at its matched sentence. Attention "
        "is correlational and does not establish causal information use.",
        "",
        "## Coverage",
        "",
        f"- Extracted prompt positions: {len(validation_rows):,}",
        f"- Environment states: {len({row['example_id'] for row in validation_rows}):,}",
        f"- Trajectories: {len({row['trajectory_id'] for row in validation_rows}):,}",
        f"- Reproduced action argmax: "
        f"{sum(str(row['action_matches']).lower() == 'true' for row in validation_rows):,}"
        f"/{len(validation_rows):,}",
        f"- Previously detected pairs with a replayed recommendation change: "
        f"{len({row['pair_id'] for row in differences if row['replayed_recommendation_change']}):,}"
        f"/{len({row['pair_id'] for row in differences}):,}",
        f"- Pairs with a replayed recommendation change and a stable matched control: "
        f"{len({row['pair_id'] for row in differences if row['isolated_replayed_recommendation_change']}):,}",
        f"- Primary pairs after requiring eager-attention argmax agreement at all four "
        f"positions: "
        f"{len({row['pair_id'] for row in differences if row['kernel_valid_isolated_replayed_recommendation_change']}):,}",
        f"- Main reported pairs after also requiring a high-quality progress match: "
        f"{len({row['pair_id'] for row in differences if row['kernel_valid_isolated_replayed_recommendation_change'] and row['match_quality'] == 'primary'}):,}",
        "",
        "## Primary estimates",
        "",
        "| Prompt region | Matched pairs | Additional attention change (percentage points per token) | 95% bootstrap interval |",
        "|---|---:|---:|---:|",
    ]
    for row in sorted(primary, key=lambda item: REGION_NAMES.index(item["region"])):
        report.append(
            f"| {row['region_label']} | {row['pairs']} | "
            f"{100 * float(row['mean_difference_in_differences']):.4f} | "
            f"[{100 * float(row['bootstrap_ci_low']):.4f}, "
            f"{100 * float(row['bootstrap_ci_high']):.4f}] |"
        )
    report.extend(
        [
            "",
            "## Limitations",
            "",
        "- Candidate positions come from an offline Bayesian change-point detector "
        "applied to an earlier action-probability series. The primary subset requires "
        "a recommendation change under the current replay and a stable matched control.",
        "- The earlier probability vectors are not exactly reproducible in the current "
        "software environment: 597 of 634 action argmaxes agree. Primary event labels "
        "therefore use the current replay rather than the earlier labels.",
        "- Explicit attention requires GPT-OSS's eager attention implementation. The "
        "primary subset excludes a pair if eager attention changes the normal-kernel "
        "action argmax at any compared position.",
            "- The boundary is retrospective: its detection uses the full probability "
            "series, although each attention measurement uses only the reasoning revealed "
            "up to that sentence.",
            "- Mean attention per token controls for region length but not for all prompt "
            "composition effects.",
            "- Attention weight is not proof that the attended information affected the "
            "recommended action.",
        ]
    )
    (args.output_dir / "run_report.md").write_text("\n".join(report) + "\n")
    caption = (
        "# Figure Captions\n\n"
        "## Attention Changes When the Recommended Action Changes\n\n"
        "Change in attention from the immediate action-readout token to each prompt "
        "region when the replayed recommended action changes, relative to the same "
        "change at progress-matched sentences from the same environment state where "
        "the replayed action remains stable. The "
        "readout token is the last prompt token before the model assigns probabilities "
        "to UP, DOWN, LEFT, and RIGHT. Points show trajectory-level means at GPT-OSS-20B "
        "layer 15; bars show 95% trajectory-bootstrap intervals. Positive values mean "
        "that attention per token increased more when the recommended action changed. This "
        "association does not establish causal information use.\n"
    )
    (args.output_dir / "FIGURE_CAPTIONS.md").write_text(caption)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pair-rows", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--candidate-rows", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--position-rows", type=Path, default=DEFAULT_POSITIONS)
    parser.add_argument("--sentence-rows", type=Path, default=DEFAULT_SENTENCES)
    parser.add_argument("--event-rows", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_SNAPSHOT)
    parser.add_argument("--layers", type=int, nargs="+", default=[15, 23])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--forward-chunk-size", type=int, default=512)
    parser.add_argument("--max-probability-error", type=float, default=0.005)
    parser.add_argument(
        "--strict-reproduction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Abort when replayed action probabilities differ from the earlier run.",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--limit-states", type=int)
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("extract", "analyze", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    if args.stage in {"extract", "all"}:
        extract(args)
    if args.stage in {"analyze", "all"}:
        analyze(args)


if __name__ == "__main__":
    main()
