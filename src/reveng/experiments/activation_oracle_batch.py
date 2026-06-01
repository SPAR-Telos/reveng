from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from tqdm import tqdm

DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"
DEFAULT_ORACLE_LORA_PATH = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
DEFAULT_ORACLE_INPUT_TYPES = ("segment", "full_seq")
SUPPORTED_ORACLE_INPUT_TYPES = {"tokens", "segment", "full_seq"}
SPECIAL_TOKEN = " ?"
LAYER_COUNTS = {DEFAULT_MODEL_NAME: 36}


class EarlyStopException(Exception):
    """Stop a forward pass after the last requested layer has been captured."""


class ActivationOracleBatchRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_id: str
    target_messages_json: list[dict[str, str]]
    oracle_prompt: str
    ground_truth: str = ""
    behavioral_label: str = ""
    representational_label: str = ""
    example_id: str = ""
    trajectory_id: str = ""
    question_id: str = ""
    source_dataset: str = ""
    observed_action: str = ""
    blackbox_action: str = ""
    whitebox_prediction_pre: str = ""
    whitebox_prediction_post: str = ""
    reasoning_reveal_pct: int | None = None
    step_index: int | None = None
    segment_start: int | None = None
    segment_end: int | None = None
    token_start: int | None = None
    token_end: int | None = None
    oracle_input_types: list[str] | None = None
    target_lora_path: str | None = None
    notes: str = ""

    @field_validator("target_messages_json", mode="before")
    @classmethod
    def _parse_target_messages(cls, value: Any) -> list[dict[str, str]]:
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, list) or not value:
            raise ValueError("target_messages_json must be a non-empty list of chat messages")
        normalized: list[dict[str, str]] = []
        for idx, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(f"target_messages_json[{idx}] must be an object")
            role = item.get("role")
            content = item.get("content")
            if not isinstance(role, str) or not role.strip():
                raise ValueError(f"target_messages_json[{idx}].role must be a non-empty string")
            if not isinstance(content, str):
                raise ValueError(f"target_messages_json[{idx}].content must be a string")
            normalized.append({"role": role, "content": content})
        return normalized

    @field_validator("oracle_input_types", mode="before")
    @classmethod
    def _parse_oracle_input_types(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                value = json.loads(stripped)
            else:
                value = [part.strip() for part in stripped.split(",") if part.strip()]
        if not isinstance(value, list) or not value:
            raise ValueError("oracle_input_types must be a non-empty list when provided")
        return value

    @field_validator("oracle_prompt")
    @classmethod
    def _validate_oracle_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("oracle_prompt must be non-empty")
        return value

    @field_validator("oracle_input_types")
    @classmethod
    def _validate_oracle_input_types(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        for item in value:
            if item not in SUPPORTED_ORACLE_INPUT_TYPES:
                raise ValueError(
                    f"oracle_input_types contains unsupported item {item!r}; expected subset of {sorted(SUPPORTED_ORACLE_INPUT_TYPES)}"
                )
        return value


class FeatureResult(BaseModel):
    feature_idx: int
    api_response: str
    prompt: str
    meta_info: Mapping[str, Any] = Field(default_factory=dict)


class TrainingDataPoint(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    datapoint_type: str
    input_ids: list[int]
    labels: list[int]
    layer: int
    steering_vectors: torch.Tensor | None
    positions: list[int]
    feature_idx: int
    target_output: str
    target_input_ids: list[int] | None
    target_positions: list[int] | None
    ds_label: str | None
    meta_info: Mapping[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_target_alignment(self) -> "TrainingDataPoint":
        if self.steering_vectors is not None:
            if len(self.positions) != self.steering_vectors.shape[0]:
                raise ValueError("positions and steering_vectors must have the same length")
        else:
            if self.target_positions is None or self.target_input_ids is None:
                raise ValueError("target_* must be provided when steering_vectors is None")
            if len(self.positions) != len(self.target_positions):
                raise ValueError("positions and target_positions must have the same length")
        return self


class BatchData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    steering_vectors: list[torch.Tensor]
    positions: list[list[int]]
    feature_indices: list[int]


@dataclass
class OracleResults:
    oracle_lora_path: str | None
    target_lora_path: str | None
    target_prompt: str
    act_key: str
    oracle_prompt: str
    ground_truth: str
    num_tokens: int
    token_responses: list[Optional[str]]
    full_sequence_responses: list[str]
    segment_responses: list[str]
    target_input_ids: list[int]
    act_layer: int


def layer_percent_to_layer(model_name: str, layer_percent: int) -> int:
    if model_name not in LAYER_COUNTS:
        raise ValueError(f"Unsupported model for v1 activation oracle runner: {model_name}")
    max_layers = LAYER_COUNTS[model_name]
    return int(max_layers * (layer_percent / 100))


def validate_qwen_only_config(
    *,
    model_name: str,
    oracle_lora_path: str,
    allow_custom_override: bool = False,
) -> None:
    if model_name != DEFAULT_MODEL_NAME and not allow_custom_override:
        raise ValueError(
            f"v1 only supports model_name={DEFAULT_MODEL_NAME!r}. "
            "Pass allow_custom_override=True only if you intentionally want a custom setup."
        )
    if oracle_lora_path != DEFAULT_ORACLE_LORA_PATH and not allow_custom_override:
        raise ValueError(
            f"v1 only supports oracle_lora_path={DEFAULT_ORACLE_LORA_PATH!r}. "
            "Pass allow_custom_override=True only if you intentionally want a custom setup."
        )


def get_hf_submodule(model: Any, layer: int, use_lora: bool = False) -> Any:
    model_name = getattr(model.config, "_name_or_path", "")
    if DEFAULT_MODEL_NAME not in model_name and model_name != DEFAULT_MODEL_NAME:
        raise ValueError(f"Unsupported model path for v1 activation oracle runner: {model_name}")
    if use_lora:
        return model.base_model.model.model.layers[layer]
    return model.model.layers[layer]


def collect_activations_multiple_layers(
    model: Any,
    submodules: dict[int, Any],
    inputs_BL: dict[str, torch.Tensor],
    min_offset: int | None,
    max_offset: int | None,
) -> dict[int, torch.Tensor]:
    if min_offset is not None:
        if max_offset is None:
            raise ValueError("max_offset must be provided when min_offset is used")
        if not (max_offset < min_offset < 0):
            raise ValueError("Expected negative offsets with max_offset < min_offset < 0")
    elif max_offset is not None:
        raise ValueError("max_offset requires min_offset")

    activations_BLD_by_layer: dict[int, torch.Tensor] = {}
    module_to_layer = {submodule: layer for layer, submodule in submodules.items()}
    max_layer = max(submodules.keys())

    def gather_target_act_hook(module: Any, _inputs: Any, outputs: Any) -> None:
        layer = module_to_layer[module]
        if isinstance(outputs, tuple):
            activations_BLD_by_layer[layer] = outputs[0]
        else:
            activations_BLD_by_layer[layer] = outputs
        if min_offset is not None:
            activations_BLD_by_layer[layer] = activations_BLD_by_layer[layer][:, max_offset:min_offset, :]
        if layer == max_layer:
            raise EarlyStopException("Early stopping after capturing activations")

    handles = [submodule.register_forward_hook(gather_target_act_hook) for submodule in submodules.values()]
    try:
        with torch.no_grad():
            _ = model(**inputs_BL)
    except EarlyStopException:
        pass
    finally:
        for handle in handles:
            handle.remove()
    return activations_BLD_by_layer


@contextlib.contextmanager
def add_hook(module: Any, hook: Callable[..., Any]):
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_hf_activation_steering_hook(
    *,
    vectors: list[torch.Tensor],
    positions: list[list[int]],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable[..., Any]:
    if len(vectors) != len(positions):
        raise ValueError("vectors and positions must have matching batch size")
    if not vectors:
        raise ValueError("Empty batch")

    normed_list = [torch.nn.functional.normalize(v_b, dim=-1).detach() for v_b in vectors]

    def hook_fn(_module: Any, _input: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            resid_BLD, *rest = output
            output_is_tuple = True
        else:
            resid_BLD = output
            rest = []
            output_is_tuple = False

        B_actual, L, _d_model_actual = resid_BLD.shape
        if B_actual != len(vectors):
            raise ValueError(f"Batch mismatch: module B={B_actual}, provided vectors B={len(vectors)}")

        if L <= 1:
            return (resid_BLD, *rest) if output_is_tuple else resid_BLD

        for b in range(B_actual):
            pos_b = torch.tensor(positions[b], dtype=torch.long, device=device)
            if pos_b.numel() == 0:
                continue
            if int(pos_b.min().item()) < 0 or int(pos_b.max().item()) >= L:
                raise ValueError("Steering position out of range for residual sequence")
            orig_KD = resid_BLD[b, pos_b, :]
            norms_K1 = orig_KD.norm(dim=-1, keepdim=True)
            steered_KD = (normed_list[b] * norms_K1 * steering_coefficient).to(dtype)
            resid_BLD[b, pos_b, :] = steered_KD.detach() + orig_KD

        return (resid_BLD, *rest) if output_is_tuple else resid_BLD

    return hook_fn


def construct_batch(training_data: list[TrainingDataPoint], tokenizer: Any, device: torch.device) -> BatchData:
    max_length = max(len(dp.input_ids) for dp in training_data)
    batch_tokens: list[torch.Tensor] = []
    batch_labels: list[torch.Tensor] = []
    batch_attn_masks: list[torch.Tensor] = []
    batch_positions: list[list[int]] = []
    batch_steering_vectors: list[torch.Tensor] = []
    batch_feature_indices: list[int] = []

    for data_point in training_data:
        padding_length = max_length - len(data_point.input_ids)
        padding_tokens = [tokenizer.pad_token_id] * padding_length
        padded_input_ids = padding_tokens + data_point.input_ids
        padded_labels = [-100] * padding_length + data_point.labels

        input_ids = torch.tensor(padded_input_ids, dtype=torch.long, device=device)
        labels = torch.tensor(padded_labels, dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
        attn_mask[:padding_length] = False

        batch_tokens.append(input_ids)
        batch_labels.append(labels)
        batch_attn_masks.append(attn_mask)

        padded_positions = [p + padding_length for p in data_point.positions]
        steering_vectors = data_point.steering_vectors
        if steering_vectors is None:
            raise ValueError("steering_vectors must be materialized before construct_batch")
        batch_positions.append(padded_positions)
        batch_steering_vectors.append(steering_vectors.to(device))
        batch_feature_indices.append(data_point.feature_idx)

    return BatchData(
        input_ids=torch.stack(batch_tokens),
        labels=torch.stack(batch_labels),
        attention_mask=torch.stack(batch_attn_masks),
        steering_vectors=batch_steering_vectors,
        positions=batch_positions,
        feature_indices=batch_feature_indices,
    )


def get_prompt_tokens_only(training_data_point: TrainingDataPoint) -> TrainingDataPoint:
    prompt_tokens: list[int] = []
    prompt_labels: list[int] = []
    response_token_seen = False
    for idx, token_id in enumerate(training_data_point.input_ids):
        if training_data_point.labels[idx] != -100:
            response_token_seen = True
            continue
        if response_token_seen:
            raise ValueError("Response token seen before prompt tokens")
        prompt_tokens.append(token_id)
        prompt_labels.append(training_data_point.labels[idx])
    new_point = training_data_point.model_copy(deep=True)
    new_point.input_ids = prompt_tokens
    new_point.labels = prompt_labels
    return new_point


def materialize_missing_steering_vectors(
    batch_points: list[TrainingDataPoint],
    tokenizer: Any,
    model: Any,
) -> list[TrainingDataPoint]:
    to_fill = [(i, dp) for i, dp in enumerate(batch_points) if dp.steering_vectors is None]
    if not to_fill:
        return batch_points

    for _, dp in to_fill:
        if dp.target_input_ids is None or dp.target_positions is None:
            raise ValueError("Datapoint has steering_vectors=None but missing target")

    pad_id = tokenizer.pad_token_id
    targets = [list(dp.target_input_ids) for _, dp in to_fill]
    positions_per_item = [list(dp.target_positions) for _, dp in to_fill]
    max_len = max(len(c) for c in targets)

    input_ids_tensors: list[torch.Tensor] = []
    attn_masks_tensors: list[torch.Tensor] = []
    left_offsets: list[int] = []
    device = next(model.parameters()).device

    for c in targets:
        pad_len = max_len - len(c)
        input_ids_tensors.append(torch.tensor([pad_id] * pad_len + c, dtype=torch.long, device=device))
        attn_masks_tensors.append(torch.tensor([False] * pad_len + [True] * len(c), dtype=torch.bool, device=device))
        left_offsets.append(pad_len)

    inputs_BL = {
        "input_ids": torch.stack(input_ids_tensors, dim=0),
        "attention_mask": torch.stack(attn_masks_tensors, dim=0),
    }

    layers_needed = sorted({dp.layer for _, dp in to_fill})
    submodules = {layer: get_hf_submodule(model, layer, use_lora=True) for layer in layers_needed}

    was_training = model.training
    model.eval()
    with model.disable_adapter():
        acts_by_layer = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL=inputs_BL,
            min_offset=None,
            max_offset=None,
        )
    if was_training:
        model.train()

    new_batch = list(batch_points)
    for b, (idx, dp) in enumerate(to_fill):
        layer = dp.layer
        acts_BLD = acts_by_layer[layer]
        idxs = [p + left_offsets[b] for p in positions_per_item[b]]
        vectors = acts_BLD[b, idxs, :].detach().contiguous()
        dp_new = dp.model_copy(deep=True)
        dp_new.steering_vectors = vectors
        new_batch[idx] = dp_new
    return new_batch


def find_pattern_in_tokens(token_ids: list[int], special_token_str: str, num_positions: int, tokenizer: Any) -> list[int]:
    special_token_id = tokenizer.encode(special_token_str, add_special_tokens=False)
    if len(special_token_id) != 1:
        raise ValueError(f"Expected SPECIAL_TOKEN to map to one token, got {len(special_token_id)}")
    special_token_id = special_token_id[0]
    positions: list[int] = []
    for i, token_id in enumerate(token_ids):
        if len(positions) == num_positions:
            break
        if token_id == special_token_id:
            positions.append(i)
    if len(positions) != num_positions:
        raise ValueError(f"Expected {num_positions} positions, got {len(positions)}")
    if positions[-1] - positions[0] != num_positions - 1:
        raise ValueError(f"Positions are not consecutive: {positions}")
    return positions


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], **kwargs: Any) -> Any:
    call_kwargs = dict(kwargs)
    for removable in ("continue_final_message", "enable_thinking"):
        try:
            return tokenizer.apply_chat_template(messages, **call_kwargs)
        except TypeError:
            call_kwargs.pop(removable, None)
    return tokenizer.apply_chat_template(messages, **call_kwargs)


def create_training_datapoint(
    *,
    datapoint_type: str,
    prompt: str,
    target_response: str,
    layer: int,
    num_positions: int,
    tokenizer: Any,
    acts_BD: torch.Tensor | None,
    feature_idx: int,
    target_input_ids: list[int] | None = None,
    target_positions: list[int] | None = None,
    ds_label: str | None = None,
    meta_info: Mapping[str, Any] | None = None,
) -> TrainingDataPoint:
    meta_info = dict(meta_info or {})
    prefix = f"Layer: {layer}\n" + (SPECIAL_TOKEN * num_positions) + " \n"
    prompt = prefix + prompt
    input_messages = [{"role": "user", "content": prompt}]

    input_prompt_ids = _apply_chat_template(
        tokenizer,
        input_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    full_messages = input_messages + [{"role": "assistant", "content": target_response}]
    full_prompt_ids = _apply_chat_template(
        tokenizer,
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )

    assistant_start_idx = len(input_prompt_ids)
    labels = list(full_prompt_ids)
    for i in range(assistant_start_idx):
        labels[i] = -100

    positions = find_pattern_in_tokens(list(full_prompt_ids), SPECIAL_TOKEN, num_positions, tokenizer)

    if acts_BD is not None:
        acts_BD = acts_BD.cpu().clone().detach()

    return TrainingDataPoint(
        input_ids=list(full_prompt_ids),
        labels=labels,
        layer=layer,
        steering_vectors=acts_BD,
        positions=positions,
        feature_idx=feature_idx,
        target_output=target_response,
        datapoint_type=datapoint_type,
        target_input_ids=target_input_ids,
        target_positions=target_positions,
        ds_label=ds_label,
        meta_info=meta_info,
    )


@torch.no_grad()
def eval_features_batch(
    *,
    eval_batch: BatchData,
    model: Any,
    submodule: Any,
    tokenizer: Any,
    device: torch.device,
    dtype: torch.dtype,
    steering_coefficient: float,
    generation_kwargs: dict[str, Any],
) -> list[FeatureResult]:
    hook_fn = get_hf_activation_steering_hook(
        vectors=eval_batch.steering_vectors,
        positions=eval_batch.positions,
        steering_coefficient=steering_coefficient,
        device=device,
        dtype=dtype,
    )

    tokenized_input = {"input_ids": eval_batch.input_ids, "attention_mask": eval_batch.attention_mask}
    decoded_prompts = tokenizer.batch_decode(eval_batch.input_ids, skip_special_tokens=False)
    feature_results: list[FeatureResult] = []

    with add_hook(submodule, hook_fn):
        output_ids = model.generate(**tokenized_input, **generation_kwargs)

    generated_tokens = output_ids[:, eval_batch.input_ids.shape[1]:]
    decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    for i in range(len(eval_batch.feature_indices)):
        feature_results.append(
            FeatureResult(
                feature_idx=eval_batch.feature_indices[i],
                api_response=decoded_output[i],
                prompt=decoded_prompts[i],
            )
        )
    return feature_results


def _run_evaluation(
    *,
    eval_data: list[TrainingDataPoint],
    model: Any,
    tokenizer: Any,
    submodule: Any,
    device: torch.device,
    dtype: torch.dtype,
    lora_adapter_name: str | None,
    eval_batch_size: int,
    steering_coefficient: float,
    generation_kwargs: dict[str, Any],
) -> list[FeatureResult]:
    if lora_adapter_name is not None:
        model.set_adapter(lora_adapter_name)

    all_feature_results: list[FeatureResult] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(eval_data), eval_batch_size), desc="Evaluating model"):
            e_batch = eval_data[i : i + eval_batch_size]
            e_batch = [get_prompt_tokens_only(dp) for dp in e_batch]
            e_batch = materialize_missing_steering_vectors(e_batch, tokenizer, model)
            e_batch = construct_batch(e_batch, tokenizer, device)
            feature_results = eval_features_batch(
                eval_batch=e_batch,
                model=model,
                submodule=submodule,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
                steering_coefficient=steering_coefficient,
                generation_kwargs=generation_kwargs,
            )
            all_feature_results.extend(feature_results)

    for feature_result, eval_data_point in zip(all_feature_results, eval_data, strict=True):
        feature_result.meta_info = eval_data_point.meta_info
    return all_feature_results


def _move_tokenized_to_device(tokenized: Any, device: torch.device) -> dict[str, torch.Tensor]:
    if hasattr(tokenized, "to"):
        tokenized = tokenized.to(device)
    if not isinstance(tokenized, dict):
        tokenized = dict(tokenized)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in tokenized.items()
    }


def encode_formatted_prompts(tokenizer: Any, formatted_prompts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    tokenized = tokenizer(formatted_prompts, return_tensors="pt", add_special_tokens=False, padding=True)
    return _move_tokenized_to_device(tokenized, device)


def sanitize_lora_name(lora_path: str) -> str:
    return lora_path.replace("/", "__").replace(".", "_")


def load_lora_adapter(model: Any, lora_path: str) -> str:
    sanitized_lora_name = sanitize_lora_name(lora_path)
    peft_config = getattr(model, "peft_config", {})
    if sanitized_lora_name not in peft_config:
        model.load_adapter(lora_path, adapter_name=sanitized_lora_name, is_trainable=False, low_cpu_mem_usage=True)
    return sanitized_lora_name


def _create_oracle_inputs(
    *,
    acts_BLD_by_layer_dict: dict[int, torch.Tensor],
    target_input_ids: list[int],
    oracle_prompt: str,
    act_layer: int,
    prompt_layer: int,
    tokenizer: Any,
    segment_start_idx: int,
    segment_end_idx: int | None,
    token_start_idx: int,
    token_end_idx: int | None,
    oracle_input_types: list[str],
    segment_repeats: int,
    full_seq_repeats: int,
    batch_idx: int = 0,
    left_pad: int = 0,
    base_meta: dict[str, Any] | None = None,
) -> list[TrainingDataPoint]:
    training_data: list[TrainingDataPoint] = []
    num_tokens = len(target_input_ids)

    if "tokens" in oracle_input_types:
        token_start = token_start_idx
        token_end = num_tokens if token_end_idx is None else token_end_idx
        if token_start < 0:
            raise ValueError(f"token_start_idx ({token_start}) must be >= 0")
        if token_end > num_tokens:
            raise ValueError(f"token_end_idx ({token_end}) exceeds sequence length ({num_tokens}). Use None for 'to the end'.")
        if token_start >= token_end:
            raise ValueError(f"token_start_idx ({token_start}) must be < token_end_idx ({token_end})")
        for i in range(token_start, token_end):
            target_positions_rel = [i]
            target_positions_abs = [left_pad + i]
            acts_BD = acts_BLD_by_layer_dict[act_layer][batch_idx, target_positions_abs]
            meta = {"dp_kind": "tokens", "token_index": i}
            if base_meta:
                meta.update(base_meta)
            training_data.append(
                create_training_datapoint(
                    datapoint_type="N/A",
                    prompt=oracle_prompt,
                    target_response="N/A",
                    layer=prompt_layer,
                    num_positions=len(target_positions_rel),
                    tokenizer=tokenizer,
                    acts_BD=acts_BD,
                    feature_idx=-1,
                    target_input_ids=target_input_ids,
                    target_positions=target_positions_rel,
                    ds_label="N/A",
                    meta_info=meta,
                )
            )

    if "segment" in oracle_input_types:
        segment_start = segment_start_idx
        segment_end = num_tokens if segment_end_idx is None else segment_end_idx
        if segment_start < 0:
            raise ValueError(f"segment_start_idx ({segment_start}) must be >= 0")
        if segment_end > num_tokens:
            raise ValueError(f"segment_end_idx ({segment_end}) exceeds sequence length ({num_tokens}). Use None for 'to the end'.")
        if segment_start >= segment_end:
            raise ValueError(f"segment_start_idx ({segment_start}) must be < segment_end_idx ({segment_end})")
        for _ in range(segment_repeats):
            target_positions_rel = list(range(segment_start, segment_end))
            target_positions_abs = [left_pad + p for p in target_positions_rel]
            acts_BD = acts_BLD_by_layer_dict[act_layer][batch_idx, target_positions_abs]
            meta = {"dp_kind": "segment"}
            if base_meta:
                meta.update(base_meta)
            training_data.append(
                create_training_datapoint(
                    datapoint_type="N/A",
                    prompt=oracle_prompt,
                    target_response="N/A",
                    layer=prompt_layer,
                    num_positions=len(target_positions_rel),
                    tokenizer=tokenizer,
                    acts_BD=acts_BD,
                    feature_idx=-1,
                    target_input_ids=target_input_ids,
                    target_positions=target_positions_rel,
                    ds_label="N/A",
                    meta_info=meta,
                )
            )

    if "full_seq" in oracle_input_types:
        for _ in range(full_seq_repeats):
            target_positions_rel = list(range(len(target_input_ids)))
            target_positions_abs = [left_pad + p for p in target_positions_rel]
            acts_BD = acts_BLD_by_layer_dict[act_layer][batch_idx, target_positions_abs]
            meta = {"dp_kind": "full_seq"}
            if base_meta:
                meta.update(base_meta)
            training_data.append(
                create_training_datapoint(
                    datapoint_type="N/A",
                    prompt=oracle_prompt,
                    target_response="N/A",
                    layer=prompt_layer,
                    num_positions=len(target_positions_rel),
                    tokenizer=tokenizer,
                    acts_BD=acts_BD,
                    feature_idx=-1,
                    target_input_ids=target_input_ids,
                    target_positions=target_positions_rel,
                    ds_label="N/A",
                    meta_info=meta,
                )
            )

    return training_data


def _collect_target_activations(
    *,
    model: Any,
    inputs_BL: dict[str, torch.Tensor],
    act_layers: list[int],
    target_lora_path: str | None,
) -> dict[int, torch.Tensor]:
    submodules = {layer: get_hf_submodule(model, layer) for layer in act_layers}
    if target_lora_path is not None:
        adapter_name = load_lora_adapter(model, target_lora_path)
        model.enable_adapters()
        model.set_adapter(adapter_name)
        return collect_activations_multiple_layers(model=model, submodules=submodules, inputs_BL=inputs_BL, min_offset=None, max_offset=None)
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            return collect_activations_multiple_layers(model=model, submodules=submodules, inputs_BL=inputs_BL, min_offset=None, max_offset=None)
    return collect_activations_multiple_layers(model=model, submodules=submodules, inputs_BL=inputs_BL, min_offset=None, max_offset=None)


def run_oracle(
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    target_prompt: str,
    target_lora_path: str | None,
    oracle_prompt: str,
    oracle_lora_path: str | None,
    segment_start_idx: int = 0,
    segment_end_idx: int | None = None,
    token_start_idx: int = 0,
    token_end_idx: int | None = 1,
    oracle_input_types: list[str] | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    ground_truth: str = "",
    segment_repeats: int = 1,
    full_seq_repeats: int = 1,
    eval_batch_size: int = 32,
    layer_percent: int = 50,
    injection_layer: int = 1,
    steering_coefficient: float = 1.0,
) -> OracleResults:
    if oracle_input_types is None:
        oracle_input_types = list(DEFAULT_ORACLE_INPUT_TYPES)
    if generation_kwargs is None:
        generation_kwargs = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 50}

    dtype = torch.bfloat16
    model_name = model.config._name_or_path
    act_layer = layer_percent_to_layer(model_name, layer_percent)
    injection_submodule = get_hf_submodule(model, injection_layer)
    inputs_BL = encode_formatted_prompts(tokenizer=tokenizer, formatted_prompts=[target_prompt], device=device)
    acts_by_layer = _collect_target_activations(
        model=model,
        inputs_BL=inputs_BL,
        act_layers=[act_layer],
        target_lora_path=target_lora_path,
    )

    seq_len = int(inputs_BL["input_ids"].shape[1])
    attn = inputs_BL["attention_mask"][0]
    real_len = int(attn.sum().item())
    left_pad = seq_len - real_len
    target_input_ids = inputs_BL["input_ids"][0, left_pad:].tolist()

    base_meta = {
        "target_lora_path": target_lora_path,
        "target_prompt": target_prompt,
        "oracle_prompt": oracle_prompt,
        "ground_truth": ground_truth,
        "combo_index": 0,
        "act_key": "lora",
        "num_tokens": len(target_input_ids),
        "target_index_within_batch": 0,
    }
    oracle_inputs = _create_oracle_inputs(
        acts_BLD_by_layer_dict=acts_by_layer,
        target_input_ids=target_input_ids,
        oracle_prompt=oracle_prompt,
        act_layer=act_layer,
        prompt_layer=act_layer,
        tokenizer=tokenizer,
        segment_start_idx=segment_start_idx,
        segment_end_idx=segment_end_idx,
        token_start_idx=token_start_idx,
        token_end_idx=token_end_idx,
        oracle_input_types=oracle_input_types,
        segment_repeats=segment_repeats,
        full_seq_repeats=full_seq_repeats,
        batch_idx=0,
        left_pad=left_pad,
        base_meta=base_meta,
    )

    oracle_adapter_name = load_lora_adapter(model, oracle_lora_path) if oracle_lora_path is not None else None
    responses = _run_evaluation(
        eval_data=oracle_inputs,
        model=model,
        tokenizer=tokenizer,
        submodule=injection_submodule,
        device=device,
        dtype=dtype,
        lora_adapter_name=oracle_adapter_name,
        eval_batch_size=eval_batch_size,
        steering_coefficient=steering_coefficient,
        generation_kwargs=generation_kwargs,
    )

    token_responses: list[Optional[str]] = [None] * len(target_input_ids)
    segment_responses: list[str] = []
    full_seq_responses: list[str] = []
    for response in responses:
        dp_kind = response.meta_info.get("dp_kind")
        if dp_kind == "tokens":
            token_responses[int(response.meta_info["token_index"])] = response.api_response
        elif dp_kind == "segment":
            segment_responses.append(response.api_response)
        elif dp_kind == "full_seq":
            full_seq_responses.append(response.api_response)

    return OracleResults(
        oracle_lora_path=oracle_lora_path,
        target_lora_path=target_lora_path,
        target_prompt=target_prompt,
        act_key="lora",
        oracle_prompt=oracle_prompt,
        ground_truth=ground_truth,
        num_tokens=len(target_input_ids),
        token_responses=token_responses,
        full_sequence_responses=full_seq_responses,
        segment_responses=segment_responses,
        target_input_ids=target_input_ids,
        act_layer=act_layer,
    )


def format_target_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return _apply_chat_template(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
        continue_final_message=False,
    )


def _token_ids_for_prompt(tokenizer: Any, formatted_prompt: str) -> list[int]:
    tokenized = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = tokenized["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        return input_ids[0].tolist()
    return list(input_ids[0])


def resolve_oracle_input_types(
    row: ActivationOracleBatchRow,
    default_oracle_input_types: tuple[str, ...] | list[str] = DEFAULT_ORACLE_INPUT_TYPES,
) -> list[str]:
    resolved = list(row.oracle_input_types or list(default_oracle_input_types))
    if not resolved:
        raise ValueError("Resolved oracle_input_types is empty")
    for item in resolved:
        if item not in SUPPORTED_ORACLE_INPUT_TYPES:
            raise ValueError(f"Unsupported oracle_input_type: {item!r}")
    return resolved


def validate_row_token_selection(
    *,
    row: ActivationOracleBatchRow,
    token_count: int,
    oracle_input_types: list[str],
) -> None:
    if "segment" in oracle_input_types:
        if row.segment_start is None:
            raise ValueError("segment_start is required when oracle_input_types includes 'segment'")
        if row.segment_start < 0:
            raise ValueError("segment_start must be >= 0")
        segment_end = token_count if row.segment_end is None else row.segment_end
        if segment_end > token_count:
            raise ValueError(f"segment_end {segment_end} exceeds token_count {token_count}")
        if row.segment_start >= segment_end:
            raise ValueError("segment_start must be < segment_end")
    if "tokens" in oracle_input_types:
        if row.token_start is None:
            raise ValueError("token_start is required when oracle_input_types includes 'tokens'")
        if row.token_start < 0:
            raise ValueError("token_start must be >= 0")
        token_end = token_count if row.token_end is None else row.token_end
        if token_end > token_count:
            raise ValueError(f"token_end {token_end} exceeds token_count {token_count}")
        if row.token_start >= token_end:
            raise ValueError("token_start must be < token_end")


def format_token_selection_preview(
    *,
    tokenizer: Any,
    input_text: str,
    segment_start: int | None = None,
    segment_end: int | None = None,
    token_start: int | None = None,
    token_end: int | None = None,
) -> str:
    input_ids = _token_ids_for_prompt(tokenizer, input_text)
    num_tokens = len(input_ids)
    resolved_segment_end = num_tokens if segment_end is None and segment_start is not None else segment_end
    resolved_token_end = num_tokens if token_end is None and token_start is not None else token_end

    lines = ["Token selection preview:", "-" * 72]
    for i, token_id in enumerate(input_ids):
        token_str = tokenizer.decode([token_id]).replace("\n", "\\n").replace("\r", "\\r")
        in_segment = segment_start is not None and resolved_segment_end is not None and segment_start <= i < resolved_segment_end
        in_token = token_start is not None and resolved_token_end is not None and token_start <= i < resolved_token_end
        marker = "ST" if in_segment and in_token else " S" if in_segment else " T" if in_token else "  "
        lines.append(f"[{i:03d}] {marker} {token_str}")
    lines.append("-" * 72)
    lines.append(f"Token count: {num_tokens}")
    if segment_start is not None:
        lines.append(f"Segment range: {segment_start}:{resolved_segment_end}")
    if token_start is not None:
        lines.append(f"Token range: {token_start}:{resolved_token_end}")
    return "\n".join(lines)


def _failure_record(*, row_id: str | None, row_index: int, stage: str, error: str, raw_row: Any | None = None) -> dict[str, Any]:
    record = {
        "row_id": row_id,
        "row_index": row_index,
        "stage": stage,
        "status": "failed",
        "error": error,
    }
    if raw_row is not None:
        record["raw_row"] = raw_row
    return record


def load_activation_oracle_batch_rows(input_path: str | Path, limit: int | None = None) -> tuple[list[ActivationOracleBatchRow], list[dict[str, Any]]]:
    rows: list[ActivationOracleBatchRow] = []
    failures: list[dict[str, Any]] = []
    with Path(input_path).open() as handle:
        for row_index, line in enumerate(handle):
            if limit is not None and len(rows) >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                failures.append(_failure_record(row_id=None, row_index=row_index, stage="json_parse", error=str(exc), raw_row=stripped))
                continue
            try:
                rows.append(ActivationOracleBatchRow.model_validate(payload))
            except ValidationError as exc:
                failures.append(
                    _failure_record(
                        row_id=payload.get("row_id") if isinstance(payload, dict) else None,
                        row_index=row_index,
                        stage="row_validation",
                        error=exc.json(),
                        raw_row=payload,
                    )
                )
    return rows, failures


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def _load_qwen_model_and_tokenizer(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    load_in_8bit: bool = True,
    device_map: str = "auto",
) -> tuple[Any, Any, torch.device]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise ImportError(
            "peft is required for the activation oracle runner. Sync the project environment "
            "(for example with `uv run ...` after dependency resolution or `uv sync`) before using this command."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if not getattr(tokenizer, "pad_token_id", None):
        tokenizer.pad_token_id = tokenizer.eos_token_id

    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=quantization_config,
        torch_dtype=dtype,
    )
    model.eval()
    if not hasattr(model, "add_adapter"):
        raise RuntimeError(
            "Loaded model does not expose the adapter interface required by the activation oracle runner"
        )
    if not hasattr(model, "peft_config") or "default" not in model.peft_config:
        dummy_config = LoraConfig()
        model.add_adapter(dummy_config, adapter_name="default")
    return model, tokenizer, device


def run_activation_oracle_batch(
    input_path: str,
    output_dir: str,
    model_name: str = DEFAULT_MODEL_NAME,
    oracle_lora_path: str = DEFAULT_ORACLE_LORA_PATH,
    default_oracle_input_types: tuple[str, ...] = DEFAULT_ORACLE_INPUT_TYPES,
    generation_max_new_tokens: int = 50,
    eval_batch_size: int = 32,
    layer_percent: int = 50,
    injection_layer: int = 1,
    steering_coefficient: float = 1.0,
    limit: int | None = None,
    load_in_8bit: bool = True,
    device_map: str = "auto",
    allow_custom_override: bool = False,
) -> dict[str, Any]:
    validate_qwen_only_config(
        model_name=model_name,
        oracle_lora_path=oracle_lora_path,
        allow_custom_override=allow_custom_override,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed_rows, failures = load_activation_oracle_batch_rows(input_path, limit=limit)
    results_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []

    if parsed_rows:
        model, tokenizer, device = _load_qwen_model_and_tokenizer(
            model_name=model_name,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
        )
        load_lora_adapter(model, oracle_lora_path)

        generation_kwargs = {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": generation_max_new_tokens,
        }

        for row_index, row in enumerate(parsed_rows):
            row_dump = row.model_dump(mode="json")
            try:
                resolved_types = resolve_oracle_input_types(row, default_oracle_input_types)
                formatted_target_prompt = format_target_prompt(tokenizer, row.target_messages_json)
                target_input_ids = _token_ids_for_prompt(tokenizer, formatted_target_prompt)
                token_count = len(target_input_ids)
                validate_row_token_selection(row=row, token_count=token_count, oracle_input_types=resolved_types)
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    _failure_record(
                        row_id=row.row_id,
                        row_index=row_index,
                        stage="resolved_validation",
                        error=str(exc),
                        raw_row=row_dump,
                    )
                )
                continue

            try:
                oracle_results = run_oracle(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    target_prompt=formatted_target_prompt,
                    target_lora_path=row.target_lora_path,
                    oracle_prompt=row.oracle_prompt,
                    oracle_lora_path=oracle_lora_path,
                    generation_kwargs=generation_kwargs,
                    ground_truth=row.ground_truth,
                    token_start_idx=row.token_start or 0,
                    token_end_idx=row.token_end,
                    segment_start_idx=row.segment_start or 0,
                    segment_end_idx=row.segment_end,
                    oracle_input_types=resolved_types,
                    eval_batch_size=eval_batch_size,
                    layer_percent=layer_percent,
                    injection_layer=injection_layer,
                    steering_coefficient=steering_coefficient,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    _failure_record(
                        row_id=row.row_id,
                        row_index=row_index,
                        stage="inference",
                        error=str(exc),
                        raw_row=row_dump,
                    )
                )
                continue

            result_record = {
                **row_dump,
                "formatted_target_prompt": formatted_target_prompt,
                "token_count": token_count,
                "resolved_oracle_input_types": resolved_types,
                "resolved_act_layer": oracle_results.act_layer,
                "segment_responses": oracle_results.segment_responses,
                "full_sequence_responses": oracle_results.full_sequence_responses,
                "status": "success",
                "warning_text": "",
            }
            results_rows.append(result_record)

            if "tokens" in resolved_types:
                for token_index, response in enumerate(oracle_results.token_responses):
                    if response is None:
                        continue
                    token_id = oracle_results.target_input_ids[token_index]
                    token_rows.append(
                        {
                            "row_id": row.row_id,
                            "token_index": token_index,
                            "token_id": token_id,
                            "token_text": tokenizer.decode([token_id]),
                            "oracle_response": response,
                        }
                    )

    _write_jsonl(out_dir / "results.jsonl", results_rows)
    _write_jsonl(out_dir / "token_responses.jsonl", token_rows)
    _write_jsonl(out_dir / "failures.jsonl", failures)

    manifest = {
        "input_path": str(input_path),
        "output_dir": str(out_dir),
        "model_name": model_name,
        "oracle_lora_path": oracle_lora_path,
        "default_oracle_input_types": list(default_oracle_input_types),
        "generation_max_new_tokens": generation_max_new_tokens,
        "eval_batch_size": eval_batch_size,
        "layer_percent": layer_percent,
        "injection_layer": injection_layer,
        "steering_coefficient": steering_coefficient,
        "limit": limit,
        "load_in_8bit": load_in_8bit,
        "device_map": device_map,
        "allow_custom_override": allow_custom_override,
        "total_rows_seen": len(parsed_rows) + len(failures),
        "parsed_rows": len(parsed_rows),
        "succeeded_rows": len(results_rows),
        "failed_rows": len(failures),
    }
    _write_json(out_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def preview_activation_oracle_tokens(
    input_path: str,
    row_id: str | None = None,
    row_index: int | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    allow_custom_override: bool = False,
) -> dict[str, Any]:
    validate_qwen_only_config(
        model_name=model_name,
        oracle_lora_path=DEFAULT_ORACLE_LORA_PATH,
        allow_custom_override=allow_custom_override,
    )

    rows, failures = load_activation_oracle_batch_rows(input_path)
    if failures:
        print(json.dumps({"parse_failures": failures}, indent=2))
    if row_id is not None:
        matched = [row for row in rows if row.row_id == row_id]
        if not matched:
            raise ValueError(f"row_id {row_id!r} not found in {input_path}")
        row = matched[0]
    else:
        idx = 0 if row_index is None else row_index
        if idx < 0 or idx >= len(rows):
            raise ValueError(f"row_index {idx} out of range for {len(rows)} parsed rows")
        row = rows[idx]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if not getattr(tokenizer, "pad_token_id", None):
        tokenizer.pad_token_id = tokenizer.eos_token_id

    formatted_target_prompt = format_target_prompt(tokenizer, row.target_messages_json)
    token_count = len(_token_ids_for_prompt(tokenizer, formatted_target_prompt))
    resolved_types = resolve_oracle_input_types(row, DEFAULT_ORACLE_INPUT_TYPES)
    validate_row_token_selection(row=row, token_count=token_count, oracle_input_types=resolved_types)
    preview = format_token_selection_preview(
        tokenizer=tokenizer,
        input_text=formatted_target_prompt,
        segment_start=row.segment_start,
        segment_end=row.segment_end,
        token_start=row.token_start,
        token_end=row.token_end,
    )
    print(preview)
    summary = {
        "row_id": row.row_id,
        "token_count": token_count,
        "resolved_oracle_input_types": resolved_types,
        "formatted_target_prompt": formatted_target_prompt,
    }
    print(json.dumps(summary, indent=2))
    return summary


__all__ = [
    "ActivationOracleBatchRow",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_ORACLE_INPUT_TYPES",
    "DEFAULT_ORACLE_LORA_PATH",
    "OracleResults",
    "format_target_prompt",
    "format_token_selection_preview",
    "load_activation_oracle_batch_rows",
    "preview_activation_oracle_tokens",
    "resolve_oracle_input_types",
    "run_activation_oracle_batch",
    "run_oracle",
    "sanitize_lora_name",
    "validate_qwen_only_config",
    "validate_row_token_selection",
]
