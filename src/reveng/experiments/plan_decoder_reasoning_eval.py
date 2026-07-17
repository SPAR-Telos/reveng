"""Published plan-decoder evaluation utilities.

This module is for the activation-level experiment:

    reasoning degrades long-horizon planning information while sharpening
    local action-readiness

It reuses the published Hugging Face decoder checkpoints, validates a local
activation-source table, and evaluates pre/post plan decodability on the same
states when real activation tensors are available.

Formal pipeline-level responsibilities:
- published plan-decoder download and checkpoint loading
- activation-row schema validation
- plan decoding and metric computation

Experiment-only responsibilities:
- writing demo/template artifacts that use a colleague-provided grid layout as
  metadata only
- writing run outputs for a specific activation-source slice
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
import torch
from torch import nn

from reveng.experiments.cognitive_map_probe_reasoning_eval import (
    DEFAULT_DOWNLOAD_CACHE_DIR,
    PUBLISHED_DECODER_PROBE_FILENAMES,
    PUBLISHED_DECODER_PROBE_REPO_ID,
    _render_original_grid,
    _render_original_grid,
    _download_published_decoder_probes,
    _fetch_published_probe_file_infos,
    _write_json,
    _write_markdown,
)


DEFAULT_OUTPUT_DIR = "data/plan_decoder_reasoning_eval"
PLAN_HORIZON = 10
PLAN_ACTION_LABELS = ["LEFT", "RIGHT", "UP", "DOWN", "__PAD__"]
PLAN_ACTION_NAME_TO_ID = {label: idx for idx, label in enumerate(PLAN_ACTION_LABELS)}
DEFAULT_PREFIX_LENGTHS = (1, 2, 3, 5, 10)
DEFAULT_WORKING_HEADROOM_GB = 2.0
DEFAULT_DOWNLOAD_SIZE_CAP_GB = 2.0
PUBLIC_ACTIVATIONS_REPO_ID = "project-telos/activations_test_full"
PUBLIC_TRAJECTORIES_REPO_ID = "project-telos/trajectories_test_full"
PUBLIC_ACTIVATIONS_REPO_ID = "project-telos/activations_test_full"
PUBLIC_TRAJECTORIES_REPO_ID = "project-telos/trajectories_test_full"
EXPECTED_PLAN_ACTIVATION_COLUMNS = (
    "example_id",
    "reasoning_split",
    "activation_path",
    "target_action_sequence_json",
    "grid_text",
)
COLLEAGUE_LAYOUT_GRID_TEXT = "\n".join(
    [
        "0 1 2 3 4 5 6 7 8",
        "0 # # # # # # # # #",
        "1 # _ _ _ # G _ _ #",
        "2 # _ _ _ # _ _ _ #",
        "3 # _ _ _ _ _ _ _ #",
        "4 # D # # # # # # #",
        "5 # _ _ _ # K _ _ #",
        "6 # _ _ _ # _ _ _ #",
        "7 # _ _ _ _ _ _ _ #",
        "8 # # # # # # # # #",
    ]
)


def _trajectory_filename(grid_size: int, trajectory_index: int) -> str:
    return f"size{grid_size}/together_ai_openai_gpt-oss-20b_size{grid_size}_comp0.0_{trajectory_index}.json"


def _load_public_trajectory(cache_dir: Path, grid_size: int, trajectory_index: int) -> tuple[str, dict[str, Any]] | None:
    filename = _trajectory_filename(grid_size, trajectory_index)
    try:
        local = hf_hub_download(
            repo_id=PUBLIC_TRAJECTORIES_REPO_ID,
            repo_type="dataset",
            filename=filename,
            cache_dir=str(cache_dir),
        )
    except EntryNotFoundError:
        return None
    return filename, json.load(open(local))


def _target_action_sequence_from_trajectory(steps: list[dict[str, Any]], start_step: int) -> list[str]:
    sequence: list[str] = []
    for step in steps[start_step : start_step + PLAN_HORIZON]:
        action = str(step.get("agent_action", "")).upper()
        if action not in PLAN_ACTION_NAME_TO_ID:
            break
        sequence.append(action)
    while len(sequence) < PLAN_HORIZON:
        sequence.append("__PAD__")
    return sequence


def _stack_public_token_activations(
    *,
    cache_dir: Path,
    grid_size: int,
    trajectory_filename: str,
    model_id: str,
    layer: int,
    step_index: int,
    reasoning_split: str,
    output_n_tokens: int,
) -> Path | None:
    traj_stem = Path(trajectory_filename).stem
    model_subdir = model_id.replace("/", "__")
    root = f"size{grid_size}/{traj_stem}/{model_subdir}/layer_{layer}/step_{step_index}"
    if reasoning_split == "pre":
        rels = [f"{root}/prompt_suffix/{i}.pt" for i in range(3)]
    else:
        rels = [f"{root}/output/{output_n_tokens - 16 + i}.pt" for i in range(3)]
    tensors: list[torch.Tensor] = []
    try:
        for rel in rels:
            local = hf_hub_download(
                repo_id=PUBLIC_ACTIVATIONS_REPO_ID,
                repo_type="dataset",
                filename=rel,
                cache_dir=str(cache_dir),
            )
            tensor = torch.load(local, map_location="cpu")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Expected tensor in {rel}")
            tensors.append(tensor.to(dtype=torch.float32).flatten())
    except EntryNotFoundError:
        return None
    packed = torch.stack(tensors, dim=0)
    out_path = cache_dir / "public_plan_decoder_packed" / f"{traj_stem}_step{step_index}_{reasoning_split}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed, out_path)
    return out_path


def build_public_plan_decoder_activation_rows(
    *,
    output_dir: Path,
    cache_dir: Path,
    grid_size: int = 7,
    max_trajectories: int = 5,
    max_steps_per_trajectory: int = 3,
    layer: int = 15,
) -> Path:
    rows: list[dict[str, Any]] = []
    for trajectory_index in range(max_trajectories):
        loaded = _load_public_trajectory(cache_dir, grid_size, trajectory_index)
        if loaded is None:
            continue
        trajectory_filename, payload = loaded
        steps = list(payload.get("steps", []))
        model_id = str(payload["model_params"]["model_id"])
        for step_index, step in enumerate(steps[:max_steps_per_trajectory]):
            target_actions = _target_action_sequence_from_trajectory(steps, step_index)
            grid_text = _render_original_grid(step["grid_state"])
            example_id = f"{Path(trajectory_filename).stem}_step{step_index}"
            for reasoning_split in ("pre", "post"):
                packed_path = _stack_public_token_activations(
                    cache_dir=cache_dir,
                    grid_size=grid_size,
                    trajectory_filename=trajectory_filename,
                    model_id=model_id,
                    layer=layer,
                    step_index=step_index,
                    reasoning_split=reasoning_split,
                    output_n_tokens=int(step["output_n_tokens"]),
                )
                if packed_path is None:
                    continue
                rows.append(
                    {
                        "example_id": example_id,
                        "reasoning_split": reasoning_split,
                        "activation_path": str(packed_path),
                        "target_action_sequence_json": json.dumps(target_actions),
                        "grid_text": grid_text,
                        "trajectory_id": Path(trajectory_filename).stem,
                        "step_index": str(step_index),
                        "observed_action": str(step.get("agent_action", "")).upper(),
                        "source_dataset": "public_plan_decoder_slice",
                    }
                )
    out_path = output_dir / "public_activation_rows.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        out_path.write_text("")
    return out_path


@dataclass(frozen=True)
class PlanDecoderEvaluationRow:
    example_id: str
    trajectory_id: str
    step_index: str
    reasoning_split: str
    activation_path: str
    grid_text: str
    observed_action: str
    source_dataset: str
    target_action_sequence_json: str
    target_action_names_json: str
    predicted_action_ids_json: str
    predicted_action_names_json: str
    next_action_correct: int
    prefix_1_exact: int
    prefix_2_exact: int
    prefix_3_exact: int
    prefix_5_exact: int
    prefix_10_exact: int


def _trajectory_filename(grid_size: int, trajectory_index: int) -> str:
    return f"size{grid_size}/together_ai_openai_gpt-oss-20b_size{grid_size}_comp0.0_{trajectory_index}.json"


def _load_public_trajectory(cache_dir: Path, grid_size: int, trajectory_index: int) -> tuple[str, dict[str, Any]] | None:
    filename = _trajectory_filename(grid_size, trajectory_index)
    try:
        local = hf_hub_download(
            repo_id=PUBLIC_TRAJECTORIES_REPO_ID,
            repo_type="dataset",
            filename=filename,
            cache_dir=str(cache_dir),
        )
    except EntryNotFoundError:
        return None
    return filename, json.load(open(local))


def _target_action_sequence_from_trajectory(steps: list[dict[str, Any]], start_step: int) -> list[str]:
    sequence: list[str] = []
    for step in steps[start_step : start_step + PLAN_HORIZON]:
        action = str(step.get("agent_action", "")).upper()
        if action not in PLAN_ACTION_NAME_TO_ID:
            break
        sequence.append(action)
    while len(sequence) < PLAN_HORIZON:
        sequence.append("__PAD__")
    return sequence


def _stack_public_token_activations(
    *,
    cache_dir: Path,
    grid_size: int,
    trajectory_filename: str,
    model_id: str,
    layer: int,
    step_index: int,
    reasoning_split: str,
    output_n_tokens: int,
) -> Path | None:
    traj_stem = Path(trajectory_filename).stem
    model_subdir = model_id.replace("/", "__")
    root = f"size{grid_size}/{traj_stem}/{model_subdir}/layer_{layer}/step_{step_index}"
    if reasoning_split == "pre":
        rels = [f"{root}/prompt_suffix/{i}.pt" for i in range(3)]
    else:
        rels = [f"{root}/output/{output_n_tokens - 16 + i}.pt" for i in range(3)]
    tensors: list[torch.Tensor] = []
    try:
        for rel in rels:
            local = hf_hub_download(
                repo_id=PUBLIC_ACTIVATIONS_REPO_ID,
                repo_type="dataset",
                filename=rel,
                cache_dir=str(cache_dir),
            )
            tensor = torch.load(local, map_location="cpu")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Expected tensor in {rel}")
            tensors.append(tensor.to(dtype=torch.float32).flatten())
    except EntryNotFoundError:
        return None
    packed = torch.stack(tensors, dim=0)
    out_path = cache_dir / "public_plan_decoder_packed" / f"{traj_stem}_step{step_index}_{reasoning_split}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed, out_path)
    return out_path


def build_public_plan_decoder_activation_rows(
    *,
    output_dir: Path,
    cache_dir: Path,
    grid_size: int = 7,
    max_trajectories: int = 5,
    max_steps_per_trajectory: int = 3,
    layer: int = 15,
) -> Path:
    rows: list[dict[str, Any]] = []
    for trajectory_index in range(max_trajectories):
        loaded = _load_public_trajectory(cache_dir, grid_size, trajectory_index)
        if loaded is None:
            continue
        trajectory_filename, payload = loaded
        steps = list(payload.get("steps", []))
        model_id = str(payload["model_params"]["model_id"])
        for step_index, step in enumerate(steps[:max_steps_per_trajectory]):
            target_actions = _target_action_sequence_from_trajectory(steps, step_index)
            grid_text = _render_original_grid(step["grid_state"])
            example_id = f"{Path(trajectory_filename).stem}_step{step_index}"
            for reasoning_split in ("pre", "post"):
                packed_path = _stack_public_token_activations(
                    cache_dir=cache_dir,
                    grid_size=grid_size,
                    trajectory_filename=trajectory_filename,
                    model_id=model_id,
                    layer=layer,
                    step_index=step_index,
                    reasoning_split=reasoning_split,
                    output_n_tokens=int(step["output_n_tokens"]),
                )
                if packed_path is None:
                    continue
                rows.append(
                    {
                        "example_id": example_id,
                        "reasoning_split": reasoning_split,
                        "activation_path": str(packed_path),
                        "target_action_sequence_json": json.dumps(target_actions),
                        "grid_text": grid_text,
                        "trajectory_id": Path(trajectory_filename).stem,
                        "step_index": str(step_index),
                        "observed_action": str(step.get("agent_action", "")).upper(),
                        "source_dataset": "public_plan_decoder_slice",
                    }
                )
    out_path = output_dir / "public_activation_rows.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        out_path.write_text("")
    return out_path


class PublishedPlanDecoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 2880,
        hidden_dim: int = 1024,
        horizon: int = PLAN_HORIZON,
        num_layers: int = 4,
        nhead: int = 8,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.num_classes = num_classes

        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=4096,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.path_queries = nn.Parameter(torch.zeros(horizon, hidden_dim))
        self.query_position_embeddings = nn.Parameter(torch.zeros(horizon, hidden_dim))
        self.action_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        if activations.ndim != 3:
            raise ValueError(
                f"Expected activations shape [batch, 3, {self.input_dim}], got {tuple(activations.shape)}"
            )
        if activations.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected activation width {self.input_dim}, got {activations.shape[-1]}"
            )
        memory = self.feature_map(activations)
        batch_size = memory.shape[0]
        queries = (self.path_queries + self.query_position_embeddings).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        decoded = self.decoder(tgt=queries, memory=memory)
        return self.action_head(decoded)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with open(path, newline="") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    raise ValueError(f"Unsupported activation rows file type: {path}")


def _validate_plan_activation_rows_schema(path: Path) -> dict[str, Any]:
    rows = _load_rows(path)
    if not rows:
        raise ValueError(f"No activation rows found in {path}")
    first_row = rows[0]
    missing = [column for column in EXPECTED_PLAN_ACTIVATION_COLUMNS if column not in first_row]
    return {
        "row_count": len(rows),
        "columns": list(first_row.keys()),
        "missing_required_columns": missing,
        "is_valid": not missing,
    }


def _resolve_checkpoint_paths(
    *,
    cache_dir: Path,
    revision: str | None,
    size_cap_gb: float,
    working_headroom_gb: float,
) -> tuple[Path, Path]:
    file_infos = _fetch_published_probe_file_infos(PUBLISHED_DECODER_PROBE_REPO_ID, revision=revision)
    snapshot_dir, _disk_check = _download_published_decoder_probes(
        repo_id=PUBLISHED_DECODER_PROBE_REPO_ID,
        cache_dir=cache_dir,
        revision=revision,
        file_infos=file_infos,
        size_cap_gb=size_cap_gb,
        working_headroom_gb=working_headroom_gb,
    )
    pre_path = snapshot_dir / PUBLISHED_DECODER_PROBE_FILENAMES[0]
    post_path = snapshot_dir / PUBLISHED_DECODER_PROBE_FILENAMES[1]
    return pre_path, post_path


def load_published_plan_decoder(checkpoint_path: Path, *, device: str = "cpu") -> PublishedPlanDecoder:
    state_dict = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state dict checkpoint at {checkpoint_path}")
    model = PublishedPlanDecoder()
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _parse_action_sequence(value: Any) -> list[int]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    if not isinstance(parsed, list):
        raise ValueError("target_action_sequence_json must decode to a list")
    action_ids: list[int] = []
    for item in parsed:
        if isinstance(item, int):
            action_id = int(item)
        elif isinstance(item, str):
            upper = item.strip().upper()
            if upper not in PLAN_ACTION_NAME_TO_ID:
                raise ValueError(f"Unknown action label {item!r}")
            action_id = PLAN_ACTION_NAME_TO_ID[upper]
        else:
            raise ValueError(f"Unsupported action token type: {type(item).__name__}")
        if action_id < 0 or action_id >= len(PLAN_ACTION_LABELS):
            raise ValueError(f"Action id {action_id} out of range")
        action_ids.append(action_id)
    if len(action_ids) != PLAN_HORIZON:
        raise ValueError(
            f"Expected exactly {PLAN_HORIZON} target actions, got {len(action_ids)}. "
            "This pipeline assumes the same fixed-horizon setup as the published plan decoder."
        )
    return action_ids


def _load_activation_tensor(path: Path, *, device: str = "cpu") -> torch.Tensor:
    payload = torch.load(path, map_location=device)
    tensor: torch.Tensor | None = None
    if isinstance(payload, torch.Tensor):
        tensor = payload
    elif isinstance(payload, dict):
        for key in ("activations", "activation", "hidden_states"):
            if isinstance(payload.get(key), torch.Tensor):
                tensor = payload[key]
                break
    if tensor is None:
        raise ValueError(f"Could not recover activation tensor from {path}")
    if tensor.ndim == 2:
        if tensor.shape != (3, 2880):
            raise ValueError(f"Expected activation tensor shape [3, 2880], got {tuple(tensor.shape)}")
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[1:] != (3, 2880) or tensor.shape[0] != 1:
            raise ValueError(f"Expected activation tensor shape [1, 3, 2880], got {tuple(tensor.shape)}")
    else:
        raise ValueError(f"Unsupported activation tensor rank: {tensor.ndim}")
    return tensor.to(device=device, dtype=torch.float32)


def _action_names(action_ids: Iterable[int]) -> list[str]:
    return [PLAN_ACTION_LABELS[int(idx)] for idx in action_ids]


def _prefix_exact(predicted_ids: list[int], target_ids: list[int], prefix_len: int) -> int:
    return int(predicted_ids[:prefix_len] == target_ids[:prefix_len])


def summarize_plan_decoder_rows(eval_rows: list[PlanDecoderEvaluationRow]) -> list[dict[str, Any]]:
    grouped: dict[str, list[PlanDecoderEvaluationRow]] = {"pre": [], "post": []}
    for row in eval_rows:
        grouped.setdefault(row.reasoning_split, []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for split, split_rows in grouped.items():
        if not split_rows:
            continue
        n = len(split_rows)
        summary_rows.append(
            {
                "reasoning_split": split,
                "n_rows": n,
                "next_action_accuracy": sum(r.next_action_correct for r in split_rows) / n,
                "prefix_1_accuracy": sum(r.prefix_1_exact for r in split_rows) / n,
                "prefix_2_accuracy": sum(r.prefix_2_exact for r in split_rows) / n,
                "prefix_3_accuracy": sum(r.prefix_3_exact for r in split_rows) / n,
                "prefix_5_accuracy": sum(r.prefix_5_exact for r in split_rows) / n,
                "prefix_10_accuracy": sum(r.prefix_10_exact for r in split_rows) / n,
            }
        )
    return summary_rows


def build_plan_decoder_hypothesis_summary(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = {str(row["reasoning_split"]): row for row in summary_rows}
    pre = by_split.get("pre")
    post = by_split.get("post")
    if pre is None or post is None:
        return {
            "status": "insufficient_splits",
            "hypothesis": (
                "reasoning degrades long-horizon planning information while sharpening local action-readiness"
            ),
        }
    local_delta = float(post["prefix_1_accuracy"]) - float(pre["prefix_1_accuracy"])
    long_horizon_delta = float(post["prefix_5_accuracy"]) - float(pre["prefix_5_accuracy"])
    longest_horizon_delta = float(post["prefix_10_accuracy"]) - float(pre["prefix_10_accuracy"])
    return {
        "status": "completed",
        "hypothesis": "reasoning degrades long-horizon planning information while sharpening local action-readiness",
        "local_action_readiness_delta_prefix_1": local_delta,
        "long_horizon_delta_prefix_5": long_horizon_delta,
        "longest_horizon_delta_prefix_10": longest_horizon_delta,
        "supports_local_sharpening": local_delta > 0.0,
        "supports_long_horizon_degradation_prefix_5": long_horizon_delta < 0.0,
        "supports_long_horizon_degradation_prefix_10": longest_horizon_delta < 0.0,
        "joint_supports_target_hypothesis": (
            local_delta > 0.0 and long_horizon_delta < 0.0 and longest_horizon_delta < 0.0
        ),
    }


def _plot_plan_decoder_summary(summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    by_split = {str(row["reasoning_split"]): row for row in summary_rows}
    if "pre" not in by_split or "post" not in by_split:
        return
    prefixes = [1, 2, 3, 5, 10]
    pre_vals = [float(by_split["pre"][f"prefix_{k}_accuracy"]) for k in prefixes]
    post_vals = [float(by_split["post"][f"prefix_{k}_accuracy"]) for k in prefixes]
    deltas = [post - pre for pre, post in zip(pre_vals, post_vals, strict=True)]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    axes[0].plot(prefixes, pre_vals, marker="o", label="Pre")
    axes[0].plot(prefixes, post_vals, marker="o", label="Post")
    axes[0].set_title("Plan Decoder Prefix Accuracy by Reasoning Split")
    axes[0].set_xlabel("Prefix length")
    axes[0].set_ylabel("Exact prefix accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend()

    axes[1].bar([str(k) for k in prefixes], deltas)
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_title("Post minus Pre Prefix Accuracy")
    axes[1].set_xlabel("Prefix length")
    axes[1].set_ylabel("Accuracy delta")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_plan_decoder_hypothesis_note(
    out_path: Path,
    summary_rows: list[dict[str, Any]],
    hypothesis_summary: dict[str, Any],
) -> None:
    by_split = {str(row["reasoning_split"]): row for row in summary_rows}
    lines = [
        "# Planning Hypothesis Summary",
        "",
        "Hypothesis tested:",
        "",
        "> reasoning degrades long-horizon planning information while sharpening local action-readiness",
        "",
    ]
    if "pre" in by_split and "post" in by_split:
        pre = by_split["pre"]
        post = by_split["post"]
        lines.extend(
            [
                f"- Prefix-1 accuracy: pre={float(pre['prefix_1_accuracy']):.3f}, post={float(post['prefix_1_accuracy']):.3f}",
                f"- Prefix-5 accuracy: pre={float(pre['prefix_5_accuracy']):.3f}, post={float(post['prefix_5_accuracy']):.3f}",
                f"- Prefix-10 accuracy: pre={float(pre['prefix_10_accuracy']):.3f}, post={float(post['prefix_10_accuracy']):.3f}",
                "",
                f"- Supports local sharpening: {hypothesis_summary.get('supports_local_sharpening')}",
                f"- Supports long-horizon degradation at prefix-5: {hypothesis_summary.get('supports_long_horizon_degradation_prefix_5')}",
                f"- Supports long-horizon degradation at prefix-10: {hypothesis_summary.get('supports_long_horizon_degradation_prefix_10')}",
                f"- Joint support for target hypothesis on this slice: {hypothesis_summary.get('joint_supports_target_hypothesis')}",
            ]
        )
    _write_markdown(out_path, "\n".join(lines) + "\n")


def evaluate_plan_decoder_rows(
    *,
    activation_rows_path: Path,
    pre_checkpoint_path: Path,
    post_checkpoint_path: Path,
    output_dir: Path,
    device: str = "cpu",
) -> None:
    rows = _load_rows(activation_rows_path)
    schema = _validate_plan_activation_rows_schema(activation_rows_path)
    if not schema["is_valid"]:
        raise ValueError(f"Activation row schema invalid: {schema['missing_required_columns']}")

    model_by_split = {
        "pre": load_published_plan_decoder(pre_checkpoint_path, device=device),
        "post": load_published_plan_decoder(post_checkpoint_path, device=device),
    }

    eval_rows: list[PlanDecoderEvaluationRow] = []
    for row in rows:
        split = str(row["reasoning_split"])
        if split not in model_by_split:
            raise ValueError(f"Unsupported reasoning_split: {split}")
        target_ids = _parse_action_sequence(row["target_action_sequence_json"])
        activation = _load_activation_tensor(Path(row["activation_path"]), device=device)
        with torch.no_grad():
            logits = model_by_split[split](activation)
        predicted_ids = logits.argmax(dim=-1).squeeze(0).tolist()
        if not isinstance(predicted_ids, list):
            predicted_ids = [int(predicted_ids)]
        eval_rows.append(
            PlanDecoderEvaluationRow(
                example_id=str(row["example_id"]),
                trajectory_id=str(row.get("trajectory_id", "")),
                step_index=str(row.get("step_index", "")),
                reasoning_split=split,
                activation_path=str(row["activation_path"]),
                grid_text=str(row.get("grid_text", "")),
                observed_action=str(row.get("observed_action", "")),
                source_dataset=str(row.get("source_dataset", "")),
                target_action_sequence_json=json.dumps(target_ids),
                target_action_names_json=json.dumps(_action_names(target_ids)),
                predicted_action_ids_json=json.dumps(predicted_ids),
                predicted_action_names_json=json.dumps(_action_names(predicted_ids)),
                next_action_correct=_prefix_exact(predicted_ids, target_ids, 1),
                prefix_1_exact=_prefix_exact(predicted_ids, target_ids, 1),
                prefix_2_exact=_prefix_exact(predicted_ids, target_ids, 2),
                prefix_3_exact=_prefix_exact(predicted_ids, target_ids, 3),
                prefix_5_exact=_prefix_exact(predicted_ids, target_ids, 5),
                prefix_10_exact=_prefix_exact(predicted_ids, target_ids, 10),
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if eval_rows:
        with open(output_dir / "plan_decoder_eval_rows.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(eval_rows[0]).keys()))
            writer.writeheader()
            for row in eval_rows:
                writer.writerow(asdict(row))

    summary_rows = summarize_plan_decoder_rows(eval_rows)
    if summary_rows:
        with open(output_dir / "plan_decoder_eval_summary.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    if {"pre", "post"}.issubset({row.reasoning_split for row in eval_rows}):
        by_example: dict[str, dict[str, PlanDecoderEvaluationRow]] = {}
        for row in eval_rows:
            by_example.setdefault(row.example_id, {})[row.reasoning_split] = row
        comparison_rows: list[dict[str, Any]] = []
        for example_id, split_rows in sorted(by_example.items()):
            if "pre" not in split_rows or "post" not in split_rows:
                continue
            pre_row = split_rows["pre"]
            post_row = split_rows["post"]
            comparison_rows.append(
                {
                    "example_id": example_id,
                    "pre_next_action_accuracy": float(pre_row.next_action_correct),
                    "post_next_action_accuracy": float(post_row.next_action_correct),
                    "delta_next_action_accuracy": float(post_row.next_action_correct - pre_row.next_action_correct),
                    "pre_prefix_10_accuracy": float(pre_row.prefix_10_exact),
                    "post_prefix_10_accuracy": float(post_row.prefix_10_exact),
                    "delta_prefix_10_accuracy": float(post_row.prefix_10_exact - pre_row.prefix_10_exact),
                }
            )
        if comparison_rows:
            with open(output_dir / "plan_decoder_eval_pre_post_comparison.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
                writer.writeheader()
                writer.writerows(comparison_rows)
    hypothesis_summary = build_plan_decoder_hypothesis_summary(summary_rows)
    _write_json(output_dir / "plan_decoder_hypothesis_summary.json", hypothesis_summary)
    _write_plan_decoder_hypothesis_note(
        output_dir / "plan_decoder_hypothesis_summary.md",
        summary_rows,
        hypothesis_summary,
    )
    _plot_plan_decoder_summary(summary_rows, output_dir / "figs" / "plan_decoder_prefix_accuracy.png")
    _write_json(output_dir / "plan_decoder_eval_status.json", {"status": "completed", "n_rows": len(eval_rows)})


def run_plan_decoder_reasoning_eval(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    activation_rows_path: str | None = None,
    cache_dir: str = DEFAULT_DOWNLOAD_CACHE_DIR,
    probe_revision: str | None = None,
    download_size_cap_gb: float = DEFAULT_DOWNLOAD_SIZE_CAP_GB,
    working_headroom_gb: float = DEFAULT_WORKING_HEADROOM_GB,
    grid_size: int = 7,
    max_trajectories: int = 5,
    max_steps_per_trajectory: int = 3,
    layer: int = 15,
    device: str = "cpu",
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    pre_checkpoint_path, post_checkpoint_path = _resolve_checkpoint_paths(
        cache_dir=cache_root,
        revision=probe_revision,
        size_cap_gb=download_size_cap_gb,
        working_headroom_gb=working_headroom_gb,
    )

    if activation_rows_path is None:
        activation_rows_file = build_public_plan_decoder_activation_rows(
            output_dir=out_dir,
            cache_dir=cache_root,
            grid_size=grid_size,
            max_trajectories=max_trajectories,
            max_steps_per_trajectory=max_steps_per_trajectory,
            layer=layer,
        )
        if not activation_rows_file.exists() or not activation_rows_file.read_text().strip():
            _write_json(
                out_dir / "status.json",
                {
                    "status": "waiting_for_activation_rows",
                    "reason": "No usable released public activation rows were available for the requested slice.",
                },
            )
            return
    else:
        activation_rows_file = Path(activation_rows_path)

    evaluate_plan_decoder_rows(
        activation_rows_path=activation_rows_file,
        pre_checkpoint_path=pre_checkpoint_path,
        post_checkpoint_path=post_checkpoint_path,
        output_dir=out_dir,
        device=device,
    )


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "PLAN_HORIZON",
    "PLAN_ACTION_LABELS",
    "PLAN_ACTION_NAME_TO_ID",
    "EXPECTED_PLAN_ACTIVATION_COLUMNS",
    "PlanDecoderEvaluationRow",
    "PublishedPlanDecoder",
    "build_public_plan_decoder_activation_rows",
    "evaluate_plan_decoder_rows",
    "load_published_plan_decoder",
    "run_plan_decoder_reasoning_eval",
]
