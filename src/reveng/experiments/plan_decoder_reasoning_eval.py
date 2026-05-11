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

import torch
from torch import nn

from reveng.experiments.cognitive_map_probe_reasoning_eval import (
    DEFAULT_DOWNLOAD_CACHE_DIR,
    PUBLISHED_DECODER_PROBE_FILENAMES,
    PUBLISHED_DECODER_PROBE_REPO_ID,
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


@dataclass(frozen=True)
class PlanDecoderEvaluationRow:
    example_id: str
    reasoning_split: str
    activation_path: str
    grid_text: str
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
        split = str(row["reasoning_split"]).strip().lower()
        if split not in model_by_split:
            raise ValueError(f"Unsupported reasoning_split {row['reasoning_split']!r}; expected pre/post")
        activation_tensor = _load_activation_tensor(Path(str(row["activation_path"])), device=device)
        target_ids = _parse_action_sequence(row["target_action_sequence_json"])
        with torch.no_grad():
            logits = model_by_split[split](activation_tensor)[0]
            predicted_ids = logits.argmax(dim=-1).tolist()
        eval_rows.append(
            PlanDecoderEvaluationRow(
                example_id=str(row["example_id"]),
                reasoning_split=split,
                activation_path=str(row["activation_path"]),
                grid_text=str(row["grid_text"]),
                target_action_sequence_json=json.dumps(target_ids),
                target_action_names_json=json.dumps(_action_names(target_ids)),
                predicted_action_ids_json=json.dumps(predicted_ids),
                predicted_action_names_json=json.dumps(_action_names(predicted_ids)),
                next_action_correct=int(predicted_ids[0] == target_ids[0]),
                prefix_1_exact=_prefix_exact(predicted_ids, target_ids, 1),
                prefix_2_exact=_prefix_exact(predicted_ids, target_ids, 2),
                prefix_3_exact=_prefix_exact(predicted_ids, target_ids, 3),
                prefix_5_exact=_prefix_exact(predicted_ids, target_ids, 5),
                prefix_10_exact=_prefix_exact(predicted_ids, target_ids, 10),
            )
        )

    fieldnames = list(asdict(eval_rows[0]).keys()) if eval_rows else list(PlanDecoderEvaluationRow.__dataclass_fields__.keys())
    with open(output_dir / "plan_decoder_eval_rows.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in eval_rows:
            writer.writerow(asdict(row))

    summary_rows: list[dict[str, Any]] = []
    for split in ("pre", "post"):
        split_rows = [row for row in eval_rows if row.reasoning_split == split]
        if not split_rows:
            continue
        denom = float(len(split_rows))
        summary_rows.append(
            {
                "reasoning_split": split,
                "n_examples": len(split_rows),
                "next_action_accuracy": sum(r.next_action_correct for r in split_rows) / denom,
                "prefix_1_accuracy": sum(r.prefix_1_exact for r in split_rows) / denom,
                "prefix_2_accuracy": sum(r.prefix_2_exact for r in split_rows) / denom,
                "prefix_3_accuracy": sum(r.prefix_3_exact for r in split_rows) / denom,
                "prefix_5_accuracy": sum(r.prefix_5_exact for r in split_rows) / denom,
                "prefix_10_accuracy": sum(r.prefix_10_exact for r in split_rows) / denom,
            }
        )

    with open(output_dir / "plan_decoder_eval_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["reasoning_split"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    paired: dict[str, dict[str, PlanDecoderEvaluationRow]] = {}
    for row in eval_rows:
        paired.setdefault(row.example_id, {})[row.reasoning_split] = row
    comparison_rows: list[dict[str, Any]] = []
    for example_id, split_rows in paired.items():
        if "pre" not in split_rows or "post" not in split_rows:
            continue
        pre = split_rows["pre"]
        post = split_rows["post"]
        comparison_rows.append(
            {
                "example_id": example_id,
                "delta_next_action_accuracy": post.next_action_correct - pre.next_action_correct,
                "delta_prefix_1": post.prefix_1_exact - pre.prefix_1_exact,
                "delta_prefix_2": post.prefix_2_exact - pre.prefix_2_exact,
                "delta_prefix_3": post.prefix_3_exact - pre.prefix_3_exact,
                "delta_prefix_5": post.prefix_5_exact - pre.prefix_5_exact,
                "delta_prefix_10": post.prefix_10_exact - pre.prefix_10_exact,
            }
        )
    if comparison_rows:
        with open(output_dir / "plan_decoder_eval_pre_post_comparison.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
            writer.writeheader()
            for row in comparison_rows:
                writer.writerow(row)


def _write_demo_layout_artifacts(output_dir: Path) -> None:
    demo_dir = output_dir / "demo_layout_only"
    demo_dir.mkdir(parents=True, exist_ok=True)
    (demo_dir / "colleague_layout_grid.txt").write_text(COLLEAGUE_LAYOUT_GRID_TEXT + "\n")
    template_rows = [
        {
            "example_id": "colleague_layout_demo_pre",
            "reasoning_split": "pre",
            "activation_path": "REQUIRED/path/to/pre_activation_tensor.pt",
            "target_action_sequence_json": "[0, 1, 2, 3, 0, 1, 2, 3, 0, 1]",
            "grid_text": COLLEAGUE_LAYOUT_GRID_TEXT,
            "notes": "Layout only. Add a full state snapshot and real activation tensor before running evaluation.",
        },
        {
            "example_id": "colleague_layout_demo_post",
            "reasoning_split": "post",
            "activation_path": "REQUIRED/path/to/post_activation_tensor.pt",
            "target_action_sequence_json": "[0, 1, 2, 3, 0, 1, 2, 3, 0, 1]",
            "grid_text": COLLEAGUE_LAYOUT_GRID_TEXT,
            "notes": "Layout only. Add a full state snapshot and real activation tensor before running evaluation.",
        },
    ]
    with open(demo_dir / "activation_rows_template.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(template_rows[0].keys()))
        writer.writeheader()
        for row in template_rows:
            writer.writerow(row)
    _write_markdown(
        demo_dir / "README.md",
        "\n".join(
            [
                "# Colleague layout demo",
                "",
                "This is a layout-only template built from the colleague-provided grid.",
                "It is not a runnable activation example by itself.",
                "",
                "To run the published plan decoder, each row still needs:",
                "- a real activation tensor shaped `[3, 2880]` from GPT-OSS-20B",
                "- a fixed 10-step target action sequence",
                "- a full state snapshot, not only a room layout",
            ]
        ),
    )


def run_plan_decoder_reasoning_eval(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    activation_rows_path: str | None = None,
    cache_dir: str = DEFAULT_DOWNLOAD_CACHE_DIR,
    probe_revision: str | None = None,
    device: str = "cpu",
    download_size_cap_gb: float = DEFAULT_DOWNLOAD_SIZE_CAP_GB,
    working_headroom_gb: float = DEFAULT_WORKING_HEADROOM_GB,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_path, post_path = _resolve_checkpoint_paths(
        cache_dir=Path(cache_dir),
        revision=probe_revision,
        size_cap_gb=download_size_cap_gb,
        working_headroom_gb=working_headroom_gb,
    )

    _write_demo_layout_artifacts(out_dir)

    metadata = {
        "repo_id": PUBLISHED_DECODER_PROBE_REPO_ID,
        "probe_revision": probe_revision,
        "pre_checkpoint_path": str(pre_path),
        "post_checkpoint_path": str(post_path),
        "assumed_label_mapping": {
            str(idx): label for idx, label in enumerate(PLAN_ACTION_LABELS)
        },
        "assumption_note": (
            "The published checkpoint has a 5-way output head. This pipeline assumes "
            "the first four classes match the repo action ids LEFT/RIGHT/UP/DOWN and "
            "the fifth class is a pad token."
        ),
    }
    _write_json(out_dir / "plan_decoder_eval_metadata.json", metadata)

    if activation_rows_path is None:
        _write_json(
            out_dir / "status.json",
            {
                "status": "waiting_for_activation_rows",
                "reason": (
                    "Published plan decoders were located and a layout-based demo template "
                    "was written, but no real activation rows were provided."
                ),
            },
        )
        return

    schema = _validate_plan_activation_rows_schema(Path(activation_rows_path))
    _write_json(out_dir / "activation_rows_schema.json", schema)
    if not schema["is_valid"]:
        _write_json(
            out_dir / "status.json",
            {
                "status": "invalid_activation_rows_schema",
                "missing_required_columns": schema["missing_required_columns"],
            },
        )
        return

    evaluate_plan_decoder_rows(
        activation_rows_path=Path(activation_rows_path),
        pre_checkpoint_path=pre_path,
        post_checkpoint_path=post_path,
        output_dir=out_dir,
        device=device,
    )
    _write_json(out_dir / "status.json", {"status": "ok"})


__all__ = [
    "COLLEAGUE_LAYOUT_GRID_TEXT",
    "DEFAULT_OUTPUT_DIR",
    "EXPECTED_PLAN_ACTIVATION_COLUMNS",
    "PLAN_ACTION_LABELS",
    "PLAN_HORIZON",
    "PublishedPlanDecoder",
    "_parse_action_sequence",
    "_validate_plan_activation_rows_schema",
    "evaluate_plan_decoder_rows",
    "load_published_plan_decoder",
    "run_plan_decoder_reasoning_eval",
]
