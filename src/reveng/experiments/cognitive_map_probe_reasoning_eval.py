"""Public cognitive-map probe evaluation against released activations and trajectories.

Formal pipeline-level responsibilities in this module:
- disk-aware Hugging Face download for released cognitive probes, trajectories,
  and activation tensors
- strict compatibility checks between probe checkpoints, trajectory metadata,
  layer/stage selection, and token-position conventions
- public cognitive-map probe loading and one-slice evaluation utilities

Experiment-only outputs written by the CLI command:
- download/compatibility manifests under the chosen output directory
- decoded-map example files for a chosen released trajectory step
- metric summaries for the evaluated released slice
"""

from __future__ import annotations

import csv
import heapq
import json
import os
from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path
from shutil import disk_usage
from typing import Any, Sequence

from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url, snapshot_download

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - import-time environment variability only
    torch = None
    nn = None


# Shared download helpers retained because the published plan-decoder pipeline imports them.
PUBLISHED_DECODER_PROBE_REPO_ID = "project-telos/decoder_probes"
PUBLISHED_DECODER_PROBE_FILENAMES = (
    "decoder_probe_layer15_pre_reasoning.pt",
    "decoder_probe_layer15_post_reasoning.pt",
)
DEFAULT_DOWNLOAD_CACHE_DIR = "data/hf/cache"
DEFAULT_OUTPUT_DIR = "data/cognitive_map_probe_reasoning_eval"
DEFAULT_DOWNLOAD_SIZE_CAP_GB = 2.0
DEFAULT_WORKING_HEADROOM_GB = 2.0
EXPECTED_ACTIVATION_ROW_COLUMNS = (
    "example_id",
    "reasoning_split",
    "activation_path",
    "grid_text",
)

# Public cognitive-map release constants.
PUBLIC_COGNITIVE_PROBE_REPO_ID = "project-telos/cognitive_map_probes"
PUBLIC_ACTIVATIONS_REPO_ID = "project-telos/activations_test_full"
PUBLIC_TRAJECTORIES_REPO_ID = "project-telos/trajectories_test_full"
PUBLIC_RESULTS_REPO_ID = "project-telos/cognitive_map_probes_results"
DEFAULT_PROBE_LAYER = 15
DEFAULT_PROBE_MODEL_TYPE = "mlp"
DEFAULT_PROBE_SCOPE = "general"
DEFAULT_GRID_SIZE = 7
DEFAULT_TRAJECTORY_INDEX = 0
DEFAULT_STEP_INDEX = 0
DEFAULT_PAD_TO_SIZE = 15
DEFAULT_MAX_TRAJECTORIES = 3
DEFAULT_MAX_STEPS_PER_TRAJECTORY = 2
DEFAULT_REASONING_STAGES = ("pre_reasoning", "post_reasoning")
DEFAULT_COORDINATE_ORDER = "row_col"
# Inferred from the released evaluation text for the public cognitive-map probes.
# The checkpoint maps prediction indices -> raw labels {0,1,2,3,7}; the released
# eval text reports those classes as A, #, G, _, + respectively.
INFERRED_RAW_LABEL_TO_SYMBOL = {
    0: "A",
    1: "#",
    2: "G",
    3: "_",
    7: "+",
}
INFERRED_SYMBOL_TO_SEMANTIC_NAME = {
    "A": "agent",
    "#": "wall",
    "G": "goal",
    "_": "open",
    "+": "padding",
}
ACTION_ID_TO_NAME = {
    0: "LEFT",
    1: "RIGHT",
    2: "UP",
    3: "DOWN",
}
ACTION_NAME_TO_DELTA = {
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "UP": (0, -1),
    "DOWN": (0, 1),
}


@dataclass(frozen=True)
class PublishedProbeFileInfo:
    filename: str
    size_bytes: int | None


@dataclass(frozen=True)
class DiskCheckResult:
    target_dir: str
    free_bytes: int
    required_bytes: int
    estimated_download_bytes: int
    headroom_bytes: int
    allowed: bool


@dataclass(frozen=True)
class ProbeCheckpointInspection:
    filename: str
    num_state_dict_keys: int
    top_level_keys_preview: list[str]
    feature_input_dim: int | None
    query_count: int | None
    output_class_count: int | None
    guessed_artifact_family: str
    compatibility_reason: str


@dataclass(frozen=True)
class CognitiveProbeCheckpointInfo:
    filename: str
    model_type: str
    input_dim: int
    num_classes: int
    hidden_dims: list[int]
    dropout: float
    idx_to_label: dict[int, int]
    raw_label_to_symbol: dict[int, str]
    raw_label_to_name: dict[int, str]
    scaler_mean_shape: tuple[int, ...]
    scaler_std_shape: tuple[int, ...]


@dataclass(frozen=True)
class CompatibilityReport:
    trajectory_filename: str
    trajectory_stem: str
    grid_size: int
    step_index: int
    model_id: str
    layer: int
    prompt_suffix_n_tokens: int
    output_n_tokens: int
    expected_pre_token_ids: list[int]
    observed_pre_token_ids: list[int]
    expected_post_token_ids: list[int]
    observed_post_token_ids: list[int]
    pre_activation_dim: int
    post_activation_dim: int
    probe_input_dim: int
    expected_probe_input_dim: int
    coordinate_order: str
    pre_feature_dim_matches_probe: bool
    post_feature_dim_matches_probe: bool
    grid_fits_probe_pad: bool
    stage_contract_matches_release: bool


@dataclass(frozen=True)
class StageEvaluationRow:
    example_id: str
    reasoning_stage: str
    trajectory_filename: str
    trajectory_stem: str
    step_index: int
    grid_size: int
    layer: int
    overall_cell_accuracy: float
    wall_cell_accuracy: float
    agent_location_exact: int
    goal_location_exact: int
    truth_agent_x: int
    truth_agent_y: int
    pred_agent_x: int | None
    pred_agent_y: int | None
    truth_goal_x: int
    truth_goal_y: int
    pred_goal_x: int | None
    pred_goal_y: int | None
    truth_wall_left: int
    pred_wall_left: int
    wall_left_correct: int
    truth_wall_right: int
    pred_wall_right: int
    wall_right_correct: int
    truth_wall_up: int
    pred_wall_up: int
    wall_up_correct: int
    truth_wall_down: int
    pred_wall_down: int
    wall_down_correct: int


def _configure_hf_transfer() -> None:
    enabled = os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return
    if find_spec("hf_transfer") is not None:
        return
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


def _bytes_from_gb(value_gb: float) -> int:
    return int(value_gb * (1024**3))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _load_activation_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".csv":
        with open(path, newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported activation rows file type: {path}")


def _validate_activation_rows_schema(path: Path) -> dict[str, Any]:
    rows = _load_activation_rows(path)
    if not rows:
        raise ValueError(f"No activation rows found in {path}")
    first_row = rows[0]
    missing = [column for column in EXPECTED_ACTIVATION_ROW_COLUMNS if column not in first_row]
    return {
        "row_count": len(rows),
        "columns": list(first_row.keys()),
        "missing_required_columns": missing,
        "is_valid": not missing,
    }


def _fetch_published_probe_file_infos(
    repo_id: str,
    *,
    revision: str | None = None,
) -> list[PublishedProbeFileInfo]:
    api = HfApi()
    info = api.repo_info(repo_id, repo_type="model", revision=revision)
    siblings = {s.rfilename: s for s in list(getattr(info, "siblings", []) or [])}
    file_infos: list[PublishedProbeFileInfo] = []
    for filename in PUBLISHED_DECODER_PROBE_FILENAMES:
        sibling = siblings.get(filename)
        size_bytes = getattr(sibling, "size", None)
        if size_bytes is None:
            try:
                metadata = get_hf_file_metadata(
                    hf_hub_url(repo_id, filename=filename, repo_type="model", revision=revision)
                )
                size_bytes = getattr(metadata, "size", None)
            except Exception:
                size_bytes = None
        file_infos.append(PublishedProbeFileInfo(filename=filename, size_bytes=size_bytes))
    return file_infos


def _check_disk_budget(
    *,
    target_dir: Path,
    estimated_download_bytes: int,
    working_headroom_bytes: int,
    size_cap_bytes: int,
) -> DiskCheckResult:
    target_dir.mkdir(parents=True, exist_ok=True)
    usage = disk_usage(target_dir)
    required = estimated_download_bytes + working_headroom_bytes
    allowed = estimated_download_bytes <= size_cap_bytes and usage.free >= required
    return DiskCheckResult(
        target_dir=str(target_dir),
        free_bytes=usage.free,
        required_bytes=required,
        estimated_download_bytes=estimated_download_bytes,
        headroom_bytes=working_headroom_bytes,
        allowed=allowed,
    )


def _download_published_decoder_probes(
    *,
    repo_id: str,
    cache_dir: Path,
    revision: str | None,
    file_infos: Sequence[PublishedProbeFileInfo],
    size_cap_gb: float,
    working_headroom_gb: float,
) -> tuple[Path, DiskCheckResult]:
    _configure_hf_transfer()
    estimated_download_bytes = sum(info.size_bytes or 0 for info in file_infos)
    disk_check = _check_disk_budget(
        target_dir=cache_dir,
        estimated_download_bytes=estimated_download_bytes,
        working_headroom_bytes=_bytes_from_gb(working_headroom_gb),
        size_cap_bytes=_bytes_from_gb(size_cap_gb),
    )
    if not disk_check.allowed:
        raise RuntimeError(
            "Refusing decoder-probe download: estimated footprint exceeds the configured "
            "size cap or local free space after headroom reservation."
        )

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        allow_patterns=list(PUBLISHED_DECODER_PROBE_FILENAMES) + ["README.md"],
        cache_dir=str(cache_dir),
    )
    return Path(snapshot_path), disk_check


def _guess_artifact_family(
    *,
    query_count: int | None,
    output_class_count: int | None,
    feature_input_dim: int | None,
) -> tuple[str, str]:
    if query_count == 10 and output_class_count == 5 and feature_input_dim == 2880:
        return (
            "sequence_decoder_candidate",
            "Checkpoint exposes 10 learned queries and a 5-way output head over 2880-d features; this looks like a small sequence decoder, not an obvious spatial cognitive-map probe.",
        )
    if feature_input_dim == 8642 and output_class_count == 5:
        return (
            "spatial_cognitive_map_candidate",
            "Checkpoint input/output dimensions are compatible with the released coordinate-conditioned cognitive-map probes.",
        )
    return (
        "unknown_decoder_family",
        "Checkpoint structure does not match the expected cognitive-map signature confidently enough.",
    )


class LogisticRegressionProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int, dropout: float) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, num_classes]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def inspect_published_probe_checkpoint(path: Path) -> ProbeCheckpointInspection:
    if torch is None:
        raise ModuleNotFoundError("torch is required to inspect published probe checkpoints.")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected state_dict-like checkpoint at {path}, got {type(checkpoint)}")

    feature_input_dim = None
    query_count = None
    output_class_count = None

    if "feature_map.0.weight" in checkpoint:
        feature_weight = checkpoint["feature_map.0.weight"]
        feature_input_dim = int(feature_weight.shape[1])
    elif checkpoint.get("model_state_dict") and "network.0.weight" in checkpoint["model_state_dict"]:
        feature_weight = checkpoint["model_state_dict"]["network.0.weight"]
        feature_input_dim = int(feature_weight.shape[1])
    elif checkpoint.get("model_state_dict") and "linear.weight" in checkpoint["model_state_dict"]:
        feature_weight = checkpoint["model_state_dict"]["linear.weight"]
        feature_input_dim = int(feature_weight.shape[1])

    if "path_queries" in checkpoint:
        query_count = int(checkpoint["path_queries"].shape[0])
    if "action_head.weight" in checkpoint:
        output_class_count = int(checkpoint["action_head.weight"].shape[0])
    elif isinstance(checkpoint.get("num_classes"), int):
        output_class_count = int(checkpoint["num_classes"])

    guessed_artifact_family, compatibility_reason = _guess_artifact_family(
        query_count=query_count,
        output_class_count=output_class_count,
        feature_input_dim=feature_input_dim,
    )

    return ProbeCheckpointInspection(
        filename=path.name,
        num_state_dict_keys=len(checkpoint),
        top_level_keys_preview=[str(key) for key in list(checkpoint.keys())[:20]],
        feature_input_dim=feature_input_dim,
        query_count=query_count,
        output_class_count=output_class_count,
        guessed_artifact_family=guessed_artifact_family,
        compatibility_reason=compatibility_reason,
    )


def _hf_file_size(repo_id: str, *, repo_type: str, filename: str, revision: str | None = None) -> int | None:
    try:
        metadata = get_hf_file_metadata(hf_hub_url(repo_id, filename=filename, repo_type=repo_type, revision=revision))
        return getattr(metadata, "size", None)
    except Exception:
        return None


def _download_checked_file(
    *,
    repo_id: str,
    repo_type: str,
    filename: str,
    cache_dir: Path,
) -> tuple[Path, int | None]:
    local = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=filename, cache_dir=str(cache_dir))
    size = _hf_file_size(repo_id, repo_type=repo_type, filename=filename)
    return Path(local), size


def _trajectory_filename(grid_size: int, trajectory_index: int) -> str:
    return f"size{grid_size}/together_ai_openai_gpt-oss-20b_size{grid_size}_comp0.0_{trajectory_index}.json"


def _probe_filename(layer: int, model_type: str, reasoning_stage: str, scope: str) -> str:
    scope_suffix = "general" if scope == "general" else f"size{scope}"
    return f"cognitive_map_probe_layer{layer}_{model_type}_{reasoning_stage}_all_{scope_suffix}.pt"


def _trajectory_indices_for_grid_size(grid_size: int, *, max_trajectories: int) -> list[int]:
    return list(range(max_trajectories))


def _parse_grid_rows(grid_state: list[str]) -> list[list[str]]:
    parsed: list[list[str]] = []
    for raw_row in grid_state[1:]:
        tokens = raw_row.split()
        if len(tokens) < 2:
            continue
        parsed.append(tokens[1:])
    return parsed


def _pad_grid(rows: list[list[str]], pad_to_size: int) -> list[list[str]]:
    if len(rows) > pad_to_size or any(len(row) > pad_to_size for row in rows):
        raise ValueError(f"Grid of shape {len(rows)}x{len(rows[0]) if rows else 0} exceeds pad size {pad_to_size}")
    width = max((len(row) for row in rows), default=0)
    padded = [row + ["+"] * (pad_to_size - len(row)) for row in rows]
    for _ in range(pad_to_size - len(padded)):
        padded.append(["+"] * pad_to_size)
    if width < pad_to_size and rows:
        padded = [row[:pad_to_size] for row in padded]
    return padded


def _agent_position(grid: list[list[str]]) -> tuple[int, int]:
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell == "A":
                return x, y
    raise ValueError("Agent cell 'A' not found in grid")


def _goal_position(grid: list[list[str]]) -> tuple[int, int]:
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell == "G":
                return x, y
    raise ValueError("Goal cell 'G' not found in grid")


def _wall_judgments(grid: list[list[str]], x: int, y: int) -> dict[str, bool]:
    max_y = len(grid) - 1
    max_x = len(grid[0]) - 1
    def cell(xx: int, yy: int) -> str:
        if xx < 0 or yy < 0 or xx > max_x or yy > max_y:
            return "#"
        return grid[yy][xx]
    return {
        "wall_left": cell(x - 1, y) == "#",
        "wall_right": cell(x + 1, y) == "#",
        "wall_up": cell(x, y - 1) == "#",
        "wall_down": cell(x, y + 1) == "#",
    }


def _compute_optimal_actions_from_grid(grid: list[list[str]], goal: tuple[int, int]) -> dict[tuple[int, int], set[int]]:
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    def is_passable(x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        return grid[y][x] != "#"

    neighbors = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]
    distances: dict[tuple[int, int], int] = {goal: 0}
    heap_queue: list[tuple[int, tuple[int, int]]] = [(0, goal)]

    while heap_queue:
        dist, (x, y) = heapq.heappop(heap_queue)
        if dist > distances.get((x, y), 10**9):
            continue
        for dx, dy, _ in neighbors:
            nx, ny = x + dx, y + dy
            if is_passable(nx, ny):
                new_dist = dist + 1
                if new_dist < distances.get((nx, ny), 10**9):
                    distances[(nx, ny)] = new_dist
                    heapq.heappush(heap_queue, (new_dist, (nx, ny)))

    optimal_actions: dict[tuple[int, int], set[int]] = {}
    for y in range(height):
        for x in range(width):
            if not is_passable(x, y):
                continue
            current_dist = distances.get((x, y), 10**9)
            if current_dist == 10**9:
                continue
            action_ids: set[int] = set()
            for dx, dy, action_id in neighbors:
                nx, ny = x + dx, y + dy
                if is_passable(nx, ny) and distances.get((nx, ny), 10**9) == current_dist - 1:
                    action_ids.add(action_id)
            optimal_actions[(x, y)] = action_ids
    optimal_actions[goal] = set()
    return optimal_actions


def _load_probe_checkpoint(path: Path) -> tuple[nn.Module, CognitiveProbeCheckpointInfo, torch.Tensor, torch.Tensor]:
    if torch is None or nn is None:
        raise ModuleNotFoundError("torch is required to load cognitive-map probe checkpoints.")
    checkpoint = torch.load(path, map_location="cpu")
    model_type = str(checkpoint["model_type"])
    input_dim = int(checkpoint["input_dim"])
    num_classes = int(checkpoint["num_classes"])
    hidden_dims = [int(v) for v in checkpoint.get("hidden_dims", [])]
    dropout = float(checkpoint.get("dropout", 0.0))
    idx_to_label = {int(k): int(v) for k, v in checkpoint["idx_to_label"].items()}
    raw_label_to_symbol = {raw: INFERRED_RAW_LABEL_TO_SYMBOL[raw] for raw in idx_to_label.values()}
    raw_label_to_name = {raw: INFERRED_SYMBOL_TO_SEMANTIC_NAME[sym] for raw, sym in raw_label_to_symbol.items()}

    if model_type == "mlp":
        model = MLPProbe(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes, dropout=dropout)
    elif model_type == "lr":
        model = LogisticRegressionProbe(input_dim=input_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported released cognitive probe model_type: {model_type}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler_mean = torch.as_tensor(checkpoint["scaler_mean"], dtype=torch.float32)
    scaler_std = torch.as_tensor(checkpoint["scaler_std"], dtype=torch.float32)
    info = CognitiveProbeCheckpointInfo(
        filename=path.name,
        model_type=model_type,
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
        idx_to_label=idx_to_label,
        raw_label_to_symbol=raw_label_to_symbol,
        raw_label_to_name=raw_label_to_name,
        scaler_mean_shape=tuple(scaler_mean.shape),
        scaler_std_shape=tuple(scaler_std.shape),
    )
    return model, info, scaler_mean, scaler_std


def _load_activation_vector(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected activation tensor at {path}, got {type(tensor)}")
    return tensor.to(dtype=torch.float32).flatten()


def _coordinate_pair(x: int, y: int, coordinate_order: str) -> tuple[float, float]:
    if coordinate_order == "col_row":
        return float(x), float(y)
    if coordinate_order == "row_col":
        return float(y), float(x)
    raise ValueError(f"Unsupported coordinate_order: {coordinate_order}")


def _build_feature(
    activation_vectors: list[torch.Tensor],
    x: int,
    y: int,
    scaler_mean: torch.Tensor,
    scaler_std: torch.Tensor,
    coordinate_order: str,
) -> torch.Tensor:
    if len(activation_vectors) != 3:
        raise ValueError(f"Expected exactly 3 token activations, got {len(activation_vectors)}")
    first_coord, second_coord = _coordinate_pair(x, y, coordinate_order)
    concat = torch.cat(
        activation_vectors + [torch.tensor([first_coord, second_coord], dtype=torch.float32)]
    )
    if concat.shape != scaler_mean.shape:
        raise ValueError(f"Feature shape {tuple(concat.shape)} does not match scaler shape {tuple(scaler_mean.shape)}")
    safe_std = torch.where(scaler_std == 0, torch.ones_like(scaler_std), scaler_std)
    return (concat - scaler_mean) / safe_std


def _predict_grid(
    *,
    model: nn.Module,
    activation_vectors: list[torch.Tensor],
    scaler_mean: torch.Tensor,
    scaler_std: torch.Tensor,
    checkpoint_info: CognitiveProbeCheckpointInfo,
    pad_to_size: int,
    coordinate_order: str,
) -> list[list[str]]:
    rows: list[list[str]] = []
    with torch.no_grad():
        for y in range(pad_to_size):
            row: list[str] = []
            for x in range(pad_to_size):
                feature = _build_feature(
                    activation_vectors,
                    x=x,
                    y=y,
                    scaler_mean=scaler_mean,
                    scaler_std=scaler_std,
                    coordinate_order=coordinate_order,
                )
                logits = model(feature.unsqueeze(0))
                pred_idx = int(logits.argmax(dim=-1).item())
                raw_label = checkpoint_info.idx_to_label[pred_idx]
                row.append(checkpoint_info.raw_label_to_symbol[raw_label])
            rows.append(row)
    return rows


def _grid_accuracy(predicted: list[list[str]], truth: list[list[str]]) -> float:
    total = 0
    correct = 0
    for pred_row, truth_row in zip(predicted, truth):
        for pred_cell, truth_cell in zip(pred_row, truth_row):
            total += 1
            if pred_cell == truth_cell:
                correct += 1
    return correct / total if total else 0.0


def _cell_accuracy_for_symbol(predicted: list[list[str]], truth: list[list[str]], symbol: str) -> float:
    total = 0
    correct = 0
    for pred_row, truth_row in zip(predicted, truth):
        for pred_cell, truth_cell in zip(pred_row, truth_row):
            if truth_cell != symbol:
                continue
            total += 1
            if pred_cell == truth_cell:
                correct += 1
    return correct / total if total else 0.0


def _first_position(grid: list[list[str]], symbol: str) -> tuple[int, int] | None:
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell == symbol:
                return x, y
    return None


def _compatibility_report(
    *,
    trajectory_filename: str,
    trajectory: dict[str, Any],
    step_index: int,
    layer: int,
    probe_input_dim: int,
    pad_to_size: int,
    coordinate_order: str,
    pre_paths: list[Path],
    post_paths: list[Path],
) -> CompatibilityReport:
    step = trajectory["steps"][step_index]
    pre_tokens = [int(path.stem) for path in pre_paths]
    post_tokens = [int(path.stem) for path in post_paths]
    pre_dim = int(_load_activation_vector(pre_paths[0]).shape[0])
    post_dim = int(_load_activation_vector(post_paths[0]).shape[0])
    expected_post = [step["output_n_tokens"] - 16 + i for i in range(3)]
    expected_probe_input_dim = (3 * pre_dim) + 2
    return CompatibilityReport(
        trajectory_filename=trajectory_filename,
        trajectory_stem=Path(trajectory_filename).stem,
        grid_size=int(trajectory["grid_params"]["grid_width"]),
        step_index=step_index,
        model_id=str(trajectory["model_params"]["model_id"]),
        layer=layer,
        prompt_suffix_n_tokens=int(step["prompt_suffix_n_tokens"]),
        output_n_tokens=int(step["output_n_tokens"]),
        expected_pre_token_ids=[0, 1, 2],
        observed_pre_token_ids=pre_tokens,
        expected_post_token_ids=expected_post,
        observed_post_token_ids=post_tokens,
        pre_activation_dim=pre_dim,
        post_activation_dim=post_dim,
        probe_input_dim=probe_input_dim,
        expected_probe_input_dim=expected_probe_input_dim,
        coordinate_order=coordinate_order,
        pre_feature_dim_matches_probe=probe_input_dim == expected_probe_input_dim,
        post_feature_dim_matches_probe=probe_input_dim == ((3 * post_dim) + 2),
        grid_fits_probe_pad=int(trajectory["grid_params"]["grid_width"]) <= pad_to_size,
        stage_contract_matches_release=(pre_tokens == [0, 1, 2]) and (post_tokens == expected_post),
    )


def _render_grid(grid: list[list[str]]) -> str:
    lines = ["  " + " ".join(str(i) for i in range(len(grid[0])))]
    for y, row in enumerate(grid):
        lines.append(f"{y:02d} " + " ".join(row))
    return "\n".join(lines)


def _render_original_grid(grid_state: list[str]) -> str:
    return "\n".join(grid_state)


def _bool_to_yes_no(value: int | bool) -> str:
    return "yes" if bool(value) else "no"


def _evaluate_stage(
    *,
    example_id: str,
    reasoning_stage: str,
    trajectory_filename: str,
    trajectory_stem: str,
    trajectory: dict[str, Any],
    step_index: int,
    layer: int,
    model: nn.Module,
    checkpoint_info: CognitiveProbeCheckpointInfo,
    scaler_mean: torch.Tensor,
    scaler_std: torch.Tensor,
    activation_paths: list[Path],
    pad_to_size: int,
    coordinate_order: str,
) -> tuple[StageEvaluationRow, list[list[str]], list[list[str]]]:
    activation_vectors = [_load_activation_vector(path) for path in activation_paths]

    truth_unpadded = _parse_grid_rows(trajectory["steps"][step_index]["grid_state"])
    truth = _pad_grid(truth_unpadded, pad_to_size=pad_to_size)
    predicted = _predict_grid(
        model=model,
        activation_vectors=activation_vectors,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        checkpoint_info=checkpoint_info,
        pad_to_size=pad_to_size,
        coordinate_order=coordinate_order,
    )

    truth_agent = _agent_position(truth)
    pred_agent = _first_position(predicted, "A")
    truth_goal = _goal_position(truth)
    pred_goal = _first_position(predicted, "G")
    truth_walls = _wall_judgments(truth, *truth_agent)
    pred_walls = _wall_judgments(predicted, *truth_agent)

    row = StageEvaluationRow(
        example_id=example_id,
        reasoning_stage=reasoning_stage,
        trajectory_filename=trajectory_filename,
        trajectory_stem=trajectory_stem,
        step_index=step_index,
        grid_size=int(trajectory["grid_params"]["grid_width"]),
        layer=layer,
        overall_cell_accuracy=_grid_accuracy(predicted, truth),
        wall_cell_accuracy=_cell_accuracy_for_symbol(predicted, truth, "#"),
        agent_location_exact=int(pred_agent == truth_agent),
        goal_location_exact=int(pred_goal == truth_goal),
        truth_agent_x=truth_agent[0],
        truth_agent_y=truth_agent[1],
        pred_agent_x=(None if pred_agent is None else pred_agent[0]),
        pred_agent_y=(None if pred_agent is None else pred_agent[1]),
        truth_goal_x=truth_goal[0],
        truth_goal_y=truth_goal[1],
        pred_goal_x=(None if pred_goal is None else pred_goal[0]),
        pred_goal_y=(None if pred_goal is None else pred_goal[1]),
        truth_wall_left=int(truth_walls["wall_left"]),
        pred_wall_left=int(pred_walls["wall_left"]),
        wall_left_correct=int(pred_walls["wall_left"] == truth_walls["wall_left"]),
        truth_wall_right=int(truth_walls["wall_right"]),
        pred_wall_right=int(pred_walls["wall_right"]),
        wall_right_correct=int(pred_walls["wall_right"] == truth_walls["wall_right"]),
        truth_wall_up=int(truth_walls["wall_up"]),
        pred_wall_up=int(pred_walls["wall_up"]),
        wall_up_correct=int(pred_walls["wall_up"] == truth_walls["wall_up"]),
        truth_wall_down=int(truth_walls["wall_down"]),
        pred_wall_down=int(pred_walls["wall_down"]),
        wall_down_correct=int(pred_walls["wall_down"] == truth_walls["wall_down"]),
    )
    return row, truth, predicted


def _write_rows_csv(path: Path, rows: list[StageEvaluationRow]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _summary_rows(rows: list[StageEvaluationRow]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_stage: dict[str, list[StageEvaluationRow]] = {}
    for row in rows:
        by_stage.setdefault(row.reasoning_stage, []).append(row)
    for stage, stage_rows in by_stage.items():
        n = len(stage_rows)
        out.append(
            {
                "reasoning_stage": stage,
                "n_examples": n,
                "mean_overall_cell_accuracy": sum(r.overall_cell_accuracy for r in stage_rows) / n,
                "mean_wall_cell_accuracy": sum(r.wall_cell_accuracy for r in stage_rows) / n,
                "agent_location_exact_rate": sum(r.agent_location_exact for r in stage_rows) / n,
                "goal_location_exact_rate": sum(r.goal_location_exact for r in stage_rows) / n,
                "wall_left_accuracy": sum(r.wall_left_correct for r in stage_rows) / n,
                "wall_right_accuracy": sum(r.wall_right_correct for r in stage_rows) / n,
                "wall_up_accuracy": sum(r.wall_up_correct for r in stage_rows) / n,
                "wall_down_accuracy": sum(r.wall_down_correct for r in stage_rows) / n,
            }
        )
    return out


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_example_markdown(
    *,
    output_dir: Path,
    compatibility: CompatibilityReport,
    stage_truth_and_predictions: dict[str, tuple[list[list[str]], list[list[str]]]],
) -> None:
    lines = [
        "# Released cognitive-map probe example",
        "",
        f"- trajectory: `{compatibility.trajectory_filename}`",
        f"- step index: `{compatibility.step_index}`",
        f"- layer: `{compatibility.layer}`",
        f"- grid size: `{compatibility.grid_size}`",
        "",
        "## Compatibility checks",
        "",
        f"- probe input dim: `{compatibility.probe_input_dim}`",
        f"- expected feature dim from released activations: `{compatibility.expected_probe_input_dim}`",
        f"- prompt_suffix token ids expected/observed: `{compatibility.expected_pre_token_ids}` / `{compatibility.observed_pre_token_ids}`",
        f"- output token ids expected/observed: `{compatibility.expected_post_token_ids}` / `{compatibility.observed_post_token_ids}`",
        f"- pre feature dim matches: `{compatibility.pre_feature_dim_matches_probe}`",
        f"- post feature dim matches: `{compatibility.post_feature_dim_matches_probe}`",
        f"- stage contract matches release: `{compatibility.stage_contract_matches_release}`",
        "",
    ]
    for stage, (truth, predicted) in stage_truth_and_predictions.items():
        lines.extend(
            [
                f"## {stage}",
                "",
                "### Ground truth",
                "",
                "```text",
                _render_grid(truth),
                "```",
                "",
                "### Predicted",
                "",
                "```text",
                _render_grid(predicted),
                "```",
                "",
            ]
        )
    _write_markdown(output_dir / "cognitive_map_probe_examples.md", "\n".join(lines))


def run_cognitive_map_probe_reasoning_eval(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    cache_dir: str = DEFAULT_DOWNLOAD_CACHE_DIR,
    download_size_cap_gb: float = DEFAULT_DOWNLOAD_SIZE_CAP_GB,
    working_headroom_gb: float = DEFAULT_WORKING_HEADROOM_GB,
    probe_layer: int = DEFAULT_PROBE_LAYER,
    probe_model_type: str = DEFAULT_PROBE_MODEL_TYPE,
    probe_scope: str = DEFAULT_PROBE_SCOPE,
    grid_size: int = DEFAULT_GRID_SIZE,
    trajectory_index: int = DEFAULT_TRAJECTORY_INDEX,
    step_index: int = DEFAULT_STEP_INDEX,
    max_trajectories: int = DEFAULT_MAX_TRAJECTORIES,
    max_steps_per_trajectory: int = DEFAULT_MAX_STEPS_PER_TRAJECTORY,
    pad_to_size: int = DEFAULT_PAD_TO_SIZE,
    coordinate_order: str = DEFAULT_COORDINATE_ORDER,
) -> None:
    """Evaluate the released cognitive-map probes on a released public trajectory step.

    This command is intentionally narrow in v1. It validates that the released
    probe checkpoint, trajectory JSON, activation tensors, layer, token
    positions, and grid padding contract all line up, then runs a real pre/post
    decode on one released step.
    """

    if torch is None or nn is None:
        raise ModuleNotFoundError("torch is required for cognitive-map probe evaluation.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    stage_probe_filenames = {
        stage: _probe_filename(probe_layer, probe_model_type, stage, probe_scope)
        for stage in DEFAULT_REASONING_STAGES
    }

    trajectory_filenames = [
        _trajectory_filename(grid_size=grid_size, trajectory_index=index)
        for index in _trajectory_indices_for_grid_size(grid_size, max_trajectories=max_trajectories)
    ]
    trajectories: list[tuple[str, dict[str, Any]]] = []
    all_files: list[tuple[str, str, str]] = []
    for stage, probe_filename in stage_probe_filenames.items():
        all_files.append((PUBLIC_COGNITIVE_PROBE_REPO_ID, "model", probe_filename))
    for trajectory_filename in trajectory_filenames:
        all_files.append((PUBLIC_TRAJECTORIES_REPO_ID, "dataset", trajectory_filename))
        trajectory_path, _ = _download_checked_file(
            repo_id=PUBLIC_TRAJECTORIES_REPO_ID,
            repo_type="dataset",
            filename=trajectory_filename,
            cache_dir=cache_root,
        )
        trajectory = json.load(open(trajectory_path))
        trajectories.append((trajectory_filename, trajectory))
        if int(trajectory["grid_params"]["grid_width"]) != grid_size:
            raise ValueError("Requested grid_size does not match downloaded trajectory metadata")
        model_id = str(trajectory["model_params"]["model_id"])
        model_subdir = model_id.replace("/", "__")
        traj_stem = Path(trajectory_filename).stem
        for local_step_index, step in enumerate(trajectory["steps"][:max_steps_per_trajectory]):
            activation_root = (
                f"size{grid_size}/{traj_stem}/{model_subdir}/layer_{probe_layer}/step_{local_step_index}"
            )
            for i in range(3):
                all_files.append(
                    (PUBLIC_ACTIVATIONS_REPO_ID, "dataset", f"{activation_root}/prompt_suffix/{i}.pt")
                )
            post_token_ids = [int(step["output_n_tokens"]) - 16 + i for i in range(3)]
            for token_id in post_token_ids:
                all_files.append(
                    (PUBLIC_ACTIVATIONS_REPO_ID, "dataset", f"{activation_root}/output/{token_id}.pt")
                )

    estimated_download_bytes = sum((_hf_file_size(repo_id, repo_type=repo_type, filename=filename) or 0) for repo_id, repo_type, filename in all_files)
    disk_check = _check_disk_budget(
        target_dir=cache_root,
        estimated_download_bytes=estimated_download_bytes,
        working_headroom_bytes=_bytes_from_gb(working_headroom_gb),
        size_cap_bytes=_bytes_from_gb(download_size_cap_gb),
    )
    if not disk_check.allowed:
        raise RuntimeError(
            "Refusing released cognitive-probe evaluation download: estimated footprint exceeds the configured size cap or local free space after headroom reservation."
        )

    stage_probe_paths = {
        stage: _download_checked_file(repo_id=PUBLIC_COGNITIVE_PROBE_REPO_ID, repo_type="model", filename=filename, cache_dir=cache_root)[0]
        for stage, filename in stage_probe_filenames.items()
    }
    loaded_stage_probes = {
        stage: _load_probe_checkpoint(path)
        for stage, path in stage_probe_paths.items()
    }
    pre_probe_info = loaded_stage_probes["pre_reasoning"][1]

    compatibility_reports: list[CompatibilityReport] = []
    rows: list[StageEvaluationRow] = []
    matched_wall_rows: list[dict[str, Any]] = []
    stage_truth_and_predictions: dict[str, tuple[list[list[str]], list[list[str]]]] = {}
    skipped_missing_activation_states: list[dict[str, Any]] = []
    for trajectory_filename, trajectory in trajectories:
        model_id = str(trajectory["model_params"]["model_id"])
        model_subdir = model_id.replace("/", "__")
        traj_stem = Path(trajectory_filename).stem
        for local_step_index, step in enumerate(trajectory["steps"][:max_steps_per_trajectory]):
            activation_root = (
                f"size{grid_size}/{traj_stem}/{model_subdir}/layer_{probe_layer}/step_{local_step_index}"
            )
            try:
                pre_paths = [
                    _download_checked_file(
                        repo_id=PUBLIC_ACTIVATIONS_REPO_ID,
                        repo_type="dataset",
                        filename=f"{activation_root}/prompt_suffix/{i}.pt",
                        cache_dir=cache_root,
                    )[0]
                    for i in range(3)
                ]
                post_token_ids = [int(step["output_n_tokens"]) - 16 + i for i in range(3)]
                post_paths = [
                    _download_checked_file(
                        repo_id=PUBLIC_ACTIVATIONS_REPO_ID,
                        repo_type="dataset",
                        filename=f"{activation_root}/output/{token_id}.pt",
                        cache_dir=cache_root,
                    )[0]
                    for token_id in post_token_ids
                ]
            except EntryNotFoundError:
                skipped_missing_activation_states.append(
                    {
                        "trajectory_filename": trajectory_filename,
                        "step_index": local_step_index,
                    }
                )
                continue
            compatibility = _compatibility_report(
                trajectory_filename=trajectory_filename,
                trajectory=trajectory,
                step_index=local_step_index,
                layer=probe_layer,
                probe_input_dim=pre_probe_info.input_dim,
                pad_to_size=pad_to_size,
                coordinate_order=coordinate_order,
                pre_paths=pre_paths,
                post_paths=post_paths,
            )
            compatibility_reports.append(compatibility)
            truth_unpadded = _parse_grid_rows(step["grid_state"])
            truth_grid_for_action = _pad_grid(truth_unpadded, pad_to_size=pad_to_size)
            truth_agent_pos = _agent_position(truth_grid_for_action)
            truth_goal_pos = _goal_position(truth_grid_for_action)
            optimal_action_ids = _compute_optimal_actions_from_grid(truth_grid_for_action, truth_goal_pos).get(
                truth_agent_pos, set()
            )
            optimal_action_names = [ACTION_ID_TO_NAME[action_id] for action_id in sorted(optimal_action_ids)]
            observed_action = str(step.get("agent_action", ""))
            observed_action_upper = observed_action.upper()
            truth_walls_for_action = _wall_judgments(truth_grid_for_action, *truth_agent_pos)
            action_to_wall_key = {
                "LEFT": "wall_left",
                "RIGHT": "wall_right",
                "UP": "wall_up",
                "DOWN": "wall_down",
            }
            wall_hit = (
                truth_walls_for_action[action_to_wall_key[observed_action_upper]]
                if observed_action_upper in action_to_wall_key
                else False
            )
            for stage in DEFAULT_REASONING_STAGES:
                activation_paths = pre_paths if stage == "pre_reasoning" else post_paths
                model, checkpoint_info, scaler_mean, scaler_std = loaded_stage_probes[stage]
                row, truth, predicted = _evaluate_stage(
                    example_id=f"{traj_stem}_step{local_step_index}",
                    reasoning_stage=stage,
                    trajectory_filename=trajectory_filename,
                    trajectory_stem=traj_stem,
                    trajectory=trajectory,
                    step_index=local_step_index,
                    layer=probe_layer,
                    model=model,
                    checkpoint_info=checkpoint_info,
                    scaler_mean=scaler_mean,
                    scaler_std=scaler_std,
                    activation_paths=activation_paths,
                    pad_to_size=pad_to_size,
                    coordinate_order=coordinate_order,
                )
                rows.append(row)
                original_grid_text = _render_original_grid(step["grid_state"])
                for direction in ("left", "right", "up", "down"):
                    matched_wall_rows.append(
                        {
                            "example_id": row.example_id,
                            "trajectory_id": traj_stem,
                            "step_index": str(local_step_index),
                            "reasoning_split": "pre" if stage == "pre_reasoning" else "post",
                            "question_id": f"wall_{direction}",
                            "ground_truth_label": _bool_to_yes_no(getattr(row, f"truth_wall_{direction}")),
                            "whitebox_prediction": _bool_to_yes_no(getattr(row, f"pred_wall_{direction}")),
                            "grid_text": original_grid_text,
                            "state_description_text": "",
                            "carrying_key": False,
                            "observed_action": observed_action_upper,
                            "optimal_actions_json": json.dumps(optimal_action_names),
                            "is_optimal_action": json.dumps(observed_action_upper in optimal_action_names),
                            "wall_hit": json.dumps(wall_hit),
                            "source_dataset": "released_cognitive_map_probe_slice",
                            "grid_size": str(grid_size),
                            "probe_layer": str(probe_layer),
                        }
                    )
                if not stage_truth_and_predictions:
                    stage_truth_and_predictions[stage] = (truth, predicted)

    _write_rows_csv(out_dir / "cognitive_map_probe_rows.csv", rows)
    summary_rows = _summary_rows(rows)
    _write_summary_csv(out_dir / "cognitive_map_probe_summary.csv", summary_rows)
    if matched_wall_rows:
        _write_json(out_dir / "matched_wall_rows_preview.json", {"rows": matched_wall_rows[:4]})
        with open(out_dir / "matched_wall_rows.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(matched_wall_rows[0].keys()))
            writer.writeheader()
            writer.writerows(matched_wall_rows)
    _write_example_markdown(
        output_dir=out_dir,
        compatibility=compatibility_reports[0],
        stage_truth_and_predictions=stage_truth_and_predictions,
    )

    manifest = {
        "status": "completed",
        "probe_repo_id": PUBLIC_COGNITIVE_PROBE_REPO_ID,
        "activations_repo_id": PUBLIC_ACTIVATIONS_REPO_ID,
        "trajectories_repo_id": PUBLIC_TRAJECTORIES_REPO_ID,
        "results_repo_id": PUBLIC_RESULTS_REPO_ID,
        "disk_check": asdict(disk_check),
        "compatibility_reports": [asdict(report) for report in compatibility_reports],
        "inferred_label_mapping": {
            str(raw): {
                "symbol": symbol,
                "semantic_name": INFERRED_SYMBOL_TO_SEMANTIC_NAME[symbol],
            }
            for raw, symbol in INFERRED_RAW_LABEL_TO_SYMBOL.items()
        },
        "summary_rows": summary_rows,
        "coordinate_order": coordinate_order,
        "n_rows": len(rows),
        "n_matched_wall_rows": len(matched_wall_rows),
        "skipped_missing_activation_states": skipped_missing_activation_states,
    }
    _write_json(out_dir / "compatibility_report.json", [asdict(report) for report in compatibility_reports])
    _write_json(out_dir / "status.json", manifest)


__all__ = [
    "DEFAULT_DOWNLOAD_CACHE_DIR",
    "DEFAULT_OUTPUT_DIR",
    "EXPECTED_ACTIVATION_ROW_COLUMNS",
    "PUBLISHED_DECODER_PROBE_FILENAMES",
    "PUBLISHED_DECODER_PROBE_REPO_ID",
    "PUBLIC_COGNITIVE_PROBE_REPO_ID",
    "PUBLIC_ACTIVATIONS_REPO_ID",
    "PUBLIC_TRAJECTORIES_REPO_ID",
    "_check_disk_budget",
    "_download_published_decoder_probes",
    "_fetch_published_probe_file_infos",
    "_validate_activation_rows_schema",
    "inspect_published_probe_checkpoint",
    "run_cognitive_map_probe_reasoning_eval",
]
