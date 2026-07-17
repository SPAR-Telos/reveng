"""Exploratory validation of released cognitive-map probes during reasoning."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from reveng.experiments.behavioral_probe_smoke_data import grid_text_to_layout
from reveng.experiments.cognitive_map_probe_reasoning_eval import (
    _agent_position,
    _build_feature,
    _cell_accuracy_for_symbol,
    _first_position,
    _goal_position,
    _grid_accuracy,
    _load_probe_checkpoint,
    _pad_grid,
    _wall_judgments,
)
from reveng.experiments.plan_decoder_reasoning_eval import (
    _target_action_sequence_from_trajectory,
    evaluate_plan_decoder_rows,
)

DEFAULT_ACTIVATION_DIR = (
    "data/behavioral_probes/step_reasoning_drift_balanced_v1_corrected_boundary_windows"
)
DEFAULT_CANDIDATES = (
    "data/behavioral_probes/trajectory_instances_recomputed_optimal/"
    "trajectory_selection_candidates.csv"
)
DEFAULT_OUTPUT_DIR = (
    "data/behavioral_probes/step_reasoning_drift_balanced_v1_corrected_boundary_windows/"
    "intermediate_cognitive_probe_eval"
)
DEFAULT_PROBE_ROOT = (
    "data/hf/cache/models--project-telos--cognitive_map_probes/snapshots/"
    "10cf53a0b877ddc48344973f745bd54f4157e1b6"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def _predict_grid_batched(
    *,
    model: torch.nn.Module,
    activation_vectors: list[torch.Tensor],
    scaler_mean: torch.Tensor,
    scaler_std: torch.Tensor,
    checkpoint_info: Any,
    pad_to_size: int,
    coordinate_order: str,
) -> list[list[str]]:
    features = torch.stack(
        [
            _build_feature(
                activation_vectors,
                x=x,
                y=y,
                scaler_mean=scaler_mean,
                scaler_std=scaler_std,
                coordinate_order=coordinate_order,
            )
            for y in range(pad_to_size)
            for x in range(pad_to_size)
        ]
    )
    device = next(model.parameters()).device
    with torch.no_grad():
        indices = model(features.to(device)).argmax(dim=-1).cpu().tolist()
    symbols = [
        checkpoint_info.raw_label_to_symbol[checkpoint_info.idx_to_label[int(index)]]
        for index in indices
    ]
    return [
        symbols[start : start + pad_to_size]
        for start in range(0, len(symbols), pad_to_size)
    ]


def run_intermediate_cognitive_probe_eval(
    *,
    activation_dir: str = DEFAULT_ACTIVATION_DIR,
    candidate_rows_path: str = DEFAULT_CANDIDATES,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    probe_root: str = DEFAULT_PROBE_ROOT,
    pad_to_size: int = 15,
    coordinate_order: str = "row_col",
    device: str = "cpu",
) -> None:
    """Apply released pre/post probes to three-token reasoning-boundary windows.

    Results are exploratory because intermediate reasoning boundaries are
    outside the released probes' training-position distribution.
    """
    activation_root = Path(activation_dir)
    out = Path(output_dir)
    activation_rows = [
        row
        for row in _read_csv(activation_root / "step_activation_rows.csv")
        if int(row["layer"]) == 15
    ]
    candidates = {row["example_id"]: row for row in _read_csv(Path(candidate_rows_path))}
    probe_paths = {
        "pre_reasoning_probe": Path(probe_root)
        / "cognitive_map_probe_layer15_mlp_pre_reasoning_all_general.pt",
        "post_reasoning_probe": Path(probe_root)
        / "cognitive_map_probe_layer15_mlp_post_reasoning_all_general.pt",
    }
    loaded = {name: _load_probe_checkpoint(path) for name, path in probe_paths.items()}
    for model, _info, _scaler_mean, _scaler_std in loaded.values():
        model.to(device)

    detailed: list[dict[str, Any]] = []
    wall_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in activation_rows:
        window_path = row.get("step_boundary_window_activation_path", "")
        if not window_path:
            skipped.append({**row, "reason": "missing boundary-window path"})
            continue
        window = torch.load(window_path, map_location="cpu", weights_only=True).to(torch.float32)
        if tuple(window.shape) != (3, 2880):
            skipped.append({**row, "reason": f"invalid boundary-window shape {tuple(window.shape)}"})
            continue
        meta = candidates.get(row["example_id"])
        if meta is None:
            skipped.append({**row, "reason": "missing candidate truth row"})
            continue
        truth = _pad_grid(grid_text_to_layout(meta["grid_text"]), pad_to_size)
        truth_agent = _agent_position(truth)
        truth_goal = _goal_position(truth)
        truth_walls = _wall_judgments(truth, *truth_agent)
        vectors = [window[index].flatten() for index in range(3)]
        for probe_name, (model, info, scaler_mean, scaler_std) in loaded.items():
            predicted = _predict_grid_batched(
                model=model,
                activation_vectors=vectors,
                scaler_mean=scaler_mean,
                scaler_std=scaler_std,
                checkpoint_info=info,
                pad_to_size=pad_to_size,
                coordinate_order=coordinate_order,
            )
            pred_agent = _first_position(predicted, "A")
            pred_goal = _first_position(predicted, "G")
            pred_walls = _wall_judgments(predicted, *truth_agent)
            base = {
                "example_id": row["example_id"],
                "trajectory_id": row["trajectory_id"],
                "step_index": row["step_index"],
                "failure_category": row["failure_category"],
                "reasoning_step_idx": int(row["reasoning_step_idx"]),
                "reasoning_progress": float(row["reasoning_progress"]),
                "progress_bucket_10": int(row["progress_bucket_10"]),
                "probe_name": probe_name,
                "probe_training_position": probe_name.removesuffix("_probe"),
                "evaluation_position": "intermediate_reasoning_boundary",
                "layer": 15,
                "boundary_window_start_token": int(row["step_boundary_window_start_token"]),
                "boundary_window_end_token": int(row["step_end_token"]),
                "overall_cell_accuracy": _grid_accuracy(predicted, truth),
                "wall_cell_accuracy": _cell_accuracy_for_symbol(predicted, truth, "#"),
                "agent_location_exact": int(pred_agent == truth_agent),
                "goal_location_exact": int(pred_goal == truth_goal),
            }
            detailed.append(base)
            for direction in ("left", "right", "up", "down"):
                key = f"wall_{direction}"
                wall_rows.append(
                    {
                        **{k: base[k] for k in (
                            "example_id", "trajectory_id", "step_index", "failure_category",
                            "reasoning_step_idx", "reasoning_progress", "progress_bucket_10",
                            "probe_name", "probe_training_position",
                        )},
                        "question_id": key,
                        "ground_truth_label": "yes" if truth_walls[key] else "no",
                        "whitebox_prediction": "yes" if pred_walls[key] else "no",
                        "correct": int(pred_walls[key] == truth_walls[key]),
                    }
                )

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in detailed:
        grouped[(row["probe_name"], int(row["progress_bucket_10"]))].append(row)
    summary = [
        {
            "probe_name": probe_name,
            "progress_bucket_10": bucket,
            "n_positions": len(rows),
            "overall_cell_accuracy": _mean(rows, "overall_cell_accuracy"),
            "wall_cell_accuracy": _mean(rows, "wall_cell_accuracy"),
            "agent_location_exact_rate": _mean(rows, "agent_location_exact"),
            "goal_location_exact_rate": _mean(rows, "goal_location_exact"),
            "wall_direction_accuracy": _mean(
                [
                    wall
                    for wall in wall_rows
                    if wall["probe_name"] == probe_name
                    and int(wall["progress_bucket_10"]) == bucket
                ],
                "correct",
            ),
        }
        for (probe_name, bucket), rows in sorted(grouped.items())
    ]
    _write_csv(out / "cognitive_map_intermediate_rows.csv", detailed)
    _write_csv(out / "wall_belief_intermediate_rows.csv", wall_rows)
    _write_csv(out / "summary_by_progress.csv", summary)
    _write_csv(out / "skipped_rows.csv", skipped)
    status = {
        "status": "completed",
        "interpretation": "exploratory_out_of_training_position_distribution",
        "n_activation_positions": len(activation_rows),
        "n_evaluated_probe_positions": len(detailed),
        "n_wall_predictions": len(wall_rows),
        "n_skipped_rows": len(skipped),
        "probe_names": list(probe_paths),
    }
    (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")


def run_intermediate_plan_decoder_eval(
    *,
    activation_dir: str = DEFAULT_ACTIVATION_DIR,
    candidate_rows_path: str = DEFAULT_CANDIDATES,
    output_dir: str = (
        "data/behavioral_probes/step_reasoning_drift_balanced_v1_corrected_boundary_windows/"
        "intermediate_plan_decoder_eval"
    ),
    decoder_root: str = (
        "data/hf/cache/models--project-telos--decoder_probes/snapshots/"
        "d267c77280d2b54eeeba41d99653cd9f58c3135b"
    ),
    device: str = "cpu",
) -> None:
    """Exploratorily apply released pre/post plan decoders at reasoning boundaries."""
    activation_rows = [
        row
        for row in _read_csv(Path(activation_dir) / "step_activation_rows.csv")
        if int(row["layer"]) == 15 and int(row.get("step_boundary_window_n_tokens", 0)) == 3
    ]
    candidates = {row["example_id"]: row for row in _read_csv(Path(candidate_rows_path))}
    trajectory_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for activation in activation_rows:
        meta = candidates[activation["example_id"]]
        source_file = meta["source_file"]
        if source_file not in trajectory_cache:
            trajectory_cache[source_file] = json.loads(Path(source_file).read_text())
        trajectory = trajectory_cache[source_file]
        state_step_index = int(meta["step_index"])
        targets = _target_action_sequence_from_trajectory(trajectory["steps"], state_step_index)
        for split in ("pre", "post"):
            rows.append(
                {
                    "example_id": (
                        f"{activation['example_id']}__reasoning_{int(activation['reasoning_step_idx']):03d}"
                    ),
                    "reasoning_split": split,
                    "activation_path": activation["step_boundary_window_activation_path"],
                    "target_action_sequence_json": json.dumps(targets),
                    "grid_text": meta["grid_text"],
                    "trajectory_id": activation["trajectory_id"],
                    "step_index": meta["step_index"],
                    "observed_action": meta["observed_action"],
                    "source_dataset": "intermediate_reasoning_boundary_exploratory",
                }
            )
    out = Path(output_dir)
    activation_rows_path = out / "intermediate_plan_decoder_activation_rows.csv"
    _write_csv(activation_rows_path, rows)
    root = Path(decoder_root)
    evaluate_plan_decoder_rows(
        activation_rows_path=activation_rows_path,
        pre_checkpoint_path=root / "decoder_probe_layer15_pre_reasoning.pt",
        post_checkpoint_path=root / "decoder_probe_layer15_post_reasoning.pt",
        output_dir=out,
        device=device,
    )
    status_path = out / "plan_decoder_eval_status.json"
    status = json.loads(status_path.read_text())
    status.update(
        {
            "interpretation": "exploratory_out_of_training_position_distribution",
            "n_reasoning_positions": len(activation_rows),
        }
    )
    status_path.write_text(json.dumps(status, indent=2) + "\n")


__all__ = ["run_intermediate_cognitive_probe_eval", "run_intermediate_plan_decoder_eval"]
