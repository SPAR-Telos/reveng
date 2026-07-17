"""Postprocess local direct-answer readouts and join sentence activations."""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open


PRIMARY_BELIEFS = ("wall_left", "wall_right", "wall_up", "wall_down", "has_key", "door_open")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: Any) -> bool | None:
    if str(value) == "True":
        return True
    if str(value) == "False":
        return False
    return None


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / denominator) if denominator else float("nan")


def _question_family(question_id: str) -> str:
    if question_id.startswith("wall_"):
        return "adjacent_wall"
    if question_id in {"has_key", "door_open"}:
        return "task_state"
    if question_id.startswith(("hit_wall_after_", "has_key_after_", "door_open_after_")):
        return "action_consequence"
    return "other"


def _annotate_commitment(rows: list[dict[str, Any]]) -> None:
    """Add retrospective full-trace commitment fields to sentence-prefix rows."""
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("analysis_unit") == "sentence":
            by_example[str(row["example_id"])].append(row)
        else:
            row["action_matches_final"] = ""
            row["action_committed"] = ""
            row["commitment_onset"] = ""
    for example_rows in by_example.values():
        ordered = sorted(example_rows, key=lambda row: int(row["reasoning_step_idx"]))
        final_action = ordered[-1]["action_label"]
        onset = len(ordered) - 1
        while onset > 0 and ordered[onset - 1]["action_label"] == final_action:
            onset -= 1
        for index, row in enumerate(ordered):
            row["final_full_trace_action"] = final_action
            row["action_matches_final"] = row["action_label"] == final_action
            row["action_committed"] = index >= onset
            row["commitment_onset"] = index == onset


def _events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    environment_mode = bool(rows and rows[0].get("analysis_unit") == "environment_step")
    for row in rows:
        group = str(row["trajectory_id"] if environment_mode else row["example_id"])
        by_example[group].append(row)
    output: list[dict[str, Any]] = []
    for example_id, example_rows in by_example.items():
        ordered = sorted(
            example_rows,
            key=lambda row: int(row["step_index"] if environment_mode else row["reasoning_step_idx"]),
        )
        for index in range(len(ordered) - 1):
            current, following = ordered[index], ordered[index + 1]
            current_opt = _bool(current["action_is_optimal"])
            next_opt = _bool(following["action_is_optimal"])
            event_types = []
            if current["action_label"] != following["action_label"]:
                event_types.append("action_change")
            if current_opt is True and next_opt is False:
                later = [
                    _bool(row["action_is_optimal"])
                    for row in ordered[index + 1 : index + 4]
                ]
                valid = [value for value in later if value is not None]
                event_types.append(
                    "sustained_optimal_to_suboptimal"
                    if valid and sum(value is False for value in valid) >= 2
                    else "transient_optimal_to_suboptimal"
                )
            if current_opt is False and next_opt is True:
                event_types.append("suboptimal_to_optimal")
            for event_type in event_types:
                output.append(
                    {
                        "event_id": f"{example_id}:{following['reasoning_step_idx']}:{event_type}",
                        "sequence_id": example_id,
                        "example_id": following["example_id"],
                        "trajectory_id": current["trajectory_id"],
                        "step_index": current["step_index"],
                        "event_type": event_type,
                        "analysis_unit": following["analysis_unit"],
                        "reasoning_step_idx": following["reasoning_step_idx"],
                        "current_reasoning_step_idx": current["reasoning_step_idx"],
                        "event_reasoning_step_idx": following["reasoning_step_idx"],
                        "reasoning_progress": following["reasoning_progress"],
                        "previous_action": current["action_label"],
                        "current_action": following["action_label"],
                        "previous_action_is_optimal": current["action_is_optimal"],
                        "current_action_is_optimal": following["action_is_optimal"],
                    }
                )
        if not environment_mode:
            commitment = next(
                (row for row in ordered if row.get("commitment_onset") is True),
                None,
            )
            if commitment is not None:
                output.append(
                    {
                        "event_id": f"{example_id}:{commitment['reasoning_step_idx']}:commitment_onset",
                        "sequence_id": example_id,
                        "example_id": commitment["example_id"],
                        "trajectory_id": commitment["trajectory_id"],
                        "step_index": commitment["step_index"],
                        "event_type": "commitment_onset",
                        "analysis_unit": commitment["analysis_unit"],
                        "reasoning_step_idx": commitment["reasoning_step_idx"],
                        "current_reasoning_step_idx": commitment["reasoning_step_idx"],
                        "event_reasoning_step_idx": commitment["reasoning_step_idx"],
                        "reasoning_progress": commitment["reasoning_progress"],
                        "previous_action": "",
                        "current_action": commitment["action_label"],
                        "previous_action_is_optimal": "",
                        "current_action_is_optimal": commitment["action_is_optimal"],
                    }
                )
    return output


def _belief_shifts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_series: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        series_id = (
            str(row["trajectory_id"])
            if row.get("analysis_unit") == "environment_step"
            else str(row["example_id"])
        )
        by_series[(series_id, str(row["question_id"]))].append(row)
    output: list[dict[str, Any]] = []
    for (series_id, question_id), series in by_series.items():
        ordered = sorted(series, key=lambda row: int(row["reasoning_step_idx"]))
        for previous, current in zip(ordered, ordered[1:]):
            if previous["answer_key"] == current["answer_key"]:
                continue
            previous_error = _bool(previous["belief_is_error"])
            current_error = _bool(current["belief_is_error"])
            if previous_error is False and current_error is True:
                shift_type = "belief_error_onset"
            elif previous_error is True and current_error is False:
                shift_type = "belief_error_recovery"
            else:
                shift_type = "belief_answer_change"
            output.append(
                {
                    "shift_id": f"{series_id}:{current['reasoning_step_idx']}:{question_id}",
                    "sequence_id": series_id,
                    "example_id": current["example_id"],
                    "trajectory_id": current["trajectory_id"],
                    "step_index": current["step_index"],
                    "reasoning_step_idx": current["reasoning_step_idx"],
                    "reasoning_progress": current["reasoning_progress"],
                    "question_id": question_id,
                    "question_family": current["question_family"],
                    "shift_type": shift_type,
                    "previous_answer": previous["answer_key"],
                    "current_answer": current["answer_key"],
                    "ground_truth_key": current["ground_truth_key"],
                }
            )
    return output


def _materialize_standard_rows(
    action_rows: list[dict[str, str]], belief_rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    environment_mode = bool(
        action_rows and action_rows[0].get("analysis_unit") == "environment_step"
    )
    beliefs_by_position: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    standard_beliefs: list[dict[str, Any]] = []
    for row in belief_rows:
        position = int(row["position_index"])
        beliefs_by_position[(row["example_id"], position)].append(row)
        standard_beliefs.append(
            {
                **row,
                "reasoning_step_idx": (
                    int(row["step_index"]) if environment_mode else position
                ),
                "question_family": _question_family(row["question_id"]),
                "answer_valid": True,
                "categorical_entropy_bits": row["entropy_bits"],
                "categorical_probs_json": row["probabilities_json"],
            }
        )
    positions: list[dict[str, Any]] = []
    max_step_by_trajectory: dict[str, int] = defaultdict(int)
    if environment_mode:
        for row in action_rows:
            max_step_by_trajectory[row["trajectory_id"]] = max(
                max_step_by_trajectory[row["trajectory_id"]], int(row["step_index"])
            )
    for row in action_rows:
        index = int(row["position_index"])
        beliefs = beliefs_by_position[(row["example_id"], index)]
        primary = [item for item in beliefs if item["question_id"] in PRIMARY_BELIEFS]
        positions.append(
            {
                **row,
                "reasoning_step_idx": int(row["step_index"]) if environment_mode else index,
                "reasoning_progress": (
                    int(row["step_index"]) / max(1, max_step_by_trajectory[row["trajectory_id"]])
                    if environment_mode
                    else row["reasoning_progress"]
                ),
                "n_beliefs": len(beliefs),
                "n_valid_beliefs": len(beliefs),
                "n_primary_belief_errors": sum(item["belief_is_error"] == "True" for item in primary),
                "primary_belief_error_rate": (
                    sum(item["belief_is_error"] == "True" for item in primary) / len(primary)
                    if primary
                    else ""
                ),
                "state_belief_entropy_bits": (
                    mean(float(item["entropy_bits"]) for item in primary) if primary else ""
                ),
                "action_confidence": max(json.loads(row["action_probabilities_json"]).values()),
            }
        )
    return positions, standard_beliefs


def _load_geometry(
    *,
    positions: list[dict[str, Any]],
    activation_index_path: Path,
) -> list[dict[str, Any]]:
    if not positions:
        return []
    if positions[0].get("analysis_unit") == "environment_step":
        return _load_environment_geometry(
            positions=positions,
            activation_index_path=activation_index_path,
        )
    wanted = {row["example_id"] for row in positions}
    index = pd.read_parquet(activation_index_path)
    index = index[(index.record_kind == "sentence") & index.trace_id.isin(wanted)]
    by_example_positions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in positions:
        if int(row["reasoning_step_idx"]) > 0:
            by_example_positions[row["example_id"]].append(row)
    output: list[dict[str, Any]] = []
    for example_id, example_positions in by_example_positions.items():
        example_index = index[index.trace_id == example_id]
        shard_path = Path(example_index.iloc[0].shard_path)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for layer in (8, 15, 23):
                for representation, key in (
                    ("sentence_mean", f"layer_{layer}.sentence_mean"),
                    ("sentence_final", f"layer_{layer}.sentence_final"),
                ):
                    matrix = handle.get_tensor(key).float()
                    ordered = sorted(example_positions, key=lambda row: int(row["reasoning_step_idx"]))
                    previous_vector: torch.Tensor | None = None
                    previous_sum: torch.Tensor | None = None
                    previous_count = 0
                    for position in ordered:
                        sentence_id = int(position["reasoning_step_idx"]) - 1
                        vector = matrix[sentence_id]
                        if previous_vector is not None and previous_sum is not None:
                            update_norm = float(
                                torch.linalg.vector_norm(vector - previous_vector)
                            )
                            adjacent_cosine = _cosine(vector, previous_vector)
                            previous_mean_cosine = _cosine(
                                vector, previous_sum / previous_count
                            )
                        else:
                            update_norm = adjacent_cosine = previous_mean_cosine = ""
                        output.append(
                            {
                                "example_id": example_id,
                                "trajectory_id": position["trajectory_id"],
                                "step_index": position["step_index"],
                                "reasoning_step_idx": position["reasoning_step_idx"],
                                "reasoning_progress": position["reasoning_progress"],
                                "layer": layer,
                                "representation": representation,
                                "update_norm": update_norm,
                                "adjacent_cosine": adjacent_cosine,
                                "previous_mean_cosine": previous_mean_cosine,
                            }
                        )
                        previous_vector = vector
                        previous_sum = (
                            vector.clone()
                            if previous_sum is None
                            else previous_sum + vector
                        )
                        previous_count += 1
    return output


def _load_environment_geometry(
    *, positions: list[dict[str, Any]], activation_index_path: Path
) -> list[dict[str, Any]]:
    index = pd.read_parquet(activation_index_path)
    state_index = index[index.record_kind == "reasoning_final"]
    shard_by_example = {
        row.trace_id: row.shard_path for row in state_index.itertuples()
    }
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in positions:
        by_trajectory[row["trajectory_id"]].append(row)
    output: list[dict[str, Any]] = []
    for trajectory_id, trajectory_rows in by_trajectory.items():
        ordered = sorted(trajectory_rows, key=lambda row: int(row["step_index"]))
        for layer in (8, 15, 23):
            for representation in ("reasoning_mean", "reasoning_final", "action_token"):
                vectors = []
                for position in ordered:
                    with safe_open(
                        Path(shard_by_example[position["example_id"]]),
                        framework="pt",
                        device="cpu",
                    ) as handle:
                        vectors.append(handle.get_tensor(f"layer_{layer}.{representation}").float())
                previous_vector: torch.Tensor | None = None
                previous_sum: torch.Tensor | None = None
                previous_count = 0
                for position, vector in zip(ordered, vectors):
                    if previous_vector is not None and previous_sum is not None:
                        update_norm = float(
                            torch.linalg.vector_norm(vector - previous_vector)
                        )
                        adjacent_cosine = _cosine(vector, previous_vector)
                        previous_mean_cosine = _cosine(
                            vector, previous_sum / previous_count
                        )
                    else:
                        update_norm = adjacent_cosine = previous_mean_cosine = ""
                    output.append(
                        {
                            "example_id": position["example_id"],
                            "trajectory_id": trajectory_id,
                            "step_index": position["step_index"],
                            "reasoning_step_idx": position["reasoning_step_idx"],
                            "reasoning_progress": position["reasoning_progress"],
                            "layer": layer,
                            "representation": representation,
                            "update_norm": update_norm,
                            "adjacent_cosine": adjacent_cosine,
                            "previous_mean_cosine": previous_mean_cosine,
                        }
                    )
                    previous_vector = vector
                    previous_sum = (
                        vector.clone()
                        if previous_sum is None
                        else previous_sum + vector
                    )
                    previous_count += 1
    return output


def _plot_outputs(
    *,
    output_dir: Path,
    positions: list[dict[str, Any]],
    events: list[dict[str, Any]],
    geometry: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 10})
    blue = "#1769AA"
    fig_dir = output_dir / "figs"
    fig_dir.mkdir(exist_ok=True)
    environment_mode = bool(
        positions and positions[0].get("analysis_unit") == "environment_step"
    )
    progress_label = (
        "Fraction of environment steps completed"
        if environment_mode
        else "Fraction of reasoning characters revealed"
    )
    bins = np.linspace(0, 1, 11)
    for field, ylabel, filename in (
        ("action_confidence", "Probability of recommended action", "action_confidence_over_progress.png"),
        ("state_belief_entropy_bits", "Mean wall, key, and door belief entropy (bits)", "state_belief_uncertainty_over_progress.png"),
    ):
        xs, ys = [], []
        for left, right in zip(bins[:-1], bins[1:]):
            values = [
                float(row[field])
                for row in positions
                if row.get(field) not in {None, ""}
                and left <= float(row["reasoning_progress"]) <= right
            ]
            if values:
                xs.append((left + right) / 2)
                ys.append(mean(values))
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.plot(xs, ys, color=blue, marker="o")
        ax.set_xlabel(progress_label)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{ylabel.split(' (')[0]}\n"
            f"{len({row['example_id'] for row in positions})} states, {len(positions):,} prefixes"
        )
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=220)
        plt.close(fig)

    if geometry:
        primary_representation = "reasoning_final" if environment_mode else "sentence_mean"
        selected = [
            row for row in geometry
            if row["layer"] == 15 and row["representation"] == primary_representation
            and row["update_norm"] != ""
        ]
        xs, ys = [], []
        for left, right in zip(bins[:-1], bins[1:]):
            values = [
                float(row["update_norm"])
                for row in selected
                if left <= float(row["reasoning_progress"]) <= right
            ]
            if values:
                xs.append((left + right) / 2)
                ys.append(mean(values))
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.plot(xs, ys, color=blue, marker="o")
        ax.set_xlabel(progress_label)
        ax.set_ylabel("Activation update norm")
        ax.set_title(
            f"Activation Change During Reasoning\n"
            f"{len({row['example_id'] for row in selected})} states, {len(selected):,} activation-backed positions"
        )
        fig.tight_layout()
        fig.savefig(fig_dir / "activation_update_norm_over_progress.png", dpi=220)
        plt.close(fig)


def build_local_immediate_analysis(
    *,
    readout_dir: str | Path,
    experiment2_dir: str | Path,
    activation_index_path: str | Path = "outputs/activation_collection/gpt_oss_20b_boundary_v1/activation_index.parquet",
) -> dict[str, Any]:
    readout = Path(readout_dir)
    exp2 = Path(experiment2_dir)
    exp2.mkdir(parents=True, exist_ok=True)
    action_rows = _read_csv(readout / "prefix_action_rows.csv")
    raw_beliefs = _read_csv(readout / "belief_rows.csv")
    positions, beliefs = _materialize_standard_rows(action_rows, raw_beliefs)
    config = json.loads((readout / "run_config.json").read_text())
    candidate_path = Path(config["candidate_rows_path"])
    candidate_metadata = {
        row["example_id"]: {
            key: row.get(key, "")
            for key in (
                "matched_pair_id",
                "matched_role",
                "trajectory_class",
                "primary_step_failure_mode",
                "selection_stage",
            )
        }
        for row in _read_csv(candidate_path)
    }
    for row in positions:
        row.update(candidate_metadata.get(row["example_id"], {}))
    for row in beliefs:
        row.update(candidate_metadata.get(row["example_id"], {}))
    _annotate_commitment(positions)
    events = _events(positions)
    belief_shifts = _belief_shifts(beliefs)
    geometry = _load_geometry(
        positions=positions,
        activation_index_path=Path(activation_index_path),
    )
    _write_csv(readout / "position_rows.csv", positions)
    _write_csv(readout / "event_rows.csv", events)
    _write_csv(readout / "action_transition_rows.csv", events)
    _write_csv(readout / "belief_shift_rows.csv", belief_shifts)
    _write_csv(readout / "geometry_rows.csv", geometry)
    _write_csv(exp2 / "position_rows.csv", positions)
    _write_csv(exp2 / "belief_rows.csv", beliefs)
    _write_csv(exp2 / "event_rows.csv", events)
    _write_csv(exp2 / "action_transition_rows.csv", events)
    _write_csv(exp2 / "belief_shift_rows.csv", belief_shifts)
    shutil.copy2(readout / "run_config.json", exp2 / "run_config.json")
    _plot_outputs(output_dir=readout, positions=positions, events=events, geometry=geometry)
    _plot_outputs(output_dir=exp2, positions=positions, events=events, geometry=[])
    summary = {
        "status": "completed",
        "positions": len(positions),
        "belief_rows": len(beliefs),
        "events": dict(Counter(row["event_type"] for row in events)),
        "belief_shifts": len(belief_shifts),
        "geometry_rows": len(geometry),
        "activation_source": str(Path(activation_index_path)),
        "entropy_source": "candidate-token logprobs; no repeated sampling",
    }
    (readout / "analysis_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (exp2 / "analysis_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


__all__ = ["build_local_immediate_analysis"]
