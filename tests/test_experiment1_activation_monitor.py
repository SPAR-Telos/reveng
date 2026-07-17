from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from reveng.experiments.experiment1_activation_monitor import (
    _sampled_rows_for_chunks,
    run_experiment1_activation_monitor,
    verify_activation_compatibility,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_chunks(path: Path) -> None:
    _write_csv(
        path,
        [
            {
                "trace_id": "traj_step_000",
                "path": "traj.json",
                "step_id": 0,
                "analysis_id": 0,
                "sentence_start": 0,
                "sentence_end": 0,
                "char_start": 0,
                "char_end": 10,
                "text": "0123456789",
            },
            {
                "trace_id": "traj_step_000",
                "path": "traj.json",
                "step_id": 0,
                "analysis_id": 1,
                "sentence_start": 1,
                "sentence_end": 1,
                "char_start": 10,
                "char_end": 20,
                "text": "abcdefghij",
            },
        ],
    )


def _write_step_activations(path: Path, tensor_dir: Path, *, mismatch: bool = False) -> None:
    tensor_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for step_idx, char_start, char_end, token_start, token_end, vector in [
        (0, "", "", 0, 0, torch.tensor([0.0, 0.0])),
        (1, 1 if mismatch else 0, 10, 1, 2, torch.tensor([1.0, 0.0])),
        (2, 10, 20, 3, 4, torch.tensor([1.0, 1.0])),
    ]:
        last = tensor_dir / f"step_{step_idx}_last.pt"
        mean = tensor_dir / f"step_{step_idx}_mean.pt"
        torch.save(vector, last)
        torch.save(vector, mean)
        rows.append(
            {
                "activation_prompt_mode": "original_trace",
                "activation_schema_version": "step_reasoning_drift_v1_nonoverlap",
                "analysis_text_path": "analysis.txt",
                "example_id": "traj_step_000",
                "failure_category": "none",
                "layer": 15,
                "local_model_name_or_path": "model",
                "progress_bucket_10": step_idx * 5,
                "prompt_text_path": "prompt.txt",
                "reasoning_progress": step_idx / 2,
                "reasoning_step_idx": step_idx,
                "selection_stage": "context",
                "source_dataset": "test",
                "source_trajectory_path": "traj.json",
                "step_end_char_in_analysis": char_end,
                "step_end_char_in_prompt": char_end if char_end != "" else 1,
                "step_end_token": token_end,
                "step_index": 0,
                "step_last_token_activation_path": str(last),
                "step_mean_activation_path": str(mean),
                "step_start_char_in_analysis": char_start,
                "step_start_char_in_prompt": char_start if char_start != "" else 0,
                "step_start_token": token_start,
                "step_text": "[pre]" if step_idx == 0 else "chunk",
                "trajectory_group": "stay_optimal",
                "trajectory_id": "traj",
            }
        )
    _write_csv(path, rows)


def test_verify_activation_compatibility_rejects_pre_post_only(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.csv"
    _write_chunks(chunks)
    activations = tmp_path / "public_activation_rows.csv"
    _write_csv(
        activations,
        [
            {
                "example_id": "e",
                "reasoning_split": "pre",
                "activation_path": "pre.pt",
                "trajectory_id": "traj",
                "step_index": 0,
            }
        ],
    )

    out = tmp_path / "outputs" / "compat_prepost"
    manifest = verify_activation_compatibility(
        activation_rows_path=str(activations),
        chunked_trajectories_path=str(chunks),
        output_dir=str(out),
        required_layers=(15,),
    )

    assert manifest["status"] == "incompatible"
    assert manifest["activation_kind"] == "pre_post_only"
    assert "pre/post" in (out / "activation_compatibility_report.md").read_text()


def test_sampled_rows_for_chunks_maps_stride_tokens_to_overlapping_spans() -> None:
    chunks = [
        {
            "trace_id": "traj_step_000",
            "path": "traj.json",
            "step_id": "0",
            "analysis_id": "0",
            "sentence_start": "0",
            "sentence_end": "0",
            "char_start": "0",
            "char_end": "6",
            "text": "abcdef",
        },
        {
            "trace_id": "traj_step_000",
            "path": "traj.json",
            "step_id": "0",
            "analysis_id": "1",
            "sentence_start": "1",
            "sentence_end": "1",
            "char_start": "6",
            "char_end": "12",
            "text": "ghijkl",
        },
    ]
    offsets = [(0, 3), (3, 6), (6, 9), (9, 12)]
    token_indices = [
        {"relative_idx": 3, "absolute_idx": 100},
        {"relative_idx": 5, "absolute_idx": 102},
    ]

    rows, meta = _sampled_rows_for_chunks(chunks=chunks, offsets=offsets, token_indices=token_indices)

    assert meta["relative_base"] == 3
    assert [row["n_sampled_tokens"] for row in rows] == [1, 1]
    assert rows[0]["sampled_matrix_rows_json"] == "[0]"
    assert rows[1]["sampled_matrix_rows_json"] == "[1]"


def test_verify_activation_compatibility_flags_chunk_span_mismatch(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.csv"
    _write_chunks(chunks)
    activations = tmp_path / "step_activation_rows.csv"
    _write_step_activations(activations, tmp_path / "tensors", mismatch=True)

    out = tmp_path / "outputs" / "compat_mismatch"
    manifest = verify_activation_compatibility(
        activation_rows_path=str(activations),
        chunked_trajectories_path=str(chunks),
        output_dir=str(out),
        required_layers=(15,),
    )

    assert manifest["status"] == "incompatible"
    rows = list(csv.DictReader((out / "activation_compatibility_rows.csv").open()))
    assert any("character span does not match" in row["reason"] for row in rows)


def test_run_experiment1_activation_monitor_writes_outputs_under_outputs(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.csv"
    sentences = tmp_path / "sentences.csv"
    _write_chunks(chunks)
    _write_csv(
        sentences,
        [
            {
                "trace_id": "traj_step_000",
                "path": "traj.json",
                "step_id": 0,
                "chunk_id": 0,
                "kind": "reasoning",
                "char_start": 0,
                "char_end": 10,
                "token_start": "",
                "token_end": "",
                "contains_action_json_mention": False,
                "chunker_version": "doorkey_sentence_v1",
                "text": "0123456789",
            }
        ],
    )
    drift = tmp_path / "drift"
    _write_csv(
        drift / "prefix_action_rows.csv",
        [
            {
                "example_id": "traj_step_000",
                "trajectory_id": "traj",
                "step_index": 0,
                "failure_category": "none",
                "reasoning_step_idx": 0,
                "reasoning_progress": 0.0,
                "action_label": "UP",
                "action_is_optimal": True,
            },
            {
                "example_id": "traj_step_000",
                "trajectory_id": "traj",
                "step_index": 0,
                "failure_category": "none",
                "reasoning_step_idx": 1,
                "reasoning_progress": 0.5,
                "action_label": "LEFT",
                "action_is_optimal": False,
            },
            {
                "example_id": "traj_step_000",
                "trajectory_id": "traj",
                "step_index": 0,
                "failure_category": "none",
                "reasoning_step_idx": 2,
                "reasoning_progress": 1.0,
                "action_label": "LEFT",
                "action_is_optimal": False,
            },
        ],
    )
    _write_csv(
        drift / "trajectory_wrong_turn_summary.csv",
        [
            {
                "example_id": "traj_step_000",
                "trajectory_id": "traj",
                "step_index": 0,
                "failure_category": "none",
                "n_reasoning_steps": 2,
                "final_action": "LEFT",
                "observed_action": "LEFT",
                "optimal_actions_json": json.dumps(["UP"]),
                "trajectory_group": "go_wrong",
                "trajectory_remains_optimal": False,
                "commitment_step": 1,
                "commitment_step_progress": 0.5,
                "wrong_turn_step_majority": 1,
                "wrong_turn_step_majority_progress": 0.5,
                "wrong_turn_step_strict": 1,
                "recovery_step": "",
            }
        ],
    )
    _write_step_activations(drift / "step_activation_rows.csv", tmp_path / "tensors")

    output_root = tmp_path / "outputs" / "experiment1_activation_monitor"
    manifest = run_experiment1_activation_monitor(
        run_name="unit",
        output_root=str(output_root),
        drift_run_dir=str(drift),
        chunked_trajectories_path=str(chunks),
        sentences_path=str(sentences),
        required_layers=(15,),
        check_tensor_shapes=True,
        verbose=False,
    )

    out = output_root / "unit"
    assert manifest["activation_compatibility_status"] == "compatible"
    assert (out / "run_manifest.json").exists()
    assert (out / "event_rows.csv").exists()
    assert (out / "event_aligned_geometry_rows.csv").exists()
    assert str(out).startswith(str(output_root))
    event_rows = list(csv.DictReader((out / "event_rows.csv").open()))
    assert {row["event_type"] for row in event_rows} >= {"action_change", "sustained_optimality_loss"}


def test_run_experiment1_rejects_data_behavioral_probes_output_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not be written"):
        run_experiment1_activation_monitor(
            output_root="data/behavioral_probes/experiment1_activation_monitor",
            drift_run_dir=str(tmp_path / "missing"),
        )
