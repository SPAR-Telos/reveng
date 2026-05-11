from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from reveng.experiments.cognitive_map_probe_reasoning_eval import (
    CompatibilityReport,
    _check_disk_budget,
    _pad_grid,
    _parse_grid_rows,
    _validate_activation_rows_schema,
    inspect_published_probe_checkpoint,
)


def test_check_disk_budget_rejects_when_requirement_exceeds_free_space(tmp_path: Path):
    result = _check_disk_budget(
        target_dir=tmp_path,
        estimated_download_bytes=10_000,
        working_headroom_bytes=10**15,
        size_cap_bytes=10**16,
    )
    assert result.allowed is False
    assert result.required_bytes > result.free_bytes


def test_validate_activation_rows_schema_accepts_csv_with_required_columns(tmp_path: Path):
    csv_path = tmp_path / "activation_rows.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["example_id", "reasoning_split", "activation_path", "grid_text", "extra"])
        writer.writeheader()
        writer.writerow(
            {
                "example_id": "ex_001",
                "reasoning_split": "pre",
                "activation_path": "acts/example.pt",
                "grid_text": "0 1\n0 A G",
                "extra": "ok",
            }
        )

    schema = _validate_activation_rows_schema(csv_path)
    assert schema["is_valid"] is True
    assert schema["missing_required_columns"] == []
    assert schema["row_count"] == 1


def test_validate_activation_rows_schema_flags_missing_columns(tmp_path: Path):
    jsonl_path = tmp_path / "activation_rows.jsonl"
    jsonl_path.write_text(json.dumps({"example_id": "ex_001", "grid_text": "grid"}) + "\n")

    schema = _validate_activation_rows_schema(jsonl_path)
    assert schema["is_valid"] is False
    assert set(schema["missing_required_columns"]) == {"reasoning_split", "activation_path"}


def test_inspect_published_probe_checkpoint_marks_sequence_decoder_candidate(tmp_path: Path):
    ckpt_path = tmp_path / "decoder_probe_layer15_pre_reasoning.pt"
    torch.save(
        {
            "path_queries": torch.zeros(10, 1024),
            "query_position_embeddings": torch.zeros(10, 1024),
            "feature_map.0.weight": torch.zeros(1024, 2880),
            "feature_map.0.bias": torch.zeros(1024),
            "action_head.weight": torch.zeros(5, 1024),
            "action_head.bias": torch.zeros(5),
        },
        ckpt_path,
    )

    inspection = inspect_published_probe_checkpoint(ckpt_path)
    assert inspection.guessed_artifact_family == "sequence_decoder_candidate"
    assert inspection.query_count == 10
    assert inspection.output_class_count == 5
    assert inspection.feature_input_dim == 2880


def test_inspect_published_probe_checkpoint_marks_cognitive_map_candidate(tmp_path: Path):
    ckpt_path = tmp_path / "cognitive_map_probe_layer15_mlp_pre_reasoning_all_general.pt"
    torch.save(
        {
            "model_state_dict": {
                "network.0.weight": torch.zeros(1024, 8642),
                "network.0.bias": torch.zeros(1024),
                "network.2.weight": torch.zeros(5, 1024),
                "network.2.bias": torch.zeros(5),
            },
            "model_type": "mlp",
            "input_dim": 8642,
            "num_classes": 5,
            "hidden_dims": [1024],
            "dropout": 0.0,
            "idx_to_label": {0: 0, 1: 1, 2: 2, 3: 3, 4: 7},
            "scaler_mean": torch.zeros(8642),
            "scaler_std": torch.ones(8642),
        },
        ckpt_path,
    )

    inspection = inspect_published_probe_checkpoint(ckpt_path)
    assert inspection.guessed_artifact_family == "spatial_cognitive_map_candidate"
    assert inspection.feature_input_dim == 8642
    assert inspection.output_class_count == 5


def test_parse_and_pad_grid_rows():
    rows = _parse_grid_rows(
        [
            "  0 1 2 ",
            "0 # # # ",
            "1 # A _ ",
            "2 # G # ",
        ]
    )
    assert rows == [["#", "#", "#"], ["#", "A", "_"], ["#", "G", "#"]]
    padded = _pad_grid(rows, pad_to_size=5)
    assert len(padded) == 5
    assert all(len(row) == 5 for row in padded)
    assert padded[0][:3] == ["#", "#", "#"]
    assert padded[-1] == ["+", "+", "+", "+", "+"]
