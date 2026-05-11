from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from reveng.experiments.plan_decoder_reasoning_eval import (
    PLAN_HORIZON,
    PublishedPlanDecoder,
    _parse_action_sequence,
    _validate_plan_activation_rows_schema,
    evaluate_plan_decoder_rows,
)


def _write_checkpoint(path: Path, winning_class: int) -> None:
    model = PublishedPlanDecoder()
    state_dict = model.state_dict()
    for key, value in state_dict.items():
        state_dict[key] = torch.zeros_like(value)
    state_dict["action_head.bias"][winning_class] = 1.0
    torch.save(state_dict, path)


def _write_activation(path: Path) -> None:
    torch.save(torch.zeros(3, 2880), path)


def test_parse_action_sequence_accepts_names_and_ids():
    assert _parse_action_sequence(list(range(4)) + [0] * 6) == [0, 1, 2, 3] + [0] * 6
    assert _parse_action_sequence(["LEFT"] * PLAN_HORIZON) == [0] * PLAN_HORIZON


def test_validate_plan_activation_rows_schema(tmp_path: Path):
    csv_path = tmp_path / "rows.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "example_id",
                "reasoning_split",
                "activation_path",
                "target_action_sequence_json",
                "grid_text",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "example_id": "ex_1",
                "reasoning_split": "pre",
                "activation_path": "a.pt",
                "target_action_sequence_json": json.dumps([0] * PLAN_HORIZON),
                "grid_text": "grid",
            }
        )
    schema = _validate_plan_activation_rows_schema(csv_path)
    assert schema["is_valid"] is True
    assert schema["row_count"] == 1


def test_evaluate_plan_decoder_rows_writes_summary(tmp_path: Path):
    pre_ckpt = tmp_path / "pre.pt"
    post_ckpt = tmp_path / "post.pt"
    _write_checkpoint(pre_ckpt, winning_class=0)
    _write_checkpoint(post_ckpt, winning_class=1)

    act_pre = tmp_path / "act_pre.pt"
    act_post = tmp_path / "act_post.pt"
    _write_activation(act_pre)
    _write_activation(act_post)

    rows_csv = tmp_path / "rows.csv"
    with open(rows_csv, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "example_id",
                "reasoning_split",
                "activation_path",
                "target_action_sequence_json",
                "grid_text",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "example_id": "shared_example",
                "reasoning_split": "pre",
                "activation_path": str(act_pre),
                "target_action_sequence_json": json.dumps([0] * PLAN_HORIZON),
                "grid_text": "grid",
            }
        )
        writer.writerow(
            {
                "example_id": "shared_example",
                "reasoning_split": "post",
                "activation_path": str(act_post),
                "target_action_sequence_json": json.dumps([0] * PLAN_HORIZON),
                "grid_text": "grid",
            }
        )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    evaluate_plan_decoder_rows(
        activation_rows_path=rows_csv,
        pre_checkpoint_path=pre_ckpt,
        post_checkpoint_path=post_ckpt,
        output_dir=out_dir,
        device="cpu",
    )

    summary_rows = list(csv.DictReader(open(out_dir / "plan_decoder_eval_summary.csv")))
    assert len(summary_rows) == 2
    by_split = {row["reasoning_split"]: row for row in summary_rows}
    assert float(by_split["pre"]["next_action_accuracy"]) == 1.0
    assert float(by_split["post"]["next_action_accuracy"]) == 0.0

    comparison_rows = list(csv.DictReader(open(out_dir / "plan_decoder_eval_pre_post_comparison.csv")))
    assert len(comparison_rows) == 1
    assert float(comparison_rows[0]["delta_next_action_accuracy"]) == -1.0
