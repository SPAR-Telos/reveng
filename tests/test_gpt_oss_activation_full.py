from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

import reveng.experiments.gpt_oss_activation_full as full
from reveng.experiments.gpt_oss_activation_pilot import PilotState


def test_completed_record_validation_checks_shard_integrity(tmp_path: Path) -> None:
    spec = PilotState("full_corpus", "trajectory", 0, 4, 2)
    shard = tmp_path / "state.safetensors"
    tensors = {}
    for layer in (8, 15, 23):
        tensors.update(
            {
                f"layer_{layer}.sentence_mean": torch.zeros((2, 4), dtype=torch.bfloat16),
                f"layer_{layer}.sentence_final": torch.zeros((2, 4), dtype=torch.bfloat16),
                f"layer_{layer}.reasoning_mean": torch.zeros(4, dtype=torch.bfloat16),
                f"layer_{layer}.reasoning_final": torch.zeros(4, dtype=torch.bfloat16),
                f"layer_{layer}.pre_window": torch.zeros((3, 4), dtype=torch.bfloat16),
                f"layer_{layer}.post_window": torch.zeros((3, 4), dtype=torch.bfloat16),
                f"layer_{layer}.action_token": torch.zeros(4, dtype=torch.bfloat16),
            }
        )
    save_file(tensors, shard, metadata={"trace_id": spec.trace_id})
    record = {
        "config_fingerprint": "fingerprint",
        "shard_path": str(shard),
        "file_size_bytes": shard.stat().st_size,
        "shard_sha256": full._sha256_file(shard),
    }
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(record))
    assert full._completed_record_is_valid(
        record_path,
        spec=spec,
        layers=(8, 15, 23),
        config_fingerprint="fingerprint",
    )

    record["shard_sha256"] = "bad"
    record_path.write_text(json.dumps(record))
    assert not full._completed_record_is_valid(
        record_path,
        spec=spec,
        layers=(8, 15, 23),
        config_fingerprint="fingerprint",
    )


def test_config_fingerprint_is_order_independent() -> None:
    assert full._config_fingerprint({"a": 1, "b": 2}) == full._config_fingerprint(
        {"b": 2, "a": 1}
    )
