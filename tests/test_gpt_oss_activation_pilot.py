from __future__ import annotations

import torch

from reveng.experiments.gpt_oss_activation_pilot import (
    BoundaryAggregateCollector,
    PilotState,
    PreparedState,
    _build_resource_projection,
    _fit_disk_projection,
    _tensor_logical_bytes,
)


class _ToyBlock(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + 1


class _ToyBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(64, 4)
        self.layers = torch.nn.ModuleList([_ToyBlock(), _ToyBlock(), _ToyBlock()])

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool,
        return_dict: bool,
    ):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return type("ToyOutput", (), {"past_key_values": past_key_values})()


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ToyBase()
        self.config = type("ToyConfig", (), {"hidden_size": 4})()


def _toy_state() -> PreparedState:
    return PreparedState(
        spec=PilotState("toy", "toy", 0, 4, 2),
        source_path=None,  # type: ignore[arg-type]
        input_ids=list(range(12)),
        sentence_rows=[
            {"sentence_id": "0", "char_start": "0", "char_end": "2"},
            {"sentence_id": "1", "char_start": "2", "char_end": "4"},
        ],
        sentence_spans=[(3, 5), (5, 7)],
        reasoning_span=(3, 7),
        pre_positions=[0, 1, 2],
        post_positions=[7, 8, 9],
        action_position=10,
        tokenizer_validation_seconds=0.0,
        token_ids_match=True,
        final_action="RIGHT",
    )


def test_boundary_collector_matches_direct_sentence_aggregation() -> None:
    torch.manual_seed(42)
    model = _ToyModel().eval()
    collector = BoundaryAggregateCollector(model, layers=(0, 2), input_device=torch.device("cpu"))
    tensors, metrics = collector.run(
        _toy_state(),
        forward_chunk_size=4,
        retain_full_for_validation=True,
    )
    assert tensors["layer_0.sentence_mean"].shape == (2, 4)
    assert tensors["layer_2.pre_window"].shape == (3, 4)
    assert tensors["layer_2.action_token"].shape == (4,)
    assert metrics["validation_max_abs_difference"] == 0.0
    assert metrics["reasoning_tokens"] == 4


def test_logical_bytes_and_disk_projection() -> None:
    tensors = {
        "a": torch.zeros((3, 4), dtype=torch.bfloat16),
        "b": torch.zeros(4, dtype=torch.bfloat16),
    }
    assert _tensor_logical_bytes(tensors) == 32
    projection = _fit_disk_projection(
        [
            {"sentences": 10, "filesystem_bytes": 1200},
            {"sentences": 20, "filesystem_bytes": 2200},
            {"sentences": 30, "filesystem_bytes": 3200},
        ]
    )
    assert projection["fitted_bytes_per_sentence"] == 100.0
    assert projection["fitted_bytes_per_environment_state"] == 200.0
    assert projection["projected_filesystem_bytes"] > 0


def test_runtime_projection_uses_all_input_tokens() -> None:
    rows = [
        {
            "length_class": "p95",
            "reasoning_tokens": 100,
            "input_tokens": 200,
            "sentences": 10,
            "forward_seconds": 2.0,
            "aggregate_seconds": 0.1,
            "serialization_seconds": 0.01,
            "filesystem_bytes": 1200,
        },
        {
            "length_class": "maximum",
            "reasoning_tokens": 200,
            "input_tokens": 300,
            "sentences": 20,
            "forward_seconds": 3.0,
            "aggregate_seconds": 0.2,
            "serialization_seconds": 0.02,
            "filesystem_bytes": 2200,
        },
    ]
    projection = _build_resource_projection(
        rows,
        model_load_seconds=1.0,
        projected_logical_bytes=123,
    )
    assert projection["central_input_tokens_per_second"] == 100.0
    assert projection["full_corpus_input_tokens"] > projection["full_corpus_reasoning_tokens"]
    assert projection["central_seconds"] > projection["central_forward_seconds"]


def test_runtime_projection_reports_separate_attention_cost() -> None:
    rows = [
        {
            "length_class": "p95",
            "reasoning_tokens": 100,
            "input_tokens": 200,
            "sentences": 10,
            "forward_seconds": 2.0,
            "aggregate_seconds": 0.1,
            "serialization_seconds": 0.01,
            "filesystem_bytes": 1200,
        },
        {
            "length_class": "maximum",
            "reasoning_tokens": 200,
            "input_tokens": 300,
            "sentences": 20,
            "forward_seconds": 3.0,
            "aggregate_seconds": 0.2,
            "serialization_seconds": 0.02,
            "filesystem_bytes": 2200,
        },
    ]
    projection = _build_resource_projection(
        rows,
        model_load_seconds=1.0,
        projected_logical_bytes=123,
        attention_benchmark={
            "prefix_input_tokens": 100,
            "prefix_forward_seconds": 1.0,
            "attention_query_seconds": 0.1,
            "attention_aggregate_seconds": 0.01,
            "serialization_seconds": 0.001,
            "sentences": 10,
            "logical_tensor_bytes": 1000,
            "file_size_bytes": 1100,
        },
    )
    assert projection["attention_projection_basis"] == "single median state; separate pass"
    assert projection["projected_attention_incremental_seconds"] > 0
    assert projection["projected_attention_filesystem_bytes"] > 0
