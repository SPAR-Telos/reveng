from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from reveng.experiments.maze_local_preflight import (
    locate_decoder_layers,
    reasoning_token_indices,
    validate_activation_tensor,
    write_local_preflight_report,
)
from reveng.experiments.maze_smoke_test import (
    DEFAULT_CONFIG,
    apply_api_model_overrides,
    apply_action,
    build_schedule,
    check_configured_api_models,
    condition_summary,
    extract_response,
    generate_grid_rows,
    parse_action,
    planned_call_breakdown,
    run_api_schedule,
    select_schedule_models,
    trajectory_rows,
    validate_config,
    write_analysis,
)


def test_frozen_grid_and_schedule_counts_and_balance():
    grids = generate_grid_rows(DEFAULT_CONFIG)
    assert len(grids) == 18
    assert len({row["grid_id"] for row in grids}) == 18
    assert {row["selection_quantile"] for row in grids} == {0.5, 0.9}
    assert all(row["candidate_pool_size"] == 32 for row in grids)
    for size in DEFAULT_CONFIG["grid_sizes"]:
        for difficulty in DEFAULT_CONFIG["difficulty_levels"]:
            selected = [
                row
                for row in grids
                if row["grid_size"] == size and row["difficulty"] == difficulty
            ]
            assert (
                selected[0]["optimal_path_length"] <= selected[1]["optimal_path_length"]
            )
    schedule = build_schedule(DEFAULT_CONFIG, grids)
    assert len(schedule) == 108
    sampled = [row for row in schedule if row["temperature"] == 0.7]
    greedy = [row for row in schedule if row["temperature"] == 0.0]
    assert len(sampled) == 72
    assert len(greedy) == 36
    assert {row["grid_id"] for row in sampled} == {row["grid_id"] for row in grids}
    for grid_id in {row["grid_id"] for row in grids}:
        assert len([row for row in sampled if row["grid_id"] == grid_id]) == 4


def test_config_rejects_design_drift():
    changed = json.loads(json.dumps(DEFAULT_CONFIG))
    changed["difficulty_levels"] = [0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="exactly"):
        validate_config(changed)


def test_endpoint_override_is_explicit_and_schedule_can_be_filtered():
    config = apply_api_model_overrides(
        DEFAULT_CONFIG, ["Qwen3-32B=account/Qwen/Qwen3-32B-endpoint"]
    )
    assert config["models"][2]["api_model_id"] == "account/Qwen/Qwen3-32B-endpoint"
    grids = generate_grid_rows(config)
    schedule = build_schedule(config, grids)
    selected = select_schedule_models(config, schedule, ["Qwen3-32B"])
    assert len(selected) == 27
    assert {row["api_model_id"] for row in selected} == {
        "account/Qwen/Qwen3-32B-endpoint"
    }


def test_api_model_availability_check_handles_missing_endpoint():
    available = [
        {"api_model_id": "openai/gpt-oss-20b"},
        {"api_model_id": "google/gemma-4-31B-it"},
    ]
    checks = check_configured_api_models(DEFAULT_CONFIG, available)
    assert [row["available"] for row in checks] == [True, True, False]


def test_action_parsing_and_wall_collision():
    assert parse_action('thinking\n{"action": "RIGHT"}') == "RIGHT"
    assert parse_action("not an action") is None
    layout = [list("#####"), list("#AG##"), list("#####")]
    assert apply_action(layout, (1, 1), "LEFT") == ((1, 1), False)
    assert apply_action(layout, (1, 1), "RIGHT") == ((2, 1), True)


def test_together_reasoning_field_and_usage_are_extracted():
    response = {
        "id": "r1",
        "choices": [
            {
                "message": {"content": '{"action":"UP"}', "reasoning": "short trace"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
        "model": "provider/resolved-model",
        "system_fingerprint": "fp_test",
    }
    got = extract_response(response)
    assert got["reasoning_content"] == "short trace"
    assert got["output_tokens"] == 7
    assert got["provider_returned_model_id"] == "provider/resolved-model"
    assert got["system_fingerprint"] == "fp_test"


def test_resumable_api_loop_and_analysis(tmp_path):
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config["models"] = [config["models"][0]]
    layout = [list("#####"), list("#AG_#"), list("#####")]
    spec = {
        "trajectory_id": "one",
        "model": "GPT-OSS-20B",
        "api_model_id": "openai/gpt-oss-20b",
        "reasoning_setting": "low",
        "reasoning_control": True,
        "temperature": 0.7,
        "sampling_condition": "sampled",
        "step_limit": 2,
        "grid_id": "tiny",
        "grid_size": 5,
        "difficulty": 0.0,
        "observed_wall_fraction": 0.0,
        "grid_replicate": 1,
        "layout": layout,
        "grid_text": "",
        "start_x": 1,
        "start_y": 1,
        "goal_x": 2,
        "goal_y": 1,
        "optimal_path_length": 1,
    }

    calls = []

    def query(**kwargs):
        calls.append(kwargs)
        return {
            "id": "r1",
            "choices": [
                {"message": {"content": '{"action":"RIGHT"}'}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }

    raw = tmp_path / "raw_api_calls.jsonl"
    first = run_api_schedule(config, [spec], raw, query_fn=query)
    second = run_api_schedule(config, [spec], raw, query_fn=query)
    assert len(first) == len(second) == 1
    assert len(calls) == 1
    trajectories = trajectory_rows(first, [spec])
    assert trajectories[0]["goal_reached"] is True
    summary = condition_summary(trajectories, first)
    assert summary[0]["goal_success_rate"] == 1.0
    status = write_analysis(tmp_path, config, [spec])
    assert status == {"calls": 1, "trajectories": 1, "scheduled": 1}
    assert (
        "Models suitable for full run: GPT-OSS-20B"
        in (tmp_path / "api_summary.md").read_text()
    )


def test_partial_trajectory_is_not_summarized_as_complete(tmp_path):
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config["models"] = [config["models"][0]]
    layout = [list("#####"), list("#A_G#"), list("#####")]
    spec = {
        "trajectory_id": "partial",
        "model": "GPT-OSS-20B",
        "api_model_id": "openai/gpt-oss-20b",
        "reasoning_setting": "low",
        "reasoning_control": True,
        "temperature": 0.7,
        "sampling_condition": "sampled",
        "step_limit": 3,
        "grid_id": "tiny",
        "grid_size": 5,
        "difficulty": 0.0,
        "observed_wall_fraction": 0.0,
        "grid_replicate": 1,
        "layout": layout,
        "grid_text": "",
        "start_x": 1,
        "start_y": 1,
        "goal_x": 3,
        "goal_y": 1,
        "optimal_path_length": 2,
    }
    call = {
        **{
            key: spec[key]
            for key in (
                "trajectory_id",
                "model",
                "api_model_id",
                "reasoning_setting",
                "temperature",
                "grid_id",
                "grid_size",
                "difficulty",
                "observed_wall_fraction",
                "grid_replicate",
                "optimal_path_length",
            )
        },
        "step_index": 0,
        "selected_action": "RIGHT",
        "valid_action": True,
        "api_success": True,
        "is_optimal_action": True,
        "truncated_call": False,
        "latency_seconds": 1.0,
        "output_tokens": 3,
        "prompt_tokens": 10,
        "total_tokens": 13,
        "estimated_cost_usd": 0.001,
        "goal_reached_after_action": False,
        "trajectory_terminal": False,
    }
    trajectories = trajectory_rows([call], [spec])
    assert trajectories[0]["trajectory_complete"] is False
    assert condition_summary(trajectories, [call]) == []


def test_dry_run_upper_bound_is_explicit():
    schedule = build_schedule(DEFAULT_CONFIG, generate_grid_rows(DEFAULT_CONFIG))
    rows = planned_call_breakdown(DEFAULT_CONFIG, schedule)
    assert {row["model"] for row in rows} == {
        model["name"] for model in DEFAULT_CONFIG["models"]
    }
    assert all(row["maximum_action_calls"] > 0 for row in rows)
    assert all("actual use should be much lower" in row["note"] for row in rows)


@pytest.mark.parametrize(
    ("trace_format", "pieces"),
    [
        (
            "gpt_oss_channels",
            [
                "<|channel|>",
                "analysis",
                "<|message|>",
                " plan",
                " maze",
                "<|end|>",
                "<|start|>",
                "assistant",
                "<|channel|>",
                "final",
                "<|message|>",
                '{"action":"UP"}',
            ],
        ),
        (
            "qwen_think_tags",
            ["<think>", " plan", " maze", "</think>", '{"action":"UP"}'],
        ),
        ("before_action_json", ["I", " reason", ". ", '{"action":"UP"}']),
    ],
)
def test_reasoning_span_detection(trace_format, pieces):
    indices = reasoning_token_indices(pieces, trace_format)
    assert indices
    joined = "".join(pieces[index] for index in indices)
    assert "action" not in joined


def test_decoder_location_and_activation_validation():
    layers = [object() for _ in range(24)]
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    assert locate_decoder_layers(model) is layers
    info = validate_activation_tensor(
        torch.ones(3, 8, dtype=torch.bfloat16), expected_tokens=3
    )
    assert info == {
        "tokens": 3,
        "hidden_size": 8,
        "dtype": "bfloat16",
        "finite": True,
        "bytes": 48,
    }
    with pytest.raises(ValueError, match="Expected 4"):
        validate_activation_tensor(torch.ones(3, 8), expected_tokens=4)


def test_local_report_stays_not_ready_until_every_model_passes(tmp_path):
    record = {
        "model": "GPT-OSS-20B",
        "checkpoint": "openai/gpt-oss-20b",
        "resolved_revision": "abc",
        "layers": [8, 15, 23],
        "reasoning_tokens_captured": 12,
        "peak_vram_bytes": 2**30,
        "activation_bytes": 4096,
        "grid_id": "g",
        "selected_action": "UP",
        "activation_shape_by_layer": {"layer_8": [12, 8]},
        "generation_and_hook_seconds": 1.2,
    }
    write_local_preflight_report(tmp_path, DEFAULT_CONFIG["models"], [record], [])
    report = (tmp_path / "local_preflight.md").read_text()
    assert "Ready for full activation run: NO" in report
    assert "Gemma-4-31B-IT: not run" in report
