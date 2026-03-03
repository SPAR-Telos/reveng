import json
from pathlib import Path

import pytest

from reveng.experiments.counterfactual_artifact_builder import (
    _read_pair_manifest,
    build_patched_trace,
)

LAYER_KEY = "model.layers.15.output"


def _grid_state(agent=(2, 2), goal=(1, 2), size=5):
    rows = [["#" for _ in range(size)] for _ in range(size)]
    for y in range(1, size - 1):
        for x in range(1, size - 1):
            rows[y][x] = "_"
    ax, ay = agent
    gx, gy = goal
    rows[ay][ax] = "A"
    rows[gy][gx] = "G"
    lines = ["  " + " ".join(str(i) for i in range(size))]
    for y, row in enumerate(rows):
        lines.append(f"{y} " + " ".join(row))
    return lines


def _trace(actions):
    steps = []
    toks = [
        {"token": "<|end|>", "token_groups": ["template"]},
        {"token": "<|start|>", "token_groups": ["template"]},
        {"token": "assistant", "token_groups": ["template"]},
    ]
    for a in actions:
        steps.append(
            {
                "grid_state": _grid_state(),
                "agent_action": a,
                "prompt_suffix_tokens": toks,
            }
        )
    return {"steps": steps}


def test_build_patched_trace_uses_b_actions_and_injects_metadata():
    trace_a = _trace(["LEFT", "LEFT", "LEFT"])
    trace_b = _trace(["RIGHT", "DOWN", "UP"])

    patched = build_patched_trace(
        trace_a=trace_a,
        trace_b=trace_b,
        goal_orig=(1, 2),
        goal_new=(3, 2),
        layer_key=LAYER_KEY,
        patch_action_source="b",
        linear_target="orig",
        synthetic_goal_prob=0.99,
    )

    assert "patch_metadata" in patched
    assert patched["patch_metadata"]["hook_tensor"] == LAYER_KEY
    assert patched["patch_metadata"]["pre_reasoning_last_n"] == 3
    assert patched["patch_metadata"]["post_reasoning_last_n"] == 3

    actions = [s["agent_action"] for s in patched["steps"]]
    assert actions == ["RIGHT", "DOWN", "UP"]

    step0 = patched["steps"][0]
    assert "pre_reasoning_prompt_suffix_tokens" in step0
    assert "prompt_suffix_tokens" in step0

    pre_tokens = step0["pre_reasoning_prompt_suffix_tokens"]
    post_tokens = step0["prompt_suffix_tokens"]

    for token in pre_tokens[-3:] + post_tokens[-3:]:
        assert "probes" in token
        probe_keys = list(token["probes"].keys())
        assert any("_mlp_" in k for k in probe_keys)
        assert any("_linear_" in k for k in probe_keys)


def test_build_patched_trace_can_keep_a_actions():
    trace_a = _trace(["LEFT", "LEFT", "LEFT"])
    trace_b = _trace(["RIGHT", "DOWN", "UP"])

    patched = build_patched_trace(
        trace_a=trace_a,
        trace_b=trace_b,
        goal_orig=(1, 2),
        goal_new=(3, 2),
        layer_key=LAYER_KEY,
        patch_action_source="a",
        linear_target="new",
        synthetic_goal_prob=0.95,
    )

    actions = [s["agent_action"] for s in patched["steps"]]
    assert actions == ["LEFT", "LEFT", "LEFT"]


def test_missing_pair_manifest_creates_template(tmp_path: Path):
    manifest_path = tmp_path / "missing" / "pair_manifest.json"

    with pytest.raises(FileNotFoundError) as exc:
        _read_pair_manifest(manifest_path)

    assert manifest_path.exists()
    created = json.loads(manifest_path.read_text())
    assert isinstance(created, list)
    assert "Created template file" in str(exc.value)
