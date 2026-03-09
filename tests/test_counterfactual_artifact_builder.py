import json
from pathlib import Path

import pytest

from reveng.experiments.counterfactual_artifact_builder import (
    _read_pair_manifest,
    build_counterfactual_patch_artifacts,
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
        goal_a=(1, 2),
        goal_b=(3, 2),
        category="goal_move",
        layer_key=LAYER_KEY,
        patch_action_source="b",
        linear_target="a",
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


def test_build_patched_trace_maps_b_actions_to_a_frame_for_rotate():
    trace_a = _trace(["LEFT"])
    trace_b = _trace(["UP"])

    patched = build_patched_trace(
        trace_a=trace_a,
        trace_b=trace_b,
        goal_a=(1, 2),
        goal_b=(2, 3),
        category="rotate_90",
        layer_key=LAYER_KEY,
        patch_action_source="b",
        linear_target="b",
        synthetic_goal_prob=0.99,
    )

    # rotate_90 mapping A->B is RIGHT->UP, so inverse maps B:UP -> A:RIGHT
    actions = [s["agent_action"] for s in patched["steps"]]
    assert actions == ["RIGHT"]


def test_build_patched_trace_can_keep_a_actions():
    trace_a = _trace(["LEFT", "LEFT", "LEFT"])
    trace_b = _trace(["RIGHT", "DOWN", "UP"])

    patched = build_patched_trace(
        trace_a=trace_a,
        trace_b=trace_b,
        goal_a=(1, 2),
        goal_b=(3, 2),
        category="goal_move",
        layer_key=LAYER_KEY,
        patch_action_source="a",
        linear_target="b",
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


def test_build_counterfactual_patch_artifacts_with_existing_traces(tmp_path: Path):
    pair_dir = tmp_path / "pair_goal_move_000"
    pair_dir.mkdir(parents=True, exist_ok=True)

    grid_a_path = pair_dir / "grid_a.txt"
    grid_b_path = pair_dir / "grid_b.txt"
    grid_a_path.write_text("\n".join(_grid_state(goal=(1, 2))) + "\n")
    grid_b_path.write_text("\n".join(_grid_state(goal=(3, 2))) + "\n")

    manifest_path = tmp_path / "pair_manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "pair_id": "pair_goal_move_000",
                    "category": "goal_move",
                    "grid_a_path": str(grid_a_path),
                    "grid_b_path": str(grid_b_path),
                    "goal_a": [1, 2],
                    "goal_b": [3, 2],
                }
            ],
            indent=2,
        )
    )

    output_dir = tmp_path / "artifacts"
    pair_artifact_dir = output_dir / "pair_goal_move_000"
    pair_artifact_dir.mkdir(parents=True, exist_ok=True)
    (pair_artifact_dir / "A.json").write_text(json.dumps(_trace(["RIGHT"]), indent=2))
    (pair_artifact_dir / "B.json").write_text(json.dumps(_trace(["LEFT"]), indent=2))

    build_counterfactual_patch_artifacts(
        pair_manifest_path=str(manifest_path),
        output_dir=str(output_dir),
        skip_trajectory_generation=True,
    )

    assert (pair_artifact_dir / "patched.json").exists()
    eval_manifest_path = output_dir / "manifest_for_counterfactual_activation_patching.json"
    assert eval_manifest_path.exists()
    rows = json.loads(eval_manifest_path.read_text())
    assert len(rows) == 1
    assert rows[0]["pair_id"] == "pair_goal_move_000"
    assert rows[0]["category"] == "goal_move"
