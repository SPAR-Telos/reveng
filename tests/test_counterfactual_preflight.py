import json
from pathlib import Path

import pytest

from reveng.experiments.counterfactual_preflight import validate_counterfactual_preflight

LAYER_KEY = "model.layers.15.output"


def _grid_text(agent=(2, 2), goal=(1, 2), size=5) -> str:
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
    return "\n".join(lines) + "\n"


def _trace_with_patch_metadata(valid: bool = True) -> dict:
    patch_metadata = {
        "pre_reasoning_last_n": 3,
        "post_reasoning_last_n": 3,
        "hook_tensor": LAYER_KEY,
    }
    if not valid:
        patch_metadata["pre_reasoning_last_n"] = 2
    return {"steps": [{"grid_state": _grid_text().strip().splitlines(), "agent_action": "RIGHT"}], "patch_metadata": patch_metadata}


def test_preflight_accepts_coordinate_coercion_and_reports_expected_k(tmp_path: Path):
    pair_dir = tmp_path / "pair_000"
    pair_dir.mkdir(parents=True, exist_ok=True)
    grid_a = pair_dir / "grid_a.txt"
    grid_b = pair_dir / "grid_b.txt"
    grid_a.write_text(_grid_text(goal=(1, 2)))
    grid_b.write_text(_grid_text(goal=(3, 2)))

    manifest = tmp_path / "pair_manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "pair_id": "pair_000",
                    "grid_a_path": str(grid_a),
                    "grid_b_path": str(grid_b),
                    "goal_orig": "(1,2)",
                    "goal_new": "3,2",
                }
            ],
            indent=2,
        )
    )

    summary = validate_counterfactual_preflight(
        pair_manifest_path=str(manifest),
        artifacts_output_dir=str(tmp_path / "artifacts"),
        eval_output_dir=str(tmp_path / "eval"),
        skip_trajectory_generation=True,
        require_api_key=False,
    )
    assert summary["status"] == "ok"
    assert summary["recommended_expected_k"] == 1


def test_preflight_requires_api_key_when_generating(tmp_path: Path, monkeypatch):
    pair_dir = tmp_path / "pair_000"
    pair_dir.mkdir(parents=True, exist_ok=True)
    grid_a = pair_dir / "grid_a.txt"
    grid_b = pair_dir / "grid_b.txt"
    grid_a.write_text(_grid_text(goal=(1, 2)))
    grid_b.write_text(_grid_text(goal=(3, 2)))
    manifest = tmp_path / "pair_manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "pair_id": "pair_000",
                    "grid_a_path": str(grid_a),
                    "grid_b_path": str(grid_b),
                    "goal_orig": [1, 2],
                    "goal_new": [3, 2],
                }
            ],
            indent=2,
        )
    )

    monkeypatch.delenv("TOGETHERAI_API_KEY", raising=False)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

    with pytest.raises(ValueError) as exc:
        validate_counterfactual_preflight(
            pair_manifest_path=str(manifest),
            skip_trajectory_generation=False,
            require_api_key=True,
        )

    assert "Missing Together API key" in str(exc.value)


def test_preflight_fails_on_invalid_patch_metadata(tmp_path: Path):
    pair_dir = tmp_path / "pair_000"
    pair_dir.mkdir(parents=True, exist_ok=True)
    grid_a = pair_dir / "grid_a.txt"
    grid_b = pair_dir / "grid_b.txt"
    grid_a.write_text(_grid_text(goal=(1, 2)))
    grid_b.write_text(_grid_text(goal=(3, 2)))
    pair_manifest = tmp_path / "pair_manifest.json"
    pair_manifest.write_text(
        json.dumps(
            [
                {
                    "pair_id": "pair_000",
                    "grid_a_path": str(grid_a),
                    "grid_b_path": str(grid_b),
                    "goal_orig": [1, 2],
                    "goal_new": [3, 2],
                }
            ],
            indent=2,
        )
    )

    artifacts_dir = tmp_path / "artifacts" / "pair_000"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    a_trace = artifacts_dir / "A.json"
    b_trace = artifacts_dir / "B.json"
    patched = artifacts_dir / "patched.json"
    base_trace = {"steps": [{"grid_state": _grid_text().strip().splitlines(), "agent_action": "RIGHT"}]}
    a_trace.write_text(json.dumps(base_trace, indent=2))
    b_trace.write_text(json.dumps(base_trace, indent=2))
    patched.write_text(json.dumps(_trace_with_patch_metadata(valid=False), indent=2))

    eval_manifest = tmp_path / "eval_manifest.json"
    eval_manifest.write_text(
        json.dumps(
            [
                {
                    "pair_id": "pair_000",
                    "grid_a_path": str(grid_a),
                    "grid_b_path": str(grid_b),
                    "goal_orig": [1, 2],
                    "goal_new": [3, 2],
                    "a_trace_path": str(a_trace),
                    "b_trace_path": str(b_trace),
                    "patched_trace_path": str(patched),
                }
            ],
            indent=2,
        )
    )

    with pytest.raises(ValueError) as exc:
        validate_counterfactual_preflight(
            pair_manifest_path=str(pair_manifest),
            eval_manifest_path=str(eval_manifest),
            skip_trajectory_generation=True,
            require_api_key=False,
        )

    assert "pre_reasoning_last_n must be 3" in str(exc.value)


def test_preflight_missing_pair_manifest_does_not_create_template(tmp_path: Path):
    missing_manifest = tmp_path / "missing" / "pair_manifest.json"
    with pytest.raises(FileNotFoundError) as exc:
        validate_counterfactual_preflight(
            pair_manifest_path=str(missing_manifest),
            skip_trajectory_generation=True,
            require_api_key=False,
        )
    assert "Pair manifest not found" in str(exc.value)
    assert missing_manifest.exists() is False
