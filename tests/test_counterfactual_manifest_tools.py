import json
from pathlib import Path

import pytest

from reveng.experiments.counterfactual_manifest_tools import (
    generate_counterfactual_eval_manifest,
    generate_counterfactual_grid_pairs,
    generate_counterfactual_pair_manifest,
)


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


def test_generate_counterfactual_pair_manifest_from_grid_dirs(tmp_path: Path):
    pair_dir = tmp_path / "cf" / "pair_000"
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "grid_a.txt").write_text(_grid_text(goal=(1, 2)))
    (pair_dir / "grid_b.txt").write_text(_grid_text(goal=(3, 2)))

    out = tmp_path / "cf" / "pair_manifest.json"
    summary = generate_counterfactual_pair_manifest(
        grids_root=str(tmp_path / "cf"),
        output_path=str(out),
        overwrite=True,
    )
    assert summary["status"] == "ok"
    rows = json.loads(out.read_text())
    assert len(rows) == 1
    assert rows[0]["pair_id"] == "pair_000"
    assert rows[0]["goal_orig"] == [1, 2]
    assert rows[0]["goal_new"] == [3, 2]


def test_generate_counterfactual_pair_manifest_rejects_no_goal_move(tmp_path: Path):
    pair_dir = tmp_path / "cf" / "pair_000"
    pair_dir.mkdir(parents=True, exist_ok=True)
    same_goal = _grid_text(goal=(1, 2))
    (pair_dir / "grid_a.txt").write_text(same_goal)
    (pair_dir / "grid_b.txt").write_text(same_goal)

    with pytest.raises(ValueError) as exc:
        generate_counterfactual_pair_manifest(
            grids_root=str(tmp_path / "cf"),
            output_path=str(tmp_path / "cf" / "pair_manifest.json"),
            overwrite=True,
        )
    assert "goal did not move" in str(exc.value)


def test_generate_counterfactual_pair_manifest_searches_nested_pair_dirs(tmp_path: Path):
    pair_dir = tmp_path / "cf" / "grids" / "pair_000"
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "grid_a.txt").write_text(_grid_text(goal=(1, 2)))
    (pair_dir / "grid_b.txt").write_text(_grid_text(goal=(3, 2)))

    out = tmp_path / "cf" / "pair_manifest.json"
    summary = generate_counterfactual_pair_manifest(
        grids_root=str(tmp_path / "cf"),
        output_path=str(out),
        overwrite=True,
    )
    assert summary["status"] == "ok"
    rows = json.loads(out.read_text())
    assert rows[0]["pair_id"] == "pair_000"


def test_generate_counterfactual_pair_manifest_has_actionable_error_for_missing_pairs(tmp_path: Path):
    (tmp_path / "cf").mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError) as exc:
        generate_counterfactual_pair_manifest(
            grids_root=str(tmp_path / "cf"),
            output_path=str(tmp_path / "cf" / "pair_manifest.json"),
            overwrite=True,
        )
    msg = str(exc.value)
    assert "No pair directories found" in msg
    assert "Expected directories like `data/cf/pair_000/`" in msg


def test_generate_counterfactual_eval_manifest_from_artifacts(tmp_path: Path):
    grid_dir = tmp_path / "cf" / "pair_000"
    grid_dir.mkdir(parents=True, exist_ok=True)
    grid_a = grid_dir / "grid_a.txt"
    grid_b = grid_dir / "grid_b.txt"
    grid_a.write_text(_grid_text(goal=(1, 2)))
    grid_b.write_text(_grid_text(goal=(3, 2)))

    pair_manifest = tmp_path / "cf" / "pair_manifest.json"
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

    artifacts = tmp_path / "cf" / "artifacts" / "pair_000"
    artifacts.mkdir(parents=True, exist_ok=True)
    base_trace = {"steps": [{"grid_state": _grid_text(goal=(1, 2)).strip().splitlines(), "agent_action": "RIGHT"}]}
    (artifacts / "A.json").write_text(json.dumps(base_trace, indent=2))
    (artifacts / "B.json").write_text(json.dumps(base_trace, indent=2))
    (artifacts / "patched.json").write_text(
        json.dumps(
            {
                **base_trace,
                "patch_metadata": {
                    "pre_reasoning_last_n": 3,
                    "post_reasoning_last_n": 3,
                    "hook_tensor": "model.layers.15.output",
                },
            },
            indent=2,
        )
    )

    out = tmp_path / "cf" / "artifacts" / "manifest_for_counterfactual_activation_patching.json"
    summary = generate_counterfactual_eval_manifest(
        pair_manifest_path=str(pair_manifest),
        artifacts_dir=str(tmp_path / "cf" / "artifacts"),
        output_path=str(out),
        overwrite=True,
    )
    assert summary["status"] == "ok"
    rows = json.loads(out.read_text())
    assert len(rows) == 1
    assert rows[0]["pair_id"] == "pair_000"
    assert rows[0]["a_trace_path"].endswith("A.json")


def test_generate_counterfactual_eval_manifest_fails_on_missing_artifacts(tmp_path: Path):
    grid_dir = tmp_path / "cf" / "pair_000"
    grid_dir.mkdir(parents=True, exist_ok=True)
    grid_a = grid_dir / "grid_a.txt"
    grid_b = grid_dir / "grid_b.txt"
    grid_a.write_text(_grid_text(goal=(1, 2)))
    grid_b.write_text(_grid_text(goal=(3, 2)))

    pair_manifest = tmp_path / "cf" / "pair_manifest.json"
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

    with pytest.raises(FileNotFoundError) as exc:
        generate_counterfactual_eval_manifest(
            pair_manifest_path=str(pair_manifest),
            artifacts_dir=str(tmp_path / "cf" / "artifacts"),
            overwrite=True,
        )
    assert "Missing artifact traces" in str(exc.value)


def test_generate_counterfactual_grid_pairs_and_pair_manifest(tmp_path: Path):
    root = tmp_path / "cf_auto"
    summary = generate_counterfactual_grid_pairs(
        output_root=str(root),
        num_pairs=2,
        grid_size=7,
        grid_complexity=0.4,
        seed=123,
        overwrite=True,
    )
    assert summary["status"] == "ok"
    assert summary["num_pairs"] == 2
    assert (root / "pair_000" / "grid_a.txt").exists()
    assert (root / "pair_000" / "grid_b.txt").exists()

    manifest_out = root / "pair_manifest.json"
    manifest_summary = generate_counterfactual_pair_manifest(
        grids_root=str(root),
        output_path=str(manifest_out),
        overwrite=True,
    )
    assert manifest_summary["status"] == "ok"
    rows = json.loads(manifest_out.read_text())
    assert len(rows) == 2
