import json
from pathlib import Path

import pytest

from reveng.experiments.counterfactual_activation_patching import (
    GridPairSpec,
    PairMetrics,
    PairRecord,
    RunArtifacts,
    aggregate_results,
    counterfactual_activation_patching,
    evaluate_pair,
)

LAYER_KEY = "model.layers.15.output"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _grid_layout(agent=(2, 2), goal=(1, 2), size=5, mutate_wall=None) -> list[list[str]]:
    rows = [["#" for _ in range(size)] for _ in range(size)]
    for y in range(1, size - 1):
        for x in range(1, size - 1):
            rows[y][x] = "_"
    ax, ay = agent
    gx, gy = goal
    rows[ay][ax] = "A"
    rows[gy][gx] = "G"
    if mutate_wall is not None:
        wx, wy, value = mutate_wall
        rows[wy][wx] = value
    return rows


def _layout_to_text(layout: list[list[str]]) -> str:
    lines = ["  " + " ".join(str(i) for i in range(len(layout[0])))]
    for y, row in enumerate(layout):
        lines.append(f"{y} " + " ".join(row))
    return "\n".join(lines) + "\n"


def _grid_state_lines(agent=(2, 2), goal=(1, 2), size=5) -> list[str]:
    return _layout_to_text(_grid_layout(agent=agent, goal=goal, size=size)).strip().splitlines()


def _probe_payload(goal_b=(3, 2), goal_a=(1, 2), include_mlp=True, include_linear=True):
    probes = {}
    if include_mlp:
        gx, gy = goal_b
        probes[
            f"cognitive_map_probe_l15_s0_suffix_-3--1_mlp_1024_full_upsample_normalize_r{gy}_c{gx}"
        ] = {LAYER_KEY: {"goal": 0.95, "empty": 0.05}}
        probes[
            "cognitive_map_probe_l15_s0_suffix_-3--1_mlp_1024_full_upsample_normalize_r2_c1"
        ] = {LAYER_KEY: {"goal": 0.10, "empty": 0.90}}

    if include_linear:
        gx, gy = goal_a
        probes[
            f"cognitive_map_probe_l15_s0_suffix_-3--1_linear_full_upsample_normalize_r{gy}_c{gx}"
        ] = {LAYER_KEY: {"goal": 0.91, "empty": 0.09}}

    return probes


def _tokens_with_probes(probes: dict) -> list[dict]:
    return [
        {"token": "<|end|>", "probes": probes},
        {"token": "<|start|>", "probes": probes},
        {"token": "assistant", "probes": probes},
    ]


def _build_trace(steps: list[dict], include_patch_metadata=False) -> dict:
    trace = {"steps": steps}
    if include_patch_metadata:
        trace["patch_metadata"] = {
            "pre_reasoning_last_n": 3,
            "post_reasoning_last_n": 3,
            "hook_tensor": LAYER_KEY,
        }
    return trace


def _build_step(action: str, include_probes=True, include_mlp=True, include_linear=True):
    probes = _probe_payload(include_mlp=include_mlp, include_linear=include_linear)
    step = {
        "grid_state": _grid_state_lines(),
        "agent_action": action,
    }
    if include_probes:
        step["pre_reasoning_prompt_suffix_tokens"] = _tokens_with_probes(probes)
        step["prompt_suffix_tokens"] = _tokens_with_probes(probes)
    return step


def _create_pair_record(
    base: Path,
    pair_id: str,
    patched_steps: list[dict],
    category: str = "goal_move",
    goal_a=(1, 2),
    goal_b=(3, 2),
    mutate_grid_b=None,
):
    pair_dir = base / pair_id
    grids_dir = pair_dir / "grids"
    artifacts_dir = pair_dir / "artifacts"
    grids_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    grid_a_path = grids_dir / "grid_a.txt"
    grid_b_path = grids_dir / "grid_b.txt"

    grid_a_layout = _grid_layout(goal=goal_a)

    if category == "goal_move":
        grid_b_layout = _grid_layout(goal=goal_b, mutate_wall=mutate_grid_b)
    elif category == "start_goal_swap":
        grid_b_layout = _grid_layout(agent=goal_a, goal=(2, 2), mutate_wall=mutate_grid_b)
    else:
        raise ValueError("test helper currently supports goal_move/start_goal_swap only")

    grid_a_path.write_text(_layout_to_text(grid_a_layout))
    grid_b_path.write_text(_layout_to_text(grid_b_layout))

    a_trace_path = artifacts_dir / "A.json"
    b_trace_path = artifacts_dir / "B.json"
    patched_trace_path = artifacts_dir / "patched.json"

    _write_json(a_trace_path, _build_trace([_build_step("RIGHT", include_probes=False)]))
    _write_json(b_trace_path, _build_trace([_build_step("RIGHT", include_probes=False)]))
    _write_json(patched_trace_path, _build_trace(patched_steps, include_patch_metadata=True))

    return PairRecord(
        spec=GridPairSpec(
            pair_id=pair_id,
            category=category,
            grid_a_path=grid_a_path,
            grid_b_path=grid_b_path,
            goal_a=goal_a,
            goal_b=goal_b,
        ),
        artifacts=RunArtifacts(
            a_trace_path=a_trace_path,
            b_trace_path=b_trace_path,
            patched_trace_path=patched_trace_path,
        ),
    )


def test_action_metric_threshold_boundaries(tmp_path: Path):
    # A_target=0.70, A_base=0.30 => Action=True
    steps = [_build_step("RIGHT") for _ in range(7)] + [_build_step("LEFT") for _ in range(3)]
    record = _create_pair_record(tmp_path, "pair_threshold", steps)

    metric = evaluate_pair(record)

    assert metric.valid_pair is True
    assert metric.evaluated_steps == 10
    assert metric.a_target == 0.7
    assert metric.a_base == 0.3
    assert metric.action_label is True


def test_disruptive_boundary_strict_less_than(tmp_path: Path):
    # A_target=0.35, A_base=0.35 => disruptive should be False (< 0.35 is strict)
    steps = (
        [_build_step("RIGHT") for _ in range(7)]
        + [_build_step("LEFT") for _ in range(7)]
        + [_build_step("DOWN") for _ in range(6)]
    )
    record = _create_pair_record(tmp_path, "pair_disruptive_boundary", steps)

    metric = evaluate_pair(record)

    assert metric.valid_pair is True
    assert metric.evaluated_steps == 20
    assert metric.a_target == 0.35
    assert metric.a_base == 0.35
    assert metric.disruptive is False


def test_belief_labels_are_reported_separately(tmp_path: Path):
    # MLP predicts target goal, linear predicts base goal
    steps = [_build_step("RIGHT", include_mlp=True, include_linear=True) for _ in range(10)]
    record = _create_pair_record(tmp_path, "pair_belief_split", steps)

    metric = evaluate_pair(record)

    assert metric.belief_mlp_match_target is True
    assert metric.belief_linear_match_target is False
    assert metric.belief_available_mlp is True
    assert metric.belief_available_linear is True
    assert metric.outcome_cell_mlp is not None
    assert metric.outcome_cell_linear is not None


def test_belief_unavailable_is_probe_specific(tmp_path: Path):
    steps = [_build_step("RIGHT", include_mlp=False, include_linear=True) for _ in range(10)]
    record = _create_pair_record(tmp_path, "pair_belief_missing", steps)

    metric = evaluate_pair(record)

    assert metric.valid_pair is True
    assert metric.belief_mlp_match_target is None
    assert metric.belief_available_mlp is False
    assert metric.belief_available_linear is True
    assert metric.outcome_cell_mlp is None
    assert metric.outcome_cell_linear is not None


def test_aggregate_reports_separate_probe_tables_and_disagreement():
    metrics = [
        PairMetrics(
            pair_id="tt",
            category="goal_move",
            evaluated_steps=10,
            a_target=0.9,
            a_base=0.1,
            action_label=True,
            disruptive=False,
            belief_mlp_match_target=True,
            belief_linear_match_target=True,
            belief_available_mlp=True,
            belief_available_linear=True,
            outcome_cell_mlp=(True, True),
            outcome_cell_linear=(True, True),
            valid_pair=True,
            invalid_reason=None,
        ),
        PairMetrics(
            pair_id="mixed",
            category="goal_move",
            evaluated_steps=10,
            a_target=0.2,
            a_base=0.8,
            action_label=False,
            disruptive=False,
            belief_mlp_match_target=True,
            belief_linear_match_target=False,
            belief_available_mlp=True,
            belief_available_linear=True,
            outcome_cell_mlp=(True, False),
            outcome_cell_linear=(False, False),
            valid_pair=True,
            invalid_reason=None,
        ),
        PairMetrics(
            pair_id="invalid",
            category="goal_move",
            evaluated_steps=0,
            a_target=0.0,
            a_base=0.0,
            action_label=None,
            disruptive=None,
            belief_mlp_match_target=None,
            belief_linear_match_target=None,
            belief_available_mlp=False,
            belief_available_linear=False,
            outcome_cell_mlp=None,
            outcome_cell_linear=None,
            valid_pair=False,
            invalid_reason="missing",
        ),
    ]

    agg = aggregate_results(metrics, stopped_early=False, early_stop_reason=None)

    assert agg.mlp_tt_count == 1
    assert agg.mlp_tf_count == 1
    assert agg.linear_tt_count == 1
    assert agg.linear_ff_count == 1
    assert agg.probe_both_available == 2
    assert agg.probe_disagree_count == 1
    assert agg.table_rows_mlp == 2
    assert agg.table_rows_linear == 2


def test_integration_outputs_written(tmp_path: Path):
    records = [
        _create_pair_record(tmp_path, "pair_tt", [_build_step("RIGHT") for _ in range(10)]),
        _create_pair_record(tmp_path, "pair_tf", [_build_step("LEFT") for _ in range(10)]),
        _create_pair_record(tmp_path, "pair_disruptive", [_build_step("DOWN") for _ in range(10)]),
    ]

    manifest_path = tmp_path / "manifest.json"
    manifest_rows = []
    for record in records:
        manifest_rows.append(
            {
                "pair_id": record.spec.pair_id,
                "category": record.spec.category,
                "grid_a_path": str(record.spec.grid_a_path),
                "grid_b_path": str(record.spec.grid_b_path),
                "goal_a": list(record.spec.goal_a),
                "goal_b": list(record.spec.goal_b),
                "a_trace_path": str(record.artifacts.a_trace_path),
                "b_trace_path": str(record.artifacts.b_trace_path),
                "patched_trace_path": str(record.artifacts.patched_trace_path),
            }
        )
    manifest_path.write_text(json.dumps(manifest_rows, indent=2))

    output_dir = tmp_path / "results"
    counterfactual_activation_patching(
        manifest_path=str(manifest_path),
        output_dir=str(output_dir),
        expected_k=3,
    )

    assert (output_dir / "per_pair_results.jsonl").exists()
    assert (output_dir / "per_pair_results.csv").exists()
    assert (output_dir / "aggregate_summary.json").exists()
    assert (output_dir / "report.md").exists()

    summary = json.loads((output_dir / "aggregate_summary.json").read_text())
    assert summary["stopped_early"] is False
    assert summary["total_pairs_processed"] == 3
    assert "mlp_tt_count" in summary
    assert "linear_tt_count" in summary


def test_failure_missing_trace_path(tmp_path: Path):
    record = _create_pair_record(tmp_path, "pair_missing_trace", [_build_step("RIGHT") for _ in range(5)])
    record.artifacts.patched_trace_path.unlink()

    metric = evaluate_pair(record)

    assert metric.valid_pair is False
    assert "trace loading failed" in (metric.invalid_reason or "")


def test_failure_category_constraints(tmp_path: Path):
    record = _create_pair_record(
        tmp_path,
        "pair_bad_topology",
        [_build_step("RIGHT") for _ in range(5)],
        mutate_grid_b=(2, 1, "#"),
    )

    metric = evaluate_pair(record)

    assert metric.valid_pair is False
    assert "non-goal topology changed" in (metric.invalid_reason or "")


def test_failure_malformed_grid_text(tmp_path: Path):
    record = _create_pair_record(tmp_path, "pair_bad_grid", [_build_step("RIGHT") for _ in range(5)])
    record.spec.grid_a_path.write_text("this is not a valid grid")

    metric = evaluate_pair(record)

    assert metric.valid_pair is False
    assert "failed to parse grid files" in (metric.invalid_reason or "")


def test_early_stop_after_first_three_non_disruptive_action_false(tmp_path: Path):
    records = [
        _create_pair_record(tmp_path, f"pair_{i}", [_build_step("LEFT") for _ in range(10)])
        for i in range(4)
    ]

    manifest_path = tmp_path / "manifest_early_stop.json"
    manifest_rows = []
    for record in records:
        manifest_rows.append(
            {
                "pair_id": record.spec.pair_id,
                "category": record.spec.category,
                "grid_a_path": str(record.spec.grid_a_path),
                "grid_b_path": str(record.spec.grid_b_path),
                "goal_a": list(record.spec.goal_a),
                "goal_b": list(record.spec.goal_b),
                "a_trace_path": str(record.artifacts.a_trace_path),
                "b_trace_path": str(record.artifacts.b_trace_path),
                "patched_trace_path": str(record.artifacts.patched_trace_path),
            }
        )
    manifest_path.write_text(json.dumps(manifest_rows, indent=2))

    output_dir = tmp_path / "results_early_stop"
    counterfactual_activation_patching(
        manifest_path=str(manifest_path),
        output_dir=str(output_dir),
        expected_k=4,
    )

    summary = json.loads((output_dir / "aggregate_summary.json").read_text())
    assert summary["stopped_early"] is True
    assert "first 3 evaluated pairs" in (summary["early_stop_reason"] or "")


def test_manifest_legacy_goal_keys_still_supported(tmp_path: Path):
    record = _create_pair_record(tmp_path, "pair_legacy", [_build_step("RIGHT") for _ in range(3)])

    manifest_path = tmp_path / "manifest_legacy.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "pair_id": record.spec.pair_id,
                    "category": record.spec.category,
                    "grid_a_path": str(record.spec.grid_a_path),
                    "grid_b_path": str(record.spec.grid_b_path),
                    "goal_orig": list(record.spec.goal_a),
                    "goal_new": list(record.spec.goal_b),
                    "a_trace_path": str(record.artifacts.a_trace_path),
                    "b_trace_path": str(record.artifacts.b_trace_path),
                    "patched_trace_path": str(record.artifacts.patched_trace_path),
                }
            ],
            indent=2,
        )
    )

    output_dir = tmp_path / "results_legacy"
    counterfactual_activation_patching(
        manifest_path=str(manifest_path),
        output_dir=str(output_dir),
        expected_k=1,
    )

    assert (output_dir / "aggregate_summary.json").exists()
