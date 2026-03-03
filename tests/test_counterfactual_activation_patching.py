import json
from pathlib import Path

from reveng.experiments.counterfactual_activation_patching import (
    PairMetrics,
    aggregate_results,
    counterfactual_activation_patching,
    evaluate_pair,
    PairRecord,
    GridPairSpec,
    RunArtifacts,
)

LAYER_KEY = "model.layers.15.output"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _grid_text(agent=(2, 2), goal=(1, 2), size=5, mutate_wall=None) -> str:
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

    lines = ["  " + " ".join(str(i) for i in range(size))]
    for y, row in enumerate(rows):
        lines.append(f"{y} " + " ".join(row))
    return "\n".join(lines) + "\n"


def _grid_state_lines(agent=(2, 2), goal=(1, 2), size=5) -> list[str]:
    return _grid_text(agent=agent, goal=goal, size=size).strip().splitlines()


def _probe_payload(goal_new=(3, 2), goal_orig=(1, 2), include_mlp=True, include_linear=True):
    probes = {}
    if include_mlp:
        gx, gy = goal_new
        probes[
            f"cognitive_map_probe_l15_s0_suffix_-3--1_mlp_1024_full_upsample_normalize_r{gy}_c{gx}"
        ] = {LAYER_KEY: {"goal": 0.95, "empty": 0.05}}
        probes[
            "cognitive_map_probe_l15_s0_suffix_-3--1_mlp_1024_full_upsample_normalize_r2_c1"
        ] = {LAYER_KEY: {"goal": 0.10, "empty": 0.90}}

    if include_linear:
        gx, gy = goal_orig
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


def _create_pair_record(base: Path, pair_id: str, patched_steps: list[dict], mutate_grid_b=None):
    pair_dir = base / pair_id
    grids_dir = pair_dir / "grids"
    artifacts_dir = pair_dir / "artifacts"
    grids_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    grid_a_path = grids_dir / "grid_a.txt"
    grid_b_path = grids_dir / "grid_b.txt"
    goal_orig = (1, 2)
    goal_new = (3, 2)

    grid_a_path.write_text(_grid_text(goal=goal_orig))
    if mutate_grid_b is None:
        grid_b_path.write_text(_grid_text(goal=goal_new))
    else:
        grid_b_path.write_text(_grid_text(goal=goal_new, mutate_wall=mutate_grid_b))

    a_trace_path = artifacts_dir / "A.json"
    b_trace_path = artifacts_dir / "B.json"
    patched_trace_path = artifacts_dir / "patched.json"

    _write_json(a_trace_path, _build_trace([_build_step("RIGHT", include_probes=False)]))
    _write_json(b_trace_path, _build_trace([_build_step("RIGHT", include_probes=False)]))
    _write_json(patched_trace_path, _build_trace(patched_steps, include_patch_metadata=True))

    return PairRecord(
        spec=GridPairSpec(
            pair_id=pair_id,
            grid_a_path=grid_a_path,
            grid_b_path=grid_b_path,
            goal_orig=goal_orig,
            goal_new=goal_new,
        ),
        artifacts=RunArtifacts(
            a_trace_path=a_trace_path,
            b_trace_path=b_trace_path,
            patched_trace_path=patched_trace_path,
        ),
    )


def test_action_metric_threshold_boundaries(tmp_path: Path):
    # A_new=0.70, A_orig=0.30 => Action=True
    steps = [_build_step("RIGHT") for _ in range(7)] + [_build_step("LEFT") for _ in range(3)]
    record = _create_pair_record(tmp_path, "pair_threshold", steps)

    metric = evaluate_pair(record)

    assert metric.valid_pair is True
    assert metric.evaluated_steps == 10
    assert metric.a_new == 0.7
    assert metric.a_orig == 0.3
    assert metric.action_label is True


def test_disruptive_boundary_strict_less_than(tmp_path: Path):
    # A_new=0.35, A_orig=0.35 => disruptive should be False (< 0.35 is strict)
    steps = (
        [_build_step("RIGHT") for _ in range(7)]
        + [_build_step("LEFT") for _ in range(7)]
        + [_build_step("DOWN") for _ in range(6)]
    )
    record = _create_pair_record(tmp_path, "pair_disruptive_boundary", steps)

    metric = evaluate_pair(record)

    assert metric.valid_pair is True
    assert metric.evaluated_steps == 20
    assert metric.a_new == 0.35
    assert metric.a_orig == 0.35
    assert metric.disruptive is False


def test_belief_label_uses_mlp_primary(tmp_path: Path):
    # MLP predicts new goal, linear predicts original goal -> Belief=True (MLP primary)
    steps = [_build_step("RIGHT", include_mlp=True, include_linear=True) for _ in range(10)]
    record = _create_pair_record(tmp_path, "pair_belief_primary", steps)

    metric = evaluate_pair(record)

    assert metric.belief_mlp_new_match is True
    assert metric.belief_linear_new_match is False
    assert metric.belief_label is True


def test_belief_unavailable_when_mlp_missing(tmp_path: Path):
    steps = [_build_step("RIGHT", include_mlp=False, include_linear=True) for _ in range(10)]
    record = _create_pair_record(tmp_path, "pair_belief_missing", steps)

    metric = evaluate_pair(record)

    assert metric.valid_pair is True
    assert metric.belief_mlp_new_match is None
    assert metric.belief_available is False
    assert metric.outcome_cell is None


def test_aggregate_2x2_and_ratios():
    metrics = [
        PairMetrics(
            pair_id="tt",
            evaluated_steps=10,
            a_new=0.9,
            a_orig=0.1,
            belief_mlp_new_match=True,
            belief_linear_new_match=True,
            action_label=True,
            belief_label=True,
            disruptive=False,
            outcome_cell=(True, True),
            valid_pair=True,
            belief_available=True,
            invalid_reason=None,
        ),
        PairMetrics(
            pair_id="tf",
            evaluated_steps=10,
            a_new=0.2,
            a_orig=0.8,
            belief_mlp_new_match=True,
            belief_linear_new_match=False,
            action_label=False,
            belief_label=True,
            disruptive=False,
            outcome_cell=(True, False),
            valid_pair=True,
            belief_available=True,
            invalid_reason=None,
        ),
        PairMetrics(
            pair_id="ff",
            evaluated_steps=10,
            a_new=0.2,
            a_orig=0.8,
            belief_mlp_new_match=False,
            belief_linear_new_match=False,
            action_label=False,
            belief_label=False,
            disruptive=False,
            outcome_cell=(False, False),
            valid_pair=True,
            belief_available=True,
            invalid_reason=None,
        ),
        PairMetrics(
            pair_id="invalid",
            evaluated_steps=0,
            a_new=0.0,
            a_orig=0.0,
            belief_mlp_new_match=None,
            belief_linear_new_match=None,
            action_label=None,
            belief_label=None,
            disruptive=None,
            outcome_cell=None,
            valid_pair=False,
            belief_available=False,
            invalid_reason="missing",
        ),
    ]

    agg = aggregate_results(metrics, stopped_early=False, early_stop_reason=None)

    assert agg.tt_count == 1
    assert agg.tf_count == 1
    assert agg.ft_count == 0
    assert agg.ff_count == 1
    assert agg.tt_over_tt_tf == 0.5
    assert agg.tt_over_tt_ft == 1.0


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
                "grid_a_path": str(record.spec.grid_a_path),
                "grid_b_path": str(record.spec.grid_b_path),
                "goal_orig": list(record.spec.goal_orig),
                "goal_new": list(record.spec.goal_new),
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
    assert summary["tt_count"] == 1
    assert summary["tf_count"] == 1


def test_failure_missing_trace_path(tmp_path: Path):
    record = _create_pair_record(tmp_path, "pair_missing_trace", [_build_step("RIGHT") for _ in range(5)])
    record.artifacts.patched_trace_path.unlink()

    metric = evaluate_pair(record)

    assert metric.valid_pair is False
    assert "trace loading failed" in (metric.invalid_reason or "")


def test_failure_goal_not_moved_only(tmp_path: Path):
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
                "grid_a_path": str(record.spec.grid_a_path),
                "grid_b_path": str(record.spec.grid_b_path),
                "goal_orig": list(record.spec.goal_orig),
                "goal_new": list(record.spec.goal_new),
                "a_trace_path": str(record.artifacts.a_trace_path),
                "b_trace_path": str(record.artifacts.b_trace_path),
                "patched_trace_path": str(record.artifacts.patched_trace_path),
            }
        )
    manifest_path.write_text(json.dumps(manifest_rows, indent=2))

    output_dir = tmp_path / "early_stop_results"
    counterfactual_activation_patching(
        manifest_path=str(manifest_path),
        output_dir=str(output_dir),
        expected_k=4,
    )

    summary = json.loads((output_dir / "aggregate_summary.json").read_text())
    assert summary["stopped_early"] is True
    assert summary["total_pairs_processed"] == 3
