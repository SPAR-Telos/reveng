from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from reveng.experiments import reasoning_belief_action as rba


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_classify_action_events_distinguishes_sustained_transient_and_recovery() -> None:
    rows = []
    for idx, optimal in enumerate([True, False, False, True, False, True]):
        rows.append(
            {
                "example_id": "e",
                "trajectory_id": "t",
                "failure_category": "short_loop",
                "reasoning_step_idx": idx,
                "reasoning_progress": idx / 5,
                "action_label": "UP" if optimal else "LEFT",
                "action_is_optimal": optimal,
            }
        )
    events = rba.classify_action_events(rows)
    event_types = [row["event_type"] for row in events]
    assert event_types.count("sustained_optimal_to_suboptimal") == 1
    assert event_types.count("transient_optimal_to_suboptimal") == 1
    assert event_types.count("suboptimal_to_optimal_recovery") == 2


def test_classify_belief_shifts_marks_persistent_error() -> None:
    rows = []
    for idx, answer in enumerate(["no", "yes", "yes", "yes"]):
        rows.append(
            {
                "example_id": "e",
                "trajectory_id": "t",
                "failure_category": "short_loop",
                "reasoning_step_idx": idx,
                "reasoning_progress": idx / 3,
                "question_id": "wall_right",
                "question_family": "wall_directional",
                "answer_key": answer,
                "ground_truth_key": "no",
                "answer_valid": True,
                "belief_is_error": answer != "no",
            }
        )
    shifts = rba.classify_belief_shifts(rows)
    assert shifts[0]["shift_type"] == "belief_error_onset"
    assert shifts[0]["persistent_belief_error"] is True


def test_validate_activation_spans_rejects_boundary_overlap(tmp_path: Path) -> None:
    path = tmp_path / "step_activation_rows.csv"
    _write_csv(
        path,
        [
            {"example_id": "e", "layer": 15, "reasoning_step_idx": 0, "step_start_token": 0, "step_end_token": 2},
            {"example_id": "e", "layer": 15, "reasoning_step_idx": 1, "step_start_token": 2, "step_end_token": 4},
        ],
    )
    result = rba.validate_activation_spans(path)
    assert result["valid"] is False
    assert result["n_overlaps"] == 1


def test_run_observational_writes_transition_outputs(tmp_path: Path) -> None:
    drift = tmp_path / "drift"
    drift.mkdir()
    prefix_rows = []
    for idx, (action, optimal) in enumerate([("UP", True), ("LEFT", False), ("LEFT", False)]):
        prefix_rows.append(
            {
                "example_id": "e",
                "trajectory_id": "t",
                "step_index": 1,
                "failure_category": "short_loop",
                "reasoning_step_idx": idx,
                "reasoning_progress": idx / 2,
                "action_label": action,
                "action_is_optimal": optimal,
                "revealed_analysis_text": "" if idx == 0 else f"reasoning {idx}",
            }
        )
    _write_csv(drift / "prefix_action_rows.csv", prefix_rows)
    truths = {
        "wall_left": "no",
        "wall_right": "no",
        "wall_up": "no",
        "wall_down": "yes",
        "has_key": "no",
        "door_open": "no",
        "agent_location": {"row": 1, "col": 1},
        "goal_location": {"row": 1, "col": 2},
        "key_location": {"row": -1, "col": -1},
        "door_location": {"row": -1, "col": -1},
        "hit_wall_after_up": "no",
        "has_key_after_up": "no",
        "door_open_after_up": "no",
        "hit_wall_after_left": "no",
        "has_key_after_left": "no",
        "door_open_after_left": "no",
    }
    candidates = tmp_path / "candidates.csv"
    _write_csv(
        candidates,
        [
            {
                "example_id": "e",
                "grid_text": "  0 1 2\n0 # # #\n1 # A G\n2 # # #",
                "carrying_key": False,
                "probe_truths_json": json.dumps(truths),
            }
        ],
    )

    def fake_query(prompt: str, seed: int):
        if '{"row": <int>, "col": <int>}' in prompt:
            if "goal's location" in prompt:
                return '{"row": 1, "col": 2}', {}
            if "agent's current location" in prompt:
                return '{"row": 1, "col": 1}', {}
            return '{"row": -1, "col": -1}', {}
        if "immediately DOWN" in prompt:
            return "A", {}
        return "B", {}

    out = tmp_path / "out"
    manifest = rba.run_reasoning_belief_action_observational(
        drift_run_dir=str(drift),
        candidate_rows_path=str(candidates),
        output_dir=str(out),
        belief_query_fn=fake_query,
        verbose=False,
    )
    assert manifest["n_positions"] == 3
    assert manifest["belief_valid_parse_rate"] == 1.0
    assert (out / "action_transition_rows.csv").exists()
    assert (out / "feature_association_fdr.csv").exists()
    events = list(csv.DictReader((out / "action_transition_rows.csv").open()))
    assert events[0]["event_type"] == "sustained_optimal_to_suboptimal"

    checkpoint = out / "belief_rows.jsonl"
    failed_retry = json.loads(checkpoint.read_text().splitlines()[0])
    failed_retry["query_error"] = "late failed retry"
    failed_retry["answer_valid"] = False
    failed_retry["answer_key"] = "invalid"
    with checkpoint.open("a") as handle:
        handle.write(json.dumps(failed_retry) + "\n")

    def should_not_query(prompt: str, seed: int):
        raise AssertionError("Successful checkpoint rows should take precedence over later failures")

    resumed = rba.run_reasoning_belief_action_observational(
        drift_run_dir=str(drift),
        candidate_rows_path=str(candidates),
        output_dir=str(out),
        belief_query_fn=should_not_query,
        verbose=False,
    )
    assert resumed["belief_valid_parse_rate"] == 1.0


def test_run_observational_resumes_after_interruption_and_truncated_tail(tmp_path: Path) -> None:
    drift = tmp_path / "drift"
    drift.mkdir()
    _write_csv(
        drift / "prefix_action_rows.csv",
        [
            {
                "example_id": "e",
                "trajectory_id": "t",
                "step_index": 1,
                "failure_category": "none",
                "reasoning_step_idx": 0,
                "reasoning_progress": 0.0,
                "action_label": "UP",
                "action_is_optimal": True,
                "revealed_analysis_text": "",
            }
        ],
    )
    truths = {
        "wall_left": "no",
        "wall_right": "no",
        "wall_up": "no",
        "wall_down": "yes",
        "has_key": "no",
        "door_open": "no",
        "agent_location": {"row": 1, "col": 1},
        "goal_location": {"row": 1, "col": 2},
        "key_location": {"row": -1, "col": -1},
        "door_location": {"row": -1, "col": -1},
        "hit_wall_after_up": "no",
        "has_key_after_up": "no",
        "door_open_after_up": "no",
    }
    candidates = tmp_path / "candidates.csv"
    _write_csv(
        candidates,
        [
            {
                "example_id": "e",
                "grid_text": "  0 1 2\n0 # # #\n1 # A G\n2 # # #",
                "carrying_key": False,
                "probe_truths_json": json.dumps(truths),
            }
        ],
    )
    out = tmp_path / "out"
    first_run_calls = 0

    def interrupting_query(prompt: str, seed: int):
        nonlocal first_run_calls
        first_run_calls += 1
        if first_run_calls == 5:
            raise KeyboardInterrupt()
        if '{"row": <int>, "col": <int>}' in prompt:
            return '{"row": -1, "col": -1}', {}
        return "B", {}

    with pytest.raises(KeyboardInterrupt):
        rba.run_reasoning_belief_action_observational(
            drift_run_dir=str(drift),
            candidate_rows_path=str(candidates),
            output_dir=str(out),
            belief_query_fn=interrupting_query,
            checkpoint_fsync_every=1,
            verbose=False,
        )
    status = json.loads((out / "run_status.json").read_text())
    assert status["status"] == "interrupted"
    assert status["successful_queries"] == 4

    with (out / "belief_rows.jsonl").open("a") as handle:
        handle.write('{"truncated":')

    resumed_prompts: list[str] = []

    def resumed_query(prompt: str, seed: int):
        resumed_prompts.append(prompt)
        if '{"row": <int>, "col": <int>}' in prompt:
            return '{"row": -1, "col": -1}', {}
        return "B", {}

    manifest = rba.run_reasoning_belief_action_observational(
        drift_run_dir=str(drift),
        candidate_rows_path=str(candidates),
        output_dir=str(out),
        belief_query_fn=resumed_query,
        checkpoint_fsync_every=1,
        verbose=False,
    )
    assert len(resumed_prompts) == 9
    assert manifest["n_belief_queries"] == 13
    final_status = json.loads((out / "run_status.json").read_text())
    assert final_status["status"] == "completed"
    assert final_status["successful_queries"] == 13
    assert final_status["remaining_queries"] == 0
    assert final_status["malformed_checkpoint_lines_ignored"] == 1


def test_run_observational_rejects_incompatible_resume(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "run_config.json").write_text(json.dumps({"different": "configuration"}))
    drift = tmp_path / "drift"
    drift.mkdir()
    _write_csv(
        drift / "prefix_action_rows.csv",
        [
            {
                "example_id": "e",
                "trajectory_id": "t",
                "step_index": 1,
                "failure_category": "none",
                "reasoning_step_idx": 0,
                "reasoning_progress": 0.0,
                "action_label": "INVALID",
                "action_is_optimal": "",
                "revealed_analysis_text": "",
            }
        ],
    )
    candidates = tmp_path / "candidates.csv"
    _write_csv(
        candidates,
        [{"example_id": "e", "grid_text": "A", "carrying_key": False, "probe_truths_json": "{}"}],
    )
    with pytest.raises(ValueError, match="saved run configuration differs"):
        rba.run_reasoning_belief_action_observational(
            drift_run_dir=str(drift),
            candidate_rows_path=str(candidates),
            output_dir=str(out),
            belief_query_fn=lambda prompt, seed: ("B", {}),
            verbose=False,
        )
