from __future__ import annotations

import csv
import json
from pathlib import Path

from reveng.experiments import step_reasoning_drift as srd


def test_segment_reasoning_trace_uses_sentence_level_chunks_with_paragraph_breaks() -> None:
    text = (
        "We should inspect the corridor.\n"
        "The right route looks open.\n\n"
        "Now compare the lower route.\n"
        "It reaches the goal faster.\n\n"
        "So we should move DOWN."
    )
    steps = srd.segment_reasoning_trace(text, segmentation_mode="paragraph_or_sentence")
    assert [step.step_idx for step in steps] == [1, 2, 3, 4, 5]
    assert "corridor" in steps[0].text
    assert "right route" in steps[1].text
    assert "lower route" in steps[2].text
    assert steps[-1].text.strip() == "So we should move DOWN."


def test_segment_reasoning_trace_caps_steps_without_losing_text() -> None:
    text = " ".join(f"Sentence {idx}." for idx in range(10))
    steps = srd.segment_reasoning_trace(text, segmentation_mode="sentence", max_steps=4)

    assert len(steps) <= 4
    assert steps[0].start_char == 0
    assert steps[-1].end_char == len(text)
    assert all(text[step.start_char:step.end_char] == step.text for step in steps)


def test_packed_chunks_balance_the_final_group() -> None:
    text = " ".join(f"Sentence {idx} has five words." for idx in range(96))
    sentences = srd.chunk_doorkey_reasoning_trace(text, trace_id="balanced")
    chunks = srd.pack_to_analysis_chunks(sentences, max_chunks=32, raw_text=text)

    word_counts = [len(chunk.text.split()) for chunk in chunks]
    assert len(chunks) == 32
    assert chunks[0].sentence_start == 0
    assert chunks[-1].sentence_end == 95
    assert max(word_counts) - min(word_counts) <= 5
    assert word_counts[-1] <= 2 * word_counts[-2]


def test_doorkey_chunker_protects_false_sentence_boundaries() -> None:
    text = (
        "I need to get the key first. The key is at (2, 5). "
        "The score is 0.7. I should avoid walls, e.g. the left wall. "
        "1. RIGHT moves closer. 2. LEFT is worse. Thus I choose RIGHT.\n"
        '{"action": "RIGHT"}'
    )

    chunks = srd.chunk_doorkey_reasoning_trace(text, trace_id="toy")
    analysis_chunks = srd.pack_to_analysis_chunks(chunks, raw_text=text)

    assert [chunk.kind for chunk in chunks][-1] == "final_action"
    assert any(chunk.text == "The key is at (2, 5)." for chunk in chunks)
    assert any(chunk.text == "The score is 0.7." for chunk in chunks)
    assert any("e.g. the left wall." in chunk.text for chunk in chunks)
    assert any(chunk.text.startswith("1. RIGHT") for chunk in chunks)
    assert any(chunk.text.startswith("2. LEFT") for chunk in chunks)
    assert srd.verify_doorkey_chunks(text, chunks, analysis_chunks) == []


def test_doorkey_chunker_flags_internal_action_json_without_final_split() -> None:
    text = 'We must output JSON: {"action":"RIGHT"}.\n\nLet us double check. Now choose UP.'

    chunks = srd.chunk_doorkey_reasoning_trace(text, trace_id="toy")

    assert all(chunk.kind == "reasoning" for chunk in chunks)
    assert any(chunk.contains_action_json_mention for chunk in chunks)
    assert chunks[0].text.startswith("We must output JSON")


def test_canonical_sentence_steps_exclude_final_action_and_validate_spans(tmp_path: Path) -> None:
    text = 'Inspect the route. Move right.\n{"action":"RIGHT"}'
    boundary_path = tmp_path / "sentences.csv"
    _rows = [
        {
            "path": "data/toy_traj.json",
            "step_id": 0,
            "sentence_id": 0,
            "kind": "reasoning",
            "char_start": 0,
            "char_end": 18,
            "text": "Inspect the route.",
        },
        {
            "path": "data/toy_traj.json",
            "step_id": 0,
            "sentence_id": 1,
            "kind": "reasoning",
            "char_start": 19,
            "char_end": 30,
            "text": "Move right.",
        },
        {
            "path": "data/toy_traj.json",
            "step_id": 0,
            "sentence_id": 2,
            "kind": "final_action",
            "char_start": 31,
            "char_end": len(text),
            "text": '{"action":"RIGHT"}',
        },
    ]
    with boundary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_rows[0]))
        writer.writeheader()
        writer.writerows(_rows)

    index = srd.load_canonical_sentence_boundaries(boundary_path)
    steps = srd.canonical_sentence_steps(
        boundary_index=index,
        trajectory_id="toy_traj",
        step_index=0,
        analysis_text=text,
    )

    assert [step.text for step in steps] == ["Inspect the route.", "Move right."]
    assert [step.canonical_sentence_id for step in steps] == [0, 1]


def test_summarize_prefix_behavior_detects_wrong_turn_commitment_and_recovery() -> None:
    rows = [
        srd.PrefixStepEvaluation(0, 0, 0, "UP", True, False, ""),
        srd.PrefixStepEvaluation(1, 1, 10, "UP", True, False, "step one"),
        srd.PrefixStepEvaluation(2, 2, 20, "LEFT", False, True, "step two"),
    ]
    summary = srd.summarize_prefix_behavior(rows, final_action="LEFT", persistence_threshold=0.5)
    assert summary["commitment_step"] == 2
    assert summary["wrong_turn_step_strict"] == 2
    assert summary["wrong_turn_step_majority"] == 2
    assert summary["recovery_step"] is None
    assert summary["trajectory_remains_optimal"] is False


def test_summarize_prefix_behavior_reports_invalid_prefixes() -> None:
    rows = [
        srd.PrefixStepEvaluation(0, 0, 0, "UP", True, False, ""),
        srd.PrefixStepEvaluation(1, 1, 10, "INVALID", None, None, "step one"),
    ]

    summary = srd.summarize_prefix_behavior(rows, final_action="LEFT", persistence_threshold=0.5)

    assert summary["n_invalid_prefixes"] == 1
    assert summary["valid_prefix_fraction"] == 0.5
    assert summary["trajectory_remains_optimal"] is False


def test_action_entropy_from_sampled_actions() -> None:
    summary = srd.summarize_action_samples(["UP", "UP", "LEFT", "invalid"])

    assert json.loads(summary["action_mc_valid_actions_json"]) == ["UP", "UP", "LEFT"]
    assert json.loads(summary["action_mc_probs_json"]) == {"LEFT": 1 / 3, "UP": 2 / 3}
    assert summary["action_mc_valid_count"] == 3
    assert summary["action_mc_invalid_count"] == 1
    assert 0.91 < summary["action_mc_entropy"] < 0.92


def test_compute_reasoning_geometry_metrics_reports_aligned_change_and_anchor_cosine() -> None:
    vectors = [
        [1.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
    ]
    rows = srd.compute_reasoning_geometry_metrics(
        vectors,
        anchor_vectors_by_bucket={
            1: [1.0, 0.0],
            2: [1.0, 0.0],
            3: [0.0, 1.0],
        },
    )
    assert len(rows) == 3
    assert rows[0]["aligned_change"] is None
    assert rows[1]["aligned_change"] is not None
    assert rows[2]["net_change"] > rows[1]["net_change"]
    assert rows[2]["optimality_anchor_cosine"] is not None


def test_run_step_reasoning_drift_rejects_public_activation_source(tmp_path: Path) -> None:
    try:
        srd.run_step_reasoning_drift_experiment(
            output_dir=str(tmp_path),
            activation_source="public_release",
        )
    except ValueError as exc:
        assert "pre/post reasoning snapshots" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for public_release activation source")


def test_local_hidden_state_collector_reports_missing_cuda(monkeypatch) -> None:
    monkeypatch.setattr(srd.torch.cuda, "is_available", lambda: False)
    try:
        srd.LocalHiddenStateCollector(
            "unused-model",
            device="cpu",
            device_map="auto",
        )
    except RuntimeError as exc:
        assert "runtime exposure" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected RuntimeError for missing CUDA")


def test_run_step_reasoning_drift_writes_behavioral_outputs(tmp_path: Path, monkeypatch) -> None:
    example = srd.DriftExample(
        example_id="toy_step_0",
        trajectory_id="toy_traj",
        step_index=0,
        grid_text="# # #\n# A G\n# # #",
        carrying_key=False,
        observed_action="LEFT",
        optimal_actions=("UP",),
        is_optimal_action=False,
        failure_category="short_loop",
        source_dataset="trajectory_candidates:short_loop",
        selection_stage="failure",
    )

    monkeypatch.setattr(srd, "_load_drift_examples", lambda **_: [example])
    monkeypatch.setattr(
        srd,
        "_load_local_trajectory_payload",
        lambda trajectory_id, trajectory_dir: {
            "steps": [
                {
                    "output_text": (
                        "<|channel|>analysis<|message|>"
                        "First inspect the route. Then choose the turn."
                        "<|end|><|start|>assistant<|channel|>final<|message|>"
                        '{"action":"LEFT"}'
                    )
                }
            ]
        },
    )

    returned_actions = iter(
        [
            ("UP", '{"action":"UP"}', {}),
            ("UP", '{"action":"UP"}', {}),
            ("LEFT", '{"action":"LEFT"}', {}),
        ]
    )

    def fake_query(prompt: str, seed: int):
        return next(returned_actions)

    srd.run_step_reasoning_drift_experiment(
        output_dir=str(tmp_path),
        collect_activations=False,
        action_query_fn=fake_query,
        verbose=False,
    )

    step_rows_path = tmp_path / "prefix_action_rows.csv"
    traj_rows_path = tmp_path / "trajectory_wrong_turn_summary.csv"
    manifest_path = tmp_path / "manifest.json"
    assert step_rows_path.exists()
    assert traj_rows_path.exists()
    assert manifest_path.exists()

    with open(step_rows_path, newline="") as handle:
        step_rows = list(csv.DictReader(handle))
    with open(traj_rows_path, newline="") as handle:
        traj_rows = list(csv.DictReader(handle))
    manifest = json.loads(manifest_path.read_text())

    assert len(step_rows) == 3
    assert traj_rows[0]["wrong_turn_step_strict"] == "2"
    assert traj_rows[0]["commitment_step"] == "2"
    assert traj_rows[0]["trajectory_group"] == "go_wrong"
    assert manifest["status"] == "completed"
    assert manifest["n_examples"] == 1


def test_run_step_reasoning_drift_writes_action_entropy_fields(tmp_path: Path, monkeypatch) -> None:
    example = srd.DriftExample(
        example_id="toy_step_0",
        trajectory_id="toy_traj",
        step_index=0,
        grid_text="# # #\n# A G\n# # #",
        carrying_key=False,
        observed_action="LEFT",
        optimal_actions=("UP",),
        is_optimal_action=False,
        failure_category="short_loop",
        source_dataset="trajectory_candidates:short_loop",
        selection_stage="failure",
    )

    monkeypatch.setattr(srd, "_load_drift_examples", lambda **_: [example])
    monkeypatch.setattr(
        srd,
        "_load_local_trajectory_payload",
        lambda trajectory_id, trajectory_dir: {
            "steps": [
                {
                    "output_text": (
                        "<|channel|>analysis<|message|>"
                        "First inspect the route."
                        "<|end|><|start|>assistant<|channel|>final<|message|>"
                        '{"action":"LEFT"}'
                    )
                }
            ]
        },
    )

    greedy_actions = iter(
        [
            ("UP", '{"action":"UP"}', {}),
            ("LEFT", '{"action":"LEFT"}', {}),
        ]
    )
    mc_actions = iter(
        [
            ("UP", '{"action":"UP"}', {}),
            ("LEFT", '{"action":"LEFT"}', {}),
            ("UP", '{"action":"UP"}', {}),
            ("INVALID", "not json", {}),
        ]
    )

    srd.run_step_reasoning_drift_experiment(
        output_dir=str(tmp_path),
        collect_activations=False,
        action_query_fn=lambda prompt, seed: next(greedy_actions),
        action_mc_query_fn=lambda prompt, seed: next(mc_actions),
        action_mc_sample_repeats=2,
        action_mc_temperature=0.7,
        verbose=False,
    )

    with open(tmp_path / "prefix_action_rows.csv", newline="") as handle:
        step_rows = list(csv.DictReader(handle))

    assert step_rows[0]["action_label"] == "UP"
    assert step_rows[0]["action_mc_sample_repeats"] == "2"
    assert json.loads(step_rows[0]["action_mc_valid_actions_json"]) == ["UP", "LEFT"]
    assert float(step_rows[0]["action_mc_entropy"]) == 1.0
    assert step_rows[1]["action_label"] == "LEFT"
    assert json.loads(step_rows[1]["action_mc_valid_actions_json"]) == ["UP"]
    assert step_rows[1]["action_mc_invalid_count"] == "1"


def test_run_step_reasoning_drift_resumes_within_example(tmp_path: Path, monkeypatch) -> None:
    example = srd.DriftExample(
        example_id="toy_step_0",
        trajectory_id="toy_traj",
        step_index=0,
        grid_text="# # #\n# A G\n# # #",
        carrying_key=False,
        observed_action="LEFT",
        optimal_actions=("UP",),
        is_optimal_action=False,
        failure_category="short_loop",
        source_dataset="trajectory_candidates:short_loop",
        selection_stage="failure",
    )
    monkeypatch.setattr(srd, "_load_drift_examples", lambda **_: [example])
    monkeypatch.setattr(
        srd,
        "_load_local_trajectory_payload",
        lambda trajectory_id, trajectory_dir: {
            "steps": [
                {
                    "output_text": (
                        "<|channel|>analysis<|message|>"
                        "First inspect the route. Then choose the turn."
                        "<|end|><|start|>assistant<|channel|>final<|message|>"
                        '{"action":"LEFT"}'
                    )
                }
            ]
        },
    )
    first_calls = 0

    def interrupted_query(prompt: str, seed: int):
        nonlocal first_calls
        first_calls += 1
        if first_calls == 2:
            raise KeyboardInterrupt
        return "UP", '{"action":"UP"}', {}

    try:
        srd.run_step_reasoning_drift_experiment(
            output_dir=str(tmp_path),
            collect_activations=False,
            action_query_fn=interrupted_query,
            verbose=False,
        )
    except KeyboardInterrupt:
        pass
    else:  # pragma: no cover
        raise AssertionError("Expected the simulated interruption.")

    prefix_checkpoints = (tmp_path / "prefix_query_checkpoints.jsonl").read_text().splitlines()
    assert len(prefix_checkpoints) == 1

    resumed_actions = iter(
        [
            ("UP", '{"action":"UP"}', {}),
            ("LEFT", '{"action":"LEFT"}', {}),
        ]
    )
    resumed_calls = 0

    def resumed_query(prompt: str, seed: int):
        nonlocal resumed_calls
        resumed_calls += 1
        return next(resumed_actions)

    srd.run_step_reasoning_drift_experiment(
        output_dir=str(tmp_path),
        collect_activations=False,
        action_query_fn=resumed_query,
        verbose=False,
    )

    with (tmp_path / "prefix_action_rows.csv").open(newline="") as handle:
        step_rows = list(csv.DictReader(handle))
    assert resumed_calls == 2
    assert len(step_rows) == 3
    assert len((tmp_path / "prefix_query_checkpoints.jsonl").read_text().splitlines()) == 3


def test_regenerate_step_activations_reuses_saved_prompts(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    prompt = source / "prompt.txt"
    prompt.write_text("before first second")
    rows = []
    for layer in (8, 15):
        for step_idx, start, end in ((0, 0, 6), (1, 6, 12), (2, 12, 19)):
            rows.append(
                {
                    "example_id": "e",
                    "trajectory_id": "t",
                    "step_index": 1,
                    "reasoning_step_idx": step_idx,
                    "layer": layer,
                    "prompt_text_path": str(prompt),
                    "step_start_char_in_prompt": start,
                    "step_end_char_in_prompt": end,
                    "local_model_name_or_path": "model",
                }
            )
    with (source / "step_activation_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    class FakeCollector:
        def __init__(self, *args, **kwargs):
            pass

        def collect(self, *, step_spans_in_prompt, layers, output_dir, **kwargs):
            output = []
            previous_end = -1
            for step_idx, _start, _end in sorted(step_spans_in_prompt):
                start = previous_end + 1
                end = start + 1
                previous_end = end
                for layer in layers:
                    output.append(
                        {
                            "reasoning_step_idx": step_idx,
                            "layer": layer,
                            "step_start_token": start,
                            "step_end_token": end,
                            "step_last_token_activation_path": "last.pt",
                            "step_mean_activation_path": "mean.pt",
                            "step_boundary_window_start_token": max(0, end - 2),
                            "step_boundary_window_activation_path": "window.pt",
                            "step_boundary_window_n_tokens": 3,
                        }
                    )
            return output

    monkeypatch.setattr(srd, "LocalHiddenStateCollector", FakeCollector)
    monkeypatch.setattr(srd, "rebuild_reasoning_geometry_from_activations", lambda **kwargs: None)
    out = tmp_path / "out"
    srd.regenerate_step_reasoning_activations(
        source_run_dir=str(source),
        output_dir=str(out),
        verbose=False,
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["behavioral_queries_rerun"] is False
    assert manifest["activation_spans_valid"] is True
    assert manifest["n_activation_rows"] == 6
