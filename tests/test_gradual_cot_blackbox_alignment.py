from pathlib import Path
import math

from reveng.experiments.gradual_cot_blackbox_alignment import (
    _extract_analysis_and_final,
    _load_pair_rows_from_failure_candidates,
    _load_pair_rows_from_trajectory_candidates,
    _parse_wall_json_label,
    _sentence_boundary_metadata,
    summarize_gradual_cot_alignment_rows,
)


def test_extract_analysis_and_final_splits_released_trace():
    output_text = (
        '<|channel|>analysis<|message|>first sentence. second sentence.'
        '<|end|><|start|>assistant<|channel|>final<|message|>{"action": "DOWN"}<|return|>'
    )
    analysis, final_text = _extract_analysis_and_final(output_text)
    assert analysis == 'first sentence. second sentence.'
    assert 'DOWN' in final_text
    assert 'analysis' not in final_text.lower()


def test_sentence_boundary_metadata_tracks_completed_sentences_only():
    full = 'Alpha. Beta. Gamma.'
    revealed = 'Alpha. Beta'
    idx, coverage = _sentence_boundary_metadata(full, revealed)
    assert idx == 1
    assert 0.0 < coverage < 1.0


def test_parse_wall_json_label_reads_json_and_falls_back():
    assert _parse_wall_json_label('{"label": "yes"}') == 'yes'
    assert _parse_wall_json_label(' { "label": "unknown" } ') == 'unknown'
    assert _parse_wall_json_label('no') == 'no'
    assert _parse_wall_json_label('{"other": "yes"}') == 'invalid'


def test_failure_candidate_loader_builds_wall_question_rows():
    rows = _load_pair_rows_from_failure_candidates(
        Path('data/behavioral_probes/trajectory_instances_recomputed_optimal/trajectory_selection_candidates.csv'),
        max_wall_hit=1,
        max_avoidable_detour=1,
        max_backtrack=1,
        max_baseline=1,
    )
    assert rows
    assert {row['question_id'] for row in rows}.issubset({'wall_left', 'wall_right', 'wall_up', 'wall_down'})
    example_ids = {row['example_id'] for row in rows}
    assert len(rows) == 4 * len(example_ids)
    assert any(row['primary_step_failure_mode'] == 'wall_hit' for row in rows)


def test_trajectory_candidate_loader_balances_failure_modes_and_context():
    rows = _load_pair_rows_from_trajectory_candidates(
        Path('data/behavioral_probes/trajectory_instances_recomputed_optimal/trajectory_selection_candidates.csv'),
        slice_type='balanced_failure_modes',
        rows_per_mode=1,
    )
    assert rows
    example_ids = {row['example_id'] for row in rows}
    assert len(rows) == 4 * len(example_ids)
    modes = {row['primary_step_failure_mode'] for row in rows}
    assert 'short_loop' in modes
    assert 'oscillation_2cycle' in modes
    assert any(row['selection_stage'] == 'context' for row in rows)


def test_trajectory_candidate_loader_supports_non_failure_controls():
    rows = _load_pair_rows_from_trajectory_candidates(
        Path('data/behavioral_probes/trajectory_instances_recomputed_optimal/trajectory_selection_candidates.csv'),
        slice_type='non_failure_controls',
        rows_per_mode=1,
        max_examples=3,
    )
    assert rows
    example_ids = {row['example_id'] for row in rows}
    assert len(example_ids) == 3
    assert len(rows) == 12
    assert all(row['primary_step_failure_mode'] == 'none' for row in rows)
    assert all(row['selection_stage'] == 'context' for row in rows)


def test_trajectory_candidate_loader_supports_specific_primary_failure_mode():
    rows = _load_pair_rows_from_trajectory_candidates(
        Path('data/behavioral_probes/trajectory_instances_recomputed_optimal/trajectory_selection_candidates.csv'),
        slice_type='primary_failure_mode:short_loop',
        rows_per_mode=1,
        max_examples=2,
    )
    assert rows
    assert len(rows) == 8
    assert all(row['primary_step_failure_mode'] == 'short_loop' for row in rows)


def test_summarize_rows_supports_logprob_only_wall_mode():
    rows = [
        {
            "example_id": "ex1",
            "trajectory_id": "traj1",
            "step_index": "0",
            "question_id": "wall_right",
            "reasoning_reveal_pct": 0,
            "ground_truth_label": "no",
            "whitebox_prediction_pre": "no",
            "whitebox_prediction_post": "no",
            "blackbox_answer": "no",
            "blackbox_correct": True,
            "blackbox_entropy": 0.01,
            "blackbox_belief_action_consistency": "potentially_consistent",
            "blackbox_belief_action_consistency_mc": "potentially_consistent",
            "blackbox_action_is_optimal": True,
            "action_mc_modal_is_optimal": True,
            "action_greedy_agreement_rate": 1.0,
            "action_mc_agreement_rate": 0.75,
            "action_mc_entropy": 0.5,
            "wall_greedy_modal_correct": None,
            "wall_greedy_agreement_rate": None,
            "wall_mc_modal_correct": None,
            "wall_mc_entropy": None,
            "wall_mc_agreement_rate": None,
            "source_dataset": "failure",
            "primary_step_failure_mode": "backtrack",
        },
        {
            "example_id": "ex1",
            "trajectory_id": "traj1",
            "step_index": "0",
            "question_id": "wall_right",
            "reasoning_reveal_pct": 100,
            "ground_truth_label": "no",
            "whitebox_prediction_pre": "no",
            "whitebox_prediction_post": "no",
            "blackbox_answer": "no",
            "blackbox_correct": True,
            "blackbox_entropy": 0.02,
            "blackbox_belief_action_consistency": "potentially_consistent",
            "blackbox_belief_action_consistency_mc": "potentially_consistent",
            "blackbox_action_is_optimal": True,
            "action_mc_modal_is_optimal": True,
            "action_greedy_agreement_rate": 1.0,
            "action_mc_agreement_rate": 1.0,
            "action_mc_entropy": 0.0,
            "wall_greedy_modal_correct": None,
            "wall_greedy_agreement_rate": None,
            "wall_mc_modal_correct": None,
            "wall_mc_entropy": None,
            "wall_mc_agreement_rate": None,
            "source_dataset": "failure",
            "primary_step_failure_mode": "backtrack",
        },
    ]
    revised, gradual, failure = summarize_gradual_cot_alignment_rows(rows, out_dir=None, reveal_pcts=(0, 100))
    assert revised[0]["BB_0"] == 1.0
    assert revised[0]["BB_100"] == 1.0
    row0 = next(r for r in gradual if r["reasoning_reveal_pct"] == 0)
    assert row0["blackbox_wall_accuracy"] == 1.0
    assert row0["action_mc_mean_entropy"] == 0.5
    assert math.isnan(row0["wall_greedy_modal_accuracy"])
    assert failure[0]["failure_mode"] == "backtrack"
