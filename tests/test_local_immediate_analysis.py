from reveng.experiments.local_immediate_analysis import (
    _annotate_commitment,
    _belief_shifts,
    _events,
    _materialize_standard_rows,
)


def test_commitment_is_first_stable_suffix_matching_full_trace_action() -> None:
    rows = [
        {
            "example_id": "e",
            "trajectory_id": "t",
            "step_index": 0,
            "reasoning_step_idx": index,
            "reasoning_progress": index / 3,
            "analysis_unit": "sentence",
            "action_label": action,
            "action_is_optimal": action == "UP",
        }
        for index, action in enumerate(("UP", "LEFT", "UP", "UP"))
    ]

    _annotate_commitment(rows)
    events = _events(rows)

    assert [row["commitment_onset"] for row in rows] == [False, False, True, False]
    assert sum(row["event_type"] == "commitment_onset" for row in events) == 1


def test_belief_shifts_distinguish_error_onset_and_recovery() -> None:
    rows = [
        {
            "example_id": "e",
            "trajectory_id": "t",
            "step_index": 0,
            "reasoning_step_idx": index,
            "reasoning_progress": index / 2,
            "question_id": "wall_left",
            "question_family": "adjacent_wall",
            "answer_key": answer,
            "ground_truth_key": "no",
            "belief_is_error": error,
        }
        for index, (answer, error) in enumerate(
            (("no", "False"), ("yes", "True"), ("no", "False"))
        )
    ]

    shifts = _belief_shifts(rows)

    assert [row["shift_type"] for row in shifts] == [
        "belief_error_onset",
        "belief_error_recovery",
    ]


def test_environment_beliefs_use_environment_step_index() -> None:
    action_rows = [
        {
            "example_id": "t_step_003",
            "trajectory_id": "t",
            "step_index": "3",
            "position_index": "42",
            "analysis_unit": "environment_step",
            "reasoning_progress": "1.0",
            "action_probabilities_json": '{"UP": 1.0}',
            "action_label": "UP",
            "action_is_optimal": "True",
        }
    ]
    belief_rows = [
        {
            "example_id": "t_step_003",
            "trajectory_id": "t",
            "step_index": "3",
            "position_index": "42",
            "analysis_unit": "environment_step",
            "reasoning_progress": "1.0",
            "question_id": "wall_up",
            "answer_key": "no",
            "ground_truth_key": "no",
            "belief_is_error": "False",
            "entropy_bits": "0.0",
            "probabilities_json": '{"no": 1.0}',
        }
    ]

    positions, beliefs = _materialize_standard_rows(action_rows, belief_rows)

    assert positions[0]["reasoning_step_idx"] == 3
    assert beliefs[0]["reasoning_step_idx"] == 3
