import numpy as np
import pandas as pd

from reveng.experiments.cpd_signal_prediction import (
    PRIMARY_BELIEFS,
    add_belief_change_summaries,
    add_future_commitment_targets,
    add_future_point_targets,
    attach_frozen_change_points,
    paired_model_comparison,
)


def belief_frame() -> pd.DataFrame:
    rows = []
    for step, yes in enumerate((0.2, 0.4, 0.1)):
        row = {"example_id": "a", "reasoning_step_idx": step}
        for belief in PRIMARY_BELIEFS:
            row[f"belief_{belief}_prob_yes"] = yes
            row[f"belief_{belief}_prob_no"] = 1.0 - yes
            row[f"belief_{belief}_prob_unknown"] = 0.0
            row[f"belief_{belief}_entropy"] = float(step) / 10
            row[f"belief_{belief}_changed"] = int(step == 2)
        rows.append(row)
    return pd.DataFrame(rows)


def test_belief_changes_use_current_and_previous_rows() -> None:
    result = add_belief_change_summaries(belief_frame())
    assert np.isnan(result.loc[0, "belief_mean_probability_shift"])
    assert np.isclose(result.loc[1, "belief_mean_probability_shift"], 0.2)
    assert np.isclose(result.loc[2, "belief_max_probability_shift"], 0.3)
    assert np.isclose(result.loc[1, "belief_mean_absolute_entropy_shift"], 0.1)
    assert result.loc[2, "belief_answer_changes"] == 6


def point_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positions = pd.DataFrame(
        {
            "example_id": ["a"] * 5,
            "position_index": range(5),
            "row_id": [f"a:{index}" for index in range(5)],
        }
    )
    detected = pd.DataFrame({"example_id": ["a", "a"], "position_index": [2, 4]})
    stable = pd.DataFrame(
        {
            "example_id": ["a", "a"],
            "position_index": [2, 4],
            "found_by_at_least_12_of_16_setups": [True, False],
        }
    )
    return positions, detected, stable


def test_future_point_targets_are_strictly_after_current_position() -> None:
    positions, detected, stable = point_tables()
    result = add_future_point_targets(positions, detected, stable, horizons=(1, 3))
    assert result.loc[1, "change_point_in_next_1"] == 1
    assert result.loc[2, "change_point_in_next_1"] == 0
    assert result.loc[1, "robust_change_point_in_next_1"] == 1
    assert result.loc[3, "robust_change_point_in_next_1"] == 0
    assert result.loc[0, "change_point_in_next_3"] == 1
    assert np.isnan(result.loc[2, "change_point_in_next_3"])


def test_attach_frozen_points_preserves_probability_and_flags() -> None:
    positions, detected, stable = point_tables()
    probabilities = positions[["example_id", "position_index"]].copy()
    probabilities["posterior_change_probability"] = [0, 0.1, 0.9, 0.2, 0.8]
    result = attach_frozen_change_points(positions, probabilities, detected, stable)
    assert result.loc[2, "offline_beast_change_probability"] == 0.9
    assert result["offline_beast_change_point_here"].tolist() == [0, 0, 1, 0, 1]
    assert result["offline_robust_change_point_here"].tolist() == [0, 0, 1, 0, 0]


def test_commitment_targets_exclude_current_onset() -> None:
    frame = pd.DataFrame(
        {
            "example_id": ["a"] * 5,
            "position_index": range(5),
            "action_committed": [False, False, True, True, True],
            "commitment_onset": [False, False, True, False, False],
        }
    )
    result = add_future_commitment_targets(frame, horizons=(1, 3))
    assert result.loc[1, "commitment_onset_h1"] == 1
    assert result.loc[2, "commitment_onset_h1"] == 0
    assert result.loc[0, "commitment_onset_h3"] == 1
    assert result["currently_uncommitted"].tolist() == [1, 1, 0, 0, 0]


def test_paired_comparison_uses_same_rows() -> None:
    predictions = pd.DataFrame(
        {
            "row_id": ["a:0", "a:0", "a:1", "a:1"],
            "example_id": ["a"] * 4,
            "trajectory_id": ["g"] * 4,
            "target": ["y"] * 4,
            "outcome": [0, 0, 1, 1],
            "model": ["candidate", "reference", "candidate", "reference"],
            "predicted_probability": [0.2, 0.3, 0.8, 0.7],
        }
    )
    paired = paired_model_comparison(
        predictions, candidate="candidate", reference="reference"
    )
    assert len(paired) == 2
    assert paired["candidate_probability"].tolist() == [0.2, 0.8]
