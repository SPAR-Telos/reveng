import json

import numpy as np
import pandas as pd

from reveng.experiments.beast_action_change_points import (
    BeastConfig,
    distance_from_initial,
    distribution_distance_from_initial,
    fit_action_distribution_states,
    fit_beast_series,
)


def test_distance_from_initial_uses_l2_distance() -> None:
    probabilities = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    assert np.allclose(distance_from_initial(probabilities), [0.0, np.sqrt(2)])


def test_distribution_distance_metrics_have_expected_simplex_extremes() -> None:
    probabilities = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    expected = {
        "l2": np.sqrt(2),
        "total_variation": 1.0,
        "hellinger": 1.0,
        "jensen_shannon": 1.0,
    }
    for metric, endpoint in expected.items():
        distances = distribution_distance_from_initial(
            probabilities,
            metric=metric,
        )
        assert np.allclose(distances, [0.0, endpoint])


def test_distribution_distance_rejects_invalid_probabilities() -> None:
    probabilities = np.asarray(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.4, 0.0, 0.0],
        ]
    )
    with np.testing.assert_raises(ValueError):
        distribution_distance_from_initial(probabilities, metric="l2")


def test_constant_series_is_rejected_before_beast_normalization() -> None:
    result = fit_beast_series(np.zeros(20), BeastConfig(samples=500))
    assert result["status"] == "constant_series"
    assert result["probability_zero_change_points"] == 1.0
    assert not result["trace_has_change_point_evidence"]
    assert result["modes"] == []


def test_beast_detects_single_abrupt_change() -> None:
    result = fit_beast_series(
        np.r_[np.zeros(20), np.ones(20)],
        BeastConfig(samples=1000, burnin=100, chains=2),
    )
    detected = [mode for mode in result["modes"] if mode["detected_change_point"]]
    assert result["trace_has_change_point_evidence"]
    assert len(detected) == 1
    assert detected[0]["position_index"] == 20
    assert detected[0]["posterior_change_probability_at_position"] > 0.99


def test_state_fit_preserves_zero_based_position_mapping() -> None:
    distributions = [
        {"UP": 1.0, "DOWN": 0.0, "LEFT": 0.0, "RIGHT": 0.0}
        for _ in range(10)
    ] + [
        {"UP": 0.0, "DOWN": 1.0, "LEFT": 0.0, "RIGHT": 0.0}
        for _ in range(10)
    ]
    rows = pd.DataFrame(
        [
            {
                "example_id": "state",
                "trajectory_id": "trajectory",
                "step_index": 2,
                "position_index": index,
                "reasoning_step_idx": index,
                "reasoning_progress": index / 19,
                "action_probabilities_json": json.dumps(distribution),
                "action_label": "UP" if index < 10 else "DOWN",
                "action_is_optimal": index < 10,
                "matched_role": "failure",
            }
            for index, distribution in enumerate(distributions)
        ]
    )
    positions, states, modes = fit_action_distribution_states(
        rows,
        BeastConfig(samples=1000, burnin=100, chains=2),
    )
    detected = modes[modes["detected_change_point"]]
    assert len(positions) == 20
    assert states.iloc[0]["n_detected_change_points"] == 1
    assert detected.iloc[0]["position_index"] == 10
