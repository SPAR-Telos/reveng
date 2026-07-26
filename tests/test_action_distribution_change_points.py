import json

import numpy as np
import pandas as pd
import pytest

from reveng.experiments.action_distribution_change_points import (
    build_event_aligned_rows,
    compute_boundary_offsets,
    compute_position_metrics,
    jensen_shannon_bits,
    parse_action_distribution,
    random_position_alignment_null,
    select_change_points,
    total_variation,
    trajectory_bootstrap_ci,
)


def position_rows(
    distributions: list[dict[str, float]],
    *,
    actions: list[str],
    commitment_position: int,
) -> pd.DataFrame:
    final_action = actions[-1]
    return pd.DataFrame(
        [
            {
                "example_id": "state",
                "trajectory_id": "trajectory",
                "step_index": 2,
                "matched_pair_id": "pair",
                "matched_role": "failure",
                "trajectory_class": "suboptimal_success",
                "primary_step_failure_mode": "wall_hit",
                "position_index": index,
                "reasoning_step_idx": index,
                "reasoning_progress": index / max(1, len(distributions) - 1),
                "action_probabilities_json": json.dumps(distribution),
                "action_label": actions[index],
                "action_is_optimal": actions[index] == "UP",
                "final_full_trace_action": final_action,
                "commitment_onset": index == commitment_position,
            }
            for index, distribution in enumerate(distributions)
        ]
    )


def test_divergence_and_total_variation_have_expected_extremes() -> None:
    up = [1.0, 0.0, 0.0, 0.0]
    down = [0.0, 1.0, 0.0, 0.0]

    assert jensen_shannon_bits(up, up) == pytest.approx(0.0)
    assert jensen_shannon_bits(up, down) == pytest.approx(1.0)
    assert total_variation(up, up) == pytest.approx(0.0)
    assert total_variation(up, down) == pytest.approx(1.0)


def test_parse_rejects_missing_negative_and_non_normalized_values() -> None:
    with pytest.raises(ValueError, match="keys differ"):
        parse_action_distribution({"UP": 1.0})
    with pytest.raises(ValueError, match="negative"):
        parse_action_distribution(
            {"UP": 1.1, "DOWN": -0.1, "LEFT": 0.0, "RIGHT": 0.0}
        )
    with pytest.raises(ValueError, match="sums"):
        parse_action_distribution(
            {"UP": 0.4, "DOWN": 0.4, "LEFT": 0.4, "RIGHT": 0.0}
        )


def test_change_points_find_jump_divergence_and_stable_suffix() -> None:
    distributions = [
        {"UP": 0.10, "DOWN": 0.05, "LEFT": 0.80, "RIGHT": 0.05},
        {"UP": 0.20, "DOWN": 0.05, "LEFT": 0.70, "RIGHT": 0.05},
        {"UP": 0.90, "DOWN": 0.03, "LEFT": 0.04, "RIGHT": 0.03},
        {"UP": 0.80, "DOWN": 0.05, "LEFT": 0.10, "RIGHT": 0.05},
    ]
    source = position_rows(
        distributions,
        actions=["LEFT", "LEFT", "UP", "UP"],
        commitment_position=2,
    )

    metrics = compute_position_metrics(source)
    points = select_change_points(metrics)
    offsets = compute_boundary_offsets(points).iloc[0]

    by_type = points.set_index("boundary_type")
    assert by_type.loc["final_action_jump", "boundary_position"] == 2
    assert by_type.loc["largest_distribution_change", "boundary_position"] == 2
    assert by_type.loc["stable_action_boundary", "boundary_position"] == 2
    assert offsets["jump_minus_stable_sentences"] == 0
    assert offsets["stable_boundary_reproduced"]


def test_tied_maximum_uses_earliest_position() -> None:
    distributions = [
        {"UP": 0.10, "DOWN": 0.10, "LEFT": 0.70, "RIGHT": 0.10},
        {"UP": 0.30, "DOWN": 0.10, "LEFT": 0.50, "RIGHT": 0.10},
        {"UP": 0.50, "DOWN": 0.10, "LEFT": 0.30, "RIGHT": 0.10},
        {"UP": 0.50, "DOWN": 0.10, "LEFT": 0.30, "RIGHT": 0.10},
    ]
    source = position_rows(
        distributions,
        actions=["LEFT", "LEFT", "UP", "UP"],
        commitment_position=2,
    )

    points = select_change_points(compute_position_metrics(source)).set_index(
        "boundary_type"
    )

    assert points.loc["final_action_jump", "boundary_position"] == 1


def test_event_alignment_and_bootstrap_are_deterministic() -> None:
    distributions = [
        {"UP": 0.10, "DOWN": 0.10, "LEFT": 0.70, "RIGHT": 0.10},
        {"UP": 0.80, "DOWN": 0.05, "LEFT": 0.10, "RIGHT": 0.05},
        {"UP": 0.85, "DOWN": 0.05, "LEFT": 0.05, "RIGHT": 0.05},
    ]
    source = position_rows(
        distributions,
        actions=["LEFT", "UP", "UP"],
        commitment_position=1,
    )
    metrics = compute_position_metrics(source)
    points = select_change_points(metrics)
    aligned = build_event_aligned_rows(metrics, points, window=1)

    jump = aligned[aligned["boundary_type"] == "final_action_jump"]
    assert set(jump["relative_sentence_offset"]) == {-1, 0, 1}

    values = pd.DataFrame(
        {
            "trajectory_id": ["a", "a", "b"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    first = trajectory_bootstrap_ci(
        values, value_column="value", repeats=100, seed=42
    )
    second = trajectory_bootstrap_ci(
        values, value_column="value", repeats=100, seed=42
    )
    assert np.allclose(first, second)


def test_random_position_alignment_null_is_deterministic() -> None:
    alignments = pd.DataFrame(
        [
            {
                "example_id": "a",
                "matched_role": "failure",
                "n_reasoning_sentences": 10,
                "stable_action_position": 8,
                "jump_minus_stable_sentences": -1,
            },
            {
                "example_id": "b",
                "matched_role": "control",
                "n_reasoning_sentences": 20,
                "stable_action_position": 15,
                "jump_minus_stable_sentences": -2,
            },
        ]
    )

    first = random_position_alignment_null(alignments, repeats=100, seed=42)
    second = random_position_alignment_null(alignments, repeats=100, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[first["state_group"] == "all", "n_states"].item() == 2
