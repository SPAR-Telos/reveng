from __future__ import annotations

import json

import numpy as np
import pandas as pd

from reveng.experiments.practical_action_event_monitor import (
    ModelSpec,
    build_belief_features,
    build_prospective_targets,
    nested_group_predictions,
    representation_dynamics,
)


def test_prospective_targets_are_future_only_and_nested() -> None:
    positions = pd.DataFrame(
        {
            "example_id": ["state"] * 5,
            "trajectory_id": ["trajectory"] * 5,
            "position_index": range(5),
            "reasoning_step_idx": range(5),
            "action_label": ["UP", "UP", "LEFT", "LEFT", "UP"],
            "action_is_optimal": [True, True, False, False, True],
        }
    )
    result = build_prospective_targets(positions)
    first = result.iloc[0]
    second = result.iloc[1]
    third = result.iloc[2]
    assert first["recommendation_change_h1"] == 0
    assert second["recommendation_change_h1"] == 1
    assert second["optimality_loss_h1"] == 1
    assert third["recovery_h1"] == 0
    assert result.iloc[3]["recovery_h1"] == 1
    assert (
        result["optimality_loss_h1"].fillna(0)
        <= result["recommendation_change_h1"].fillna(0)
    ).all()


def test_representation_dynamics_uses_current_and_preceding_sentences() -> None:
    base = np.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32
    )
    result = representation_dynamics(
        {
            8: base + 0.1,
            15: base,
            23: base - 0.1,
        },
        rolling_window=2,
    )
    assert len(result) == 3
    assert np.isnan(result.iloc[0]["activation_similarity_to_preceding_mean"])
    assert np.isfinite(result.iloc[1]["activation_similarity_to_preceding_mean"])
    assert result.iloc[2]["rolling_representation_dispersion"] > 0
    assert (result["sparse_cross_layer_change"] >= 0).all()


def test_belief_features_separate_observable_and_verified_values() -> None:
    positions = pd.DataFrame(
        {
            "example_id": ["state"],
            "reasoning_step_idx": [0],
            "position_index": [0],
            "action_label": ["UP"],
        }
    )
    rows = []
    for question_id in (
        "wall_left",
        "wall_right",
        "wall_up",
        "wall_down",
        "has_key",
        "door_open",
        "hit_wall_after_up",
        "has_key_after_up",
        "door_open_after_up",
    ):
        rows.append(
            {
                "example_id": "state",
                "reasoning_step_idx": 0,
                "question_id": question_id,
                "answer_key": "yes" if question_id == "wall_up" else "no",
                "belief_is_error": question_id == "wall_left",
                "entropy_bits": 0.5,
                "probabilities_json": json.dumps(
                    {"yes": 0.2, "no": 0.7, "unknown": 0.1}
                ),
            }
        )
    features, observable, verified = build_belief_features(
        positions, pd.DataFrame(rows)
    )
    assert features.iloc[0]["belief_action_conflict"] == 1
    assert "belief_wall_left_entropy" in observable
    assert "belief_wall_left_error" not in observable
    assert "belief_wall_left_error" in verified


def test_grouped_predictions_keep_trajectories_in_one_test_fold() -> None:
    rows = []
    for group_index in range(8):
        for position in range(8):
            rows.append(
                {
                    "row_id": f"{group_index}:{position}",
                    "example_id": f"state-{group_index}",
                    "trajectory_id": f"trajectory-{group_index}",
                    "reasoning_step_idx": position,
                    "feature": float(position),
                    "target": int((position + group_index) % 4 == 0),
                }
            )
    predictions, _ = nested_group_predictions(
        pd.DataFrame(rows),
        spec=ModelSpec("test", ("feature",)),
        target_column="target",
        outer_splits=4,
        seed=42,
    )
    assert len(predictions) == len(rows)
    assert (
        predictions.groupby("trajectory_id")["fold"].nunique().eq(1).all()
    )
