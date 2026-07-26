import numpy as np
import pandas as pd

from reveng.experiments.grouped_event_prediction import (
    binary_auc,
    circular_shift_within_states,
    grouped_logistic_predictions,
)


def test_binary_auc_has_expected_direction() -> None:
    labels = np.asarray([0, 0, 1, 1])

    assert binary_auc(labels, np.asarray([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert binary_auc(labels, np.asarray([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_grouped_prediction_holds_out_complete_groups() -> None:
    rows = []
    for group_index in range(10):
        for position in range(8):
            rows.append(
                {
                    "example_id": f"state_{group_index}",
                    "trajectory_id": f"trajectory_{group_index}",
                    "feature": float(position),
                    "event": int(position >= 6),
                }
            )
    frame = pd.DataFrame(rows)

    predictions = grouped_logistic_predictions(
        frame,
        feature_columns=("feature",),
        target_column="event",
    )

    assert len(predictions) == len(frame)
    assert predictions.groupby("trajectory_id")["fold"].nunique().max() == 1
    assert binary_auc(
        predictions["event"].to_numpy(),
        predictions["predicted_probability"].to_numpy(),
    ) > 0.95


def test_circular_shift_preserves_values_within_each_state() -> None:
    frame = pd.DataFrame(
        {
            "example_id": ["a", "a", "a", "b", "b"],
            "value": [1, 2, 3, 10, 20],
        }
    )
    shifted = circular_shift_within_states(
        frame,
        columns=("value",),
        rng=np.random.default_rng(42),
    )

    for state in ("a", "b"):
        original = sorted(frame[frame["example_id"] == state]["value"])
        permuted = sorted(shifted[shifted["example_id"] == state]["value"])
        assert original == permuted
