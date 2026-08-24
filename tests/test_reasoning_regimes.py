import pandas as pd

from reveng.experiments.reasoning_regimes import (
    assign_regimes,
    recurrence_returns,
    summarize_trace,
)


def frame(actions, optimal):
    rows = []
    for index, (action, is_optimal) in enumerate(zip(actions, optimal, strict=True)):
        rows.append(
            {
                "position_index": index,
                "reasoning_progress": index / max(len(actions) - 1, 1),
                "argmax_action": action,
                "action_is_optimal": is_optimal,
                "prob_up": 0.7 if action == "UP" else 0.1,
                "prob_down": 0.7 if action == "DOWN" else 0.1,
                "prob_left": 0.1,
                "prob_right": 0.1,
            }
        )
    return pd.DataFrame(rows)


def test_regimes_are_exclusive_and_stable_suffix_is_retrospective():
    labelled = assign_regimes(
        frame(["UP", "DOWN", "UP", "UP"], [True, False, True, True])
    )
    assert labelled.regime.tolist() == [
        "unsettled_optimal",
        "unsettled_suboptimal",
        "stable_optimal",
        "stable_optimal",
    ]
    assert labelled.stable_action_position.unique().tolist() == [2]


def test_recurrence_requires_leaving_and_returning():
    assert recurrence_returns(["a", "a", "b", "b", "a"]) == 1
    assert recurrence_returns(["a", "a", "a"]) == 0


def test_trace_summary_counts_optimality_loss_and_recovery():
    labelled = assign_regimes(
        frame(["UP", "DOWN", "UP", "UP"], [True, False, True, True])
    )
    summary = summarize_trace(labelled)
    assert summary["optimality_losses"] == 1
    assert summary["optimality_gains"] == 1
    assert summary["optimality_recovered"] is True
    assert summary["recurrence_returns"] == 0
