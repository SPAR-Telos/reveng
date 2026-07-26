from scripts.run_attention_decision_relevance import (
    choose_control_center,
    summarize_event_rows,
    window_indices,
)
from reveng.experiments.attention_event_windows import summarize_attention_rows
import pandas as pd


def test_control_window_is_nonoverlapping_and_excludes_same_event_type() -> None:
    width = 3
    center = 10
    forbidden = {10, 17}

    control = choose_control_center(center, sentence_count=30, width=width, forbidden_centers=forbidden)

    assert control is not None
    assert set(window_indices(center, 30, width)).isdisjoint(window_indices(control, 30, width))
    assert all(abs(control - event_center) > width for event_center in forbidden)


def test_event_summary_weights_trajectories_not_event_count() -> None:
    base = {
        "event_type": "action_change",
        "layer": 15,
        "matched_role": "failure",
        "event_attention_mean_across_heads": 1.0,
        "control_attention_mean_across_heads": 0.0,
    }
    rows = [
        {**base, "trace_id": "s1", "trajectory_id": "trajectory_a", "event_minus_control_attention": 1.0},
        {**base, "trace_id": "s2", "trajectory_id": "trajectory_a", "event_minus_control_attention": 1.0},
        {**base, "trace_id": "s3", "trajectory_id": "trajectory_b", "event_minus_control_attention": -1.0},
    ]

    all_group = next(
        row
        for row in summarize_event_rows(rows)
        if row["matched_role"] == "all"
    )

    assert all_group["events"] == 3
    assert all_group["trajectories"] == 2
    assert all_group["mean_event_minus_control_attention"] == 0.0


def test_reusable_attention_summary_accepts_numpy_comparison_results() -> None:
    rows = pd.DataFrame(
        [
            {
                "boundary_type": "final_action_jump",
                "layer": 15,
                "matched_role": "failure",
                "example_id": "s1",
                "trajectory_id": "t1",
                "event_attention_mean_across_heads": 0.2,
                "control_attention_mean_across_heads": 0.1,
                "event_minus_control_attention": 0.1,
            },
            {
                "boundary_type": "final_action_jump",
                "layer": 15,
                "matched_role": "failure",
                "example_id": "s2",
                "trajectory_id": "t2",
                "event_attention_mean_across_heads": 0.1,
                "control_attention_mean_across_heads": 0.2,
                "event_minus_control_attention": -0.1,
            },
        ]
    )

    summary = summarize_attention_rows(rows)

    assert summary.iloc[0]["fraction_trajectories_positive"] == 0.5
