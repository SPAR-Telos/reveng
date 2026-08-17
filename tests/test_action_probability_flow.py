import json

import numpy as np
import pandas as pd
import pytest

from reveng.experiments.action_probability_flow import (
    add_probability_flow_columns,
    build_window_changes,
    clustered_paired_event_summary,
    match_change_points_to_controls,
    matched_role_event_summary,
    paired_bootstrap_summary,
    parse_optimal_actions,
    summarize_states,
    threshold_sensitivity,
)


def make_positions() -> pd.DataFrame:
    rows = []
    distributions = {
        "failure": [
            [0.30, 0.20, 0.30, 0.20],
            [0.50, 0.10, 0.20, 0.20],
            [0.20, 0.20, 0.40, 0.20],
            [0.60, 0.10, 0.20, 0.10],
            [0.40, 0.20, 0.20, 0.20],
            [0.70, 0.10, 0.10, 0.10],
            [0.80, 0.05, 0.10, 0.05],
        ],
        "control": [
            [0.30, 0.20, 0.30, 0.20],
            [0.40, 0.20, 0.20, 0.20],
            [0.50, 0.10, 0.20, 0.20],
            [0.60, 0.10, 0.20, 0.10],
            [0.70, 0.10, 0.10, 0.10],
            [0.80, 0.05, 0.10, 0.05],
            [0.90, 0.04, 0.03, 0.03],
        ],
    }
    for role, vectors in distributions.items():
        for position, vector in enumerate(vectors):
            rows.append(
                {
                    "example_id": role,
                    "trajectory_id": f"trajectory_{role}",
                    "matched_pair_id": "pair_1",
                    "matched_role": role,
                    "position_index": position,
                    "reasoning_progress": position / (len(vectors) - 1),
                    "action_is_optimal": int(np.argmax(vector)) == 0,
                    "prob_up": vector[0],
                    "prob_down": vector[1],
                    "prob_left": vector[2],
                    "prob_right": vector[3],
                }
            )
    return pd.DataFrame(rows)


def make_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": ["failure", "control"],
            "optimal_actions_json": [json.dumps(["UP"]), json.dumps(["UP"])],
        }
    )


def test_parse_optimal_actions_validates_input() -> None:
    assert parse_optimal_actions('["RIGHT", "UP"]') == ("RIGHT", "UP")
    with pytest.raises(ValueError, match="non-empty"):
        parse_optimal_actions("[]")
    with pytest.raises(ValueError, match="Invalid"):
        parse_optimal_actions('["WAIT"]')


def test_probability_flow_preserves_signed_direction() -> None:
    flow = add_probability_flow_columns(make_positions(), make_metadata())
    failure = flow[flow["example_id"] == "failure"].sort_values("position_index")

    assert failure["optimal_action_probability"].iloc[0] == pytest.approx(0.30)
    assert failure["change_in_optimal_action_probability"].iloc[1] == pytest.approx(
        0.20
    )
    assert failure["change_in_optimal_action_probability"].iloc[2] == pytest.approx(
        -0.30
    )
    assert failure["probability_flow_direction"].iloc[1] == "toward optimal actions"
    assert failure["probability_flow_direction"].iloc[2] == "away from optimal actions"


def test_state_summary_and_paired_bootstrap_use_complete_pairs() -> None:
    flow = add_probability_flow_columns(make_positions(), make_metadata())
    states = summarize_states(flow)
    summary = paired_bootstrap_summary(
        states,
        ["mean_optimal_action_probability"],
        repeats=100,
        seed=7,
    ).iloc[0]

    assert len(states) == 2
    assert summary["n_matched_pairs"] == 1
    assert summary["failure_minus_control"] < 0
    assert summary["difference_ci_low"] == pytest.approx(
        summary["failure_minus_control"]
    )


def test_threshold_sensitivity_uses_strict_minimum_change() -> None:
    flow = add_probability_flow_columns(make_positions(), make_metadata())
    summary = threshold_sensitivity(
        flow,
        thresholds=(0.0, 0.25),
        repeats=100,
        seed=3,
    )
    away = summary[summary["metric"] == "fraction_away"].set_index("minimum_change")

    assert away.loc[0.0, "failure_mean"] > away.loc[0.25, "failure_mean"]
    assert away.loc[0.25, "control_mean"] == 0


def test_change_point_controls_are_same_state_unique_and_not_near_change_point() -> (
    None
):
    flow = add_probability_flow_columns(make_positions(), make_metadata())
    windows = build_window_changes(flow, window=1)
    change_points = pd.DataFrame(
        {
            "example_id": ["failure"],
            "position_index": [3],
            "reasoning_progress": [0.5],
        }
    )
    pairs = match_change_points_to_controls(windows, change_points, exclusion_radius=1)

    assert pairs["event_id"].nunique() == 1
    assert set(pairs["row_type"]) == {"change point", "matched position"}
    control_position = pairs.loc[
        pairs["row_type"] == "matched position", "position_index"
    ].item()
    assert abs(control_position - 3) > 1


def test_clustered_event_summary_is_deterministic() -> None:
    pairs = pd.DataFrame(
        {
            "event_id": ["a", "a", "b", "b"],
            "trajectory_id": ["t1", "t1", "t2", "t2"],
            "row_type": ["change point", "matched position"] * 2,
            "absolute_window_change": [0.4, 0.1, 0.3, 0.2],
            "signed_window_change": [0.4, 0.1, -0.3, -0.2],
        }
    )
    first = clustered_paired_event_summary(pairs, repeats=100, seed=9)
    second = clustered_paired_event_summary(pairs, repeats=100, seed=9)

    pd.testing.assert_frame_equal(first, second)
    absolute = first.set_index("metric").loc["absolute_window_change"]
    assert absolute["change_point_minus_matched"] == pytest.approx(0.2)


def test_matched_role_event_summary_uses_complete_pairs() -> None:
    pairs = pd.DataFrame(
        {
            "event_id": ["fc", "fc", "cc", "cc", "fm", "fm", "cm", "cm"],
            "matched_pair_id": ["p1"] * 8,
            "matched_role": ["failure", "failure", "control", "control"] * 2,
            "row_type": ["change point"] * 4 + ["matched position"] * 4,
            "absolute_window_change": [0.3, 0.3, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1],
            "signed_window_change": [0.1, 0.1, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0],
        }
    )
    summary = matched_role_event_summary(pairs, repeats=100, seed=4)
    row = summary[
        (summary["row_type"] == "change point")
        & (summary["metric"] == "signed_window_change")
    ].iloc[0]

    assert row["n_complete_matched_pairs"] == 1
    assert row["control_minus_failure"] == pytest.approx(0.3)
    assert row["difference_ci_low"] == pytest.approx(0.3)
