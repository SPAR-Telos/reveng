from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reveng.experiments.epistemic_state_pilot import (
    PilotConfig,
    add_next_three_outcomes,
    aggregate_regime,
    classify_belief,
    classify_positions,
    parse_probabilities,
    relevant_question_ids,
    run_pilot,
    trajectory_bootstrap_rate,
)


def test_probability_parsing_normalizes_and_rejects_bad_values() -> None:
    parsed = parse_probabilities('{"yes": 2, "no": 1, "unknown": 1}')
    assert parsed == {"yes": 0.5, "no": 0.25, "unknown": 0.25}
    with pytest.raises(ValueError):
        parse_probabilities({"yes": 1, "no": -1, "unknown": 1})
    with pytest.raises(ValueError):
        parse_probabilities({"yes": 1, "no": 0})


def test_action_to_question_matching() -> None:
    assert relevant_question_ids("LEFT") == {
        "chosen_wall": "wall_left",
        "hit_wall": "hit_wall_after_left",
        "has_key_after": "has_key_after_left",
    }
    with pytest.raises(ValueError):
        relevant_question_ids("pickup")


def test_category_precedence_and_unknown_diagnostic() -> None:
    wrong = classify_belief({"yes": 0.9, "no": 0.05, "unknown": 0.05}, "no", 0.8)
    unresolved = classify_belief({"yes": 0.1, "no": 0.3, "unknown": 0.6}, "no", 0.8)
    correct = classify_belief({"yes": 0.05, "no": 0.9, "unknown": 0.05}, "no", 0.8)
    assert wrong["state"] == "confidently_wrong"
    assert unresolved["state"] == "unresolved"
    assert unresolved["prefers_unknown"] is True
    assert correct["state"] == "confidently_correct"
    assert aggregate_regime([wrong["state"], unresolved["state"], correct["state"]]) == "confidently_wrong"
    assert aggregate_regime([correct["state"], unresolved["state"], correct["state"]]) == "unresolved"
    assert aggregate_regime([correct["state"]] * 3) == "confidently_correct"


def test_next_three_outcomes_require_a_complete_window() -> None:
    positions = pd.DataFrame(
        {
            "example_id": ["e"] * 6,
            "position_index": range(6),
            "action_is_optimal": [True, True, False, False, True, True],
        }
    )
    result = add_next_three_outcomes(positions)
    assert result["has_complete_next3"].tolist() == [True, True, True, False, False, False]
    assert result.loc[0, "next3_optimality_loss"] == 1
    assert result.loc[1, "next3_optimality_loss"] == 1
    assert np.isnan(result.loc[2, "next3_optimality_loss"])
    assert result.loc[2, "next3_recovery"] == 1
    assert np.isnan(result.loc[3, "next3_recovery"])


def test_bootstrap_resamples_trajectories_not_positions() -> None:
    frame = pd.DataFrame(
        {
            "trajectory_id": ["long"] * 10 + ["short"],
            "outcome": [1.0] * 10 + [0.0],
        }
    )
    point, low, high = trajectory_bootstrap_rate(frame, "outcome", repeats=2_000, seed=42)
    assert point == pytest.approx(10 / 11)
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(1.0)


def _synthetic_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    positions: list[dict[str, object]] = []
    beliefs: list[dict[str, object]] = []
    regimes = {
        "correct": {"yes": 0.10, "no": 0.85, "unknown": 0.05},
        "unresolved": {"yes": 0.15, "no": 0.60, "unknown": 0.25},
        "wrong": {"yes": 0.85, "no": 0.10, "unknown": 0.05},
    }
    action_patterns = {
        "correct": [True, True, True, False, True, True],
        "unresolved": [True, False, False, True, True, True],
        "wrong": [True, False, True, True, False, True],
    }
    for trajectory_index, (regime, probabilities) in enumerate(regimes.items()):
        trajectory_id = f"trajectory-{trajectory_index}"
        example_id = f"example-{trajectory_index}"
        for position_index, action_is_optimal in enumerate(action_patterns[regime]):
            positions.append(
                {
                    "example_id": example_id,
                    "trajectory_id": trajectory_id,
                    "position_index": position_index,
                    "action_label": "RIGHT",
                    "action_is_optimal": action_is_optimal,
                }
            )
            for question_id in relevant_question_ids("RIGHT").values():
                beliefs.append(
                    {
                        "example_id": example_id,
                        "position_index": position_index,
                        "question_id": question_id,
                        "probabilities_json": json.dumps(probabilities),
                        "ground_truth_key": "no",
                    }
                )
    return pd.DataFrame(positions), pd.DataFrame(beliefs)


def test_threshold_sensitivity_changes_only_confidence_cutoff() -> None:
    positions, beliefs = _synthetic_tables()
    at_70 = classify_positions(positions, beliefs, 0.70)
    at_90 = classify_positions(positions, beliefs, 0.90)
    assert (at_70["epistemic_regime"] == "confidently_correct").sum() == 6
    assert (at_70["epistemic_regime"] == "confidently_wrong").sum() == 6
    assert (at_90["epistemic_regime"] == "unresolved").sum() == 18
    assert at_70["current_suboptimal"].equals(at_90["current_suboptimal"])


def test_synthetic_integration_writes_expected_outputs(tmp_path: Path) -> None:
    positions, beliefs = _synthetic_tables()
    position_path = tmp_path / "positions.csv"
    belief_path = tmp_path / "beliefs.csv"
    output = tmp_path / "output"
    positions.to_csv(position_path, index=False)
    beliefs.to_csv(belief_path, index=False)

    paths = run_pilot(
        PilotConfig(
            belief_rows_path=belief_path,
            position_rows_path=position_path,
            output_dir=output,
            bootstrap_repeats=100,
            seed=42,
        )
    )
    assert all(path.exists() for path in paths.values())
    assert paths["figure"].name == "epistemic_state_action_outcomes.png"
    assert paths["sensitivity_figure"].name == "threshold_sensitivity.png"
    rows = pd.read_csv(paths["rows"])
    summary = pd.read_csv(paths["summary"])
    sensitivity = pd.read_csv(paths["sensitivity"])
    contrasts = pd.read_csv(paths["contrasts"])
    audit = pd.read_csv(paths["audit"])
    assert len(rows) == 18
    assert set(rows["epistemic_regime"]) == {
        "confidently_correct",
        "unresolved",
        "confidently_wrong",
    }
    assert set(summary["epistemic_regime"]) == set(rows["epistemic_regime"])
    assert set(sensitivity["confidence_threshold"]) == {0.7, 0.8, 0.9}
    assert set(contrasts["outcome"]) == {
        "current_suboptimal",
        "next3_optimality_loss",
        "next3_recovery",
    }
    assert set(audit["audit_scope"]) == {"overall", "chosen_wall", "hit_wall", "has_key_after"}
    assert {"n_missing_belief_rows", "n_invalid_probability_rows"} <= set(audit.columns)
    required_row_columns = {
        "epistemic_regime",
        "chosen_wall_belief_state",
        "hit_wall_belief_state",
        "has_key_after_belief_state",
        "current_suboptimal",
        "next3_optimality_loss",
        "next3_recovery",
    }
    assert required_row_columns <= set(rows.columns)
