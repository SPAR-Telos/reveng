from reveng.experiments.belief_transition_models import (
    _auc,
    _feature_names,
    _group_folds,
    _grouped_bootstrap_comparison,
    _grouped_bootstrap_intervals,
    _metrics,
)


def test_belief_action_chain_has_action_conditioned_interactions() -> None:
    global_beliefs = _feature_names("global_beliefs")
    belief_dynamics = _feature_names("belief_dynamics")
    action_conditioned = _feature_names("action_conditioned")
    combined = _feature_names("combined")
    chain = _feature_names("belief_action_chain")

    assert "error_wall_left" in global_beliefs
    assert "error_door_open" not in global_beliefs
    assert "n_global_error_onsets" in belief_dynamics
    assert "chosen_wall_reports_blocked" in action_conditioned
    assert "chosen_wall_error" in combined
    assert "chosen_wall_error" in chain
    assert "chosen_effect_error" in chain
    assert "blocked_report*predicted_hit" in chain
    assert len(chain) > len(global_beliefs)


def test_transition_metrics_report_perfect_ranking() -> None:
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.8, 0.9]

    assert _auc(labels, probabilities) == 1.0
    assert _metrics(labels, probabilities)["brier_score"] < 0.05


def test_group_folds_keep_validation_groups_together() -> None:
    rows = [
        {"validation_group": "pair-a", "outcome": 0},
        {"validation_group": "pair-a", "outcome": 1},
        {"validation_group": "trajectory-b", "outcome": 0},
        {"validation_group": "trajectory-c", "outcome": 1},
    ]

    folds = _group_folds(rows, n_folds=2, seed=42)

    assert sum("pair-a" in fold for fold in folds) == 1


def test_grouped_bootstrap_intervals_are_reproducible() -> None:
    rows = [
        {"validation_group": "a", "outcome": 0, "probability": 0.1},
        {"validation_group": "a", "outcome": 1, "probability": 0.9},
        {"validation_group": "b", "outcome": 0, "probability": 0.2},
        {"validation_group": "b", "outcome": 1, "probability": 0.8},
    ]

    first = _grouped_bootstrap_intervals(rows, n_bootstrap=20, seed=7)
    second = _grouped_bootstrap_intervals(rows, n_bootstrap=20, seed=7)

    assert first == second
    assert first["roc_auc_ci_low"] == 1.0


def test_grouped_bootstrap_comparison_pairs_positions() -> None:
    model = [
        {"example_id": "a", "reasoning_step_idx": 1, "validation_group": "g1", "outcome": 0, "probability": 0.1},
        {"example_id": "b", "reasoning_step_idx": 1, "validation_group": "g2", "outcome": 1, "probability": 0.9},
    ]
    reference = [
        {"example_id": "a", "reasoning_step_idx": 1, "validation_group": "g1", "outcome": 0, "probability": 0.5},
        {"example_id": "b", "reasoning_step_idx": 1, "validation_group": "g2", "outcome": 1, "probability": 0.5},
    ]

    result = _grouped_bootstrap_comparison(model, reference, n_bootstrap=20, seed=3)

    assert result["n_paired_rows"] == 2
    assert result["brier_score_difference_ci_high"] < 0
