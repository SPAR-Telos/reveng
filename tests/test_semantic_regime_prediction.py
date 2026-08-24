import numpy as np
import pandas as pd

from reveng.experiments.semantic_regime_prediction import (
    DESTINATIONS,
    REGIMES,
    add_history_features,
    combine_hurdle_probabilities,
    combine_semantic_runs,
    semantic_feature_table,
)


def test_multilabels_become_simultaneous_binary_features():
    annotations = pd.DataFrame(
        {
            "example_id": ["x"],
            "sentence_number": [1],
            "semantic_labels": ['["verification", "new_inference"]'],
            "explicitly_revises_prior_reasoning": ["no"],
            "evaluates_prior_route_or_claim": ["yes"],
            "repeats_prior_content": ["no"],
            "introduces_new_information_or_plan": ["yes"],
        }
    )
    features = semantic_feature_table(annotations, "original")
    assert features.loc[0, "original__label__verification"] == 1
    assert features.loc[0, "original__label__new_inference"] == 1
    assert features.loc[0, "original__label__route_planning"] == 0
    assert features.loc[0, "original__cue__evaluates_prior_route_or_claim"] == 1


def test_semantic_intersection_and_union():
    original = pd.DataFrame(
        {
            "example_id": ["x"],
            "sentence_number": [1],
            "original__valid": [1.0],
            "original__label__verification": [1.0],
        }
    )
    replicate = pd.DataFrame(
        {
            "example_id": ["x"],
            "sentence_number": [1],
            "replicate__valid": [1.0],
            "replicate__label__verification": [0.0],
        }
    )
    intersection = combine_semantic_runs(original, replicate, "intersection")
    union = combine_semantic_runs(original, replicate, "union")
    assert intersection.loc[0, "semantic__label__verification"] == 0
    assert union.loc[0, "semantic__label__verification"] == 1


def test_history_window_uses_prior_regimes_with_trace_start_padding():
    frame = pd.DataFrame(
        {
            "example_id": ["x", "x", "x"],
            "position_index": [0, 1, 2],
            "current_regime": ["a", "a", "b"],
        }
    )
    output = add_history_features(frame, 3)
    assert output.regime_lag_1.tolist() == ["trace_start", "a", "a"]
    assert output.regime_lag_2.tolist() == ["trace_start", "trace_start", "a"]
    assert output.regime_duration.tolist() == [1.0, 2.0, 1.0]


def test_hurdle_probabilities_form_exact_regime_distribution():
    destination = np.zeros((2, len(DESTINATIONS)))
    destination[0] = [0.5, 0.4, 0.1]
    destination[1] = [0.2, 0.7, 0.1]
    combined = combine_hurdle_probabilities(
        ["unsettled_optimal", "unsettled_suboptimal"],
        np.array([0.2, 0.4]),
        destination,
    )
    assert np.allclose(combined.sum(axis=1), 1.0)
    assert np.isclose(combined[0, REGIMES.index("unsettled_optimal")], 0.8)
    assert np.isclose(combined[0, REGIMES.index("unsettled_suboptimal")], 0.1)
    assert np.isclose(combined[1, REGIMES.index("stable_optimal")], 0.28)
