import numpy as np

from reveng.experiments.reasoning_dynamics_graph import (
    benjamini_hochberg,
    fractional_path_counts,
    operator_features,
    transition_information_test,
)


def test_fractional_paths_preserve_one_unit_per_sentence_window():
    counts = fractional_path_counts(
        [[{"verification", "correction"}, {"consolidation"}]], order=2
    )
    assert counts[("verification", "consolidation")] == 0.5
    assert counts[("correction", "consolidation")] == 0.5
    assert sum(counts.values()) == 1.0


def test_transition_information_detects_deterministic_alternation():
    result = transition_information_test(
        [["a", "b"] * 20, ["b", "a"] * 20], permutations=200, seed=3
    )
    assert result["excess_information_bits"] > 0.5
    assert result["permutation_p_upper"] < 0.05


def test_operator_features_detect_switch_and_sharpening():
    probabilities = np.array(
        [[0.6, 0.4, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]]
    )
    result = operator_features(probabilities, 2, window=2)
    assert result["argmax_changed"] == 1.0
    assert result["confidence_delta"] > 0
    assert result["total_variation"] > 0.5


def test_bh_adjustment_is_monotone_in_rank():
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.04, 0.04])
