import numpy as np
import pandas as pd

from reveng.experiments.beast_cpd_robustness import (
    default_robustness_analysis_setups,
    event_proximity_rows,
    generate_synthetic_probability_trace,
    match_detected_locations,
)


def test_default_robustness_grid_changes_one_choice_at_a_time() -> None:
    analysis_setups = default_robustness_analysis_setups()
    assert analysis_setups[0].analysis_setup_id == "original"
    assert len({setup.analysis_setup_id for setup in analysis_setups}) == len(
        analysis_setups
    )
    original = analysis_setups[0].to_record()
    ignored = {"analysis_setup_id", "analysis_setup_family"}
    for analysis_setup in analysis_setups[1:]:
        record = analysis_setup.to_record()
        differences = [
            key
            for key in original
            if key not in ignored and record[key] != original[key]
        ]
        if analysis_setup.analysis_setup_family == "input_noise_check":
            assert set(differences) == {
                "perturbation_sd_fraction",
                "perturbation_seed",
            }
        else:
            assert len(differences) == 1


def test_location_matching_is_one_to_one_and_tolerance_limited() -> None:
    reference = pd.DataFrame(
        {
            "example_id": ["a", "a", "b"],
            "position_index": [10, 20, 5],
        }
    )
    candidate = pd.DataFrame(
        {
            "example_id": ["a", "a", "a", "b"],
            "position_index": [9, 12, 22, 12],
        }
    )
    matched, summary = match_detected_locations(
        reference,
        candidate,
        tolerance=3,
    )
    assert len(matched) == 2
    assert summary["reference_recovery"] == 2 / 3
    assert summary["candidate_precision"] == 0.5
    assert set(matched["absolute_location_difference"]) == {1, 2}


def test_progress_matched_event_expectation_is_exact() -> None:
    positions = pd.DataFrame(
        {
            "example_id": ["a"] * 10,
            "position_index": np.arange(10),
            "reasoning_progress": np.arange(10) / 10,
        }
    )
    detected = pd.DataFrame(
        {
            "example_id": ["a"],
            "trajectory_id": ["t"],
            "position_index": [5],
            "reasoning_progress": [0.5],
        }
    )
    events = pd.DataFrame(
        {
            "example_id": ["a"],
            "event_type": ["action_change"],
            "position_index": [6],
        }
    )
    rows = event_proximity_rows(
        positions,
        detected,
        events,
        event_types=["action_change"],
        window=1,
    )
    assert rows.iloc[0]["event_within_window"] == 1.0
    assert np.isclose(
        rows.iloc[0]["progress_matched_expected_probability"],
        2 / 9,
    )


def test_synthetic_level_shift_has_prespecified_magnitude() -> None:
    distributions, expected = generate_synthetic_probability_trace(
        "one_level_shift",
        length=100,
        seed=3,
        concentration=1e9,
    )
    assert expected == [50]
    mean_before = distributions[:50].mean(axis=0)
    mean_after = distributions[50:].mean(axis=0)
    assert np.isclose(
        np.linalg.norm(mean_after - mean_before),
        0.18 * np.sqrt(2),
        atol=1e-4,
    )


def test_synthetic_two_shift_trace_changes_distance_twice() -> None:
    distributions, expected = generate_synthetic_probability_trace(
        "two_level_shifts",
        length=90,
        seed=5,
        concentration=1e9,
    )
    assert expected == [30, 60]
    distances = np.linalg.norm(distributions - distributions[0], axis=1)
    assert distances[:30].mean() < 0.001
    assert 0.24 < distances[30:60].mean() < 0.27
    assert 0.49 < distances[60:].mean() < 0.52
