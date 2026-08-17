"""Robustness utilities for action-distribution BEAST change points."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from reveng.experiments.beast_action_change_points import (
    BeastConfig,
    distribution_distance_from_initial,
    fit_action_distribution_states,
    fit_beast_series,
)


@dataclass(frozen=True)
class CPDAnalysisSetup:
    """One complete set of choices for running the BEAST analysis."""

    analysis_setup_id: str
    analysis_setup_family: str
    distance_metric: str = "l2"
    min_segment_length: int = 3
    location_probability_threshold: float = 0.70
    trace_bayes_factor_threshold: float = 9.0
    seed: int = 42
    perturbation_sd_fraction: float = 0.0
    perturbation_seed: int | None = None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def default_robustness_analysis_setups() -> list[CPDAnalysisSetup]:
    """Return the finite set of analysis setups, with the original run first."""
    original_setup = CPDAnalysisSetup("original", "original")
    return [
        original_setup,
        replace(
            original_setup,
            analysis_setup_id="distance_total_variation",
            analysis_setup_family="distance_calculation",
            distance_metric="total_variation",
        ),
        replace(
            original_setup,
            analysis_setup_id="distance_hellinger",
            analysis_setup_family="distance_calculation",
            distance_metric="hellinger",
        ),
        replace(
            original_setup,
            analysis_setup_id="distance_jensen_shannon",
            analysis_setup_family="distance_calculation",
            distance_metric="jensen_shannon",
        ),
        replace(
            original_setup,
            analysis_setup_id="minimum_segment_5",
            analysis_setup_family="minimum_segment_length",
            min_segment_length=5,
        ),
        replace(
            original_setup,
            analysis_setup_id="minimum_segment_10",
            analysis_setup_family="minimum_segment_length",
            min_segment_length=10,
        ),
        replace(
            original_setup,
            analysis_setup_id="location_cutoff_0_50",
            analysis_setup_family="location_cutoff",
            location_probability_threshold=0.50,
        ),
        replace(
            original_setup,
            analysis_setup_id="location_cutoff_0_90",
            analysis_setup_family="location_cutoff",
            location_probability_threshold=0.90,
        ),
        replace(
            original_setup,
            analysis_setup_id="evidence_cutoff_3",
            analysis_setup_family="evidence_cutoff",
            trace_bayes_factor_threshold=3.0,
        ),
        replace(
            original_setup,
            analysis_setup_id="evidence_cutoff_20",
            analysis_setup_family="evidence_cutoff",
            trace_bayes_factor_threshold=20.0,
        ),
        *[
            replace(
                original_setup,
                analysis_setup_id=f"random_seed_{seed}",
                analysis_setup_family="random_seed",
                seed=seed,
            )
            for seed in (7, 19, 73, 101)
        ],
        replace(
            original_setup,
            analysis_setup_id="small_noise_check_1",
            analysis_setup_family="input_noise_check",
            perturbation_sd_fraction=0.03,
            perturbation_seed=1,
        ),
        replace(
            original_setup,
            analysis_setup_id="small_noise_check_2",
            analysis_setup_family="input_noise_check",
            perturbation_sd_fraction=0.03,
            perturbation_seed=2,
        ),
    ]


def beast_config_for_analysis_setup(
    analysis_setup: CPDAnalysisSetup,
    *,
    samples: int,
    burnin: int = 200,
    chains: int = 3,
    thinning: int = 5,
) -> BeastConfig:
    return BeastConfig(
        min_segment_length=analysis_setup.min_segment_length,
        location_probability_threshold=(
            analysis_setup.location_probability_threshold
        ),
        trace_bayes_factor_threshold=(
            analysis_setup.trace_bayes_factor_threshold
        ),
        seed=analysis_setup.seed,
        samples=samples,
        burnin=burnin,
        chains=chains,
        thinning=thinning,
    )


def fit_robustness_analysis_setup(
    rows: pd.DataFrame,
    analysis_setup: CPDAnalysisSetup,
    *,
    samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = beast_config_for_analysis_setup(analysis_setup, samples=samples)
    positions, states, modes = fit_action_distribution_states(
        rows,
        config,
        distance_metric=analysis_setup.distance_metric,
        perturbation_sd_fraction=analysis_setup.perturbation_sd_fraction,
        perturbation_seed=analysis_setup.perturbation_seed,
    )
    metadata = analysis_setup.to_record()
    for frame in (positions, states, modes):
        for column, value in reversed(tuple(metadata.items())):
            if column in frame.columns:
                if not frame[column].eq(value).all():
                    raise ValueError(
                        f"Conflicting {column} values in robustness output"
                    )
                continue
            frame.insert(0, column, value)
    return positions, states, modes


def match_detected_locations(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    tolerance: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """One-to-one match candidate locations to reference locations by state."""
    matched_rows: list[dict[str, Any]] = []
    reference_total = len(reference)
    candidate_total = len(candidate)
    for example_id, reference_group in reference.groupby("example_id", sort=True):
        candidate_group = candidate[candidate["example_id"] == example_id]
        if candidate_group.empty:
            continue
        reference_group = reference_group.sort_values("position_index")
        candidate_group = candidate_group.sort_values("position_index")
        reference_positions = reference_group["position_index"].to_numpy(dtype=int)
        candidate_positions = candidate_group["position_index"].to_numpy(dtype=int)
        costs = np.abs(
            reference_positions[:, np.newaxis]
            - candidate_positions[np.newaxis, :]
        )
        reference_indices, candidate_indices = linear_sum_assignment(costs)
        for reference_index, candidate_index in zip(
            reference_indices,
            candidate_indices,
            strict=True,
        ):
            distance = int(costs[reference_index, candidate_index])
            if distance > tolerance:
                continue
            reference_row = reference_group.iloc[int(reference_index)]
            candidate_row = candidate_group.iloc[int(candidate_index)]
            matched_rows.append(
                {
                    "example_id": example_id,
                    "reference_position": int(reference_row["position_index"]),
                    "candidate_position": int(candidate_row["position_index"]),
                    "absolute_location_difference": distance,
                }
            )
    matched = pd.DataFrame(matched_rows)
    matched_count = len(matched)
    summary: dict[str, float | int] = {
        "reference_count": reference_total,
        "candidate_count": candidate_total,
        "matched_within_tolerance": matched_count,
        "reference_recovery": (
            matched_count / reference_total if reference_total else np.nan
        ),
        "candidate_precision": (
            matched_count / candidate_total if candidate_total else np.nan
        ),
        "median_absolute_location_difference": (
            float(matched["absolute_location_difference"].median())
            if matched_count
            else np.nan
        ),
    }
    return matched, summary


def event_proximity_rows(
    positions: pd.DataFrame,
    detected: pd.DataFrame,
    events: pd.DataFrame,
    *,
    event_types: Iterable[str],
    window: int,
) -> pd.DataFrame:
    """Compare detected locations with their exact progress-matched expectation."""
    event_types = tuple(event_types)
    event_lookup = {
        (str(example_id), str(event_type)): group["position_index"]
        .astype(int)
        .to_numpy()
        for (example_id, event_type), group in events.groupby(
            ["example_id", "event_type"]
        )
    }
    position_lookup = {
        str(example_id): group.sort_values("position_index")
        for example_id, group in positions.groupby("example_id")
    }
    rows: list[dict[str, Any]] = []
    for point in detected.itertuples():
        state = position_lookup[str(point.example_id)]
        progress = float(point.reasoning_progress)
        decile = min(9, int(progress * 10))
        candidates = state[
            state["reasoning_progress"].astype(float).between(
                decile / 10,
                (decile + 1) / 10,
                inclusive="left",
            )
        ]["position_index"].astype(int)
        candidates = candidates[candidates != int(point.position_index)]
        if candidates.empty:
            candidates = state["position_index"].astype(int)
            candidates = candidates[candidates != int(point.position_index)]
        candidate_array = candidates.to_numpy()
        for event_type in event_types:
            event_positions = event_lookup.get(
                (str(point.example_id), event_type),
                np.asarray([], dtype=int),
            )
            nearest = (
                int(np.min(np.abs(event_positions - int(point.position_index))))
                if len(event_positions)
                else np.nan
            )
            if len(event_positions):
                close_by_candidate = np.min(
                    np.abs(
                        candidate_array[:, np.newaxis]
                        - event_positions[np.newaxis, :]
                    ),
                    axis=1,
                ) <= window
                random_probability = float(np.mean(close_by_candidate))
            else:
                random_probability = 0.0
            observed = float(np.isfinite(nearest) and nearest <= window)
            rows.append(
                {
                    "example_id": point.example_id,
                    "trajectory_id": point.trajectory_id,
                    "change_point_position": int(point.position_index),
                    "event_type": event_type,
                    "event_within_window": observed,
                    "progress_matched_expected_probability": random_probability,
                    "observed_minus_expected": observed - random_probability,
                    "nearest_event_distance_sentences": nearest,
                }
            )
    return pd.DataFrame(rows)


def summarize_event_proximity(
    rows: pd.DataFrame,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize proximity effects with a trajectory-level bootstrap interval."""
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for event_type, event_rows in rows.groupby("event_type", sort=True):
        grouped = {
            str(key): group["observed_minus_expected"].to_numpy(dtype=float)
            for key, group in event_rows.groupby("trajectory_id")
        }
        keys = np.asarray(sorted(grouped), dtype=object)
        bootstrap: list[float] = []
        for _ in range(bootstrap_repeats):
            sampled = rng.choice(keys, len(keys), replace=True)
            values = np.concatenate([grouped[key] for key in sampled])
            bootstrap.append(float(np.mean(values)))
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        output.append(
            {
                "event_type": event_type,
                "n_detected_change_points": len(event_rows),
                "observed_fraction": event_rows["event_within_window"].mean(),
                "progress_matched_expected_fraction": event_rows[
                    "progress_matched_expected_probability"
                ].mean(),
                "observed_minus_expected_fraction": event_rows[
                    "observed_minus_expected"
                ].mean(),
                "trajectory_bootstrap_ci_low": float(low),
                "trajectory_bootstrap_ci_high": float(high),
            }
        )
    return pd.DataFrame(output)


def generate_synthetic_probability_trace(
    scenario: str,
    *,
    length: int,
    seed: int,
    concentration: float = 400.0,
) -> tuple[np.ndarray, list[int]]:
    """Generate simplex-valued null and change-point benchmark traces."""
    if length < 21:
        raise ValueError("Synthetic trace length must be at least 21")
    rng = np.random.default_rng(seed)
    start = np.asarray([0.70, 0.10, 0.10, 0.10])
    shifted = np.asarray([0.52, 0.28, 0.10, 0.10])
    shifted_again = np.asarray([0.34, 0.46, 0.10, 0.10])

    expected: list[int] = []
    means: list[np.ndarray]
    if scenario == "constant":
        return np.tile(start, (length, 1)), expected
    if scenario == "stable_noise":
        means = [start] * length
    elif scenario == "smooth_drift":
        means = [
            (1.0 - fraction) * start + fraction * shifted
            for fraction in np.linspace(0.0, 1.0, length)
        ]
    elif scenario == "one_level_shift":
        boundary = length // 2
        expected = [boundary]
        means = [start if index < boundary else shifted for index in range(length)]
    elif scenario == "one_slope_change":
        boundary = length // 2
        expected = [boundary]
        means = []
        for index in range(length):
            if index < boundary:
                means.append(start)
            else:
                fraction = (index - boundary) / max(1, length - boundary - 1)
                means.append((1.0 - fraction) * start + fraction * shifted)
    elif scenario == "two_level_shifts":
        first, second = length // 3, 2 * length // 3
        expected = [first, second]
        means = [
            start if index < first else shifted if index < second else shifted_again
            for index in range(length)
        ]
    else:
        raise ValueError(f"Unknown synthetic scenario: {scenario}")

    distributions = np.vstack(
        [rng.dirichlet(np.maximum(mean * concentration, 1e-6)) for mean in means]
    )
    return distributions, expected


def evaluate_synthetic_trace(
    distributions: np.ndarray,
    expected_positions: list[int],
    analysis_setup: CPDAnalysisSetup,
    *,
    samples: int,
    tolerance: int,
    perturbation_seed: int,
) -> dict[str, Any]:
    distances = distribution_distance_from_initial(
        distributions,
        metric=analysis_setup.distance_metric,
    )
    if analysis_setup.perturbation_sd_fraction:
        rng = np.random.default_rng(perturbation_seed)
        distances = distances + rng.normal(
            0.0,
            analysis_setup.perturbation_sd_fraction * float(np.ptp(distances)),
            len(distances),
        )
    result = fit_beast_series(
        distances,
        beast_config_for_analysis_setup(analysis_setup, samples=samples),
    )
    detected = [
        int(mode["position_index"])
        for mode in result["modes"]
        if mode["detected_change_point"]
    ]
    matched_distances: list[int] = []
    if expected_positions and detected:
        costs = np.abs(
            np.asarray(expected_positions)[:, np.newaxis]
            - np.asarray(detected)[np.newaxis, :]
        )
        expected_indices, detected_indices = linear_sum_assignment(costs)
        matched_distances = [
            int(costs[expected_index, detected_index])
            for expected_index, detected_index in zip(
                expected_indices,
                detected_indices,
                strict=True,
            )
            if costs[expected_index, detected_index] <= tolerance
        ]
    return {
        "n_expected_change_points": len(expected_positions),
        "n_detected_change_points": len(detected),
        "n_matched_change_points": len(matched_distances),
        "trace_false_positive": bool(not expected_positions and detected),
        "trace_detected": bool(detected),
        "change_point_recall": (
            len(matched_distances) / len(expected_positions)
            if expected_positions
            else np.nan
        ),
        "median_localization_error": (
            float(np.median(matched_distances)) if matched_distances else np.nan
        ),
        "detected_positions": detected,
    }
