"""Bayesian change-point detection for sentence-level action distributions."""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import Rbeast as rb

from reveng.experiments.action_distribution_change_points import (
    ACTIONS,
    parse_action_distribution,
)

IMPLEMENTATION_VERSION = "beast_action_cpd_v1_exact_sentence_probability"
SUPPORTED_DISTANCE_METRICS = (
    "l2",
    "total_variation",
    "hellinger",
    "jensen_shannon",
)


@dataclass(frozen=True)
class BeastConfig:
    """Prespecified BEAST settings for the offline action-distribution analysis."""

    max_change_points: int = 10
    min_segment_length: int = 3
    location_probability_threshold: float = 0.70
    trace_bayes_factor_threshold: float = 9.0
    minimum_dynamic_range: float = 1e-8
    seed: int = 42
    burnin: int = 200
    chains: int = 3
    thinning: int = 5
    samples: int = 8000


def distance_from_initial(distributions: np.ndarray) -> np.ndarray:
    """Return the L2 distance between each action distribution and the first."""
    return distribution_distance_from_initial(distributions, metric="l2")


def distribution_distance_from_initial(
    distributions: np.ndarray,
    *,
    metric: str,
) -> np.ndarray:
    """Return a named distance from each action distribution to the first.

    Jensen--Shannon distance is the square root of the divergence measured in bits,
    so all four measures are distances with zero denoting identical distributions.
    """
    matrix = np.asarray(distributions, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(ACTIONS):
        raise ValueError(f"Expected an n-by-{len(ACTIONS)} probability matrix")
    if metric not in SUPPORTED_DISTANCE_METRICS:
        raise ValueError(
            f"Unsupported distance metric {metric!r}; expected one of "
            f"{SUPPORTED_DISTANCE_METRICS}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Action distributions contain non-finite values")
    if np.any(matrix < 0):
        raise ValueError("Action distributions contain negative probabilities")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("Action distributions must sum to one")

    reference = matrix[0]
    if metric == "l2":
        return np.linalg.norm(matrix - reference, axis=1)
    if metric == "total_variation":
        return 0.5 * np.abs(matrix - reference).sum(axis=1)
    if metric == "hellinger":
        return np.sqrt(
            0.5
            * np.square(np.sqrt(matrix) - np.sqrt(reference)).sum(axis=1)
        )

    midpoint = 0.5 * (matrix + reference)
    with np.errstate(divide="ignore", invalid="ignore"):
        left_terms = np.where(
            matrix > 0,
            matrix * np.log2(matrix / midpoint),
            0.0,
        )
        right_terms = np.where(
            reference > 0,
            reference * np.log2(reference / midpoint),
            0.0,
        )
    divergence = 0.5 * (left_terms.sum(axis=1) + right_terms.sum(axis=1))
    return np.sqrt(np.maximum(divergence, 0.0))


def _stable_noise_seed(seed: int, example_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{example_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _finite_array(value: Any, length: int, fill: float = np.nan) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if len(array) >= length:
        return array[:length]
    return np.pad(array, (0, length - len(array)), constant_values=fill)


def _trace_bayes_factor(probability_zero: float) -> float:
    if probability_zero <= 0:
        return float("inf")
    return float((1.0 - probability_zero) / probability_zero)


def fit_beast_series(
    values: np.ndarray,
    config: BeastConfig = BeastConfig(),
) -> dict[str, Any]:
    """Fit trend-only BEAST and return posterior summaries in zero-based positions."""
    series = np.asarray(values, dtype=float).reshape(-1)
    if len(series) < 2 * config.min_segment_length + 1:
        raise ValueError("Series is too short for the configured segment length")
    if not np.all(np.isfinite(series)):
        raise ValueError("Series contains non-finite values")

    dynamic_range = float(np.ptp(series))
    if dynamic_range <= config.minimum_dynamic_range:
        return {
            "status": "constant_series",
            "dynamic_range": dynamic_range,
            "fitted_values": series.copy(),
            "change_probability": np.zeros(len(series), dtype=float),
            "posterior_count_probabilities": np.asarray([1.0]),
            "posterior_mean_count": 0.0,
            "posterior_count_q10": 0,
            "probability_zero_change_points": 1.0,
            "trace_bayes_factor": 0.0,
            "trace_has_change_point_evidence": False,
            "modes": [],
        }

    max_supported = max(
        1,
        (len(series) - 2 * config.min_segment_length)
        // config.min_segment_length,
    )
    max_change_points = min(config.max_change_points, max_supported)
    output = rb.beast(
        series,
        start=1,
        deltat=1,
        season="none",
        tcp_minmax=[0, max_change_points],
        torder_minmax=[1, 1],
        tseg_minlength=config.min_segment_length,
        tseg_leftmargin=config.min_segment_length,
        tseg_rightmargin=config.min_segment_length,
        mcmc_seed=config.seed,
        mcmc_burbin=config.burnin,
        mcmc_chains=config.chains,
        mcmc_thin=config.thinning,
        mcmc_samples=config.samples,
        print_param=False,
        print_progress=False,
        print_warning=False,
        quiet=True,
    )

    count_probabilities = _finite_array(
        output.trend.ncpPr, max_change_points + 1, fill=0.0
    )
    probability_zero = float(count_probabilities[0])
    bayes_factor = _trace_bayes_factor(probability_zero)
    trace_has_evidence = bayes_factor > config.trace_bayes_factor_threshold

    locations = _finite_array(output.trend.cp, max_change_points)
    mode_probabilities = _finite_array(output.trend.cpPr, max_change_points)
    change_probabilities = _finite_array(
        output.trend.cpOccPr, len(series), fill=0.0
    )
    changes = _finite_array(output.trend.cpAbruptChange, max_change_points)
    intervals = np.asarray(output.trend.cpCI, dtype=float).reshape(-1, 2)
    modes: list[dict[str, Any]] = []
    for rank, (location, mode_probability, abrupt_change) in enumerate(
        zip(locations, mode_probabilities, changes, strict=True),
        start=1,
    ):
        if not np.isfinite(location) or not np.isfinite(mode_probability):
            continue
        interval = intervals[rank - 1] if rank - 1 < len(intervals) else [np.nan, np.nan]
        position = int(round(float(location))) - 1
        position_probability = float(change_probabilities[position])
        modes.append(
            {
                "mode_rank": rank,
                "position_index": position,
                "mode_probability": float(mode_probability),
                "posterior_change_probability_at_position": position_probability,
                "abrupt_change_in_distance": float(abrupt_change),
                "position_ci_low": (
                    float(interval[0] - 1) if np.isfinite(interval[0]) else np.nan
                ),
                "position_ci_high": (
                    float(interval[1] - 1) if np.isfinite(interval[1]) else np.nan
                ),
                "passes_location_threshold": (
                    position_probability >= config.location_probability_threshold
                ),
                "detected_change_point": (
                    trace_has_evidence
                    and position_probability
                    >= config.location_probability_threshold
                ),
            }
        )

    return {
        "status": "fitted",
        "dynamic_range": dynamic_range,
        "fitted_values": _finite_array(output.trend.Y, len(series)),
        "change_probability": change_probabilities,
        "posterior_count_probabilities": count_probabilities,
        "posterior_mean_count": float(np.asarray(output.trend.ncp).reshape(-1)[0]),
        "posterior_count_q10": int(
            round(float(np.asarray(output.trend.ncp_pct10).reshape(-1)[0]))
        ),
        "probability_zero_change_points": probability_zero,
        "trace_bayes_factor": bayes_factor,
        "trace_has_change_point_evidence": trace_has_evidence,
        "modes": modes,
    }


def fit_action_distribution_states(
    rows: pd.DataFrame,
    config: BeastConfig = BeastConfig(),
    *,
    distance_metric: str = "l2",
    perturbation_sd_fraction: float = 0.0,
    perturbation_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit BEAST independently to each fixed-state action-distribution series."""
    if distance_metric not in SUPPORTED_DISTANCE_METRICS:
        raise ValueError(
            f"Unsupported distance metric {distance_metric!r}; expected one of "
            f"{SUPPORTED_DISTANCE_METRICS}"
        )
    if perturbation_sd_fraction < 0:
        raise ValueError("perturbation_sd_fraction must be non-negative")
    if perturbation_sd_fraction > 0 and perturbation_seed is None:
        raise ValueError("perturbation_seed is required when perturbing the series")

    required = {
        "example_id",
        "trajectory_id",
        "step_index",
        "position_index",
        "reasoning_step_idx",
        "reasoning_progress",
        "action_probabilities_json",
        "action_label",
        "action_is_optimal",
        "matched_role",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Position rows missing required columns: {missing}")

    position_output: list[dict[str, Any]] = []
    state_output: list[dict[str, Any]] = []
    mode_output: list[dict[str, Any]] = []
    for example_id, group in rows.groupby("example_id", sort=True):
        ordered = group.sort_values("position_index").copy()
        positions = ordered["position_index"].astype(int).to_numpy()
        if not np.array_equal(positions, np.arange(len(positions))):
            raise ValueError(f"{example_id}: position_index must be contiguous from zero")
        distributions = np.vstack(
            [
                parse_action_distribution(value)
                for value in ordered["action_probabilities_json"]
            ]
        )
        raw_l2_distances = distance_from_initial(distributions)
        distances = distribution_distance_from_initial(
            distributions,
            metric=distance_metric,
        )
        beast_input = distances.copy()
        if perturbation_sd_fraction > 0:
            dynamic_range = float(np.ptp(distances))
            noise_sd = perturbation_sd_fraction * dynamic_range
            rng = np.random.default_rng(
                _stable_noise_seed(int(perturbation_seed), str(example_id))
            )
            beast_input = beast_input + rng.normal(0.0, noise_sd, len(beast_input))
        result = fit_beast_series(beast_input, config)
        detected_modes = [
            mode for mode in result["modes"] if mode["detected_change_point"]
        ]
        metadata = {
            "example_id": example_id,
            "trajectory_id": str(ordered.iloc[0]["trajectory_id"]),
            "step_index": int(ordered.iloc[0]["step_index"]),
            "matched_pair_id": ordered.iloc[0].get("matched_pair_id", ""),
            "matched_role": ordered.iloc[0]["matched_role"],
            "trajectory_class": ordered.iloc[0].get("trajectory_class", ""),
            "primary_step_failure_mode": ordered.iloc[0].get(
                "primary_step_failure_mode", ""
            ),
        }
        for row, raw_l2_distance, distance, input_distance, fitted, probability in zip(
            ordered.to_dict("records"),
            raw_l2_distances,
            distances,
            beast_input,
            result["fitted_values"],
            result["change_probability"],
            strict=True,
        ):
            distribution = parse_action_distribution(row["action_probabilities_json"])
            position_output.append(
                {
                    **metadata,
                    "position_index": int(row["position_index"]),
                    "reasoning_step_idx": int(row["reasoning_step_idx"]),
                    "reasoning_progress": float(row["reasoning_progress"]),
                    "action_label": row["action_label"],
                    "action_is_optimal": row["action_is_optimal"],
                    **{
                        f"prob_{action.lower()}": float(value)
                        for action, value in zip(ACTIONS, distribution, strict=True)
                    },
                    "distance_metric": distance_metric,
                    "distance_from_initial_action_distribution": float(distance),
                    "beast_input_distance": float(input_distance),
                    "l2_distance_from_initial_action_distribution": float(
                        raw_l2_distance
                    ),
                    "beast_fitted_distance": float(fitted),
                    "posterior_change_probability": float(probability),
                }
            )

        state_output.append(
            {
                **metadata,
                "n_positions": len(ordered),
                "n_reasoning_sentences": len(ordered) - 1,
                "distance_metric": distance_metric,
                "perturbation_sd_fraction": perturbation_sd_fraction,
                "fit_status": result["status"],
                "distance_dynamic_range": result["dynamic_range"],
                "posterior_mean_change_points": result["posterior_mean_count"],
                "posterior_count_q10": result["posterior_count_q10"],
                "probability_zero_change_points": result[
                    "probability_zero_change_points"
                ],
                "trace_bayes_factor": result["trace_bayes_factor"],
                "trace_has_change_point_evidence": result[
                    "trace_has_change_point_evidence"
                ],
                "n_detected_change_points": len(detected_modes),
                "posterior_count_probabilities_json": json.dumps(
                    result["posterior_count_probabilities"].tolist()
                ),
            }
        )
        for mode in result["modes"]:
            position = int(mode["position_index"])
            if not 0 <= position < len(ordered):
                raise ValueError(f"{example_id}: BEAST returned position {position}")
            source = ordered.iloc[position]
            mode_output.append(
                {
                    **metadata,
                    **mode,
                    "reasoning_step_idx": int(source["reasoning_step_idx"]),
                    "reasoning_progress": float(source["reasoning_progress"]),
                    "action_label": source["action_label"],
                    "action_is_optimal": source["action_is_optimal"],
                }
            )

    return (
        pd.DataFrame(position_output),
        pd.DataFrame(state_output),
        pd.DataFrame(mode_output),
    )
