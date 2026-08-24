"""Continuous trajectory metrics for sentence-level reasoning activations.

The metrics in this module deliberately avoid PCA and clustering.  They treat a
reasoning trace as an ordered curve, normalize away its common activation
offset and scale, and measure recurrence, path tortuosity, directional
reversals, and late-stage consolidation.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


METRIC_DIRECTIONS = {
    "nonlocal_recurrence": "higher_failure",
    "path_tortuosity": "higher_failure",
    "directional_reversal_rate": "higher_failure",
    "late_to_early_dispersion_ratio": "higher_failure",
}


def resample_curve(values: np.ndarray, n_points: int = 32) -> np.ndarray:
    """Linearly resample an ordered activation curve by normalized progress."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2:
        raise ValueError("values must have shape (steps, dimensions) with >=2 steps")
    if n_points < 8:
        raise ValueError("n_points must be at least 8")
    source = np.linspace(0.0, 1.0, len(x))
    target = np.linspace(0.0, 1.0, n_points)
    out = np.empty((n_points, x.shape[1]), dtype=np.float64)
    for column in range(x.shape[1]):
        out[:, column] = np.interp(target, source, x[:, column])
    return out


def normalize_curve(values: np.ndarray) -> np.ndarray:
    """Remove a trace's common offset and normalize its RMS state norm."""
    x = np.asarray(values, dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("activation curve has negligible variation")
    return centered / scale


def _mean_dispersion(values: np.ndarray) -> float:
    centered = values - values.mean(axis=0, keepdims=True)
    return float(np.mean(np.sum(centered * centered, axis=1)))


def curve_metrics(
    values: np.ndarray,
    *,
    n_points: int = 32,
    minimum_recurrence_gap: int = 4,
) -> dict[str, float]:
    """Compute length-controlled continuous topology proxies for one trace.

    ``nonlocal_recurrence`` is the mean exponential proximity to the closest
    position at least ``minimum_recurrence_gap`` away. Distances are measured
    after within-trace RMS normalization. It is a recurrence proxy, not a
    claim that a discrete graph cycle exists.
    """
    curve = normalize_curve(resample_curve(values, n_points=n_points))
    index = np.arange(n_points)
    eligible = np.abs(index[:, None] - index[None, :]) >= minimum_recurrence_gap
    pairwise_distance = np.linalg.norm(curve[:, None, :] - curve[None, :, :], axis=2)
    nearest_nonlocal = np.where(eligible, pairwise_distance, np.inf).min(axis=1)
    recurrence_by_position = np.exp(-nearest_nonlocal)

    updates = np.diff(curve, axis=0)
    step_lengths = np.linalg.norm(updates, axis=1)
    direct = float(np.linalg.norm(curve[-1] - curve[0]))
    tortuosity = float(step_lengths.sum() / max(direct, 1e-12))

    adjacent_updates = updates[:-1]
    following_updates = updates[1:]
    update_denom = np.linalg.norm(adjacent_updates, axis=1) * np.linalg.norm(
        following_updates, axis=1
    )
    update_cosine = np.sum(adjacent_updates * following_updates, axis=1) / np.maximum(
        update_denom, 1e-12
    )
    reversal_rate = float(np.mean(update_cosine < 0.0))

    quarter = n_points // 4
    early_dispersion = _mean_dispersion(curve[:quarter])
    late_dispersion = _mean_dispersion(curve[-quarter:])
    consolidation_ratio = float(late_dispersion / max(early_dispersion, 1e-12))

    return {
        "nonlocal_recurrence": float(np.mean(recurrence_by_position)),
        "path_tortuosity": tortuosity,
        "directional_reversal_rate": reversal_rate,
        "late_to_early_dispersion_ratio": consolidation_ratio,
    }


def paired_inference(
    differences: Iterable[float],
    *,
    groups: Iterable[object] | None = None,
    seed: int = 42,
    bootstrap_samples: int = 20_000,
    permutation_samples: int = 100_000,
) -> dict[str, float]:
    """Paired inference while keeping dependent pairs in the same group."""
    delta = np.asarray(list(differences), dtype=np.float64)
    group_array = (
        np.arange(len(delta), dtype=object)
        if groups is None
        else np.asarray(list(groups), dtype=object)
    )
    if len(group_array) != len(delta):
        raise ValueError("groups and differences must have equal length")
    finite = np.isfinite(delta)
    delta = delta[finite]
    group_array = group_array[finite]
    if len(delta) < 2:
        raise ValueError("at least two finite paired differences are required")
    unique_groups = np.unique(group_array)
    clusters = [delta[group_array == group] for group in unique_groups]
    rng = np.random.default_rng(seed)
    observed = float(delta.mean())
    boot_group_indices = rng.integers(
        0, len(clusters), size=(bootstrap_samples, len(clusters))
    )
    boot_means = np.asarray(
        [
            np.concatenate([clusters[index] for index in sample]).mean()
            for sample in boot_group_indices
        ]
    )
    if len(clusters) <= 16:
        patterns = np.arange(2 ** len(clusters), dtype=np.uint32)[:, None]
        bits = (patterns >> np.arange(len(clusters), dtype=np.uint32)) & 1
        group_signs = 2.0 * bits - 1.0
    else:
        group_signs = rng.choice(
            np.array([-1.0, 1.0]), size=(permutation_samples, len(clusters))
        )
    null_means = np.asarray(
        [
            np.concatenate(
                [clusters[index] * signs[index] for index in range(len(clusters))]
            ).mean()
            for signs in group_signs
        ]
    )
    p_value = float(np.mean(np.abs(null_means) >= abs(observed)))
    return {
        "n_pairs": int(len(delta)),
        "n_validation_groups": int(len(clusters)),
        "mean_difference": observed,
        "ci_low": float(np.quantile(boot_means, 0.025)),
        "ci_high": float(np.quantile(boot_means, 0.975)),
        "sign_flip_p_two_sided": p_value,
    }
