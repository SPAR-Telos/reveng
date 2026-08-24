"""Utilities for multi-horizon closure and temporal-process graph tests."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import product

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values in input order."""
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def fractional_path_counts(
    traces: Sequence[Sequence[set[str]]], order: int
) -> Counter[tuple[str, ...]]:
    """Count ordered multilabel paths without inflating multilabel sentences."""
    if order < 1:
        raise ValueError("order must be positive")
    counts: Counter[tuple[str, ...]] = Counter()
    for trace in traces:
        for start in range(len(trace) - order + 1):
            window = trace[start : start + order]
            if any(not labels for labels in window):
                continue
            weight = float(np.prod([1.0 / len(labels) for labels in window]))
            for path in product(*(sorted(labels) for labels in window)):
                counts[path] += weight
    return counts


def path_permutation_test(
    traces: Sequence[Sequence[set[str]]],
    *,
    order: int,
    trace_strata: Sequence[Sequence[object]] | None = None,
    permutations: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare path counts with within-trace order permutations."""
    categories = sorted({label for trace in traces for labels in trace for label in labels})
    category_index = {label: index for index, label in enumerate(categories)}
    matrices = []
    for trace in traces:
        matrix = np.zeros((len(trace), len(categories)), dtype=float)
        for row, labels in enumerate(trace):
            for label in labels:
                matrix[row, category_index[label]] = 1.0 / len(labels)
        matrices.append(matrix)

    def count_paths(values: Sequence[np.ndarray]) -> np.ndarray:
        shape = (len(categories),) * order
        output = np.zeros(shape, dtype=float)
        for matrix in values:
            if len(matrix) < order:
                continue
            if order == 2:
                output += np.einsum("ti,tj->ij", matrix[:-1], matrix[1:])
            elif order == 3:
                output += np.einsum(
                    "ti,tj,tk->ijk", matrix[:-2], matrix[1:-1], matrix[2:]
                )
            else:
                raise ValueError("vectorized permutation test supports orders 2 and 3")
        return output

    observed_array = count_paths(matrices)
    path_indices = np.argwhere(observed_array > 0)
    paths = [tuple(categories[index] for index in indices) for indices in path_indices]
    observed = observed_array[tuple(path_indices.T)]
    null = np.zeros((permutations, len(paths)), dtype=float)
    rng = np.random.default_rng(seed)
    for draw in range(permutations):
        shuffled = []
        for trace_index, matrix in enumerate(matrices):
            strata = None if trace_strata is None else trace_strata[trace_index]
            shuffled.append(np.asarray(_shuffle_within_strata(matrix, strata, rng)))
        counts = count_paths(shuffled)
        null[draw] = counts[tuple(path_indices.T)]
    rows = []
    for column, path in enumerate(paths):
        values = null[:, column]
        value = observed[column]
        rows.append(
            {
                "path": " -> ".join(path),
                "order": order,
                "observed_weight": value,
                "null_mean": float(values.mean()),
                "enrichment": float(value - values.mean()),
                "enrichment_ratio": float(value / max(values.mean(), 1e-12)),
                "permutation_p_upper": float(
                    (1 + np.sum(values >= value)) / (permutations + 1)
                ),
            }
        )
    output = pd.DataFrame(rows)
    output["bh_q"] = benjamini_hochberg(output.permutation_p_upper)
    return output.sort_values(
        ["bh_q", "enrichment"], ascending=[True, False]
    ).reset_index(drop=True)


def categorical_transition_information(traces: Sequence[Sequence[str]]) -> float:
    """Empirical mutual information in bits between adjacent symbols."""
    pairs = [(a, b) for trace in traces for a, b in zip(trace[:-1], trace[1:])]
    if not pairs:
        return float("nan")
    pair_counts = Counter(pairs)
    left_counts = Counter(a for a, _ in pairs)
    right_counts = Counter(b for _, b in pairs)
    total = float(len(pairs))
    information = 0.0
    for (left, right), count in pair_counts.items():
        joint = count / total
        information += joint * np.log2(
            joint / ((left_counts[left] / total) * (right_counts[right] / total))
        )
    return float(information)


def transition_information_test(
    traces: Sequence[Sequence[str]],
    *,
    trace_strata: Sequence[Sequence[object]] | None = None,
    permutations: int = 5_000,
    seed: int = 42,
) -> dict[str, float]:
    """Test adjacent-symbol information against within-trace shuffled order."""
    observed = categorical_transition_information(traces)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=float)
    for draw in range(permutations):
        shuffled = []
        for trace_index, trace in enumerate(traces):
            strata = None if trace_strata is None else trace_strata[trace_index]
            shuffled.append(_shuffle_within_strata(trace, strata, rng))
        null[draw] = categorical_transition_information(shuffled)
    return {
        "observed_mutual_information_bits": observed,
        "null_mean_bits": float(null.mean()),
        "excess_information_bits": float(observed - null.mean()),
        "permutation_p_upper": float(
            (1 + np.sum(null >= observed)) / (permutations + 1)
        ),
    }


def _shuffle_within_strata(
    values: Sequence[object],
    strata: Sequence[object] | None,
    rng: np.random.Generator,
) -> list:
    """Shuffle values globally or only among positions sharing a stratum."""
    output = list(values)
    if strata is None:
        indices = rng.permutation(len(values))
        return [values[index] for index in indices]
    if len(strata) != len(values):
        raise ValueError("trace and stratum lengths differ")
    stratum_array = np.asarray(strata, dtype=object)
    for stratum in pd.unique(stratum_array):
        indices = np.flatnonzero(stratum_array == stratum)
        shuffled = rng.permutation(indices)
        for target, source in zip(indices, shuffled, strict=True):
            output[target] = values[source]
    return output


def probability_entropy(probabilities: np.ndarray) -> float:
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    return float(-np.sum(values * np.log2(values)))


def jensen_shannon_bits(left: np.ndarray, right: np.ndarray) -> float:
    left = np.clip(np.asarray(left, dtype=float), 1e-12, 1.0)
    right = np.clip(np.asarray(right, dtype=float), 1e-12, 1.0)
    middle = (left + right) / 2.0
    return float(
        0.5 * np.sum(left * np.log2(left / middle))
        + 0.5 * np.sum(right * np.log2(right / middle))
    )


def operator_features(
    probabilities: np.ndarray, position: int, *, window: int = 3
) -> dict[str, float]:
    """Action-identity-invariant dynamics around one BEAST boundary."""
    values = np.asarray(probabilities, dtype=float)
    if position < 1 or position >= len(values):
        raise ValueError("position must have observations on both sides")
    before = values[max(0, position - window) : position].mean(axis=0)
    after = values[position : min(len(values), position + window)].mean(axis=0)
    return {
        "total_variation": float(0.5 * np.abs(after - before).sum()),
        "jensen_shannon_bits": jensen_shannon_bits(before, after),
        "entropy_delta_bits": probability_entropy(after) - probability_entropy(before),
        "confidence_delta": float(after.max() - before.max()),
        "argmax_changed": float(np.argmax(before) != np.argmax(after)),
    }


def select_operator_clusters(
    features: np.ndarray,
    *,
    cluster_counts: Iterable[int] = range(2, 7),
    minimum_cluster_size: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, pd.DataFrame, int]:
    """Select K by silhouette among solutions satisfying a size floor."""
    values = np.asarray(features, dtype=float)
    scaled = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-12)
    rows = []
    candidates: list[tuple[float, int, np.ndarray]] = []
    for count in cluster_counts:
        labels = KMeans(n_clusters=count, n_init=50, random_state=seed).fit_predict(
            scaled
        )
        sizes = np.bincount(labels, minlength=count)
        score = float(silhouette_score(scaled, labels))
        eligible = bool(sizes.min() >= minimum_cluster_size)
        rows.append(
            {
                "n_clusters": count,
                "silhouette": score,
                "minimum_cluster_size": int(sizes.min()),
                "eligible": eligible,
            }
        )
        if eligible:
            candidates.append((score, count, labels))
    if not candidates:
        raise ValueError("no cluster solution satisfies minimum size")
    _, selected_count, selected_labels = max(candidates, key=lambda item: item[0])
    return selected_labels, pd.DataFrame(rows), selected_count
