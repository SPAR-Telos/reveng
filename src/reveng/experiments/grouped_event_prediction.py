"""Small grouped logistic models for sentence-level event prediction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = int(labels.sum())
    negative = int(len(labels) - positive)
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float(
        (ranks[labels == 1].sum() - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "auroc": binary_auc(labels, probabilities),
        "log_loss": float(
            -np.mean(
                labels * np.log(probabilities)
                + (1 - labels) * np.log(1 - probabilities)
            )
        ),
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
    }


def grouped_folds(
    frame: pd.DataFrame,
    *,
    group_column: str,
    target_column: str,
    n_folds: int = 5,
    seed: int = 42,
) -> list[set[str]]:
    grouped = frame.groupby(group_column)[target_column].agg(["sum", "size"])
    group_ids = grouped.index.astype(str).tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(group_ids)
    group_ids.sort(
        key=lambda group_id: (
            -int(grouped.loc[group_id, "sum"]),
            -int(grouped.loc[group_id, "size"]),
        )
    )
    folds = [set() for _ in range(min(n_folds, len(group_ids)))]
    fold_events = [0 for _ in folds]
    fold_rows = [0 for _ in folds]
    for group_id in group_ids:
        fold_index = min(
            range(len(folds)),
            key=lambda index: (fold_events[index], fold_rows[index]),
        )
        folds[fold_index].add(group_id)
        fold_events[fold_index] += int(grouped.loc[group_id, "sum"])
        fold_rows[fold_index] += int(grouped.loc[group_id, "size"])
    return folds


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2_penalty: float,
    max_iterations: int = 50,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(features)), features])
    coefficients = np.zeros(design.shape[1], dtype=float)
    penalty = np.eye(design.shape[1], dtype=float) * l2_penalty
    penalty[0, 0] = 0.0
    for _ in range(max_iterations):
        logits = np.clip(design @ coefficients, -30, 30)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probabilities - labels) + penalty @ coefficients
        weights = np.clip(probabilities * (1 - probabilities), 1e-6, None)
        hessian = design.T @ (design * weights[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return coefficients


def grouped_logistic_predictions(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    target_column: str,
    group_column: str = "trajectory_id",
    n_folds: int = 5,
    l2_penalty: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    required = {target_column, group_column, *feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction frame missing columns: {missing}")
    source = frame.dropna(subset=list(required)).reset_index(drop=True)
    if source[target_column].nunique() != 2:
        raise ValueError(f"{target_column} must contain two classes")
    output: list[dict[str, Any]] = []
    folds = grouped_folds(
        source,
        group_column=group_column,
        target_column=target_column,
        n_folds=n_folds,
        seed=seed,
    )
    for fold_index, test_groups in enumerate(folds):
        is_test = source[group_column].astype(str).isin(test_groups).to_numpy()
        train = source[~is_test]
        test = source[is_test]
        if train.empty or test.empty or train[target_column].nunique() != 2:
            continue
        train_x = train[list(feature_columns)].to_numpy(dtype=float)
        test_x = test[list(feature_columns)].to_numpy(dtype=float)
        mean = train_x.mean(axis=0)
        standard_deviation = train_x.std(axis=0)
        standard_deviation[standard_deviation < 1e-6] = 1.0
        train_x = (train_x - mean) / standard_deviation
        test_x = (test_x - mean) / standard_deviation
        coefficients = _fit_logistic(
            train_x,
            train[target_column].to_numpy(dtype=int),
            l2_penalty=l2_penalty,
        )
        logits = np.clip(
            np.column_stack([np.ones(len(test_x)), test_x]) @ coefficients,
            -30,
            30,
        )
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        for source_row, probability in zip(
            test.to_dict("records"), probabilities, strict=True
        ):
            output.append(
                {
                    **source_row,
                    "fold": fold_index,
                    "predicted_probability": float(probability),
                }
            )
    return pd.DataFrame(output)


def circular_shift_within_states(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    state_column: str = "example_id",
    rng: np.random.Generator,
) -> pd.DataFrame:
    shifted = frame.copy()
    for _, indices in shifted.groupby(state_column).groups.items():
        index = np.asarray(list(indices), dtype=int)
        if len(index) < 2:
            continue
        offset = int(rng.integers(1, len(index)))
        for column in columns:
            shifted.loc[index, column] = np.roll(
                shifted.loc[index, column].to_numpy(), offset
            )
    return shifted


def permutation_p_value(
    frame: pd.DataFrame,
    *,
    baseline_features: tuple[str, ...],
    added_features: tuple[str, ...],
    target_column: str,
    repeats: int = 200,
    seed: int = 42,
) -> tuple[float, float, float]:
    baseline = grouped_logistic_predictions(
        frame,
        feature_columns=baseline_features,
        target_column=target_column,
        seed=seed,
    )
    extended = grouped_logistic_predictions(
        frame,
        feature_columns=baseline_features + added_features,
        target_column=target_column,
        seed=seed,
    )
    baseline_auc = binary_auc(
        baseline[target_column].to_numpy(),
        baseline["predicted_probability"].to_numpy(),
    )
    extended_auc = binary_auc(
        extended[target_column].to_numpy(),
        extended["predicted_probability"].to_numpy(),
    )
    observed = extended_auc - baseline_auc
    rng = np.random.default_rng(seed)
    null_improvements: list[float] = []
    for repeat in range(repeats):
        permuted = circular_shift_within_states(
            frame,
            columns=added_features,
            rng=rng,
        )
        predictions = grouped_logistic_predictions(
            permuted,
            feature_columns=baseline_features + added_features,
            target_column=target_column,
            seed=seed,
        )
        permuted_auc = binary_auc(
            predictions[target_column].to_numpy(),
            predictions["predicted_probability"].to_numpy(),
        )
        if math.isfinite(permuted_auc):
            null_improvements.append(permuted_auc - baseline_auc)
    p_value = float(
        (1 + np.sum(np.asarray(null_improvements) >= observed))
        / (1 + len(null_improvements))
    )
    return observed, p_value, float(np.mean(null_improvements))
