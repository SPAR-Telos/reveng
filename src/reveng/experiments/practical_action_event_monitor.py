"""Prospective grouped prediction utilities for reasoning-time action events."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
PRIMARY_BELIEFS = (
    "wall_left",
    "wall_right",
    "wall_up",
    "wall_down",
    "has_key",
    "door_open",
)
CONSEQUENCE_PREFIXES = (
    "hit_wall_after_",
    "has_key_after_",
    "door_open_after_",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    feature_columns: tuple[str, ...]
    aggregate_standardized_features: bool = False


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Cannot parse Boolean value {value!r}")


def parse_probabilities(value: str | dict[str, float]) -> dict[str, float]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError("Categorical probabilities must be a JSON object")
    return {str(key): float(probability) for key, probability in parsed.items()}


def build_prospective_targets(
    positions: pd.DataFrame,
    *,
    horizons: Sequence[int] = (1, 3),
) -> pd.DataFrame:
    """Label events strictly after each current reasoning position."""
    required = {
        "example_id",
        "trajectory_id",
        "position_index",
        "reasoning_step_idx",
        "action_label",
        "action_is_optimal",
    }
    missing = sorted(required - set(positions.columns))
    if missing:
        raise ValueError(f"Position rows missing required columns: {missing}")

    output: list[dict[str, Any]] = []
    for example_id, group in positions.groupby("example_id", sort=True):
        ordered = group.sort_values("position_index").reset_index(drop=True)
        expected = np.arange(len(ordered))
        actual = ordered["position_index"].astype(int).to_numpy()
        if not np.array_equal(actual, expected):
            raise ValueError(f"{example_id}: position_index is not contiguous")
        actions = ordered["action_label"].astype(str).str.upper().tolist()
        optimal = [as_bool(value) for value in ordered["action_is_optimal"]]
        for index, row in ordered.iterrows():
            record = row.to_dict()
            record["row_id"] = f"{example_id}:{int(row['position_index'])}"
            record["current_action_is_optimal"] = int(optimal[index])
            record["has_sentence_representation"] = int(
                int(row["reasoning_step_idx"]) > 0
            )
            for horizon in horizons:
                complete = index + horizon < len(ordered)
                record[f"complete_horizon_{horizon}"] = int(complete)
                if not complete:
                    record[f"recommendation_change_h{horizon}"] = np.nan
                    record[f"optimality_loss_h{horizon}"] = np.nan
                    record[f"recovery_h{horizon}"] = np.nan
                    continue
                transition_indices = range(index + 1, index + horizon + 1)
                record[f"recommendation_change_h{horizon}"] = int(
                    any(
                        actions[next_index] != actions[next_index - 1]
                        for next_index in transition_indices
                    )
                )
                record[f"optimality_loss_h{horizon}"] = int(
                    optimal[index]
                    and any(not optimal[next_index] for next_index in transition_indices)
                )
                record[f"recovery_h{horizon}"] = int(
                    (not optimal[index])
                    and any(optimal[next_index] for next_index in transition_indices)
                )
            output.append(record)
    result = pd.DataFrame(output)
    for horizon in horizons:
        loss = result[f"optimality_loss_h{horizon}"].fillna(0).astype(bool)
        recovery = result[f"recovery_h{horizon}"].fillna(0).astype(bool)
        change = result[f"recommendation_change_h{horizon}"].fillna(0).astype(bool)
        if bool((loss & ~change).any()) or bool((recovery & ~change).any()):
            raise ValueError(
                f"Horizon {horizon}: optimality transitions must involve an action change"
            )
    return result


def add_recent_action_distribution_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "example_id",
        "position_index",
        "action_probabilities_json",
        "action_confidence",
        "action_entropy_bits",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Action rows missing required columns: {missing}")
    result = frame.copy()
    probability_rows: list[dict[str, float]] = []
    for value in result["action_probabilities_json"]:
        probabilities = parse_probabilities(value)
        probability_rows.append(
            {f"action_probability_{action.lower()}": probabilities[action] for action in ACTIONS}
        )
    result = pd.concat(
        [result.reset_index(drop=True), pd.DataFrame(probability_rows)], axis=1
    )
    result["recent_action_js_bits"] = np.nan
    result["next_action_js_bits"] = np.nan
    for _, indices in result.groupby("example_id", sort=False).groups.items():
        ordered_index = (
            result.loc[list(indices)]
            .sort_values("position_index")
            .index.to_numpy()
        )
        distributions = result.loc[
            ordered_index,
            [f"action_probability_{action.lower()}" for action in ACTIONS],
        ].to_numpy(dtype=float)
        adjacent = [
            jensen_shannon_bits(distributions[index - 1], distributions[index])
            for index in range(1, len(distributions))
        ]
        result.loc[ordered_index[1:], "recent_action_js_bits"] = adjacent
        result.loc[ordered_index[:-1], "next_action_js_bits"] = adjacent
    return result


def jensen_shannon_bits(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)

    def kl(source: np.ndarray, target: np.ndarray) -> float:
        mask = source > 0
        return float(
            np.sum(source[mask] * np.log2(source[mask] / target[mask]))
        )

    return 0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)


def build_belief_features(
    positions: pd.DataFrame,
    beliefs: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Build observable and simulator-verified belief features at each position."""
    required = {
        "example_id",
        "reasoning_step_idx",
        "question_id",
        "answer_key",
        "belief_is_error",
        "entropy_bits",
        "probabilities_json",
    }
    missing = sorted(required - set(beliefs.columns))
    if missing:
        raise ValueError(f"Belief rows missing required columns: {missing}")

    belief_lookup = {
        (str(row.example_id), int(row.reasoning_step_idx), str(row.question_id)): row
        for row in beliefs.itertuples()
    }
    observable_columns: set[str] = set()
    verified_columns: set[str] = set()
    output: list[dict[str, Any]] = []
    for example_id, group in positions.groupby("example_id", sort=True):
        ordered = group.sort_values("position_index")
        previous_answers: dict[str, str] = {}
        for row in ordered.itertuples():
            step = int(row.reasoning_step_idx)
            action = str(row.action_label).lower()
            features: dict[str, Any] = {
                "example_id": str(example_id),
                "reasoning_step_idx": step,
            }
            observable_errors: list[float] = []
            verified_errors: list[float] = []
            for question_id in PRIMARY_BELIEFS:
                belief = belief_lookup.get((str(example_id), step, question_id))
                if belief is None:
                    continue
                prefix = f"belief_{question_id}"
                probabilities = parse_probabilities(str(belief.probabilities_json))
                for answer in ("yes", "no", "unknown"):
                    column = f"{prefix}_prob_{answer}"
                    features[column] = probabilities.get(answer, 0.0)
                    observable_columns.add(column)
                entropy_column = f"{prefix}_entropy"
                change_column = f"{prefix}_changed"
                error_column = f"{prefix}_error"
                answer = str(belief.answer_key)
                features[entropy_column] = float(belief.entropy_bits)
                features[change_column] = int(
                    question_id in previous_answers
                    and answer != previous_answers[question_id]
                )
                features[error_column] = int(as_bool(belief.belief_is_error))
                previous_answers[question_id] = answer
                observable_columns.update((entropy_column, change_column))
                verified_columns.add(error_column)
                observable_errors.append(float(features[entropy_column]))
                verified_errors.append(float(features[error_column]))

            consequence_errors: list[float] = []
            consequence_entropies: list[float] = []
            for consequence_prefix in CONSEQUENCE_PREFIXES:
                question_id = f"{consequence_prefix}{action}"
                belief = belief_lookup.get((str(example_id), step, question_id))
                short_name = consequence_prefix.removesuffix("_after_")
                if belief is None:
                    continue
                prefix = f"chosen_action_{short_name}"
                probabilities = parse_probabilities(str(belief.probabilities_json))
                for answer in ("yes", "no", "unknown"):
                    column = f"{prefix}_prob_{answer}"
                    features[column] = probabilities.get(answer, 0.0)
                    observable_columns.add(column)
                entropy_column = f"{prefix}_entropy"
                change_column = f"{prefix}_changed"
                error_column = f"{prefix}_error"
                answer = str(belief.answer_key)
                previous_key = f"chosen::{short_name}"
                features[entropy_column] = float(belief.entropy_bits)
                features[change_column] = int(
                    previous_key in previous_answers
                    and answer != previous_answers[previous_key]
                )
                features[error_column] = int(as_bool(belief.belief_is_error))
                previous_answers[previous_key] = answer
                observable_columns.update((entropy_column, change_column))
                verified_columns.add(error_column)
                consequence_entropies.append(float(features[entropy_column]))
                consequence_errors.append(float(features[error_column]))

            wall_question = f"wall_{action}"
            wall_belief = belief_lookup.get((str(example_id), step, wall_question))
            hit_belief = belief_lookup.get(
                (str(example_id), step, f"hit_wall_after_{action}")
            )
            features["chosen_direction_reported_blocked"] = int(
                wall_belief is not None and str(wall_belief.answer_key) == "yes"
            )
            features["chosen_action_predicted_wall_hit"] = int(
                hit_belief is not None and str(hit_belief.answer_key) == "yes"
            )
            features["belief_action_conflict"] = int(
                features["chosen_direction_reported_blocked"]
                or features["chosen_action_predicted_wall_hit"]
            )
            observable_columns.update(
                (
                    "chosen_direction_reported_blocked",
                    "chosen_action_predicted_wall_hit",
                    "belief_action_conflict",
                )
            )
            features["current_belief_mean_entropy"] = (
                float(np.mean(observable_errors)) if observable_errors else np.nan
            )
            features["chosen_consequence_mean_entropy"] = (
                float(np.mean(consequence_entropies))
                if consequence_entropies
                else np.nan
            )
            features["current_belief_error_rate"] = (
                float(np.mean(verified_errors)) if verified_errors else np.nan
            )
            features["chosen_consequence_error_rate"] = (
                float(np.mean(consequence_errors)) if consequence_errors else np.nan
            )
            observable_columns.update(
                ("current_belief_mean_entropy", "chosen_consequence_mean_entropy")
            )
            verified_columns.update(
                ("current_belief_error_rate", "chosen_consequence_error_rate")
            )
            output.append(features)
    return (
        pd.DataFrame(output),
        tuple(sorted(observable_columns)),
        tuple(sorted(verified_columns)),
    )


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def representation_dynamics(
    layer_vectors: dict[int, np.ndarray],
    *,
    rolling_window: int = 5,
) -> pd.DataFrame:
    """Compute causal sentence-level representation dynamics for one state."""
    if set(layer_vectors) != {8, 15, 23}:
        raise ValueError("Representation dynamics require layers 8, 15, and 23")
    lengths = {len(matrix) for matrix in layer_vectors.values()}
    if len(lengths) != 1:
        raise ValueError("Layer matrices have different sentence counts")
    layer_15 = layer_vectors[15].astype(np.float64)
    output: list[dict[str, float]] = []
    for index in range(len(layer_15)):
        current = layer_15[index]
        history = layer_15[:index]
        window = layer_15[max(0, index - rolling_window + 1) : index + 1]
        center = window.mean(axis=0)
        dispersion = float(
            np.nanmean([1.0 - cosine_similarity(vector, center) for vector in window])
        )
        output.append(
            {
                "sentence_id": index,
                "activation_update_norm": (
                    float(np.linalg.norm(current - layer_15[index - 1]))
                    if index > 0
                    else np.nan
                ),
                "activation_adjacent_cosine_distance": (
                    1.0 - cosine_similarity(current, layer_15[index - 1])
                    if index > 0
                    else np.nan
                ),
                "activation_similarity_to_preceding_mean": (
                    cosine_similarity(current, history.mean(axis=0))
                    if len(history)
                    else np.nan
                ),
                "rolling_representation_dispersion": dispersion,
                "sparse_cross_layer_change": float(
                    np.nanmean(
                        [
                            1.0
                            - cosine_similarity(
                                layer_vectors[8][index], layer_vectors[15][index]
                            ),
                            1.0
                            - cosine_similarity(
                                layer_vectors[15][index], layer_vectors[23][index]
                            ),
                        ]
                    )
                ),
            }
        )
    return pd.DataFrame(output)


def balanced_group_splits(
    frame: pd.DataFrame,
    *,
    target_column: str,
    group_column: str,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    grouped = frame.groupby(group_column)[target_column].agg(["sum", "size"])
    groups = grouped.index.astype(str).tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    groups.sort(
        key=lambda group: (
            -float(grouped.loc[group, "sum"]),
            -int(grouped.loc[group, "size"]),
        )
    )
    fold_groups = [set() for _ in range(min(n_splits, len(groups)))]
    fold_events = [0.0 for _ in fold_groups]
    fold_sizes = [0 for _ in fold_groups]
    for group in groups:
        fold = min(
            range(len(fold_groups)),
            key=lambda index: (fold_events[index], fold_sizes[index]),
        )
        fold_groups[fold].add(group)
        fold_events[fold] += float(grouped.loc[group, "sum"])
        fold_sizes[fold] += int(grouped.loc[group, "size"])
    group_values = frame[group_column].astype(str)
    splits = []
    for test_groups in fold_groups:
        test = np.flatnonzero(group_values.isin(test_groups).to_numpy())
        train = np.flatnonzero(~group_values.isin(test_groups).to_numpy())
        if len(train) and len(test):
            splits.append((train, test))
    return splits


def make_estimator(
    *,
    c_value: float,
    l1_ratio: float,
    aggregate_standardized_features: bool,
    seed: int,
) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ]
    if aggregate_standardized_features:
        steps.append(
            (
                "equal_weight_average",
                FunctionTransformer(
                    lambda matrix: np.mean(matrix, axis=1, keepdims=True)
                ),
            )
        )
    steps.append(
        (
            "model",
            LogisticRegression(
                C=c_value,
                solver="saga",
                l1_ratio=l1_ratio,
                max_iter=5000,
                tol=1e-3,
                random_state=seed,
            ),
        )
    )
    return Pipeline(steps)


def select_hyperparameters(
    frame: pd.DataFrame,
    *,
    spec: ModelSpec,
    target_column: str,
    group_column: str,
    seed: int,
    c_grid: Sequence[float] = (0.03, 0.3, 3.0),
    l1_grid: Sequence[float] = (0.0, 0.5, 1.0),
) -> tuple[float, float]:
    if frame[target_column].nunique() < 2:
        return 0.3, 0.0
    splits = balanced_group_splits(
        frame,
        target_column=target_column,
        group_column=group_column,
        n_splits=3,
        seed=seed,
    )
    best: tuple[float, float, float] | None = None
    for c_value in c_grid:
        for l1_ratio in l1_grid:
            fold_losses: list[float] = []
            for train_index, valid_index in splits:
                train = frame.iloc[train_index]
                valid = frame.iloc[valid_index]
                if train[target_column].nunique() < 2:
                    continue
                estimator = make_estimator(
                    c_value=c_value,
                    l1_ratio=l1_ratio,
                    aggregate_standardized_features=spec.aggregate_standardized_features,
                    seed=seed,
                )
                estimator.fit(
                    train[list(spec.feature_columns)],
                    train[target_column].astype(int),
                )
                probability = estimator.predict_proba(
                    valid[list(spec.feature_columns)]
                )[:, 1]
                fold_losses.append(
                    log_loss(
                        valid[target_column].astype(int),
                        probability,
                        labels=[0, 1],
                    )
                )
            if not fold_losses:
                continue
            candidate = (float(np.mean(fold_losses)), c_value, l1_ratio)
            if best is None or candidate < best:
                best = candidate
    return (best[1], best[2]) if best is not None else (0.3, 0.0)


def nested_group_predictions(
    frame: pd.DataFrame,
    *,
    spec: ModelSpec,
    target_column: str,
    group_column: str = "trajectory_id",
    outer_splits: int = 5,
    seed: int = 42,
    conditional_train_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create out-of-fold predictions with tuning restricted to training groups."""
    required = {"row_id", group_column, target_column, *spec.feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction frame missing columns: {missing}")
    source = frame.dropna(subset=[target_column]).reset_index(drop=True)
    splits = balanced_group_splits(
        source,
        target_column=target_column,
        group_column=group_column,
        n_splits=outer_splits,
        seed=seed,
    )
    predictions: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    for fold, (train_index, test_index) in enumerate(splits):
        train = source.iloc[train_index].copy()
        test = source.iloc[test_index].copy()
        fit_train = train
        if conditional_train_column is not None:
            fit_train = train[train[conditional_train_column].astype(int) == 1]
        if fit_train.empty or fit_train[target_column].nunique() < 2:
            prevalence = float(train[target_column].mean())
            probability = np.full(len(test), prevalence)
            c_value, l1_ratio = np.nan, np.nan
            estimator = None
        else:
            c_value, l1_ratio = select_hyperparameters(
                fit_train,
                spec=spec,
                target_column=target_column,
                group_column=group_column,
                seed=seed + fold,
            )
            estimator = make_estimator(
                c_value=c_value,
                l1_ratio=l1_ratio,
                aggregate_standardized_features=spec.aggregate_standardized_features,
                seed=seed + fold,
            )
            estimator.fit(
                fit_train[list(spec.feature_columns)],
                fit_train[target_column].astype(int),
            )
            probability = estimator.predict_proba(
                test[list(spec.feature_columns)]
            )[:, 1]
        for row, score in zip(test.to_dict("records"), probability, strict=True):
            predictions.append(
                {
                    "row_id": row["row_id"],
                    "example_id": row["example_id"],
                    "trajectory_id": row["trajectory_id"],
                    "reasoning_step_idx": int(row["reasoning_step_idx"]),
                    "fold": fold,
                    "model": spec.name,
                    "target": target_column,
                    "outcome": int(row[target_column]),
                    "predicted_probability": float(score),
                    "selected_c": c_value,
                    "selected_l1_ratio": l1_ratio,
                }
            )
        if estimator is None:
            continue
        model = estimator.named_steps["model"]
        feature_names: Iterable[str]
        if spec.aggregate_standardized_features:
            feature_names = ("equal_weight_standardized_average",)
        else:
            feature_names = estimator.named_steps["imputer"].get_feature_names_out(
                spec.feature_columns
            )
        for feature, coefficient in zip(
            feature_names, model.coef_[0], strict=True
        ):
            coefficients.append(
                {
                    "fold": fold,
                    "model": spec.name,
                    "target": target_column,
                    "feature": feature,
                    "coefficient": float(coefficient),
                    "selected_c": c_value,
                    "selected_l1_ratio": l1_ratio,
                }
            )
    return pd.DataFrame(predictions), pd.DataFrame(coefficients)


def nested_large_distribution_predictions(
    frame: pd.DataFrame,
    *,
    spec: ModelSpec,
    value_column: str = "next_action_js_bits",
    quantile: float = 0.9,
    group_column: str = "trajectory_id",
    outer_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict unusually large next-sentence distribution changes.

    The binary threshold is estimated from each outer training fold. A global
    provisional label is used only to balance trajectory assignment to folds.
    """
    source = frame.dropna(subset=[value_column]).reset_index(drop=True).copy()
    provisional_threshold = float(source[value_column].quantile(quantile))
    source["_provisional_large_change"] = (
        source[value_column].astype(float) >= provisional_threshold
    ).astype(int)
    splits = balanced_group_splits(
        source,
        target_column="_provisional_large_change",
        group_column=group_column,
        n_splits=outer_splits,
        seed=seed,
    )
    predictions: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    for fold, (train_index, test_index) in enumerate(splits):
        train = source.iloc[train_index].copy()
        test = source.iloc[test_index].copy()
        threshold = float(train[value_column].quantile(quantile))
        target_column = "large_distribution_change_h1"
        train[target_column] = (train[value_column] >= threshold).astype(int)
        test[target_column] = (test[value_column] >= threshold).astype(int)
        c_value, l1_ratio = select_hyperparameters(
            train,
            spec=spec,
            target_column=target_column,
            group_column=group_column,
            seed=seed + fold,
        )
        estimator = make_estimator(
            c_value=c_value,
            l1_ratio=l1_ratio,
            aggregate_standardized_features=spec.aggregate_standardized_features,
            seed=seed + fold,
        )
        estimator.fit(
            train[list(spec.feature_columns)], train[target_column].astype(int)
        )
        probability = estimator.predict_proba(
            test[list(spec.feature_columns)]
        )[:, 1]
        for row, score in zip(test.to_dict("records"), probability, strict=True):
            predictions.append(
                {
                    "row_id": row["row_id"],
                    "example_id": row["example_id"],
                    "trajectory_id": row["trajectory_id"],
                    "reasoning_step_idx": int(row["reasoning_step_idx"]),
                    "fold": fold,
                    "model": spec.name,
                    "target": target_column,
                    "outcome": int(row[target_column]),
                    "predicted_probability": float(score),
                    "target_threshold_js_bits": threshold,
                    "next_action_js_bits": float(row[value_column]),
                    "selected_c": c_value,
                    "selected_l1_ratio": l1_ratio,
                }
            )
        model = estimator.named_steps["model"]
        feature_names: Iterable[str]
        if spec.aggregate_standardized_features:
            feature_names = ("equal_weight_standardized_average",)
        else:
            feature_names = estimator.named_steps["imputer"].get_feature_names_out(
                spec.feature_columns
            )
        for feature, coefficient in zip(
            feature_names, model.coef_[0], strict=True
        ):
            coefficients.append(
                {
                    "fold": fold,
                    "model": spec.name,
                    "target": target_column,
                    "feature": feature,
                    "coefficient": float(coefficient),
                    "selected_c": c_value,
                    "selected_l1_ratio": l1_ratio,
                }
            )
    return pd.DataFrame(predictions), pd.DataFrame(coefficients)


def prediction_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(labels)) < 2:
        return {
            "auprc": np.nan,
            "auroc": np.nan,
            "log_loss": np.nan,
            "brier_score": np.nan,
            "expected_calibration_error": np.nan,
        }
    return {
        "auprc": float(average_precision_score(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "expected_calibration_error": expected_calibration_error(
            labels, probabilities
        ),
    }


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        mask = (probabilities >= lower) & (
            probabilities <= upper if upper == 1.0 else probabilities < upper
        )
        if not np.any(mask):
            continue
        error += float(mask.mean()) * abs(
            float(labels[mask].mean()) - float(probabilities[mask].mean())
        )
    return float(error) if total else np.nan


def top_alert_metrics(
    predictions: pd.DataFrame,
    *,
    horizon: int,
    alert_fraction: float = 0.05,
) -> dict[str, float]:
    if predictions.empty:
        return {}
    threshold = float(
        predictions["predicted_probability"].quantile(1.0 - alert_fraction)
    )
    alerted = predictions["predicted_probability"] >= threshold
    labels = predictions["outcome"].astype(int)
    true_alerts = alerted & labels.eq(1)
    return {
        "top_alert_threshold": threshold,
        "top_5_percent_precision": (
            float(labels[alerted].mean()) if bool(alerted.any()) else np.nan
        ),
        "top_5_percent_recall": (
            float(true_alerts.sum() / labels.sum()) if int(labels.sum()) else np.nan
        ),
        "false_alerts_per_100_sentences": float(
            100.0 * (alerted & labels.eq(0)).sum() / len(predictions)
        ),
        "maximum_warning_horizon_sentences": float(horizon),
    }


def bootstrap_metric_difference(
    paired: pd.DataFrame,
    *,
    metric: str,
    repeats: int = 300,
    seed: int = 42,
) -> tuple[float, float]:
    groups = {
        str(group): rows
        for group, rows in paired.groupby("trajectory_id", sort=False)
    }
    group_ids = np.asarray(sorted(groups), dtype=object)
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(repeats):
        sample = rng.choice(group_ids, len(group_ids), replace=True)
        sampled = pd.concat([groups[group] for group in sample], ignore_index=True)
        labels = sampled["outcome"].to_numpy(dtype=int)
        baseline_probability = np.clip(
            sampled["baseline_probability"].to_numpy(dtype=float), 1e-6, 1 - 1e-6
        )
        candidate_probability = np.clip(
            sampled["predicted_probability"].to_numpy(dtype=float), 1e-6, 1 - 1e-6
        )
        if metric == "auprc":
            if len(np.unique(labels)) < 2:
                continue
            difference = average_precision_score(
                labels, candidate_probability
            ) - average_precision_score(labels, baseline_probability)
        elif metric == "log_loss":
            baseline_loss = -np.mean(
                labels * np.log(baseline_probability)
                + (1 - labels) * np.log(1 - baseline_probability)
            )
            candidate_loss = -np.mean(
                labels * np.log(candidate_probability)
                + (1 - labels) * np.log(1 - candidate_probability)
            )
            difference = baseline_loss - candidate_loss
        else:
            raise ValueError(f"Unsupported bootstrap metric: {metric}")
        if math.isfinite(difference):
            differences.append(float(difference))
    if not differences:
        return np.nan, np.nan
    return tuple(float(value) for value in np.quantile(differences, [0.025, 0.975]))
