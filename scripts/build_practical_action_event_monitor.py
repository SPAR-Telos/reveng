#!/usr/bin/env python3
"""Build prospective action-change and optimality-transition monitors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import font_manager
from safetensors import safe_open
from sklearn.metrics import precision_recall_curve
from transformers import AutoModel, AutoTokenizer

from reveng.experiments.practical_action_event_monitor import (
    ModelSpec,
    add_recent_action_distribution_features,
    bootstrap_metric_difference,
    build_belief_features,
    build_prospective_targets,
    make_estimator,
    nested_group_predictions,
    nested_large_distribution_predictions,
    prediction_metrics,
    representation_dynamics,
)


BLUE = "#1769AA"
DARK_BLUE = "#0B3C5D"
MID_BLUE = "#4F9BC8"
LIGHT_BLUE = "#A7D4ED"
PALE_BLUE = "#DCEEF8"
GRAY = "#66737C"
GRID = "#E2ECF2"
TARGET_LABELS = {
    "recommendation_change": "Recommendation change",
    "optimality_loss": "Optimality loss",
    "recovery": "Recovery",
    "large_distribution_change": "Large action-distribution change",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_matplotlib() -> None:
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update(
        {
            "font.family": "Arial" if "Arial" in fonts else "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_sentence_text(
    sentence_path: Path,
    example_ids: set[str],
) -> pd.DataFrame:
    sentences = pd.read_csv(sentence_path)
    sentences = sentences[
        sentences["trace_id"].astype(str).isin(example_ids)
        & sentences["kind"].astype(str).eq("reasoning")
    ].copy()
    sentences["example_id"] = sentences["trace_id"].astype(str)
    sentences["reasoning_step_idx"] = sentences["sentence_id"].astype(int) + 1
    if sentences.duplicated(["example_id", "reasoning_step_idx"]).any():
        raise ValueError("Canonical sentence table has duplicate sentence boundaries")
    return sentences[
        ["example_id", "reasoning_step_idx", "text", "char_start", "char_end"]
    ]


def encode_sentences(
    sentence_rows: pd.DataFrame,
    *,
    model_name: str,
    batch_size: int,
    cache_path: Path,
) -> np.ndarray:
    if cache_path.exists():
        cache = np.load(cache_path)
        cached_ids = cache["row_keys"].astype(str)
        expected_ids = (
            sentence_rows["example_id"].astype(str)
            + ":"
            + sentence_rows["reasoning_step_idx"].astype(str)
        ).to_numpy()
        if np.array_equal(cached_ids, expected_ids):
            return cache["embeddings"].astype(np.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    embeddings: list[np.ndarray] = []
    texts = sentence_rows["text"].fillna("").astype(str).tolist()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            embeddings.append(pooled.float().cpu().numpy())
    matrix = np.concatenate(embeddings, axis=0).astype(np.float32)
    row_keys = (
        sentence_rows["example_id"].astype(str)
        + ":"
        + sentence_rows["reasoning_step_idx"].astype(str)
    ).to_numpy(dtype=str)
    np.savez_compressed(cache_path, row_keys=row_keys, embeddings=matrix)
    return matrix


def build_semantic_features(
    sentence_rows: pd.DataFrame,
    embeddings: np.ndarray,
) -> pd.DataFrame:
    if len(sentence_rows) != len(embeddings):
        raise ValueError("Sentence rows and embeddings have different lengths")
    source = sentence_rows.reset_index(drop=True).copy()
    output: list[dict[str, Any]] = []
    for example_id, indices in source.groupby("example_id", sort=True).groups.items():
        ordered = source.loc[list(indices)].sort_values("reasoning_step_idx")
        ordered_indices = ordered.index.to_numpy(dtype=int)
        vectors = embeddings[ordered_indices]
        for local_index, row in enumerate(ordered.itertuples()):
            current = vectors[local_index]
            previous = vectors[:local_index]
            if len(previous):
                similarities = previous @ current
                preceding_mean = previous.mean(axis=0)
                preceding_mean /= max(float(np.linalg.norm(preceding_mean)), 1e-12)
                maximum_similarity = float(np.max(similarities))
                mean_similarity = float(current @ preceding_mean)
                adjacent_similarity = float(current @ previous[-1])
            else:
                maximum_similarity = np.nan
                mean_similarity = np.nan
                adjacent_similarity = np.nan
            output.append(
                {
                    "example_id": example_id,
                    "reasoning_step_idx": int(row.reasoning_step_idx),
                    "refrain_max_semantic_similarity": maximum_similarity,
                    "semantic_similarity_to_preceding_mean": mean_similarity,
                    "semantic_adjacent_similarity": adjacent_similarity,
                    "semantic_novelty_from_preceding_mean": (
                        1.0 - mean_similarity
                        if np.isfinite(mean_similarity)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(output)


def load_activation_features(
    index_path: Path,
    example_ids: set[str],
) -> pd.DataFrame:
    index = pd.read_parquet(index_path)
    sentence_index = index[
        index["record_kind"].astype(str).eq("sentence")
        & index["trace_id"].astype(str).isin(example_ids)
        & index["layer"].astype(int).isin((8, 15, 23))
    ].copy()
    output: list[pd.DataFrame] = []
    for example_id, state_index in sentence_index.groupby("trace_id", sort=True):
        shard_paths = state_index["shard_path"].astype(str).unique()
        if len(shard_paths) != 1:
            raise ValueError(f"{example_id}: expected one activation shard")
        matrices: dict[int, np.ndarray] = {}
        shard_path = Path(shard_paths[0])
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for layer in (8, 15, 23):
                layer_rows = state_index[
                    state_index["layer"].astype(int).eq(layer)
                ].sort_values("sentence_id")
                sentence_ids = layer_rows["sentence_id"].astype(int).to_numpy()
                if not np.array_equal(sentence_ids, np.arange(len(sentence_ids))):
                    raise ValueError(
                        f"{example_id}: layer {layer} sentence ids are incomplete"
                    )
                keys = layer_rows["mean_tensor_key"].astype(str).unique()
                if len(keys) != 1:
                    raise ValueError(
                        f"{example_id}: layer {layer} has multiple mean tensors"
                    )
                tensor = handle.get_tensor(keys[0]).float().numpy()
                tensor_rows = layer_rows["tensor_row"].astype(int).to_numpy()
                matrices[layer] = tensor[tensor_rows]
        state_features = representation_dynamics(matrices)
        state_features["example_id"] = str(example_id)
        state_features["reasoning_step_idx"] = (
            state_features["sentence_id"].astype(int) + 1
        )
        output.append(state_features.drop(columns="sentence_id"))
    result = pd.concat(output, ignore_index=True)
    if result["example_id"].nunique() != len(example_ids):
        raise ValueError(
            f"Activations cover {result['example_id'].nunique()} of {len(example_ids)} states"
        )
    return result


def make_model_specs(
    observable_beliefs: tuple[str, ...],
    verified_beliefs: tuple[str, ...],
) -> list[ModelSpec]:
    baseline = (
        "reasoning_progress",
        "action_confidence",
        "recent_action_js_bits",
    )
    activation = (
        "activation_similarity_to_preceding_mean",
        "activation_adjacent_cosine_distance",
        "activation_update_norm",
        "rolling_representation_dispersion",
        "sparse_cross_layer_change",
    )
    coarse = (
        "rolling_representation_dispersion",
        "sparse_cross_layer_change",
    )
    text = (
        "refrain_max_semantic_similarity",
        "semantic_similarity_to_preceding_mean",
        "semantic_adjacent_similarity",
        "semantic_novelty_from_preceding_mean",
    )
    return [
        ModelSpec("baseline", baseline),
        ModelSpec("activation", baseline + activation),
        ModelSpec(
            "coarse_representation_dynamics",
            coarse,
            aggregate_standardized_features=True,
        ),
        ModelSpec("observable_beliefs", baseline + observable_beliefs),
        ModelSpec(
            "verified_beliefs",
            baseline + observable_beliefs + verified_beliefs,
        ),
        ModelSpec("text_semantic", baseline + text),
        ModelSpec(
            "combined_observable",
            baseline + activation + observable_beliefs + text,
        ),
        ModelSpec(
            "combined_verified",
            baseline + activation + observable_beliefs + verified_beliefs + text,
        ),
    ]


def summarize_predictions(
    predictions: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, target_rows in predictions.groupby("target", sort=True):
        baseline = target_rows[target_rows["model"].eq("baseline")][
            ["row_id", "trajectory_id", "outcome", "predicted_probability"]
        ].rename(columns={"predicted_probability": "baseline_probability"})
        for model, model_rows in target_rows.groupby("model", sort=True):
            metrics = prediction_metrics(
                model_rows["outcome"].to_numpy(dtype=int),
                model_rows["predicted_probability"].to_numpy(dtype=float),
            )
            alert = event_warning_metrics(
                model_rows,
                feature_rows,
                target=str(target),
            )
            row = {
                "target": target,
                "model": model,
                "n_rows": len(model_rows),
                "n_trajectories": model_rows["trajectory_id"].nunique(),
                "n_positive_windows": int(model_rows["outcome"].sum()),
                "prevalence": float(model_rows["outcome"].mean()),
                **metrics,
                **alert,
            }
            paired = model_rows[
                ["row_id", "trajectory_id", "outcome", "predicted_probability"]
            ].merge(
                baseline,
                on=["row_id", "trajectory_id", "outcome"],
                how="inner",
            )
            if model != "baseline" and not paired.empty:
                baseline_metrics = prediction_metrics(
                    paired["outcome"].to_numpy(dtype=int),
                    paired["baseline_probability"].to_numpy(dtype=float),
                )
                row["delta_auprc_vs_baseline"] = (
                    metrics["auprc"] - baseline_metrics["auprc"]
                )
                row["delta_auroc_vs_baseline"] = (
                    metrics["auroc"] - baseline_metrics["auroc"]
                )
                row["log_loss_improvement_vs_baseline"] = (
                    baseline_metrics["log_loss"] - metrics["log_loss"]
                )
                for metric in ("auprc", "log_loss"):
                    low, high = bootstrap_metric_difference(
                        paired,
                        metric=metric,
                        repeats=bootstrap_repeats,
                        seed=seed,
                    )
                    prefix = (
                        "delta_auprc"
                        if metric == "auprc"
                        else "log_loss_improvement"
                    )
                    row[f"{prefix}_ci_low"] = low
                    row[f"{prefix}_ci_high"] = high
            else:
                row["delta_auprc_vs_baseline"] = 0.0
                row["delta_auroc_vs_baseline"] = 0.0
                row["log_loss_improvement_vs_baseline"] = 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def event_warning_metrics(
    predictions: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    target: str,
    alert_fraction: float = 0.05,
) -> dict[str, float]:
    threshold = float(
        predictions["predicted_probability"].quantile(1.0 - alert_fraction)
    )
    alerts = predictions[
        predictions["predicted_probability"].astype(float) >= threshold
    ].copy()
    positive_alerts = alerts["outcome"].astype(int).eq(1)
    result = {
        "top_alert_threshold": threshold,
        "top_5_percent_precision": (
            float(alerts["outcome"].mean()) if not alerts.empty else np.nan
        ),
        "false_alerts_per_100_sentences": float(
            100.0 * (~positive_alerts).sum() / len(predictions)
        ),
        "event_recall_within_horizon": np.nan,
        "median_warning_sentences": np.nan,
        "n_unique_events": np.nan,
    }
    if target == "large_distribution_change_h1":
        result["event_recall_within_horizon"] = float(
            positive_alerts.sum() / predictions["outcome"].sum()
        )
        result["n_unique_events"] = float(predictions["outcome"].sum())
        return result

    event_name, horizon_text = target.rsplit("_h", 1)
    horizon = int(horizon_text)
    one_step_column = f"{event_name}_h1"
    event_rows = feature_rows[
        feature_rows[one_step_column].fillna(0).astype(int).eq(1)
    ][["example_id", "reasoning_step_idx"]].copy()
    event_rows["event_sentence"] = event_rows["reasoning_step_idx"].astype(int) + 1
    alert_lookup = {
        str(example_id): sorted(group["reasoning_step_idx"].astype(int).tolist())
        for example_id, group in alerts.groupby("example_id")
    }
    leads: list[int] = []
    for event in event_rows.itertuples():
        candidates = [
            step
            for step in alert_lookup.get(str(event.example_id), [])
            if 1 <= int(event.event_sentence) - step <= horizon
        ]
        if candidates:
            leads.append(int(event.event_sentence) - min(candidates))
    result["n_unique_events"] = float(len(event_rows))
    result["event_recall_within_horizon"] = (
        float(len(leads) / len(event_rows)) if len(event_rows) else np.nan
    )
    result["median_warning_sentences"] = (
        float(np.median(leads)) if leads else np.nan
    )
    return result


def add_hierarchical_predictions(
    frame: pd.DataFrame,
    direct_predictions: pd.DataFrame,
    specs: list[ModelSpec],
    *,
    horizons: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_names = {"baseline", "combined_observable", "combined_verified"}
    selected_specs = [spec for spec in specs if spec.name in selected_names]
    hierarchical_predictions: list[pd.DataFrame] = []
    coefficient_rows: list[pd.DataFrame] = []
    for horizon in horizons:
        change_target = f"recommendation_change_h{horizon}"
        for event_name, optimal_value in (
            ("optimality_loss", 1),
            ("recovery", 0),
        ):
            target = f"{event_name}_h{horizon}"
            risk = frame[
                frame["current_action_is_optimal"].astype(int).eq(optimal_value)
                & frame[f"complete_horizon_{horizon}"].astype(int).eq(1)
            ].copy()
            for spec in selected_specs:
                conditional_spec = ModelSpec(
                    name=f"{spec.name}_conditional",
                    feature_columns=spec.feature_columns,
                    aggregate_standardized_features=spec.aggregate_standardized_features,
                )
                conditional, coefficients = nested_group_predictions(
                    risk,
                    spec=conditional_spec,
                    target_column=target,
                    conditional_train_column=change_target,
                    seed=79 + horizon,
                )
                change = direct_predictions[
                    direct_predictions["target"].eq(change_target)
                    & direct_predictions["model"].eq(spec.name)
                ][["row_id", "predicted_probability"]].rename(
                    columns={"predicted_probability": "change_probability"}
                )
                merged = conditional.merge(change, on="row_id", how="inner")
                merged["conditional_event_probability"] = merged[
                    "predicted_probability"
                ]
                merged["predicted_probability"] = (
                    merged["change_probability"]
                    * merged["conditional_event_probability"]
                )
                merged["model"] = f"{spec.name}_hierarchical"
                merged["target"] = target
                hierarchical_predictions.append(merged)
                if not coefficients.empty:
                    coefficients["target"] = f"{target}_given_change"
                    coefficient_rows.append(coefficients)
    return (
        pd.concat(hierarchical_predictions, ignore_index=True),
        (
            pd.concat(coefficient_rows, ignore_index=True)
            if coefficient_rows
            else pd.DataFrame()
        ),
    )


def coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    if coefficients.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in coefficients.groupby(
        ["target", "model", "feature"], sort=True
    ):
        values = group["coefficient"].astype(float).to_numpy()
        nonzero = np.abs(values) > 1e-8
        rows.append(
            {
                "target": keys[0],
                "model": keys[1],
                "feature": keys[2],
                "n_folds": len(values),
                "mean_standardized_coefficient": float(np.mean(values)),
                "median_standardized_coefficient": float(np.median(values)),
                "nonzero_fold_fraction": float(np.mean(nonzero)),
                "positive_fold_fraction": (
                    float(np.mean(values[nonzero] > 0))
                    if bool(nonzero.any())
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def refit_direct_fold_coefficients(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    specs: list[ModelSpec],
) -> pd.DataFrame:
    """Recover fold coefficients from saved out-of-fold prediction metadata."""
    spec_by_name = {spec.name: spec for spec in specs}
    output: list[dict[str, Any]] = []
    direct = predictions[predictions["model"].isin(spec_by_name)].copy()
    for (target, model_name, fold), fold_predictions in direct.groupby(
        ["target", "model", "fold"], sort=True
    ):
        spec = spec_by_name[str(model_name)]
        if target == "large_distribution_change_h1":
            risk = frame[frame["complete_horizon_1"].astype(int).eq(1)].copy()
            threshold = float(
                fold_predictions["target_threshold_js_bits"].dropna().iloc[0]
            )
            risk[target] = (
                risk["next_action_js_bits"].astype(float) >= threshold
            ).astype(int)
        else:
            horizon = int(str(target).rsplit("_h", 1)[1])
            risk = frame[frame[f"complete_horizon_{horizon}"].astype(int).eq(1)]
            if str(target).startswith("optimality_loss"):
                risk = risk[risk["current_action_is_optimal"].astype(int).eq(1)]
            elif str(target).startswith("recovery"):
                risk = risk[risk["current_action_is_optimal"].astype(int).eq(0)]
        test_trajectories = set(
            fold_predictions["trajectory_id"].astype(str).unique()
        )
        train = risk[
            ~risk["trajectory_id"].astype(str).isin(test_trajectories)
        ].copy()
        if train.empty or train[target].nunique() < 2:
            continue
        c_value = float(fold_predictions["selected_c"].dropna().iloc[0])
        l1_ratio = float(
            fold_predictions["selected_l1_ratio"].dropna().iloc[0]
        )
        estimator = make_estimator(
            c_value=c_value,
            l1_ratio=l1_ratio,
            aggregate_standardized_features=spec.aggregate_standardized_features,
            seed=42 + int(fold),
        )
        estimator.fit(
            train[list(spec.feature_columns)], train[target].astype(int)
        )
        if spec.aggregate_standardized_features:
            feature_names = ("equal_weight_standardized_average",)
        else:
            feature_names = estimator.named_steps[
                "imputer"
            ].get_feature_names_out(spec.feature_columns)
        for feature, coefficient in zip(
            feature_names,
            estimator.named_steps["model"].coef_[0],
            strict=True,
        ):
            output.append(
                {
                    "fold": int(fold),
                    "model": model_name,
                    "target": target,
                    "feature": feature,
                    "coefficient": float(coefficient),
                    "selected_c": c_value,
                    "selected_l1_ratio": l1_ratio,
                }
            )
    return pd.DataFrame(output)


def calibration_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target, model), group in predictions.groupby(
        ["target", "model"], sort=True
    ):
        usable = group.copy()
        usable["calibration_bin"] = pd.qcut(
            usable["predicted_probability"],
            q=min(10, usable["predicted_probability"].nunique()),
            duplicates="drop",
        )
        for bin_index, (_, bin_rows) in enumerate(
            usable.groupby("calibration_bin", observed=True)
        ):
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "calibration_bin": bin_index,
                    "n": len(bin_rows),
                    "mean_predicted_probability": float(
                        bin_rows["predicted_probability"].mean()
                    ),
                    "observed_event_rate": float(bin_rows["outcome"].mean()),
                }
            )
    return pd.DataFrame(rows)


def target_sample_sizes(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in (1, 3):
        complete = frame[frame[f"complete_horizon_{horizon}"].astype(int).eq(1)]
        for target, risk_description, risk in (
            ("recommendation_change", "all sentence positions", complete),
            (
                "optimality_loss",
                "positions with a currently optimal recommendation",
                complete[complete["current_action_is_optimal"].astype(int).eq(1)],
            ),
            (
                "recovery",
                "positions with a currently suboptimal recommendation",
                complete[complete["current_action_is_optimal"].astype(int).eq(0)],
            ),
        ):
            column = f"{target}_h{horizon}"
            rows.append(
                {
                    "target": target,
                    "horizon_sentences": horizon,
                    "at_risk_population": risk_description,
                    "n_positions": len(risk),
                    "n_trajectories": risk["trajectory_id"].nunique(),
                    "n_positive_windows": int(risk[column].sum()),
                    "positive_window_rate": float(risk[column].mean()),
                    "n_unique_transitions": int(
                        frame[f"{target}_h1"].fillna(0).astype(int).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_precision_recall(
    predictions: pd.DataFrame,
    output_dir: Path,
) -> None:
    models = (
        "baseline",
        "activation",
        "observable_beliefs",
        "verified_beliefs",
        "text_semantic",
        "combined_verified",
        "combined_verified_hierarchical",
    )
    labels = {
        "baseline": "Behavioral baseline",
        "activation": "Baseline and activations",
        "observable_beliefs": "Baseline and observable beliefs",
        "verified_beliefs": "Baseline and verified belief errors",
        "text_semantic": "Baseline and text semantics",
        "combined_verified": "All signals, direct",
        "combined_verified_hierarchical": "All signals, hierarchical",
    }
    colors = [GRAY, BLUE, MID_BLUE, DARK_BLUE, LIGHT_BLUE, "#2E718E", "#0A91AB"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), sharey=True)
    for ax, horizon in zip(axes, (1, 3), strict=True):
        target = f"optimality_loss_h{horizon}"
        target_rows = predictions[predictions["target"].eq(target)]
        for model, color in zip(models, colors, strict=True):
            rows = target_rows[target_rows["model"].eq(model)]
            if rows.empty:
                continue
            precision, recall, _ = precision_recall_curve(
                rows["outcome"].astype(int),
                rows["predicted_probability"].astype(float),
            )
            metrics = prediction_metrics(
                rows["outcome"].to_numpy(dtype=int),
                rows["predicted_probability"].to_numpy(dtype=float),
            )
            ax.plot(
                recall,
                precision,
                color=color,
                linewidth=1.8,
                label=f"{labels[model]} ({metrics['auprc']:.3f})",
            )
        prevalence = float(
            target_rows[target_rows["model"].eq("baseline")]["outcome"].mean()
        )
        ax.axhline(prevalence, color="#999999", linestyle=":", linewidth=1)
        n_rows = len(target_rows[target_rows["model"].eq("baseline")])
        n_events = int(
            target_rows[target_rows["model"].eq("baseline")]["outcome"].sum()
        )
        ax.set_title(
            f"Within {horizon} sentence{'s' if horizon > 1 else ''}\n"
            f"n={n_rows:,} at-risk positions, positives={n_events:,}"
        )
        ax.set_xlabel("Recall of optimality-loss windows")
        ax.grid(color=GRID)
    axes[0].set_ylabel("Precision of optimality-loss alerts")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.suptitle("Prediction of Upcoming Optimality Loss")
    fig.tight_layout()
    fig.savefig(output_dir / "figs" / "optimality_loss_precision_recall.png", dpi=220)
    plt.close(fig)


def plot_auprc_improvements(summary: pd.DataFrame, output_dir: Path) -> None:
    models = [
        "activation",
        "observable_beliefs",
        "verified_beliefs",
        "text_semantic",
        "combined_observable",
        "combined_verified",
    ]
    model_labels = [
        "Activations",
        "Observable beliefs",
        "Verified belief errors",
        "Text semantics",
        "Combined observable",
        "Combined verified",
    ]
    colors = [BLUE, MID_BLUE, DARK_BLUE, LIGHT_BLUE, "#5B8FA8", "#0A91AB"]
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.0), sharey=True)
    for ax, horizon in zip(axes, (1, 3), strict=True):
        targets = [
            f"recommendation_change_h{horizon}",
            f"optimality_loss_h{horizon}",
            f"recovery_h{horizon}",
        ]
        x = np.arange(len(targets))
        width = 0.12
        for model_index, (model, label, color) in enumerate(
            zip(models, model_labels, colors, strict=True)
        ):
            values = []
            lower_errors = []
            upper_errors = []
            for target in targets:
                match = summary[
                    summary["target"].eq(target) & summary["model"].eq(model)
                ]
                if match.empty:
                    values.append(np.nan)
                    lower_errors.append(np.nan)
                    upper_errors.append(np.nan)
                    continue
                value = float(match["delta_auprc_vs_baseline"].iloc[0])
                low = float(match["delta_auprc_ci_low"].iloc[0])
                high = float(match["delta_auprc_ci_high"].iloc[0])
                values.append(value)
                lower_errors.append(max(0.0, value - low))
                upper_errors.append(max(0.0, high - value))
            ax.bar(
                x + (model_index - 2.5) * width,
                values,
                width=width,
                color=color,
                label=label,
                yerr=np.asarray([lower_errors, upper_errors]),
                capsize=2,
                error_kw={"elinewidth": 0.7, "capthick": 0.7},
            )
        ax.axhline(0, color="#555555", linewidth=1)
        tick_labels = []
        for target, short_label in zip(
            targets,
            ("Recommendation\nchange", "Optimality\nloss", "Recovery"),
            strict=True,
        ):
            baseline_row = summary[
                summary["target"].eq(target)
                & summary["model"].eq("baseline")
            ].iloc[0]
            tick_labels.append(
                f"{short_label}\n"
                f"n={int(baseline_row['n_rows']):,}\n"
                f"positive windows={int(baseline_row['n_positive_windows']):,}"
            )
        ax.set_xticks(x, labels=tick_labels, fontsize=8)
        ax.set_xlabel("Future action event")
        ax.set_title(
            f"Events within {horizon} sentence{'s' if horizon > 1 else ''}"
        )
        ax.grid(axis="y", color=GRID)
    axes[0].set_ylabel("Held-out AUPRC change from behavioral baseline")
    axes[1].legend(frameon=False, fontsize=7.8, ncols=2)
    fig.suptitle("Incremental Value of Activation, Belief, and Text Signals")
    fig.tight_layout(pad=1.8)
    fig.savefig(output_dir / "figs" / "auprc_improvement_by_signal.png", dpi=220)
    plt.close(fig)


def write_report(
    output_dir: Path,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    target_sizes: pd.DataFrame,
) -> None:
    lines = [
        "# Practical Prediction of Action Changes and Optimality Loss",
        "",
        "This analysis forecasts events after the current reasoning sentence. It does not classify an event using the activation or belief readout from the event sentence itself.",
        "",
        f"- Environment states: {frame['example_id'].nunique():,}",
        f"- Trajectories: {frame['trajectory_id'].nunique():,}",
        f"- Sentence-backed positions before horizon filtering: {len(frame):,}",
        "- Validation: five grouped outer folds by trajectory, with elastic-net tuning in grouped inner folds.",
        "- Primary metric: area under the precision-recall curve (AUPRC).",
        "",
        "## Targets",
        "",
        "| Target | Horizon | At-risk positions | Positive windows | Rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in target_sizes.itertuples():
        lines.append(
            f"| {TARGET_LABELS[row.target]} | {row.horizon_sentences} | "
            f"{row.n_positions:,} | {row.n_positive_windows:,} | "
            f"{row.positive_window_rate:.3f} |"
        )
    lines.extend(
        [
            "",
            "An optimality-loss prediction is evaluated only while the current recommendation is planner-optimal. A recovery prediction is evaluated only while it is suboptimal. The three-sentence labels indicate whether an event occurs at any of the next three sentence boundaries.",
            "",
            "## Held-Out Results",
            "",
            "| Target | Horizon | Best direct model | AUPRC | Baseline AUPRC | AUPRC change | Log-loss improvement |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for target_name in ("recommendation_change", "optimality_loss", "recovery"):
        for horizon in (1, 3):
            target = f"{target_name}_h{horizon}"
            rows = summary[
                summary["target"].eq(target)
                & ~summary["model"].str.endswith("_hierarchical")
            ].copy()
            baseline = rows[rows["model"].eq("baseline")]
            candidates = rows[~rows["model"].eq("baseline")]
            if baseline.empty or candidates.empty:
                continue
            best = candidates.sort_values(
                ["auprc", "log_loss_improvement_vs_baseline"],
                ascending=[False, False],
            ).iloc[0]
            lines.append(
                f"| {TARGET_LABELS[target_name]} | {horizon} | "
                f"{str(best['model']).replace('_', ' ')} | "
                f"{float(best['auprc']):.3f} | {float(baseline['auprc'].iloc[0]):.3f} | "
                f"{float(best['delta_auprc_vs_baseline']):+.3f} | "
                f"{float(best['log_loss_improvement_vs_baseline']):+.4f} |"
            )
    lines.extend(
        [
            "",
            "A positive AUPRC change is not sufficient by itself. The result is treated as useful only when held-out log loss also improves and the trajectory-bootstrap interval supports the same direction.",
            "",
            "## Headline",
            "",
        ]
    )
    observable_loss = summary[
        summary["target"].eq("optimality_loss_h3")
        & summary["model"].eq("observable_beliefs")
    ].iloc[0]
    lines.extend(
        [
            "Observable belief readouts provide the only clear improvement for the primary failure target. "
            f"They increase three-sentence optimality-loss AUPRC by {float(observable_loss['delta_auprc_vs_baseline']):+.3f} "
            f"(trajectory-bootstrap 95% interval {float(observable_loss['delta_auprc_ci_low']):+.3f} to {float(observable_loss['delta_auprc_ci_high']):+.3f}) "
            f"and improve log loss by {float(observable_loss['log_loss_improvement_vs_baseline']):+.4f} "
            f"({float(observable_loss['log_loss_improvement_ci_low']):+.4f} to {float(observable_loss['log_loss_improvement_ci_high']):+.4f}).",
            "",
            "The effect is not conclusive at the one-sentence horizon. Activation features do not improve optimality-loss prediction beyond the behavioral baseline, and the combined model does not outperform observable beliefs alone. This supports a short-horizon association between explicit belief readouts and upcoming optimality loss, not a general claim that all internal signals improve failure prediction.",
            "",
            "## Hierarchical Prediction",
            "",
            "| Target | Horizon | Direct combined-observable AUPRC | Hierarchical AUPRC | Direct log loss | Hierarchical log loss |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for target_name in ("optimality_loss", "recovery"):
        for horizon in (1, 3):
            target = f"{target_name}_h{horizon}"
            direct = summary[
                summary["target"].eq(target)
                & summary["model"].eq("combined_observable")
            ].iloc[0]
            hierarchical = summary[
                summary["target"].eq(target)
                & summary["model"].eq("combined_observable_hierarchical")
            ].iloc[0]
            lines.append(
                f"| {TARGET_LABELS[target_name]} | {horizon} | "
                f"{float(direct['auprc']):.3f} | "
                f"{float(hierarchical['auprc']):.3f} | "
                f"{float(direct['log_loss']):.4f} | "
                f"{float(hierarchical['log_loss']):.4f} |"
            )
    lines.extend(
        [
            "",
            "The hierarchical score multiplies the predicted probability of a recommendation change by the predicted probability that the change is an optimality loss or recovery. It does not consistently outperform direct prediction.",
            "",
            "## Alert Utility for Three-Sentence Optimality Loss",
            "",
            "| Model | Precision at top 5% alerts | Unique transitions detected | False alerts per 100 sentences | Median warning |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in (
        "baseline",
        "observable_beliefs",
        "combined_observable_hierarchical",
    ):
        row = summary[
            summary["target"].eq("optimality_loss_h3")
            & summary["model"].eq(model)
        ].iloc[0]
        lines.append(
            f"| {model.replace('_', ' ')} | "
            f"{float(row['top_5_percent_precision']):.3f} | "
            f"{float(row['event_recall_within_horizon']):.3f} | "
            f"{float(row['false_alerts_per_100_sentences']):.2f} | "
            f"{float(row['median_warning_sentences']):.1f} sentences |"
        )
    semantic_change = summary[
        summary["target"].eq("large_distribution_change_h1")
        & summary["model"].eq("text_semantic")
    ].iloc[0]
    lines.extend(
        [
            "",
            "At a fixed 5% alert budget, observable beliefs increase alert precision and reduce false alerts, but they do not clearly increase the fraction of unique optimality-loss transitions detected relative to the behavioral baseline.",
            "",
            "## Large Action-Distribution Changes",
            "",
            f"The text-semantic model improves next-sentence large-distribution-change AUPRC by {float(semantic_change['delta_auprc_vs_baseline']):+.3f} "
            f"({float(semantic_change['delta_auprc_ci_low']):+.3f} to {float(semantic_change['delta_auprc_ci_high']):+.3f}) "
            f"and log loss by {float(semantic_change['log_loss_improvement_vs_baseline']):+.4f} "
            f"({float(semantic_change['log_loss_improvement_ci_low']):+.4f} to {float(semantic_change['log_loss_improvement_ci_high']):+.4f}). "
            "This is a small but consistent association with changes in the full action distribution, not evidence that text semantics predicts planner failure.",
            "",
            "## Predictor Definitions",
            "",
            "- The behavioral baseline uses reasoning progress, current action confidence, and the Jensen-Shannon divergence from the preceding action distribution.",
            "- Observable belief signals use probe probabilities, entropy, answer changes, and conflicts between the recommended action and reported walls or consequences. They do not require simulator truth.",
            "- Verified belief errors compare those answers with the fixed DoorKey state and transition model.",
            "- Activation signals use GPT-OSS-20B layer-15 sentence means plus sparse cross-layer comparisons at layers 8, 15, and 23.",
            "- Text semantics use the REFRAIN `all-MiniLM-L6-v2` maximum similarity to preceding reasoning sentences and related sentence-embedding novelty measures.",
            "",
            "## Representation-Dynamics Baseline",
            "",
            "The coarse representation baseline averages training-fold-standardized rolling sentence dispersion and sparse cross-layer change. It is not D²H. Exact D²H requires token-level states at every layer and attention-guided token selection, which are unavailable in the stored sentence aggregates.",
            "",
            "## Commitment",
            "",
            "Retrospective commitment onset is not used as a deployable prediction target because it is defined by checking that the selected action remains unchanged through the rest of the trace. It remains a descriptive boundary.",
        ]
    )
    (output_dir / "prediction_report.md").write_text("\n".join(lines) + "\n")
    captions = (
        "# Figure Captions\n\n"
        "## optimality_loss_precision_recall.png\n\n"
        "Precision-recall curves for forecasting a change from a planner-optimal recommendation to a suboptimal recommendation. Predictors use only information available at the current sentence. The one-sentence panel predicts the next sentence boundary; the three-sentence panel predicts any loss within the next three boundaries. Values in parentheses are held-out AUPRC. The dotted horizontal line is the event rate.\n\n"
        "## auprc_improvement_by_signal.png\n\n"
        "Change in trajectory-held-out AUPRC after adding activation, belief, or sentence-semantic signals to a behavioral baseline. Error bars are trajectory-bootstrap 95% intervals. The baseline uses reasoning progress, current action confidence, and recent change in the four-action probability distribution. Positive values indicate better ranking of future event windows; improvements are considered credible only when held-out log loss also improves.\n"
    )
    (output_dir / "FIGURE_CAPTIONS.md").write_text(captions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment1-dir",
        default="outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1",
    )
    parser.add_argument(
        "--experiment2-dir",
        default="outputs/experiment2_behavioral_beliefs/gpt_oss_local_sentence_matched46_v1",
    )
    parser.add_argument(
        "--activation-index",
        default="outputs/activation_collection/gpt_oss_20b_boundary_v1/activation_index.parquet",
    )
    parser.add_argument(
        "--sentences",
        default="data/behavioral_probes/doorkey_chunking_validation/sentences.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hypothesis_tests/practical_action_event_monitor_v1",
    )
    parser.add_argument(
        "--sentence-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-repeats", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Reuse saved out-of-fold predictions and regenerate summaries.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    setup_matplotlib()
    output_dir = Path(args.output_dir)
    (output_dir / "figs").mkdir(parents=True, exist_ok=True)

    exp1 = Path(args.experiment1_dir)
    exp2 = Path(args.experiment2_dir)
    position_path = exp1 / "position_rows.csv"
    belief_path = exp2 / "belief_rows.csv"
    positions = pd.read_csv(position_path)
    targets = build_prospective_targets(positions)
    frame = add_recent_action_distribution_features(targets)
    example_ids = set(frame["example_id"].astype(str))

    sentence_rows = load_sentence_text(Path(args.sentences), example_ids)
    embedding_cache = output_dir / "sentence_embeddings_all_minilm_l6_v2.npz"
    embeddings = encode_sentences(
        sentence_rows,
        model_name=args.sentence_model,
        batch_size=args.embedding_batch_size,
        cache_path=embedding_cache,
    )
    semantic_features = build_semantic_features(sentence_rows, embeddings)
    semantic_features.to_parquet(
        output_dir / "sentence_semantic_features.parquet", index=False
    )

    activation_features = load_activation_features(
        Path(args.activation_index), example_ids
    )
    activation_features.to_parquet(
        output_dir / "sentence_activation_features.parquet", index=False
    )

    belief_features, observable_columns, verified_columns = build_belief_features(
        positions, pd.read_csv(belief_path)
    )
    belief_features.to_parquet(
        output_dir / "sentence_belief_features.parquet", index=False
    )

    for features in (semantic_features, activation_features, belief_features):
        frame = frame.merge(
            features,
            on=["example_id", "reasoning_step_idx"],
            how="left",
            validate="one_to_one",
        )
    common = frame[
        frame["reasoning_step_idx"].astype(int).gt(0)
        & frame["activation_similarity_to_preceding_mean"].notna()
        & frame["refrain_max_semantic_similarity"].notna()
    ].copy()
    common.to_parquet(output_dir / "prospective_feature_rows.parquet", index=False)
    specs = make_model_specs(observable_columns, verified_columns)

    prediction_path = output_dir / "out_of_fold_predictions.parquet"
    if args.report_only:
        if not prediction_path.exists():
            raise FileNotFoundError(
                "--report-only requires out_of_fold_predictions.parquet"
            )
        all_predictions = pd.read_parquet(prediction_path)
        all_coefficients = refit_direct_fold_coefficients(
            common, all_predictions, specs
        )
    else:
        prediction_frames: list[pd.DataFrame] = []
        coefficient_frames: list[pd.DataFrame] = []
        for horizon in (1, 3):
            complete = common[
                common[f"complete_horizon_{horizon}"].astype(int).eq(1)
            ]
            for target_name, risk_value in (
                ("recommendation_change", None),
                ("optimality_loss", 1),
                ("recovery", 0),
            ):
                target = f"{target_name}_h{horizon}"
                risk = (
                    complete
                    if risk_value is None
                    else complete[
                        complete["current_action_is_optimal"]
                        .astype(int)
                        .eq(risk_value)
                    ]
                )
                for spec in specs:
                    print(f"Fitting {target}: {spec.name}", flush=True)
                    predictions, coefficients = nested_group_predictions(
                        risk,
                        spec=spec,
                        target_column=target,
                        seed=args.seed + horizon,
                    )
                    prediction_frames.append(predictions)
                    coefficient_frames.append(coefficients)

        direct_predictions = pd.concat(prediction_frames, ignore_index=True)
        direct_coefficients = pd.concat(coefficient_frames, ignore_index=True)
        hierarchical, hierarchical_coefficients = add_hierarchical_predictions(
            common,
            direct_predictions,
            specs,
            horizons=(1, 3),
        )
        all_predictions = pd.concat(
            [direct_predictions, hierarchical], ignore_index=True
        )
        all_coefficients = pd.concat(
            [direct_coefficients, hierarchical_coefficients], ignore_index=True
        )

        complete_next = common[common["complete_horizon_1"].astype(int).eq(1)]
        for spec in specs:
            print(
                f"Fitting large_distribution_change_h1: {spec.name}",
                flush=True,
            )
            predictions, coefficients = nested_large_distribution_predictions(
                complete_next,
                spec=spec,
                seed=args.seed,
            )
            all_predictions = pd.concat(
                [all_predictions, predictions], ignore_index=True
            )
            all_coefficients = pd.concat(
                [all_coefficients, coefficients], ignore_index=True
            )

        all_predictions.to_parquet(prediction_path, index=False)
    all_coefficients.to_parquet(
        output_dir / "fold_feature_coefficients.parquet", index=False
    )
    summary = summarize_predictions(
        all_predictions,
        common,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    summary.to_csv(output_dir / "model_comparison.csv", index=False)
    coefficient_stability(all_coefficients).to_csv(
        output_dir / "feature_coefficient_stability.csv", index=False
    )
    calibration_rows(all_predictions).to_csv(
        output_dir / "calibration_summary.csv", index=False
    )
    target_sizes = target_sample_sizes(common)
    target_sizes.to_csv(output_dir / "target_sample_sizes.csv", index=False)
    summary[
        [
            "target",
            "model",
            "top_alert_threshold",
            "top_5_percent_precision",
            "event_recall_within_horizon",
            "false_alerts_per_100_sentences",
            "median_warning_sentences",
            "n_unique_events",
        ]
    ].to_csv(output_dir / "warning_time_summary.csv", index=False)

    plot_precision_recall(all_predictions, output_dir)
    plot_auprc_improvements(summary, output_dir)
    write_report(output_dir, common, summary, target_sizes)
    manifest = {
        "status": "complete",
        "seed": args.seed,
        "sentence_model": args.sentence_model,
        "environment_states": int(common["example_id"].nunique()),
        "trajectories": int(common["trajectory_id"].nunique()),
        "sentence_backed_positions": len(common),
        "input_files": {
            str(position_path): sha256(position_path),
            str(belief_path): sha256(belief_path),
            str(args.activation_index): sha256(Path(args.activation_index)),
            str(args.sentences): sha256(Path(args.sentences)),
        },
        "target_horizons_sentences": [1, 3],
        "outer_grouping": "trajectory_id",
        "outer_folds": 5,
        "inner_folds": 3,
        "elastic_net_c_grid": [0.03, 0.3, 3.0],
        "elastic_net_l1_ratio_grid": [0.0, 0.5, 1.0],
        "large_distribution_change_quantile": 0.9,
        "exact_d2h_implemented": False,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
