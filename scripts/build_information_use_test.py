#!/usr/bin/env python3
"""Low-cost information-use tests from existing GPT-OSS activations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from scipy.stats import rankdata


EVENT_TYPES = {
    "action_change": {"action_change"},
    "optimal_to_suboptimal": {
        "transient_optimal_to_suboptimal",
        "sustained_optimal_to_suboptimal",
    },
    "suboptimal_to_optimal": {"suboptimal_to_optimal"},
    "commitment_onset": {"commitment_onset"},
}
LAYERS = (8, 15, 23)
REPRESENTATIONS = ("sentence_mean", "sentence_final")
PRIMARY_LAYER = 15
PRIMARY_REPRESENTATION = "sentence_mean"
BASELINE_PROGRESS = ("reasoning_progress",)
BASELINE_ACTION_CONFIDENCE = ("reasoning_progress", "action_confidence")
BELIEF_FEATURES = (
    "current_state_error_rate",
    "current_state_entropy",
    "transition_belief_error_rate",
    "transition_belief_entropy",
)
BLUE = "#1769AA"
LIGHT_BLUE = "#8CC8E8"
DARK = "#0B3C5D"
GRID = "#E5EEF5"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def auc(labels: Iterable[int], scores: Iterable[float]) -> float | None:
    y = np.asarray(list(labels), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return None
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def metrics(labels: list[int], probs: list[float]) -> dict[str, Any]:
    clipped = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(labels, dtype=int)
    return {
        "n_rows": len(labels),
        "n_positive": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "roc_auc": auc(labels, clipped),
        "brier_score": float(np.mean((clipped - y) ** 2)) if len(y) else float("nan"),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))) if len(y) else float("nan"),
    }


def grouped_folds(rows: list[dict[str, Any]], *, seed: int = 42, n_folds: int = 5) -> list[set[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["validation_group"])].append(row)
    group_ids = list(grouped)
    random.Random(seed).shuffle(group_ids)
    group_ids.sort(key=lambda g: (-sum(int(r["outcome"]) for r in grouped[g]), -len(grouped[g])))
    folds = [set() for _ in range(min(n_folds, len(group_ids)))]
    fold_pos = [0] * len(folds)
    fold_n = [0] * len(folds)
    for group_id in group_ids:
        idx = min(range(len(folds)), key=lambda i: (fold_pos[i], fold_n[i]))
        folds[idx].add(group_id)
        fold_pos[idx] += sum(int(r["outcome"]) for r in grouped[group_id])
        fold_n[idx] += len(grouped[group_id])
    return folds


def standardize_train_test(train: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train.mean(0)
    std = train.std(0).clamp_min(1e-5)
    return (train - mean) / std, (test - mean) / std


def pca_project_train_test(
    train: torch.Tensor,
    test: torch.Tensor,
    *,
    n_components: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = min(n_components, train.shape[0] - 1, train.shape[1] - 1)
    if q <= 0:
        return train, test
    _u, _s, v = torch.pca_lowrank(train, q=q, center=False, niter=2)
    return train @ v[:, :q], test @ v[:, :q]


def fit_predict_logistic(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, *, seed: int) -> list[float]:
    torch.manual_seed(seed)
    if train_x.shape[1] == 0:
        p = float(train_y.mean())
        return [p] * test_x.shape[0]
    model = torch.nn.Linear(train_x.shape[1], 1)
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=150, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        logits = model(train_x).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_y)
        loss = loss + 0.01 * model.weight.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return torch.sigmoid(model(test_x).squeeze(-1)).tolist()


def load_activation_matrix(
    rows: pd.DataFrame,
    activation_index_path: Path,
    *,
    layer: int,
    representation: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    index = pd.read_parquet(activation_index_path)
    index = index[(index.record_kind == "sentence") & (index.layer == layer)].copy()
    key_col = "mean_tensor_key" if representation == "sentence_mean" else "final_tensor_key"
    needed = rows[["example_id", "trajectory_id", "step_index", "sentence_id"]].copy()
    merged = needed.merge(
        index[["trajectory_id", "step_index", "sentence_id", "tensor_row", "shard_path", key_col]],
        on=["trajectory_id", "step_index", "sentence_id"],
        how="inner",
    )
    if len(merged) != len(rows):
        raise ValueError(
            f"Activation join lost rows for layer={layer} representation={representation}: "
            f"{len(merged)} of {len(rows)}"
        )
    vectors: list[np.ndarray] = []
    for shard_path, shard_rows in merged.groupby("shard_path", sort=False):
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            tensor_cache: dict[str, torch.Tensor] = {}
            for record in shard_rows.itertuples():
                tensor_key = getattr(record, key_col)
                if tensor_key not in tensor_cache:
                    tensor_cache[tensor_key] = handle.get_tensor(tensor_key).float()
                vectors.append(tensor_cache[tensor_key][int(record.tensor_row)].numpy())
    return merged.reset_index(drop=True), np.stack(vectors).astype("float32")


def build_analysis_rows(exp1_dir: Path, transition_rows_path: Path) -> pd.DataFrame:
    positions = pd.read_csv(exp1_dir / "position_rows.csv")
    events = pd.read_csv(exp1_dir / "event_rows.csv")
    positions = positions[positions.reasoning_step_idx > 0].copy()
    positions["sentence_id"] = positions.reasoning_step_idx.astype(int) - 1
    event_lookup: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in events.itertuples():
        event_lookup[(row.example_id, int(row.reasoning_step_idx))].add(row.event_type)
    for target, event_types in EVENT_TYPES.items():
        positions[target] = [
            int(bool(event_lookup.get((row.example_id, int(row.reasoning_step_idx)), set()) & event_types))
            for row in positions.itertuples()
        ]

    transitions = pd.read_csv(transition_rows_path)
    transitions = transitions[
        [
            "example_id",
            "reasoning_step_idx",
            "current_state_error_rate",
            "current_state_entropy",
            "transition_belief_error_rate",
            "transition_belief_entropy",
        ]
    ].copy()
    merged = positions.merge(transitions, on=["example_id", "reasoning_step_idx"], how="left")
    merged["validation_group"] = merged["matched_pair_id"].fillna(merged["trajectory_id"]).astype(str)
    for col in (
        "reasoning_progress",
        "action_confidence",
        "action_entropy_bits",
        "current_state_error_rate",
        "current_state_entropy",
        "transition_belief_error_rate",
        "transition_belief_entropy",
    ):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["current_state_error_rate"] = merged["current_state_error_rate"].fillna(merged["primary_belief_error_rate"])
    merged["current_state_entropy"] = merged["current_state_entropy"].fillna(merged["state_belief_entropy_bits"])
    merged["transition_belief_error_rate"] = merged["transition_belief_error_rate"].fillna(0.0)
    merged["transition_belief_entropy"] = merged["transition_belief_entropy"].fillna(0.0)
    return merged.reset_index(drop=True)


def feature_matrix(frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    if not features:
        return np.zeros((len(frame), 0), dtype="float32")
    return frame[list(features)].astype(float).fillna(0.0).to_numpy(dtype="float32")


def cross_validated_predictions(
    frame: pd.DataFrame,
    target: str,
    model_type: str,
    *,
    scalar_features: tuple[str, ...],
    activation_matrix: np.ndarray | None = None,
    n_components: int = 32,
    seed: int = 42,
    layer: int | None = None,
    representation: str | None = None,
) -> list[dict[str, Any]]:
    labels = frame[target].astype(int).to_numpy()
    if len(set(labels.tolist())) < 2:
        return []
    rows = frame.to_dict("records")
    for row, label in zip(rows, labels, strict=True):
        row["outcome"] = int(label)
    scalar = feature_matrix(frame, scalar_features)
    predictions: list[dict[str, Any]] = []
    for fold_idx, test_groups in enumerate(grouped_folds(rows, seed=seed)):
        train_idx = [i for i, row in enumerate(rows) if str(row["validation_group"]) not in test_groups]
        test_idx = [i for i, row in enumerate(rows) if str(row["validation_group"]) in test_groups]
        if len(set(labels[train_idx].tolist())) < 2 or not test_idx:
            continue
        train_parts: list[torch.Tensor] = []
        test_parts: list[torch.Tensor] = []
        if scalar.shape[1] > 0:
            train_scalar = torch.tensor(scalar[train_idx], dtype=torch.float32)
            test_scalar = torch.tensor(scalar[test_idx], dtype=torch.float32)
            train_scalar, test_scalar = standardize_train_test(train_scalar, test_scalar)
            train_parts.append(train_scalar)
            test_parts.append(test_scalar)
        if activation_matrix is not None:
            train_act = torch.tensor(activation_matrix[train_idx], dtype=torch.float32)
            test_act = torch.tensor(activation_matrix[test_idx], dtype=torch.float32)
            train_act, test_act = standardize_train_test(train_act, test_act)
            train_act, test_act = pca_project_train_test(train_act, test_act, n_components=n_components)
            train_parts.append(train_act)
            test_parts.append(test_act)
        train_x = torch.cat(train_parts, dim=1) if train_parts else torch.zeros((len(train_idx), 0))
        test_x = torch.cat(test_parts, dim=1) if test_parts else torch.zeros((len(test_idx), 0))
        train_y = torch.tensor(labels[train_idx], dtype=torch.float32)
        probs = fit_predict_logistic(train_x, train_y, test_x, seed=seed + fold_idx)
        for idx, prob in zip(test_idx, probs, strict=True):
            row = rows[idx]
            predictions.append(
                {
                    "target": target,
                    "model_type": model_type,
                    "fold": fold_idx,
                    "example_id": row["example_id"],
                    "trajectory_id": row["trajectory_id"],
                    "validation_group": row["validation_group"],
                    "reasoning_step_idx": int(row["reasoning_step_idx"]),
                    "outcome": int(row["outcome"]),
                    "probability": float(prob),
                    "layer": "" if layer is None else layer,
                    "representation": "" if representation is None else representation,
                }
            )
    return predictions


def summarize_predictions(predictions: list[dict[str, Any]], *, seed: int = 42) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[
            (
                str(row["target"]),
                str(row["model_type"]),
                str(row.get("layer", "")),
                str(row.get("representation", "")),
            )
        ].append(row)
    rows: list[dict[str, Any]] = []
    for (target, model_type, layer, representation), group in sorted(grouped.items()):
        labels = [int(row["outcome"]) for row in group]
        probs = [float(row["probability"]) for row in group]
        metric = metrics(labels, probs)
        rows.append(
            {
                "target": target,
                "model_type": model_type,
                "layer": layer,
                "representation": representation,
                "n_trajectories": len({row["trajectory_id"] for row in group}),
                "n_validation_groups": len({row["validation_group"] for row in group}),
                **metric,
                **bootstrap_metric_ci(group, seed=seed),
            }
        )
    return rows


def bootstrap_metric_ci(rows: list[dict[str, Any]], *, seed: int, repeats: int = 300) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["validation_group"])].append(row)
    group_ids = sorted(grouped)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(repeats):
        sample = [row for group_id in (rng.choice(group_ids) for _ in group_ids) for row in grouped[group_id]]
        metric = metrics([int(row["outcome"]) for row in sample], [float(row["probability"]) for row in sample])
        for key in ("roc_auc", "log_loss", "brier_score"):
            value = metric[key]
            if value is not None and np.isfinite(value):
                samples[key].append(float(value))
    out: dict[str, float] = {}
    for key, values in samples.items():
        if values:
            out[f"{key}_ci_low"] = float(np.quantile(values, 0.025))
            out[f"{key}_ci_high"] = float(np.quantile(values, 0.975))
    return out


def pca_components(matrix: np.ndarray, *, n_components: int = 32) -> np.ndarray:
    x = torch.tensor(matrix, dtype=torch.float32)
    x = (x - x.mean(0)) / x.std(0).clamp_min(1e-5)
    q = min(n_components, x.shape[0] - 1, x.shape[1] - 1)
    if q <= 0:
        return x.numpy()
    _u, _s, v = torch.pca_lowrank(x, q=q, center=False, niter=2)
    return (x @ v[:, :q]).numpy().astype("float32")


def linear_cka_with_binary_label(components: np.ndarray, labels: np.ndarray) -> float:
    x = np.asarray(components, dtype="float64")
    y = np.asarray(labels, dtype="float64").reshape(-1, 1)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    numerator = float(np.linalg.norm(x.T @ y, ord="fro") ** 2)
    x_norm = float(np.linalg.norm(x.T @ x, ord="fro"))
    y_norm = float(np.linalg.norm(y.T @ y, ord="fro"))
    if x_norm <= 0 or y_norm <= 0:
        return float("nan")
    return numerator / (x_norm * y_norm)


def permutation_scores_by_group(
    components: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    unique_groups = np.array(sorted(set(groups.tolist())), dtype=object)
    scores = []
    for _ in range(repeats):
        permuted = np.empty_like(labels)
        for group in unique_groups:
            indices = np.where(groups == group)[0]
            permuted[indices] = rng.permutation(labels[indices])
        scores.append(linear_cka_with_binary_label(components, permuted))
    return scores


def bootstrap_information_ci(
    components: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique_groups = np.array(sorted(set(groups.tolist())), dtype=object)
    estimates = []
    for _ in range(repeats):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([np.where(groups == group)[0] for group in sampled_groups])
        estimates.append(linear_cka_with_binary_label(components[indices], labels[indices]))
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def information_scores(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    *,
    layer: int,
    representation: str,
    permutation_repeats: int,
    bootstrap_repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    components = pca_components(matrix, n_components=32)
    groups = frame["validation_group"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for target_idx, target in enumerate(EVENT_TYPES):
        labels = frame[target].astype(int).to_numpy()
        if len(set(labels.tolist())) < 2:
            continue
        score = linear_cka_with_binary_label(components, labels)
        null_scores = permutation_scores_by_group(
            components,
            labels,
            groups,
            repeats=permutation_repeats,
            seed=seed + target_idx,
        )
        ci_low, ci_high = bootstrap_information_ci(
            components,
            labels,
            groups,
            repeats=bootstrap_repeats,
            seed=seed + 100 + target_idx,
        )
        p_value = (1 + sum(float(null) >= float(score) for null in null_scores)) / (1 + len(null_scores))
        rows.append(
            {
                "target": target,
                "layer": layer,
                "representation": representation,
                "n_rows": len(frame),
                "n_positive": int(labels.sum()),
                "n_validation_groups": len(set(groups.tolist())),
                "information_score_linear_cka": float(score),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "permutation_null_mean": float(np.mean(null_scores)),
                "permutation_null_sd": float(np.std(null_scores)),
                "permutation_p_value": float(p_value),
            }
        )
    return rows


def add_improvements(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_target: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        if row["model_type"] == "baseline_progress_action_confidence":
            by_target[str(row["target"])] = row
    out = []
    for row in summary_rows:
        baseline = by_target.get(str(row["target"]))
        enriched = dict(row)
        if baseline is not None:
            baseline_auc = baseline.get("roc_auc")
            baseline_log = baseline.get("log_loss")
            model_auc = row.get("roc_auc")
            model_log = row.get("log_loss")
            enriched["delta_auroc_vs_action_confidence_baseline"] = (
                float(model_auc) - float(baseline_auc)
                if model_auc is not None and baseline_auc is not None
                else float("nan")
            )
            enriched["log_loss_improvement_vs_action_confidence_baseline"] = (
                float(baseline_log) - float(model_log)
                if model_log is not None and baseline_log is not None
                else float("nan")
            )
        out.append(enriched)
    return out


def plot_predictive_improvement(summary: pd.DataFrame, output_dir: Path) -> None:
    primary = summary[
        (
            summary["model_type"].isin(
                [
                    "baseline_progress",
                    "baseline_progress_action_confidence",
                    "baseline_plus_beliefs",
                    "baseline_plus_primary_activation",
                    "baseline_plus_beliefs_and_primary_activation",
                ]
            )
        )
    ].copy()
    pivot = primary.pivot_table(
        index="target",
        columns="model_type",
        values="roc_auc",
        aggfunc="first",
    )
    order = list(EVENT_TYPES)
    models = [
        "baseline_plus_beliefs",
        "baseline_plus_primary_activation",
        "baseline_plus_beliefs_and_primary_activation",
    ]
    labels = [
        "Add beliefs",
        "Add activation",
        "Add beliefs and activation",
    ]
    colors = [LIGHT_BLUE, BLUE, DARK]
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    x = np.arange(len(order))
    width = 0.2
    baseline_values = [
        float(pivot.loc[target, "baseline_progress_action_confidence"])
        if target in pivot.index and "baseline_progress_action_confidence" in pivot.columns
        else np.nan
        for target in order
    ]
    ax.scatter(
        x,
        baseline_values,
        marker="D",
        s=54,
        color="#555555",
        label="Baseline: reasoning progress and action confidence",
        zorder=3,
    )
    for offset, (model, label, color) in enumerate(zip(models, labels, colors, strict=True)):
        values = [float(pivot.loc[target, model]) if target in pivot.index and model in pivot.columns else np.nan for target in order]
        ax.bar(x + (offset - 1) * width, values, width=width, label=label, color=color)
        for x_i, baseline, value in zip(x + (offset - 1) * width, baseline_values, values, strict=True):
            if np.isfinite(baseline) and np.isfinite(value):
                ax.plot([x_i, x_i], [baseline, value], color="#9CA3AF", linewidth=0.8, zorder=1)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks(x, labels=[target.replace("_", " ") for target in order], rotation=12, ha="right")
    ax.set_ylim(0.45, 0.9)
    ax.set_ylabel("Held-out event prediction AUROC")
    ax.set_xlabel("Action event")
    ax.set_title("Do Belief and Activation Features Improve Event Prediction?")
    ax.legend(frameon=False, ncols=2, fontsize=9)
    ax.grid(axis="y", color=GRID)
    fig.tight_layout()
    fig.savefig(output_dir / "figs" / "predictive_improvement_by_event.png", dpi=220)
    plt.close(fig)


def plot_information_scores(info: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, representation in zip(axes, REPRESENTATIONS, strict=True):
        subset = info[info["representation"] == representation].copy()
        x = np.arange(len(EVENT_TYPES))
        width = 0.22
        for idx, layer in enumerate(LAYERS):
            values = []
            for target in EVENT_TYPES:
                match = subset[(subset["target"] == target) & (subset["layer"] == layer)]
                values.append(float(match["information_score_linear_cka"].iloc[0]) if not match.empty else np.nan)
            ax.bar(x + (idx - 1) * width, values, width=width, label=f"Layer {layer}")
        ax.set_xticks(x, labels=[target.replace("_", " ") for target in EVENT_TYPES], rotation=12, ha="right")
        ax.set_xlabel("Action event")
        ax.set_title(representation.replace("_", " ").title())
        ax.grid(axis="y", color=GRID)
    axes[0].set_ylabel("Linear CKA score between activation and event label")
    axes[1].legend(frameon=False)
    fig.suptitle("Activation Dependence on Action Event Labels")
    fig.tight_layout()
    fig.savefig(output_dir / "figs" / "information_score_by_layer.png", dpi=220)
    plt.close(fig)


def write_report(output_dir: Path, info_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    info = pd.DataFrame(info_rows)
    summary = pd.DataFrame(summary_rows)
    lines = [
        "# Information-Use Test",
        "",
        "This analysis uses existing GPT-OSS-20B sentence activations and existing matched-46 action-event labels. It does not collect attention tensors.",
        "",
        f"- Sentence positions: {len(frame):,}",
        f"- Environment states: {frame['example_id'].nunique():,}",
        f"- Trajectories: {frame['trajectory_id'].nunique():,}",
        f"- Validation groups: {frame['validation_group'].nunique():,}",
        "- Primary activation representation: layer 15 sentence mean.",
        "- Predictive models use grouped cross-validation by matched pair or trajectory.",
        "- Dependence score is linear CKA between PCA-reduced activations and the binary event label; it is used as a low-cost HSIC-style proxy, not as a direct mutual-information estimate.",
        "- The event label is at the same sentence position as the activation. This is an event-classification test, not a leading-indicator test.",
        "- The permutation null shuffles labels within validation groups to break feature-label alignment while preserving each group's event count.",
        "",
        "## Difference From the MI-Peaks Paper",
        "",
        "The MI-peaks paper measures token-level HSIC/MI between each generated token representation and the gold answer representation, then searches for sparse peaks during reasoning. This run instead measures whether sentence-level GPT-OSS-20B activations are statistically dependent on action-event labels and whether they improve held-out event prediction. Therefore, the numbers here should be read as activation-event dependence scores, not as reproduced MI-peaks values.",
        "",
        "## Headline Predictive Improvement",
        "",
        "| Target | Best added signal | AUROC change | Log-loss improvement | Interpretation |",
        "|---|---|---:|---:|---|",
    ]
    preferred = summary[
        summary["model_type"].isin(
            [
                "baseline_plus_beliefs",
                "baseline_plus_primary_activation",
                "baseline_plus_beliefs_and_primary_activation",
            ]
        )
    ].copy()
    for target in EVENT_TYPES:
        target_rows = preferred[preferred["target"] == target].copy()
        if target_rows.empty:
            continue
        target_rows["score"] = target_rows["log_loss_improvement_vs_action_confidence_baseline"].astype(float)
        best = target_rows.sort_values("score", ascending=False).iloc[0]
        delta_auc = float(best["delta_auroc_vs_action_confidence_baseline"])
        delta_log = float(best["log_loss_improvement_vs_action_confidence_baseline"])
        if delta_log > 0 and delta_auc > 0:
            interp = "positive held-out improvement"
        elif delta_log > 0:
            interp = "small log-loss improvement only"
        else:
            interp = "no held-out improvement"
        lines.append(
            f"| {target.replace('_', ' ')} | {str(best['model_type']).replace('_', ' ')} | "
            f"{delta_auc:.3f} | {delta_log:.4f} | {interp} |"
        )
    lines.extend(
        [
            "",
        "## Strongest Dependence Scores",
            "",
            "| Target | Layer | Representation | Score | Null mean | Permutation p |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for target in EVENT_TYPES:
        target_info = info[info["target"] == target].copy()
        if target_info.empty:
            continue
        best = target_info.sort_values("information_score_linear_cka", ascending=False).iloc[0]
        lines.append(
            f"| {target.replace('_', ' ')} | {int(best['layer'])} | {best['representation']} | "
            f"{float(best['information_score_linear_cka']):.5f} | "
            f"{float(best['permutation_null_mean']):.5f} | {float(best['permutation_p_value']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- High activation-event dependence with no predictive improvement means the activation is associated with events but does not clearly improve action-event prediction.",
            "- Low dependence with no predictive improvement means there is no current evidence for this information-use claim.",
            "- Predictive improvement from belief features is stronger evidence for a belief-action link than raw activation dependence alone.",
            "- Attention extraction should remain deferred unless a follow-up targets the positive recommendation-change signal.",
        ]
    )
    (output_dir / "information_use_report.md").write_text("\n".join(lines) + "\n")

    captions = (
        "# Figure Captions\n\n"
        "## predictive_improvement_by_event.png\n\n"
        "Held-out AUROC for action-event classification in the matched 46-state GPT-OSS-20B run. "
        "The gray diamond is the baseline model using reasoning progress and action confidence. Reasoning progress is the fraction of the reasoning text revealed at the current sentence. Action confidence is the probability assigned to the currently recommended action. "
        "Bars show the same baseline plus belief features, layer 15 sentence-mean activation features, or both. "
        "The activation features are projected to 32 PCA components inside each training fold. "
        "The event label is at the same sentence position as the activation, so this plot tests event classification rather than advance prediction.\n\n"
        "## information_score_by_layer.png\n\n"
        "Activation-event dependence for action events across GPT-OSS-20B layers 8, 15, and 23. "
        "For each layer and sentence representation, activations are standardized, projected to 32 PCA components, and compared with the binary event label using linear CKA. "
        "This is a low-cost HSIC-style dependence proxy, not the exact mutual-information estimator used by the MI-peaks paper. "
        "Higher values indicate stronger association between activations and the event label, but do not by themselves show that the information is used for action selection.\n"
    )
    (output_dir / "FIGURE_CAPTIONS.md").write_text(captions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp1-dir", default="outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1")
    parser.add_argument("--transition-rows", default="outputs/hypothesis_tests/transition_activation_commitment_v1/transition_belief_rows.csv")
    parser.add_argument("--activation-index", default="outputs/activation_collection/gpt_oss_20b_boundary_v1/activation_index.parquet")
    parser.add_argument("--output-dir", default="outputs/hypothesis_tests/information_use_v1")
    parser.add_argument("--permutation-repeats", type=int, default=200)
    parser.add_argument("--bootstrap-repeats", type=int, default=300)
    args = parser.parse_args()

    torch.manual_seed(42)
    output_dir = Path(args.output_dir)
    (output_dir / "figs").mkdir(parents=True, exist_ok=True)

    frame = build_analysis_rows(Path(args.exp1_dir), Path(args.transition_rows))
    frame.to_csv(output_dir / "information_use_rows.csv", index=False)

    predictions: list[dict[str, Any]] = []
    info_rows: list[dict[str, Any]] = []
    primary_matrix: np.ndarray | None = None
    for layer in LAYERS:
        for representation in REPRESENTATIONS:
            _merged, matrix = load_activation_matrix(
                frame,
                Path(args.activation_index),
                layer=layer,
                representation=representation,
            )
            info_rows.extend(
                information_scores(
                    frame,
                    matrix,
                    layer=layer,
                    representation=representation,
                    permutation_repeats=args.permutation_repeats,
                    bootstrap_repeats=args.bootstrap_repeats,
                    seed=42 + layer,
                )
            )
            if layer == PRIMARY_LAYER and representation == PRIMARY_REPRESENTATION:
                primary_matrix = matrix

    if primary_matrix is None:
        raise RuntimeError("Primary activation matrix was not loaded.")

    for target in EVENT_TYPES:
        predictions.extend(
            cross_validated_predictions(
                frame,
                target,
                "baseline_progress",
                scalar_features=BASELINE_PROGRESS,
                seed=42,
            )
        )
        predictions.extend(
            cross_validated_predictions(
                frame,
                target,
                "baseline_progress_action_confidence",
                scalar_features=BASELINE_ACTION_CONFIDENCE,
                seed=42,
            )
        )
        predictions.extend(
            cross_validated_predictions(
                frame,
                target,
                "baseline_plus_beliefs",
                scalar_features=BASELINE_ACTION_CONFIDENCE + BELIEF_FEATURES,
                seed=42,
            )
        )
        predictions.extend(
            cross_validated_predictions(
                frame,
                target,
                "baseline_plus_primary_activation",
                scalar_features=BASELINE_ACTION_CONFIDENCE,
                activation_matrix=primary_matrix,
                layer=PRIMARY_LAYER,
                representation=PRIMARY_REPRESENTATION,
                seed=42,
            )
        )
        predictions.extend(
            cross_validated_predictions(
                frame,
                target,
                "baseline_plus_beliefs_and_primary_activation",
                scalar_features=BASELINE_ACTION_CONFIDENCE + BELIEF_FEATURES,
                activation_matrix=primary_matrix,
                layer=PRIMARY_LAYER,
                representation=PRIMARY_REPRESENTATION,
                seed=42,
            )
        )

    write_csv(output_dir / "predictive_improvement_predictions.csv", predictions)
    summary = add_improvements(summarize_predictions(predictions))
    write_csv(output_dir / "predictive_improvement_summary.csv", summary)
    write_csv(output_dir / "information_use_summary.csv", info_rows)

    plot_predictive_improvement(pd.DataFrame(summary), output_dir)
    plot_information_scores(pd.DataFrame(info_rows), output_dir)
    write_report(output_dir, info_rows, summary, frame)

    manifest = {
        "status": "completed",
        "exp1_dir": args.exp1_dir,
        "transition_rows": args.transition_rows,
        "activation_index": args.activation_index,
        "output_dir": args.output_dir,
        "rows": len(frame),
        "states": int(frame["example_id"].nunique()),
        "trajectories": int(frame["trajectory_id"].nunique()),
        "validation_groups": int(frame["validation_group"].nunique()),
        "primary_layer": PRIMARY_LAYER,
        "primary_representation": PRIMARY_REPRESENTATION,
        "permutation_repeats": args.permutation_repeats,
        "bootstrap_repeats": args.bootstrap_repeats,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
