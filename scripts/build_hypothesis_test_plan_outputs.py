#!/usr/bin/env python3
"""Build hypothesis-test outputs for transition beliefs, activation monitors, and commitment tails."""

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
    "optimal_to_suboptimal": {"transient_optimal_to_suboptimal", "sustained_optimal_to_suboptimal"},
    "suboptimal_to_optimal": {"suboptimal_to_optimal"},
    "commitment_onset": {"commitment_onset"},
}
CURRENT_STATE_IDS = ("wall_left", "wall_right", "wall_up", "wall_down", "has_key", "door_open")
TRANSITION_PREFIXES = ("hit_wall_after_", "has_key_after_", "door_open_after_")
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
DIRECTION_BY_ACTION = {"UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right"}
BLUE = "#1769aa"
LIGHT_BLUE = "#9bd0f5"
DARK = "#0b3c5d"
GRID = "#e5eef5"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_bool(value: Any) -> bool | None:
    if str(value) == "True":
        return True
    if str(value) == "False":
        return False
    return None


def parse_probs(value: str) -> dict[str, float]:
    try:
        return {str(k): float(v) for k, v in json.loads(value).items()}
    except Exception:
        return {}


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
    clipped = [min(max(float(p), 1e-6), 1 - 1e-6) for p in probs]
    return {
        "n_rows": len(labels),
        "n_positive": int(sum(labels)),
        "positive_rate": float(sum(labels) / len(labels)) if labels else float("nan"),
        "roc_auc": auc(labels, probs),
        "brier_score": float(np.mean([(p - y) ** 2 for p, y in zip(clipped, labels, strict=True)])) if labels else float("nan"),
        "log_loss": float(-np.mean([y * math.log(p) + (1 - y) * math.log(1 - p) for p, y in zip(clipped, labels, strict=True)])) if labels else float("nan"),
    }


def bootstrap_ci(rows: list[dict[str, Any]], *, seed: int, repeats: int = 300) -> dict[str, Any]:
    if not rows:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["validation_group"])].append(row)
    groups = sorted(grouped)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(repeats):
        sample = [r for g in (rng.choice(groups) for _ in groups) for r in grouped[g]]
        m = metrics([int(r["outcome"]) for r in sample], [float(r["probability"]) for r in sample])
        for key in ("roc_auc", "brier_score", "log_loss"):
            if m[key] is not None and np.isfinite(m[key]):
                samples[key].append(float(m[key]))
    out: dict[str, Any] = {}
    for key, vals in samples.items():
        if vals:
            out[f"{key}_ci_low"] = float(np.quantile(vals, 0.025))
            out[f"{key}_ci_high"] = float(np.quantile(vals, 0.975))
    return out


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
    for g in group_ids:
        idx = min(range(len(folds)), key=lambda i: (fold_pos[i], fold_n[i]))
        folds[idx].add(g)
        fold_pos[idx] += sum(int(r["outcome"]) for r in grouped[g])
        fold_n[idx] += len(grouped[g])
    return folds


def fit_predict_logistic(train: list[dict[str, Any]], test: list[dict[str, Any]], features: list[str], *, seed: int = 42) -> list[float]:
    if not features:
        p = sum(int(r["outcome"]) for r in train) / len(train)
        return [float(p)] * len(test)
    torch.manual_seed(seed)
    x_train = torch.tensor([[safe_float(r.get(f), 0.0) for f in features] for r in train], dtype=torch.float32)
    y_train = torch.tensor([int(r["outcome"]) for r in train], dtype=torch.float32)
    x_test = torch.tensor([[safe_float(r.get(f), 0.0) for f in features] for r in test], dtype=torch.float32)
    mean = x_train.mean(0)
    std = x_train.std(0).clamp_min(1e-5)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    model = torch.nn.Linear(x_train.shape[1], 1)
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=150, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        logits = model(x_train).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        loss = loss + 0.01 * model.weight.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return torch.sigmoid(model(x_test).squeeze(-1)).tolist()


def cross_validated_predictions(rows: list[dict[str, Any]], model_name: str, features: list[str], *, seed: int = 42) -> list[dict[str, Any]]:
    if len({int(r["outcome"]) for r in rows}) < 2:
        return []
    predictions: list[dict[str, Any]] = []
    for fold_idx, test_groups in enumerate(grouped_folds(rows, seed=seed)):
        train = [r for r in rows if str(r["validation_group"]) not in test_groups]
        test = [r for r in rows if str(r["validation_group"]) in test_groups]
        if not train or not test or len({int(r["outcome"]) for r in train}) < 2:
            continue
        probs = fit_predict_logistic(train, test, features, seed=seed + fold_idx)
        for row, prob in zip(test, probs, strict=True):
            predictions.append({
                "model_type": model_name,
                "fold": fold_idx,
                "example_id": row["example_id"],
                "trajectory_id": row["trajectory_id"],
                "validation_group": row["validation_group"],
                "reasoning_step_idx": row["reasoning_step_idx"],
                "outcome": int(row["outcome"]),
                "probability": float(prob),
            })
    return predictions


def summarize_predictions(predictions: list[dict[str, Any]], *, family_key: str = "target", seed: int = 42) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_key[(str(row[family_key]), str(row["model_type"]))].append(row)
    for (family, model_type), group in sorted(by_key.items()):
        label = [int(r["outcome"]) for r in group]
        prob = [float(r["probability"]) for r in group]
        rows.append({
            family_key: family,
            "model_type": model_type,
            "n_trajectories": len({r["trajectory_id"] for r in group}),
            **metrics(label, prob),
            **bootstrap_ci(group, seed=seed),
        })
    return rows


def add_metadata_from_positions(frame: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [c for c in ["matched_pair_id", "matched_role", "trajectory_class", "primary_step_failure_mode", "selection_stage"] if c in positions.columns]
    meta = positions[["example_id", "reasoning_step_idx", *meta_cols]].copy()
    return frame.merge(meta, on=["example_id", "reasoning_step_idx"], how="left")


def build_transition_belief_rows(exp1: Path, out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positions = pd.read_csv(exp1 / "position_rows.csv")
    beliefs = pd.read_csv(exp1 / "belief_rows.csv")
    if "reasoning_step_idx" not in beliefs.columns:
        beliefs["reasoning_step_idx"] = beliefs["position_index"]
    belief_map = {(r.example_id, int(r.reasoning_step_idx), r.question_id): r for r in beliefs.itertuples()}
    event_map: dict[tuple[str, int], set[str]] = defaultdict(set)
    for r in pd.read_csv(exp1 / "event_rows.csv").itertuples():
        event_map[(r.example_id, int(r.reasoning_step_idx))].add(r.event_type)
    rows: list[dict[str, Any]] = []
    for example_id, group in positions.groupby("example_id"):
        ordered = group.sort_values("reasoning_step_idx").reset_index(drop=True)
        for i in range(0, len(ordered) - 1):
            cur = ordered.iloc[i]
            nxt = ordered.iloc[i + 1]
            step = int(cur.reasoning_step_idx)
            features: dict[str, Any] = {}
            current_errors: list[float] = []
            current_entropy: list[float] = []
            for qid in CURRENT_STATE_IDS:
                b = belief_map.get((example_id, step, qid))
                if b is None:
                    continue
                err = int(str(b.belief_is_error) == "True")
                features[f"current_error_{qid}"] = err
                features[f"current_entropy_{qid}"] = safe_float(b.entropy_bits, 0.0)
                current_errors.append(err)
                current_entropy.append(safe_float(b.entropy_bits, 0.0))
            chosen_action = str(cur.action_label).lower()
            transition_errors: list[float] = []
            transition_entropy: list[float] = []
            for prefix in TRANSITION_PREFIXES:
                qid = f"{prefix}{chosen_action}"
                b = belief_map.get((example_id, step, qid))
                if b is None:
                    continue
                short = prefix.removesuffix("_after_")
                err = int(str(b.belief_is_error) == "True")
                features[f"transition_error_{short}"] = err
                features[f"transition_entropy_{short}"] = safe_float(b.entropy_bits, 0.0)
                transition_errors.append(err)
                transition_entropy.append(safe_float(b.entropy_bits, 0.0))
            events_next = event_map.get((example_id, int(nxt.reasoning_step_idx)), set())
            row = {
                "example_id": example_id,
                "trajectory_id": cur.trajectory_id,
                "validation_group": cur.matched_pair_id if "matched_pair_id" in ordered.columns and isinstance(cur.matched_pair_id, str) else cur.trajectory_id,
                "reasoning_step_idx": step,
                "next_reasoning_step_idx": int(nxt.reasoning_step_idx),
                "reasoning_progress": safe_float(cur.reasoning_progress, 0.0),
                "action_confidence": safe_float(cur.action_confidence, 0.0),
                "current_state_error_rate": float(np.mean(current_errors)) if current_errors else float("nan"),
                "current_state_entropy": float(np.mean(current_entropy)) if current_entropy else float("nan"),
                "transition_belief_error_rate": float(np.mean(transition_errors)) if transition_errors else float("nan"),
                "transition_belief_entropy": float(np.mean(transition_entropy)) if transition_entropy else float("nan"),
                "action_change": int("action_change" in events_next),
                "optimal_to_suboptimal": int(bool({"transient_optimal_to_suboptimal", "sustained_optimal_to_suboptimal"} & events_next)),
                "suboptimal_to_optimal": int("suboptimal_to_optimal" in events_next),
                "commitment_onset": int("commitment_onset" in events_next),
                "transition_belief_scope": "available_chosen_action_consequence_probes_not_allocentric",
                **features,
            }
            rows.append(row)
    write_csv(out / "transition_belief_rows.csv", rows)

    model_features = {
        "prevalence_baseline": [],
        "progress_baseline": ["reasoning_progress"],
        "current_state_beliefs": ["reasoning_progress", "current_state_error_rate", "current_state_entropy"],
        "transition_beliefs": ["reasoning_progress", "transition_belief_error_rate", "transition_belief_entropy"],
        "current_plus_transition": ["reasoning_progress", "current_state_error_rate", "current_state_entropy", "transition_belief_error_rate", "transition_belief_entropy"],
    }
    predictions: list[dict[str, Any]] = []
    for target in ("action_change", "optimal_to_suboptimal", "suboptimal_to_optimal", "commitment_onset"):
        target_rows = [{**r, "outcome": int(r[target])} for r in rows if np.isfinite(safe_float(r.get("transition_belief_error_rate")))]
        for model_name, features in model_features.items():
            preds = cross_validated_predictions(target_rows, model_name, features)
            for pred in preds:
                pred["target"] = target
            predictions.extend(preds)
    summary = summarize_predictions(predictions, family_key="target")
    write_csv(out / "transition_belief_model_summary.csv", summary)

    lines = [
        "# Current-State Versus Transition-Belief Comparison",
        "",
        "This first-pass comparison uses the transition-belief probes already present in the matched-46 run: the three action-consequence probes for the action recommended at each prefix. These probes are action-conditioned but not allocentric. Separately collected allocentric readouts are not included in this model comparison.",
        "",
        "| Target | Best learned model | AUROC | Events | Interpretation |",
        "|---|---|---:|---:|---|",
    ]
    for target in ("action_change", "optimal_to_suboptimal", "suboptimal_to_optimal", "commitment_onset"):
        candidates = [r for r in summary if r["target"] == target and r["model_type"] != "prevalence_baseline"]
        if not candidates:
            continue
        best = max(candidates, key=lambda r: -1 if r["roc_auc"] is None else float(r["roc_auc"]))
        interp = "promising" if best["model_type"] == "transition_beliefs" and float(best["roc_auc"] or 0) >= 0.6 else "not decisive"
        lines.append(f"| {target.replace('_', ' ')} | {best['model_type'].replace('_', ' ')} | {float(best['roc_auc']):.3f} | {best['n_positive']} | {interp} |")
    lines.extend([
        "",
        "Success criterion for the planned allocentric test: transition-belief AUROC must beat both progress and current-state baselines by at least 0.03 with a grouped-bootstrap lower bound above zero.",
    ])
    (out / "current_vs_transition_belief_comparison.md").write_text("\n".join(lines) + "\n")
    return rows, summary


def write_allocentric_probe_design(exp1: Path, out: Path) -> None:
    positions = pd.read_csv(exp1 / "position_rows.csv")
    rows: list[dict[str, Any]] = []
    # This design samples balanced counterfactual cells and actions. Labels are
    # intentionally left as pending because collection requires new model readouts.
    for idx, row in positions.drop_duplicates("example_id").head(46).iterrows():
        for case_idx, action in enumerate(ACTIONS):
            rows.append({
                "example_id": row.example_id,
                "trajectory_id": row.trajectory_id,
                "step_index": int(row.step_index),
                "case_id": f"{row.example_id}:allocentric:{case_idx}",
                "probe_kind": "allocentric_transition_belief",
                "action": action,
                "source_cell_policy": "balanced_traversable_non_agent_cell_from_same_grid",
                "key_status_policy": "balanced_true_false_across_cases",
                "door_status_policy": "balanced_open_closed_when_door_exists",
                "questions": "hit_wall,next_has_key,next_door_open",
                "status": "design_ready_label_and_readout_pending",
            })
    write_csv(out / "allocentric_transition_probe_design.csv", rows)


def load_activation_matrix(exp1: Path, activation_index_path: Path, *, layer: int, representation: str) -> tuple[pd.DataFrame, np.ndarray]:
    positions = pd.read_csv(exp1 / "position_rows.csv")
    positions = positions[positions.reasoning_step_idx > 0].copy()
    positions["sentence_id"] = positions.reasoning_step_idx.astype(int) - 1
    index = pd.read_parquet(activation_index_path)
    index = index[(index.record_kind == "sentence") & (index.layer == layer)]
    key_col = "mean_tensor_key" if representation == "sentence_mean" else "final_tensor_key"
    merged = positions.merge(
        index[["trajectory_id", "step_index", "sentence_id", "tensor_row", "shard_path", key_col]],
        on=["trajectory_id", "step_index", "sentence_id"],
        how="inner",
    )
    vectors: list[np.ndarray] = []
    for shard_path, shard_rows in merged.groupby("shard_path", sort=False):
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            tensor_cache: dict[str, torch.Tensor] = {}
            for r in shard_rows.itertuples():
                tensor_key = getattr(r, key_col)
                if tensor_key not in tensor_cache:
                    tensor_cache[tensor_key] = handle.get_tensor(tensor_key).float()
                tensor = tensor_cache[tensor_key]
                row_idx = int(r.tensor_row)
                vectors.append(tensor[row_idx].numpy())
    matrix = np.stack(vectors).astype("float32")
    return merged.reset_index(drop=True), matrix


def pca_logistic_predictions(frame: pd.DataFrame, x: np.ndarray, target: str, *, n_components: int = 32, seed: int = 42) -> list[dict[str, Any]]:
    rows = frame.to_dict("records")
    labels = frame[target].astype(int).to_numpy()
    for row, y in zip(rows, labels, strict=True):
        row["outcome"] = int(y)
        row["validation_group"] = str(row.get("matched_pair_id") or row["trajectory_id"])
    if len(set(labels.tolist())) < 2:
        return []
    preds: list[dict[str, Any]] = []
    folds = grouped_folds(rows, seed=seed)
    for fold_idx, test_groups in enumerate(folds):
        train_idx = [i for i, r in enumerate(rows) if str(r["validation_group"]) not in test_groups]
        test_idx = [i for i, r in enumerate(rows) if str(r["validation_group"]) in test_groups]
        if len(set(labels[train_idx].tolist())) < 2 or not test_idx:
            continue
        x_train = torch.tensor(x[train_idx], dtype=torch.float32)
        x_test = torch.tensor(x[test_idx], dtype=torch.float32)
        mean = x_train.mean(0)
        std = x_train.std(0).clamp_min(1e-5)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        q = min(n_components, x_train.shape[0] - 1, x_train.shape[1] - 1)
        if q > 0:
            _u, _s, v = torch.pca_lowrank(x_train, q=q, center=False, niter=2)
            x_train = x_train @ v[:, :q]
            x_test = x_test @ v[:, :q]
        y_train = torch.tensor(labels[train_idx], dtype=torch.float32)
        model = torch.nn.Linear(x_train.shape[1], 1)
        opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=150, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            logits = model(x_train).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train)
            loss = loss + 0.01 * model.weight.square().sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            prob = torch.sigmoid(model(x_test).squeeze(-1)).tolist()
        for idx, p in zip(test_idx, prob, strict=True):
            r = rows[idx]
            preds.append({
                "target": target,
                "model_type": "activation_pca_logistic",
                "fold": fold_idx,
                "example_id": r["example_id"],
                "trajectory_id": r["trajectory_id"],
                "validation_group": r["validation_group"],
                "reasoning_step_idx": int(r["reasoning_step_idx"]),
                "outcome": int(r["outcome"]),
                "probability": float(p),
            })
    return preds


def build_supervised_activation_monitor(exp1: Path, activation_index_path: Path, out: Path) -> list[dict[str, Any]]:
    positions = pd.read_csv(exp1 / "position_rows.csv")
    events = pd.read_csv(exp1 / "event_rows.csv")
    event_lookup: dict[tuple[str, int], set[str]] = defaultdict(set)
    for r in events.itertuples():
        event_lookup[(r.example_id, int(r.reasoning_step_idx))].add(r.event_type)
    for target, event_types in EVENT_TYPES.items():
        positions[target] = [int(bool(event_lookup.get((r.example_id, int(r.reasoning_step_idx)), set()) & event_types)) for r in positions.itertuples()]
    scalar_features = ["reasoning_progress", "action_confidence", "action_entropy_bits", "state_belief_entropy_bits"]
    predictions: list[dict[str, Any]] = []
    for target in EVENT_TYPES:
        usable = positions[positions.reasoning_step_idx > 0].copy()
        base_rows = []
        for r in usable.itertuples():
            base_rows.append({
                "example_id": r.example_id,
                "trajectory_id": r.trajectory_id,
                "validation_group": getattr(r, "matched_pair_id", r.trajectory_id),
                "reasoning_step_idx": int(r.reasoning_step_idx),
                "outcome": int(getattr(r, target)),
                **{f: safe_float(getattr(r, f), 0.0) for f in scalar_features},
            })
        for model_name, features in {
            "prevalence_baseline": [],
            "reasoning_progress": ["reasoning_progress"],
            "scalar_readouts": scalar_features,
        }.items():
            preds = cross_validated_predictions(base_rows, model_name, features)
            for p in preds:
                p["target"] = target
                p["layer"] = ""
                p["representation"] = "scalar"
            predictions.extend(preds)
    # Activation PCA models. Keep all requested layers and sentence mean/final, but
    # process one matrix at a time to avoid high memory use.
    for layer in (8, 15, 23):
        for representation in ("sentence_mean", "sentence_final"):
            frame, matrix = load_activation_matrix(exp1, activation_index_path, layer=layer, representation=representation)
            frame = add_metadata_from_positions(frame, positions)
            for target, event_types in EVENT_TYPES.items():
                frame[target] = [int(bool(event_lookup.get((r.example_id, int(r.reasoning_step_idx)), set()) & event_types)) for r in frame.itertuples()]
                preds = pca_logistic_predictions(frame, matrix, target, seed=42 + layer)
                for p in preds:
                    p["layer"] = layer
                    p["representation"] = representation.replace("sentence_", "sentence ")
                    p["model_type"] = f"activation_pca_{layer}_{representation}"
                predictions.extend(preds)
    write_csv(out / "supervised_activation_monitor_predictions.csv", predictions)
    summary = summarize_predictions(predictions, family_key="target")
    write_csv(out / "supervised_activation_monitor_summary.csv", summary)

    comparison_rows: list[dict[str, Any]] = []
    lines = [
        "# Supervised Activation Monitor Baseline Comparison",
        "",
        "This supervised monitor uses grouped cross-validation by matched pair or trajectory. PCA is fit inside each training fold before ridge logistic regression. The current result is predictive and not causal.",
        "",
        "| Target | Best activation model | Activation AUROC | Reasoning-progress AUROC | Scalar-readout AUROC | Interpretation |",
        "|---|---|---:|---:|---:|---|",
    ]
    for target in EVENT_TYPES:
        target_rows = [r for r in summary if r["target"] == target]
        activation_rows = [r for r in target_rows if str(r["model_type"]).startswith("activation_pca")]
        progress_rows = [r for r in target_rows if r["model_type"] == "reasoning_progress"]
        scalar_rows = [r for r in target_rows if r["model_type"] == "scalar_readouts"]
        if not activation_rows:
            continue
        best_activation = max(activation_rows, key=lambda r: -1 if r["roc_auc"] is None else float(r["roc_auc"]))
        progress_auc = float(progress_rows[0]["roc_auc"]) if progress_rows else float("nan")
        scalar_auc = float(scalar_rows[0]["roc_auc"]) if scalar_rows else float("nan")
        diff_progress = float(best_activation["roc_auc"]) - progress_auc
        diff_scalar = float(best_activation["roc_auc"]) - scalar_auc
        if diff_progress >= 0.05 and diff_scalar < 0:
            interp = "beats reasoning progress, but not scalar readouts"
        elif diff_progress >= 0.05:
            interp = "passes +0.05 over reasoning progress"
        else:
            interp = "does not pass +0.05 over reasoning progress"
        comparison_rows.append(
            {
                "target": target,
                "best_activation_model": best_activation["model_type"],
                "activation_auc": best_activation["roc_auc"],
                "reasoning_progress_auc": progress_auc,
                "scalar_readout_auc": scalar_auc,
                "activation_minus_progress_auc": diff_progress,
                "activation_minus_scalar_auc": diff_scalar,
                "interpretation": interp,
            }
        )
        lines.append(
            f"| {target.replace('_', ' ')} | {best_activation['model_type']} | "
            f"{float(best_activation['roc_auc']):.3f} | {progress_auc:.3f} | {scalar_auc:.3f} | {interp} |"
        )
    (out / "activation_monitor_baseline_comparison.md").write_text("\n".join(lines) + "\n")
    write_csv(out / "activation_monitor_baseline_comparisons.csv", comparison_rows)

    fig_rows = [r for r in summary if str(r["model_type"]).startswith("activation_pca")]
    if fig_rows:
        fig_df = pd.DataFrame(fig_rows)
        best = fig_df.sort_values("roc_auc", ascending=False).groupby("target", as_index=False).first()
        fig, ax = plt.subplots(figsize=(9.4, 4.8))
        y_labels = [
            f"{row.target.replace('_', ' ')}\n"
            f"n={int(row.n_rows):,}, events={int(row.n_positive):,}"
            for row in best.itertuples()
        ]
        y_pos = np.arange(len(best))
        auc_values = best["roc_auc"].astype(float).to_numpy()
        ax.barh(y_pos, auc_values, color=BLUE)
        ax.set_yticks(y_pos, labels=y_labels)
        ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
        for y_i, value in zip(y_pos, auc_values, strict=True):
            ax.text(
                min(value + 0.01, 0.98),
                y_i,
                f"{value:.3f}",
                va="center",
                fontsize=9,
                color=DARK,
            )
        ax.set_xlim(0.45, 0.75)
        ax.set_xlabel("Held-out AUROC for event prediction")
        ax.set_ylabel("Action event")
        ax.set_title("Best Activation-Only Monitor by Event")
        ax.grid(axis="x", color=GRID)
        fig.tight_layout()
        (out / "figs").mkdir(exist_ok=True)
        fig.savefig(out / "figs" / "supervised_activation_monitor_auc.png", dpi=220)
        plt.close(fig)
    return summary


def total_variation(left: dict[str, float], right: dict[str, float]) -> float:
    labels = set(left) | set(right)
    return 0.5 * sum(abs(left.get(label, 0.0) - right.get(label, 0.0)) for label in labels)


def build_post_commitment(exp1: Path, out: Path) -> list[dict[str, Any]]:
    positions = pd.read_csv(exp1 / "position_rows.csv")
    beliefs = pd.read_csv(exp1 / "belief_rows.csv")
    if "reasoning_step_idx" not in beliefs.columns:
        beliefs["reasoning_step_idx"] = beliefs["position_index"]
    rows: list[dict[str, Any]] = []
    for example_id, group in positions.groupby("example_id"):
        ordered = group.sort_values("reasoning_step_idx").reset_index(drop=True)
        commit = ordered[ordered.commitment_onset == True]
        if commit.empty:
            continue
        commit_idx = int(commit.iloc[0].reasoning_step_idx)
        total = int(ordered.reasoning_step_idx.max())
        post = ordered[ordered.reasoning_step_idx >= commit_idx].copy()
        commit_probs = parse_probs(str(commit.iloc[0].action_probabilities_json))
        final_action = str(commit.iloc[0].final_full_trace_action)
        tvs = [total_variation(commit_probs, parse_probs(str(r.action_probabilities_json))) for r in post.itertuples()]
        b = beliefs[(beliefs.example_id == example_id) & (beliefs.reasoning_step_idx >= commit_idx)].copy()
        current = b[b.question_id.isin(CURRENT_STATE_IDS)]
        transition = b[b.question_id.str.startswith(TRANSITION_PREFIXES)]
        rows.append({
            "example_id": example_id,
            "trajectory_id": commit.iloc[0].trajectory_id,
            "validation_group": commit.iloc[0].matched_pair_id if "matched_pair_id" in commit.columns else commit.iloc[0].trajectory_id,
            "trajectory_class": commit.iloc[0].trajectory_class,
            "primary_step_failure_mode": commit.iloc[0].primary_step_failure_mode,
            "commitment_step": commit_idx,
            "total_sentence_prefixes": total,
            "commitment_sentence_fraction": commit_idx / max(1, total),
            "commitment_progress": safe_float(commit.iloc[0].reasoning_progress),
            "post_commitment_sentences": max(0, total - commit_idx),
            "post_commitment_fraction": max(0.0, (total - commit_idx) / max(1, total)),
            "action_confidence_at_commitment": safe_float(commit.iloc[0].action_confidence),
            "action_entropy_at_commitment": safe_float(commit.iloc[0].action_entropy_bits),
            "mean_post_commitment_action_confidence": float(post.action_confidence.mean()),
            "mean_post_commitment_action_entropy": float(post.action_entropy_bits.mean()),
            "max_post_commitment_action_distribution_drift": float(max(tvs) if tvs else 0.0),
            "mean_post_commitment_action_distribution_drift": float(np.mean(tvs) if tvs else 0.0),
            "final_action_probability_at_commitment": commit_probs.get(final_action, float("nan")),
            "mean_post_current_state_entropy": float(current.entropy_bits.astype(float).mean()) if not current.empty else float("nan"),
            "mean_post_transition_belief_entropy": float(transition.entropy_bits.astype(float).mean()) if not transition.empty else float("nan"),
            "mean_post_current_state_error": float((current.belief_is_error == True).mean()) if not current.empty else float("nan"),
            "mean_post_transition_belief_error": float((transition.belief_is_error == True).mean()) if not transition.empty else float("nan"),
            "transition_belief_scope": "available_chosen_action_consequence_probes_not_allocentric",
        })
    write_csv(out / "commitment_tail_rows.csv", rows)

    model_specs = {
        "action_confidence_only": ["action_confidence_at_commitment"],
        "current_state_uncertainty": ["action_confidence_at_commitment", "mean_post_current_state_entropy", "mean_post_current_state_error"],
        "transition_belief_uncertainty": ["action_confidence_at_commitment", "mean_post_transition_belief_entropy", "mean_post_transition_belief_error"],
        "combined": ["action_confidence_at_commitment", "mean_post_current_state_entropy", "mean_post_current_state_error", "mean_post_transition_belief_entropy", "mean_post_transition_belief_error"],
    }
    model_rows: list[dict[str, Any]] = []
    if rows:
        df = pd.DataFrame(rows)
        y = df["post_commitment_fraction"].astype(float).to_numpy()
        for name, features in model_specs.items():
            x = df[features].astype(float).fillna(df[features].astype(float).mean()).to_numpy()
            pred = np.zeros_like(y, dtype=float)
            for i in range(len(df)):
                train = np.ones(len(df), dtype=bool)
                train[i] = False
                x_train, y_train = x[train], y[train]
                mean_x = x_train.mean(0)
                std_x = np.maximum(x_train.std(0), 1e-5)
                zx = (x_train - mean_x) / std_x
                ztest = (x[[i]] - mean_x) / std_x
                lam = 1.0
                beta = np.linalg.solve(zx.T @ zx + lam * np.eye(zx.shape[1]), zx.T @ y_train)
                intercept = y_train.mean() - (zx.mean(0) @ beta)
                pred[i] = float((intercept + ztest @ beta).item())
            ss_res = float(((y - pred) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            model_rows.append({
                "model_type": name,
                "n_states": len(df),
                "outcome": "post_commitment_fraction",
                "mae": float(np.abs(y - pred).mean()),
                "rmse": float(np.sqrt(((y - pred) ** 2).mean())),
                "cross_validated_r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                "features": ",".join(features),
            })
    write_csv(out / "post_commitment_model_summary.csv", model_rows)

    lines = [
        "# Post-Commitment Reasoning",
        "",
        f"States with a commitment boundary: {len(rows)}.",
        "Commitment is retrospective: the first prefix where the recommended action equals the full-trace action and remains stable.",
        "Post-commitment reasoning is action-epiphenomenal when the action distribution changes little after commitment, but this does not imply that the text is useless for belief calibration.",
        "",
    ]
    if rows:
        df = pd.DataFrame(rows)
        df["state_group"] = np.where(
            df["trajectory_class"].astype(str).str.contains("suboptimal|failure", case=False, regex=True),
            "Suboptimal final action",
            "Optimal final action",
        )
        group_rows = []
        for group_name, group in df.groupby("state_group"):
            group_rows.append({
                "state_group": group_name,
                "states": len(group),
                "mean_commitment_sentence_fraction": float(group.commitment_sentence_fraction.mean()),
                "median_commitment_sentence_fraction": float(group.commitment_sentence_fraction.median()),
                "mean_post_commitment_sentences": float(group.post_commitment_sentences.mean()),
                "median_post_commitment_sentences": float(group.post_commitment_sentences.median()),
                "mean_post_commitment_fraction": float(group.post_commitment_fraction.mean()),
            })
        write_csv(out / "commitment_tail_group_summary.csv", group_rows)
        early_cut = float(df.commitment_sentence_fraction.median())
        long_cut = float(df.post_commitment_sentences.median())
        df["commitment_timing_group"] = np.where(
            df.commitment_sentence_fraction <= early_cut,
            "earlier commitment",
            "later commitment",
        )
        df["post_commitment_length_group"] = np.where(
            df.post_commitment_sentences >= long_cut,
            "longer post-commitment reasoning",
            "shorter post-commitment reasoning",
        )
        quadrant_rows = []
        for (timing, length), group in df.groupby(["commitment_timing_group", "post_commitment_length_group"]):
            failures = group["state_group"].eq("Suboptimal final action")
            quadrant_rows.append({
                "commitment_timing_group": timing,
                "post_commitment_length_group": length,
                "states": len(group),
                "suboptimal_final_action_states": int(failures.sum()),
                "suboptimal_final_action_rate": float(failures.mean()),
                "median_cut_commitment_sentence_fraction": early_cut,
                "median_cut_post_commitment_sentences": long_cut,
            })
        write_csv(out / "commitment_tail_group_failure_rates.csv", quadrant_rows)
        lines.extend([
            f"- Mean post-commitment tail: {df.post_commitment_sentences.mean():.1f} sentences ({df.post_commitment_fraction.mean():.3f} of the trace).",
            f"- Mean max post-commitment action-distribution drift: {df.max_post_commitment_action_distribution_drift.mean():.3f} total variation.",
            f"- Mean post-commitment current-state entropy: {df.mean_post_current_state_entropy.mean():.3f} bits.",
            f"- Mean post-commitment transition-belief entropy: {df.mean_post_transition_belief_entropy.mean():.3f} bits.",
        ])
    lines.extend([
        "",
        "## Predicting Post-Commitment Length",
        "",
        "| Model | MAE | RMSE | Cross-validated R2 |",
        "|---|---:|---:|---:|",
    ])
    for r in model_rows:
        lines.append(f"| {r['model_type'].replace('_', ' ')} | {float(r['mae']):.3f} | {float(r['rmse']):.3f} | {float(r['cross_validated_r2']):.3f} |")
    (out / "post_commitment_reasoning_report.md").write_text("\n".join(lines) + "\n")

    if rows:
        df = pd.DataFrame(rows)
        df["state_group"] = np.where(
            df["trajectory_class"].astype(str).str.contains("suboptimal|failure", case=False, regex=True),
            "Suboptimal final action",
            "Optimal final action",
        )
        (out / "figs").mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        colors = {
            "Optimal final action": BLUE,
            "Suboptimal final action": "#0B3C5D",
        }
        for group_name, group in df.groupby("state_group"):
            ax.scatter(
                group.commitment_sentence_fraction,
                group.post_commitment_sentences,
                c=colors.get(group_name, BLUE),
                alpha=0.82,
                label=f"{group_name} (n={len(group)})",
            )
        ax.set_xlabel("Fraction of reasoning sentences before commitment")
        ax.set_ylabel("Sentences after commitment")
        ax.set_title("Commitment Boundary and Remaining Reasoning")
        ax.legend(frameon=False)
        ax.grid(color=GRID)
        fig.tight_layout()
        fig.savefig(out / "figs" / "commitment_boundary_tail_length.png", dpi=220)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
        panel_specs = [
            (
                "mean_post_current_state_entropy",
                "Current-state belief entropy (bits)",
                LIGHT_BLUE,
            ),
            (
                "mean_post_transition_belief_entropy",
                "Transition-belief entropy (bits)",
                DARK,
            ),
        ]
        for ax, (column, xlabel, color) in zip(axes, panel_specs, strict=True):
            x = df[column].astype(float)
            y = df.post_commitment_sentences.astype(float)
            ax.scatter(x, y, c=color, alpha=0.82)
            mask = np.isfinite(x) & np.isfinite(y)
            if int(mask.sum()) >= 2 and float(x[mask].std()) > 0:
                slope, intercept = np.polyfit(x[mask], y[mask], 1)
                xs = np.linspace(float(x[mask].min()), float(x[mask].max()), 50)
                ax.plot(xs, intercept + slope * xs, color="#243746", linewidth=1)
            ax.set_xlabel(xlabel)
            ax.grid(color=GRID)
            ax.set_title(f"n={len(df)} states")
        axes[0].set_ylabel("Sentences after commitment")
        fig.suptitle("Post-Commitment Reasoning and Belief Uncertainty")
        fig.tight_layout()
        fig.savefig(out / "figs" / "post_commitment_length_vs_uncertainty.png", dpi=220)
        plt.close(fig)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp1-dir", default="outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1")
    parser.add_argument("--output-dir", default="outputs/hypothesis_tests/transition_activation_commitment_v1")
    parser.add_argument("--activation-index", default="outputs/activation_collection/gpt_oss_20b_boundary_v1/activation_index.parquet")
    args = parser.parse_args()
    exp1 = Path(args.exp1_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    transition_rows, transition_summary = build_transition_belief_rows(exp1, out)
    write_allocentric_probe_design(exp1, out)
    activation_summary = build_supervised_activation_monitor(exp1, Path(args.activation_index), out)
    tail_rows = build_post_commitment(exp1, out)
    allocentric_manifest_path = out / "allocentric_readouts" / "run_manifest.json"
    if allocentric_manifest_path.exists():
        allocentric_manifest = json.loads(allocentric_manifest_path.read_text())
        allocentric_status: str | dict[str, Any] = {
            "status": allocentric_manifest.get("status", "unknown"),
            "path": str(
                out
                / "allocentric_readouts"
                / "allocentric_transition_belief_rows.csv"
            ),
            "positions": allocentric_manifest.get("positions_in_checkpoint"),
            "rows": allocentric_manifest.get("rows"),
        }
    else:
        allocentric_status = "not_collected_in_this_analysis; design file emitted"
    manifest = {
        "status": "completed",
        "exp1_dir": str(exp1),
        "activation_index": str(args.activation_index),
        "transition_belief_rows": len(transition_rows),
        "transition_model_rows": len(transition_summary),
        "activation_model_rows": len(activation_summary),
        "commitment_tail_rows": len(tail_rows),
        "allocentric_probe_readouts": allocentric_status,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
