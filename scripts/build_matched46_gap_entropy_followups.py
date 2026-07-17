#!/usr/bin/env python3
"""Follow-up figures/tables for matched-46 entropy and belief-action gap analyses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

DIRECTION_BY_ACTION = {"UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right"}
EVENT_TYPES = {
    "action_change": {"action_change"},
    "optimal_to_suboptimal": {"transient_optimal_to_suboptimal", "sustained_optimal_to_suboptimal"},
    "suboptimal_to_optimal": {"suboptimal_to_optimal"},
    "commitment_onset": {"commitment_onset"},
}
BLUE = "#1769aa"
LIGHT = "#9bd0f5"
DARK = "#0b3c5d"
GRID = "#e5eef5"


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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except Exception:
        return default


def auc(labels: list[int], scores: list[float]) -> float | None:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return None
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(r["outcome"]) for r in rows]
    probs = [min(max(float(r["probability"]), 1e-6), 1 - 1e-6) for r in rows]
    return {
        "n_rows": len(rows),
        "n_positive": sum(labels),
        "positive_rate": sum(labels) / len(labels),
        "roc_auc": auc(labels, probs),
        "brier_score": float(np.mean([(p - y) ** 2 for p, y in zip(probs, labels, strict=True)])),
        "log_loss": float(-np.mean([y * math.log(p) + (1 - y) * math.log(1 - p) for p, y in zip(probs, labels, strict=True)])),
    }


def grouped_folds(rows: list[dict[str, Any]], *, n_folds: int = 5, seed: int = 42) -> list[set[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["validation_group"]].append(row)
    groups = list(grouped)
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda g: (-sum(int(r["outcome"]) for r in grouped[g]), -len(grouped[g])))
    folds = [set() for _ in range(min(n_folds, len(groups)))]
    pos = [0] * len(folds)
    total = [0] * len(folds)
    for group in groups:
        idx = min(range(len(folds)), key=lambda i: (pos[i], total[i]))
        folds[idx].add(group)
        pos[idx] += sum(int(r["outcome"]) for r in grouped[group])
        total[idx] += len(grouped[group])
    return folds


def fit_predict(train: list[dict[str, Any]], test: list[dict[str, Any]], features: list[str], *, seed: int) -> list[float]:
    if not features:
        p = sum(int(r["outcome"]) for r in train) / len(train)
        return [float(p)] * len(test)
    torch.manual_seed(seed)
    x_train = torch.tensor([[safe_float(r.get(f)) for f in features] for r in train], dtype=torch.float32)
    y_train = torch.tensor([int(r["outcome"]) for r in train], dtype=torch.float32)
    x_test = torch.tensor([[safe_float(r.get(f)) for f in features] for r in test], dtype=torch.float32)
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


def cross_validate(rows: list[dict[str, Any]], model_type: str, features: list[str]) -> list[dict[str, Any]]:
    if len({int(r["outcome"]) for r in rows}) < 2:
        return []
    output: list[dict[str, Any]] = []
    for fold_idx, test_groups in enumerate(grouped_folds(rows)):
        train = [r for r in rows if r["validation_group"] not in test_groups]
        test = [r for r in rows if r["validation_group"] in test_groups]
        if len({int(r["outcome"]) for r in train}) < 2:
            continue
        probs = fit_predict(train, test, features, seed=42 + fold_idx)
        for row, prob in zip(test, probs, strict=True):
            output.append({
                "model_type": model_type,
                "fold": fold_idx,
                "example_id": row["example_id"],
                "trajectory_id": row["trajectory_id"],
                "validation_group": row["validation_group"],
                "reasoning_step_idx": row["reasoning_step_idx"],
                "outcome": row["outcome"],
                "probability": prob,
            })
    return output


def build_rows(root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    positions = pd.read_csv(root / "position_rows.csv")
    beliefs = pd.read_csv(root / "belief_rows.csv")
    if "reasoning_step_idx" not in beliefs.columns:
        beliefs["reasoning_step_idx"] = beliefs["position_index"]
    belief_map = {(r.example_id, int(r.reasoning_step_idx), r.question_id): r for r in beliefs.itertuples()}
    event_map: dict[tuple[str, int], set[str]] = defaultdict(set)
    for r in pd.read_csv(root / "event_rows.csv").itertuples():
        event_map[(r.example_id, int(r.reasoning_step_idx))].add(r.event_type)
    rows: list[dict[str, Any]] = []
    for example_id, group in positions.groupby("example_id"):
        ordered = group.sort_values("reasoning_step_idx").reset_index(drop=True)
        for i in range(len(ordered) - 1):
            cur = ordered.iloc[i]
            nxt = ordered.iloc[i + 1]
            direction = DIRECTION_BY_ACTION.get(str(cur.action_label), "")
            chosen_wall = belief_map.get((example_id, int(cur.reasoning_step_idx), f"wall_{direction}"))
            chosen_hit = belief_map.get((example_id, int(cur.reasoning_step_idx), f"hit_wall_after_{direction}"))
            wall_reports_blocked = int(getattr(chosen_wall, "answer_key", "") == "yes") if chosen_wall is not None else 0
            hit_reports_blocked = int(getattr(chosen_hit, "answer_key", "") == "yes") if chosen_hit is not None else 0
            events_next = event_map.get((example_id, int(nxt.reasoning_step_idx)), set())
            rows.append({
                "example_id": example_id,
                "trajectory_id": cur.trajectory_id,
                "validation_group": cur.matched_pair_id if isinstance(cur.matched_pair_id, str) else cur.trajectory_id,
                "reasoning_step_idx": int(cur.reasoning_step_idx),
                "next_reasoning_step_idx": int(nxt.reasoning_step_idx),
                "matched_role": cur.matched_role,
                "trajectory_class": cur.trajectory_class,
                "reasoning_progress": safe_float(cur.reasoning_progress),
                "action_confidence": safe_float(cur.action_confidence),
                "state_belief_entropy_bits": safe_float(cur.state_belief_entropy_bits),
                "primary_belief_error_rate": safe_float(cur.primary_belief_error_rate),
                "chosen_wall_reports_blocked": wall_reports_blocked,
                "chosen_hit_reports_blocked": hit_reports_blocked,
                "belief_action_conflict": int(wall_reports_blocked or hit_reports_blocked),
                "action_change": int("action_change" in events_next),
                "optimal_to_suboptimal": int(bool({"transient_optimal_to_suboptimal", "sustained_optimal_to_suboptimal"} & events_next)),
                "suboptimal_to_optimal": int("suboptimal_to_optimal" in events_next),
                "commitment_onset": int("commitment_onset" in events_next),
            })
    return positions, rows


def plot_entropy_split(positions: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 10})
    bins = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    summary_rows = []
    for label, color in (("failure", DARK), ("control", LIGHT)):
        subset = positions[positions.matched_role == label]
        xs, ys = [], []
        for left, right in zip(bins[:-1], bins[1:]):
            values = subset[(subset.reasoning_progress >= left) & (subset.reasoning_progress <= right)].state_belief_entropy_bits.astype(float)
            if len(values):
                xs.append((left + right) / 2)
                ys.append(float(values.mean()))
                summary_rows.append({"matched_role": label, "bin_left": left, "bin_right": right, "n_prefixes": len(values), "mean_state_belief_entropy_bits": float(values.mean())})
        ax.plot(xs, ys, marker="o", color=color, label=f"{label} (states={subset.example_id.nunique()}, prefixes={len(subset)})")
    ax.set_xlabel("Fraction of reasoning characters revealed")
    ax.set_ylabel("Mean wall, key, and door belief entropy (bits)")
    ax.set_title("State-belief uncertainty by failure status")
    ax.grid(color=GRID)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "figs" / "state_belief_entropy_by_failure_status.png", dpi=220)
    plt.close(fig)
    write_csv(out / "state_belief_entropy_by_failure_status.csv", summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1")
    args = parser.parse_args()
    root = Path(args.run_dir)
    (root / "figs").mkdir(exist_ok=True)
    positions, rows = build_rows(root)
    plot_entropy_split(positions, root)
    write_csv(root / "belief_action_gap_rows.csv", rows)

    gap_summary = []
    df = pd.DataFrame(rows)
    for group_name, group in [("all", df), *[(str(k), v) for k, v in df.groupby("matched_role")]]:
        gap_summary.append({
            "group": group_name,
            "n_prefixes": len(group),
            "n_states": group.example_id.nunique(),
            "belief_action_conflict_rate": float(group.belief_action_conflict.mean()),
            "chosen_wall_reports_blocked_rate": float(group.chosen_wall_reports_blocked.mean()),
            "chosen_hit_reports_blocked_rate": float(group.chosen_hit_reports_blocked.mean()),
            "mean_primary_belief_error_rate": float(group.primary_belief_error_rate.mean()),
            "mean_state_belief_entropy_bits": float(group.state_belief_entropy_bits.mean()),
        })
    write_csv(root / "belief_action_gap_summary.csv", gap_summary)

    feature_sets = {
        "prevalence_baseline": [],
        "progress_baseline": ["reasoning_progress"],
        "belief_error_entropy": ["reasoning_progress", "primary_belief_error_rate", "state_belief_entropy_bits"],
        "belief_action_conflict": ["reasoning_progress", "belief_action_conflict", "chosen_wall_reports_blocked", "chosen_hit_reports_blocked"],
        "combined_gap_belief": ["reasoning_progress", "primary_belief_error_rate", "state_belief_entropy_bits", "belief_action_conflict", "chosen_wall_reports_blocked", "chosen_hit_reports_blocked"],
    }
    prediction_rows = []
    summary_rows = []
    for target in EVENT_TYPES:
        target_rows = [{**r, "outcome": int(r[target])} for r in rows]
        for model_type, features in feature_sets.items():
            preds = cross_validate(target_rows, model_type, features)
            for pred in preds:
                pred["target"] = target
            prediction_rows.extend(preds)
            if preds:
                m = metrics(preds)
                summary_rows.append({"target": target, "model_type": model_type, "n_trajectories": len({p['trajectory_id'] for p in preds}), **m})
    write_csv(root / "belief_action_gap_event_predictions.csv", prediction_rows)
    write_csv(root / "belief_action_gap_event_summary.csv", summary_rows)

    lines = [
        "# Belief-Action Gap Follow-Up",
        "",
        "Belief-action conflict is defined as recommending an action while the model's own elicited belief says that action is blocked: either the chosen-direction wall probe answers yes, or the chosen-action hit-wall consequence probe answers yes. This is a simple empirical gap and does not assume a full planner.",
        "",
        "## Descriptive Summary",
        "",
        "| Group | States | Prefixes | Conflict rate | Mean belief error | Mean belief entropy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in gap_summary:
        lines.append(f"| {row['group']} | {row['n_states']} | {row['n_prefixes']} | {row['belief_action_conflict_rate']:.3f} | {row['mean_primary_belief_error_rate']:.3f} | {row['mean_state_belief_entropy_bits']:.3f} |")
    lines.extend([
        "",
        "## Predictive Summary",
        "",
        "| Target | Best learned model | AUROC | Events |",
        "|---|---|---:|---:|",
    ])
    for target in EVENT_TYPES:
        candidates = [r for r in summary_rows if r["target"] == target and r["model_type"] != "prevalence_baseline"]
        if candidates:
            best = max(candidates, key=lambda r: -1 if r["roc_auc"] is None else float(r["roc_auc"]))
            lines.append(f"| {target.replace('_', ' ')} | {best['model_type'].replace('_', ' ')} | {float(best['roc_auc']):.3f} | {best['n_positive']} |")
    (root / "BELIEF_ACTION_GAP_FOLLOWUP.md").write_text("\n".join(lines) + "\n")
    print(root / "BELIEF_ACTION_GAP_FOLLOWUP.md")


if __name__ == "__main__":
    main()
