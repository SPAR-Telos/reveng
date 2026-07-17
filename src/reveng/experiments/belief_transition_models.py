"""Trajectory-held-out models of belief errors and action-optimality transitions."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from scipy.stats import rankdata

GLOBAL_BELIEFS = ("wall_left", "wall_right", "wall_up", "wall_down", "has_key")
DIRECTION_BY_ACTION = {
    "LEFT": "left",
    "RIGHT": "right",
    "UP": "up",
    "DOWN": "down",
}
MODEL_FEATURES = {
    "prevalence_baseline": ("intercept_only",),
    "baseline": ("reasoning_progress",),
    "global_beliefs": (
        "reasoning_progress",
        *(f"error_{belief}" for belief in GLOBAL_BELIEFS),
    ),
    "belief_dynamics": (
        "reasoning_progress",
        "n_global_belief_changes",
        "n_global_error_onsets",
        "n_global_error_recoveries",
    ),
    "global_beliefs_and_dynamics": (
        "reasoning_progress",
        *(f"error_{belief}" for belief in GLOBAL_BELIEFS),
        "n_global_belief_changes",
        "n_global_error_onsets",
        "n_global_error_recoveries",
    ),
    "action_conditioned": (
        "reasoning_progress",
        "chosen_wall_error",
        "chosen_wall_reports_blocked",
        "chosen_effect_error",
        "chosen_effect_reports_hit",
    ),
    "combined": (
        "reasoning_progress",
        *(f"error_{belief}" for belief in GLOBAL_BELIEFS),
        "chosen_wall_error",
        "chosen_wall_reports_blocked",
        "chosen_effect_error",
        "chosen_effect_reports_hit",
    ),
    "belief_action_chain": (
        "reasoning_progress",
        *(f"error_{belief}" for belief in GLOBAL_BELIEFS),
        "chosen_wall_error",
        "chosen_wall_reports_blocked",
        "chosen_effect_error",
        "chosen_effect_reports_hit",
        "blocked_report*predicted_hit",
        "chosen_wall_error*chosen_effect_error",
    ),
}
DEFAULT_RUN_DIR = "data/behavioral_probes/reasoning_belief_action_balanced_v1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _as_optional_bool(value: str) -> bool | None:
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def _validation_groups(run_dir: Path, positions: list[dict[str, str]]) -> dict[str, str]:
    """Keep source trajectories and both members of each matched pair together."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    trajectory_by_example = {
        row["example_id"]: row["trajectory_id"]
        for row in positions
    }
    for trajectory_id in trajectory_by_example.values():
        find(f"trajectory:{trajectory_id}")

    config_path = run_dir / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        candidates_path = Path(config.get("candidate_rows_path", ""))
        if candidates_path.exists():
            for row in _read_csv(candidates_path):
                trajectory_id = trajectory_by_example.get(row.get("example_id", ""))
                matched_pair_id = row.get("matched_pair_id", "")
                if trajectory_id and matched_pair_id:
                    union(f"trajectory:{trajectory_id}", f"pair:{matched_pair_id}")

    return {
        trajectory_id: find(f"trajectory:{trajectory_id}")
        for trajectory_id in set(trajectory_by_example.values())
    }


def _build_model_rows(run_dir: Path) -> list[dict[str, Any]]:
    positions = _read_csv(run_dir / "position_rows.csv")
    beliefs = _read_csv(run_dir / "belief_rows.csv")
    validation_groups = _validation_groups(run_dir, positions)
    belief_lookup = {
        (row["example_id"], int(row["reasoning_step_idx"]), row["question_id"]): row
        for row in beliefs
        if row["question_id"] in GLOBAL_BELIEFS
        or row["question_id"].startswith("hit_wall_after_")
    }
    by_example: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in positions:
        by_example[row["example_id"]].append(row)
    rows: list[dict[str, Any]] = []
    for example_rows in by_example.values():
        ordered = sorted(example_rows, key=lambda row: int(row["reasoning_step_idx"]))
        for position_idx in range(1, len(ordered) - 1):
            previous = ordered[position_idx - 1]
            current = ordered[position_idx]
            following = ordered[position_idx + 1]
            current_optimal = _as_optional_bool(current["action_is_optimal"])
            next_optimal = _as_optional_bool(following["action_is_optimal"])
            if current_optimal is None or next_optimal is None:
                continue
            direction = DIRECTION_BY_ACTION.get(current["action_label"])
            if direction is None:
                continue
            features: dict[str, int] = {}
            belief_changes = 0
            error_onsets = 0
            error_recoveries = 0
            valid = True
            for question in GLOBAL_BELIEFS:
                belief = belief_lookup.get(
                    (current["example_id"], int(current["reasoning_step_idx"]), question)
                )
                previous_belief = belief_lookup.get(
                    (previous["example_id"], int(previous["reasoning_step_idx"]), question)
                )
                if (
                    belief is None
                    or belief["answer_valid"] != "True"
                    or previous_belief is None
                    or previous_belief["answer_valid"] != "True"
                ):
                    valid = False
                    break
                features[f"error_{question}"] = int(belief["belief_is_error"] == "True")
                belief_changes += int(belief["answer_key"] != previous_belief["answer_key"])
                error_onsets += int(
                    belief["belief_is_error"] == "True"
                    and previous_belief["belief_is_error"] == "False"
                )
                error_recoveries += int(
                    belief["belief_is_error"] == "False"
                    and previous_belief["belief_is_error"] == "True"
                )
            chosen_wall = belief_lookup.get(
                (
                    current["example_id"],
                    int(current["reasoning_step_idx"]),
                    f"wall_{direction}",
                )
            )
            chosen_effect = belief_lookup.get(
                (
                    current["example_id"],
                    int(current["reasoning_step_idx"]),
                    f"hit_wall_after_{direction}",
                )
            )
            if (
                chosen_wall is None
                or chosen_wall["answer_valid"] != "True"
                or chosen_effect is None
                or chosen_effect["answer_valid"] != "True"
            ):
                valid = False
            if not valid:
                continue
            features.update(
                {
                    "n_global_belief_changes": belief_changes,
                    "n_global_error_onsets": error_onsets,
                    "n_global_error_recoveries": error_recoveries,
                    "chosen_wall_error": int(chosen_wall["belief_is_error"] == "True"),
                    "chosen_wall_reports_blocked": int(chosen_wall["answer_key"] == "yes"),
                    "chosen_effect_error": int(chosen_effect["belief_is_error"] == "True"),
                    "chosen_effect_reports_hit": int(chosen_effect["answer_key"] == "yes"),
                }
            )
            features["blocked_report*predicted_hit"] = (
                features["chosen_wall_reports_blocked"] * features["chosen_effect_reports_hit"]
            )
            features["chosen_wall_error*chosen_effect_error"] = (
                features["chosen_wall_error"] * features["chosen_effect_error"]
            )
            rows.append(
                {
                    "example_id": current["example_id"],
                    "trajectory_id": current["trajectory_id"],
                    "validation_group": validation_groups[current["trajectory_id"]],
                    "reasoning_step_idx": int(current["reasoning_step_idx"]),
                    "intercept_only": 0,
                    "reasoning_progress": float(current["reasoning_progress"]),
                    "current_action_is_optimal": current_optimal,
                    "next_action_is_optimal": next_optimal,
                    **features,
                }
            )
    return rows


def _group_folds(rows: list[dict[str, Any]], n_folds: int, seed: int) -> list[set[str]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["validation_group"]].append(row)
    ordered = list(groups)
    rng.shuffle(ordered)
    ordered.sort(
        key=lambda group: (
            -sum(int(row["outcome"]) for row in groups[group]),
            -len(groups[group]),
        )
    )
    folds = [set() for _ in range(min(n_folds, len(ordered)))]
    fold_positive = [0] * len(folds)
    fold_total = [0] * len(folds)
    for group in ordered:
        target = min(range(len(folds)), key=lambda idx: (fold_positive[idx], fold_total[idx]))
        folds[target].add(group)
        fold_positive[target] += sum(int(row["outcome"]) for row in groups[group])
        fold_total[target] += len(groups[group])
    return folds


def _feature_names(model_type: str) -> list[str]:
    try:
        return list(MODEL_FEATURES[model_type])
    except KeyError as exc:
        raise ValueError(f"Unknown model_type: {model_type}") from exc


def _vector(row: dict[str, Any], model_type: str) -> list[float]:
    return [float(row[name]) for name in _feature_names(model_type)]


def _fit_predict(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    model_type: str,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[list[float], torch.nn.Linear]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    x_train = torch.tensor([_vector(row, model_type) for row in train_rows], dtype=torch.float32)
    y_train = torch.tensor([row["outcome"] for row in train_rows], dtype=torch.float32)
    x_test = torch.tensor([_vector(row, model_type) for row in test_rows], dtype=torch.float32)
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0).clamp_min(1e-5)
    x_train = ((x_train - mean) / std).to(device)
    x_test = ((x_test - mean) / std).to(device)
    y_train = y_train.to(device)
    model = torch.nn.Linear(x_train.shape[1], 1).to(device)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=learning_rate,
        max_iter=epochs,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train).squeeze(-1)
        # Unweighted logistic loss preserves probabilistic interpretation for
        # held-out Brier score and log loss. AUROC handles class imbalance.
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        loss = loss + weight_decay * model.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x_test).squeeze(-1)).cpu().tolist()
    return [float(value) for value in probabilities], model.cpu()


def _auc(labels: list[int], probabilities: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(probabilities)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label == 1)
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def _metrics(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
    clipped = [min(max(value, 1e-6), 1 - 1e-6) for value in probabilities]
    return {
        "n_rows": len(labels),
        "n_positive": sum(labels),
        "positive_rate": sum(labels) / len(labels),
        "roc_auc": _auc(labels, probabilities),
        "brier_score": sum((prob - label) ** 2 for prob, label in zip(probabilities, labels, strict=True))
        / len(labels),
        "log_loss": -sum(
            label * math.log(prob) + (1 - label) * math.log(1 - prob)
            for prob, label in zip(clipped, labels, strict=True)
        )
        / len(labels),
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _grouped_bootstrap_intervals(
    prediction_rows: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | None]:
    """Bootstrap complete validation groups, preserving repeated positions."""
    if n_bootstrap <= 0:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[row["validation_group"]].append(row)
    group_ids = sorted(grouped)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_bootstrap):
        sampled = [rng.choice(group_ids) for _ in group_ids]
        rows = [row for group_id in sampled for row in grouped[group_id]]
        metrics = _metrics(
            [int(row["outcome"]) for row in rows],
            [float(row["probability"]) for row in rows],
        )
        for metric in ("roc_auc", "brier_score", "log_loss"):
            value = metrics[metric]
            if value is not None:
                samples[metric].append(float(value))
    intervals: dict[str, float | None] = {}
    for metric in ("roc_auc", "brier_score", "log_loss"):
        intervals[f"{metric}_ci_low"] = _percentile(samples[metric], 0.025)
        intervals[f"{metric}_ci_high"] = _percentile(samples[metric], 0.975)
    return intervals


def _grouped_bootstrap_comparison(
    model_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    key = lambda row: (row["example_id"], row["reasoning_step_idx"])
    reference = {key(row): row for row in reference_rows}
    paired = [(row, reference[key(row)]) for row in model_rows if key(row) in reference]
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in paired:
        grouped[pair[0]["validation_group"]].append(pair)
    group_ids = sorted(grouped)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_bootstrap):
        sampled = [rng.choice(group_ids) for _ in group_ids]
        pairs = [pair for group_id in sampled for pair in grouped[group_id]]
        labels = [int(pair[0]["outcome"]) for pair in pairs]
        model_metrics = _metrics(labels, [float(pair[0]["probability"]) for pair in pairs])
        reference_metrics = _metrics(labels, [float(pair[1]["probability"]) for pair in pairs])
        for metric in ("roc_auc", "brier_score", "log_loss"):
            model_value, reference_value = model_metrics[metric], reference_metrics[metric]
            if model_value is not None and reference_value is not None:
                samples[metric].append(float(model_value) - float(reference_value))
    row: dict[str, Any] = {"n_paired_rows": len(paired)}
    for metric in ("roc_auc", "brier_score", "log_loss"):
        values = samples[metric]
        row[f"{metric}_difference_ci_low"] = _percentile(values, 0.025)
        row[f"{metric}_difference_ci_high"] = _percentile(values, 0.975)
    return row


def _format_interval(row: dict[str, Any], metric: str) -> str:
    value = row.get(metric)
    low = row.get(f"{metric}_ci_low")
    high = row.get(f"{metric}_ci_high")
    if value is None:
        return "NA"
    if low is None or high is None:
        return f"{float(value):.3f}"
    return f"{float(value):.3f} [{float(low):.3f}, {float(high):.3f}]"


def _write_report(
    root: Path,
    summary_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    status: dict[str, Any],
) -> None:
    by_family = {
        family: [row for row in summary_rows if row["transition_family"] == family]
        for family in ("optimal_to_suboptimal", "suboptimal_to_optimal")
    }
    lines = [
        "# Belief-Action Transition Models",
        "",
        "These models test whether behavioral belief readouts at reasoning position `k` predict "
        "whether the recommended action changes optimality at position `k+1`. Evaluation holds "
        "out complete source trajectories and keeps exact-matched pairs in the same fold. "
        "Confidence intervals bootstrap those validation groups. The constant prevalence baseline "
        "is a descriptive null. The results are predictive associations, not causal effects.",
        "",
        "The primary analysis excludes `door_open` because its ground truth is constant in the "
        "analyzed cohort. The action-conditioned models test whether the model reports a wall in the "
        "direction it recommends moving and whether it predicts that move will hit a wall.",
        "",
        "## Results",
        "",
        "| Transition | Model | Trajectories | Events | AUROC [95% CI] | Brier [95% CI] | Log loss [95% CI] |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for family in by_family:
        for row in by_family[family]:
            lines.append(
                f"| {family.replace('_', ' ')} | {row['model_type'].replace('_', ' ')} | "
                f"{row['n_trajectories']} | {row['n_positive']} | {_format_interval(row, 'roc_auc')} | "
                f"{_format_interval(row, 'brier_score')} | {_format_interval(row, 'log_loss')} |"
            )
    loss_rows = [
        row
        for row in by_family["optimal_to_suboptimal"]
        if row["model_type"] != "prevalence_baseline"
    ]
    recovery_rows = [
        row
        for row in by_family["suboptimal_to_optimal"]
        if row["model_type"] != "prevalence_baseline"
    ]
    best_loss = max(loss_rows, key=lambda row: float(row["roc_auc"])) if loss_rows else None
    best_recovery = max(recovery_rows, key=lambda row: float(row["roc_auc"])) if recovery_rows else None
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"- For upcoming optimal-to-suboptimal transitions, the highest learned-model "
                f"held-out AUROC is "
                f"{float(best_loss['roc_auc']):.3f} from `{best_loss['model_type']}`."
                if best_loss
                else "- No optimal-to-suboptimal model was estimable."
            ),
            (
                f"- For recoveries, the highest learned-model held-out AUROC is "
                f"{float(best_recovery['roc_auc']):.3f} from `{best_recovery['model_type']}`."
                if best_recovery
                else "- No recovery model was estimable."
            ),
            "- If action-conditioned or belief-action-chain models do not outperform the baseline, "
            "the current behavioral probes do not yet provide evidence for a stable compositional "
            "belief chain explaining action changes.",
            "- The constant prevalence null has AUROC 0.500. Compare learned models against it as "
            "well as against the reasoning-progress baseline.",
            "- Confidence intervals are grouped-bootstrap intervals and can remain wide when few "
            "independent validation groups contain transition events.",
            "",
            "## Paired Comparisons With Reasoning Progress",
            "",
            "Differences below are model minus reasoning-progress baseline. Positive AUROC is "
            "better; negative Brier score and log loss are better.",
            "",
            "| Transition | Model | AUROC difference [95% CI] | Brier difference [95% CI] | Log loss difference [95% CI] |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in comparison_rows:
        if row["reference_type"] != "baseline":
            continue
        lines.append(
            f"| {row['transition_family'].replace('_', ' ')} | "
            f"{row['model_type'].replace('_', ' ')} | "
            f"{float(row['roc_auc_difference']):.3f} "
            f"[{float(row['roc_auc_difference_ci_low']):.3f}, "
            f"{float(row['roc_auc_difference_ci_high']):.3f}] | "
            f"{float(row['brier_score_difference']):.3f} "
            f"[{float(row['brier_score_difference_ci_low']):.3f}, "
            f"{float(row['brier_score_difference_ci_high']):.3f}] | "
            f"{float(row['log_loss_difference']):.3f} "
            f"[{float(row['log_loss_difference_ci_low']):.3f}, "
            f"{float(row['log_loss_difference_ci_high']):.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Model Definitions",
            "",
        ]
    )
    for model_type, features in status["model_feature_definitions"].items():
        lines.append(f"- `{model_type}`: {', '.join(features)}")
    (root / "TRANSITION_MODEL_SUMMARY.md").write_text("\n".join(lines) + "\n")


def run_belief_transition_models(
    *,
    run_dir: str = DEFAULT_RUN_DIR,
    n_folds: int = 5,
    epochs: int = 300,
    learning_rate: float = 1.0,
    weight_decay: float = 0.01,
    seed: int = 42,
    device: str = "cpu",
    n_bootstrap: int = 1000,
) -> None:
    """Compare group-held-out general and action-conditioned belief models."""
    root = Path(run_dir)
    all_rows = _build_model_rows(root)
    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for transition_family, current_optimal in (
        ("optimal_to_suboptimal", True),
        ("suboptimal_to_optimal", False),
    ):
        risk_rows = [
            {
                **row,
                "outcome": int(row["next_action_is_optimal"] != current_optimal),
            }
            for row in all_rows
            if row["current_action_is_optimal"] is current_optimal
        ]
        folds = _group_folds(risk_rows, n_folds=n_folds, seed=seed)
        for model_type in MODEL_FEATURES:
            family_predictions: list[dict[str, Any]] = []
            if model_type == "prevalence_baseline":
                null_probability = sum(row["outcome"] for row in risk_rows) / len(risk_rows)
                fold_predictions = [null_probability] * len(risk_rows)
                fold_rows = risk_rows
                fold_idx = -1
                for row, probability in zip(fold_rows, fold_predictions, strict=True):
                    family_predictions.append(
                        {
                            "transition_family": transition_family,
                            "model_type": model_type,
                            "fold": fold_idx,
                            "example_id": row["example_id"],
                            "trajectory_id": row["trajectory_id"],
                            "validation_group": row["validation_group"],
                            "reasoning_step_idx": row["reasoning_step_idx"],
                            "outcome": row["outcome"],
                            "probability": probability,
                        }
                    )
            else:
                for fold_idx, test_groups in enumerate(folds):
                    train = [row for row in risk_rows if row["validation_group"] not in test_groups]
                    test = [row for row in risk_rows if row["validation_group"] in test_groups]
                    if not train or not test or len({row["outcome"] for row in train}) < 2:
                        continue
                    probabilities, _model = _fit_predict(
                        train,
                        test,
                        model_type=model_type,
                        device=device,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        seed=seed + fold_idx,
                    )
                    for row, probability in zip(test, probabilities, strict=True):
                        family_predictions.append(
                            {
                                "transition_family": transition_family,
                                "model_type": model_type,
                                "fold": fold_idx,
                                "example_id": row["example_id"],
                                "trajectory_id": row["trajectory_id"],
                                "validation_group": row["validation_group"],
                                "reasoning_step_idx": row["reasoning_step_idx"],
                                "outcome": row["outcome"],
                                "probability": probability,
                            }
                        )
            prediction_rows.extend(family_predictions)
            if family_predictions:
                labels = [int(row["outcome"]) for row in family_predictions]
                probabilities = [float(row["probability"]) for row in family_predictions]
                summary_rows.append(
                    {
                        "transition_family": transition_family,
                        "model_type": model_type,
                        "n_trajectories": len({row["trajectory_id"] for row in family_predictions}),
                        **_metrics(labels, probabilities),
                        **_grouped_bootstrap_intervals(
                            family_predictions,
                            n_bootstrap=n_bootstrap,
                            seed=seed,
                        ),
                    }
                )
            if model_type != "prevalence_baseline" and len({row["outcome"] for row in risk_rows}) >= 2:
                _probabilities, full_model = _fit_predict(
                    risk_rows,
                    risk_rows[:1],
                    model_type=model_type,
                    device=device,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    seed=seed,
                )
                for name, coefficient in zip(
                    _feature_names(model_type),
                    full_model.weight.detach().flatten().tolist(),
                    strict=True,
                ):
                    coefficient_rows.append(
                        {
                            "transition_family": transition_family,
                            "model_type": model_type,
                            "feature": name,
                            "standardized_coefficient": coefficient,
                        }
                    )
    _write_csv(root / "transition_model_predictions.csv", prediction_rows)
    _write_csv(root / "transition_model_summary.csv", summary_rows)
    _write_csv(root / "transition_model_full_data_coefficients.csv", coefficient_rows)
    comparison_rows: list[dict[str, Any]] = []
    for transition_family in ("optimal_to_suboptimal", "suboptimal_to_optimal"):
        family_rows = [
            row for row in prediction_rows if row["transition_family"] == transition_family
        ]
        by_model = {
            model_type: [row for row in family_rows if row["model_type"] == model_type]
            for model_type in MODEL_FEATURES
        }
        summary_by_model = {
            row["model_type"]: row
            for row in summary_rows
            if row["transition_family"] == transition_family
        }
        for model_type in MODEL_FEATURES:
            if model_type == "prevalence_baseline" or not by_model[model_type]:
                continue
            for reference_type in ("prevalence_baseline", "baseline"):
                if model_type == reference_type or not by_model[reference_type]:
                    continue
                model_summary = summary_by_model[model_type]
                reference_summary = summary_by_model[reference_type]
                comparison_rows.append(
                    {
                        "transition_family": transition_family,
                        "model_type": model_type,
                        "reference_type": reference_type,
                        "roc_auc_difference": float(model_summary["roc_auc"])
                        - float(reference_summary["roc_auc"]),
                        "brier_score_difference": float(model_summary["brier_score"])
                        - float(reference_summary["brier_score"]),
                        "log_loss_difference": float(model_summary["log_loss"])
                        - float(reference_summary["log_loss"]),
                        **_grouped_bootstrap_comparison(
                            by_model[model_type],
                            by_model[reference_type],
                            n_bootstrap=n_bootstrap,
                            seed=seed,
                        ),
                    }
                )
    _write_csv(root / "transition_model_comparisons.csv", comparison_rows)
    status = {
        "status": "completed",
        "interpretation": "trajectory_and_matched_pair_grouped_predictive_not_causal",
        "n_complete_case_positions": len(all_rows),
        "n_trajectories": len({row["trajectory_id"] for row in all_rows}),
        "n_validation_groups": len({row["validation_group"] for row in all_rows}),
        "global_beliefs": list(GLOBAL_BELIEFS),
        "excluded_primary_beliefs": {
            "door_open": "Ground-truth door state is constant in the analyzed cohort, so errors are not "
            "informative state variation."
        },
        "models": list(MODEL_FEATURES),
        "prevalence_baseline_note": "Uses the cohort event rate as a constant descriptive null; "
        "all learned models hold out source trajectories and exact-matched pairs.",
        "model_feature_definitions": {
            model_type: list(features) for model_type, features in MODEL_FEATURES.items()
        },
        "parameters": {
            "optimizer": "LBFGS",
            "n_folds": n_folds,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "n_bootstrap": n_bootstrap,
        },
    }
    (root / "transition_model_status.json").write_text(json.dumps(status, indent=2) + "\n")
    _write_report(root, summary_rows, comparison_rows, status)


__all__ = ["run_belief_transition_models"]
