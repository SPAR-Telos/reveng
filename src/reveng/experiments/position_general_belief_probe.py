"""Grouped-state feasibility test for position-general belief probes."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

DEFAULT_ACTIVATION_ROWS = (
    "data/behavioral_probes/step_reasoning_drift_balanced_v1_corrected_boundary_windows/"
    "step_activation_rows.csv"
)
DEFAULT_CANDIDATES = (
    "data/behavioral_probes/trajectory_instances_recomputed_optimal/"
    "trajectory_selection_candidates.csv"
)
DEFAULT_OUTPUT_DIR = (
    "data/behavioral_probes/step_reasoning_drift_balanced_v1_corrected_boundary_windows/"
    "position_general_belief_probe_feasibility"
)
PRIMARY_BELIEFS = ("wall_left", "wall_right", "wall_up", "wall_down", "has_key", "door_open")
REPRESENTATIONS = ("last_token", "sentence_mean", "boundary_window_mean")


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


def _load_representation(row: dict[str, str], representation: str) -> torch.Tensor:
    if representation == "last_token":
        path = row["step_last_token_activation_path"]
        return torch.load(path, map_location="cpu", weights_only=True).to(torch.float32).flatten()
    if representation == "sentence_mean":
        path = row["step_mean_activation_path"]
        return torch.load(path, map_location="cpu", weights_only=True).to(torch.float32).flatten()
    if representation == "boundary_window_mean":
        path = row["step_boundary_window_activation_path"]
        tensor = torch.load(path, map_location="cpu", weights_only=True).to(torch.float32)
        return tensor.mean(dim=0).flatten()
    raise ValueError(f"Unsupported representation: {representation}")


def _stratified_group_folds(
    labels_by_group: dict[str, int],
    *,
    n_folds: int,
    seed: int,
) -> list[set[str]]:
    rng = random.Random(seed)
    by_label: dict[int, list[str]] = defaultdict(list)
    for group, label in labels_by_group.items():
        by_label[label].append(group)
    folds = [set() for _ in range(n_folds)]
    for groups in by_label.values():
        rng.shuffle(groups)
        for index, group in enumerate(groups):
            folds[index % n_folds].add(group)
    return folds


def _balanced_accuracy(labels: list[int], predictions: list[int]) -> float:
    recalls = []
    for label in (0, 1):
        indices = [index for index, value in enumerate(labels) if value == label]
        if indices:
            recalls.append(sum(predictions[index] == label for index in indices) / len(indices))
    return sum(recalls) / len(recalls) if recalls else 0.0


def _fit_linear_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    groups_train: list[str],
    x_test: torch.Tensor,
    *,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0).clamp_min(1e-5)
    train = ((x_train - mean) / std).to(device)
    test = ((x_test - mean) / std).to(device)
    labels = y_train.to(device)
    model = torch.nn.Linear(train.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    group_counts = Counter(groups_train)
    class_counts = Counter(int(value) for value in y_train.tolist())
    weights = torch.tensor(
        [
            (1.0 / group_counts[group]) * (1.0 / class_counts[int(label)])
            for group, label in zip(groups_train, y_train.tolist(), strict=True)
        ],
        dtype=torch.float32,
        device=device,
    )
    weights = weights / weights.mean()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train).squeeze(-1)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        loss = (losses * weights).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(test).squeeze(-1)).cpu()
    return probabilities, (probabilities >= 0.5).to(torch.int64)


def run_position_general_belief_probe_feasibility(
    *,
    activation_rows_path: str = DEFAULT_ACTIVATION_ROWS,
    candidate_rows_path: str = DEFAULT_CANDIDATES,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    layer: int = 15,
    n_folds: int = 5,
    epochs: int = 100,
    learning_rate: float = 0.01,
    weight_decay: float = 0.0001,
    seed: int = 42,
    device: str = "cpu",
) -> None:
    """Train once across positions and evaluate on entirely held-out states."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    activation_rows = [
        row
        for row in _read_csv(Path(activation_rows_path))
        if int(row["layer"]) == layer
    ]
    candidates = {row["example_id"]: row for row in _read_csv(Path(candidate_rows_path))}
    groups = [row["example_id"] for row in activation_rows]
    truths = {
        example_id: json.loads(row["probe_truths_json"])
        for example_id, row in candidates.items()
    }
    out = Path(output_dir)
    prediction_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    matrices = {
        representation: torch.stack(
            [_load_representation(row, representation) for row in activation_rows]
        )
        for representation in REPRESENTATIONS
    }
    for question_idx, question_id in enumerate(PRIMARY_BELIEFS):
        labels_by_group = {
            group: int(str(truths[group][question_id]).lower() == "yes")
            for group in sorted(set(groups))
        }
        if len(set(labels_by_group.values())) < 2:
            skipped.append(
                {
                    "question_id": question_id,
                    "reason": "only one class is present across independent states",
                    "class_counts_json": json.dumps(dict(Counter(labels_by_group.values()))),
                }
            )
            continue
        folds = _stratified_group_folds(
            labels_by_group,
            n_folds=n_folds,
            seed=seed + question_idx,
        )
        labels = torch.tensor([labels_by_group[group] for group in groups], dtype=torch.float32)
        for representation, matrix in matrices.items():
            for fold_idx, test_groups in enumerate(folds):
                test_indices = [index for index, group in enumerate(groups) if group in test_groups]
                train_indices = [index for index, group in enumerate(groups) if group not in test_groups]
                probabilities, predictions = _fit_linear_probe(
                    matrix[train_indices],
                    labels[train_indices],
                    [groups[index] for index in train_indices],
                    matrix[test_indices],
                    device=device,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                )
                for local_idx, row_idx in enumerate(test_indices):
                    source = activation_rows[row_idx]
                    prediction_rows.append(
                        {
                            "example_id": source["example_id"],
                            "reasoning_step_idx": int(source["reasoning_step_idx"]),
                            "reasoning_progress": float(source["reasoning_progress"]),
                            "progress_bucket_10": int(source["progress_bucket_10"]),
                            "question_id": question_id,
                            "representation": representation,
                            "fold": fold_idx,
                            "ground_truth": int(labels[row_idx].item()),
                            "probability_yes": float(probabilities[local_idx].item()),
                            "prediction": int(predictions[local_idx].item()),
                            "correct": int(predictions[local_idx].item() == labels[row_idx].item()),
                        }
                    )

    summary: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_progress: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[(row["question_id"], row["representation"])].append(row)
        by_progress[
            (row["question_id"], row["representation"], int(row["progress_bucket_10"]))
        ].append(row)
    for (question_id, representation), rows in sorted(grouped.items()):
        labels = [int(row["ground_truth"]) for row in rows]
        predictions = [int(row["prediction"]) for row in rows]
        summary.append(
            {
                "question_id": question_id,
                "representation": representation,
                "n_positions": len(rows),
                "n_states": len({row["example_id"] for row in rows}),
                "accuracy": sum(row["correct"] for row in rows) / len(rows),
                "balanced_accuracy": _balanced_accuracy(labels, predictions),
                "positive_state_fraction": sum(
                    labels_by_group_value == 1
                    for labels_by_group_value in {
                        row["example_id"]: int(row["ground_truth"]) for row in rows
                    }.values()
                )
                / len({row["example_id"] for row in rows}),
            }
        )
    progress_summary = []
    for (question_id, representation, bucket), rows in sorted(by_progress.items()):
        labels = [int(row["ground_truth"]) for row in rows]
        predictions = [int(row["prediction"]) for row in rows]
        progress_summary.append(
            {
                "question_id": question_id,
                "representation": representation,
                "progress_bucket_10": bucket,
                "n_positions": len(rows),
                "accuracy": sum(row["correct"] for row in rows) / len(rows),
                "balanced_accuracy": _balanced_accuracy(labels, predictions),
            }
        )
    _write_csv(out / "prediction_rows.csv", prediction_rows)
    _write_csv(out / "summary.csv", summary)
    _write_csv(out / "summary_by_progress.csv", progress_summary)
    _write_csv(out / "skipped_questions.csv", skipped)
    status = {
        "status": "completed",
        "interpretation": "feasibility_pilot_only",
        "layer": layer,
        "n_independent_states": len(set(groups)),
        "n_positions": len(activation_rows),
        "n_folds": n_folds,
        "grouping_contract": "all positions from one state remain in one fold",
        "representations": list(REPRESENTATIONS),
        "questions_evaluated": sorted({row["question_id"] for row in prediction_rows}),
        "questions_skipped": skipped,
        "training_parameters": {
            "model": "linear_logistic_probe",
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "state_and_class_weighted_loss": True,
            "seed": seed,
        },
    }
    (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")


__all__ = ["run_position_general_belief_probe_feasibility"]
