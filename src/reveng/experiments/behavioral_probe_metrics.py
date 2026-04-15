"""Metrics for behavioral-probe experiments."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from reveng.experiments.behavioral_probe_parse import (
    COORD_MISSING,
    VALID_BEHAVIORAL_PROBE_ANSWERS,
    coordinate_manhattan_distance,
    coordinate_to_key,
)

ANSWER_ORDER = {answer: idx for idx, answer in enumerate(VALID_BEHAVIORAL_PROBE_ANSWERS)}
ACTION_BY_DIRECTION = {
    "wall_left": "LEFT",
    "wall_right": "RIGHT",
    "wall_up": "UP",
    "wall_down": "DOWN",
}


def sampled_answer_from_probabilities(probabilities: dict[str, float]) -> str:
    if not probabilities:
        return "invalid"
    ordered = {
        answer: float(probabilities.get(answer, 0.0))
        for answer in VALID_BEHAVIORAL_PROBE_ANSWERS
    }
    return max(
        ordered.items(),
        key=lambda item: (item[1], -ANSWER_ORDER[item[0]]),
    )[0]


def shannon_entropy(probabilities: dict[str, float]) -> float:
    entropy = 0.0
    for prob in probabilities.values():
        if prob > 0.0:
            entropy -= prob * math.log2(prob)
    return entropy


def yes_no_answer_from_probabilities(probabilities: dict[str, float]) -> str:
    yes_prob = float(probabilities.get("yes", 0.0))
    no_prob = float(probabilities.get("no", 0.0))
    if yes_prob > no_prob:
        return "yes"
    if no_prob > yes_prob:
        return "no"
    return "unknown"


def valid_parse_rate(*answers: str) -> float:
    if not answers:
        return 0.0
    valid = sum(answer != "invalid" for answer in answers)
    return valid / len(answers)


def valid_parse_rate_from_answers(answers: list[str]) -> float:
    if not answers:
        return 0.0
    valid = sum(answer != "invalid" for answer in answers)
    return valid / len(answers)


def _consistency_target_action(question_id: str) -> str | None:
    if question_id in ACTION_BY_DIRECTION:
        return ACTION_BY_DIRECTION[question_id]
    if question_id.startswith("hit_wall_after_"):
        return question_id.removeprefix("hit_wall_after_").upper()
    return None


def belief_action_consistency(
    belief_answer: str,
    observed_action: str,
    question_id: str,
) -> str:
    target_action = _consistency_target_action(question_id)
    if target_action is None:
        return "not_applicable"
    if belief_answer in {"unknown", "invalid"}:
        return "unknown_based"
    if observed_action != target_action:
        return "not_applicable"
    if belief_answer == "yes":
        return "inconsistent"
    if belief_answer == "no":
        return "potentially_consistent"
    return "unknown_based"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _row_value(row: dict[str, Any], key: str, fallback_key: str | None = None) -> Any:
    if key in row:
        return row[key]
    if fallback_key is not None and fallback_key in row:
        return row[fallback_key]
    raise KeyError(key)


def _summarize_label3_rows(question_rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_rows = len(question_rows)
    consistency_counts = Counter(
        row["belief_action_consistency"]
        for row in question_rows
        if row["belief_action_consistency"] != "not_applicable"
    )
    mc_consistency_counts = Counter(
        row["mc_belief_action_consistency"]
        for row in question_rows
        if row["mc_belief_action_consistency"] != "not_applicable"
    )

    return {
        "question_id": question_rows[0]["question_id"],
        "target_variable": question_rows[0]["target_variable"],
        "question_family": question_rows[0]["question_family"],
        "answer_space": "label3",
        "n_rows": n_rows,
        "valid_parse_rate": _mean([float(row["valid_parse_rate_t0_for_row"]) for row in question_rows]),
        "valid_parse_rate_t07": _mean([float(row["valid_parse_rate_t07_for_row"]) for row in question_rows]),
        "valid_parse_rate_t1": _mean([float(row["valid_parse_rate_t1_for_row"]) for row in question_rows]),
        "mc_valid_parse_rate": _mean([float(row["mc_valid_parse_rate_for_row"]) for row in question_rows]),
        "greedy_accuracy": _mean(
            [1.0 if row["greedy_answer"] == row["ground_truth_label"] else 0.0 for row in question_rows]
        ),
        "greedy_repeated_modal_accuracy": _mean(
            [1.0 if row.get("greedy_modal_answer", row["greedy_answer"]) == row["ground_truth_label"] else 0.0 for row in question_rows]
        ),
        "greedy_repeated_valid_parse_rate": _mean(
            [float(row.get("greedy_valid_parse_rate_for_row", 1.0)) for row in question_rows]
        ),
        "greedy_repeated_agreement_rate": _mean(
            [float(row.get("greedy_repeated_agreement_rate_for_row", 1.0)) for row in question_rows]
        ),
        "mean_greedy_entropy": _mean([float(row.get("greedy_entropy", 0.0)) for row in question_rows]),
        "logprob_yes_no_accuracy_t0": _mean(
            [1.0 if _row_value(row, "logprob_yes_no_answer_t0", "logprob_answer_t0") == row["ground_truth_label"] else 0.0 for row in question_rows]
        ),
        "logprob_yes_no_accuracy_t07": _mean(
            [1.0 if _row_value(row, "logprob_yes_no_answer_t07", "logprob_answer_t07") == row["ground_truth_label"] else 0.0 for row in question_rows]
        ),
        "logprob_yes_no_accuracy_t1": _mean(
            [1.0 if _row_value(row, "logprob_yes_no_answer_t1", "logprob_answer_t1") == row["ground_truth_label"] else 0.0 for row in question_rows]
        ),
        "mc_yes_no_accuracy": _mean(
            [1.0 if _row_value(row, "mc_yes_no_answer", "mc_answer") == row["ground_truth_label"] else 0.0 for row in question_rows]
        ),
        "logprob_t0_accuracy": _mean(
            [1.0 if row["logprob_answer_t0"] == row["ground_truth_label"] else 0.0 for row in question_rows]
        ),
        "logprob_t07_accuracy": _mean(
            [1.0 if row["logprob_answer_t07"] == row["ground_truth_label"] else 0.0 for row in question_rows]
        ),
        "logprob_t1_accuracy": _mean(
            [1.0 if row["logprob_answer_t1"] == row["ground_truth_label"] else 0.0 for row in question_rows]
        ),
        "mc_accuracy": _mean(
            [1.0 if row["mc_answer"] == row["ground_truth_label"] else 0.0 for row in question_rows]
        ),
        "mean_entropy_t0": _mean([float(row["entropy_t0"]) for row in question_rows]),
        "mean_entropy_t07": _mean([float(row["entropy_t07"]) for row in question_rows]),
        "mean_entropy_t1": _mean([float(row["entropy_t1"]) for row in question_rows]),
        "mean_mc_entropy": _mean([float(row["mc_entropy"]) for row in question_rows]),
        "fraction_unknown_t0": _mean(
            [1.0 if row["logprob_answer_t0"] == "unknown" else 0.0 for row in question_rows]
        ),
        "fraction_unknown_t07": _mean(
            [1.0 if row["logprob_answer_t07"] == "unknown" else 0.0 for row in question_rows]
        ),
        "fraction_unknown_t1": _mean(
            [1.0 if row["logprob_answer_t1"] == "unknown" else 0.0 for row in question_rows]
        ),
        "mc_fraction_unknown": _mean(
            [1.0 if row["mc_answer"] == "unknown" else 0.0 for row in question_rows]
        ),
        "greedy_logprob_t0_agreement": _mean(
            [1.0 if row["greedy_answer"] == row["logprob_answer_t0"] else 0.0 for row in question_rows]
        ),
        "greedy_vs_greedy_repeated_modal_agreement": _mean(
            [1.0 if row["greedy_answer"] == row.get("greedy_modal_answer", row["greedy_answer"]) else 0.0 for row in question_rows]
        ),
        "greedy_logprob_yes_no_t0_agreement": _mean(
            [1.0 if row["greedy_answer"] == _row_value(row, "logprob_yes_no_answer_t0", "logprob_answer_t0") else 0.0 for row in question_rows]
        ),
        "greedy_logprob_yes_no_t07_agreement": _mean(
            [1.0 if row["greedy_answer"] == _row_value(row, "logprob_yes_no_answer_t07", "logprob_answer_t07") else 0.0 for row in question_rows]
        ),
        "greedy_logprob_yes_no_t1_agreement": _mean(
            [1.0 if row["greedy_answer"] == _row_value(row, "logprob_yes_no_answer_t1", "logprob_answer_t1") else 0.0 for row in question_rows]
        ),
        "greedy_mc_yes_no_agreement": _mean(
            [1.0 if row["greedy_answer"] == _row_value(row, "mc_yes_no_answer", "mc_answer") else 0.0 for row in question_rows]
        ),
        "greedy_logprob_t07_agreement": _mean(
            [1.0 if row["greedy_answer"] == row["logprob_answer_t07"] else 0.0 for row in question_rows]
        ),
        "greedy_logprob_t1_agreement": _mean(
            [1.0 if row["greedy_answer"] == row["logprob_answer_t1"] else 0.0 for row in question_rows]
        ),
        "greedy_mc_agreement": _mean(
            [1.0 if row["greedy_answer"] == row["mc_answer"] else 0.0 for row in question_rows]
        ),
        "consistency_inconsistent": consistency_counts.get("inconsistent", 0),
        "consistency_potentially_consistent": consistency_counts.get("potentially_consistent", 0),
        "consistency_unknown_based": consistency_counts.get("unknown_based", 0),
        "mc_consistency_inconsistent": mc_consistency_counts.get("inconsistent", 0),
        "mc_consistency_potentially_consistent": mc_consistency_counts.get("potentially_consistent", 0),
        "mc_consistency_unknown_based": mc_consistency_counts.get("unknown_based", 0),
        # Backward-compatible aliases
        "sampled_accuracy": _mean(
            [1.0 if row["logprob_answer_t1"] == row["ground_truth_label"] else 0.0 for row in question_rows]
        ),
        "mean_entropy": _mean([float(row["entropy_t1"]) for row in question_rows]),
        "fraction_unknown": _mean(
            [1.0 if row["logprob_answer_t1"] == "unknown" else 0.0 for row in question_rows]
        ),
        "greedy_sampled_agreement": _mean(
            [1.0 if row["greedy_answer"] == row["logprob_answer_t1"] else 0.0 for row in question_rows]
        ),
    }


def _summarize_coord_rows(question_rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_rows = len(question_rows)
    greedy_distances = [
        float(row["greedy_manhattan_distance"])
        for row in question_rows
        if row["greedy_manhattan_distance"] != ""
    ]
    mc_modal_distances = [
        float(row["mc_modal_manhattan_distance"])
        for row in question_rows
        if row["mc_modal_manhattan_distance"] != ""
    ]
    mc_mean_distances = [
        float(row["mc_mean_manhattan_distance"])
        for row in question_rows
        if row["mc_mean_manhattan_distance"] != ""
    ]
    return {
        "question_id": question_rows[0]["question_id"],
        "target_variable": question_rows[0]["target_variable"],
        "question_family": question_rows[0]["question_family"],
        "answer_space": "coord_json",
        "n_rows": n_rows,
        "greedy_exact_match_accuracy": _mean(
            [float(row["greedy_exact_match"]) for row in question_rows]
        ),
        "mc_modal_exact_match_accuracy": _mean(
            [float(row["mc_modal_exact_match"]) for row in question_rows]
        ),
        "greedy_invalid_rate": _mean(
            [1.0 if row["greedy_coordinate_key"] == "invalid" else 0.0 for row in question_rows]
        ),
        "mc_invalid_rate": _mean(
            [1.0 - float(row["mc_valid_parse_rate_for_row"]) for row in question_rows]
        ),
        "mean_greedy_manhattan_distance": _mean(greedy_distances),
        "mean_mc_modal_manhattan_distance": _mean(mc_modal_distances),
        "mean_mc_sample_manhattan_distance": _mean(mc_mean_distances),
        "mean_mc_unique_coordinate_count": _mean(
            [float(row["mc_unique_coordinate_count"]) for row in question_rows]
        ),
    }


def summarize_probe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["question_id"]].append(row)

    summary_rows: list[dict[str, Any]] = []
    for _, question_rows in sorted(grouped.items()):
        answer_space = question_rows[0]["answer_space"]
        if answer_space == "coord_json":
            summary_rows.append(_summarize_coord_rows(question_rows))
        else:
            summary_rows.append(_summarize_label3_rows(question_rows))
    return summary_rows


def summarize_coordinate_samples(
    predictions: list[dict[str, int] | None],
    ground_truth: dict[str, int],
) -> dict[str, Any]:
    valid_predictions = [prediction for prediction in predictions if prediction is not None]
    counts = Counter(coordinate_to_key(prediction) for prediction in valid_predictions)
    if counts:
        modal_key, _ = counts.most_common(1)[0]
        row_str, col_str = modal_key.split(",")
        modal_coord = {"row": int(row_str), "col": int(col_str)}
    else:
        modal_coord = dict(COORD_MISSING)

    distances = [
        coordinate_manhattan_distance(prediction, ground_truth)
        for prediction in valid_predictions
    ]
    return {
        "modal_coordinate": modal_coord,
        "valid_parse_rate": len(valid_predictions) / len(predictions) if predictions else 0.0,
        "unique_coordinate_count": len(counts),
        "mean_manhattan_distance": _mean(
            [float(distance) for distance in distances if distance is not None]
        ),
        "counts": dict(counts),
    }


__all__ = [
    "belief_action_consistency",
    "yes_no_answer_from_probabilities",
    "sampled_answer_from_probabilities",
    "shannon_entropy",
    "summarize_coordinate_samples",
    "summarize_probe_rows",
    "valid_parse_rate",
    "valid_parse_rate_from_answers",
]
