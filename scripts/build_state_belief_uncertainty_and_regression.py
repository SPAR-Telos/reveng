#!/usr/bin/env python3
"""Build state-belief uncertainty and exploratory regression tables.

This script reuses the stored Experiment 2 behavioral belief prompts. It queries
label logprobs for the six primary state-belief questions, converts the
yes/no/unknown probabilities into entropy, aggregates that entropy per reasoning
prefix, and fits small logistic regression models for next-prefix action events.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from reveng.experiments.behavioral_probe_runner import (
    BehavioralProbeLLM,
    _collect_label3_logprob_readout,
)
from reveng.experiments.reasoning_belief_action import PRIMARY_QUESTION_IDS


DEFAULT_OUTPUT_DIR = Path("outputs/experiment2_behavioral_beliefs/weisheng_8_state_v1")
DEFAULT_MODEL_NAME = "together_ai/openai/gpt-oss-20b"
LABELS = ("yes", "no", "unknown")


def _action_entropy_value(row: dict[str, Any]) -> Any:
    """Use the primary logprob action entropy, with MC entropy as legacy fallback."""
    value = row.get("action_logprob_entropy_bits", "")
    if value not in {None, ""}:
        return value
    return row.get("action_mc_entropy", "")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _load_jsonl_rows(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = (str(row["example_id"]), int(row["reasoning_step_idx"]), str(row["question_id"]))
        except Exception:
            continue
        if not row.get("query_error"):
            rows[key] = row
    return rows


def _as_bool(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _entropy_bits_label3(probabilities: dict[str, float]) -> tuple[float | None, float | None, float]:
    values = [max(0.0, float(probabilities.get(label, 0.0))) for label in LABELS]
    total = sum(values)
    if total <= 0:
        return None, None, 0.0
    normalized = [value / total for value in values]
    entropy = -sum(prob * math.log2(prob) for prob in normalized if prob > 0)
    return entropy, entropy / math.log2(len(LABELS)), total


def _query_uncertainty_row(
    row: dict[str, str],
    *,
    model_name: str,
    temperature: float,
    top_logprobs: int,
    seed: int,
    thread_local: threading.local,
) -> dict[str, Any]:
    client = getattr(thread_local, "client", None)
    if client is None:
        client = BehavioralProbeLLM(model_name=model_name, temperature=temperature)
        thread_local.client = client
    base = {
        "example_id": row["example_id"],
        "trajectory_id": row.get("trajectory_id", ""),
        "step_index": row.get("step_index", ""),
        "failure_category": row.get("failure_category", ""),
        "reasoning_step_idx": int(row["reasoning_step_idx"]),
        "reasoning_progress": row.get("reasoning_progress", ""),
        "reasoning_character_progress": row.get("reasoning_character_progress", ""),
        "analysis_unit": row.get("analysis_unit", "packed_chunk"),
        "analysis_unit_index": row.get("analysis_unit_index", row["reasoning_step_idx"]),
        "analysis_unit_char_start": row.get("analysis_unit_char_start", ""),
        "analysis_unit_char_end": row.get("analysis_unit_char_end", ""),
        "canonical_sentence_id": row.get("canonical_sentence_id", ""),
        "question_id": row["question_id"],
        "answer_key_greedy": row.get("answer_key", ""),
        "ground_truth_key": row.get("ground_truth_key", ""),
        "belief_is_error_greedy": row.get("belief_is_error", ""),
        "logprob_temperature": temperature,
        "top_logprobs": top_logprobs,
        "seed": seed,
    }
    try:
        result = _collect_label3_logprob_readout(
            client=client,
            prompt=row["prompt"],
            seed=seed,
            top_logprobs=top_logprobs,
        )
        probs = result["semantic_probs"]
        entropy_bits, entropy_normalized, valid_mass = _entropy_bits_label3(probs)
        return {
            **base,
            "query_error": "",
            "visible_answer": result["visible_answer"],
            "logprob_answer": result["answer"],
            "p_yes": probs.get("yes", 0.0),
            "p_no": probs.get("no", 0.0),
            "p_unknown": probs.get("unknown", 0.0),
            "p_invalid": probs.get("invalid", 0.0),
            "valid_label_probability_mass": valid_mass,
            "state_belief_entropy_bits": entropy_bits,
            "state_belief_entropy_normalized": entropy_normalized,
            "semantic_probs_json": json.dumps(probs, sort_keys=True),
            "raw_probs_json": json.dumps(result["raw_probs"], sort_keys=True),
            "answer_token_logprobs_json": json.dumps(result["answer_token_logprobs"], sort_keys=True),
            "raw_output": result["raw_output"],
            "cost_usd": result["usage"].get("cost_usd", 0.0),
            "prompt_tokens": result["usage"].get("prompt_tokens", 0),
            "completion_tokens": result["usage"].get("completion_tokens", 0),
            "total_tokens": result["usage"].get("total_tokens", 0),
            "retry_count": result["usage"].get("retry_count", 0),
        }
    except Exception as exc:
        return {
            **base,
            "query_error": repr(exc),
            "visible_answer": "",
            "logprob_answer": "",
            "p_yes": "",
            "p_no": "",
            "p_unknown": "",
            "p_invalid": "",
            "valid_label_probability_mass": "",
            "state_belief_entropy_bits": "",
            "state_belief_entropy_normalized": "",
            "semantic_probs_json": "",
            "raw_probs_json": "",
            "answer_token_logprobs_json": "",
            "raw_output": "",
            "cost_usd": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "retry_count": 0,
        }


def build_uncertainty_rows(
    *,
    output_dir: Path,
    model_name: str,
    temperature: float,
    top_logprobs: int,
    seed: int,
    max_workers: int,
    limit: int | None,
    skip_queries: bool,
) -> list[dict[str, Any]]:
    belief_rows = _read_csv(output_dir / "belief_rows.csv")
    target_rows = [
        row for row in belief_rows
        if row.get("answer_space") == "label3"
        and row.get("question_id") in set(PRIMARY_QUESTION_IDS)
    ]
    if limit is not None:
        target_rows = target_rows[:limit]

    checkpoint_path = output_dir / "state_belief_uncertainty_rows.jsonl"
    existing = _load_jsonl_rows(checkpoint_path)
    tasks = [
        row for row in target_rows
        if (str(row["example_id"]), int(row["reasoning_step_idx"]), str(row["question_id"])) not in existing
    ]
    if skip_queries and tasks:
        raise RuntimeError(
            f"{len(tasks)} uncertainty rows are missing from {checkpoint_path}; rerun without --skip-queries."
        )

    lock = threading.Lock()
    thread_local = threading.local()
    completed = len(existing)
    if tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _query_uncertainty_row,
                    row,
                    model_name=model_name,
                    temperature=temperature,
                    top_logprobs=top_logprobs,
                    seed=seed,
                    thread_local=thread_local,
                )
                for row in tasks
            ]
            for future in as_completed(futures):
                result = future.result()
                key = (
                    str(result["example_id"]),
                    int(result["reasoning_step_idx"]),
                    str(result["question_id"]),
                )
                existing[key] = result
                _append_jsonl(checkpoint_path, result, lock)
                completed += 1
                if completed % 50 == 0 or completed == len(target_rows):
                    print(f"uncertainty rows completed: {completed}/{len(target_rows)}", flush=True)

    ordered = [
        existing[(str(row["example_id"]), int(row["reasoning_step_idx"]), str(row["question_id"]))]
        for row in target_rows
        if (str(row["example_id"]), int(row["reasoning_step_idx"]), str(row["question_id"])) in existing
    ]
    _write_csv(output_dir / "state_belief_uncertainty_rows.csv", ordered)
    return ordered


def build_position_uncertainty(output_dir: Path, uncertainty_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    position_rows = _read_csv(output_dir / "position_rows.csv")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in uncertainty_rows:
        if row.get("query_error"):
            continue
        entropy = _as_float(row.get("state_belief_entropy_bits"))
        normalized = _as_float(row.get("state_belief_entropy_normalized"))
        if entropy is None or normalized is None:
            continue
        key = (str(row["example_id"]), int(row["reasoning_step_idx"]))
        grouped.setdefault(key, []).append(row)

    out_rows: list[dict[str, Any]] = []
    for row in position_rows:
        key = (str(row["example_id"]), int(row["reasoning_step_idx"]))
        values = grouped.get(key, [])
        entropy_values = [_as_float(item.get("state_belief_entropy_bits")) for item in values]
        normalized_values = [_as_float(item.get("state_belief_entropy_normalized")) for item in values]
        entropy_values = [value for value in entropy_values if value is not None]
        normalized_values = [value for value in normalized_values if value is not None]
        out_rows.append(
            {
                **row,
                "action_entropy_bits": _action_entropy_value(row),
                "action_entropy_method": (
                    "four-action token logprobs"
                    if row.get("action_logprob_entropy_bits") not in {None, ""}
                    else "sampled actions"
                ),
                "n_state_uncertainty_beliefs": len(values),
                "state_belief_entropy_mean_bits": (
                    sum(entropy_values) / len(entropy_values) if entropy_values else ""
                ),
                "state_belief_entropy_max_bits": max(entropy_values) if entropy_values else "",
                "state_belief_entropy_mean_normalized": (
                    sum(normalized_values) / len(normalized_values) if normalized_values else ""
                ),
                "state_belief_entropy_max_normalized": max(normalized_values) if normalized_values else "",
            }
        )
    _write_csv(output_dir / "state_uncertainty_position_rows.csv", out_rows)
    return out_rows


def build_event_uncertainty(output_dir: Path, position_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    action_events = _read_csv(output_dir / "action_transition_rows.csv")
    by_key = {
        (str(row["example_id"]), int(row["reasoning_step_idx"])): row
        for row in position_rows
    }
    out_rows: list[dict[str, Any]] = []
    for event in action_events:
        idx = int(event["reasoning_step_idx"])
        current = by_key.get((event["example_id"], idx), {})
        previous = by_key.get((event["example_id"], idx - 1), {})
        prev_entropy = _as_float(previous.get("state_belief_entropy_mean_bits"))
        curr_entropy = _as_float(current.get("state_belief_entropy_mean_bits"))
        out_rows.append(
            {
                **event,
                "previous_state_belief_entropy_mean_bits": prev_entropy if prev_entropy is not None else "",
                "current_state_belief_entropy_mean_bits": curr_entropy if curr_entropy is not None else "",
                "delta_state_belief_entropy_mean_bits": (
                    curr_entropy - prev_entropy
                    if prev_entropy is not None and curr_entropy is not None
                    else ""
                ),
                "previous_action_entropy_bits": _action_entropy_value(previous),
                "current_action_entropy_bits": _action_entropy_value(current),
                "action_entropy_method": (
                    "four-action token logprobs"
                    if current.get("action_logprob_entropy_bits") not in {None, ""}
                    else "sampled actions"
                ),
            }
        )
    _write_csv(output_dir / "state_uncertainty_event_rows.csv", out_rows)
    return out_rows


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40, 40)))


def _fit_logistic_model(
    rows: list[dict[str, Any]],
    *,
    outcome: str,
    predictors: list[str],
    model_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_rows: list[dict[str, Any]] = []
    y_values: list[float] = []
    x_values: list[list[float]] = []
    for row in rows:
        y_bool = _as_bool(row.get(outcome))
        if y_bool is None:
            continue
        xs: list[float] = []
        missing = False
        for predictor in predictors:
            if predictor.startswith("is_") or predictor in {
                "current_action_is_optimal",
                "chosen_action_conflicts_with_reported_state",
                "blocked_report_and_predicted_hit",
            }:
                value_bool = _as_bool(row.get(predictor))
                if value_bool is None:
                    missing = True
                    break
                xs.append(1.0 if value_bool else 0.0)
            else:
                value = _as_float(row.get(predictor))
                if value is None:
                    missing = True
                    break
                xs.append(value)
        if missing:
            continue
        model_rows.append(row)
        y_values.append(1.0 if y_bool else 0.0)
        x_values.append(xs)

    n = len(y_values)
    events = int(sum(y_values))
    non_events = n - events
    status = {
        "model_name": model_name,
        "outcome": outcome,
        "predictors": predictors,
        "n": n,
        "events": events,
        "non_events": non_events,
        "converged": False,
        "warning": "",
    }
    if 0 < min(events, non_events) < 10:
        status["warning"] = (
            f"rare event model: only {min(events, non_events)} rows in the smaller class; "
            "coefficients and p-values are unstable"
        )
    if n == 0 or events == 0 or non_events == 0:
        status["warning"] = "not fitted: outcome has no variation"
        return [], status

    X = np.column_stack([np.ones(n), np.asarray(x_values, dtype=float)])
    y = np.asarray(y_values, dtype=float)

    def objective(beta: np.ndarray) -> float:
        p = _sigmoid(X @ beta)
        eps = 1e-9
        return float(-np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    def gradient(beta: np.ndarray) -> np.ndarray:
        p = _sigmoid(X @ beta)
        return X.T @ (p - y)

    result = minimize(
        objective,
        np.zeros(X.shape[1]),
        jac=gradient,
        method="BFGS",
        options={"maxiter": 1000, "gtol": 1e-6},
    )
    beta = np.asarray(result.x, dtype=float)
    p_hat = _sigmoid(X @ beta)
    weights = p_hat * (1.0 - p_hat)
    hessian = X.T @ (X * weights[:, None])
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(hessian)
        status["warning"] = "used pseudo-inverse covariance; estimates may be unstable"
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    names = ["Intercept", *predictors]
    table_rows: list[dict[str, Any]] = []
    for name, coef, se_value in zip(names, beta, se):
        z_value = coef / se_value if se_value > 0 else float("nan")
        p_value = 2 * norm.sf(abs(z_value)) if math.isfinite(z_value) else float("nan")
        ci_low = coef - 1.96 * se_value
        ci_high = coef + 1.96 * se_value
        table_rows.append(
            {
                "model_name": model_name,
                "outcome": outcome,
                "term": name,
                "coefficient": coef,
                "std_error": se_value,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "z_value": z_value,
                "p_value": p_value,
                "odds_ratio": math.exp(coef) if abs(coef) < 700 else "",
                "odds_ratio_ci_95_low": math.exp(ci_low) if abs(ci_low) < 700 else "",
                "odds_ratio_ci_95_high": math.exp(ci_high) if abs(ci_high) < 700 else "",
                "n": n,
                "events": events,
                "non_events": non_events,
                "converged": bool(result.success),
                "model_warning": status["warning"],
            }
        )
    status["converged"] = bool(result.success)
    if not result.success:
        status["warning"] = (status["warning"] + "; " if status["warning"] else "") + str(result.message)
    if min(p_hat) < 1e-5 or max(p_hat) > 1 - 1e-5:
        status["warning"] = (status["warning"] + "; " if status["warning"] else "") + (
            "near-perfect fitted probabilities; possible separation"
        )
    return table_rows, status


def build_regression_tables(output_dir: Path, position_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    factor_rows = _read_csv(output_dir / "factorization_position_rows.csv")
    positions_by_key = {
        (str(row["example_id"]), int(row["reasoning_step_idx"])): row
        for row in position_rows
    }
    merged: list[dict[str, Any]] = []
    for row in factor_rows:
        key = (str(row["example_id"]), int(row["reasoning_step_idx"]))
        position = positions_by_key.get(key, {})
        merged.append(
            {
                **row,
                "primary_belief_error_rate": position.get("primary_belief_error_rate", ""),
                "action_entropy_bits": _action_entropy_value(position),
                "action_entropy_method": (
                    "four-action token logprobs"
                    if position.get("action_logprob_entropy_bits") not in {None, ""}
                    else "sampled actions"
                ),
                "state_belief_entropy_mean_bits": position.get("state_belief_entropy_mean_bits", ""),
                "state_belief_entropy_per_0p01_bits": (
                    (_as_float(position.get("state_belief_entropy_mean_bits")) or 0.0) / 0.01
                    if _as_float(position.get("state_belief_entropy_mean_bits")) is not None
                    else ""
                ),
                "state_belief_entropy_mean_normalized": position.get("state_belief_entropy_mean_normalized", ""),
            }
        )
    _write_csv(output_dir / "regression_model_rows.csv", merged)

    specs = [
        (
            "action_change_next_main",
            "action_change_next",
            [
                "reasoning_progress",
                "action_entropy_bits",
                "state_belief_entropy_per_0p01_bits",
                "primary_belief_error_rate",
                "chosen_action_conflicts_with_reported_state",
            ],
        ),
        (
            "optimality_loss_next_compact",
            "optimality_loss_next",
            [
                "reasoning_progress",
                "action_entropy_bits",
                "state_belief_entropy_per_0p01_bits",
                "primary_belief_error_rate",
            ],
        ),
        (
            "optimality_recovery_next_compact",
            "optimality_recovery_next",
            [
                "reasoning_progress",
                "action_entropy_bits",
                "state_belief_entropy_per_0p01_bits",
                "primary_belief_error_rate",
            ],
        ),
    ]
    table_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for model_name, outcome, predictors in specs:
        rows, status = _fit_logistic_model(
            merged,
            outcome=outcome,
            predictors=predictors,
            model_name=model_name,
        )
        table_rows.extend(rows)
        status_rows.append(status)
    _write_csv(output_dir / "exploratory_logistic_regression_table.csv", table_rows)
    _write_csv(output_dir / "exploratory_logistic_regression_status.csv", status_rows)
    return table_rows, status_rows


def _format_float(value: Any, digits: int = 3) -> str:
    if value == "" or value is None:
        return ""
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def write_markdown_summary(
    *,
    output_dir: Path,
    uncertainty_rows: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    regression_rows: list[dict[str, Any]],
    regression_status: list[dict[str, Any]],
    temperature: float,
) -> None:
    completed = [row for row in uncertainty_rows if not row.get("query_error")]
    costs = [_as_float(row.get("cost_usd")) or 0.0 for row in uncertainty_rows]
    entropies = [_as_float(row.get("state_belief_entropy_bits")) for row in completed]
    entropies = [value for value in entropies if value is not None]
    position_entropies = [_as_float(row.get("state_belief_entropy_mean_bits")) for row in position_rows]
    position_entropies = [value for value in position_entropies if value is not None]

    lines = [
        "# State-Belief Uncertainty And Exploratory Regression",
        "",
        "State-belief uncertainty is computed from logprob readouts for the six primary yes/no/unknown state-belief probes: wall left, wall right, wall up, wall down, key held, and door open.",
        "For each probe, yes/no/unknown probabilities are normalized after excluding invalid mass, then Shannon entropy is computed in bits. Per-position uncertainty is the mean entropy across available primary probes.",
        f"The label logprobs were requested at temperature {temperature:g}.",
        "",
        "The regression table is exploratory. It treats repeated reasoning prefixes as rows and reports model-based logistic-regression standard errors, confidence intervals, and p-values. Because this run has 8 states from one source trajectory group, these p-values are diagnostics, not a population-level inferential claim.",
        "In the regression, state-belief entropy is scaled in units of 0.01 bits because the logprob readouts are very confident and raw entropy values are close to zero.",
        "",
        "## Query Summary",
        "",
        f"- Uncertainty rows requested: {len(uncertainty_rows)}",
        f"- Successful logprob rows: {len(completed)}",
        f"- Failed logprob rows: {len(uncertainty_rows) - len(completed)}",
        f"- Approximate query cost in USD: {_format_float(sum(costs), 4)}",
        f"- Mean probe entropy in bits: {_format_float(sum(entropies) / len(entropies) if entropies else None)}",
        f"- Mean per-position state-belief entropy in bits: {_format_float(sum(position_entropies) / len(position_entropies) if position_entropies else None)}",
        "",
        "## Event-Level State-Belief Uncertainty",
        "",
        "| event type | n | mean previous entropy | mean current entropy | mean change |",
        "|---|---:|---:|---:|---:|",
    ]
    by_event: dict[str, list[dict[str, Any]]] = {}
    for row in event_rows:
        by_event.setdefault(str(row["event_type"]), []).append(row)
    for event_type, rows in sorted(by_event.items()):
        prev = [_as_float(row.get("previous_state_belief_entropy_mean_bits")) for row in rows]
        curr = [_as_float(row.get("current_state_belief_entropy_mean_bits")) for row in rows]
        delta = [_as_float(row.get("delta_state_belief_entropy_mean_bits")) for row in rows]
        prev = [value for value in prev if value is not None]
        curr = [value for value in curr if value is not None]
        delta = [value for value in delta if value is not None]
        lines.append(
            "| "
            + " | ".join(
                [
                    event_type.replace("_", " "),
                    str(len(rows)),
                    _format_float(sum(prev) / len(prev) if prev else None),
                    _format_float(sum(curr) / len(curr) if curr else None),
                    _format_float(sum(delta) / len(delta) if delta else None),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Regression Status",
            "",
            "| model | outcome | n | events | converged | warning |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for row in regression_status:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_name"]),
                    str(row["outcome"]),
                    str(row["n"]),
                    str(row["events"]),
                    str(row["converged"]),
                    str(row.get("warning", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Regression Coefficients",
            "",
            "| model | term | coefficient | SE | 95% CI | p | odds ratio |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in regression_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_name"]),
                    str(row["term"]),
                    _format_float(row["coefficient"]),
                    _format_float(row["std_error"]),
                    f"[{_format_float(row['ci_95_low'])}, {_format_float(row['ci_95_high'])}]",
                    _format_float(row["p_value"], 4),
                    _format_float(row["odds_ratio"]),
                ]
            )
            + " |"
        )
    (output_dir / "STATE_UNCERTAINTY_AND_REGRESSION.md").write_text("\n".join(lines) + "\n")


def write_uncertainty_predictor_outputs(
    *,
    output_dir: Path,
    regression_rows: list[dict[str, Any]],
    regression_status: list[dict[str, Any]],
    state_belief_temperature: float,
) -> None:
    """Write a focused uncertainty table and coefficient plot."""
    predictor_labels = {
        "action_entropy_bits": "Action uncertainty (1 bit)",
        "state_belief_entropy_per_0p01_bits": "State-belief uncertainty (0.01 bits)",
    }
    outcome_labels = {
        "action_change_next": "Action changes at next prefix",
        "optimality_loss_next": "Optimal action becomes suboptimal",
        "optimality_recovery_next": "Suboptimal action becomes optimal",
    }
    status_by_model = {str(row["model_name"]): row for row in regression_status}
    selected = [row for row in regression_rows if row.get("term") in predictor_labels]
    summary_rows: list[dict[str, Any]] = []
    for row in selected:
        status = status_by_model.get(str(row["model_name"]), {})
        summary_rows.append(
            {
                "outcome": outcome_labels.get(str(row["outcome"]), str(row["outcome"])),
                "predictor": predictor_labels[str(row["term"])],
                "coefficient_log_odds": row["coefficient"],
                "std_error": row["std_error"],
                "ci_95_low": row["ci_95_low"],
                "ci_95_high": row["ci_95_high"],
                "p_value": row["p_value"],
                "odds_ratio": row["odds_ratio"],
                "n_prefix_transitions": row["n"],
                "n_events": row["events"],
                "warning": status.get("warning", ""),
            }
        )
    _write_csv(output_dir / "uncertainty_predictor_summary.csv", summary_rows)

    lines = [
        "# Uncertainty Predictors of Action Events",
        "",
        f"Action uncertainty is Shannon entropy over the normalized UP, DOWN, LEFT, and RIGHT token probabilities from a temperature 0.7 logprob query. State-belief uncertainty is the mean Shannon entropy over the six primary yes/no/unknown logprob readouts at temperature {state_belief_temperature:g}.",
        "",
        "| outcome | predictor | coefficient | SE | 95% CI | p | odds ratio | events |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["outcome"]),
                    str(row["predictor"]),
                    _format_float(row["coefficient_log_odds"]),
                    _format_float(row["std_error"]),
                    f"[{_format_float(row['ci_95_low'])}, {_format_float(row['ci_95_high'])}]",
                    _format_float(row["p_value"], 4),
                    _format_float(row["odds_ratio"]),
                    str(row["n_events"]),
                ]
            )
            + " |"
        )
    status_by_outcome = {
        str(row.get("outcome")): row for row in regression_status
    }
    action_events = status_by_outcome.get("action_change_next", {}).get("events", "")
    loss_events = status_by_outcome.get("optimality_loss_next", {}).get("events", "")
    recovery_events = status_by_outcome.get("optimality_recovery_next", {}).get("events", "")
    lines.extend(
        [
            "",
            f"The action-change model has {action_events} events. The optimality-loss and recovery models have {loss_events} and {recovery_events} events, respectively. Coefficients and p-values are unstable when event counts are small. All standard errors are model-based; repeated prefixes and the 8-state sample mean these are exploratory diagnostics rather than population-level inference.",
        ]
    )
    (output_dir / "UNCERTAINTY_PREDICTOR_SUMMARY.md").write_text("\n".join(lines) + "\n")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    model_order = [
        "action_change_next_main",
        "optimality_loss_next_compact",
        "optimality_recovery_next_compact",
    ]
    figure_rows: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        figure_rows.setdefault(str(row["model_name"]), []).append(row)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    colors = {
        "action_entropy_bits": "#1f5f9f",
        "state_belief_entropy_per_0p01_bits": "#8bb8df",
    }
    short_labels = {
        "action_entropy_bits": "Action entropy\n(+1 bit)",
        "state_belief_entropy_per_0p01_bits": "State-belief entropy\n(+0.01 bits)",
    }
    for ax, model_name in zip(axes, model_order):
        rows = sorted(figure_rows.get(model_name, []), key=lambda row: str(row["term"]))
        status = status_by_model.get(model_name, {})
        for y, row in enumerate(rows):
            coef = float(row["coefficient"])
            low = float(row["ci_95_low"])
            high = float(row["ci_95_high"])
            ax.errorbar(
                coef,
                y,
                xerr=[[coef - low], [high - coef]],
                fmt="o",
                color=colors[str(row["term"])],
                capsize=3,
                markersize=5,
                linewidth=1.4,
            )
        ax.axvline(0, color="#64748b", linewidth=1)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([short_labels[str(row["term"])] for row in rows])
        outcome = str(rows[0]["outcome"]) if rows else ""
        title = outcome_labels.get(outcome, outcome)
        ax.set_title(f"{title}\n(n={status.get('n', 0)}, events={status.get('events', 0)})")
        ax.set_xlabel("Regression coefficient\n(positive = event more likely; bars = 95% CI)")
        ax.grid(axis="x", color="#d7e3f0", linewidth=0.8)
    position_rows = _read_csv(output_dir / "position_rows.csv")
    analysis_units = {row.get("analysis_unit", "packed_chunk") for row in position_rows}
    analysis_unit = next(iter(analysis_units)) if len(analysis_units) == 1 else "analysis unit"
    next_unit = "Sentence" if analysis_unit == "sentence" else "Packed Reasoning Unit"
    fig.suptitle(
        f"Does Current Uncertainty Predict a Change at the Next {next_unit}?",
        fontsize=11,
        color="#172033",
    )
    fig.tight_layout()
    fig_dir = output_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "uncertainty_predictor_coefficients.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    captions_path = output_dir / "FIGURE_CAPTIONS.md"
    caption = (
        "\n## `uncertainty_predictor_coefficients.png`\n"
        f"Exploratory logistic-regression coefficients for whether an action event occurs after the next {analysis_unit.replace('_', ' ')}. The y-axis lists the uncertainty predictor and the increase represented by one coefficient unit. The x-axis is the change in predicted log odds: positive values mean that greater uncertainty is associated with a more likely event, zero means no association, and negative values mean a less likely event. Action entropy is computed from the normalized UP, DOWN, LEFT, and RIGHT token probabilities at temperature 0.7. State-belief entropy uses wall, key, and door answer probabilities at temperature {state_belief_temperature:g} and is scaled per 0.01 bits. Points are coefficients and bars are model-based 95% confidence intervals.\n"
    )
    existing = captions_path.read_text() if captions_path.exists() else "# Figure Captions\n"
    heading = "## `uncertainty_predictor_coefficients.png`"
    if heading in existing:
        updated = re.sub(
            rf"\n{re.escape(heading)}\n.*?(?=\n## `|\Z)",
            caption.rstrip(),
            existing,
            flags=re.S,
        )
        captions_path.write_text(updated.rstrip() + "\n")
    else:
        captions_path.write_text(existing.rstrip() + "\n" + caption)


def write_uncertainty_over_reasoning_plot(
    *,
    output_dir: Path,
    position_rows: list[dict[str, Any]],
    state_belief_temperature: float,
) -> None:
    """Plot action and state-belief entropy over normalized reasoning progress."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    metrics = (
        (
            "action_entropy_bits",
            "Action entropy from UP/DOWN/LEFT/RIGHT token probabilities (T=0.7)",
            "Action entropy (bits)",
        ),
        (
            "state_belief_entropy_mean_bits",
            f"Mean entropy of wall, key, and door answers (T={state_belief_temperature:g})",
            "State-belief entropy (bits)",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7), sharex=True)
    summary_rows: list[dict[str, Any]] = []
    for ax, (metric, title, ylabel) in zip(axes, metrics):
        by_state_bin: dict[tuple[str, int], list[float]] = {}
        for row in position_rows:
            progress = _as_float(
                row.get("reasoning_character_progress") or row.get("reasoning_progress")
            )
            value = _as_float(row.get(metric))
            if progress is None or value is None:
                continue
            bin_index = min(9, max(0, int(progress * 10)))
            by_state_bin.setdefault((str(row["example_id"]), bin_index), []).append(value)
        by_bin: dict[int, list[float]] = {}
        for (_, bin_index), values in by_state_bin.items():
            by_bin.setdefault(bin_index, []).append(sum(values) / len(values))
        xs: list[float] = []
        means: list[float] = []
        errors: list[float] = []
        for bin_index in sorted(by_bin):
            values = by_bin[bin_index]
            value_mean = sum(values) / len(values)
            variance = (
                sum((value - value_mean) ** 2 for value in values) / (len(values) - 1)
                if len(values) > 1
                else 0.0
            )
            xs.append((bin_index + 0.5) / 10)
            means.append(value_mean)
            standard_error = math.sqrt(variance / len(values))
            errors.append(standard_error)
            summary_rows.append(
                {
                    "metric": metric,
                    "progress_bin_start": bin_index / 10,
                    "progress_bin_end": (bin_index + 1) / 10,
                    "progress_bin_center": (bin_index + 0.5) / 10,
                    "n_states": len(values),
                    "mean_entropy_bits": value_mean,
                    "standard_error_across_states": standard_error,
                }
            )
        lower = [value - error for value, error in zip(means, errors)]
        upper = [value + error for value, error in zip(means, errors)]
        ax.plot(xs, means, marker="o", color="#1f5f9f", linewidth=1.8)
        ax.fill_between(xs, lower, upper, color="#dbeafe", alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Fraction of reasoning characters revealed")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", color="#d7e3f0", linewidth=0.8)
    fig.suptitle("Uncertainty as More of the Reasoning Trace Is Revealed", fontsize=11, color="#172033")
    fig.tight_layout()
    fig_dir = output_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "uncertainty_over_reasoning_progress.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    _write_csv(output_dir / "uncertainty_over_reasoning_progress.csv", summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-queries", action="store_true")
    args = parser.parse_args()

    uncertainty_rows = build_uncertainty_rows(
        output_dir=args.output_dir,
        model_name=args.model_name,
        temperature=args.temperature,
        top_logprobs=args.top_logprobs,
        seed=args.seed,
        max_workers=args.max_workers,
        limit=args.limit,
        skip_queries=args.skip_queries,
    )
    position_rows = build_position_uncertainty(args.output_dir, uncertainty_rows)
    event_rows = build_event_uncertainty(args.output_dir, position_rows)
    regression_rows, regression_status = build_regression_tables(args.output_dir, position_rows)
    write_markdown_summary(
        output_dir=args.output_dir,
        uncertainty_rows=uncertainty_rows,
        position_rows=position_rows,
        event_rows=event_rows,
        regression_rows=regression_rows,
        regression_status=regression_status,
        temperature=args.temperature,
    )
    write_uncertainty_predictor_outputs(
        output_dir=args.output_dir,
        regression_rows=regression_rows,
        regression_status=regression_status,
        state_belief_temperature=args.temperature,
    )
    write_uncertainty_over_reasoning_plot(
        output_dir=args.output_dir,
        position_rows=position_rows,
        state_belief_temperature=args.temperature,
    )
    _write_json(
        args.output_dir / "state_uncertainty_manifest.json",
        {
            "model_name": args.model_name,
            "temperature": args.temperature,
            "top_logprobs": args.top_logprobs,
            "seed": args.seed,
            "primary_question_ids": list(PRIMARY_QUESTION_IDS),
            "uncertainty_definition": (
                "Shannon entropy in bits over logprob-normalized yes/no/unknown "
                "probabilities, excluding invalid probability mass."
            ),
            "n_uncertainty_rows": len(uncertainty_rows),
            "maximum_uncertainty_rows": len(position_rows) * len(PRIMARY_QUESTION_IDS),
            "n_position_rows": len(position_rows),
            "n_event_rows": len(event_rows),
            "n_regression_coefficients": len(regression_rows),
        },
    )


if __name__ == "__main__":
    main()
