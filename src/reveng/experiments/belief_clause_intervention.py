"""Scaled belief-replacement and verifier-clause action experiment."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from reveng.experiments.behavioral_probe_runner import _behavioral_probe_preamble
from reveng.experiments.belief_replacement_action import (
    ACTIONS,
    BELIEF_IDS,
    BELIEF_LABELS,
    PROBABILITY_LABELS,
    _check_or_write_config,
    _final_action_rows,
    _final_belief_rows,
    _initial_action_rows,
    _parse_bool,
    _sha256,
    _write_json,
    action_question,
    jensen_shannon_divergence,
    parse_probabilities,
    total_variation,
)
from reveng.experiments.gpt_oss_activation_pilot import DEFAULT_MODEL_SNAPSHOT
from reveng.experiments.reasoningflow_transfer import gpu_status


CONDITIONS = (
    "model_reports",
    "verified_truth",
    "counterfactual_wall",
    "irrelevant_control",
)
WALL_IDS = BELIEF_IDS[:4]
IRRELEVANT_LABELS = (
    "The grid display contains numbered row labels",
    "The grid display contains numbered column labels",
    "The legend uses # to denote wall cells",
    "The legend uses _ to denote empty cells",
    "The agent marker in the grid is the letter A",
    "Cardinal move names are written as uppercase labels",
)


@dataclass(frozen=True)
class ClauseExperimentConfig:
    candidate_rows_path: Path
    prefix_action_rows_path: Path
    belief_rows_path: Path
    output_dir: Path
    model_path: Path = DEFAULT_MODEL_SNAPSHOT
    temperature: float = 0.7
    gpu_index: int = 0
    minimum_free_gib: float = 18.0
    seed: int = 42
    bootstrap_repetitions: int = 5000


def _one_hot(label: str) -> dict[str, float | str]:
    if label not in {"yes", "no"}:
        raise ValueError(f"verified truth must be yes or no, got {label!r}")
    return {
        "yes": float(label == "yes"),
        "no": float(label == "no"),
        "unknown": 0.0,
        "top_label": label,
    }


def _flip(
    payload: Mapping[str, Mapping[str, Any]], question_id: str
) -> dict[str, dict[str, Any]]:
    result = {key: dict(value) for key, value in payload.items()}
    probabilities = parse_probabilities(result[question_id], PROBABILITY_LABELS)
    observed = max(probabilities, key=probabilities.get)
    if observed not in {"yes", "no"}:
        raise ValueError("counterfactual source must have a binary top label")
    result[question_id] = _one_hot("no" if observed == "yes" else "yes")
    return result


def select_scaled_cohort(
    candidates: pd.DataFrame,
    actions: pd.DataFrame,
    beliefs: pd.DataFrame,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Join all matched examples to final actions, reports, and simulator truth."""
    required = (
        "example_id",
        "trajectory_id",
        "grid_text",
        "carrying_key",
        "probe_truths_json",
        "optimal_actions_json",
    )
    if missing := set(required) - set(candidates.columns):
        raise ValueError(f"candidate table is missing columns: {sorted(missing)}")
    if candidates["example_id"].duplicated().any():
        raise ValueError("candidate table has duplicate examples")

    final_actions = _final_action_rows(actions)
    initial_actions = _initial_action_rows(actions)
    expected = set(candidates["example_id"].astype(str))
    if set(final_actions["example_id"].astype(str)) != expected:
        raise ValueError("candidate and final-action example sets differ")
    if set(initial_actions["example_id"].astype(str)) != expected:
        raise ValueError("candidate and grid-only action example sets differ")

    final_positions = final_actions.set_index("example_id")["position_index"].to_dict()
    final_beliefs = _final_belief_rows(beliefs, final_positions)
    counts = final_beliefs.groupby("example_id")["question_id"].nunique()
    if set(counts.index.astype(str)) != expected or not counts.eq(len(BELIEF_IDS)).all():
        raise ValueError("every example must have all six final-position state beliefs")

    cohort = candidates[list(required)].merge(
        final_actions[
            ["example_id", "position_index", "action_label", "action_probabilities_json"]
        ].rename(
            columns={
                "position_index": "final_position_index",
                "action_label": "full_cot_action",
                "action_probabilities_json": "full_cot_action_probabilities_json",
            }
        ),
        on="example_id",
        validate="one_to_one",
    ).merge(
        initial_actions[
            ["example_id", "action_label", "action_probabilities_json"]
        ].rename(
            columns={
                "action_label": "grid_only_action",
                "action_probabilities_json": "grid_only_action_probabilities_json",
            }
        ),
        on="example_id",
        validate="one_to_one",
    )

    report_lookup = {
        str(example_id): group.set_index("question_id")
        for example_id, group in final_beliefs.groupby("example_id", sort=False)
    }
    model_payloads: list[str] = []
    verified_payloads: list[str] = []
    counterfactual_payloads: list[str] = []
    counterfactual_ids: list[str] = []
    error_counts: list[int] = []
    for row in cohort.itertuples(index=False):
        group = report_lookup[str(row.example_id)]
        model_payload: dict[str, Any] = {}
        for question_id in BELIEF_IDS:
            probabilities = parse_probabilities(
                group.loc[question_id, "probabilities_json"], PROBABILITY_LABELS
            )
            model_payload[question_id] = {
                **probabilities,
                "top_label": max(probabilities, key=probabilities.get),
            }
        truths = json.loads(str(row.probe_truths_json))
        verified_payload = {
            question_id: _one_hot(str(truths[question_id]))
            for question_id in BELIEF_IDS
        }
        rank = hashlib.sha256(f"{seed}\x1f{row.example_id}".encode()).digest()
        counterfactual_id = WALL_IDS[
            int.from_bytes(rank[:4], "big") % len(WALL_IDS)
        ]
        model_payloads.append(json.dumps(model_payload, sort_keys=True))
        verified_payloads.append(json.dumps(verified_payload, sort_keys=True))
        counterfactual_payloads.append(
            json.dumps(_flip(verified_payload, counterfactual_id), sort_keys=True)
        )
        counterfactual_ids.append(counterfactual_id)
        error_counts.append(int(group["belief_is_error"].map(_parse_bool).sum()))

    cohort["model_beliefs_json"] = model_payloads
    cohort["verified_beliefs_json"] = verified_payloads
    cohort["counterfactual_beliefs_json"] = counterfactual_payloads
    cohort["counterfactual_question_id"] = counterfactual_ids
    cohort["n_incorrect_state_beliefs"] = error_counts
    cohort["belief_error_group"] = np.where(
        cohort["n_incorrect_state_beliefs"].eq(0),
        "all_correct",
        "at_least_one_incorrect",
    )
    cohort = cohort.sort_values(["trajectory_id", "example_id"]).reset_index(drop=True)
    cohort.insert(0, "scaled_order", np.arange(1, len(cohort) + 1))
    return cohort


def _preparation_config(config: ClauseExperimentConfig) -> dict[str, Any]:
    return {
        "analysis": "belief_clause_intervention_matched46_v1",
        "candidate_rows_path": str(config.candidate_rows_path.resolve()),
        "candidate_rows_sha256": _sha256(config.candidate_rows_path),
        "prefix_action_rows_path": str(config.prefix_action_rows_path.resolve()),
        "prefix_action_rows_sha256": _sha256(config.prefix_action_rows_path),
        "belief_rows_path": str(config.belief_rows_path.resolve()),
        "belief_rows_sha256": _sha256(config.belief_rows_path),
        "selection": "all_matched_46_examples",
        "counterfactual_selection": "one_wall_fact_by_seeded_example_hash",
        "seed": config.seed,
    }


def prepare_experiment(config: ClauseExperimentConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _check_or_write_config(
        config.output_dir / "preparation_manifest.json", _preparation_config(config)
    )
    cohort = select_scaled_cohort(
        pd.read_csv(config.candidate_rows_path),
        pd.read_csv(config.prefix_action_rows_path),
        pd.read_csv(config.belief_rows_path),
        seed=config.seed,
    )
    path = config.output_dir / "cohort.csv"
    cohort.to_csv(path, index=False)
    return path


def _format_belief_lines(
    beliefs: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    lines: list[str] = []
    for question_id in BELIEF_IDS:
        probabilities = parse_probabilities(beliefs[question_id], PROBABILITY_LABELS)
        lines.append(
            f"- {BELIEF_LABELS[question_id]}: "
            f"yes={probabilities['yes']:.6f}, no={probabilities['no']:.6f}, "
            f"unknown={probabilities['unknown']:.6f}"
        )
    return lines


def build_clause_context(
    *,
    grid_text: str,
    carrying_key: Any,
    condition: str,
    model_beliefs: Mapping[str, Mapping[str, Any]],
    verified_beliefs: Mapping[str, Mapping[str, Any]],
    counterfactual_beliefs: Mapping[str, Mapping[str, Any]],
) -> str:
    """Render a no-CoT context for one preregistered clause condition."""
    if condition == "model_reports":
        heading = (
            "The original reasoning trace is unavailable. It is replaced by six "
            "model-generated belief readouts. They may be uncertain or mistaken."
        )
        lines = _format_belief_lines(model_beliefs)
    elif condition in {"verified_truth", "counterfactual_wall"}:
        heading = (
            "The original reasoning trace is unavailable. It is replaced by six "
            "categorical readouts supplied by a trusted external verifier."
        )
        payload = (
            verified_beliefs
            if condition == "verified_truth"
            else counterfactual_beliefs
        )
        lines = _format_belief_lines(payload)
    elif condition == "irrelevant_control":
        heading = (
            "The original reasoning trace is unavailable. It is replaced by six "
            "categorical readouts supplied by a trusted external verifier."
        )
        lines = [
            f"- {label}: yes=1.000000, no=0.000000, unknown=0.000000"
            for label in IRRELEVANT_LABELS
        ]
    else:
        raise ValueError(f"unknown clause condition: {condition}")
    return (
        _behavioral_probe_preamble("cardinal_action_explicit")
        + "\n# Inputs\n\nCurrent grid state:\n\n"
        + str(grid_text)
        + "\n\nAgent status:\n- Carrying key: "
        + str(_parse_bool(carrying_key)).lower()
        + "\n\nReplacement readouts:\n"
        + heading
        + " Each line gives probabilities over yes, no, and unknown.\n"
        + "\n".join(lines)
        + "\n\n"
    )


def _query_config(config: ClauseExperimentConfig, cohort_path: Path) -> dict[str, Any]:
    return {
        "analysis": "belief_clause_intervention_matched46_v1",
        "cohort_sha256": _sha256(cohort_path),
        "conditions": list(CONDITIONS),
        "model_path": str(config.model_path.resolve()),
        "temperature": config.temperature,
        "reasoning_effort": "low",
        "scoring": "direct_candidate_token_probabilities",
        "gpu_index": config.gpu_index,
        "minimum_free_gib": config.minimum_free_gib,
        "seed": config.seed,
    }


def _blocked_report(config: ClauseExperimentConfig, status: Mapping[str, Any]) -> Path:
    path = config.output_dir / "run_report.md"
    path.write_text(
        "# Matched-46 belief-clause intervention\n\n"
        f"**Status: `{status.get('status', 'blocked')}`.**\n\n"
        f"{status.get('message', 'Insufficient free GPU memory.')}\n"
    )
    return path


def query_experiment(
    config: ClauseExperimentConfig,
    *,
    reader_factory: Callable[..., Any] | None = None,
    gpu_status_fn: Callable[[int, float], Mapping[str, Any]] = gpu_status,
) -> dict[str, Any]:
    """Score every example/condition pair, resuming an fsynced checkpoint."""
    cohort_path = config.output_dir / "cohort.csv"
    if not cohort_path.exists():
        raise FileNotFoundError("run the prepare stage before query")
    query_config = _query_config(config, cohort_path)
    _check_or_write_config(config.output_dir / "query_config.json", query_config)
    checkpoint = config.output_dir / "action_readouts.jsonl"
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed[(str(row["example_id"]), str(row["condition"]))] = row

    cohort = pd.read_csv(cohort_path)
    planned = [
        (str(row.example_id), condition)
        for row in cohort.itertuples(index=False)
        for condition in CONDITIONS
    ]
    pending = [key for key in planned if key not in completed]
    manifest = {
        **query_config,
        "planned_readouts": len(planned),
        "completed_readouts": len(completed),
        "pending_readouts": len(pending),
    }
    if not pending:
        manifest["status"] = "completed"
        _write_json(config.output_dir / "run_manifest.json", manifest)
        return manifest

    status = dict(gpu_status_fn(config.gpu_index, config.minimum_free_gib))
    manifest.update(status)
    _write_json(config.output_dir / "run_manifest.json", manifest)
    if status.get("status") != "ready":
        _blocked_report(config, status)
        return manifest
    if reader_factory is None:
        if config.gpu_index != 0:
            raise ValueError("launch with CUDA_VISIBLE_DEVICES set to the requested GPU")
        from reveng.experiments.local_immediate_behavioral import LocalImmediateReadout

        reader_factory = LocalImmediateReadout
    reader = reader_factory(config.model_path, temperature=config.temperature)
    cohort_lookup = cohort.set_index("example_id")
    with checkpoint.open("a") as handle:
        for example_id, condition in pending:
            row = cohort_lookup.loc[example_id]
            context = build_clause_context(
                grid_text=str(row["grid_text"]),
                carrying_key=row["carrying_key"],
                condition=condition,
                model_beliefs=json.loads(str(row["model_beliefs_json"])),
                verified_beliefs=json.loads(str(row["verified_beliefs_json"])),
                counterfactual_beliefs=json.loads(
                    str(row["counterfactual_beliefs_json"])
                ),
            )
            result = reader.score_questions(
                context=context, questions=[action_question()]
            )["action"]
            output = {
                "example_id": example_id,
                "trajectory_id": str(row["trajectory_id"]),
                "condition": condition,
                "action": str(result["answer"]),
                "action_probabilities": result["probabilities"],
                "action_entropy_bits": float(result["entropy_bits"]),
                "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                "context_characters": len(context),
                "counterfactual_question_id": (
                    str(row["counterfactual_question_id"])
                    if condition == "counterfactual_wall"
                    else None
                ),
            }
            handle.write(json.dumps(output, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            completed[(example_id, condition)] = output
            manifest["completed_readouts"] = len(completed)
            manifest["pending_readouts"] = len(planned) - len(completed)
            _write_json(config.output_dir / "run_manifest.json", manifest)
    manifest["status"] = "completed"
    _write_json(config.output_dir / "run_manifest.json", manifest)
    return manifest


def build_analysis_rows(
    cohort: pd.DataFrame, readouts: Sequence[Mapping[str, Any]]
) -> pd.DataFrame:
    lookup = {
        (str(row["example_id"]), str(row["condition"])): row for row in readouts
    }
    output: list[dict[str, Any]] = []
    for source in cohort.to_dict("records"):
        full_probabilities = parse_probabilities(
            source["full_cot_action_probabilities_json"], ACTIONS
        )
        grid_probabilities = parse_probabilities(
            source["grid_only_action_probabilities_json"], ACTIONS
        )
        optimal_actions = set(json.loads(str(source["optimal_actions_json"])))
        for condition in CONDITIONS:
            key = (str(source["example_id"]), condition)
            if key not in lookup:
                raise ValueError(f"missing action readout for {key}")
            readout = lookup[key]
            condition_probabilities = parse_probabilities(
                readout["action_probabilities"], ACTIONS
            )
            action = str(readout["action"])
            output.append(
                {
                    "scaled_order": int(source["scaled_order"]),
                    "example_id": source["example_id"],
                    "trajectory_id": source["trajectory_id"],
                    "condition": condition,
                    "n_incorrect_state_beliefs": int(
                        source["n_incorrect_state_beliefs"]
                    ),
                    "belief_error_group": source["belief_error_group"],
                    "counterfactual_question_id": source[
                        "counterfactual_question_id"
                    ],
                    "optimal_actions_json": json.dumps(sorted(optimal_actions)),
                    "full_cot_action": source["full_cot_action"],
                    "grid_only_action": source["grid_only_action"],
                    "condition_action": action,
                    "full_cot_is_optimal": source["full_cot_action"]
                    in optimal_actions,
                    "grid_only_is_optimal": source["grid_only_action"]
                    in optimal_actions,
                    "condition_is_optimal": action in optimal_actions,
                    "condition_disagrees_with_full_cot": action
                    != source["full_cot_action"],
                    "condition_differs_from_grid_only": action
                    != source["grid_only_action"],
                    "condition_rescues_full_cot_action": (
                        source["grid_only_action"] != source["full_cot_action"]
                        and action == source["full_cot_action"]
                    ),
                    "condition_corrects_suboptimal_full_cot": (
                        source["full_cot_action"] not in optimal_actions
                        and action in optimal_actions
                    ),
                    "condition_degrades_optimal_full_cot": (
                        source["full_cot_action"] in optimal_actions
                        and action not in optimal_actions
                    ),
                    "full_cot_action_probabilities_json": json.dumps(
                        full_probabilities, sort_keys=True
                    ),
                    "grid_only_action_probabilities_json": json.dumps(
                        grid_probabilities, sort_keys=True
                    ),
                    "condition_action_probabilities_json": json.dumps(
                        condition_probabilities, sort_keys=True
                    ),
                    "grid_only_total_variation_from_full_cot": total_variation(
                        grid_probabilities, full_probabilities
                    ),
                    "condition_total_variation_from_full_cot": total_variation(
                        condition_probabilities, full_probabilities
                    ),
                    "condition_total_variation_from_grid_only": total_variation(
                        condition_probabilities, grid_probabilities
                    ),
                    "condition_js_divergence_from_full_cot_bits": (
                        jensen_shannon_divergence(
                            condition_probabilities, full_probabilities
                        )
                    ),
                }
            )
    return pd.DataFrame(output).sort_values(["condition", "scaled_order"])


def _cluster_interval(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    seed: int,
    repetitions: int,
) -> tuple[float, float]:
    unique = np.unique(clusters)
    grouped = {cluster: values[clusters == cluster] for cluster in unique}
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        estimates[index] = np.concatenate(
            [grouped[item] for item in sampled]
        ).mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def summarize_analysis(
    rows: pd.DataFrame, *, seed: int = 42, repetitions: int = 5000
) -> pd.DataFrame:
    primary = rows.loc[rows["condition"].eq(CONDITIONS[0])]
    inputs: list[tuple[str, pd.Series, pd.Series, pd.Series]] = [
        (
            "full_cot",
            pd.Series(False, index=primary.index),
            pd.Series(0.0, index=primary.index),
            primary["full_cot_is_optimal"],
        ),
        (
            "grid_only",
            primary["grid_only_action"] != primary["full_cot_action"],
            primary["grid_only_total_variation_from_full_cot"],
            primary["grid_only_is_optimal"],
        ),
    ]
    for condition in CONDITIONS:
        subset = rows.loc[rows["condition"].eq(condition)]
        inputs.append(
            (
                condition,
                subset["condition_disagrees_with_full_cot"],
                subset["condition_total_variation_from_full_cot"],
                subset["condition_is_optimal"],
            )
        )
    clusters = primary["trajectory_id"].astype(str).to_numpy()
    records: list[dict[str, Any]] = []
    for condition, disagreement, variation, optimal in inputs:
        disagreement_values = disagreement.astype(float).to_numpy()
        variation_values = variation.astype(float).to_numpy()
        optimal_values = optimal.astype(float).to_numpy()
        disagreement_ci = _cluster_interval(
            disagreement_values, clusters, seed=seed, repetitions=repetitions
        )
        variation_ci = _cluster_interval(
            variation_values, clusters, seed=seed + 1, repetitions=repetitions
        )
        optimal_ci = _cluster_interval(
            optimal_values, clusters, seed=seed + 2, repetitions=repetitions
        )
        records.append(
            {
                "condition": condition,
                "n_examples": len(primary),
                "n_trajectories": int(primary["trajectory_id"].nunique()),
                "n_disagree_with_full_cot": int(disagreement_values.sum()),
                "disagreement_rate": float(disagreement_values.mean()),
                "disagreement_ci_low": disagreement_ci[0],
                "disagreement_ci_high": disagreement_ci[1],
                "mean_total_variation_from_full_cot": float(
                    variation_values.mean()
                ),
                "mean_total_variation_ci_low": variation_ci[0],
                "mean_total_variation_ci_high": variation_ci[1],
                "n_optimal": int(optimal_values.sum()),
                "optimal_rate": float(optimal_values.mean()),
                "optimal_rate_ci_low": optimal_ci[0],
                "optimal_rate_ci_high": optimal_ci[1],
            }
        )
    return pd.DataFrame(records)


def build_pairwise_contrasts(
    cohort: pd.DataFrame,
    readouts: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    repetitions: int = 5000,
) -> pd.DataFrame:
    query_lookup = {
        (str(row["example_id"]), str(row["condition"])): row for row in readouts
    }
    pairs = (
        ("model_reports", "grid_only"),
        ("model_reports", "irrelevant_control"),
        ("model_reports", "verified_truth"),
        ("verified_truth", "grid_only"),
        ("counterfactual_wall", "verified_truth"),
        ("irrelevant_control", "verified_truth"),
        ("verified_truth", "full_cot"),
    )
    records: list[dict[str, Any]] = []
    for pair_index, (condition_a, condition_b) in enumerate(pairs):
        differences: list[float] = []
        variations: list[float] = []
        optimal_differences: list[float] = []
        clusters: list[str] = []
        for source in cohort.to_dict("records"):
            optimal = set(json.loads(str(source["optimal_actions_json"])))

            def values(condition: str) -> tuple[str, dict[str, float]]:
                if condition == "full_cot":
                    return str(source["full_cot_action"]), parse_probabilities(
                        source["full_cot_action_probabilities_json"], ACTIONS
                    )
                if condition == "grid_only":
                    return str(source["grid_only_action"]), parse_probabilities(
                        source["grid_only_action_probabilities_json"], ACTIONS
                    )
                row = query_lookup[(str(source["example_id"]), condition)]
                return str(row["action"]), parse_probabilities(
                    row["action_probabilities"], ACTIONS
                )

            action_a, probabilities_a = values(condition_a)
            action_b, probabilities_b = values(condition_b)
            differences.append(float(action_a != action_b))
            variations.append(total_variation(probabilities_a, probabilities_b))
            optimal_differences.append(
                float(action_a in optimal) - float(action_b in optimal)
            )
            clusters.append(str(source["trajectory_id"]))
        difference_values = np.asarray(differences)
        variation_values = np.asarray(variations)
        optimal_values = np.asarray(optimal_differences)
        cluster_values = np.asarray(clusters)
        difference_ci = _cluster_interval(
            difference_values,
            cluster_values,
            seed=seed + 10 * pair_index,
            repetitions=repetitions,
        )
        variation_ci = _cluster_interval(
            variation_values,
            cluster_values,
            seed=seed + 10 * pair_index + 1,
            repetitions=repetitions,
        )
        optimal_ci = _cluster_interval(
            optimal_values,
            cluster_values,
            seed=seed + 10 * pair_index + 2,
            repetitions=repetitions,
        )
        records.append(
            {
                "condition_a": condition_a,
                "condition_b": condition_b,
                "n_examples": len(cohort),
                "n_actions_differ": int(difference_values.sum()),
                "action_difference_rate": float(difference_values.mean()),
                "action_difference_ci_low": difference_ci[0],
                "action_difference_ci_high": difference_ci[1],
                "mean_total_variation": float(variation_values.mean()),
                "mean_total_variation_ci_low": variation_ci[0],
                "mean_total_variation_ci_high": variation_ci[1],
                "optimal_rate_a_minus_b": float(optimal_values.mean()),
                "optimal_difference_ci_low": optimal_ci[0],
                "optimal_difference_ci_high": optimal_ci[1],
            }
        )
    return pd.DataFrame(records)


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _write_report(
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    output: Path,
) -> None:
    indexed = summary.set_index("condition")
    contrast_lookup = contrasts.set_index(["condition_a", "condition_b"])
    verified = indexed.loc["verified_truth"]
    model = indexed.loc["model_reports"]
    grid = indexed.loc["grid_only"]
    counterfactual = contrast_lookup.loc[
        ("counterfactual_wall", "verified_truth")
    ]
    verifier_gap = contrast_lookup.loc[("verified_truth", "full_cot")]
    verified_rows = rows.loc[rows["condition"].eq("verified_truth")]
    corrected = int(
        verified_rows["condition_corrects_suboptimal_full_cot"].sum()
    )
    degraded = int(verified_rows["condition_degrades_optimal_full_cot"].sum())
    labels = {
        "full_cot": "Full CoT",
        "grid_only": "Grid only",
        "model_reports": "Model reports",
        "verified_truth": "Verified truth",
        "counterfactual_wall": "One flipped wall fact",
        "irrelevant_control": "Irrelevant control",
    }
    table = [
        "| Condition | Differs from full CoT | Mean TV from full CoT | Optimal top action |",
        "|---|---:|---:|---:|",
    ]
    for condition, label in labels.items():
        row = indexed.loc[condition]
        table.append(
            f"| {label} | {int(row.n_disagree_with_full_cot)}/"
            f"{int(row.n_examples)} ({_percent(row.disagreement_rate)}) | "
            f"{row.mean_total_variation_from_full_cot:.3f} | "
            f"{int(row.n_optimal)}/{int(row.n_examples)} "
            f"({_percent(row.optimal_rate)}) |"
        )
    output.write_text(
        "# Matched-46 belief-replacement and verifier-clause experiment\n\n"
        "## Design\n\n"
        f"The cohort contains {int(verified.n_examples)} decisions from "
        f"{int(verified.n_trajectories)} trajectories. Every query retains the "
        "original grid and key status, removes the CoT, and adds one preregistered "
        "replacement block. Actions are scored from candidate-token probabilities "
        "at temperature 0.7.\n\n"
        + "\n".join(table)
        + "\n\n## Main comparisons\n\n"
        f"Model reports disagree with the full-CoT top action in "
        f"{int(model.n_disagree_with_full_cot)}/{int(model.n_examples)} cases, "
        f"compared with {int(grid.n_disagree_with_full_cot)}/{int(grid.n_examples)} "
        "for grid only. Verified clauses disagree with full CoT in "
        f"{int(verified.n_disagree_with_full_cot)}/{int(verified.n_examples)} cases. "
        f"Their top action is optimal in {int(verified.n_optimal)}/"
        f"{int(verified.n_examples)} cases. Relative to full CoT, verified clauses "
        f"correct {corrected} suboptimal actions and degrade {degraded} optimal "
        "actions.\n\n"
        f"The verified-clause and full-CoT actions differ in "
        f"{int(verifier_gap.n_actions_differ)}/{int(verifier_gap.n_examples)} cases "
        f"(mean TV {verifier_gap.mean_total_variation:.3f}). Flipping one "
        "deterministically selected wall fact changes the action relative to the "
        f"verified-truth condition in {int(counterfactual.n_actions_differ)}/"
        f"{int(counterfactual.n_examples)} cases (mean TV "
        f"{counterfactual.mean_total_variation:.3f}).\n\n"
        "Intervals in summary.csv and pairwise_contrasts.csv use "
        "trajectory-clustered bootstrap resampling.\n\n"
        "## Interpretation limits\n\n"
        "This is a prompt intervention, not evidence that the elicited clauses are "
        "the latent beliefs used during the original forward pass. Verified facts "
        "are also visible in the grid, so improvements measure whether an explicit "
        "trusted summary changes action selection. The counterfactual condition is "
        "a sensitivity test; it is not a valid task state. Direct candidate scoring "
        "is deterministic, so repeated sampling seeds would not measure generation "
        "variance.\n"
    )


def analyze_experiment(config: ClauseExperimentConfig) -> dict[str, Path]:
    cohort_path = config.output_dir / "cohort.csv"
    checkpoint = config.output_dir / "action_readouts.jsonl"
    if not cohort_path.exists() or not checkpoint.exists():
        raise FileNotFoundError("prepare and query stages must complete before analysis")
    cohort = pd.read_csv(cohort_path)
    readouts = [
        json.loads(line) for line in checkpoint.read_text().splitlines() if line
    ]
    rows = build_analysis_rows(cohort, readouts)
    summary = summarize_analysis(
        rows, seed=config.seed, repetitions=config.bootstrap_repetitions
    )
    contrasts = build_pairwise_contrasts(
        cohort,
        readouts,
        seed=config.seed,
        repetitions=config.bootstrap_repetitions,
    )
    subgroups = _summarize_subgroups(rows)
    paths = {
        "cohort": cohort_path,
        "readouts": checkpoint,
        "rows": config.output_dir / "analysis_rows.csv",
        "summary": config.output_dir / "summary.csv",
        "contrasts": config.output_dir / "pairwise_contrasts.csv",
        "subgroups": config.output_dir / "subgroup_summary.csv",
        "report": config.output_dir / "run_report.md",
    }
    rows.to_csv(paths["rows"], index=False)
    summary.to_csv(paths["summary"], index=False)
    contrasts.to_csv(paths["contrasts"], index=False)
    subgroups.to_csv(paths["subgroups"], index=False)
    _write_report(rows, summary, contrasts, paths["report"])
    return paths


def run_all(config: ClauseExperimentConfig) -> dict[str, Any]:
    prepare_experiment(config)
    manifest = query_experiment(config)
    if manifest.get("status") != "completed":
        return {
            "manifest": config.output_dir / "run_manifest.json",
            "status": manifest,
        }
    return {"paths": analyze_experiment(config), "status": manifest}


__all__ = [
    "CONDITIONS",
    "ClauseExperimentConfig",
    "analyze_experiment",
    "build_analysis_rows",
    "build_clause_context",
    "build_pairwise_contrasts",
    "prepare_experiment",
    "query_experiment",
    "run_all",
    "select_scaled_cohort",
    "summarize_analysis",
]


def _summarize_subgroups(rows: pd.DataFrame) -> pd.DataFrame:
    """Exploratory exact counts split by final report correctness."""
    records: list[dict[str, Any]] = []
    for group_name, grouped_rows in rows.groupby("belief_error_group", sort=True):
        primary = grouped_rows.loc[grouped_rows["condition"].eq(CONDITIONS[0])]
        inputs: list[tuple[str, pd.Series, pd.Series]] = [
            (
                "full_cot",
                pd.Series(False, index=primary.index),
                primary["full_cot_is_optimal"],
            ),
            (
                "grid_only",
                primary["grid_only_action"] != primary["full_cot_action"],
                primary["grid_only_is_optimal"],
            ),
        ]
        for condition in CONDITIONS:
            subset = grouped_rows.loc[grouped_rows["condition"].eq(condition)]
            inputs.append(
                (
                    condition,
                    subset["condition_disagrees_with_full_cot"],
                    subset["condition_is_optimal"],
                )
            )
        for condition, disagreement, optimal in inputs:
            records.append(
                {
                    "belief_error_group": group_name,
                    "condition": condition,
                    "n_examples": len(primary),
                    "n_trajectories": int(primary["trajectory_id"].nunique()),
                    "n_disagree_with_full_cot": int(disagreement.astype(bool).sum()),
                    "disagreement_rate": float(disagreement.astype(float).mean()),
                    "n_optimal": int(optimal.astype(bool).sum()),
                    "optimal_rate": float(optimal.astype(float).mean()),
                }
            )
    return pd.DataFrame(records)
