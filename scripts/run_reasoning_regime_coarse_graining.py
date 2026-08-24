#!/usr/bin/env python3
"""Run a Fields-inspired behavioral reasoning-regime coarse-graining pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reveng.experiments.reasoning_regimes import (
    ACTION_COLUMNS,
    REGIMES,
    assign_regimes,
    smoothed_transition_probabilities,
    summarize_trace,
    transition_rows,
)
from reveng.experiments.reasoning_topology import paired_inference


DEFAULT_POSITIONS = Path(
    "outputs/hypothesis_tests/action_distribution_change_points_v1/"
    "position_distribution_metrics.csv"
)
DEFAULT_TOPOLOGY = Path(
    "outputs/hypothesis_tests/reasoning_topology_v1/trace_metrics.csv"
)
DEFAULT_OUTPUT = Path("outputs/hypothesis_tests/reasoning_regime_coarse_graining_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=Path, default=DEFAULT_POSITIONS)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoothing-alpha", type=float, default=0.5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    ordered = values.sort_values()
    ranks = np.arange(1, len(ordered) + 1, dtype=float)
    adjusted = ordered.to_numpy(dtype=float) * len(ordered) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = pd.Series(np.minimum(adjusted, 1.0), index=ordered.index)
    return result.reindex(values.index)


def validation_groups(metadata: pd.DataFrame) -> dict[str, str]:
    """Connect matched pairs sharing any source trajectory."""
    pair_ids = [str(value) for value in metadata.matched_pair_id.unique()]
    parent = {pair_id: pair_id for pair_id in pair_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for _, rows in metadata.groupby("trajectory_id"):
        shared = [str(value) for value in rows.matched_pair_id.unique()]
        for pair_id in shared[1:]:
            union(shared[0], pair_id)
    roots = {pair_id: find(pair_id) for pair_id in pair_ids}
    canonical = {
        root: f"validation_group_{index:02d}"
        for index, root in enumerate(sorted(set(roots.values())), start=1)
    }
    return {pair_id: canonical[root] for pair_id, root in roots.items()}


def build_regime_tables(
    positions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    position_frames: list[pd.DataFrame] = []
    transition_frames: list[pd.DataFrame] = []
    trace_rows: list[dict[str, Any]] = []
    metadata_columns = (
        "example_id",
        "trajectory_id",
        "matched_pair_id",
        "matched_role",
        "trajectory_class",
        "primary_step_failure_mode",
    )
    for example_id, source in positions.groupby("example_id", sort=True):
        labelled = assign_regimes(source)
        meta = {column: labelled.iloc[0][column] for column in metadata_columns}
        position_frames.append(labelled)
        transitions = transition_rows(labelled)
        for column, value in meta.items():
            transitions[column] = value
        transition_frames.append(transitions)
        trace_rows.append({**meta, **summarize_trace(labelled)})
    return (
        pd.concat(position_frames, ignore_index=True),
        pd.concat(transition_frames, ignore_index=True),
        pd.DataFrame(trace_rows),
    )


def score_transition_surprisal(
    transitions: pd.DataFrame,
    *,
    alpha: float,
) -> pd.DataFrame:
    """Score each edge from controls outside its trajectory-connected group."""
    scored: list[pd.DataFrame] = []
    for validation_group, held_out in transitions.groupby("validation_group"):
        training = transitions[
            (transitions.validation_group != validation_group)
            & (transitions.matched_role == "control")
        ]
        probabilities = smoothed_transition_probabilities(training, alpha=alpha)
        fold = held_out.copy()
        fold["control_transition_probability"] = [
            probabilities[(current, following)]
            for current, following in zip(
                fold.current_regime, fold.next_regime, strict=True
            )
        ]
        fold["transition_surprisal_bits"] = -np.log2(
            fold["control_transition_probability"].astype(float)
        )
        scored.append(fold)
    output = pd.concat(scored, ignore_index=True)
    control_nonself = output[
        (output.matched_role == "control")
        & ~output.is_self_transition
        & ~output.enters_stable_selection
    ]
    control_cutoff = float(control_nonself.transition_surprisal_bits.quantile(0.95))
    output["unusual_transition"] = (
        ~output.is_self_transition
        & ~output.enters_stable_selection
        & output.transition_surprisal_bits.gt(control_cutoff)
    )
    output["unusual_cutoff_bits"] = control_cutoff
    return output


def graph_edges(transitions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for role, group in transitions.groupby("matched_role", sort=True):
        counts = group.groupby(["current_regime", "next_regime"]).size()
        totals = group.groupby("current_regime").size()
        trajectory_support = group.groupby(
            ["current_regime", "next_regime"]
        ).trajectory_id.nunique()
        for current in REGIMES:
            for following in REGIMES:
                count = int(counts.get((current, following), 0))
                rows.append(
                    {
                        "matched_role": role,
                        "current_regime": current,
                        "next_regime": following,
                        "edge_count": count,
                        "source_total": int(totals.get(current, 0)),
                        "transition_probability": (
                            count / int(totals[current])
                            if current in totals
                            else np.nan
                        ),
                        "supporting_trajectories": int(
                            trajectory_support.get((current, following), 0)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def one_hot(values: pd.Series, categories: tuple[str, ...]) -> np.ndarray:
    mapping = {value: index for index, value in enumerate(categories)}
    output = np.zeros((len(values), len(categories)), dtype=float)
    for row_index, value in enumerate(values.astype(str)):
        if value in mapping:
            output[row_index, mapping[value]] = 1.0
    return output


def feature_matrix(frame: pd.DataFrame, model: str) -> np.ndarray:
    progress = frame[["reasoning_progress"]].to_numpy(dtype=float)
    if model == "progress_only":
        return progress
    current = one_hot(frame.current_regime, REGIMES)
    if model == "first_order_regime":
        return np.column_stack([progress, current])
    if model == "second_order_regime":
        previous_categories = ("trace_start", *REGIMES)
        previous = one_hot(frame.previous_regime, previous_categories)
        return np.column_stack([progress, current, previous])
    if model == "full_action_distribution":
        action_columns = [*ACTION_COLUMNS, "computed_action_entropy_bits"]
        return np.column_stack([progress, frame[action_columns].to_numpy(dtype=float)])
    raise ValueError(f"unknown predictive model: {model}")


def grouped_predictive_closure(
    transitions: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate next-regime prediction before stable selection."""
    data = transitions[transitions.current_regime.str.startswith("unsettled_")].copy()
    class_index = {regime: index for index, regime in enumerate(REGIMES)}
    data["target_index"] = data.next_regime.map(class_index).astype(int)
    trace_sizes = data.groupby("example_id").size()
    data["fit_weight"] = data.example_id.map(lambda value: 1.0 / trace_sizes[value])
    models = (
        "progress_only",
        "first_order_regime",
        "second_order_regime",
        "full_action_distribution",
    )
    prediction_rows: list[dict[str, Any]] = []
    for validation_group in sorted(data.validation_group.unique()):
        training = data[data.validation_group != validation_group]
        testing = data[data.validation_group == validation_group]
        for model_name in models:
            x_train = feature_matrix(training, model_name)
            x_test = feature_matrix(testing, model_name)
            y_train = training.target_index.to_numpy(dtype=int)
            probabilities = np.full((len(testing), len(REGIMES)), 1e-12)
            observed_classes = np.unique(y_train)
            if len(observed_classes) == 1:
                probabilities[:, observed_classes[0]] = 1.0
            else:
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=2_000,
                        random_state=seed,
                    ),
                )
                model.fit(
                    x_train,
                    y_train,
                    logisticregression__sample_weight=training.fit_weight.to_numpy(
                        dtype=float
                    ),
                )
                fold_probabilities = model.predict_proba(x_test)
                for source_column, class_value in enumerate(
                    model.named_steps["logisticregression"].classes_
                ):
                    probabilities[:, int(class_value)] = fold_probabilities[
                        :, source_column
                    ]
                probabilities /= probabilities.sum(axis=1, keepdims=True)
            targets = testing.target_index.to_numpy(dtype=int)
            predicted = np.argmax(probabilities, axis=1)
            for source, target, prediction, probability in zip(
                testing.itertuples(index=False),
                targets,
                predicted,
                probabilities,
                strict=True,
            ):
                prediction_rows.append(
                    {
                        "example_id": source.example_id,
                        "trajectory_id": source.trajectory_id,
                        "matched_pair_id": source.matched_pair_id,
                        "matched_role": source.matched_role,
                        "validation_group": source.validation_group,
                        "position_index": source.position_index,
                        "model": model_name,
                        "target_regime": REGIMES[target],
                        "predicted_regime": REGIMES[prediction],
                        "correct": bool(target == prediction),
                        "negative_log_likelihood": float(
                            -np.log(max(probability[target], 1e-12))
                        ),
                    }
                )
    predictions = pd.DataFrame(prediction_rows)
    per_trace = predictions.groupby(["model", "example_id"], as_index=False).agg(
        validation_group=("validation_group", "first"),
        matched_role=("matched_role", "first"),
        mean_negative_log_likelihood=("negative_log_likelihood", "mean"),
        accuracy=("correct", "mean"),
        n_transitions=("correct", "size"),
    )
    baseline = per_trace[per_trace.model == "progress_only"].set_index("example_id")
    first_order = per_trace[per_trace.model == "first_order_regime"].set_index(
        "example_id"
    )
    summary_rows: list[dict[str, Any]] = []
    for model_name in models:
        subset = per_trace[per_trace.model == model_name].set_index("example_id")
        improvements = (
            baseline.mean_negative_log_likelihood - subset.mean_negative_log_likelihood
        )
        inference = paired_inference(
            improvements,
            groups=subset.validation_group,
            seed=seed,
        )
        versus_first = (
            first_order.mean_negative_log_likelihood
            - subset.mean_negative_log_likelihood
        )
        first_inference = paired_inference(
            versus_first,
            groups=subset.validation_group,
            seed=seed + 1,
        )
        summary_rows.append(
            {
                "model": model_name,
                "n_traces": int(len(subset)),
                "mean_log_loss": float(subset.mean_negative_log_likelihood.mean()),
                "mean_accuracy": float(subset.accuracy.mean()),
                "log_loss_improvement_vs_progress": float(improvements.mean()),
                "improvement_ci_low": inference["ci_low"],
                "improvement_ci_high": inference["ci_high"],
                "improvement_sign_flip_p": inference["sign_flip_p_two_sided"],
                "log_loss_improvement_vs_first_order": float(versus_first.mean()),
                "first_order_improvement_ci_low": first_inference["ci_low"],
                "first_order_improvement_ci_high": first_inference["ci_high"],
                "first_order_improvement_sign_flip_p": first_inference[
                    "sign_flip_p_two_sided"
                ],
            }
        )
    return predictions, pd.DataFrame(summary_rows)


def paired_failure_control(
    traces: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    metrics = {
        "recurrence_returns_per_100_positions": "higher_failure",
        "optimality_losses_per_100_positions": "higher_failure",
        "unusual_nonself_transition_fraction": "higher_failure",
        "maximum_unsettled_transition_surprisal_bits": "higher_failure",
        "stable_suffix_fraction": "lower_failure",
    }
    rows: list[dict[str, Any]] = []
    for metric, direction in metrics.items():
        wide = traces.pivot(
            index="matched_pair_id", columns="matched_role", values=metric
        )
        differences = wide.failure.astype(float) - wide.control.astype(float)
        pair_groups = (
            traces[["matched_pair_id", "validation_group"]]
            .drop_duplicates()
            .set_index("matched_pair_id")
        )
        inference = paired_inference(
            differences,
            groups=pair_groups.loc[differences.index, "validation_group"],
            seed=seed,
        )
        rows.append(
            {
                "metric": metric,
                "prespecified_direction": direction,
                "failure_mean": float(
                    traces.loc[traces.matched_role == "failure", metric]
                    .astype(float)
                    .mean()
                ),
                "control_mean": float(
                    traces.loc[traces.matched_role == "control", metric]
                    .astype(float)
                    .mean()
                ),
                **inference,
            }
        )
    summary = pd.DataFrame(rows)
    summary["bh_q"] = benjamini_hochberg(summary.sign_flip_p_two_sided)
    return summary


def resolution_sensitivity(positions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for example_id, source in positions.groupby("example_id", sort=True):
        ordered = source.sort_values("position_index").reset_index(drop=True)
        for label, indices in (
            ("sentence", np.arange(len(ordered))),
            (
                "stride_2",
                np.unique(np.r_[np.arange(0, len(ordered), 2), len(ordered) - 1]),
            ),
            (
                "stride_4",
                np.unique(np.r_[np.arange(0, len(ordered), 4), len(ordered) - 1]),
            ),
            (
                "progress_16",
                np.unique(
                    [
                        int(
                            np.abs(
                                ordered.reasoning_progress.to_numpy() - target
                            ).argmin()
                        )
                        for target in np.linspace(0.0, 1.0, 16)
                    ]
                ),
            ),
        ):
            sampled = assign_regimes(ordered.iloc[indices].copy())
            summary = summarize_trace(sampled)
            rows.append(
                {
                    "example_id": example_id,
                    "matched_role": ordered.iloc[0].matched_role,
                    "resolution": label,
                    "sampled_positions": len(sampled),
                    "recurrence_returns": summary["recurrence_returns"],
                    "recurrence_returns_per_100_positions": summary[
                        "recurrence_returns_per_100_positions"
                    ],
                    "optimality_losses": summary["optimality_losses"],
                    "optimality_losses_per_100_positions": summary[
                        "optimality_losses_per_100_positions"
                    ],
                    "final_action_optimal": summary["final_action_optimal"],
                    "stable_suffix_fraction": summary["stable_suffix_fraction"],
                }
            )
    return pd.DataFrame(rows)


def activation_overlay(traces: pd.DataFrame, topology_path: Path) -> pd.DataFrame:
    if not topology_path.exists():
        return pd.DataFrame()
    topology = pd.read_csv(topology_path)
    topology = topology[topology.layer == 15][
        [
            "example_id",
            "late_to_early_dispersion_ratio",
            "nonlocal_recurrence",
        ]
    ]
    merged = traces.merge(topology, on="example_id", validate="one_to_one")
    outcomes = (
        "recurrence_returns",
        "optimality_losses",
        "mean_transition_surprisal_bits",
        "stable_suffix_fraction",
    )
    rows: list[dict[str, Any]] = []
    for activation_metric in (
        "late_to_early_dispersion_ratio",
        "nonlocal_recurrence",
    ):
        for outcome in outcomes:
            correlation, p_value = spearmanr(
                merged[activation_metric].astype(float),
                merged[outcome].astype(float),
            )
            rows.append(
                {
                    "activation_metric": activation_metric,
                    "behavioral_metric": outcome,
                    "n_traces": len(merged),
                    "spearman_rho": float(correlation),
                    "uncorrected_p": float(p_value),
                }
            )
    output = pd.DataFrame(rows)
    output["bh_q"] = benjamini_hochberg(output.uncorrected_p)
    return output


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    labels = [column.replace("_", " ") for column in columns]
    lines = [
        "| " + " | ".join(labels) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    positions = pd.read_csv(args.positions)
    regime_positions, transitions, traces = build_regime_tables(positions)
    metadata = traces[
        ["example_id", "trajectory_id", "matched_pair_id", "matched_role"]
    ]
    group_by_pair = validation_groups(metadata)
    for frame in (regime_positions, transitions, traces):
        frame["validation_group"] = frame.matched_pair_id.map(group_by_pair)

    transitions = score_transition_surprisal(transitions, alpha=args.smoothing_alpha)
    transition_aggregates = transitions.groupby("example_id").agg(
        mean_transition_surprisal_bits=("transition_surprisal_bits", "mean"),
        maximum_transition_surprisal_bits=("transition_surprisal_bits", "max"),
        unusual_transition_count=("unusual_transition", "sum"),
        transition_count=("transition_surprisal_bits", "size"),
    )
    nonself_aggregates = (
        transitions[
            ~transitions.is_self_transition & ~transitions.enters_stable_selection
        ]
        .groupby("example_id")
        .agg(
            nonself_transition_count=("transition_surprisal_bits", "size"),
            maximum_unsettled_transition_surprisal_bits=(
                "transition_surprisal_bits",
                "max",
            ),
        )
    )
    transition_aggregates = transition_aggregates.join(nonself_aggregates)
    transition_aggregates["nonself_transition_count"] = transition_aggregates[
        "nonself_transition_count"
    ].fillna(0)
    transition_aggregates["maximum_unsettled_transition_surprisal_bits"] = (
        transition_aggregates["maximum_unsettled_transition_surprisal_bits"].fillna(0.0)
    )
    transition_aggregates["unusual_nonself_transition_fraction"] = (
        transition_aggregates["unusual_transition_count"]
        / transition_aggregates["nonself_transition_count"].clip(lower=1)
    )
    traces = traces.merge(
        transition_aggregates, on="example_id", how="left", validate="one_to_one"
    )

    edges = graph_edges(transitions)
    predictions, predictive_summary = grouped_predictive_closure(
        transitions, seed=args.seed
    )
    failure_summary = paired_failure_control(traces, seed=args.seed)
    resolution = resolution_sensitivity(positions)
    overlay = activation_overlay(traces, args.topology)
    unusual_edges = (
        transitions[transitions.unusual_transition]
        .groupby(["matched_role", "current_regime", "next_regime"], as_index=False)
        .agg(
            edge_count=("example_id", "size"),
            supporting_states=("example_id", "nunique"),
            supporting_trajectories=("trajectory_id", "nunique"),
        )
    )

    regime_positions.to_csv(args.output_dir / "regime_positions.csv", index=False)
    transitions.to_csv(args.output_dir / "transition_rows.csv", index=False)
    traces.to_csv(args.output_dir / "trace_regime_summary.csv", index=False)
    edges.to_csv(args.output_dir / "graph_edges.csv", index=False)
    predictions.to_csv(args.output_dir / "predictive_rows.csv", index=False)
    predictive_summary.to_csv(
        args.output_dir / "predictive_closure_summary.csv", index=False
    )
    failure_summary.to_csv(args.output_dir / "failure_control_summary.csv", index=False)
    resolution.to_csv(args.output_dir / "resolution_sensitivity.csv", index=False)
    overlay.to_csv(args.output_dir / "activation_overlay.csv", index=False)
    unusual_edges.to_csv(args.output_dir / "unusual_edge_summary.csv", index=False)

    first_order = predictive_summary[
        predictive_summary.model == "first_order_regime"
    ].iloc[0]
    second_order = predictive_summary[
        predictive_summary.model == "second_order_regime"
    ].iloc[0]
    strict_failures = traces[
        (traces.matched_role == "failure") & ~traces.final_action_optimal
    ]
    corrected = failure_summary[
        (failure_summary.bh_q < 0.05)
        & (
            (failure_summary.prespecified_direction == "higher_failure")
            & (failure_summary.mean_difference > 0)
            | (failure_summary.prespecified_direction == "lower_failure")
            & (failure_summary.mean_difference < 0)
        )
    ]
    recurrence_row = failure_summary[
        failure_summary.metric == "recurrence_returns_per_100_positions"
    ].iloc[0]
    loss_row = failure_summary[
        failure_summary.metric == "optimality_losses_per_100_positions"
    ].iloc[0]
    lines = [
        "# Fields-Inspired Reasoning-Regime Coarse-Graining Pilot",
        "",
        "## Question",
        "",
        "Can sentence-level action dynamics be coarse-grained into a small, reproducible behavioral state process, and do recurrence or unusual transitions distinguish rollout-failure states from exact-matched controls?",
        "",
        "## Deterministic state space",
        "",
        "Every valid position receives exactly one of four regimes: `unsettled_optimal`, `unsettled_suboptimal`, `stable_optimal`, or `stable_suboptimal`. Stability is the first position in the final constant argmax-action suffix. Optimality uses the stored DoorKey-aware planner label. Probability revision and optimality loss are transition events, not regimes.",
        "",
        "Behavioral recurrence is a return to a previously left regime after consecutive repeats have been collapsed. Transition surprisal is `-log2 P_control(next_regime | current_regime)`, estimated from controls outside the held-out trajectory-connected validation group with additive smoothing. To avoid calling every state change unusual merely because self-transitions dominate—and to avoid defining rarity from the final outcome—the descriptive unusual-transition cutoff is the 95th percentile among held-out-control non-self edges that remain unsettled. It is applied only to such edges.",
        "",
        "## Data and dependence",
        "",
        f"- States: {len(traces)} in {traces.matched_pair_id.nunique()} exact matched pairs",
        f"- Source trajectories: {traces.trajectory_id.nunique()}",
        f"- Trajectory-connected validation groups: {traces.validation_group.nunique()}",
        f"- Sentence-boundary transitions: {len(transitions)}",
        f"- Strict final-suboptimal failure states: {len(strict_failures)}",
        f"- Traces contributing to next-regime prediction: {int(predictive_summary.n_traces.max())} (one trace begins in its stable suffix)",
        "",
        "## Predictive closure before stable selection",
        "",
        "The held-out target is the next regime at positions that are not yet in the stable suffix. Each source state receives equal total fitting weight, and validation holds out complete trajectory-connected groups.",
        "",
        *markdown_table(
            predictive_summary,
            [
                "model",
                "mean_log_loss",
                "mean_accuracy",
                "log_loss_improvement_vs_progress",
                "improvement_ci_low",
                "improvement_ci_high",
                "improvement_sign_flip_p",
                "log_loss_improvement_vs_first_order",
            ],
        ),
        "",
        "## Failure-control contrasts",
        "",
        *markdown_table(
            failure_summary,
            [
                "metric",
                "failure_mean",
                "control_mean",
                "mean_difference",
                "ci_low",
                "ci_high",
                "sign_flip_p_two_sided",
                "bh_q",
            ],
        ),
        "",
        "## Decision",
        "",
        f"The first-order regime model changes held-out log loss versus progress alone by {first_order.log_loss_improvement_vs_progress:+.4f} (grouped interval [{first_order.improvement_ci_low:+.4f}, {first_order.improvement_ci_high:+.4f}]). Adding one more regime of history improves log loss over the first-order model by {second_order.log_loss_improvement_vs_first_order:+.4f} (grouped interval [{second_order.first_order_improvement_ci_low:+.4f}, {second_order.first_order_improvement_ci_high:+.4f}]). This rejects exact first-order closure in this pilot; an equivalence margin was not prespecified.",
        "",
        f"Failure minus control recurrence returns per 100 positions: {recurrence_row.mean_difference:+.4f} (interval [{recurrence_row.ci_low:+.4f}, {recurrence_row.ci_high:+.4f}]). Failure minus control optimality losses per 100 positions: {loss_row.mean_difference:+.4f} (interval [{loss_row.ci_low:+.4f}, {loss_row.ci_high:+.4f}]).",
        "",
        "In this four-state construction, behavioural recurrence is almost entirely alternation between `unsettled_optimal` and `unsettled_suboptimal`; recurrence and optimality-loss metrics are therefore overlapping descriptions, not independent mechanisms.",
        "",
        f"Rare unsettled edge types observed above the held-out-control cutoff: {len(unusual_edges)} aggregated role/edge combinations. With only two possible unsettled non-self edge types, this graph has little capacity to identify unusual transition structure beyond optimality oscillation.",
        "",
        (
            "Prespecified failure-control metrics surviving BH correction: "
            + ", ".join(corrected.metric)
            if len(corrected)
            else "No prespecified failure-control metric survives BH correction at q < 0.05."
        ),
        "",
        "This is an offline coarse-graining pilot, not evidence of thermodynamic entropy production or a literal arrow of time. Stable selection is retrospective, self-transitions are structurally common, and the strict final-suboptimal subset is too small for a confirmatory graph comparison. Semantic ReasoningFlow labels remain outside the primary graph because their manual validation is incomplete.",
        "",
        "## Next gate",
        "",
        "Confirm any supported transition pattern on new repeated continuations. In particular, continue low-stabilization strict failures for controlled additional lengths and test whether they enter `stable_optimal`, remain `stable_suboptimal`, or continue recurrent unsettled transitions.",
    ]
    (args.output_dir / "run_report.md").write_text("\n".join(lines) + "\n")
    write_json(
        args.output_dir / "run_manifest.json",
        {
            "analysis": "reasoning_regime_coarse_graining_v1",
            "positions": str(args.positions),
            "positions_sha256": sha256(args.positions),
            "topology": str(args.topology) if args.topology.exists() else None,
            "topology_sha256": (
                sha256(args.topology) if args.topology.exists() else None
            ),
            "regimes": list(REGIMES),
            "smoothing_alpha": args.smoothing_alpha,
            "unusual_transition_definition": "held-out-control 95th percentile of leave-group-out transition surprisal among non-self edges that do not enter stable selection; applied only to those edges",
            "uses_pca": False,
            "uses_clustering": False,
            "seed": args.seed,
            "counts": {
                "states": len(traces),
                "matched_pairs": int(traces.matched_pair_id.nunique()),
                "source_trajectories": int(traces.trajectory_id.nunique()),
                "validation_groups": int(traces.validation_group.nunique()),
                "transitions": len(transitions),
                "strict_final_suboptimal_failures": len(strict_failures),
            },
        },
    )


if __name__ == "__main__":
    main()
