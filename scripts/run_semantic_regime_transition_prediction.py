#!/usr/bin/env python3
"""Run persistence-aware semantic regime-transition prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reveng.experiments.reasoning_topology import paired_inference
from reveng.experiments.semantic_regime_prediction import (
    CUE_COLUMNS,
    DESTINATIONS,
    REGIMES,
    SEMANTIC_LABELS,
    add_history_features,
    combine_hurdle_probabilities,
    combine_semantic_runs,
    destination_type,
    semantic_feature_table,
)


DEFAULT_TRANSITIONS = Path(
    "outputs/hypothesis_tests/reasoning_regime_coarse_graining_v1/"
    "transition_rows.csv"
)
DEFAULT_SEMANTIC_ROOT = Path(
    "outputs/hypothesis_tests/semantic_reasoning_classification_v1/general_corpus_v1"
)
DEFAULT_ORIGINAL = "annotations_gpt_oss_20b_multilabel_v3_full.csv"
DEFAULT_REPLICATE = "annotations_gpt_oss_20b_multilabel_v3_full_replicate.csv"
DEFAULT_OUTPUT = Path(
    "outputs/hypothesis_tests/semantic_regime_transition_prediction_v1"
)
HISTORY_SIZES = (1, 2, 3, 5, 8)
SEMANTIC_CONFIGS = (
    "none",
    "original_all",
    "replicate_all",
    "intersection_all",
    "union_all",
    "mean_all",
    "mean_labels",
    "mean_cues",
)
ACTION_COLUMNS = (
    "prob_up",
    "prob_down",
    "prob_left",
    "prob_right",
    "computed_action_entropy_bits",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--semantic-root", type=Path, default=DEFAULT_SEMANTIC_ROOT)
    parser.add_argument("--original", default=DEFAULT_ORIGINAL)
    parser.add_argument("--replicate", default=DEFAULT_REPLICATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def one_hot(values: pd.Series, categories: tuple[str, ...]) -> np.ndarray:
    mapping = {value: index for index, value in enumerate(categories)}
    output = np.zeros((len(values), len(categories)), dtype=float)
    for row, value in enumerate(values.astype(str)):
        if value in mapping:
            output[row, mapping[value]] = 1.0
    return output


def semantic_columns(frame: pd.DataFrame, config: str) -> list[str]:
    if config == "none":
        return []
    if config.endswith("_labels"):
        return [column for column in frame if column.startswith("semantic__label__")]
    if config.endswith("_cues"):
        return [column for column in frame if column.startswith("semantic__cue__")]
    return [
        column
        for column in frame
        if column.startswith("semantic__label__")
        or column.startswith("semantic__cue__")
    ]


def feature_matrix(
    frame: pd.DataFrame,
    *,
    history_size: int,
    semantic_config: str,
) -> np.ndarray:
    numeric = frame[
        ["reasoning_progress", "regime_duration", *ACTION_COLUMNS]
    ].to_numpy(dtype=float)
    regime_categories = ("trace_start", *REGIMES)
    history = np.column_stack(
        [
            one_hot(frame[f"regime_lag_{lag}"], regime_categories)
            for lag in range(history_size)
        ]
    )
    selected_semantics = semantic_columns(frame, semantic_config)
    if not selected_semantics:
        return np.column_stack([numeric, history])
    semantic = frame[selected_semantics].to_numpy(dtype=float)
    current = one_hot(frame.current_regime, REGIMES)
    interactions = np.column_stack(
        [semantic * current[:, [index]] for index in range(len(REGIMES))]
    )
    return np.column_stack([numeric, history, semantic, interactions])


def fit_binary(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    classes = np.unique(y_train)
    if len(classes) == 1:
        return np.full(len(x_test), float(classes[0]))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2_000, random_state=seed),
    )
    model.fit(x_train, y_train, logisticregression__sample_weight=weights)
    return model.predict_proba(x_test)[:, 1]


def fit_destination(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    output = np.full((len(x_test), len(DESTINATIONS)), 1e-12, dtype=float)
    classes = np.unique(y_train)
    if len(classes) == 1:
        output[:, classes[0]] = 1.0
        return output / output.sum(axis=1, keepdims=True)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2_000, random_state=seed),
    )
    model.fit(x_train, y_train, logisticregression__sample_weight=weights)
    probabilities = model.predict_proba(x_test)
    for source_column, target_column in enumerate(
        model.named_steps["logisticregression"].classes_
    ):
        output[:, int(target_column)] = probabilities[:, source_column]
    return output / output.sum(axis=1, keepdims=True)


def prepare_semantics(
    original_path: Path,
    replicate_path: Path,
) -> dict[str, pd.DataFrame]:
    original = semantic_feature_table(pd.read_csv(original_path), "original")
    replicate = semantic_feature_table(pd.read_csv(replicate_path), "replicate")
    outputs: dict[str, pd.DataFrame] = {}
    for mode in ("original", "replicate", "intersection", "union", "mean"):
        outputs[mode] = combine_semantic_runs(original, replicate, mode)
    return outputs


def semantic_mode(config: str) -> str:
    return "mean" if config.startswith("mean_") else config.removesuffix("_all")


def prepare_analysis_frame(
    transitions: pd.DataFrame,
    semantics: pd.DataFrame,
    *,
    alignment: str,
) -> pd.DataFrame:
    sentence_position = (
        "position_index" if alignment == "prospective" else "next_position_index"
    )
    merged = transitions.merge(
        semantics,
        left_on=["example_id", sentence_position],
        right_on=["example_id", "sentence_number"],
        how="left",
        validate="many_to_one",
    )
    merged = merged[
        merged.current_regime.str.startswith("unsettled_")
        & merged.semantic_valid.eq(1.0)
    ].copy()
    merged["change_target"] = (~merged.is_self_transition).astype(int)
    merged["destination_name"] = [
        destination_type(current, following)
        for current, following in zip(
            merged.current_regime, merged.next_regime, strict=True
        )
    ]
    destination_index = {value: index for index, value in enumerate(DESTINATIONS)}
    merged["destination_target"] = merged.destination_name.map(destination_index)
    return merged


def trace_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("example_id").size()
    return frame.example_id.map(lambda value: 1.0 / counts[value]).to_numpy(dtype=float)


def run_specification(
    frame: pd.DataFrame,
    *,
    alignment: str,
    semantic_config: str,
    history_size: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    destination_index = {value: index for index, value in enumerate(DESTINATIONS)}
    regime_index = {value: index for index, value in enumerate(REGIMES)}
    for fold_number, validation_group in enumerate(
        sorted(frame.validation_group.unique())
    ):
        training = frame[frame.validation_group != validation_group]
        testing = frame[frame.validation_group == validation_group]
        x_train = feature_matrix(
            training,
            history_size=history_size,
            semantic_config=semantic_config,
        )
        x_test = feature_matrix(
            testing,
            history_size=history_size,
            semantic_config=semantic_config,
        )
        hazard = fit_binary(
            x_train,
            training.change_target.to_numpy(dtype=int),
            trace_balanced_weights(training),
            x_test,
            seed=seed + fold_number,
        )
        changed_training = training[training.change_target == 1]
        changed_x = feature_matrix(
            changed_training,
            history_size=history_size,
            semantic_config=semantic_config,
        )
        destination = fit_destination(
            changed_x,
            changed_training.destination_target.to_numpy(dtype=int),
            trace_balanced_weights(changed_training),
            x_test,
            seed=seed + 100 + fold_number,
        )
        exact = combine_hurdle_probabilities(
            testing.current_regime.to_numpy(dtype=object), hazard, destination
        )
        true_regime = testing.next_regime.map(regime_index).to_numpy(dtype=int)
        true_change = testing.change_target.to_numpy(dtype=int)
        for source_index, source in enumerate(testing.itertuples(index=False)):
            true_destination = (
                destination_index[source.destination_name]
                if source.change_target
                else None
            )
            rows.append(
                {
                    "alignment": alignment,
                    "semantic_config": semantic_config,
                    "history_size": history_size,
                    "example_id": source.example_id,
                    "trajectory_id": source.trajectory_id,
                    "matched_pair_id": source.matched_pair_id,
                    "matched_role": source.matched_role,
                    "validation_group": source.validation_group,
                    "position_index": source.position_index,
                    "current_regime": source.current_regime,
                    "next_regime": source.next_regime,
                    "is_change": bool(true_change[source_index]),
                    "destination_name": source.destination_name,
                    "predicted_change_probability": float(hazard[source_index]),
                    "hazard_log_loss": float(
                        -np.log(
                            max(
                                (
                                    hazard[source_index]
                                    if true_change[source_index]
                                    else 1.0 - hazard[source_index]
                                ),
                                1e-12,
                            )
                        )
                    ),
                    "hazard_brier": float(
                        (hazard[source_index] - true_change[source_index]) ** 2
                    ),
                    "destination_log_loss": (
                        float(
                            -np.log(
                                max(destination[source_index, true_destination], 1e-12)
                            )
                        )
                        if true_destination is not None
                        else np.nan
                    ),
                    "destination_correct": (
                        bool(np.argmax(destination[source_index]) == true_destination)
                        if true_destination is not None
                        else np.nan
                    ),
                    "exact_regime_log_loss": float(
                        -np.log(
                            max(exact[source_index, true_regime[source_index]], 1e-12)
                        )
                    ),
                    "exact_regime_correct": bool(
                        np.argmax(exact[source_index]) == true_regime[source_index]
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["alignment", "semantic_config", "history_size"]
    trace = predictions.groupby([*keys, "example_id"], as_index=False).agg(
        validation_group=("validation_group", "first"),
        n_transitions=("is_change", "size"),
        change_rate=("is_change", "mean"),
        hazard_log_loss=("hazard_log_loss", "mean"),
        hazard_brier=("hazard_brier", "mean"),
        exact_log_loss=("exact_regime_log_loss", "mean"),
        exact_accuracy=("exact_regime_correct", "mean"),
    )
    self_rows = (
        predictions[~predictions.is_change]
        .groupby([*keys, "example_id"], as_index=False)
        .agg(self_log_loss=("exact_regime_log_loss", "mean"))
    )
    change_rows = (
        predictions[predictions.is_change]
        .groupby([*keys, "example_id"], as_index=False)
        .agg(
            change_log_loss=("exact_regime_log_loss", "mean"),
            destination_log_loss=("destination_log_loss", "mean"),
            destination_accuracy=("destination_correct", "mean"),
        )
    )
    trace = trace.merge(self_rows, on=[*keys, "example_id"], validate="one_to_one")
    trace = trace.merge(change_rows, on=[*keys, "example_id"], validate="one_to_one")
    trace["balanced_exact_log_loss"] = (
        trace.self_log_loss + trace.change_log_loss
    ) / 2.0
    summary_rows: list[dict[str, Any]] = []
    for specification, group in predictions.groupby(keys, sort=True):
        alignment, semantic_config, history_size = specification
        trace_group = trace[
            (trace.alignment == alignment)
            & (trace.semantic_config == semantic_config)
            & (trace.history_size == history_size)
        ]
        summary_rows.append(
            {
                "alignment": alignment,
                "semantic_config": semantic_config,
                "history_size": history_size,
                "n_traces": int(trace_group.example_id.nunique()),
                "n_transitions": int(len(group)),
                "n_changes": int(group.is_change.sum()),
                "change_rate": float(group.is_change.mean()),
                "hazard_log_loss": float(trace_group.hazard_log_loss.mean()),
                "hazard_brier": float(trace_group.hazard_brier.mean()),
                "hazard_average_precision": float(
                    average_precision_score(
                        group.is_change.astype(int),
                        group.predicted_change_probability,
                    )
                ),
                "hazard_auroc": float(
                    roc_auc_score(
                        group.is_change.astype(int),
                        group.predicted_change_probability,
                    )
                ),
                "destination_log_loss": float(trace_group.destination_log_loss.mean()),
                "destination_accuracy": float(trace_group.destination_accuracy.mean()),
                "overall_exact_log_loss": float(trace_group.exact_log_loss.mean()),
                "self_exact_log_loss": float(trace_group.self_log_loss.mean()),
                "change_exact_log_loss": float(trace_group.change_log_loss.mean()),
                "balanced_exact_log_loss": float(
                    trace_group.balanced_exact_log_loss.mean()
                ),
                "exact_regime_accuracy": float(trace_group.exact_accuracy.mean()),
            }
        )
    return trace, pd.DataFrame(summary_rows)


def incremental_semantic_tests(trace: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    metrics = (
        "hazard_log_loss",
        "destination_log_loss",
        "exact_log_loss",
        "change_log_loss",
        "balanced_exact_log_loss",
    )
    rows: list[dict[str, Any]] = []
    for (alignment, history_size), comparison in trace.groupby(
        ["alignment", "history_size"]
    ):
        baseline = comparison[comparison.semantic_config == "none"].set_index(
            "example_id"
        )
        for semantic_config in sorted(comparison.semantic_config.unique()):
            if semantic_config == "none":
                continue
            extended = comparison[
                comparison.semantic_config == semantic_config
            ].set_index("example_id")
            for metric in metrics:
                improvement = baseline[metric] - extended[metric]
                inference = paired_inference(
                    improvement,
                    groups=extended.validation_group,
                    seed=seed + history_size,
                )
                rows.append(
                    {
                        "alignment": alignment,
                        "history_size": history_size,
                        "semantic_config": semantic_config,
                        "metric": metric,
                        "baseline_mean": float(baseline[metric].mean()),
                        "semantic_mean": float(extended[metric].mean()),
                        "improvement": float(improvement.mean()),
                        "ci_low": inference["ci_low"],
                        "ci_high": inference["ci_high"],
                        "sign_flip_p": inference["sign_flip_p_two_sided"],
                    }
                )
    output = pd.DataFrame(rows)
    output["bh_q_across_all_tests"] = np.nan
    for indices in output.groupby("alignment").groups.values():
        p_values = output.loc[indices, "sign_flip_p"].to_numpy(dtype=float)
        order = np.argsort(p_values)
        ranked = p_values[order]
        adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        restored = np.empty_like(adjusted)
        restored[order] = np.minimum(adjusted, 1.0)
        output.loc[indices, "bh_q_across_all_tests"] = restored
    return output


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    labels = [column.replace("_", " ") for column in columns]
    lines = [
        "| " + " | ".join(labels) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for value in row:
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_path = args.semantic_root / args.original
    replicate_path = args.semantic_root / args.replicate
    transitions = add_history_features(
        pd.read_csv(args.transitions), maximum_history=max(HISTORY_SIZES)
    )
    semantics = prepare_semantics(original_path, replicate_path)
    predictions: list[pd.DataFrame] = []
    alignment_specs = {
        "prospective": [
            (config, history)
            for config in SEMANTIC_CONFIGS
            for history in HISTORY_SIZES
        ],
        "operator": [(config, 2) for config in ("none", "mean_all")],
    }
    prepared: dict[tuple[str, str], pd.DataFrame] = {}
    for alignment, specifications in alignment_specs.items():
        for semantic_config, history_size in specifications:
            mode = (
                "mean" if semantic_config == "none" else semantic_mode(semantic_config)
            )
            key = (alignment, mode)
            if key not in prepared:
                prepared[key] = prepare_analysis_frame(
                    transitions, semantics[mode], alignment=alignment
                )
            predictions.append(
                run_specification(
                    prepared[key],
                    alignment=alignment,
                    semantic_config=semantic_config,
                    history_size=history_size,
                    seed=args.seed,
                )
            )
    prediction_rows = pd.concat(predictions, ignore_index=True)
    trace_summary, model_summary = summarize_predictions(prediction_rows)
    incremental = incremental_semantic_tests(trace_summary, seed=args.seed)
    prediction_rows.to_csv(args.output_dir / "prediction_rows.csv", index=False)
    trace_summary.to_csv(args.output_dir / "trace_metric_summary.csv", index=False)
    model_summary.to_csv(args.output_dir / "model_summary.csv", index=False)
    incremental.to_csv(args.output_dir / "semantic_incremental_tests.csv", index=False)

    history = model_summary[
        (model_summary.alignment == "prospective")
        & (model_summary.semantic_config == "none")
    ].sort_values("history_size")
    semantic_sensitivity = model_summary[
        (model_summary.alignment == "prospective") & (model_summary.history_size == 2)
    ].sort_values("semantic_config")
    primary_tests = incremental[
        (incremental.alignment == "prospective")
        & (incremental.history_size == 2)
        & (incremental.semantic_config == "mean_all")
    ]
    operator = model_summary[
        (model_summary.alignment == "operator") & (model_summary.history_size == 2)
    ].sort_values("semantic_config")
    prospective_rows = prediction_rows[
        (prediction_rows.alignment == "prospective")
        & (prediction_rows.semantic_config == "none")
        & (prediction_rows.history_size == 1)
    ]
    destination_counts = prospective_rows.loc[
        prospective_rows.is_change, "destination_name"
    ].value_counts()
    change_counts_by_group = prospective_rows.groupby("validation_group").is_change.sum()
    lines = [
        "# Semantic-Informed Regime-Transition Prediction",
        "",
        "## Question",
        "",
        "Do semantic multilabels improve prediction of regime persistence, regime-change hazard, and the exact destination conditional on change, beyond progress, behavioral action features, regime duration, and a finite window of regime history?",
        "",
        "## Definitions",
        "",
        "`history_size = k` is a sliding window containing the current regime and the preceding `k-1` regimes. Every model also includes reasoning progress, current regime duration, the four action probabilities, and action entropy. The prospective semantic model uses the most recently revealed sentence label; the operator sensitivity uses the label of the sentence revealed during the transition and is contemporaneous rather than forecasting.",
        "",
        "The hurdle model first predicts `change = (z_next != z_current)`, then predicts one of `switch_unsettled`, `enter_stable_optimal`, or `enter_stable_suboptimal` conditional on change. These probabilities are recombined into one exact four-regime distribution.",
        "",
        "## Data",
        "",
        f"- Prospective transitions: {int(history.n_transitions.iloc[0])}",
        f"- Prospective changes: {int(history.n_changes.iloc[0])}",
        f"- Change prevalence: {history.change_rate.iloc[0]:.2%}",
        f"- Trajectory-connected validation groups: {transitions.validation_group.nunique()}",
        "- Prospective destination support: "
        f"{int(destination_counts.get('switch_unsettled', 0))} unsettled switches, "
        f"{int(destination_counts.get('enter_stable_optimal', 0))} stable-optimal entries, "
        f"and {int(destination_counts.get('enter_stable_suboptimal', 0))} stable-suboptimal entries.",
        f"- Change events per validation group range from {int(change_counts_by_group.min())} to {int(change_counts_by_group.max())}; inference therefore uses trajectory-level summaries and trajectory-connected groups, but remains data-limited.",
        "",
        "## Regime-history window without semantics",
        "",
        *markdown_table(
            history,
            [
                "history_size",
                "hazard_log_loss",
                "hazard_average_precision",
                "destination_log_loss",
                "overall_exact_log_loss",
                "change_exact_log_loss",
                "balanced_exact_log_loss",
            ],
        ),
        "",
        "## Prospective semantic sensitivity at history size 2",
        "",
        *markdown_table(
            semantic_sensitivity,
            [
                "semantic_config",
                "hazard_log_loss",
                "hazard_average_precision",
                "destination_log_loss",
                "overall_exact_log_loss",
                "change_exact_log_loss",
                "balanced_exact_log_loss",
            ],
        ),
        "",
        "Positive incremental values below mean lower loss after adding mean original/replicate semantic features to the same nonsemantic model.",
        "",
        *markdown_table(
            primary_tests,
            [
                "metric",
                "baseline_mean",
                "semantic_mean",
                "improvement",
                "ci_low",
                "ci_high",
                "sign_flip_p",
                "bh_q_across_all_tests",
            ],
        ),
        "",
        "## Contemporaneous semantic-operator sensitivity",
        "",
        *markdown_table(
            operator,
            [
                "semantic_config",
                "hazard_log_loss",
                "hazard_average_precision",
                "destination_log_loss",
                "overall_exact_log_loss",
                "change_exact_log_loss",
            ],
        ),
        "",
        "## Conclusions",
        "",
        "The best nonsemantic specification uses history size 1. Adding more raw regime lags does not improve change-hazard, conditional-destination, or exact-regime loss; history size 8 is distinctly worse. Thus the current regime plus the continuously valued duration and behavioral state carry more useful information than a longer literal regime window.",
        "",
        "The prespecified prospective mean-semantic model does not improve any loss with a confidence interval excluding zero, and no prospective semantic specification has a robust positive incremental result. The labels therefore do not yet provide evidence of predictive information beyond the nonsemantic state variables. This is an informative null, not evidence that the semantic functions are causally irrelevant.",
        "",
        "The contemporaneous operator alignment also fails to improve prediction and substantially worsens conditional-destination loss in this high-dimensional linear specification. Treat it as an overfitting warning rather than evidence of a harmful semantic effect.",
        "",
        "## Interpretation guardrails",
        "",
        "Overall exact-regime loss intentionally retains the natural persistence distribution. Hazard and conditional-destination results diagnose whether apparent performance is only persistence prediction. The balanced exact loss weights persistence and change equally as a diagnostic, not as the natural deployment distribution.",
        "",
        "Semantic labels are generated measurements, not randomized treatments. Prospective associations may encode latent reasoning content or annotation artifacts; operator-aligned associations are contemporaneous and cannot by themselves establish that a semantic function caused a regime transition.",
        "",
        "All semantic models use fixed L2 regularization rather than nested hyperparameter selection. With only 354 prospective changes—and only 8 stable-suboptimal entries—the conditional destination analysis has limited power, especially for interactions between current regime and semantic features.",
    ]
    (args.output_dir / "run_report.md").write_text("\n".join(lines) + "\n")
    write_json(
        args.output_dir / "run_manifest.json",
        {
            "analysis": "semantic_regime_transition_prediction_v1",
            "transitions": str(args.transitions),
            "transitions_sha256": sha256(args.transitions),
            "original_annotations": str(original_path),
            "original_sha256": sha256(original_path),
            "replicate_annotations": str(replicate_path),
            "replicate_sha256": sha256(replicate_path),
            "history_sizes": list(HISTORY_SIZES),
            "semantic_labels": list(SEMANTIC_LABELS),
            "cue_columns": list(CUE_COLUMNS),
            "semantic_configs": list(SEMANTIC_CONFIGS),
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
