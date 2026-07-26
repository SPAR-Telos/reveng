#!/usr/bin/env python3
"""Build concise Experiment 1 and 2 reports for the local matched-46 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


EVENTS = {
    "Recommendation change": {"action_change"},
    "Commitment onset": {"commitment_onset"},
    "Optimal to suboptimal": {
        "sustained_optimal_to_suboptimal",
        "transient_optimal_to_suboptimal",
    },
    "Suboptimal to optimal": {"suboptimal_to_optimal"},
}
METRICS = {
    "update_norm": ("Euclidean distance between adjacent sentence representations", False),
    "adjacent_cosine": ("Cosine distance between adjacent sentence representations", True),
    "previous_mean_cosine": ("Cosine similarity to the mean of earlier sentence representations", False),
}


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = int(labels.sum())
    negative = len(labels) - positive
    if not positive or not negative:
        return float("nan")
    ranks = rankdata(scores)
    return float(
        (ranks[labels == 1].sum() - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def bootstrap_auc(
    frame: pd.DataFrame,
    *,
    label_column: str,
    score_column: str,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    grouped = {key: group for key, group in frame.groupby("trajectory_id")}
    keys = np.array(sorted(grouped), dtype=object)
    values: list[float] = []
    for _ in range(repeats):
        sample = pd.concat(
            [grouped[key] for key in rng.choice(keys, size=len(keys), replace=True)],
            ignore_index=True,
        )
        value = auc(
            sample[label_column].to_numpy(dtype=int),
            sample[score_column].to_numpy(dtype=float),
        )
        if np.isfinite(value):
            values.append(value)
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def build_activation_monitor(exp1: Path, *, bootstrap_repeats: int) -> pd.DataFrame:
    geometry = pd.read_csv(exp1 / "geometry_rows.csv")
    events = pd.read_csv(exp1 / "event_rows.csv")
    primary = geometry[
        (geometry.layer == 15)
        & (geometry.representation == "sentence_mean")
    ].copy()
    event_lookup = {
        label: {
            (row.example_id, int(row.reasoning_step_idx))
            for row in events[events.event_type.isin(types)].itertuples()
        }
        for label, types in EVENTS.items()
    }
    rows = []
    for event_index, (event_label, positions) in enumerate(event_lookup.items()):
        labels = np.array(
            [
                int((row.example_id, int(row.reasoning_step_idx)) in positions)
                for row in primary.itertuples()
            ]
        )
        for metric_index, (metric, (metric_label, convert_from_cosine)) in enumerate(METRICS.items()):
            valid = primary[primary[metric].notna()].copy()
            valid["event_label"] = [
                int((row.example_id, int(row.reasoning_step_idx)) in positions)
                for row in valid.itertuples()
            ]
            valid["monitor_score"] = (
                1.0 - valid[metric].astype(float)
                if convert_from_cosine
                else valid[metric].astype(float)
            )
            point = auc(
                valid.event_label.to_numpy(dtype=int),
                valid.monitor_score.to_numpy(dtype=float),
            )
            low, high = bootstrap_auc(
                valid,
                label_column="event_label",
                score_column="monitor_score",
                repeats=bootstrap_repeats,
                seed=42 + 10 * event_index + metric_index,
            )
            event_values = valid.loc[valid.event_label == 1, "monitor_score"]
            other_values = valid.loc[valid.event_label == 0, "monitor_score"]
            rows.append(
                {
                    "event": event_label,
                    "activation_metric": metric_label,
                    "layer": 15,
                    "representation": "sentence mean",
                    "n_event_positions": len(event_values),
                    "n_other_positions": len(other_values),
                    "mean_at_event": event_values.mean(),
                    "mean_at_other_positions": other_values.mean(),
                    "monitor_auc": point,
                    "monitor_auc_ci_low": low,
                    "monitor_auc_ci_high": high,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(exp1 / "activation_event_monitor_summary.csv", index=False)

    colors = ["#0B3C5D", "#328CC1", "#8CC8E8"]
    offsets = [-0.18, 0.0, 0.18]
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    y_base = np.arange(len(EVENTS))
    for color, offset, metric_label in zip(colors, offsets, [value[0] for value in METRICS.values()]):
        subset = result[result.activation_metric == metric_label].set_index("event").loc[list(EVENTS)]
        values = subset.monitor_auc.to_numpy()
        lower = values - subset.monitor_auc_ci_low.to_numpy()
        upper = subset.monitor_auc_ci_high.to_numpy() - values
        ax.errorbar(
            values,
            y_base + offset,
            xerr=np.vstack([lower, upper]),
            fmt="o",
            color=color,
            capsize=2,
            label=metric_label,
        )
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    event_labels = []
    for event_label in EVENTS:
        event_n = int(
            result[
                (result.event == event_label)
                & (result.activation_metric == METRICS["update_norm"][0])
            ].n_event_positions.iloc[0]
        )
        event_labels.append(f"{event_label} (n={event_n})")
    ax.set_yticks(y_base, labels=event_labels)
    ax.set_xlim(0.35, 0.75)
    ax.set_xlabel("AUROC for distinguishing event positions from other sentence positions (0.5 = chance)")
    ax.set_ylabel("Action-selection event")
    ax.set_title(
        f"Can Simple Activation Metrics Distinguish Action-Selection Events?\n"
        f"46 states, {len(primary):,} sentence positions"
    )
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(axis="x", color="#E5EEF5", linewidth=0.8)
    fig.tight_layout()
    figure_dir = exp1 / "figs"
    figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir / "activation_event_monitor_auc.png", dpi=220)
    plt.close(fig)
    return result


def write_reports(exp1: Path, exp2: Path, monitor: pd.DataFrame) -> None:
    manifest = json.loads((exp1 / "analysis_manifest.json").read_text())
    positions = pd.read_csv(exp1 / "position_rows.csv")
    commitments = positions[positions.commitment_onset == True]  # noqa: E712
    primary = monitor[monitor.activation_metric == METRICS["update_norm"][0]]
    exp1_lines = [
        "# Experiment 1: Activation Monitor of Action Events",
        "",
        f"The matched analysis contains 46 DoorKey states from {positions.trajectory_id.nunique()} trajectories, "
        f"{len(positions):,} sentence prefixes, and {manifest['geometry_rows']:,} activation rows.",
        "Activations are GPT-OSS-20B layer 15 sentence means for the prespecified primary analysis. "
        "AUROC measures whether each activation metric distinguishes event positions from other sentence positions. "
        "Intervals bootstrap complete trajectories and are descriptive, not causal.",
        "",
        f"A retrospective commitment boundary was identifiable for all {len(commitments)} states. It is the first "
        "prefix whose recommended action equals the full-trace recommendation and remains unchanged thereafter.",
        "",
        "| Event | Positions | Change-magnitude AUROC | 95% interval |",
        "|---|---:|---:|---:|",
    ]
    for row in primary.itertuples():
        exp1_lines.append(
            f"| {row.event} | {row.n_event_positions} | {row.monitor_auc:.3f} | "
            f"[{row.monitor_auc_ci_low:.3f}, {row.monitor_auc_ci_high:.3f}] |"
        )
    exp1_lines.extend(
        [
            "",
            "The complete metric table is `activation_event_monitor_summary.csv`. "
            "The figure is `figs/activation_event_monitor_auc.png`.",
            "",
            "The empirical optimal-trace anchor from the original proposal is not included yet; these results use "
            "only adjacent change magnitude, distance from the preceding sentence, and similarity to the mean of "
            "preceding reasoning sentences.",
        ]
    )
    (exp1 / "run_report.md").write_text("\n".join(exp1_lines) + "\n")

    model_summary = pd.read_csv(exp2 / "transition_model_summary.csv")
    exp2_lines = [
        "# Experiment 2: Behavioral Beliefs and Action Optimality",
        "",
        f"The analysis contains {manifest['belief_rows']:,} categorical belief readouts and "
        f"{manifest['belief_shifts']:,} adjacent-prefix belief changes. Action and categorical belief uncertainty "
        "use candidate-token logprobs at temperature 0.7; repeated sampling is not used.",
        "",
        "Models hold out complete trajectories and keep matched pairs in the same validation group. Results are "
        "predictive associations rather than causal effects.",
        "",
        "| Outcome | Best learned model | AUROC | Events |",
        "|---|---|---:|---:|",
    ]
    for family in ("optimal_to_suboptimal", "suboptimal_to_optimal"):
        candidates = model_summary[
            (model_summary.transition_family == family)
            & (model_summary.model_type != "prevalence_baseline")
        ]
        best = candidates.loc[candidates.roc_auc.idxmax()]
        exp2_lines.append(
            f"| {family.replace('_', ' ')} | {best.model_type.replace('_', ' ')} | "
            f"{best.roc_auc:.3f} | {int(best.n_positive)} |"
        )
    exp2_lines.extend(
        [
            "",
            "`door_open` is excluded from the fitted models because its ground truth is always `no` in this matched "
            "cohort. Consequently, the proposed key-door interaction cannot be estimated here.",
            "",
            "Coordinate beliefs remain deferred because they require generated coordinate answers rather than the "
            "categorical candidate-logprob protocol used in this run.",
            "",
            "See `TRANSITION_MODEL_SUMMARY.md` for held-out metrics and bootstrap intervals.",
        ]
    )
    (exp2 / "run_report.md").write_text("\n".join(exp2_lines) + "\n")

    captions = (
        "# Figure Captions\n\n"
        "## activation_event_monitor_auc.png\n\n"
        "Ability of three GPT-OSS-20B layer 15 sentence-mean activation metrics to distinguish "
        "sentence positions containing recommendation changes, retrospective commitment onset, optimality loss, "
        "or optimality recovery from other sentence positions. Points are AUROC values and bars are 95 percent "
        "intervals from bootstrapping complete trajectories. AUROC 0.5 indicates chance discrimination. "
        "Cosine similarity to the mean of earlier sentence representations compares the current sentence-mean "
        "activation with the average sentence-mean activation over all earlier reasoning sentences for the same "
        "environment state. This is a contemporaneous univariate association, not a forecast of a later event, "
        "and it does not adjust for reasoning progress or sentence content.\n"
    )
    (exp1 / "FIGURE_CAPTIONS.md").write_text(captions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment1-dir",
        type=Path,
        default=Path("outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1"),
    )
    parser.add_argument(
        "--experiment2-dir",
        type=Path,
        default=Path("outputs/experiment2_behavioral_beliefs/gpt_oss_local_sentence_matched46_v1"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()
    monitor = build_activation_monitor(
        args.experiment1_dir,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    write_reports(args.experiment1_dir, args.experiment2_dir, monitor)


if __name__ == "__main__":
    main()
