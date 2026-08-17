#!/usr/bin/env python3
"""Build a compact evidence summary for the belief-action reasoning experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


BLUE = "#1769aa"
LIGHT_BLUE = "#9ecae1"
PALE_BLUE = "#deebf7"
DARK = "#17324d"
GRAY = "#66788a"
GRID = "#d9e3ec"
ORANGE = "#d97706"
TEAL = "#0f766e"

SEMANTIC_ORDER = [
    "state_readout",
    "route_deliberation",
    "summary_or_restatement",
    "other",
]
SEMANTIC_LABELS = {
    "state_readout": "State readout",
    "route_deliberation": "Route deliberation",
    "summary_or_restatement": "Summary or restatement",
    "other": "Other",
}


def configure_style() -> None:
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        font_family = "Arial"
    except ValueError:
        font_family = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": "#9aabb8",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_belief_forecast(model_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(model_path)
    selected = frame[
        frame["target"].isin(["optimality_loss_h1", "optimality_loss_h3"])
        & frame["model"].isin(["baseline", "observable_beliefs"])
    ].copy()
    horizon_order = ["optimality_loss_h1", "optimality_loss_h3"]
    labels = ["Next sentence", "Next 3 sentences"]
    x = np.arange(2)
    width = 0.34

    fig, (ax, delta_ax) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.6),
        gridspec_kw={"width_ratios": [1.45, 1]},
    )
    for offset, model, label, color in [
        (-width / 2, "baseline", "Behavioral baseline", LIGHT_BLUE),
        (width / 2, "observable_beliefs", "Baseline and belief readouts", BLUE),
    ]:
        values = [
            float(
                selected[(selected["target"] == target) & (selected["model"] == model)][
                    "auprc"
                ].iloc[0]
            )
            for target in horizon_order
        ]
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

    sample_labels = []
    for target, label in zip(horizon_order, labels, strict=True):
        row = selected[
            (selected["target"] == target) & (selected["model"] == "baseline")
        ].iloc[0]
        sample_labels.append(
            f"{label}\n{int(row.n_rows):,} positions, "
            f"{int(row.n_positive_windows):,} positive"
        )
    ax.set_xticks(x, sample_labels)
    ax.set_ylabel("Held-out area under the precision-recall curve")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylim(0, 0.39)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)

    belief_rows = selected[selected["model"] == "observable_beliefs"].set_index(
        "target"
    )
    deltas = np.array(
        [
            float(belief_rows.loc[target, "delta_auprc_vs_baseline"])
            for target in horizon_order
        ]
    )
    low = np.array(
        [
            float(belief_rows.loc[target, "delta_auprc_ci_low"])
            for target in horizon_order
        ]
    )
    high = np.array(
        [
            float(belief_rows.loc[target, "delta_auprc_ci_high"])
            for target in horizon_order
        ]
    )
    delta_ax.errorbar(
        deltas,
        x,
        xerr=np.vstack([deltas - low, high - deltas]),
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        capsize=4,
        markersize=7,
    )
    delta_ax.axvline(0, color=GRAY, linestyle="--", linewidth=1)
    delta_ax.set_yticks(x, labels)
    delta_ax.invert_yaxis()
    delta_ax.set_xlabel("AUPRC improvement from belief readouts")
    delta_ax.set_ylabel("Forecast horizon")
    delta_ax.grid(axis="x", color=GRID, linewidth=0.8)
    for y, value in zip(x, deltas, strict=True):
        delta_ax.text(value + 0.004, y - 0.13, f"{value:+.3f}", color=DARK, fontsize=9)

    fig.suptitle("Behavioral Belief Readouts Forecast Upcoming Optimality Loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_change_point_proximity(summary_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(summary_path)
    order = ["action_change", "optimal_to_suboptimal", "suboptimal_to_optimal"]
    frame = frame.set_index("event_type").loc[order].reset_index()
    labels = [
        "Recommended action changes",
        "Recommendation becomes suboptimal",
        "Recommendation becomes optimal",
    ]
    y = np.arange(len(frame))
    offset = 0.13
    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    observed = frame["observed_fraction"].to_numpy()
    observed_low = frame["observed_ci_low"].to_numpy()
    observed_high = frame["observed_ci_high"].to_numpy()
    random = frame["progress_matched_random_fraction"].to_numpy()
    random_low = frame["random_ci_low"].to_numpy()
    random_high = frame["random_ci_high"].to_numpy()
    ax.errorbar(
        observed,
        y - offset,
        xerr=np.vstack([observed - observed_low, observed_high - observed]),
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        capsize=4,
        markersize=7,
        label="Detected action-distribution change points",
    )
    ax.errorbar(
        random,
        y + offset,
        xerr=np.vstack([random - random_low, random_high - random]),
        fmt="o",
        color=GRAY,
        ecolor=GRAY,
        capsize=4,
        markersize=6,
        label="Progress-matched sentences",
    )
    for y_pos, value in zip(y - offset, observed, strict=True):
        ax.text(
            value + 0.018, y_pos, f"{value:.1%}", va="center", color=DARK, fontsize=9
        )
    for y_pos, value in zip(y + offset, random, strict=True):
        ax.text(
            value + 0.018, y_pos, f"{value:.1%}", va="center", color=GRAY, fontsize=9
        )

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Fraction within 3 sentences before or after the event")
    ax.set_ylabel("Reasoning event")
    ax.set_title(
        "Action-Distribution Changes Cluster Near Recommendation Changes\n"
        "160 detected change points across 44 of 46 environment states"
    )
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=2,
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_monitor_comparison(summary_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(summary_path)
    targets = [
        "action_change",
        "optimal_to_suboptimal",
        "suboptimal_to_optimal",
        "commitment_onset",
    ]
    labels = [
        "Recommendation change",
        "Optimal to suboptimal",
        "Suboptimal to optimal",
        "Retrospective commitment",
    ]
    best_activation = (
        frame[frame["model_type"].str.startswith("activation_pca")]
        .sort_values("roc_auc", ascending=False)
        .groupby("target", as_index=False)
        .first()
        .set_index("target")
    )
    by_model = frame.set_index(["target", "model_type"])
    y = np.arange(len(targets))
    offsets = [-0.22, 0, 0.22]
    models = [
        ("Best activation monitor", BLUE),
        ("Reasoning progress", LIGHT_BLUE),
        ("Behavioral readouts", GRAY),
    ]

    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    for offset, (label, color) in zip(offsets, models, strict=True):
        values = []
        low = []
        high = []
        for target in targets:
            if label == "Best activation monitor":
                row = best_activation.loc[target]
            elif label == "Reasoning progress":
                row = by_model.loc[(target, "reasoning_progress")]
            else:
                row = by_model.loc[(target, "scalar_readouts")]
            values.append(float(row.roc_auc))
            low.append(float(row.roc_auc_ci_low))
            high.append(float(row.roc_auc_ci_high))
        values_arr = np.asarray(values)
        ax.errorbar(
            values_arr,
            y + offset,
            xerr=np.vstack([values_arr - low, np.asarray(high) - values_arr]),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=6,
            label=label,
        )

    event_labels = []
    for target, label in zip(targets, labels, strict=True):
        row = best_activation.loc[target]
        event_labels.append(
            f"{label}\n{int(row.n_positive):,} events in {int(row.n_rows):,} positions"
        )
    ax.set_yticks(y, event_labels)
    ax.invert_yaxis()
    ax.axvline(0.5, color="#888888", linestyle="--", linewidth=1, label="Chance")
    ax.set_xlim(0.38, 0.88)
    ax.set_xlabel("Trajectory-held-out AUROC")
    ax.set_ylabel("Action event")
    ax.set_title("Activation Monitors Do Not Outperform Behavioral Readouts")
    ax.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_semantic_cross_analysis(
    matched_path: Path,
    model_path: Path,
    event_path: Path,
    outcome_test_path: Path,
    attention_path: Path,
    attention_test_path: Path,
    output_path: Path,
) -> None:
    matched = pd.read_csv(matched_path)
    matched = matched[
        matched["analysis_set"].eq("primary_matches")
        & matched["label_family"].eq("broad")
    ].copy()
    matched["semantic_feature"] = pd.Categorical(
        matched["semantic_feature"], categories=SEMANTIC_ORDER, ordered=True
    )
    matched = matched.sort_values("semantic_feature")

    models = pd.read_csv(model_path)
    model_order = [
        "baseline_plus_semantics",
        "baseline_plus_activation",
        "baseline_plus_semantics_and_activation",
    ]
    models = models.set_index("model").loc[model_order].reset_index()

    events = pd.read_csv(event_path)
    entropy = events[events["outcome"].eq("mean_belief_entropy_bits")].copy()
    entropy["broad_label"] = pd.Categorical(
        entropy["broad_label"], categories=SEMANTIC_ORDER, ordered=True
    )
    entropy = entropy.sort_values("broad_label")

    outcome_tests = pd.read_csv(outcome_test_path)
    entropy_test = outcome_tests[
        outcome_tests["outcome"].eq("mean_belief_entropy_bits")
    ].iloc[0]

    attention = pd.read_csv(attention_path)
    attention = attention[attention["pairs"].gt(0)].copy()
    attention["broad_label"] = pd.Categorical(
        attention["broad_label"], categories=SEMANTIC_ORDER, ordered=True
    )
    attention = attention.sort_values("broad_label")
    attention_test = json.loads(attention_test_path.read_text())

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.4))
    axes = axes.ravel()

    y = np.arange(len(matched))
    differences = matched["paired_rate_difference"].to_numpy()
    low = matched["trajectory_bootstrap_ci_low"].to_numpy()
    high = matched["trajectory_bootstrap_ci_high"].to_numpy()
    axes[0].errorbar(
        differences,
        y,
        xerr=np.vstack([differences - low, high - differences]),
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        capsize=4,
        markersize=7,
    )
    axes[0].axvline(0, color=GRAY, linestyle="--", linewidth=1)
    axes[0].set_yticks(
        y,
        [
            SEMANTIC_LABELS[str(value)]
            for value in matched["semantic_feature"].astype("object")
        ],
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Change-point rate minus matched-sentence rate")
    axes[0].set_title("A. No stable semantic signature (omnibus p = 0.219)", loc="left")
    axes[0].xaxis.set_major_formatter(lambda value, _: f"{value:+.0%}")
    axes[0].grid(axis="x", color=GRID, linewidth=0.8)

    model_labels = [
        "Semantic function",
        "Activation geometry",
        "Both",
    ]
    y = np.arange(len(models))
    delta = models["delta_auroc_vs_baseline"].to_numpy()
    low = models["delta_auroc_ci_low"].to_numpy()
    high = models["delta_auroc_ci_high"].to_numpy()
    axes[1].errorbar(
        delta,
        y,
        xerr=np.vstack([delta - low, high - delta]),
        fmt="o",
        color=TEAL,
        ecolor=TEAL,
        capsize=4,
        markersize=7,
    )
    axes[1].axvline(0, color=GRAY, linestyle="--", linewidth=1)
    axes[1].set_yticks(y, model_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Held-out AUROC change vs progress + length")
    axes[1].set_title("B. Limited incremental discrimination", loc="left")
    axes[1].grid(axis="x", color=GRID, linewidth=0.8)
    for y_position, value in zip(y, delta, strict=True):
        axes[1].text(
            high[y_position] + 0.004,
            y_position,
            f"{value:+.3f}",
            va="center",
            fontsize=9,
            color=DARK,
        )

    x = np.arange(len(entropy))
    means = entropy["mean_or_event_rate"].to_numpy()
    low = entropy["trajectory_bootstrap_ci_low"].to_numpy()
    high = entropy["trajectory_bootstrap_ci_high"].to_numpy()
    axes[2].bar(
        x,
        means,
        color=[LIGHT_BLUE, BLUE, TEAL, GRAY],
        width=0.66,
    )
    axes[2].errorbar(
        x,
        means,
        yerr=np.vstack([means - low, high - means]),
        fmt="none",
        ecolor=DARK,
        capsize=4,
        linewidth=1.2,
    )
    axes[2].set_xticks(
        x,
        [
            {
                "state_readout": "State\nreadout",
                "route_deliberation": "Route\ndeliberation",
                "summary_or_restatement": "Summary or\nrestatement",
                "other": "Other",
            }[str(value)]
            + f"\n(n={int(sentences)})"
            for value, sentences in zip(
                entropy["broad_label"].astype("object"),
                entropy["sentences"],
                strict=True,
            )
        ],
    )
    axes[2].set_ylabel("Mean categorical-belief entropy (bits)")
    axes[2].set_title(
        "C. Route deliberation: highest belief uncertainty",
        loc="left",
    )
    axes[2].text(
        0.02,
        0.94,
        f"Within-trajectory permutation q = "
        f"{entropy_test['bh_q_across_outcomes']:.3f}",
        transform=axes[2].transAxes,
        va="top",
        fontsize=9,
        color=GRAY,
    )
    axes[2].grid(axis="y", color=GRID, linewidth=0.8)

    y = np.arange(len(attention))
    attention_pp = (
        attention["mean_attention_difference_in_differences"].to_numpy() * 100
    )
    attention_se_pp = attention["trajectory_mean_standard_error"].to_numpy() * 100
    axes[3].errorbar(
        attention_pp,
        y,
        xerr=1.96 * attention_se_pp,
        fmt="o",
        color=ORANGE,
        ecolor=ORANGE,
        capsize=4,
        markersize=7,
    )
    axes[3].axvline(0, color=GRAY, linestyle="--", linewidth=1)
    axes[3].set_yticks(
        y,
        [
            f"{SEMANTIC_LABELS[str(label)]}\n" f"({int(pairs)} pairs)"
            for label, pairs in zip(
                attention["broad_label"].astype("object"),
                attention["pairs"],
                strict=True,
            )
        ],
    )
    axes[3].invert_yaxis()
    axes[3].set_xlabel(
        "Newest-sentence attention difference\n" "(percentage points per token)"
    )
    axes[3].set_title(
        "D. Summary attention is largest "
        f"(heterogeneity p = {attention_test['within_trajectory_permutation_p']:.3f})",
        loc="left",
    )
    axes[3].grid(axis="x", color=GRID, linewidth=0.8)

    fig.suptitle(
        "What Sentence-Level Semantic Labels Add to the Action-Selection Story",
        y=0.995,
        fontsize=15,
    )
    fig.text(
        0.5,
        0.008,
        "Calibrated four-way AI-assisted labels; primary close matches only "
        "(152 BEAST change-point/control pairs). Panels C–D are exploratory "
        "associations, not causal effects.",
        ha="center",
        color=GRAY,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.975), h_pad=2.6, w_pad=2.1)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_semantic_tables(
    sentence_path: Path,
    matched_path: Path,
    model_path: Path,
    event_path: Path,
    outcome_test_path: Path,
    attention_path: Path,
    attention_test_path: Path,
    output_dir: Path,
) -> dict[str, float | int]:
    sentences = pd.read_csv(sentence_path)
    matched = pd.read_csv(matched_path)
    models = pd.read_csv(model_path)
    events = pd.read_csv(event_path)
    outcome_tests = pd.read_csv(outcome_test_path)
    attention = pd.read_csv(attention_path)
    attention_test = json.loads(attention_test_path.read_text())

    primary_change_points = sentences[
        sentences["item_role"].eq("detected_change_point")
        & sentences["match_quality"].eq("primary")
    ].copy()
    loss_rows = primary_change_points[
        primary_change_points["optimality_loss_here"].eq(True)
    ].copy()
    candidates = loss_rows[
        loss_rows["mean_belief_error_rate"].fillna(np.inf).eq(0)
    ].copy()
    candidate_columns = [
        "sentence_id",
        "annotation_id",
        "trajectory_id",
        "step_index",
        "position_index",
        "target_sentence",
        "primary_label",
        "broad_label",
        "confidence",
        "previous_action_label",
        "action_label",
        "mean_belief_error_rate",
        "mean_belief_entropy_bits",
        "adjacent_js_divergence_bits",
        "posterior_change_probability",
    ]
    candidates[candidate_columns].sort_values(
        ["broad_label", "trajectory_id", "step_index", "position_index"]
    ).to_csv(output_dir / "semantic_intervention_candidates.csv", index=False)

    matched_broad = matched[
        matched["analysis_set"].eq("primary_matches")
        & matched["label_family"].eq("broad")
    ].set_index("semantic_feature")
    semantic_model = models.set_index("model").loc["baseline_plus_semantics"]
    entropy_test = outcome_tests.set_index("outcome").loc["mean_belief_entropy_bits"]
    exact_action_tests = outcome_tests[
        outcome_tests["outcome"].isin(
            [
                "recommendation_changes_here",
                "optimality_loss_here",
                "recovery_here",
                "commitment_onset_here",
            ]
        )
    ]
    commitment_test = outcome_tests.set_index("outcome").loc["commitment_onset_here"]
    activation_tests = outcome_tests[
        outcome_tests["outcome"].isin(
            ["update_norm", "adjacent_cosine", "previous_mean_cosine"]
        )
    ]
    error_test = outcome_tests.set_index("outcome").loc["mean_belief_error_rate"]
    entropy_profiles = events[
        events["outcome"].eq("mean_belief_entropy_bits")
    ].set_index("broad_label")
    attention_profiles = attention.set_index("broad_label")

    rows = [
        {
            "analysis_family": "Previous belief-action-gap evidence",
            "semantic_question": (
                "Do the current labels directly explain the earlier wall-hit and "
                "reasoning-patching results?"
            ),
            "result": (
                "No direct row join; the earlier failure slice was not labeled "
                "with this taxonomy"
            ),
            "estimate": np.nan,
            "interval_low": np.nan,
            "interval_high": np.nan,
            "p_or_q": np.nan,
            "metric": "scope audit",
            "semantic_utility": "Interpretive bridge only",
            "central_claim_implication": (
                "The labels motivate a heterogeneous intervention design but do "
                "not re-estimate the published behavioral or causal effects."
            ),
        },
        {
            "analysis_family": "Offline action-distribution change points",
            "semantic_question": "Does a distinctive sentence function identify a change point?",
            "result": "No reliable four-way composition difference",
            "estimate": 0.07236842105263161,
            "interval_low": np.nan,
            "interval_high": np.nan,
            "p_or_q": 0.21897810218978103,
            "metric": "paired total-variation distance; permutation p",
            "semantic_utility": "Constrains interpretation",
            "central_claim_implication": (
                "Action-routing boundaries cut across verbal functions; avoid an "
                "'aha/correction sentence' mechanism claim."
            ),
        },
        {
            "analysis_family": "Offline action-distribution change points",
            "semantic_question": "Is route deliberation enriched at change points?",
            "result": "Numerically enriched, interval includes zero",
            "estimate": matched_broad.loc[
                "route_deliberation", "paired_rate_difference"
            ],
            "interval_low": matched_broad.loc[
                "route_deliberation", "trajectory_bootstrap_ci_low"
            ],
            "interval_high": matched_broad.loc[
                "route_deliberation", "trajectory_bootstrap_ci_high"
            ],
            "p_or_q": matched_broad.loc[
                "route_deliberation", "mcnemar_bh_q_within_set"
            ],
            "metric": "paired rate difference; McNemar q",
            "semantic_utility": "Weak descriptive context",
            "central_claim_implication": (
                "Routing language is common, but not specific enough to define "
                "action-distribution transitions."
            ),
        },
        {
            "analysis_family": "Change-point discrimination",
            "semantic_question": (
                "Do labels add held-out discrimination beyond progress and length?"
            ),
            "result": "Small, uncertain AUROC increase",
            "estimate": semantic_model["delta_auroc_vs_baseline"],
            "interval_low": semantic_model["delta_auroc_ci_low"],
            "interval_high": semantic_model["delta_auroc_ci_high"],
            "p_or_q": np.nan,
            "metric": "trajectory-held-out AUROC change",
            "semantic_utility": "Not a dependable monitor",
            "central_claim_implication": (
                "Sentence function is not a practical substitute for action and "
                "belief readouts."
            ),
        },
        {
            "analysis_family": "Exact action events",
            "semantic_question": (
                "Do semantic groups differ in recommendation change, loss, "
                "recovery, or commitment rates?"
            ),
            "result": "No association survives correction",
            "estimate": np.nan,
            "interval_low": np.nan,
            "interval_high": np.nan,
            "p_or_q": exact_action_tests["bh_q_across_outcomes"].min(),
            "metric": "smallest BH q across four event outcomes",
            "semantic_utility": "Constrains event narratives",
            "central_claim_implication": (
                "No named discourse operation reliably explains whether an action "
                "change helps or harms performance."
            ),
        },
        {
            "analysis_family": "Commitment and post-commitment reasoning",
            "semantic_question": (
                "Does broad sentence function identify retrospective commitment "
                "or explain the post-commitment tail?"
            ),
            "result": (
                "No commitment-onset association; labels do not cover all "
                "post-commitment tail sentences"
            ),
            "estimate": np.nan,
            "interval_low": np.nan,
            "interval_high": np.nan,
            "p_or_q": commitment_test["bh_q_across_outcomes"],
            "metric": "commitment-onset within-trajectory permutation q",
            "semantic_utility": "Constrains commitment interpretation",
            "central_claim_implication": (
                "Do not equate retrospective commitment with one sentence "
                "function or infer that later reasoning is semantically useless."
            ),
        },
        {
            "analysis_family": "Activation geometry",
            "semantic_question": (
                "Do semantic groups explain sentence-level activation transitions?"
            ),
            "result": "No activation association survives correction",
            "estimate": np.nan,
            "interval_low": np.nan,
            "interval_high": np.nan,
            "p_or_q": activation_tests["bh_q_across_outcomes"].min(),
            "metric": "smallest BH q across activation outcomes",
            "semantic_utility": "No mechanistic localization",
            "central_claim_implication": (
                "Generic representation movement cannot be assigned to one broad "
                "reasoning function."
            ),
        },
        {
            "analysis_family": "Behavioral beliefs",
            "semantic_question": "Do semantic groups differ in belief error?",
            "result": "No reliable difference in mean belief error",
            "estimate": np.nan,
            "interval_low": np.nan,
            "interval_high": np.nan,
            "p_or_q": error_test["bh_q_across_outcomes"],
            "metric": "within-trajectory permutation q",
            "semantic_utility": "No error explanation",
            "central_claim_implication": (
                "Semantic function does not explain action changes through measured "
                "belief correctness."
            ),
        },
        {
            "analysis_family": "Behavioral beliefs",
            "semantic_question": "Do semantic groups differ in belief uncertainty?",
            "result": (
                "Route deliberation has the highest mean categorical-belief entropy"
            ),
            "estimate": entropy_profiles.loc[
                "route_deliberation", "mean_or_event_rate"
            ],
            "interval_low": entropy_profiles.loc[
                "route_deliberation", "trajectory_bootstrap_ci_low"
            ],
            "interval_high": entropy_profiles.loc[
                "route_deliberation", "trajectory_bootstrap_ci_high"
            ],
            "p_or_q": entropy_test["bh_q_across_outcomes"],
            "metric": "bits; within-trajectory permutation q",
            "semantic_utility": "Useful uncertainty stratifier",
            "central_claim_implication": (
                "Route deliberation marks an uncertainty regime, not a reliably "
                "incorrect-belief regime."
            ),
        },
        {
            "analysis_family": "Immediate-action attention",
            "semantic_question": (
                "Is the newest-sentence attention increase concentrated by function?"
            ),
            "result": (
                "Largest for summaries/restatements; heterogeneity is borderline"
            ),
            "estimate": attention_profiles.loc[
                "summary_or_restatement",
                "mean_attention_difference_in_differences",
            ]
            * 100,
            "interval_low": np.nan,
            "interval_high": np.nan,
            "p_or_q": attention_test["within_trajectory_permutation_p"],
            "metric": "percentage points per token; permutation p",
            "semantic_utility": "Candidate prioritization only",
            "central_claim_implication": (
                "Action readout may emphasize compressed conclusions even though "
                "those sentences are not selectively enriched at change points."
            ),
        },
        {
            "analysis_family": "Central-claim candidate slice",
            "semantic_question": (
                "Can an action become suboptimal while measured categorical beliefs "
                "remain correct?"
            ),
            "result": (
                f"{len(candidates)} of {len(loss_rows)} primary change-point "
                "optimality losses have zero error across nine categorical "
                "belief readouts"
            ),
            "estimate": len(candidates) / len(loss_rows) if len(loss_rows) else np.nan,
            "interval_low": np.nan,
            "interval_high": np.nan,
            "p_or_q": np.nan,
            "metric": "descriptive fraction",
            "semantic_utility": "High-value intervention sampling",
            "central_claim_implication": (
                "These are direct candidates for testing information-to-action "
                "conversion failure."
            ),
        },
        {
            "analysis_family": "Prospective action-event monitor",
            "semantic_question": (
                "Are the monitor's text-semantic features the current four-way "
                "sentence labels?"
            ),
            "result": (
                "No; the monitor uses embedding similarity and novelty features"
            ),
            "estimate": np.nan,
            "interval_low": np.nan,
            "interval_high": np.nan,
            "p_or_q": np.nan,
            "metric": "feature-definition audit",
            "semantic_utility": "Prevents construct conflation",
            "central_claim_implication": (
                "The small prospective text-feature result cannot be attributed "
                "to verification, planning, correction, or other taxonomy labels."
            ),
        },
    ]
    decision_table = pd.DataFrame(rows)
    source_by_family = {
        "Previous belief-action-gap evidence": "belief_action_gap.tex",
        "Offline action-distribution change points": (
            "final_analysis/matched_semantic_comparisons.csv"
        ),
        "Change-point discrimination": (
            "final_analysis/incremental_model_comparison.csv"
        ),
        "Exact action events": ("final_analysis/semantic_outcome_omnibus_tests.csv"),
        "Commitment and post-commitment reasoning": (
            "final_analysis/semantic_outcome_omnibus_tests.csv; "
            "transition_activation_commitment_v1/post_commitment_reasoning_report.md"
        ),
        "Activation geometry": ("final_analysis/semantic_outcome_omnibus_tests.csv"),
        "Behavioral beliefs": (
            "final_analysis/semantic_event_composition.csv; "
            "semantic_outcome_omnibus_tests.csv"
        ),
        "Immediate-action attention": (
            "final_analysis/attention_by_semantic_function.csv"
        ),
        "Central-claim candidate slice": (
            "final_analysis/sentence_label_cpd_analysis_table.csv"
        ),
        "Prospective action-event monitor": (
            "practical_action_event_monitor_v1/prediction_report.md"
        ),
    }
    direct_join_families = {
        "Offline action-distribution change points",
        "Change-point discrimination",
        "Exact action events",
        "Commitment and post-commitment reasoning",
        "Activation geometry",
        "Behavioral beliefs",
        "Immediate-action attention",
        "Central-claim candidate slice",
    }
    decision_table["evidence_relation"] = decision_table["analysis_family"].map(
        lambda value: (
            "direct semantic-label join"
            if value in direct_join_families
            else "scope or construct comparison"
        )
    )
    decision_table["source_artifact"] = decision_table["analysis_family"].map(
        source_by_family
    )
    decision_table.to_csv(
        output_dir / "semantic_cross_analysis_decision_table.csv", index=False
    )

    metrics = {
        "primary_pairs": int(
            sentences[sentences["match_quality"].eq("primary")]["pair_id"].nunique()
        ),
        "primary_change_points": len(primary_change_points),
        "primary_optimality_losses": len(loss_rows),
        "zero_error_loss_candidates": len(candidates),
        "zero_error_loss_fraction": (
            len(candidates) / len(loss_rows) if len(loss_rows) else float("nan")
        ),
        "entropy_route_mean": float(
            entropy_profiles.loc["route_deliberation", "mean_or_event_rate"]
        ),
        "entropy_q": float(entropy_test["bh_q_across_outcomes"]),
        "attention_summary_pp": float(
            attention_profiles.loc[
                "summary_or_restatement",
                "mean_attention_difference_in_differences",
            ]
            * 100
        ),
        "attention_p": float(attention_test["within_trajectory_permutation_p"]),
        "exact_action_min_q": float(exact_action_tests["bh_q_across_outcomes"].min()),
        "activation_min_q": float(activation_tests["bh_q_across_outcomes"].min()),
    }
    manifest = {
        "generated_by": "scripts/build_belief_action_reasoning_summary.py",
        "semantic_taxonomy": "four-way calibrated AI-assisted sentence function",
        "scope": (
            "retrospective BEAST change-point sentences and same-state matched "
            "comparison sentences"
        ),
        "sources": [
            str(sentence_path),
            str(matched_path),
            str(model_path),
            str(event_path),
            str(outcome_test_path),
            str(attention_path),
            str(attention_test_path),
        ],
        "generated_artifacts": [
            "figs/semantic_labels_across_analyses.png",
            "figs/semantic_labels_across_analyses.pdf",
            "semantic_cross_analysis_decision_table.csv",
            "semantic_intervention_candidates.csv",
            "SEMANTIC_CROSS_ANALYSIS_SYNTHESIS.md",
            "SEMANTIC_INTERVENTION_CANDIDATES.md",
            "SEMANTIC_PAPER_INSERT.tex",
        ],
        "headline_metrics": metrics,
    }
    (output_dir / "semantic_synthesis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return metrics


def write_semantic_synthesis(
    output_dir: Path, semantic_metrics: dict[str, float | int]
) -> None:
    text = f"""# Semantic Labels Across the Experiment Chain

## Bottom line

The semantic labels sharpen the project mainly by ruling out an overly simple
mechanism. Retrospective action-distribution change points are behaviorally meaningful,
but they are not a stable class of corrections, verifications, summaries, or other
named discourse operations. Broad sentence function also does not reliably explain
which change points are recommendation changes, optimality losses, recoveries, or
commitment onsets, and it does not localize the sentence-level activation transition.

The useful positive result is narrower: route-deliberation sentences mark a
high-uncertainty belief regime, and semantic labels identify concrete optimality-loss
sentences where all nine categorical belief readouts at that boundary are correct. The
labels are therefore better for interpretation and intervention sampling than for
monitoring.

## What changes in the central claim

Recommended wording:

> Task-relevant state information can remain recoverable while reasoning redirects the
> model's action distribution. These redirects occur across several verbal reasoning
> functions rather than at one stereotyped “aha” or correction sentence. Sentence
> function is therefore not the missing conversion mechanism itself; it helps identify
> uncertainty regimes and candidate reasoning content for causal intervention.

This is sharper than saying that action changes are “semantic transitions.” The current
evidence instead supports a **reasoning-mediated conversion failure** whose observable
sentence-level form is heterogeneous.

## Cross-analysis conclusions

| Analysis | What the labels contribute | Conclusion |
|---|---|---|
| Previous belief--action-gap evidence | Interpretive bridge only; the old failure slice was not labeled with this taxonomy | Correct local reports can coexist with inconsistent actions, but current labels do not retrospectively explain those earlier cases |
| BEAST change points | Direct matched test | No reliable four-way semantic signature across {semantic_metrics['primary_pairs']} close pairs |
| Exact action and optimality events | Direct semantic join | No event association survives correction; smallest q = {semantic_metrics['exact_action_min_q']:.3f} |
| Commitment and post-commitment reasoning | Partial direct join | Commitment onset has no semantic association; the labels do not cover all tail sentences, so they cannot establish that post-commitment text is semantically useless |
| Activation transitions | Direct semantic join | No activation-geometry association survives correction; smallest q = {semantic_metrics['activation_min_q']:.3f} |
| Behavioral belief transitions | Direct semantic join | Belief error does not vary reliably, but belief entropy does: route deliberation averages {semantic_metrics['entropy_route_mean']:.3f} bits, q = {semantic_metrics['entropy_q']:.3f} |
| Immediate-action attention | Direct join on 67 matched attention pairs | The increase is largest for summary/restatement sentences ({semantic_metrics['attention_summary_pp']:.3f} percentage points per token), but semantic heterogeneity is exploratory (p = {semantic_metrics['attention_p']:.3f}) |
| Prospective event monitor | Not a direct join: its “text semantic” features are embedding novelty, not these labels | Do not attribute its small large-distribution-change result to the four-way taxonomy |

The machine-readable version is `semantic_cross_analysis_decision_table.csv`.

## How this sharpens the experiment chain

**Previous behavioral and causal evidence.** The paper's strongest behavioral example
is that all three eligible wall-hit cases preserve a correct local wall report while
the model still takes the blocked action. Its strongest intervention result is that
patched outputs follow the donor reasoning action in 100% of cases, with a 76% action
flip rate, whereas pre-reasoning activation patching alone has little detectable
effect. The current semantic labels do not re-estimate those effects because they were
collected on a different matched sentence cohort. They sharpen the follow-up question:
which *kind of reasoning content* should be deleted or replaced when correct measured
beliefs coexist with an optimality loss?

**Action distributions and BEAST.** BEAST supplies retrospective candidate boundaries,
not a linguistic mechanism. The semantic null is useful here: it prevents us from
renaming statistical change points as correction, verification, or commitment
sentences. The strongest warranted interpretation remains an action-routing boundary.

**Activation and belief-transition analyses.** Sentence activation geometry is
associated with some same-position events but is not a reliable prospective failure
monitor, and its variation is not explained by broad semantic function. Behavioral
belief readouts do provide modest three-sentence warning of optimality loss, while the
semantic labels instead distinguish the uncertainty regime at the boundary. These are
complementary roles: beliefs carry the prospective signal; semantics interprets what
the model is verbally doing.

**Attention.** The immediate action readout increases attention to the newest sentence
when its recommendation changes. The semantic breakdown suggests that compressed
summaries or restatements may receive the largest increase, but the heterogeneity test
is only borderline. This is a prioritization result for patching, not evidence that
summaries generally cause action changes.

## Expected and counterintuitive observations

Expected:

- Route deliberation dominates both change-point and comparison sentences and is
  numerically more common at change points.
- Belief uncertainty is highest while the model is checking, revising, or constructing
  routes.

Counterintuitive:

- Corrections, checks, and summaries do not form a distinctive change-point class.
- Semantic function does not reliably distinguish harmful action changes from
  recoveries.
- The attention increase is numerically largest for summaries/restatements, whereas
  belief uncertainty is largest for route deliberation. Attention routing and belief
  uncertainty therefore appear to be different parts of the process.
- {semantic_metrics['zero_error_loss_candidates']} of {semantic_metrics['primary_optimality_losses']}
  exact optimality losses at primary change points
  ({semantic_metrics['zero_error_loss_fraction']:.1%}) occur with correct answers on
  all nine available categorical belief readouts at that boundary. Their semantic
  functions are heterogeneous.

## How the labels should be used

Use them to:

1. prevent post-hoc “aha moment” stories;
2. stratify belief uncertainty by reasoning function;
3. select heterogeneous intervention candidates;
4. choose clear qualitative examples while preserving annotation uncertainty.

Do not use them as:

- a real-time change-point predictor;
- human-ground-truth reasoning states;
- token-level mechanism labels;
- evidence that a discourse function caused an action transition.

## Best causal follow-up

Start with `semantic_intervention_candidates.csv`. These rows are primary BEAST
change points where the recommendation becomes suboptimal while all nine available
categorical belief readouts are correct. Delete, replace, or resample the target
sentence, then re-elicit the immediate action distribution and the same belief probes.
Stratify the sample across state readout, route deliberation, summary or restatement,
and other functions instead of selecting only the most intuitive examples.

If changing a sentence reverses the action without changing the measured beliefs, that
would directly support a failure in converting available information into action. If
the belief reports also change, the result is better described as a belief-mediated
transition. If neither changes, the sentence is a marker rather than a causal driver.

## Scope

The semantic taxonomy is AI-assisted and has stronger reliability at the broad
four-way level than at the fine-label level. BEAST is retrospective, and the unit is a
whole sentence. The results do not rule out sparse token-level forks within otherwise
ordinary sentences.
"""
    (output_dir / "SEMANTIC_CROSS_ANALYSIS_SYNTHESIS.md").write_text(text)

    candidates = pd.read_csv(output_dir / "semantic_intervention_candidates.csv")
    candidate_lines = [
        "# Semantic Intervention Candidates",
        "",
        "Primary retrospective change points where the recommendation changes from",
        "planner-optimal to suboptimal and all nine categorical belief readouts",
        "(six current-state and three chosen-action consequences) are correct at",
        "that sentence boundary.",
        "",
        "| ID | Sentence | Semantic function | Action shift | Belief entropy | BEAST posterior |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in candidates.itertuples():
        sentence = str(row.target_sentence).replace("|", "\\|").replace("\n", " ")
        if len(sentence) > 110:
            sentence = sentence[:107].rstrip() + "..."
        candidate_lines.append(
            f"| `{row.sentence_id}` | “{sentence}” | "
            f"{str(row.primary_label).replace('_', ' ')} | "
            f"{row.previous_action_label} → {row.action_label} | "
            f"{row.mean_belief_entropy_bits:.3f} | "
            f"{row.posterior_change_probability:.3f} |"
        )
    candidate_lines.extend(
        [
            "",
            "These are observational candidates, not causal examples. Zero probe",
            "error covers the nine available categorical questions; it does not prove that",
            "every task-relevant latent belief is correct. Fine semantic labels should",
            "also be checked against the context before quoting an individual row.",
        ]
    )
    (output_dir / "SEMANTIC_INTERVENTION_CANDIDATES.md").write_text(
        "\n".join(candidate_lines) + "\n"
    )

    paper_insert = rf"""\paragraph{{Semantic interpretation of action-distribution change points.}}
We classified the discourse function of 160 retrospectively detected
action-distribution change-point sentences and 160 same-state matched sentences while
blinding the judge to change-point status, actions, activations, and behavioral
beliefs. The primary analysis uses {semantic_metrics['primary_pairs']} close matched
pairs and a calibrated four-way grouping. Semantic composition did not differ
reliably between change points and matched sentences (paired permutation
$p=.219$), and adding sentence function to reasoning progress and sentence length
changed trajectory-held-out AUROC by only $+.029$ (95\% trajectory-bootstrap interval
$[-.005,+.066]$). Semantic function also did not reliably distinguish recommendation
changes, optimality losses, recoveries, or retrospective commitment onsets (smallest
corrected $q={semantic_metrics['exact_action_min_q']:.3f}$). Thus, the detected
boundaries should be interpreted as action-routing transitions rather than as one
stereotyped verbal operation.

The positive semantic result was a difference in behavioral-belief uncertainty:
mean categorical-belief entropy was highest during route deliberation
({semantic_metrics['entropy_route_mean']:.3f} bits; within-trajectory permutation
$q={semantic_metrics['entropy_q']:.3f}$). In the immediate-action attention subset,
the newest-sentence attention increase was numerically largest for summaries or
restatements ({semantic_metrics['attention_summary_pp']:.3f} percentage points per
token), although heterogeneity across semantic functions was exploratory
($p={semantic_metrics['attention_p']:.3f}$). Finally,
{semantic_metrics['zero_error_loss_candidates']} of {semantic_metrics['primary_optimality_losses']}
optimality losses at primary change points occurred while all nine available
categorical belief readouts were correct at that boundary. These heterogeneous cases
are candidates for sentence deletion, replacement, and resampling interventions; they
do not by themselves establish a causal sentence mechanism.

\begin{{figure}}[t]
    \centering
    \includegraphics[width=\linewidth]{{outputs/reader_facing/belief_action_reasoning_summary_v1/figs/semantic_labels_across_analyses.pdf}}
    \caption{{What sentence-level semantic labels add to the action-selection
    analysis. Panel A compares broad semantic-function rates in retrospective BEAST
    change-point sentences and same-state, progress- and length-matched sentences.
    Panel B reports incremental trajectory-held-out change-point discrimination.
    Panel C shows behavioral-belief entropy at change points. Panel D stratifies the
    immediate-action attention increase to the newest sentence. The four-way labels
    are calibrated AI-assisted judgments rather than human ground truth; panels C and
    D are exploratory, and all analyses are observational.}}
    \label{{fig:semantic_action_change_synthesis}}
\end{{figure}}

\begin{{table}}[t]
\centering
\small
\begin{{tabular}}{{p{{0.25\linewidth}}p{{0.27\linewidth}}p{{0.38\linewidth}}}}
\toprule
Analysis & Semantic result & Implication \\
\midrule
Matched change points & No reliable four-way signature ($p=.219$) &
Do not equate a statistical change point with a correction or ``aha'' sentence. \\
Held-out discrimination & AUROC change $+.029$ $[-.005,+.066]$ &
Sentence function is not a dependable action-change monitor. \\
Exact action events & No association survives correction (minimum $q={semantic_metrics['exact_action_min_q']:.3f}$) &
No named discourse operation reliably separates harmful changes from recoveries. \\
Behavioral beliefs & Route deliberation has highest entropy ({semantic_metrics['entropy_route_mean']:.3f} bits; $q={semantic_metrics['entropy_q']:.3f}$) &
Semantic function identifies an uncertainty regime rather than an error regime. \\
Immediate-action attention & Summary/restatement largest; heterogeneity $p={semantic_metrics['attention_p']:.3f}$ &
Useful for intervention prioritization, not a causal conclusion. \\
\bottomrule
\end{{tabular}}
\caption{{Cross-analysis utility of the broad semantic-function labels.}}
\label{{tab:semantic_action_change_synthesis}}
\end{{table}}
"""
    (output_dir / "SEMANTIC_PAPER_INSERT.tex").write_text(paper_insert)


def write_docs(output_dir: Path, semantic_metrics: dict[str, float | int]) -> None:
    captions = """# Figure Captions

## belief_readouts_forecast_optimality_loss.png

Trajectory-held-out prediction of whether the model's recommended action will change
from planner-optimal to suboptimal after the current reasoning sentence. The behavioral
baseline uses reasoning progress, current action confidence, and the recent change in
the probability distribution over UP, DOWN, LEFT, and RIGHT. The belief model adds
behavioral readout probabilities, uncertainty, answer changes, and conflicts between
the recommended action and reported walls or action consequences. The right panel shows
the AUPRC improvement and 95% trajectory-bootstrap interval. Belief readouts improve the
three-sentence forecast but not conclusively the next-sentence forecast.

## action_distribution_changes_near_events.png

Fraction of 160 offline Bayesian change points in the elicited action-probability
sequence that fall within three reasoning sentences before or after each event.
Progress-matched comparisons sample sentences from the same environment state and the
same tenth of reasoning progress. Error bars show 95% trajectory-bootstrap intervals
for detected change points and the central 95% of 2,000 randomizations for comparisons.
The association is strongest for changes in the highest-probability recommended action.
It does not establish that a reasoning sentence caused the action change.

## activation_monitor_baseline_comparison.png

Trajectory-held-out classification of action events at the current sentence position
in 7,038 positions from 46 environment states and 31 trajectories. The activation
result is the best ridge logistic monitor across GPT-OSS-20B layers 8, 15, and 23 and
sentence-mean or sentence-final representations. Behavioral readouts are scalar action
and belief variables available at the same position. Points are AUROC estimates and
bars are 95% trajectory-bootstrap intervals. This is contemporaneous classification,
not advance prediction.

## attention_changes_when_recommended_action_changes.png

Change in attention from the immediate action-readout token to each prompt region when
the replayed recommended action changes, relative to a progress-matched sentence from
the same environment state where the action remains stable. Points show layer-15 means
for 67 matched pairs from 35 states and 28 trajectories; bars are 95% trajectory-
bootstrap intervals. Attention per token increases most for the newest reasoning
sentence. Attention is correlational and does not prove causal use.

## semantic_labels_across_analyses.png

Cross-analysis role of broad sentence-function labels in 152 close, same-state matched
pairs. Panel A shows paired differences in semantic-function rates between BEAST
change-point sentences and matched sentences. Panel B shows the change in
trajectory-held-out change-point AUROC when semantic labels, activation geometry, or
both are added to reasoning progress and sentence length. Panel C shows categorical
behavioral-belief entropy at change points by semantic function, with
trajectory-bootstrap intervals. Panel D shows the semantic breakdown of the
immediate-action attention increase to the newest sentence; bars are 1.96 times the
trajectory-level standard error. Panels C and D are exploratory. BEAST and the
semantic comparisons are retrospective, and none of the panels is a causal test.
"""
    (output_dir / "FIGURE_CAPTIONS.md").write_text(captions)

    index = """# Reader-Facing Evidence

## Recommended figures

1. `figs/belief_readouts_forecast_optimality_loss.png`
   - Main prospective result: observable belief readouts provide modest warning of
     optimality loss within the next three reasoning sentences.
2. `figs/action_distribution_changes_near_events.png`
   - Behavioral structure: abrupt changes in action probabilities cluster around
     changes in the recommended action.
3. `figs/attention_changes_when_recommended_action_changes.png`
   - Mechanistic clue: the immediate action readout increases attention to the newest
     reasoning sentence when its recommendation changes.
4. `figs/activation_monitor_baseline_comparison.png`
   - Negative comparison: supervised activation monitors do not outperform behavioral
     action and belief readouts.
5. `figs/semantic_labels_across_analyses.png`
   - Interpretive synthesis: semantic labels constrain simple discourse-function
     stories, identify a high-belief-uncertainty regime, and prioritize causal
     intervention candidates.

## Semantic synthesis

Use `SEMANTIC_CROSS_ANALYSIS_SYNTHESIS.md` for the claims that can be carried into the
paper and `semantic_cross_analysis_decision_table.csv` for their numerical provenance.
`SEMANTIC_PAPER_INSERT.tex` contains a copy-ready results paragraph, figure block, and
compact table.
The main semantic conclusion is not that one reasoning function causes action changes.
It is that action-routing boundaries are semantically heterogeneous, while route
deliberation marks elevated belief uncertainty.

## Main paper context

The existing paper results remain necessary context: state facts are decodable before
reasoning, inferred action values favor optimal actions more often than realized
behavior, reasoning-trace patching changes final actions, and correct wall reports can
coexist with wall-hit actions. The new figures do not replace those results.

## Appendix or diagnostic only

- Generic activation-distance curves: associations are difficult to interpret and do
  not prospectively predict optimality loss.
- State-belief entropy over reasoning progress: failure and control curves are similar
  in this pilot.
- Event-aligned belief-error curves: useful for inspection, but uncertainty is omitted
  and the main averages are mostly flat.
- Commitment timing and post-commitment length: descriptive and based on a
  retrospective boundary.
- Final-action attention to event windows: replaced by immediate action-readout
  attention with progress-matched controls.
"""
    (output_dir / "READER_FACING_FIGURES.md").write_text(index)

    audit = f"""# Evidence Audit and Central Claim

## Defensible conclusion

In these DoorKey trajectories, failures cannot be explained only by absent state
information. Relevant state facts are often recoverable, while action selection changes
with the model's reasoning context. Behavioral belief readouts provide modest advance
warning of optimality loss, abrupt action-probability changes occur near recommendation
changes, and the immediate action readout places more attention on the newest reasoning
sentence when its recommendation changes. Together with the paper's reasoning-trace
interventions, the evidence supports reasoning-mediated action selection rather than a
simple loss-of-state-knowledge account.

This conclusion is specific to GPT-OSS-20B in DoorKey and should use "some failures,"
not "all failures."

## Evidence and limits

| Evidence | Result | Role in the conclusion | Main limitation |
|---|---|---|---|
| State and value readouts in the paper | State facts are decodable, and inferred values favor optimal actions more often than realized actions | Establishes information availability and a belief-action discrepancy | Probe validity and task scope |
| Reasoning-trace intervention in the paper | Patched reasoning determines the final action far more strongly than pre-reasoning activation changes | Strongest evidence that reasoning context controls the final action | The current paper table should report intervention sample sizes and uncertainty |
| Prospective belief monitor | Three-sentence optimality-loss AUPRC rises from 0.273 to 0.334, change +0.061, 95% CI [+0.014, +0.123] | Shows practical short-horizon association between observable beliefs and failure | Next-sentence result is inconclusive; only 31 trajectories |
| Offline action-distribution change points | 86.9% are near recommendation changes versus 65.3% for progress-matched sentences | Identifies candidate reasoning boundaries for intervention | Offline and correlational; earlier probability vectors are not exactly reproducible |
| Immediate action attention | Layer-15 attention to the newest reasoning sentence rises by 0.0838 percentage points per token, 95% CI [0.0442, 0.1314] | Suggests changed recommendations are associated with the newest reasoning content | Attention is not causal evidence; only 67 matched pairs |
| Supervised activation monitors | Best activation AUROC is below behavioral readouts for every event | Rules out a simple claim that generic activation features are the best monitor | Monitors use selected layers and aggregate sentence representations |
| Immediate belief-transition models | Current-state and action-consequence readouts do not robustly improve immediate transition prediction | Prevents claiming that measured belief changes directly explain action changes | Probe set and behavioral elicitation may miss the relevant transition belief |
| Blinded sentence-function labels | No reliable change-point signature or exact action-event profile; route deliberation has the highest belief entropy | Rules out a single verbal-operation account and identifies an uncertainty regime | AI-assisted sentence labels; offline change points; no token or causal localization |

## Claims not supported

- Measured belief changes directly cause recommendation changes.
- Attention proves that grid information was used or ignored.
- Generic activation drift reliably predicts failure.
- Action uncertainty necessarily decreases at commitment.
- Reasoning after retrospective commitment is semantically useless.
- Corrections, verifications, or summaries are a general mechanism for action changes.

## Strong next hypothesis

A reasoning sentence can redirect the action distribution despite stable, recoverable
state information. Test sentences where the recommendation changes from optimal to
suboptimal, state-belief reports remain correct, and attention to the newest sentence
increases. Delete or replace that sentence and re-evaluate the immediate action
distribution. The hypothesis predicts restoration of the previous or planner-optimal
action without a corresponding change in state-belief reports.

This intervention would join the current descriptive results into one causal test:
information remains available, a particular reasoning sentence changes action
selection, and changing that sentence reverses the failure.

## Candidate sentences

Across the matched run, 66 of 161 optimal-to-suboptimal transitions have the same
correct answers immediately before and after the transition for wall-left,
wall-right, wall-up, wall-down, key possession, and door status. Representative
candidates include:

| Environment state and position | Sentence | Elicited action change | Transition |
|---|---|---|---|
| `keepdoor_0`, step 2, sentence 9 | "Row4: # _ # # # D # # #." | LEFT to RIGHT | Sustained optimal to suboptimal |
| `keepdoor_26`, step 8, sentence 76 | "From (5,3) go up to (4,3) which is '#', blocked." | UP to RIGHT | Sustained optimal to suboptimal |
| `keepdoor_33`, step 1, sentence 28 | "Must pick key before door if door blocks path." | DOWN to LEFT | Sustained optimal to suboptimal |

These are observational candidates. The action was elicited after each sentence, so
temporal coincidence does not show that the sentence caused the action change.

The semantic-label join yields a stricter candidate set:
{semantic_metrics['zero_error_loss_candidates']} of
{semantic_metrics['primary_optimality_losses']} exact optimality losses at primary
BEAST change points have correct answers across all nine categorical belief readouts
at that boundary. See `semantic_intervention_candidates.csv`; semantic functions are
deliberately heterogeneous so a causal follow-up does not select only intuitive
examples.
"""
    (output_dir / "EVIDENCE_AUDIT.md").write_text(audit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reader_facing/belief_action_reasoning_summary_v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    figure_dir = output_dir / "figs"
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    plot_belief_forecast(
        Path(
            "outputs/hypothesis_tests/practical_action_event_monitor_v1/model_comparison.csv"
        ),
        figure_dir / "belief_readouts_forecast_optimality_loss.png",
    )
    plot_change_point_proximity(
        Path(
            "outputs/hypothesis_tests/action_distribution_cpd_beast_v1/event_proximity_summary.csv"
        ),
        figure_dir / "action_distribution_changes_near_events.png",
    )
    (
        Path(
            "outputs/hypothesis_tests/action_distribution_cpd_beast_v1/"
            "figs/change_points_near_decision_events.png"
        )
    ).write_bytes(
        (figure_dir / "action_distribution_changes_near_events.png").read_bytes()
    )
    monitor_summary = Path(
        "outputs/hypothesis_tests/transition_activation_commitment_v1/"
        "supervised_activation_monitor_summary.csv"
    )
    plot_monitor_comparison(
        monitor_summary,
        figure_dir / "activation_monitor_baseline_comparison.png",
    )
    plot_monitor_comparison(
        monitor_summary,
        Path(
            "outputs/hypothesis_tests/transition_activation_commitment_v1/"
            "figs/supervised_activation_monitor_auc.png"
        ),
    )

    attention_source = Path(
        "outputs/hypothesis_tests/prefix_action_attention_v1/"
        "figs/attention_changes_when_recommended_action_changes.png"
    )
    attention_target = figure_dir / attention_source.name
    attention_target.write_bytes(attention_source.read_bytes())

    semantic_root = Path(
        "outputs/hypothesis_tests/semantic_reasoning_classification_v1/"
        "final_analysis"
    )
    sentence_path = semantic_root / "sentence_label_cpd_analysis_table.csv"
    matched_path = semantic_root / "matched_semantic_comparisons.csv"
    semantic_model_path = semantic_root / "incremental_model_comparison.csv"
    semantic_event_path = semantic_root / "semantic_event_composition.csv"
    semantic_outcome_path = semantic_root / "semantic_outcome_omnibus_tests.csv"
    semantic_attention_path = semantic_root / "attention_by_semantic_function.csv"
    semantic_attention_test_path = (
        semantic_root / "attention_semantic_omnibus_test.json"
    )
    plot_semantic_cross_analysis(
        matched_path,
        semantic_model_path,
        semantic_event_path,
        semantic_outcome_path,
        semantic_attention_path,
        semantic_attention_test_path,
        figure_dir / "semantic_labels_across_analyses.png",
    )
    semantic_metrics = write_semantic_tables(
        sentence_path,
        matched_path,
        semantic_model_path,
        semantic_event_path,
        semantic_outcome_path,
        semantic_attention_path,
        semantic_attention_test_path,
        output_dir,
    )
    write_semantic_synthesis(output_dir, semantic_metrics)
    write_docs(output_dir, semantic_metrics)
    print(f"Wrote reader-facing evidence summary to {output_dir}")


if __name__ == "__main__":
    main()
