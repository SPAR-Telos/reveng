#!/usr/bin/env python3
"""Build reader-facing scaled matched-46 figures and tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


BLUE = "#1769aa"
DARK = "#0b3c5d"
LIGHT = "#9bd0f5"
PALE = "#dbeafe"
GRID = "#e5eef5"
TEXT = "#172033"

STATE_BELIEFS = ["wall_left", "wall_right", "wall_up", "wall_down", "has_key", "door_open"]
WALL_BELIEFS = ["wall_left", "wall_right", "wall_up", "wall_down"]
ACTION_EFFECT_PREFIXES = ["hit_wall_after", "has_key_after", "door_open_after"]
EVENT_FAMILIES = {
    "action_change": {"action_change"},
    "optimal_to_suboptimal": {"transient_optimal_to_suboptimal", "sustained_optimal_to_suboptimal"},
    "sustained_optimal_to_suboptimal": {"sustained_optimal_to_suboptimal"},
    "suboptimal_to_optimal": {"suboptimal_to_optimal"},
    "commitment_onset": {"commitment_onset"},
}


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
        }
    )


def bootstrap_mean_ci(values: pd.Series | np.ndarray, *, seed: int = 42, repeats: int = 2000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(repeats, len(arr)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(samples, [0.025, 0.975]))


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def event_family(event_type: str) -> str:
    if event_type in {"transient_optimal_to_suboptimal", "sustained_optimal_to_suboptimal"}:
        return "optimal_to_suboptimal"
    return event_type


def load_inputs(exp1: Path, exp2: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positions = pd.read_csv(exp1 / "position_rows.csv")
    events = pd.read_csv(exp1 / "event_rows.csv")
    beliefs = pd.read_csv(exp2 / "belief_rows.csv")
    positions["commitment_onset_bool"] = normalize_bool(positions["commitment_onset"])
    positions["action_committed_bool"] = normalize_bool(positions["action_committed"])
    return positions, events, beliefs


def build_commitment_timing(positions: pd.DataFrame, out: Path) -> pd.DataFrame:
    commitments = positions[positions["commitment_onset_bool"]].copy()
    commitments["commitment_sentence_index"] = commitments["position_index"].astype(int)
    commitments["commitment_progress"] = commitments["reasoning_progress"].astype(float)
    total_sentences = positions.groupby("example_id").size().rename("total_sentence_prefixes")
    commitments = commitments.join(total_sentences, on="example_id")
    rows: list[dict[str, Any]] = []
    for group_name, group in [("all", commitments), ("failure", commitments[commitments["matched_role"] == "failure"]), ("control", commitments[commitments["matched_role"] == "control"])]:
        low, high = bootstrap_mean_ci(group["commitment_sentence_index"], seed=43)
        progress_low, progress_high = bootstrap_mean_ci(group["commitment_progress"], seed=44)
        rows.append(
            {
                "group": group_name,
                "n_states": group["example_id"].nunique(),
                "mean_commitment_sentence_index": group["commitment_sentence_index"].mean(),
                "median_commitment_sentence_index": group["commitment_sentence_index"].median(),
                "commitment_sentence_ci_low": low,
                "commitment_sentence_ci_high": high,
                "mean_commitment_progress": group["commitment_progress"].mean(),
                "median_commitment_progress": group["commitment_progress"].median(),
                "commitment_progress_ci_low": progress_low,
                "commitment_progress_ci_high": progress_high,
                "mean_total_sentence_prefixes": group["total_sentence_prefixes"].mean(),
                "max_total_sentence_prefixes": int(group["total_sentence_prefixes"].max()) if len(group) else 0,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "commitment_timing_by_state_group.csv", index=False)

    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    order = ["control", "failure"]
    colors = {"control": LIGHT, "failure": BLUE}
    rng = np.random.default_rng(42)
    for ax, value_col, low_col, high_col, y_label in [
        (
            axes[0],
            "commitment_sentence_index",
            "commitment_sentence_ci_low",
            "commitment_sentence_ci_high",
            "Reasoning sentences revealed",
        ),
        (
            axes[1],
            "commitment_progress",
            "commitment_progress_ci_low",
            "commitment_progress_ci_high",
            "Reasoning characters revealed (fraction of trace)",
        ),
    ]:
        for i, group_name in enumerate(order):
            group = commitments[commitments["matched_role"] == group_name]
            jitter = rng.normal(0.0, 0.035, size=len(group))
            ax.scatter(
                np.full(len(group), i) + jitter,
                group[value_col],
                s=26,
                color=colors[group_name],
                alpha=0.55,
                edgecolor="white",
                linewidth=0.4,
            )
            row = summary[summary["group"] == group_name].iloc[0]
            mean_col = "mean_" + value_col
            ax.errorbar(
                i,
                row[mean_col],
                yerr=np.array(
                    [
                        [row[mean_col] - row[low_col]],
                        [row[high_col] - row[mean_col]],
                    ]
                ),
                fmt="o",
                markersize=7,
                capsize=4,
                color=DARK,
                label="Mean with 95% bootstrap CI" if i == 0 else None,
            )
        labels = []
        for group_name in order:
            row = summary[summary.group == group_name].iloc[0]
            label = "Control" if group_name == "control" else "Failure"
            labels.append(f"{label}\n(n={int(row.n_states)} states)")
        ax.set_xticks([0, 1], labels=labels)
        ax.set_ylabel(y_label)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            ax.legend(frameon=False, fontsize=8)
    labels = [
        f"Control\n(n={int(summary[summary.group == 'control'].n_states.iloc[0])})",
        f"Failure\n(n={int(summary[summary.group == 'failure'].n_states.iloc[0])})",
    ]
    fig.suptitle("Action Commitment Timing by State Group")
    fig.tight_layout()
    fig.savefig(out / "figs" / "commitment_timing_by_state_group.png", dpi=220)
    plt.close(fig)
    return summary


def build_entropy_decision_events(positions: pd.DataFrame, events: pd.DataFrame, out: Path) -> pd.DataFrame:
    position_lookup = {
        (row.example_id, int(row.reasoning_step_idx)): row
        for row in positions.itertuples()
    }
    rows: list[dict[str, Any]] = []
    for event in events.itertuples():
        cur_idx = int(event.event_reasoning_step_idx)
        prev_idx = cur_idx - 1
        cur = position_lookup.get((event.example_id, cur_idx))
        prev = position_lookup.get((event.example_id, prev_idx))
        if cur is None or prev is None:
            continue
        family = event_family(event.event_type)
        rows.append(
            {
                "event_type": event.event_type,
                "event_family": family,
                "example_id": event.example_id,
                "trajectory_id": event.trajectory_id,
                "matched_role": getattr(cur, "matched_role", ""),
                "current_reasoning_step_idx": cur_idx,
                "previous_action": event.previous_action,
                "current_action": event.current_action,
                "previous_action_entropy_bits": float(prev.action_entropy_bits),
                "current_action_entropy_bits": float(cur.action_entropy_bits),
                "delta_action_entropy_bits": float(cur.action_entropy_bits) - float(prev.action_entropy_bits),
                "previous_action_confidence": float(prev.action_confidence),
                "current_action_confidence": float(cur.action_confidence),
                "delta_action_confidence": float(cur.action_confidence) - float(prev.action_confidence),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "action_entropy_decision_event_rows.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    order = ["action_change", "optimal_to_suboptimal", "sustained_optimal_to_suboptimal", "suboptimal_to_optimal", "commitment_onset"]
    for family in order:
        subset = frame[frame["event_family"].eq(family) if family != "sustained_optimal_to_suboptimal" else frame["event_type"].eq(family)]
        if subset.empty:
            continue
        low, high = bootstrap_mean_ci(subset["delta_action_entropy_bits"], seed=45)
        conf_low, conf_high = bootstrap_mean_ci(subset["delta_action_confidence"], seed=46)
        summary_rows.append(
            {
                "event_family": family,
                "n_events": len(subset),
                "mean_delta_action_entropy_bits": subset["delta_action_entropy_bits"].mean(),
                "median_delta_action_entropy_bits": subset["delta_action_entropy_bits"].median(),
                "delta_action_entropy_ci_low": low,
                "delta_action_entropy_ci_high": high,
                "mean_delta_action_confidence": subset["delta_action_confidence"].mean(),
                "median_delta_action_confidence": subset["delta_action_confidence"].median(),
                "delta_action_confidence_ci_low": conf_low,
                "delta_action_confidence_ci_high": conf_high,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "action_entropy_decision_event_summary.csv", index=False)

    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10.2, 4.5))
    labels = {
        "action_change": "Action\nchanges",
        "optimal_to_suboptimal": "Becomes\nsuboptimal",
        "sustained_optimal_to_suboptimal": "Becomes suboptimal\nand stays suboptimal",
        "suboptimal_to_optimal": "Becomes\noptimal",
        "commitment_onset": "Action first\nremains stable",
    }
    plot_rows = summary[summary["event_family"].isin(order)].set_index("event_family").loc[
        [x for x in order if x in set(summary["event_family"])]
    ]
    x = np.arange(len(plot_rows))
    means = plot_rows["mean_delta_action_entropy_bits"].to_numpy(dtype=float)
    yerr = np.vstack(
        [
            means - plot_rows["delta_action_entropy_ci_low"].to_numpy(dtype=float),
            plot_rows["delta_action_entropy_ci_high"].to_numpy(dtype=float) - means,
        ]
    )
    colors = [BLUE if m < 0 else LIGHT for m in means]
    ax.bar(x, means, color=colors, edgecolor=DARK, linewidth=0.7)
    ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor=DARK, capsize=3, linewidth=1)
    ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks(x, [f"{labels[idx]}\n(n={int(plot_rows.loc[idx, 'n_events'])})" for idx in plot_rows.index])
    ax.set_ylabel("Entropy after event minus entropy before event (bits)")
    ax.set_title("Change in Action Uncertainty at Recommendation Events")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "figs" / "action_entropy_change_decision_events.png", dpi=220)
    plt.close(fig)
    return summary


def _belief_family(question_id: str, question_family: str) -> str:
    if question_id in WALL_BELIEFS or question_id.startswith("hit_wall_after"):
        return "wall"
    if question_id == "has_key" or question_id.startswith("has_key_after"):
        return "key"
    if question_id == "door_open" or question_id.startswith("door_open_after"):
        return "door"
    return question_family


def build_belief_error_alignment(beliefs: pd.DataFrame, events: pd.DataFrame, out: Path) -> pd.DataFrame:
    beliefs = beliefs.copy()
    beliefs["belief_group"] = [
        _belief_family(qid, fam) for qid, fam in zip(beliefs["question_id"], beliefs["question_family"], strict=True)
    ]
    event_rows = events[
        events["event_type"].isin(
            {"action_change", "transient_optimal_to_suboptimal", "sustained_optimal_to_suboptimal", "suboptimal_to_optimal"}
        )
    ].copy()
    event_rows["event_family"] = event_rows["event_type"].map(event_family)

    rows: list[dict[str, Any]] = []
    groups = ["all", "wall", "key", "door", "wall_down"]
    for event in event_rows.itertuples():
        for offset in range(-5, 6):
            step = int(event.event_reasoning_step_idx) + offset
            subset = beliefs[(beliefs["example_id"] == event.example_id) & (beliefs["reasoning_step_idx"].astype(int) == step)]
            if subset.empty:
                continue
            for group_name in groups:
                if group_name == "all":
                    group = subset[subset["question_id"].isin(STATE_BELIEFS)]
                elif group_name == "wall_down":
                    group = subset[subset["question_id"].eq("wall_down")]
                else:
                    group = subset[subset["belief_group"].eq(group_name)]
                if group.empty:
                    continue
                rows.append(
                    {
                        "event_id": event.event_id,
                        "event_family": event.event_family,
                        "event_type": event.event_type,
                        "example_id": event.example_id,
                        "trajectory_id": event.trajectory_id,
                        "relative_sentence_offset": offset,
                        "belief_group": group_name,
                        "n_probe_rows": len(group),
                        "belief_error_rate": pd.to_numeric(group["belief_is_error"], errors="coerce").mean(),
                    }
                )
    aligned = pd.DataFrame(rows)
    aligned.to_csv(out / "belief_error_around_optimality_change_rows.csv", index=False)
    summary = (
        aligned.groupby(["event_family", "relative_sentence_offset", "belief_group"], as_index=False)
        .agg(
            n_events=("event_id", "nunique"),
            mean_belief_error_rate=("belief_error_rate", "mean"),
            median_belief_error_rate=("belief_error_rate", "median"),
        )
    )
    summary.to_csv(out / "belief_error_around_optimality_change_summary.csv", index=False)

    granular = (
        beliefs[beliefs["question_id"].isin(STATE_BELIEFS)]
        .groupby(["question_id", "matched_role"], as_index=False)
        .agg(
            n_probe_rows=("belief_is_error", "size"),
            mean_belief_error_rate=("belief_is_error", "mean"),
            mean_entropy_bits=("categorical_entropy_bits", "mean"),
        )
    )
    granular.to_csv(out / "granular_current_state_belief_summary.csv", index=False)

    setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3), sharey=True)
    event_order = ["action_change", "optimal_to_suboptimal", "suboptimal_to_optimal"]
    colors = {"all": DARK, "wall": BLUE, "key": "#3f8fc7", "door": LIGHT, "wall_down": "#6b8fb3"}
    labels = {"all": "All current-state beliefs", "wall": "Wall beliefs", "key": "Key belief", "door": "Door belief", "wall_down": "Wall-down belief"}
    for ax, family in zip(axes, event_order, strict=True):
        family_summary = summary[summary["event_family"] == family]
        for group_name in ["all", "wall", "key", "door", "wall_down"]:
            subset = family_summary[family_summary["belief_group"] == group_name].sort_values("relative_sentence_offset")
            if subset.empty:
                continue
            ax.plot(
                subset["relative_sentence_offset"],
                subset["mean_belief_error_rate"],
                marker="o",
                markersize=3,
                linewidth=1.8,
                color=colors[group_name],
                label=labels[group_name],
            )
        ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
        if family == "action_change":
            ax.set_title("Recommended action\nchanges")
            ax.set_xlabel("Sentence position relative to action change")
        elif family == "optimal_to_suboptimal":
            ax.set_title("Recommended action\nbecomes suboptimal")
            ax.set_xlabel("Sentence position relative to optimality loss")
        else:
            ax.set_title("Recommended action\nbecomes optimal")
            ax.set_xlabel("Sentence position relative to optimality recovery")
        ax.grid(color=GRID, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Fraction of belief questions answered incorrectly")
    axes[2].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("Wall, Key, and Door Belief Errors Around Action and Optimality Changes", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "figs" / "belief_errors_before_after_optimality_changes.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


def build_belief_entropy_around_commitment(
    positions: pd.DataFrame,
    beliefs: pd.DataFrame,
    out: Path,
    *,
    window: int = 10,
) -> pd.DataFrame:
    commitments = positions[positions["commitment_onset_bool"]].set_index("example_id")["position_index"].astype(int).to_dict()
    belief_frame = beliefs[beliefs["question_id"].isin(STATE_BELIEFS)].copy()
    belief_frame["commitment_position_index"] = belief_frame["example_id"].map(commitments)
    belief_frame = belief_frame[belief_frame["commitment_position_index"].notna()].copy()
    belief_frame["relative_commitment_offset"] = (
        belief_frame["position_index"].astype(int) - belief_frame["commitment_position_index"].astype(int)
    )
    belief_frame = belief_frame[belief_frame["relative_commitment_offset"].between(-window, window)].copy()
    belief_frame["belief_group"] = [
        _belief_family(qid, fam)
        for qid, fam in zip(belief_frame["question_id"], belief_frame["question_family"], strict=True)
    ]
    belief_frame["entropy_value"] = pd.to_numeric(belief_frame["categorical_entropy_bits"], errors="coerce")

    rows: list[dict[str, Any]] = []
    for offset, group in belief_frame.groupby("relative_commitment_offset"):
        current = group[group["question_id"].isin(STATE_BELIEFS)]
        rows.append(
            {
                "level": "aggregate",
                "belief": "all_current_state_beliefs",
                "relative_commitment_offset": int(offset),
                "n_states": current["example_id"].nunique(),
                "n_probe_rows": len(current),
                "mean_entropy_bits": current["entropy_value"].mean(),
                "median_entropy_bits": current["entropy_value"].median(),
            }
        )
    for (offset, belief_group), group in belief_frame.groupby(["relative_commitment_offset", "belief_group"]):
        rows.append(
            {
                "level": "family",
                "belief": belief_group,
                "relative_commitment_offset": int(offset),
                "n_states": group["example_id"].nunique(),
                "n_probe_rows": len(group),
                "mean_entropy_bits": group["entropy_value"].mean(),
                "median_entropy_bits": group["entropy_value"].median(),
            }
        )
    for (offset, question_id), group in belief_frame.groupby(["relative_commitment_offset", "question_id"]):
        rows.append(
            {
                "level": "granular",
                "belief": question_id,
                "relative_commitment_offset": int(offset),
                "n_states": group["example_id"].nunique(),
                "n_probe_rows": len(group),
                "mean_entropy_bits": group["entropy_value"].mean(),
                "median_entropy_bits": group["entropy_value"].median(),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "belief_entropy_around_commitment_summary.csv", index=False)

    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    subset = summary[(summary["level"] == "aggregate") & (summary["belief"] == "all_current_state_beliefs")].sort_values("relative_commitment_offset")
    ax.plot(subset["relative_commitment_offset"], subset["mean_entropy_bits"], marker="o", linewidth=2.0, color=DARK)
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_title("Current-State Belief Uncertainty Around Action Commitment")
    ax.set_xlabel("Sentence position relative to action commitment")
    ax.set_ylabel("Mean entropy over wall, key, and door answers (bits)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "figs" / "belief_entropy_around_commitment_aggregate.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    colors = {"wall": BLUE, "key": "#3f8fc7", "door": LIGHT}
    labels = {"wall": "Wall beliefs", "key": "Key belief", "door": "Door belief"}
    for group_name in ["wall", "key", "door"]:
        subset = summary[(summary["level"] == "family") & (summary["belief"] == group_name)].sort_values("relative_commitment_offset")
        if subset.empty:
            continue
        ax.plot(
            subset["relative_commitment_offset"],
            subset["mean_entropy_bits"],
            marker="o",
            markersize=3,
            linewidth=1.8,
            color=colors[group_name],
            label=labels[group_name],
        )
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_title("Wall, Key, and Door Belief Uncertainty Around Action Commitment")
    ax.set_xlabel("Sentence position relative to action commitment")
    ax.set_ylabel("Mean answer entropy (bits)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "figs" / "belief_entropy_around_commitment_by_family.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    colors = {
        "wall_left": "#0b3c5d",
        "wall_right": "#1769aa",
        "wall_up": "#3f8fc7",
        "wall_down": "#6b8fb3",
        "has_key": "#9bd0f5",
        "door_open": "#b9dff7",
    }
    labels = {
        "wall_left": "wall left",
        "wall_right": "wall right",
        "wall_up": "wall up",
        "wall_down": "wall down",
        "has_key": "key held",
        "door_open": "door open",
    }
    for question_id in STATE_BELIEFS:
        subset = summary[(summary["level"] == "granular") & (summary["belief"] == question_id)].sort_values("relative_commitment_offset")
        if subset.empty:
            continue
        ax.plot(
            subset["relative_commitment_offset"],
            subset["mean_entropy_bits"],
            marker="o",
            markersize=3,
            linewidth=1.6,
            color=colors[question_id],
            label=labels[question_id],
        )
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_title("Uncertainty for Individual State Questions Around Action Commitment")
    ax.set_xlabel("Sentence position relative to action commitment")
    ax.set_ylabel("Mean answer entropy (bits)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(out / "figs" / "belief_entropy_around_commitment_by_question.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


def build_factorization_table(positions: pd.DataFrame, beliefs: pd.DataFrame, events: pd.DataFrame, out: Path) -> pd.DataFrame:
    event_map: dict[tuple[str, int], set[str]] = {}
    for row in events.itertuples():
        key = (row.example_id, int(row.event_reasoning_step_idx))
        event_map.setdefault(key, set()).add(row.event_type)

    belief_pivot = beliefs.pivot_table(
        index=["example_id", "reasoning_step_idx"],
        columns="question_id",
        values="belief_is_error",
        aggfunc="max",
    )
    answer_pivot = beliefs.pivot_table(
        index=["example_id", "reasoning_step_idx"],
        columns="question_id",
        values="answer_key",
        aggfunc="first",
    )

    rows: list[dict[str, Any]] = []
    ordered = positions.sort_values(["example_id", "reasoning_step_idx"])
    for example_id, group in ordered.groupby("example_id"):
        group = group.sort_values("reasoning_step_idx")
        max_idx = int(group["reasoning_step_idx"].max())
        for pos in group.itertuples():
            step = int(pos.reasoning_step_idx)
            if step >= max_idx:
                continue
            key = (example_id, step)
            next_events = event_map.get((example_id, step + 1), set())
            b = belief_pivot.loc[key] if key in belief_pivot.index else pd.Series(dtype=object)
            a = answer_pivot.loc[key] if key in answer_pivot.index else pd.Series(dtype=object)
            action = str(pos.action_label).lower()
            chosen_wall = f"wall_{action}" if action in {"up", "down", "left", "right"} else ""
            chosen_hit = f"hit_wall_after_{action}" if action in {"up", "down", "left", "right"} else ""
            feature_row: dict[str, Any] = {
                "example_id": example_id,
                "trajectory_id": pos.trajectory_id,
                "matched_role": pos.matched_role,
                "reasoning_step_idx": step,
                "action_change": int("action_change" in next_events),
                "optimal_to_suboptimal": int(bool(next_events & {"transient_optimal_to_suboptimal", "sustained_optimal_to_suboptimal"})),
                "sustained_optimal_to_suboptimal": int("sustained_optimal_to_suboptimal" in next_events),
                "suboptimal_to_optimal": int("suboptimal_to_optimal" in next_events),
                "commitment_onset": int("commitment_onset" in next_events),
            }
            for q in STATE_BELIEFS:
                feature_row[f"error_{q}"] = int(bool(b.get(q, False))) if q in b.index else 0
            feature_row["any_wall_error"] = int(any(feature_row[f"error_{q}"] for q in WALL_BELIEFS))
            feature_row["any_current_state_error"] = int(any(feature_row[f"error_{q}"] for q in STATE_BELIEFS))
            feature_row["chosen_wall_error"] = int(bool(b.get(chosen_wall, False))) if chosen_wall in b.index else 0
            feature_row["chosen_wall_reports_blocked"] = int(str(a.get(chosen_wall, "")).lower() == "yes") if chosen_wall in a.index else 0
            feature_row["chosen_effect_error"] = int(bool(b.get(chosen_hit, False))) if chosen_hit in b.index else 0
            feature_row["chosen_effect_reports_hit"] = int(str(a.get(chosen_hit, "")).lower() == "yes") if chosen_hit in a.index else 0
            feature_row["belief_action_conflict"] = int(feature_row["chosen_wall_reports_blocked"] or feature_row["chosen_effect_reports_hit"])
            for prefix in ACTION_EFFECT_PREFIXES:
                cols = [c for c in b.index if str(c).startswith(prefix)]
                feature_row[f"any_{prefix}_error"] = int(any(bool(b.get(c, False)) for c in cols))
            rows.append(feature_row)

    feature_rows = pd.DataFrame(rows)
    feature_rows.to_csv(out / "factorization_position_rows_matched46.csv", index=False)
    features = [
        "any_current_state_error",
        "any_wall_error",
        "error_wall_left",
        "error_wall_right",
        "error_wall_up",
        "error_wall_down",
        "error_has_key",
        "error_door_open",
        "chosen_wall_error",
        "chosen_wall_reports_blocked",
        "chosen_effect_error",
        "chosen_effect_reports_hit",
        "belief_action_conflict",
        "any_hit_wall_after_error",
        "any_has_key_after_error",
        "any_door_open_after_error",
    ]
    outcomes = ["action_change", "optimal_to_suboptimal", "sustained_optimal_to_suboptimal", "suboptimal_to_optimal", "commitment_onset"]
    summary_rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        for feature in features:
            with_feature = feature_rows[feature_rows[feature] == 1]
            without_feature = feature_rows[feature_rows[feature] == 0]
            rate_with = with_feature[outcome].mean() if len(with_feature) else np.nan
            rate_without = without_feature[outcome].mean() if len(without_feature) else np.nan
            summary_rows.append(
                {
                    "Outcome": outcome,
                    "Feature": feature,
                    "n with feature": len(with_feature),
                    "Rate with": rate_with,
                    "Rate without": rate_without,
                    "Difference": rate_with - rate_without if np.isfinite(rate_with) and np.isfinite(rate_without) else np.nan,
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "factorization_feature_outcome_table.csv", index=False)
    return summary


def write_report(
    out: Path,
    *,
    commitment: pd.DataFrame,
    entropy: pd.DataFrame,
    factorization: pd.DataFrame,
) -> None:
    opt = entropy[entropy["event_family"] == "optimal_to_suboptimal"].iloc[0]
    failure_commit = commitment[commitment["group"] == "failure"].iloc[0]
    control_commit = commitment[commitment["group"] == "control"].iloc[0]
    top_factor = (
        factorization[factorization["Outcome"].eq("optimal_to_suboptimal")]
        .sort_values("Difference", ascending=False)
        .head(5)
    )
    lines = [
        "# Scaled Matched-46 Reader-Facing Outputs",
        "",
        "These figures use the GPT-OSS-20B matched 46-state sentence-prefix run. Action entropy is from a single temperature 0.7 candidate-logprob distribution over UP, DOWN, LEFT, and RIGHT, not repeated sampling.",
        "",
        "## Headline Checks",
        "",
        f"- Failure states commit at mean sentence index {failure_commit['mean_commitment_sentence_index']:.1f}; control states commit at {control_commit['mean_commitment_sentence_index']:.1f}.",
        f"- Optimal-to-suboptimal events have mean action-entropy change {opt['mean_delta_action_entropy_bits']:.3f} bits with 95% bootstrap interval [{opt['delta_action_entropy_ci_low']:.3f}, {opt['delta_action_entropy_ci_high']:.3f}].",
        "- The factorisation table reports descriptive next-prefix event rates, not causal effects.",
        "",
        "## Top Descriptive Features for Optimal-to-Suboptimal Events",
        "",
        "| Feature | n with feature | Rate with | Rate without | Difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in top_factor.iterrows():
        lines.append(
            f"| {row['Feature']} | {int(row['n with feature'])} | "
            f"{row['Rate with']:.3f} | {row['Rate without']:.3f} | {row['Difference']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `figs/commitment_timing_by_state_group.png`",
            "- `figs/action_entropy_change_decision_events.png`",
            "- `figs/belief_errors_before_after_optimality_changes.png`",
            "- `figs/belief_entropy_around_commitment_aggregate.png`",
            "- `figs/belief_entropy_around_commitment_by_family.png`",
            "- `figs/belief_entropy_around_commitment_by_question.png`",
            "- `factorization_feature_outcome_table.csv`",
        ]
    )
    (out / "SCALED_READER_FACING_SUMMARY.md").write_text("\n".join(lines) + "\n")

    captions = """# Figure Captions

## commitment_timing_by_state_group.png

Action commitment timing for failure and control states in the matched 46-state GPT-OSS-20B run. The recommendation is measured after every reasoning sentence. Commitment is the first sentence boundary where the recommended action equals the full-trace recommendation and remains stable through the rest of the trace. The left panel reports the number of reasoning sentences revealed at commitment; the right panel divides the number of reasoning characters revealed by the total number of reasoning characters in that trace. The 46 traces contain 20 to 365 reasoning sentences. Points are environment states; dark markers show group means with 95 percent bootstrap confidence intervals over states.

## action_entropy_change_decision_events.png

Change in action uncertainty when the recommendation changes or first remains stable. For each event, the plotted value is entropy after the event minus entropy at the preceding sentence boundary. Action entropy is Shannon entropy over the temperature 0.7 candidate-logprob distribution on UP, DOWN, LEFT, and RIGHT. Negative values mean the model became more confident in one action. The plot does not use activations.

## belief_errors_before_after_optimality_changes.png

Mean belief error rates around recommendation changes and changes in whether the recommended action is optimal. Position 0 is the first sentence boundary where the event occurs. Error rate is the fraction of wall, key-possession, or door-state questions whose answer disagrees with the ground-truth DoorKey state. The plot reports each belief family and an aggregate across all six current-state questions.

## belief_entropy_around_commitment_aggregate.png

Mean current-state belief uncertainty around retrospective action commitment. Position 0 is the first sentence boundary where the recommended action equals the full-trace recommendation and remains stable. Entropy is averaged over wall-left, wall-right, wall-up, wall-down, key-held, and door-open answer distributions at temperature 0.7.

## belief_entropy_around_commitment_by_family.png

Mean answer entropy around retrospective action commitment, separated into wall, key-possession, and door-state questions. This checks whether post-commitment reasoning is associated with residual uncertainty about particular parts of the environment state.

## belief_entropy_around_commitment_by_question.png

Mean belief entropy around retrospective action commitment for each current-state belief probe: wall left, wall right, wall up, wall down, key held, and door open.
"""
    (out / "FIGURE_CAPTIONS.md").write_text(captions)


def audit_directories(out: Path) -> None:
    lines = [
        "# Output Directory Audit",
        "",
        "## Recommended reader-facing location",
        "",
        "- `outputs/reader_facing/experiment1_2_gpt_oss_matched46_v1/`: scaled matched-46 figures and tables generated for sharing.",
        "",
        "## Keep as source experiment outputs",
        "",
        "- `outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1/`: source Experiment 1 matched-46 action, entropy, and activation outputs.",
        "- `outputs/experiment2_behavioral_beliefs/gpt_oss_local_sentence_matched46_v1/`: source Experiment 2 matched-46 belief outputs and predictive models.",
        "- `outputs/experiment1_activation_monitor/gpt_oss_local_environment_step_all1276_v1/` and `outputs/experiment2_behavioral_beliefs/gpt_oss_local_environment_step_all1276_v1/`: environment-step robustness outputs, not the same population as matched-46 sentence figures.",
        "- Some belief-related follow-up CSVs also exist in the Experiment 1 matched-46 folder because earlier combined analysis scripts joined action, entropy, and belief rows there. Treat them as source artifacts, not reader-facing figures.",
        "",
        "## Potentially confusing legacy or diagnostic outputs",
        "",
        "- `outputs/experiment1_activation_monitor/weisheng_8_state_*` and `outputs/experiment2_behavioral_beliefs/weisheng_8_state_*`: earlier 8-state pilot outputs; keep for provenance but do not mix with scaled matched-46 figures.",
        "- `outputs/experiment1_activation_monitor/weisheng_sentence_logprob_gate*` and `outputs/experiment2_behavioral_beliefs/weisheng_sentence_logprob_gate*`: feasibility-gate outputs; move to an archive folder if sharing the repo externally.",
        "- `outputs/reader_facing/experiment1_2_weisheng_8_state_logprob_v1/`: earlier reader-facing 8-state output; superseded for scaled matched-46 reporting.",
        "- Duplicated progress figures inside Experiment 1 and Experiment 2 source folders are acceptable as diagnostics, but the new `outputs/reader_facing/experiment1_2_gpt_oss_matched46_v1/` folder should be the shared location for scaled-run figures.",
        "",
        "## Do not delete automatically",
        "",
        "No files were deleted. The safest cleanup is to move legacy Weisheng pilot and gate folders under an explicit archive directory after confirming no open notebook references them.",
    ]
    (out / "OUTPUT_DIRECTORY_AUDIT.md").write_text("\n".join(lines) + "\n")


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reader_facing/experiment1_2_gpt_oss_matched46_v1"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figs").mkdir(parents=True, exist_ok=True)

    positions, events, beliefs = load_inputs(args.experiment1_dir, args.experiment2_dir)
    commitment = build_commitment_timing(positions, args.output_dir)
    entropy = build_entropy_decision_events(positions, events, args.output_dir)
    build_belief_error_alignment(beliefs, events, args.output_dir)
    build_belief_entropy_around_commitment(positions, beliefs, args.output_dir)
    factorization = build_factorization_table(positions, beliefs, events, args.output_dir)
    write_report(args.output_dir, commitment=commitment, entropy=entropy, factorization=factorization)
    audit_directories(args.output_dir)
    manifest = {
        "experiment1_dir": str(args.experiment1_dir),
        "experiment2_dir": str(args.experiment2_dir),
        "output_dir": str(args.output_dir),
        "n_states": int(positions["example_id"].nunique()),
        "n_trajectories": int(positions["trajectory_id"].nunique()),
        "n_sentence_prefixes": int(len(positions)),
        "n_belief_rows": int(len(beliefs)),
        "action_entropy_protocol": "temperature_0.7_candidate_logprob_distribution_over_UP_DOWN_LEFT_RIGHT",
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote scaled reader-facing outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
