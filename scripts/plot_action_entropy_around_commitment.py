#!/usr/bin/env python3
"""Plot action entropy around retrospective commitment boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


BLUE = "#1769aa"
LIGHT_BLUE = "#9bd0f5"
DARK_BLUE = "#0b3c5d"
GRID = "#e5eef5"
TEXT = "#172033"


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _bootstrap_ci(values: list[float], *, seed: int = 42, repeats: int = 2000) -> tuple[float, float]:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(clean) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(repeats, len(clean)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _safe_json_sum(value: Any) -> float:
    try:
        parsed = json.loads(value)
    except Exception:
        return float("nan")
    if not isinstance(parsed, dict):
        return float("nan")
    try:
        return float(sum(float(v) for v in parsed.values()))
    except Exception:
        return float("nan")


def _prepare_aligned_rows(positions: pd.DataFrame, *, window: int) -> pd.DataFrame:
    positions = positions.copy()
    positions = positions.sort_values(["example_id", "position_index"]).copy()
    positions["sentence_char_length"] = (
        positions.groupby("example_id")["analysis_char_end"]
        .diff()
        .fillna(positions["analysis_char_end"])
        .clip(lower=0)
    )
    positions["commitment_onset_bool"] = _as_bool(positions["commitment_onset"])
    positions["action_committed_bool"] = _as_bool(positions["action_committed"])
    commitments = positions[positions["commitment_onset_bool"]]
    n_commitments = len(commitments)
    n_states = positions["example_id"].nunique()
    if n_commitments != n_states:
        raise ValueError(f"Expected one commitment boundary per state, found {n_commitments} for {n_states} states")

    commitment_index = commitments.set_index("example_id")["position_index"].astype(int).to_dict()
    positions["commitment_position_index"] = positions["example_id"].map(commitment_index)
    positions["relative_commitment_offset"] = (
        positions["position_index"].astype(int) - positions["commitment_position_index"].astype(int)
    )
    aligned = positions[
        positions["relative_commitment_offset"].between(-window, window)
    ].copy()
    aligned["action_probability_sum"] = aligned["action_probabilities_json"].map(_safe_json_sum)
    keep = [
        "example_id",
        "trajectory_id",
        "matched_role",
        "trajectory_class",
        "primary_step_failure_mode",
        "position_index",
        "reasoning_step_idx",
        "reasoning_progress",
        "relative_commitment_offset",
        "action_label",
        "final_full_trace_action",
        "action_entropy_bits",
        "action_confidence",
        "state_belief_entropy_bits",
        "sentence_char_length",
        "action_probability_sum",
        "action_committed_bool",
        "commitment_onset_bool",
    ]
    return aligned[keep].sort_values(["example_id", "relative_commitment_offset"])


def _grouped_offset_summary(aligned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [("all", aligned)]
    if set(aligned["matched_role"].dropna()) >= {"failure", "control"}:
        groups.extend([
            ("failure", aligned[aligned["matched_role"] == "failure"]),
            ("control", aligned[aligned["matched_role"] == "control"]),
        ])
    for group_name, group in groups:
        for offset, subset in group.groupby("relative_commitment_offset"):
            rows.append(
                {
                    "group": group_name,
                    "relative_commitment_offset": int(offset),
                    "n_states": subset["example_id"].nunique(),
                    "n_prefixes": len(subset),
                    "median_action_entropy_bits": subset["action_entropy_bits"].median(),
                    "q25_action_entropy_bits": subset["action_entropy_bits"].quantile(0.25),
                    "q75_action_entropy_bits": subset["action_entropy_bits"].quantile(0.75),
                    "median_action_confidence": subset["action_confidence"].median(),
                    "q25_action_confidence": subset["action_confidence"].quantile(0.25),
                    "q75_action_confidence": subset["action_confidence"].quantile(0.75),
                    "median_state_belief_entropy_bits": subset["state_belief_entropy_bits"].median(),
                    "mean_sentence_char_length": subset["sentence_char_length"].mean(),
                    "mean_action_probability_sum": subset["action_probability_sum"].mean(),
                }
            )
    return pd.DataFrame(rows)


def _state_window_values(aligned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for example_id, group in aligned.groupby("example_id"):
        group = group.copy()
        pre = group[group["relative_commitment_offset"].between(-5, -1)]
        commit = group[group["relative_commitment_offset"] == 0]
        post = group[group["relative_commitment_offset"].between(1, 5)]
        if commit.empty:
            continue
        row = {
            "example_id": example_id,
            "trajectory_id": commit["trajectory_id"].iloc[0],
            "matched_role": commit["matched_role"].iloc[0],
            "trajectory_class": commit["trajectory_class"].iloc[0],
            "pre_action_entropy_mean": pre["action_entropy_bits"].mean(),
            "commitment_action_entropy": commit["action_entropy_bits"].iloc[0],
            "post_action_entropy_mean": post["action_entropy_bits"].mean(),
            "pre_action_confidence_mean": pre["action_confidence"].mean(),
            "commitment_action_confidence": commit["action_confidence"].iloc[0],
            "post_action_confidence_mean": post["action_confidence"].mean(),
            "pre_state_belief_entropy_mean": pre["state_belief_entropy_bits"].mean(),
            "commitment_state_belief_entropy": commit["state_belief_entropy_bits"].iloc[0],
            "post_state_belief_entropy_mean": post["state_belief_entropy_bits"].mean(),
            "pre_low_action_entropy_rate": float((pre["action_entropy_bits"] <= 0.1).mean()) if len(pre) else np.nan,
            "n_pre_prefixes": len(pre),
            "n_post_prefixes": len(post),
        }
        row["delta_commitment_minus_pre_action_entropy"] = (
            row["commitment_action_entropy"] - row["pre_action_entropy_mean"]
        )
        row["delta_post_minus_pre_action_entropy"] = (
            row["post_action_entropy_mean"] - row["pre_action_entropy_mean"]
        )
        row["delta_post_minus_pre_action_confidence"] = (
            row["post_action_confidence_mean"] - row["pre_action_confidence_mean"]
        )
        row["delta_post_minus_pre_state_belief_entropy"] = (
            row["post_state_belief_entropy_mean"] - row["pre_state_belief_entropy_mean"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _window_summary(state_values: pd.DataFrame) -> pd.DataFrame:
    measures = [
        "pre_action_entropy_mean",
        "commitment_action_entropy",
        "post_action_entropy_mean",
        "delta_commitment_minus_pre_action_entropy",
        "delta_post_minus_pre_action_entropy",
        "pre_action_confidence_mean",
        "commitment_action_confidence",
        "post_action_confidence_mean",
        "delta_post_minus_pre_action_confidence",
        "pre_state_belief_entropy_mean",
        "commitment_state_belief_entropy",
        "post_state_belief_entropy_mean",
        "delta_post_minus_pre_state_belief_entropy",
        "pre_low_action_entropy_rate",
    ]
    rows: list[dict[str, Any]] = []
    groups = [("all", state_values)]
    if set(state_values["matched_role"].dropna()) >= {"failure", "control"}:
        groups.extend([
            ("failure", state_values[state_values["matched_role"] == "failure"]),
            ("control", state_values[state_values["matched_role"] == "control"]),
        ])
    for group_name, group in groups:
        for idx, measure in enumerate(measures):
            values = pd.to_numeric(group[measure], errors="coerce").dropna().to_numpy(dtype=float)
            low, high = _bootstrap_ci(values.tolist(), seed=42 + idx)
            rows.append(
                {
                    "group": group_name,
                    "measure": measure,
                    "n_states_with_value": len(values),
                    "mean": float(np.mean(values)) if len(values) else np.nan,
                    "median": float(np.median(values)) if len(values) else np.nan,
                    "bootstrap_mean_ci_low": low,
                    "bootstrap_mean_ci_high": high,
                }
            )
    rows.extend(
        [
            {
                "group": "all",
                "measure": "tied_optimal_action_split",
                "n_states_with_value": 0,
                "mean": np.nan,
                "median": np.nan,
                "bootstrap_mean_ci_low": np.nan,
                "bootstrap_mean_ci_high": np.nan,
                "note": "Not computed: position_rows.csv does not contain optimal_actions_json.",
            },
            {
                "group": "all",
                "measure": "raw_candidate_probability_mass",
                "n_states_with_value": 0,
                "mean": np.nan,
                "median": np.nan,
                "bootstrap_mean_ci_low": np.nan,
                "bootstrap_mean_ci_high": np.nan,
                "note": "Not computed: action_probabilities_json is normalized over UP, DOWN, LEFT, and RIGHT.",
            },
        ]
    )
    return pd.DataFrame(rows)


def _plot(offset_summary: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
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
    colors = {"all": DARK_BLUE, "failure": BLUE, "control": LIGHT_BLUE}
    labels = {"all": "All states", "failure": "Failure states", "control": "Control states"}
    all_rows = offset_summary[offset_summary["group"] == "all"]
    mean_chars = float(all_rows["mean_sentence_char_length"].mean()) if len(all_rows) else float("nan")
    max_abs_offset = int(max(abs(all_rows["relative_commitment_offset"].min()), abs(all_rows["relative_commitment_offset"].max()))) if len(all_rows) else 0
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True)
    for group_name in ["all", "failure", "control"]:
        subset = offset_summary[offset_summary["group"] == group_name].sort_values("relative_commitment_offset")
        if subset.empty:
            continue
        x = subset["relative_commitment_offset"].to_numpy(dtype=float)
        color = colors[group_name]
        axes[0].plot(
            x,
            subset["median_action_entropy_bits"].to_numpy(dtype=float),
            marker="o",
            markersize=3,
            linewidth=1.8,
            color=color,
            label=f"{labels[group_name]} (max n={int(subset['n_states'].max())})",
        )
        axes[0].fill_between(
            x,
            subset["q25_action_entropy_bits"].to_numpy(dtype=float),
            subset["q75_action_entropy_bits"].to_numpy(dtype=float),
            color=color,
            alpha=0.13,
            linewidth=0,
        )
        axes[1].plot(
            x,
            subset["median_action_confidence"].to_numpy(dtype=float),
            marker="o",
            markersize=3,
            linewidth=1.8,
            color=color,
            label=f"{labels[group_name]} (max n={int(subset['n_states'].max())})",
        )
        axes[1].fill_between(
            x,
            subset["q25_action_confidence"].to_numpy(dtype=float),
            subset["q75_action_confidence"].to_numpy(dtype=float),
            color=color,
            alpha=0.13,
            linewidth=0,
        )
    for ax in axes:
        ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
        ax.grid(color=GRID, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Action entropy over UP, DOWN,\nLEFT, and RIGHT (bits)")
    axes[0].set_title(
        f"Action uncertainty around commitment\n"
        f"Window = +/-{max_abs_offset} sentences; mean sentence length in window = {mean_chars:.1f} characters"
    )
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].set_ylabel("Probability assigned to\nthe recommended action")
    axes[1].set_xlabel("Sentence offset from commitment")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _append_caption(caption_path: Path) -> None:
    caption = (
        "\n## action_entropy_around_commitment.png\n\n"
        "Action uncertainty and action confidence around retrospective commitment in the matched 46-state "
        "sentence-prefix run. Offset 0 is the first reasoning prefix where the recommended action equals the "
        "full-trace recommendation and remains unchanged for the rest of the trace. Action entropy is Shannon "
        "entropy over the temperature 0.7 logprob distribution on UP, DOWN, LEFT, and RIGHT. Action confidence "
        "is the probability assigned to the recommended action. Lines show medians across states and bands show "
        "interquartile ranges; prefixes outside the available trace window are excluded. The plot uses a "
        "+/-10 sentence window; in this run, sentences in that window average about 43 characters.\n"
    )
    existing = caption_path.read_text() if caption_path.exists() else "# Figure Captions\n"
    if "## action_entropy_around_commitment.png" in existing:
        before = existing.split("## action_entropy_around_commitment.png")[0].rstrip()
        caption_path.write_text(before + "\n" + caption)
    else:
        caption_path.write_text(existing.rstrip() + "\n" + caption)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment1-dir",
        type=Path,
        default=Path("outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1"),
    )
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    root = args.experiment1_dir
    positions = pd.read_csv(root / "position_rows.csv")
    required = {
        "example_id",
        "position_index",
        "analysis_char_end",
        "matched_role",
        "commitment_onset",
        "action_committed",
        "action_entropy_bits",
        "action_confidence",
        "state_belief_entropy_bits",
        "action_probabilities_json",
    }
    missing = sorted(required - set(positions.columns))
    if missing:
        raise ValueError(f"Missing required columns in position_rows.csv: {missing}")

    aligned = _prepare_aligned_rows(positions, window=args.window)
    offset_summary = _grouped_offset_summary(aligned)
    state_values = _state_window_values(aligned)
    window_summary = _window_summary(state_values)

    aligned.to_csv(root / "action_entropy_commitment_aligned_rows.csv", index=False)
    offset_summary.to_csv(root / "action_entropy_commitment_offset_counts.csv", index=False)
    state_values.to_csv(root / "action_entropy_commitment_state_windows.csv", index=False)
    window_summary.to_csv(root / "action_entropy_commitment_summary.csv", index=False)
    _plot(offset_summary, root / "figs" / "action_entropy_around_commitment.png")
    _append_caption(root / "FIGURE_CAPTIONS.md")

    print(f"Wrote {root / 'figs' / 'action_entropy_around_commitment.png'}")
    print(f"Wrote {root / 'action_entropy_commitment_summary.csv'}")


if __name__ == "__main__":
    main()
