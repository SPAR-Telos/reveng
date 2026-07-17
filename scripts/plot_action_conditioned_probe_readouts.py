#!/usr/bin/env python3
"""Plot existing action-conditioned consequence probe accuracy and uncertainty."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

FAMILIES = {
    "hit_wall_after": "Hit wall after\nrecommended action",
    "has_key_after": "Has key after\nrecommended action",
    "door_open_after": "Door open after\nrecommended action",
}
COLORS = ["#8CC8E8", "#1769AA", "#0B3C5D"]
GRID = "#E5EEF5"


def _family(question_id: str) -> str | None:
    for prefix in FAMILIES:
        if question_id.startswith(prefix + "_"):
            return prefix
    return None


def _bootstrap_ci(frame: pd.DataFrame, column: str, *, repeats: int, seed: int) -> tuple[float, float]:
    grouped = {key: group for key, group in frame.groupby("trajectory_id")}
    keys = np.array(sorted(grouped), dtype=object)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        sample = pd.concat([grouped[key] for key in rng.choice(keys, len(keys), replace=True)], ignore_index=True)
        estimates.append(float(sample[column].mean()))
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def build_plot(input_path: Path, output_dir: Path, *, bootstrap_repeats: int) -> None:
    frame = pd.read_csv(input_path)
    frame["family"] = frame["question_id"].map(_family)
    frame = frame[frame["family"].notna()].copy()
    frame["correct"] = ~frame["belief_is_error"].astype(bool)
    # These rows are action-conditioned on the recommended action from the same sentence.
    if "action_label" in frame.columns:
        frame["question_action"] = frame["question_id"].str.extract(r"_(up|down|left|right)$")[0].str.upper()
        recommended_only = frame[frame["question_action"] == frame["action_label"]]
        if len(recommended_only):
            frame = recommended_only.copy()

    rows = []
    for idx, (family, label) in enumerate(FAMILIES.items()):
        subset = frame[frame["family"] == family].copy()
        acc_low, acc_high = _bootstrap_ci(subset, "correct", repeats=bootstrap_repeats, seed=42 + idx)
        ent_low, ent_high = _bootstrap_ci(subset, "entropy_bits", repeats=bootstrap_repeats, seed=142 + idx)
        rows.append(
            {
                "probe_family": family,
                "label": label.replace("\n", " "),
                "rows": len(subset),
                "states": subset["example_id"].nunique(),
                "trajectories": subset["trajectory_id"].nunique(),
                "accuracy": float(subset["correct"].mean()),
                "accuracy_ci_low": acc_low,
                "accuracy_ci_high": acc_high,
                "mean_entropy_bits": float(subset["entropy_bits"].mean()),
                "entropy_ci_low": ent_low,
                "entropy_ci_high": ent_high,
            }
        )
    summary = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "action_conditioned_probe_summary.csv", index=False)

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update(
        {
            "font.family": "Arial" if "Arial" in available_fonts else "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    x = np.arange(len(summary))
    labels = [FAMILIES[row.probe_family] for row in summary.itertuples()]

    accuracy = summary["accuracy"].to_numpy()
    accuracy_error = np.vstack(
        [accuracy - summary["accuracy_ci_low"].to_numpy(), summary["accuracy_ci_high"].to_numpy() - accuracy]
    )
    axes[0].bar(x, accuracy, color=COLORS)
    axes[0].errorbar(x, accuracy, yerr=accuracy_error, fmt="none", ecolor="#243746", capsize=3)
    axes[0].axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(x, labels=labels)
    axes[0].set_ylabel("Exact-match accuracy")
    axes[0].set_xlabel("Action-conditioned consequence probe")
    axes[0].set_title("Probe Accuracy")
    axes[0].grid(axis="y", color=GRID)
    for i, row in enumerate(summary.itertuples()):
        axes[0].text(i, row.accuracy + 0.035, f"{row.accuracy:.1%}\nn={row.rows:,}", ha="center", fontsize=9)

    entropy = summary["mean_entropy_bits"].to_numpy()
    entropy_error = np.vstack(
        [entropy - summary["entropy_ci_low"].to_numpy(), summary["entropy_ci_high"].to_numpy() - entropy]
    )
    axes[1].bar(x, entropy, color=COLORS)
    axes[1].errorbar(x, entropy, yerr=entropy_error, fmt="none", ecolor="#243746", capsize=3)
    axes[1].set_ylim(0, np.log2(3))
    axes[1].set_xticks(x, labels=labels)
    axes[1].set_ylabel("Mean answer entropy (bits)")
    axes[1].set_xlabel("Action-conditioned consequence probe")
    axes[1].set_title("Probe Uncertainty")
    axes[1].grid(axis="y", color=GRID)

    fig.suptitle(
        "Action-Conditioned Consequence Readouts for the Recommended Action\n"
        f"{frame['example_id'].nunique()} states, {frame['trajectory_id'].nunique()} trajectories, {len(frame):,} readouts",
        y=1.04,
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "action_conditioned_probe_accuracy_uncertainty.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "# Figure Caption\n\n"
        "## action_conditioned_probe_accuracy_uncertainty.png\n\n"
        "Accuracy and uncertainty for the existing GPT-OSS-20B action-conditioned consequence probes in the matched 46-state sentence run. "
        "At each reasoning sentence, the probe asks about the action currently recommended by the model, not all four possible actions. "
        "Exact-match accuracy compares the highest-probability yes, no, or unknown answer with the DoorKey transition label for the real agent marker in the grid. "
        "Entropy is Shannon entropy over the same temperature 0.7 yes/no/unknown distribution. Error bars are 95 percent intervals from bootstrapping complete trajectories.\n"
    )
    (output_dir / "ACTION_CONDITIONED_PROBE_CAPTION.md").write_text(caption)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=Path("outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1/belief_rows.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hypothesis_tests/transition_activation_commitment_v1/figs"))
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()
    build_plot(args.input_path, args.output_dir, bootstrap_repeats=args.bootstrap_repeats)


if __name__ == "__main__":
    main()
