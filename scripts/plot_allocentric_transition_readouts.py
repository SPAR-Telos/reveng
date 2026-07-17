#!/usr/bin/env python3
"""Plot accuracy and uncertainty of allocentric transition-belief readouts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


PROBES = {
    "hit_wall": "Wall collision",
    "has_key": "Key held after action",
    "door_open": "Door open after action",
}
COLORS = ["#8CC8E8", "#2878B5", "#0B3C5D"]


def _probe_family(question_id: str) -> str:
    for suffix in PROBES:
        if question_id.endswith(suffix):
            return suffix
    raise ValueError(f"Unknown transition probe: {question_id}")


def _bootstrap_interval(
    frame: pd.DataFrame,
    column: str,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    groups = {key: value for key, value in frame.groupby("trajectory_id")}
    keys = np.array(sorted(groups), dtype=object)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sample = pd.concat(
            [groups[key] for key in rng.choice(keys, len(keys), replace=True)],
            ignore_index=True,
        )
        values.append(float(sample[column].mean()))
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def build_figure(input_path: Path, *, bootstrap_repeats: int) -> None:
    frame = pd.read_csv(input_path)
    frame["probe_family"] = frame["question_id"].map(_probe_family)
    frame["correct"] = ~frame["belief_is_error"].astype(bool)

    rows = []
    for index, (family, label) in enumerate(PROBES.items()):
        subset = frame[frame["probe_family"] == family].copy()
        accuracy_low, accuracy_high = _bootstrap_interval(
            subset,
            "correct",
            repeats=bootstrap_repeats,
            seed=42 + index,
        )
        entropy_low, entropy_high = _bootstrap_interval(
            subset,
            "entropy_bits",
            repeats=bootstrap_repeats,
            seed=142 + index,
        )
        rows.append(
            {
                "transition_belief": label,
                "n_readouts": len(subset),
                "n_states": subset["example_id"].nunique(),
                "n_trajectories": subset["trajectory_id"].nunique(),
                "accuracy": float(subset["correct"].mean()),
                "accuracy_ci_low": accuracy_low,
                "accuracy_ci_high": accuracy_high,
                "mean_entropy_bits": float(subset["entropy_bits"].mean()),
                "entropy_ci_low": entropy_low,
                "entropy_ci_high": entropy_high,
            }
        )
    summary = pd.DataFrame(rows)
    output_dir = input_path.parent
    summary.to_csv(output_dir / "allocentric_transition_probe_summary.csv", index=False)

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update(
        {
            "font.family": "Arial" if "Arial" in available_fonts else "DejaVu Sans",
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    x = np.arange(len(summary))
    labels = summary["transition_belief"].tolist()

    accuracy = summary["accuracy"].to_numpy()
    accuracy_error = np.vstack(
        [
            accuracy - summary["accuracy_ci_low"].to_numpy(),
            summary["accuracy_ci_high"].to_numpy() - accuracy,
        ]
    )
    axes[0].bar(x, accuracy, color=COLORS)
    axes[0].errorbar(
        x,
        accuracy,
        yerr=accuracy_error,
        fmt="none",
        ecolor="#243746",
        capsize=3,
    )
    axes[0].axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 0.7)
    axes[0].set_xticks(x, labels=labels, rotation=12, ha="right")
    axes[0].set_ylabel("Exact-match accuracy")
    axes[0].set_xlabel("Counterfactual transition question")
    axes[0].set_title("Transition-Belief Accuracy")
    axes[0].grid(axis="y", color="#E5EEF5", linewidth=0.8)

    entropy = summary["mean_entropy_bits"].to_numpy()
    entropy_error = np.vstack(
        [
            entropy - summary["entropy_ci_low"].to_numpy(),
            summary["entropy_ci_high"].to_numpy() - entropy,
        ]
    )
    axes[1].bar(x, entropy, color=COLORS)
    axes[1].errorbar(
        x,
        entropy,
        yerr=entropy_error,
        fmt="none",
        ecolor="#243746",
        capsize=3,
    )
    axes[1].set_ylim(0, np.log2(3))
    axes[1].set_xticks(x, labels=labels, rotation=12, ha="right")
    axes[1].set_ylabel("Mean answer entropy (bits)")
    axes[1].set_xlabel("Counterfactual transition question")
    axes[1].set_title("Transition-Belief Uncertainty")
    axes[1].grid(axis="y", color="#E5EEF5", linewidth=0.8)

    fig.suptitle(
        "Allocentric Transition-Belief Readouts\n"
        f"{frame['example_id'].nunique()} states, "
        f"{frame['trajectory_id'].nunique()} trajectories, "
        f"{len(frame):,} readouts",
        fontsize=14,
        y=1.04,
    )
    fig.tight_layout()
    figure_dir = output_dir / "figs"
    figure_dir.mkdir(exist_ok=True)
    fig.savefig(
        figure_dir / "allocentric_transition_probe_accuracy_uncertainty.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    caption = (
        "# Figure Caption\n\n"
        "## allocentric_transition_probe_accuracy_uncertainty.png\n\n"
        "Accuracy and uncertainty of GPT-OSS-20B behavioral readouts for "
        "counterfactual DoorKey transitions at arbitrary traversable cells in "
        "the current grid. Each reasoning position uses four balanced cases "
        "and asks whether the selected action would hit a wall, leave the "
        "agent holding the key, or leave the door open. Accuracy compares the "
        "highest-probability yes, no, or unknown answer with the DoorKey "
        "transition model. Entropy is Shannon entropy over the same temperature "
        "0.7 answer distribution. Error bars are 95 percent intervals from "
        "bootstrapping complete trajectories. The dashed line marks 0.5 "
        "accuracy; yes and no ground-truth labels are approximately balanced.\n"
    )
    (output_dir / "FIGURE_CAPTION.md").write_text(caption)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/transition_activation_commitment_v1/"
            "allocentric_readouts/allocentric_transition_belief_rows.csv"
        ),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()
    build_figure(args.input_path, bootstrap_repeats=args.bootstrap_repeats)


if __name__ == "__main__":
    main()
