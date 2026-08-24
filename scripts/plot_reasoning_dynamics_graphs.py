#!/usr/bin/env python3
"""Render deterministic views of the semantic and BEAST operator graphs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

from reveng.experiments.semantic_regime_prediction import SEMANTIC_LABELS


DEFAULT_ROOT = Path("outputs/hypothesis_tests/reasoning_dynamics_graph_v1")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def circular_positions(labels: list[str]) -> dict[str, np.ndarray]:
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(labels), endpoint=False)
    return {label: np.array([np.cos(angle), np.sin(angle)]) for label, angle in zip(labels, angles, strict=True)}


def draw_arrow(ax, start, end, *, width, color, alpha=0.8, self_loop=False):
    if self_loop:
        patch = FancyArrowPatch(
            start + np.array([0.02, 0.09]), start + np.array([0.09, 0.02]),
            connectionstyle="arc3,rad=2.2", arrowstyle="-|>", mutation_scale=10,
            linewidth=width, color=color, alpha=alpha,
        )
    else:
        patch = FancyArrowPatch(
            start, end, connectionstyle="arc3,rad=0.12", arrowstyle="-|>",
            mutation_scale=10, shrinkA=24, shrinkB=24, linewidth=width,
            color=color, alpha=alpha,
        )
    ax.add_patch(patch)


def semantic_graph(root: Path, output: Path) -> None:
    edges = pd.read_csv(root / "semantic_process_graph_edges.csv")
    labels = list(SEMANTIC_LABELS)
    positions = circular_positions(labels)
    fig, ax = plt.subplots(figsize=(10, 9))
    for edge in edges.itertuples(index=False):
        source, target = edge.path.split(" -> ")
        draw_arrow(
            ax, positions[source], positions[target],
            width=0.8 + 1.8 * np.log2(max(edge.mean_enrichment_ratio, 1.0)),
            color="#315b7d", self_loop=source == target,
        )
    connected = set(" -> ".join(edges.path).split(" -> "))
    for label in labels:
        x, y = positions[label]
        color = "#d9e9f2" if label in connected else "#eeeeee"
        ax.scatter([x], [y], s=1700, color=color, edgecolor="#263746", linewidth=1.3, zorder=3)
        ax.text(x, y, label.replace("_", "\n"), ha="center", va="center", fontsize=8.5, zorder=4)
    ax.text(
        0, -1.35,
        "Edges are enriched at BH q < .05 in both annotation runs; width encodes mean enrichment ratio.",
        ha="center", fontsize=9,
    )
    ax.set_title("Replicated semantic process graph", fontsize=15)
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.35); ax.axis("off")
    fig.tight_layout(); fig.savefig(output, dpi=220, bbox_inches="tight"); plt.close(fig)


def operator_name(row) -> str:
    switch = row.argmax_change_rate >= 0.5
    if switch and row.entropy_delta_bits < -0.2:
        return "switch +\nsharpen"
    if switch and row.entropy_delta_bits > 0.2:
        return "switch +\nflatten"
    if switch:
        return "large /\nrebalanced switch"
    if row.entropy_delta_bits < 0:
        return "no switch +\nsharpen"
    return "no switch +\nflatten"


def operator_graph(root: Path, output: Path) -> None:
    nodes = pd.read_csv(root / "beast_operator_cluster_summary.csv")
    edges = pd.read_csv(root / "beast_operator_graph_edges.csv")
    labels = nodes.operator_cluster.tolist()
    positions = circular_positions(labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    maximum = max(edges.observed_weight.max(), 1)
    for edge in edges.itertuples(index=False):
        source, target = edge.path.split(" -> ")
        draw_arrow(
            ax, positions[source], positions[target],
            width=0.25 + 2.5 * edge.observed_weight / maximum,
            color="#9a9a9a", alpha=0.42, self_loop=source == target,
        )
    limit = max(abs(nodes.entropy_delta_bits).max(), 1e-6)
    cmap = plt.get_cmap("coolwarm")
    for row in nodes.itertuples(index=False):
        x, y = positions[row.operator_cluster]
        color = cmap((row.entropy_delta_bits / limit + 1) / 2)
        ax.scatter([x], [y], s=2100, color=color, edgecolor="#333333", linewidth=1.3, zorder=3)
        ax.text(x, y + 0.04, operator_name(row), ha="center", va="center", fontsize=8.5, zorder=4)
        ax.text(x, y - 0.18, f"n={row.n_events}", ha="center", fontsize=8, zorder=4)
    ax.text(0, -1.34, "Gray edges show observed transitions; none survives BH correction.", ha="center", fontsize=9)
    ax.set_title("BEAST change-operator graph (exploratory)", fontsize=15)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.42, 1.3); ax.axis("off")
    fig.tight_layout(); fig.savefig(output, dpi=220, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    args = arguments()
    output = args.output_dir or args.input_dir / "figs"
    output.mkdir(parents=True, exist_ok=True)
    semantic_graph(args.input_dir, output / "semantic_process_graph.png")
    operator_graph(args.input_dir, output / "beast_operator_graph.png")
    (args.input_dir / "FIGURE_GUIDE.md").write_text(
        "# Graph figures\n\n"
        "- `figs/semantic_process_graph.png`: the confirmatory, replicated type-level semantic graph.\n"
        "- `figs/beast_operator_graph.png`: the exploratory BEAST operator graph; no edge is statistically enriched.\n"
    )


if __name__ == "__main__":
    main()
