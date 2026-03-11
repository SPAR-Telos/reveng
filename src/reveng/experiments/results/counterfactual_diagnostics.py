"""Diagnostics for counterfactual A_target/A_base analysis.

This script focuses on:
1) Threshold-free analysis of A_target and A_base.
2) Visual inspection of disruptive pairs where both A and G moved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


def _parse_grid_text(path: Path) -> list[list[str]]:
    lines = path.read_text().splitlines()
    rows: list[list[str]] = []
    for line in lines[1:]:
        parts = line.strip().split()
        if not parts:
            continue
        rows.append(parts[1:])
    return rows


def _find_positions(grid: list[list[str]]) -> tuple[tuple[int, int], tuple[int, int]]:
    agent = (-1, -1)
    goal = (-1, -1)
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell == "A":
                agent = (x, y)
            elif cell == "G":
                goal = (x, y)
    return agent, goal


def _cell_to_int(cell: str) -> int:
    mapping = {"#": 0, "_": 1, "A": 2, "G": 3}
    return mapping.get(cell, 1)


def _draw_grid(ax: Any, grid: list[list[str]], title: str) -> None:
    arr = np.array([[_cell_to_int(c) for c in row] for row in grid], dtype=int)
    cmap = ListedColormap(["#2f3640", "#ecf0f1", "#27ae60", "#f1c40f"])
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=3)
    h, w = arr.shape
    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=10)
    ax.grid(which="both", color="#7f8c8d", linewidth=0.4)


def _load_enriched_dataframe(manifest_path: Path, per_pair_csv: Path) -> pd.DataFrame:
    manifest = pd.read_json(manifest_path)
    per_pair = pd.read_csv(per_pair_csv)

    keep_cols = [
        "pair_id",
        "category",
        "a_target",
        "a_base",
        "action_label",
        "disruptive",
        "evaluated_steps",
        "valid_pair",
    ]
    merged = manifest.merge(per_pair[keep_cols], on=["pair_id", "category"], how="left")

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        grid_a = _parse_grid_text(Path(row["grid_a_path"]))
        grid_b = _parse_grid_text(Path(row["grid_b_path"]))
        a_pos_a, g_pos_a = _find_positions(grid_a)
        a_pos_b, g_pos_b = _find_positions(grid_b)

        a_move = abs(a_pos_a[0] - a_pos_b[0]) + abs(a_pos_a[1] - a_pos_b[1])
        g_move = abs(g_pos_a[0] - g_pos_b[0]) + abs(g_pos_a[1] - g_pos_b[1])

        out = dict(row)
        out["a_move_l1"] = a_move
        out["g_move_l1"] = g_move
        out["total_move_l1"] = a_move + g_move
        out["both_moved"] = bool(a_move > 0 and g_move > 0)
        out["delta_a"] = float(row["a_target"]) - float(row["a_base"])
        rows.append(out)

    return pd.DataFrame(rows)


def _plot_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    valid = df[df["valid_pair"] == True].copy()  # noqa: E712

    mask_disruptive = valid["disruptive"] == True  # noqa: E712
    mask_non_disruptive = valid["disruptive"] == False  # noqa: E712
    mask_unevaluable = valid["disruptive"].isna()

    ax.scatter(
        valid.loc[mask_non_disruptive, "a_base"],
        valid.loc[mask_non_disruptive, "a_target"],
        c="#1f77b4",
        label="non-disruptive",
        alpha=0.8,
    )
    ax.scatter(
        valid.loc[mask_disruptive, "a_base"],
        valid.loc[mask_disruptive, "a_target"],
        c="#d62728",
        label="disruptive",
        alpha=0.8,
    )
    ax.scatter(
        valid.loc[mask_unevaluable, "a_base"],
        valid.loc[mask_unevaluable, "a_target"],
        c="#7f8c8d",
        label="unevaluable",
        alpha=0.8,
    )

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.axhline(0.7, linestyle=":", color="#2c3e50", linewidth=1)
    ax.axhline(0.35, linestyle=":", color="#7f8c8d", linewidth=1)
    ax.axvline(0.35, linestyle=":", color="#7f8c8d", linewidth=1)

    ax.set_xlabel("A_base")
    ax.set_ylabel("A_target")
    ax.set_title("A_target vs A_base (threshold-free view)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "a_target_vs_a_base_scatter.png", dpi=200)
    plt.close(fig)


def _plot_delta_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    valid = df[df["valid_pair"] == True].copy()  # noqa: E712
    ax.hist(valid["delta_a"], bins=16, color="#2c7fb8", edgecolor="white")
    ax.axvline(0.0, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Delta A = A_target - A_base")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Delta A across pairs")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "delta_a_distribution.png", dpi=200)
    plt.close(fig)


def _plot_category_bars(df: pd.DataFrame, out_dir: Path) -> None:
    valid = df[df["valid_pair"] == True].copy()  # noqa: E712
    grouped = (
        valid.groupby("category")
        .agg(a_target_mean=("a_target", "mean"), a_base_mean=("a_base", "mean"))
        .reset_index()
    )
    grouped = grouped.sort_values("category")

    x = np.arange(len(grouped))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width / 2, grouped["a_target_mean"], width, label="A_target", color="#1f77b4")
    ax.bar(x + width / 2, grouped["a_base_mean"], width, label="A_base", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["category"], rotation=20, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Mean rate")
    ax.set_title("Per-category mean A_target and A_base")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "category_mean_a_target_a_base.png", dpi=200)
    plt.close(fig)


def _select_example_pairs(df: pd.DataFrame, n_examples: int) -> pd.DataFrame:
    valid = df[df["valid_pair"] == True].copy()  # noqa: E712

    preferred = valid[
        (valid["both_moved"] == True)  # noqa: E712
        & (valid["disruptive"] == True)  # noqa: E712
    ].copy()
    preferred = preferred.sort_values(
        by=["total_move_l1", "a_target", "a_base"], ascending=[False, True, True]
    )

    if len(preferred) >= n_examples:
        return preferred.head(n_examples)

    needed = n_examples - len(preferred)
    fallback = valid[
        (valid["both_moved"] == True)  # noqa: E712
        & (~valid["pair_id"].isin(preferred["pair_id"]))
    ].copy()
    fallback = fallback.sort_values(
        by=["disruptive", "total_move_l1", "a_target"], ascending=[False, False, True]
    )
    return pd.concat([preferred, fallback.head(needed)], ignore_index=True)


def _plot_examples(df_examples: pd.DataFrame, out_dir: Path) -> None:
    n = len(df_examples)
    fig, axes = plt.subplots(n, 2, figsize=(10, max(3, int(2.5 * n))))
    if n == 1:
        axes = np.array([axes])

    for i, (_, row) in enumerate(df_examples.iterrows()):
        grid_a = _parse_grid_text(Path(row["grid_a_path"]))
        grid_b = _parse_grid_text(Path(row["grid_b_path"]))

        _draw_grid(axes[i, 0], grid_a, f"{row['pair_id']} (A)")
        _draw_grid(axes[i, 1], grid_b, f"{row['pair_id']} (B)")

        meta = (
            f"{row['category']} | disruptive={row['disruptive']} | "
            f"A_target={row['a_target']:.3f} | A_base={row['a_base']:.3f} | "
            f"A_move={row['a_move_l1']} | G_move={row['g_move_l1']}"
        )
        axes[i, 0].set_ylabel(meta, fontsize=8, rotation=90, labelpad=25)

    fig.suptitle(
        "Disruption-focused examples (prioritized: both A and G moved)",
        fontsize=13,
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "disruption_examples_both_moved_top10.png", dpi=220)
    plt.close(fig)


def _write_summary(df: pd.DataFrame, df_examples: pd.DataFrame, out_dir: Path) -> None:
    valid = df[df["valid_pair"] == True].copy()  # noqa: E712

    both = valid[valid["both_moved"] == True]  # noqa: E712
    not_both = valid[valid["both_moved"] == False]  # noqa: E712

    lines = [
        "# Counterfactual Diagnostics (A_target / A_base)",
        "",
        "## Overall",
        f"- total_pairs: {len(df)}",
        f"- valid_pairs: {len(valid)}",
        f"- disruptive_pairs: {int((valid['disruptive'] == True).sum())}",  # noqa: E712
        f"- mean_A_target: {valid['a_target'].mean():.4f}",
        f"- mean_A_base: {valid['a_base'].mean():.4f}",
        f"- mean_delta_A: {valid['delta_a'].mean():.4f}",
        "",
        "## Movement vs Disruption",
        f"- both_moved_count: {len(both)}",
        f"- both_moved_disruptive_rate: {float((both['disruptive'] == True).mean()):.4f}",  # noqa: E712
        f"- not_both_moved_count: {len(not_both)}",
        f"- not_both_moved_disruptive_rate: {float((not_both['disruptive'] == True).mean()):.4f}",  # noqa: E712
        "",
        "## Category Means",
    ]

    category_tbl = (
        valid.groupby("category")
        .agg(
            pairs=("pair_id", "count"),
            mean_a_target=("a_target", "mean"),
            mean_a_base=("a_base", "mean"),
            disruptive_rate=("disruptive", lambda s: float((s == True).mean())),  # noqa: E712
        )
        .reset_index()
    )
    lines.append(category_tbl.to_markdown(index=False))
    lines.extend(["", "## Example Pairs (Top 10)", df_examples.to_markdown(index=False), ""])

    (out_dir / "diagnostics_summary.md").write_text("\n".join(lines))


def run(manifest_path: Path, per_pair_csv: Path, out_dir: Path, n_examples: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_enriched_dataframe(manifest_path, per_pair_csv)
    df.to_csv(out_dir / "enriched_per_pair_results.csv", index=False)

    _plot_scatter(df, out_dir)
    _plot_delta_distribution(df, out_dir)
    _plot_category_bars(df, out_dir)

    df_examples = _select_example_pairs(df, n_examples=n_examples)
    df_examples.to_csv(out_dir / "selected_disruption_examples.csv", index=False)
    _plot_examples(df_examples, out_dir)

    _write_summary(df, df_examples, out_dir)

    print(f"Wrote diagnostics to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfactual diagnostics for A_target/A_base.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/cf/pair_manifest.json"),
    )
    parser.add_argument(
        "--per-pair-csv",
        type=Path,
        default=Path("data/cf/eval_results/per_pair_results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cf/eval_results/diagnostics"),
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
    )
    args = parser.parse_args()
    run(
        manifest_path=args.manifest_path,
        per_pair_csv=args.per_pair_csv,
        out_dir=args.output_dir,
        n_examples=args.num_examples,
    )


if __name__ == "__main__":
    main()
