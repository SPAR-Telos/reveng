"""Compare step-level activation geometry across prefix-action outcome classes."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt


METRICS = (
    ("aligned_change", "Mean aligned change"),
    ("adjacent_step_cosine", "Mean adjacent-step cosine"),
    ("optimality_anchor_cosine", "Mean optimal-action anchor cosine"),
    ("update_norm", "Mean update norm"),
)
LAYERS = (8, 15, 23)
CLASS_ORDER = (
    "sustained optimal-to-suboptimal recommendation transition",
    "mixed/transient suboptimal recommendations",
    "all valid prefix-elicited actions planner-optimal",
)
CLASS_LABELS = {
    CLASS_ORDER[0]: "Sustained\ntransition",
    CLASS_ORDER[1]: "Transient\nsuboptimality",
    CLASS_ORDER[2]: "All actions\noptimal",
}
CLASS_COLORS = {
    CLASS_ORDER[0]: "#1769AA",
    CLASS_ORDER[1]: "#67A3CC",
    CLASS_ORDER[2]: "#B9D7EA",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_class_geometry_figure(
    *,
    run_dir: str | Path = "data/behavioral_probes/step_reasoning_drift_balanced_v1",
) -> None:
    """Write per-state summaries, class summaries, and a comparison figure."""
    run_dir = Path(run_dir)
    geometry_rows = _read_csv(run_dir / "geometry_rows.csv")
    outcome_rows = _read_csv(run_dir / "state_outcome_accounting.csv")
    outcomes = {row["example_id"]: row for row in outcome_rows}

    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in geometry_rows:
        if row["representation_kind"] != "mean_pool":
            continue
        if int(row["reasoning_step_idx"]) == 0:
            continue
        key = (row["example_id"], int(row["layer"]))
        for metric, _ in METRICS:
            value = row.get(metric, "")
            if value not in {"", None}:
                grouped[key][metric].append(float(value))

    state_rows: list[dict[str, Any]] = []
    for (example_id, layer), metric_values in sorted(grouped.items()):
        outcome = outcomes[example_id]
        state_rows.append(
            {
                "example_id": example_id,
                "state_group": outcome["state_group"],
                "failure_category": outcome["failure_category"],
                "outcome_class": outcome["outcome_class"],
                "layer": layer,
                "representation_kind": "mean_pool",
                **{
                    f"mean_{metric}": mean(metric_values[metric])
                    for metric, _ in METRICS
                },
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for layer in LAYERS:
        for outcome_class in CLASS_ORDER:
            selected = [
                row
                for row in state_rows
                if row["layer"] == layer and row["outcome_class"] == outcome_class
            ]
            for metric, _ in METRICS:
                values = [float(row[f"mean_{metric}"]) for row in selected]
                summary_rows.append(
                    {
                        "layer": layer,
                        "representation_kind": "mean_pool",
                        "outcome_class": outcome_class,
                        "metric": metric,
                        "n_states": len(values),
                        "mean": mean(values),
                        "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
                        "minimum": min(values),
                        "maximum": max(values),
                    }
                )

    _write_csv(run_dir / "state_class_geometry_rows.csv", state_rows)
    _write_csv(run_dir / "class_geometry_summary.csv", summary_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(
        len(METRICS),
        len(LAYERS),
        figsize=(10.7, 10.6),
        sharex=True,
    )
    fig.patch.set_facecolor("white")
    rng = random.Random(20260606)

    for metric_idx, (metric, metric_label) in enumerate(METRICS):
        for layer_idx, layer in enumerate(LAYERS):
            ax = axes[metric_idx][layer_idx]
            for class_idx, outcome_class in enumerate(CLASS_ORDER):
                values = [
                    float(row[f"mean_{metric}"])
                    for row in state_rows
                    if row["layer"] == layer and row["outcome_class"] == outcome_class
                ]
                jitter = [rng.uniform(-0.11, 0.11) for _ in values]
                ax.scatter(
                    [class_idx + value for value in jitter],
                    values,
                    s=27,
                    color=CLASS_COLORS[outcome_class],
                    edgecolor="white",
                    linewidth=0.6,
                    alpha=0.9,
                    zorder=3,
                )
                class_mean = mean(values)
                ax.plot(
                    [class_idx - 0.17, class_idx + 0.17],
                    [class_mean, class_mean],
                    color="#12344D",
                    linewidth=2.0,
                    zorder=4,
                )
            if metric_idx == 0:
                ax.set_title(f"Layer {layer}", pad=8)
            if layer_idx == 0:
                ax.set_ylabel(metric_label)
            ax.set_xticks(range(len(CLASS_ORDER)))
            ax.set_xticklabels([CLASS_LABELS[value] for value in CLASS_ORDER])
            ax.grid(axis="y", color="#DCEAF4", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#A9C3D5")

    fig.suptitle(
        "Mean-pooled activation geometry by prefix-action outcome",
        fontsize=14,
        fontweight="bold",
        color="#12344D",
        y=0.995,
    )
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    footer = (
        "Corrected activation artifact: adjacent reasoning-step token spans are disjoint."
        if manifest.get("activation_spans_valid") is True
        else "Preliminary: balanced v1 token spans overlap at 79.0% of chunk boundaries."
    )
    fig.text(
        0.5,
        0.012,
        "Dots are state-level means; horizontal lines are class means. " + footer,
        ha="center",
        fontsize=8.5,
        color="#8B3A3A",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.975), h_pad=1.5, w_pad=1.1)
    output_path = run_dir / "figs" / "activation_metrics_by_outcome_class.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    build_class_geometry_figure()
