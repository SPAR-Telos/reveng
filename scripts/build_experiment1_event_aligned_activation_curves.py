"""Build clearer event-aligned activation metric curves for Experiment 1."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

METRICS = (
    "aligned_change",
    "update_norm",
    "adjacent_step_cosine",
    "optimality_anchor_cosine",
)
METRIC_LABELS = {
    "aligned_change": "Update direction compared with\nfirst-to-last activation direction",
    "update_norm": "Magnitude of activation change\nfrom the previous chunk",
    "adjacent_step_cosine": "Cosine similarity to\nthe previous chunk",
    "optimality_anchor_cosine": "Cosine similarity to the mean of\nalways-optimal prefixes",
}
EVENT_LABELS = {
    "action_change": "Action change",
    "commitment_onset": "Commitment onset",
    "optimality_loss": "Optimality loss",
    "optimality_recovery": "Optimality recovery",
    "sustained_optimality_loss": "Sustained loss",
}
PALETTE = {
    "action_change": "#4f8fc7",
    "commitment_onset": "#82b9df",
    "optimality_loss": "#0b4f8a",
    "optimality_recovery": "#b6d7ee",
    "sustained_optimality_loss": "#15324b",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except Exception:
        return None


def as_int(value: Any) -> int | None:
    try:
        if value in {None, ""}:
            return None
        return int(float(value))
    except Exception:
        return None


def build_curve_rows(root: Path, *, layer: int, representation_kind: str, window: int) -> list[dict[str, Any]]:
    geometry_path = root / "geometry_rows.csv"
    if not geometry_path.exists():
        from reveng.experiments.experiment1_activation_monitor import (
            _load_geometry_from_activations,
            _read_csv,
            _write_csv,
        )
        geometry_rows = _load_geometry_from_activations(
            _read_csv(root / "usable_activation_rows.csv"),
            _read_csv(root / "trajectory_commitment_summary.csv"),
        )
        _write_csv(geometry_path, geometry_rows)
    events = read_csv(root / "event_rows.csv")
    geometry_rows = [
        row for row in read_csv(geometry_path)
        if as_int(row.get("layer")) == layer and row.get("representation_kind") == representation_kind
    ]
    by_example: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in geometry_rows:
        by_example[row["example_id"]].append(row)
    # Z-score within each trajectory to put the four metrics on comparable axes.
    z_lookup: dict[tuple[str, int, str], float] = {}
    for example_id, rows in by_example.items():
        for metric in METRICS:
            vals = [as_float(row.get(metric)) for row in rows]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            mu = mean(vals)
            sigma = pstdev(vals) or 1.0
            for row in rows:
                step = as_int(row.get("reasoning_step_idx"))
                val = as_float(row.get(metric))
                if step is not None and val is not None:
                    z_lookup[(example_id, step, metric)] = (val - mu) / sigma
    curve_rows: list[dict[str, Any]] = []
    for event in events:
        example_id = event["example_id"]
        event_step = as_int(event.get("next_reasoning_step_idx"))
        if event_step is None:
            continue
        for offset in range(-window, window + 1):
            step = event_step + offset
            for metric in METRICS:
                value = z_lookup.get((example_id, step, metric))
                if value is None:
                    continue
                curve_rows.append(
                    {
                        "event_id": event["event_id"],
                        "example_id": example_id,
                        "event_type": event["event_type"],
                        "event_step": event_step,
                        "offset": offset,
                        "layer": layer,
                        "representation_kind": representation_kind,
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "z_value": value,
                    }
                )
    return curve_rows


def plot_curves(root: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.4), sharex=True)
    event_types = [event for event in EVENT_LABELS if any(row["event_type"] == event for row in rows)]
    event_counts = {
        event: len({row["event_id"] for row in rows if row["event_type"] == event})
        for event in event_types
    }
    for ax, metric in zip(axes.ravel(), METRICS):
        for event_type in event_types:
            by_offset: dict[int, list[float]] = defaultdict(list)
            for row in rows:
                if row["metric"] == metric and row["event_type"] == event_type:
                    by_offset[int(row["offset"])].append(float(row["z_value"]))
            xs = sorted(by_offset)
            if not xs:
                continue
            ax.plot(
                xs,
                [mean(by_offset[x]) for x in xs],
                marker="o",
                linewidth=1.8,
                color=PALETTE.get(event_type, "#1f5f9f"),
                label=f"{EVENT_LABELS.get(event_type, event_type.replace('_', ' '))} (n={event_counts[event_type]})",
            )
        ax.axvline(0, color="#15324b", linewidth=1, linestyle="--")
        ax.axhline(0, color="#64748b", linewidth=1)
        ax.grid(axis="y", color="#d7e3f0", linewidth=0.8)
        ax.set_title(METRIC_LABELS[metric])
    fig.supxlabel("Reasoning-chunk offset (0 = first prefix showing the new action or optimality status)")
    fig.supylabel("Mean standardized activation metric\n(0 = that state's average)")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Activation Metrics Before and After Action-Decision Changes", y=1.04, fontsize=11)
    fig.tight_layout()
    out = root / "figs" / "event_aligned_activation_metric_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="outputs/experiment1_activation_monitor/weisheng_8_state_canonical_experiment1")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--representation-kind", default="chunk_sampled_token_mean")
    parser.add_argument("--window", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.run_dir)
    rows = build_curve_rows(root, layer=args.layer, representation_kind=args.representation_kind, window=args.window)
    write_csv(root / "event_aligned_activation_curve_rows.csv", rows)
    plot_curves(root, rows)


if __name__ == "__main__":
    main()
