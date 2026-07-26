"""Reusable attention-window comparisons for sentence-level events."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open


def window_indices(center: int, sentence_count: int, width: int) -> list[int]:
    return [
        index
        for index in range(center - width, center + width + 1)
        if 0 <= index < sentence_count
    ]


def choose_control_center(
    center: int,
    sentence_count: int,
    width: int,
    forbidden_centers: set[int],
) -> int | None:
    candidates = [
        index
        for index in range(width, sentence_count - width)
        if abs(index - center) > 2 * width
        and not any(abs(index - other) <= width for other in forbidden_centers)
    ]
    if not candidates:
        return None
    denominator = max(1, sentence_count - 1)
    target_fraction = center / denominator
    return min(
        candidates,
        key=lambda index: (
            abs(index / denominator - target_fraction),
            abs(index - center),
        ),
    )


def load_attention_tensor(record: dict[str, Any], layer: int, metric: str) -> torch.Tensor:
    key = f"layer_{int(layer)}.final_action_to_sentence_{metric}"
    with safe_open(record["shard_path"], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).float()


def compare_attention_at_boundaries(
    change_points: pd.DataFrame,
    *,
    attention_index_path: Path,
    layers: tuple[int, ...] = (15, 23),
    width: int = 3,
    metric: str = "mass",
) -> pd.DataFrame:
    """Compare event-window attention with same-state nonoverlapping controls."""
    index = pd.read_csv(attention_index_path)
    records = {str(row.trace_id): row._asdict() for row in index.itertuples()}
    boundaries = change_points.copy()
    boundaries["center_sentence_index"] = (
        boundaries["boundary_reasoning_step_idx"].astype(int) - 1
    )
    forbidden: dict[str, set[int]] = defaultdict(set)
    for row in boundaries.itertuples():
        forbidden[str(row.example_id)].add(int(row.center_sentence_index))

    cache: dict[tuple[str, int], torch.Tensor] = {}
    output: list[dict[str, Any]] = []
    for boundary in boundaries.itertuples():
        trace_id = str(boundary.example_id)
        record = records.get(trace_id)
        if record is None:
            continue
        sentence_count = int(record["sentence_count"])
        center = int(boundary.center_sentence_index)
        if center < width or center >= sentence_count - width:
            continue
        control = choose_control_center(
            center,
            sentence_count,
            width,
            forbidden[trace_id],
        )
        if control is None:
            continue
        event_window = window_indices(center, sentence_count, width)
        control_window = window_indices(control, sentence_count, width)
        for layer in layers:
            key = (trace_id, layer)
            if key not in cache:
                cache[key] = load_attention_tensor(record, layer, metric)
            attention = cache[key]
            if not bool(torch.count_nonzero(attention)):
                continue
            event_by_head = attention[:, event_window].sum(dim=1)
            control_by_head = attention[:, control_window].sum(dim=1)
            output.append(
                {
                    "example_id": trace_id,
                    "trajectory_id": boundary.trajectory_id,
                    "matched_role": boundary.matched_role,
                    "boundary_type": boundary.boundary_type,
                    "boundary_position": int(boundary.boundary_position),
                    "layer": layer,
                    "event_window_n_sentences": len(event_window),
                    "control_position": control + 1,
                    "event_attention_mean_across_heads": float(event_by_head.mean()),
                    "control_attention_mean_across_heads": float(control_by_head.mean()),
                    "event_minus_control_attention": float(
                        (event_by_head - control_by_head).mean()
                    ),
                    "fraction_heads_positive": float(
                        ((event_by_head - control_by_head) > 0).float().mean()
                    ),
                }
            )
    return pd.DataFrame(output)


def _trajectory_bootstrap_interval(
    values: dict[str, float],
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    trajectories = np.asarray(sorted(values), dtype=object)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        sampled = rng.choice(trajectories, size=len(trajectories), replace=True)
        estimates.append(float(np.mean([values[trajectory] for trajectory in sampled])))
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def summarize_attention_rows(
    rows: pd.DataFrame,
    *,
    bootstrap_repeats: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    if rows.empty:
        return pd.DataFrame()
    for boundary_type in sorted(rows["boundary_type"].unique()):
        boundary_rows = rows[rows["boundary_type"] == boundary_type]
        for layer in sorted(boundary_rows["layer"].astype(int).unique()):
            layer_rows = boundary_rows[boundary_rows["layer"].astype(int) == layer]
            for matched_role in ("all", "failure", "control"):
                group = (
                    layer_rows
                    if matched_role == "all"
                    else layer_rows[layer_rows["matched_role"] == matched_role]
                )
                if group.empty:
                    continue
                by_trajectory = {
                    str(trajectory_id): float(
                        trajectory_group[
                            "event_minus_control_attention"
                        ].astype(float).mean()
                    )
                    for trajectory_id, trajectory_group in group.groupby(
                        "trajectory_id"
                    )
                }
                low, high = _trajectory_bootstrap_interval(
                    by_trajectory,
                    repeats=bootstrap_repeats,
                    seed=seed + int(layer),
                )
                values = list(by_trajectory.values())
                output.append(
                    {
                        "boundary_type": boundary_type,
                        "layer": int(layer),
                        "matched_role": matched_role,
                        "n_states": group["example_id"].nunique(),
                        "n_trajectories": group["trajectory_id"].nunique(),
                        "mean_event_attention": group[
                            "event_attention_mean_across_heads"
                        ].mean(),
                        "mean_control_attention": group[
                            "control_attention_mean_across_heads"
                        ].mean(),
                        "mean_event_minus_control_attention": mean(values),
                        "event_minus_control_ci_low": low,
                        "event_minus_control_ci_high": high,
                        "fraction_trajectories_positive": mean(
                            1.0 if value > 0 else 0.0 for value in values
                        ),
                    }
                )
    return pd.DataFrame(output)
