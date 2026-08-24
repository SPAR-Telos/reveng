#!/usr/bin/env python3
"""Run a matched continuous-trajectory topology test on stored activations."""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
from pathlib import Path

import pandas as pd
import torch
from safetensors import safe_open

from reveng.experiments.reasoning_topology import (
    METRIC_DIRECTIONS,
    curve_metrics,
    paired_inference,
)


DEFAULT_COHORT = Path(
    "data/behavioral_probes/reasoning_belief_action_cohorts/matched_46_candidates.csv"
)
DEFAULT_WRONG_TURNS = Path(
    "data/behavioral_probes/step_reasoning_drift_matched_46_behavioral/"
    "trajectory_wrong_turn_summary.csv"
)
DEFAULT_ACTIVATIONS = Path("outputs/activation_collection/gpt_oss_20b_boundary_v1")
DEFAULT_OUTPUT = Path("outputs/hypothesis_tests/reasoning_topology_v1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tensor(path: Path, key: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--wrong-turns", type=Path, default=DEFAULT_WRONG_TURNS)
    parser.add_argument("--activation-root", type=Path, default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", nargs="+", type=int, default=[8, 15, 23])
    parser.add_argument("--primary-layer", type=int, default=15)
    parser.add_argument("--n-points", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    """Return Benjamini-Hochberg adjusted p-values in original row order."""
    ordered = values.sort_values()
    ranks = np.arange(1, len(ordered) + 1, dtype=float)
    adjusted = ordered.to_numpy(dtype=float) * len(ordered) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = pd.Series(index=ordered.index, data=np.minimum(adjusted, 1.0))
    return result.reindex(values.index)


def connected_validation_groups(cohort: pd.DataFrame) -> dict[str, str]:
    """Join matched pairs that share any source trajectory."""
    pair_ids = [str(value) for value in cohort.matched_pair_id.unique()]
    parent = {pair_id: pair_id for pair_id in pair_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for _, rows in cohort.groupby("trajectory_id"):
        shared_pairs = [str(value) for value in rows.matched_pair_id.unique()]
        for pair_id in shared_pairs[1:]:
            union(shared_pairs[0], pair_id)
    return {pair_id: find(pair_id) for pair_id in pair_ids}


def main() -> None:
    args = parse_args()
    cohort = pd.read_csv(args.cohort)
    wrong = pd.read_csv(args.wrong_turns)[
        ["example_id", "final_action", "optimal_actions_json", "wrong_turn_detected"]
    ]
    wrong["final_action_optimal"] = wrong.apply(
        lambda row: row.final_action in json.loads(row.optimal_actions_json), axis=1
    )
    cohort = cohort.merge(
        wrong[["example_id", "final_action_optimal", "wrong_turn_detected"]],
        on="example_id",
        how="left",
        validate="one_to_one",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for item in cohort.itertuples(index=False):
        shard = args.activation_root / "shards" / f"{item.example_id}.safetensors"
        if not shard.exists():
            raise FileNotFoundError(f"missing activation shard: {shard}")
        for layer in args.layers:
            values = load_tensor(shard, f"layer_{layer}.sentence_mean").numpy()
            metrics = curve_metrics(values, n_points=args.n_points)
            rows.append(
                {
                    "example_id": item.example_id,
                    "trajectory_id": item.trajectory_id,
                    "matched_pair_id": item.matched_pair_id,
                    "matched_role": item.matched_role,
                    "failure_category": item.primary_step_failure_mode,
                    "final_action_optimal": bool(item.final_action_optimal),
                    "wrong_turn_detected": bool(item.wrong_turn_detected),
                    "layer": layer,
                    "original_sentences": len(values),
                    **metrics,
                }
            )
    metric_rows = pd.DataFrame(rows)
    metric_rows.to_csv(args.output_dir / "trace_metrics.csv", index=False)

    validation_group_by_pair = connected_validation_groups(cohort)
    paired_rows: list[dict[str, object]] = []
    for layer in args.layers:
        subset = metric_rows[metric_rows.layer == layer]
        for metric, direction in METRIC_DIRECTIONS.items():
            wide = subset.pivot(
                index="matched_pair_id", columns="matched_role", values=metric
            )
            differences = wide["failure"] - wide["control"]
            inference = paired_inference(
                differences,
                groups=[
                    validation_group_by_pair[str(pair_id)]
                    for pair_id in differences.index
                ],
                seed=args.seed + layer,
            )
            paired_rows.append(
                {
                    "layer": layer,
                    "metric": metric,
                    "prespecified_direction": direction,
                    "failure_mean": float(
                        subset.loc[subset.matched_role == "failure", metric].mean()
                    ),
                    "control_mean": float(
                        subset.loc[subset.matched_role == "control", metric].mean()
                    ),
                    **inference,
                }
            )
    summary = pd.DataFrame(paired_rows)
    summary["bh_q_within_layer"] = summary.groupby("layer", group_keys=False)[
        "sign_flip_p_two_sided"
    ].transform(benjamini_hochberg)
    summary.to_csv(args.output_dir / "paired_summary.csv", index=False)

    primary = summary[summary.layer == args.primary_layer].copy()
    nominal_supported = primary[
        (primary.mean_difference > 0.0) & (primary.ci_low > 0.0)
    ]
    corrected_supported = nominal_supported[nominal_supported.bh_q_within_layer < 0.05]
    strict_failures = metric_rows[
        (metric_rows.layer == args.primary_layer)
        & (metric_rows.matched_role == "failure")
        & (~metric_rows.final_action_optimal)
    ]
    report_lines = [
        "# Continuous Reasoning-Trajectory Topology Test",
        "",
        "## Question",
        "",
        "Do exact-matched rollout-failure states show more continuous representational recurrence, tortuosity, directional reversal, or weaker late consolidation than controls?",
        "",
        "This test uses no PCA and no clustering. Every trace is linearly resampled to 32 reasoning-progress points, centered within trace, and RMS-normalized. The metrics are continuous trajectory summaries; `nonlocal_recurrence` measures normalized return proximity and is not a claim that a discrete graph cycle exists.",
        "",
        "## Data",
        "",
        f"- States: {metric_rows.example_id.nunique()} in {cohort.matched_pair_id.nunique()} exact matched pairs",
        f"- Source trajectories: {metric_rows.trajectory_id.nunique()}",
        f"- Independent trajectory-connected validation groups: {len(set(validation_group_by_pair.values()))}",
        f"- Strict final-suboptimal failure states at layer {args.primary_layer}: {len(strict_failures)}",
        f"- Primary representation: layer {args.primary_layer} sentence means",
        "- Sensitivity layers: 8 and 23",
        "- Intervals: bootstrap trajectory-connected groups; p-values: exact grouped sign flips",
        "",
        "## Primary layer results",
        "",
        "| Metric | Failure mean | Control mean | Difference | 95% interval | Two-sided p | BH q |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples(index=False):
        report_lines.append(
            f"| {row.metric} | {row.failure_mean:.4f} | {row.control_mean:.4f} | "
            f"{row.mean_difference:+.4f} | [{row.ci_low:+.4f}, {row.ci_high:+.4f}] | "
            f"{row.sign_flip_p_two_sided:.4f} | {row.bh_q_within_layer:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "Multiplicity-corrected support: "
                + ", ".join(corrected_supported.metric.tolist())
                if len(corrected_supported)
                else "None of the four layer-15 metrics survives within-layer false-discovery correction at q < 0.05."
            ),
            (
                "Nominal directional evidence: "
                + ", ".join(nominal_supported.metric.tolist())
                if len(nominal_supported)
                else "No metric has a grouped-bootstrap interval wholly above zero."
            ),
            "",
            "This is a matched observational test. Rollout-failure labels describe the source trajectory and are not identical to a final suboptimal recommendation. The strict final-suboptimal subset is retained in `trace_metrics.csv` but is too small for a primary independent-trajectory test.",
        ]
    )
    (args.output_dir / "run_report.md").write_text("\n".join(report_lines) + "\n")
    manifest = {
        "analysis": "reasoning_topology_v1",
        "status": "completed",
        "uses_pca": False,
        "uses_clustering": False,
        "cohort": str(args.cohort),
        "cohort_sha256": sha256(args.cohort),
        "wrong_turns": str(args.wrong_turns),
        "wrong_turns_sha256": sha256(args.wrong_turns),
        "activation_root": str(args.activation_root),
        "layers": args.layers,
        "primary_layer": args.primary_layer,
        "representation": "sentence_mean",
        "resampled_points": args.n_points,
        "seed": args.seed,
        "states": int(metric_rows.example_id.nunique()),
        "matched_pairs": int(cohort.matched_pair_id.nunique()),
        "validation_groups": int(len(set(validation_group_by_pair.values()))),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(args.output_dir / "run_report.md")


if __name__ == "__main__":
    main()
