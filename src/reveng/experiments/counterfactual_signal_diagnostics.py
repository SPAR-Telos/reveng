"""Diagnose why the current counterfactual signal is weak."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from reveng.experiments.counterfactual_activation_patching import _read_manifest
from reveng.experiments.counterfactual_artifact_builder import (
    _extract_agent_goal,
    _parse_grid_text_file,
)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_ratio(numer: int, denom: int) -> float | None:
    if denom == 0:
        return None
    return numer / denom


def _load_per_pair_rows(per_pair_csv: Path) -> list[dict[str, str]]:
    with open(per_pair_csv, newline="") as handle:
        return list(csv.DictReader(handle))


def _movement_flags_by_pair(eval_manifest_path: Path) -> dict[str, dict[str, bool]]:
    flags: dict[str, dict[str, bool]] = {}
    for record in _read_manifest(eval_manifest_path):
        layout_a = _parse_grid_text_file(record.spec.grid_a_path)
        layout_b = _parse_grid_text_file(record.spec.grid_b_path)
        agent_a, goal_a = _extract_agent_goal(layout_a)
        agent_b, goal_b = _extract_agent_goal(layout_b)
        flags[record.spec.pair_id] = {
            "agent_moved": agent_a != agent_b,
            "goal_moved": goal_a != goal_b,
            "both_moved": (agent_a != agent_b) and (goal_a != goal_b),
        }
    return flags


def _category_breakdown(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)

    summary: list[dict[str, Any]] = []
    for category, category_rows in sorted(grouped.items()):
        valid_rows = [r for r in category_rows if r["valid_pair"] == "True"]
        usable_rows = [
            r
            for r in valid_rows
            if r["disruptive"] == "False" and r["action_label"] in {"True", "False"}
        ]
        action_true = sum(r["action_label"] == "True" for r in usable_rows)
        disruptive = sum(r["disruptive"] == "True" for r in valid_rows)
        summary.append(
            {
                "category": category,
                "valid_pairs": len(valid_rows),
                "disruptive_pairs": disruptive,
                "disruptive_rate": _safe_ratio(disruptive, len(valid_rows)),
                "retained_non_disruptive_pairs": len(usable_rows),
                "action_true_count": action_true,
                "action_true_rate": _safe_ratio(action_true, len(usable_rows)),
            }
        )
    return summary


def _movement_breakdown(
    rows: list[dict[str, str]],
    movement_flags: dict[str, dict[str, bool]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {"both_moved": [], "not_both_moved": []}
    for row in rows:
        if row["valid_pair"] != "True":
            continue
        key = "both_moved" if movement_flags.get(row["pair_id"], {}).get("both_moved") else "not_both_moved"
        grouped[key].append(row)

    summary: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        usable_rows = [
            r
            for r in group_rows
            if r["disruptive"] == "False" and r["action_label"] in {"True", "False"}
        ]
        disruptive = sum(r["disruptive"] == "True" for r in group_rows)
        action_true = sum(r["action_label"] == "True" for r in usable_rows)
        summary.append(
            {
                "movement_group": key,
                "valid_pairs": len(group_rows),
                "disruptive_pairs": disruptive,
                "disruptive_rate": _safe_ratio(disruptive, len(group_rows)),
                "retained_non_disruptive_pairs": len(usable_rows),
                "action_true_count": action_true,
                "action_true_rate": _safe_ratio(action_true, len(usable_rows)),
            }
        )
    return summary


def _layer_sweep_summary(
    expansion_metrics_csv: Path,
    selection_threshold: float,
) -> dict[str, Any]:
    if not expansion_metrics_csv.exists():
        return {
            "available": False,
            "selection_threshold": selection_threshold,
        }

    with open(expansion_metrics_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected = [
        row
        for row in rows
        if row.get("status") == "ok"
        and abs(_safe_float(row.get("action_threshold")) - selection_threshold) < 1e-9
    ]
    if not selected:
        return {
            "available": False,
            "selection_threshold": selection_threshold,
        }

    action_true_values = [_safe_float(row["action_true_rate"]) for row in selected]
    disruptive_values = [_safe_float(row["disruptive_rate"]) for row in selected]
    action_true_range = max(action_true_values) - min(action_true_values)
    disruptive_range = max(disruptive_values) - min(disruptive_values)

    return {
        "available": True,
        "selection_threshold": selection_threshold,
        "layers": [int(_safe_float(row["layer"])) for row in selected],
        "action_true_range": action_true_range,
        "disruptive_rate_range": disruptive_range,
        "flat_action_true": action_true_range <= 1e-9,
        "flat_disruptive_rate": disruptive_range <= 1e-9,
    }


def diagnose_counterfactual_signal(
    eval_dir: str = "data/cf/eval_results",
    expansion_dir: str = "data/cf/eval_results/expansion",
    eval_manifest_path: str = "data/cf/artifacts/manifest_for_counterfactual_activation_patching.json",
    output_dir: str = "data/cf/eval_results/signal_diagnostics",
    selection_threshold: float = 0.70,
) -> None:
    """Summarize why the current counterfactual signal is weak."""
    eval_root = Path(eval_dir)
    expansion_root = Path(expansion_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    per_pair_csv = eval_root / "per_pair_results.csv"
    aggregate_summary_path = eval_root / "aggregate_summary.json"
    if not per_pair_csv.exists():
        raise FileNotFoundError(f"Missing per-pair results: {per_pair_csv}")
    if not aggregate_summary_path.exists():
        raise FileNotFoundError(f"Missing aggregate summary: {aggregate_summary_path}")

    rows = _load_per_pair_rows(per_pair_csv)
    aggregate = json.loads(aggregate_summary_path.read_text())
    valid_rows = [row for row in rows if row["valid_pair"] == "True"]
    usable_rows = [
        row
        for row in valid_rows
        if row["disruptive"] == "False" and row["action_label"] in {"True", "False"}
    ]
    action_true_count = sum(row["action_label"] == "True" for row in usable_rows)
    movement_flags = _movement_flags_by_pair(Path(eval_manifest_path))

    category_rows = _category_breakdown(rows)
    movement_rows = _movement_breakdown(rows, movement_flags)
    layer_summary = _layer_sweep_summary(
        expansion_root / "layer_threshold_metrics.csv",
        selection_threshold=selection_threshold,
    )

    summary = {
        "current_signature": {
            "valid_pairs": len(valid_rows),
            "disruptive_pairs": aggregate.get("total_pairs_disruptive"),
            "disruptive_rate": aggregate.get("disruptive_rate"),
            "retained_non_disruptive_pairs": len(usable_rows),
            "action_true_count": action_true_count,
            "action_true_rate": _safe_ratio(action_true_count, len(usable_rows)),
            "categories": dict(Counter(row["category"] for row in valid_rows)),
        },
        "desired_signature": {
            "lower_disruptive_rate": True,
            "enough_non_disruptive_pairs_for_belief_action_gap": True,
            "layer_dependent_intervention_curve": True,
        },
        "category_breakdown": category_rows,
        "movement_breakdown": movement_rows,
        "layer_sweep_summary": layer_summary,
    }

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Counterfactual Signal Diagnostics",
        "",
        "## Current Signature",
        f"- valid_pairs: {len(valid_rows)}",
        f"- disruptive_pairs: {aggregate.get('total_pairs_disruptive')}",
        f"- disruptive_rate: {aggregate.get('disruptive_rate')}",
        f"- retained_non_disruptive_pairs: {len(usable_rows)}",
        f"- action_true_count: {action_true_count}",
        f"- action_true_rate: {_safe_ratio(action_true_count, len(usable_rows))}",
        "",
        "## Why The Current Result Looks Weak",
        "- The signal is dominated by disruptive runs, so too many pairs never make it into the belief-action interpretation table.",
        "- The retained non-disruptive denominator is small, which makes it hard to study a meaningful belief-action gap rather than mostly failures.",
        "- The current layer sweep is effectively flat, so it does not show the sharp layer transition expected from a stronger patching effect.",
        "",
        "## Desired vs Current",
        "- Desired: lower disruptive rate, more retained non-disruptive pairs, and a clear layer-dependent intervention curve.",
        "- Current: high disruption, too few usable pairs, and little to no layer dependence in the existing sweep.",
        "",
        "## Category Breakdown",
        "| category | valid_pairs | disruptive_pairs | disruptive_rate | retained_non_disruptive_pairs | action_true_count | action_true_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in category_rows:
        lines.append(
            f"| {row['category']} | {row['valid_pairs']} | {row['disruptive_pairs']} | "
            f"{row['disruptive_rate']} | {row['retained_non_disruptive_pairs']} | "
            f"{row['action_true_count']} | {row['action_true_rate']} |"
        )

    lines.extend(
        [
            "",
            "## Movement Breakdown",
            "| movement_group | valid_pairs | disruptive_pairs | disruptive_rate | retained_non_disruptive_pairs | action_true_count | action_true_rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in movement_rows:
        lines.append(
            f"| {row['movement_group']} | {row['valid_pairs']} | {row['disruptive_pairs']} | "
            f"{row['disruptive_rate']} | {row['retained_non_disruptive_pairs']} | "
            f"{row['action_true_count']} | {row['action_true_rate']} |"
        )

    lines.extend(["", "## Layer Sweep Flatness"])
    if layer_summary["available"]:
        lines.extend(
            [
                f"- selection_threshold: {layer_summary['selection_threshold']:.2f}",
                f"- evaluated_layers: {', '.join(str(layer) for layer in layer_summary['layers'])}",
                f"- action_true_range: {layer_summary['action_true_range']}",
                f"- disruptive_rate_range: {layer_summary['disruptive_rate_range']}",
                f"- flat_action_true: {layer_summary['flat_action_true']}",
                f"- flat_disruptive_rate: {layer_summary['flat_disruptive_rate']}",
            ]
        )
    else:
        lines.append("- No layer sweep summary was available.")

    (out_root / "summary.md").write_text("\n".join(lines) + "\n")


__all__ = ["diagnose_counterfactual_signal"]
