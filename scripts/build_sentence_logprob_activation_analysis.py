#!/usr/bin/env python3
"""Join sentence-level logprob events to Weisheng's sentence activations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reveng.experiments.experiment1_activation_monitor import (
    _build_commitment_summary,
    _classify_events,
    _event_aligned_geometry,
    _geometry_class_summary,
    _load_geometry_from_activations,
    _plot_outputs,
)


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
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--activation-rows",
        type=Path,
        default=Path(
            "outputs/experiment1_activation_monitor/"
            "weisheng_sentence_averaged_activations/sentence_activation_rows.csv"
        ),
    )
    args = parser.parse_args()

    prefixes = read_csv(args.run_dir / "prefix_action_rows.csv")
    trajectories = read_csv(args.run_dir / "trajectory_wrong_turn_summary.csv")
    activations = read_csv(args.activation_rows)
    if {row.get("analysis_unit") for row in prefixes} != {"sentence"}:
        raise ValueError("Sentence activation analysis requires sentence-prefix rows.")

    prefix_keys = {
        (
            row["example_id"],
            int(row["canonical_sentence_id"]),
        )
        for row in prefixes
        if row.get("canonical_sentence_id") not in {None, ""}
    }
    activation_keys = {
        (row["example_id"], int(row["sentence_id"])) for row in activations
    }
    missing_prefix_activations = sorted(prefix_keys - activation_keys)
    outside_prefix_scope = sorted(activation_keys - prefix_keys)
    if missing_prefix_activations or outside_prefix_scope:
        raise ValueError(
            "Sentence activation join is not exact: "
            f"missing={len(missing_prefix_activations)}, "
            f"outside_scope={len(outside_prefix_scope)}"
        )

    events = _classify_events(prefixes)
    commitments = _build_commitment_summary(trajectories, prefixes)
    write_csv(args.run_dir / "event_rows.csv", events)
    write_csv(args.run_dir / "trajectory_commitment_summary.csv", commitments)

    primary_rows = [
        row
        for row in activations
        if row["layer"] == "8"
        and row["representation_kind"] == "sentence_sampled_token_mean"
    ]
    primary_geometry = _load_geometry_from_activations(primary_rows, trajectories)
    primary_events = _event_aligned_geometry(events, primary_geometry)
    write_csv(args.run_dir / "geometry_rows.csv", primary_geometry)
    write_csv(args.run_dir / "event_aligned_geometry_rows.csv", primary_events)
    write_csv(
        args.run_dir / "geometry_class_summary.csv",
        _geometry_class_summary(primary_geometry, trajectories),
    )

    sensitivity_geometry: list[dict[str, Any]] = []
    grouped = sorted(
        {
            (int(row["layer"]), row["representation_kind"])
            for row in activations
            if not (
                row["layer"] == "8"
                and row["representation_kind"] == "sentence_sampled_token_mean"
            )
        }
    )
    for layer, representation in grouped:
        rows = [
            row
            for row in activations
            if int(row["layer"]) == layer
            and row["representation_kind"] == representation
        ]
        sensitivity_geometry.extend(_load_geometry_from_activations(rows, trajectories))
    write_csv(args.run_dir / "geometry_sensitivity_rows.csv", sensitivity_geometry)
    write_csv(
        args.run_dir / "event_aligned_geometry_sensitivity_rows.csv",
        _event_aligned_geometry(events, sensitivity_geometry),
    )
    _plot_outputs(args.run_dir, events, primary_events, commitments)

    report = {
        "status": "completed",
        "analysis_unit": "sentence",
        "n_prefix_positions": len(prefixes),
        "n_nonzero_sentence_prefixes": len(prefix_keys),
        "n_activation_backed_sentences": len(activation_keys),
        "prefix_zero_has_activation": False,
        "activation_join_missing": 0,
        "activation_join_outside_scope": 0,
        "primary_layer": 8,
        "primary_representation": "sentence_sampled_token_mean",
        "sensitivity_layers": sorted({int(row["layer"]) for row in activations}),
        "sensitivity_representations": sorted(
            {row["representation_kind"] for row in activations}
        ),
        "n_primary_geometry_rows": len(primary_geometry),
        "n_primary_event_aligned_rows": len(primary_events),
        "n_sensitivity_geometry_rows": len(sensitivity_geometry),
        "event_counts": dict(Counter(row["event_type"] for row in events)),
    }
    (args.run_dir / "sentence_activation_join_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    event_counts = Counter(row["event_type"] for row in events)
    lines = [
        "# Sentence-Level Action Commitment and Activation Results",
        "",
        "The recommended action and its uncertainty come from one GPT-OSS-20B request per sentence prefix at temperature 0.7. The action is the argmax over UP, DOWN, LEFT, and RIGHT token probabilities; uncertainty is Shannon entropy over the same normalized distribution.",
        "",
        f"- Environment states: {len(trajectories)}",
        f"- Prefix positions: {len(prefixes)}",
        f"- Activation-backed sentence positions: {len(activation_keys)}",
        "- Prefix zero activation: unavailable by design",
        "- Primary activation analysis: layer 8, mean over all stride-2 sampled positions in each sentence",
        "- Sensitivity analyses: trailing up to three sampled positions and all other available even layers",
        "",
        "| Event | Count |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name.replace('_', ' ')} | {count} |"
        for name, count in sorted(event_counts.items())
    )
    (args.run_dir / "run_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
