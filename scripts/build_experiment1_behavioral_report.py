#!/usr/bin/env python3
"""Build Experiment 1 behavioral events and figures without activation reuse."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from reveng.experiments.experiment1_activation_monitor import (
    _build_commitment_summary,
    _classify_events,
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
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/experiment1_activation_monitor/weisheng_8_state_sentence_experiment1"),
    )
    args = parser.parse_args()
    prefix_rows = read_csv(args.run_dir / "prefix_action_rows.csv")
    trajectory_rows = read_csv(args.run_dir / "trajectory_wrong_turn_summary.csv")
    events = _classify_events(prefix_rows)
    commitments = _build_commitment_summary(trajectory_rows, prefix_rows)
    write_csv(args.run_dir / "event_rows.csv", events)
    write_csv(args.run_dir / "trajectory_commitment_summary.csv", commitments)
    _plot_outputs(args.run_dir, events, [], commitments)
    counts = Counter(row["event_type"] for row in events)
    commitment_progress = [
        float(row["commitment_character_progress"])
        for row in commitments
        if row.get("commitment_character_progress") not in {None, ""}
    ]
    lines = [
        "# Experiment 1 Behavioral Sentence-Boundary Results",
        "",
        "This run uses every canonical reasoning sentence boundary. No activation geometry is reported because the available Weisheng tensors are averages over different packed spans.",
        "",
        f"- Environment states: {len(commitments)}",
        f"- Prefix positions: {len(prefix_rows)}",
        f"- Analysis unit: {prefix_rows[0].get('analysis_unit', '') if prefix_rows else ''}",
        f"- Mean commitment position by character progress: {mean(commitment_progress):.3f}" if commitment_progress else "- Mean commitment position: unavailable",
        "",
        "| Event | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {name.replace('_', ' ')} | {count} |" for name, count in sorted(counts.items()))
    (args.run_dir / "run_report.md").write_text("\n".join(lines) + "\n")
    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "behavioral_report_status": "completed",
            "activation_geometry_status": "not_run_incompatible_packed_averages",
            "n_prefix_positions": len(prefix_rows),
            "maximum_action_queries": len(prefix_rows)
            * (1 + int(manifest.get("action_mc_sample_repeats", 0))),
            "n_action_events": len(events),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
