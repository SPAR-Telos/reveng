#!/usr/bin/env python3
"""Validate sentence-boundary Experiment 1 and 2 output contracts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp1-dir",
        type=Path,
        default=Path("outputs/experiment1_activation_monitor/weisheng_8_state_sentence_experiment1"),
    )
    parser.add_argument(
        "--exp2-dir",
        type=Path,
        default=Path("outputs/experiment2_behavioral_beliefs/weisheng_8_state_sentence_v1"),
    )
    parser.add_argument(
        "--sentences-path",
        type=Path,
        default=Path("data/behavioral_probes/doorkey_chunking_validation/sentences.csv"),
    )
    args = parser.parse_args()
    prefixes = read_csv(args.exp1_dir / "prefix_action_rows.csv")
    beliefs = read_csv(args.exp2_dir / "belief_rows.csv")
    uncertainty = read_csv(args.exp2_dir / "state_belief_uncertainty_rows.csv")
    canonical = {
        (Path(row["path"]).stem, int(row["step_id"]), int(row["sentence_id"])): row
        for row in read_csv(args.sentences_path)
        if row.get("kind") == "reasoning"
    }
    failures: list[str] = []
    if len(prefixes) != 662:
        failures.append(f"expected 662 prefix positions, found {len(prefixes)}")
    n_states = len({row["example_id"] for row in prefixes})
    if n_states != 8:
        failures.append(f"expected 8 states, found {n_states}")
    if {row.get("analysis_unit") for row in prefixes} != {"sentence"}:
        failures.append("prefix rows do not exclusively use analysis_unit=sentence")
    for row in prefixes:
        sentence_id = row.get("canonical_sentence_id")
        if not sentence_id:
            if int(row["reasoning_step_idx"]) != 0:
                failures.append(f"missing sentence id at {row['example_id']}:{row['reasoning_step_idx']}")
            continue
        key = (row["trajectory_id"], int(row["step_index"]), int(sentence_id))
        boundary = canonical.get(key)
        if boundary is None:
            failures.append(f"canonical sentence missing for {key}")
            continue
        if int(row["analysis_unit_char_end"]) != int(boundary["char_end"]):
            failures.append(f"character endpoint mismatch for {key}")
        if int(row["revealed_analysis_chars"]) != int(boundary["char_end"]):
            failures.append(f"revealed character count mismatch for {key}")
    if len(beliefs) > 662 * 13:
        failures.append(f"belief query count exceeds expected maximum: {len(beliefs)}")
    categorical = [row for row in beliefs if row.get("answer_space") == "label3"]
    coordinates = [row for row in beliefs if row.get("answer_space") == "coord_json"]
    if categorical and len(categorical) != 662 * 9:
        failures.append(
            f"expected {662 * 9} categorical belief rows, found {len(categorical)}"
        )
    if coordinates and len(coordinates) != 662 * 4:
        failures.append(
            f"expected {662 * 4} coordinate belief rows, found {len(coordinates)}"
        )
    action_candidate_coverage = sum(
        row.get("action_logprob_all_candidates_present") == "True" for row in prefixes
    )
    categorical_candidate_coverage = sum(
        row.get("categorical_all_candidates_present") == "True" for row in categorical
    )
    combined_logprob_rows = len(prefixes) + len(categorical)
    combined_candidate_coverage = (
        (action_candidate_coverage + categorical_candidate_coverage)
        / combined_logprob_rows
        if combined_logprob_rows
        else 0.0
    )
    if combined_candidate_coverage < 0.99:
        failures.append(
            "combined action and categorical candidate coverage is below 99 percent"
        )
    if len(uncertainty) > 662 * 6:
        failures.append(f"uncertainty query count exceeds expected maximum: {len(uncertainty)}")
    activation_join = args.exp1_dir / "sentence_activation_join_report.json"
    activation_geometry_reused = False
    if activation_join.exists():
        join_report = json.loads(activation_join.read_text())
        activation_geometry_reused = join_report.get("status") == "completed"
        if join_report.get("activation_join_missing") or join_report.get(
            "activation_join_outside_scope"
        ):
            failures.append("sentence activation join is incomplete")
    report: dict[str, Any] = {
        "status": "failed" if failures else "passed",
        "n_states": n_states,
        "n_prefix_positions": len(prefixes),
        "n_behavioral_belief_queries": len(beliefs),
        "maximum_behavioral_belief_queries": 662 * 13,
        "n_state_uncertainty_queries": len(uncertainty),
        "maximum_state_uncertainty_queries": 662 * 6,
        "n_categorical_belief_queries": len(categorical),
        "n_coordinate_belief_queries": len(coordinates),
        "action_candidate_coverage": action_candidate_coverage / len(prefixes) if prefixes else 0,
        "categorical_candidate_coverage": (
            categorical_candidate_coverage / len(categorical) if categorical else 0
        ),
        "combined_candidate_coverage": combined_candidate_coverage,
        "activation_geometry_reused": activation_geometry_reused,
        "failures": failures,
    }
    (args.exp1_dir / "sentence_boundary_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    lines = [
        "# Sentence-Boundary Validation",
        "",
        f"Status: `{report['status']}`",
        "",
        f"- Environment states: {n_states}",
        f"- Prefix positions: {len(prefixes)}",
        f"- Behavioral belief queries: {len(beliefs)} of at most {662 * 13}",
        f"- State-belief uncertainty queries: {len(uncertainty)} of at most {662 * 6}",
        f"- Sentence-aligned activation geometry included: {'yes' if activation_geometry_reused else 'no'}",
    ]
    if failures:
        lines.extend(["", "## Failures", "", *[f"- {failure}" for failure in failures]])
    (args.exp1_dir / "sentence_boundary_validation.md").write_text("\n".join(lines) + "\n")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
