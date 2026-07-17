#!/usr/bin/env python3
"""Validate action and categorical-belief candidate coverage for a feasibility gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp1-dir", type=Path, required=True)
    parser.add_argument("--exp2-dir", type=Path, required=True)
    parser.add_argument("--expected-prefixes", type=int, default=20)
    parser.add_argument("--minimum-coverage", type=float, default=0.99)
    args = parser.parse_args()

    action_rows = read_csv(args.exp1_dir / "prefix_action_rows.csv")
    belief_rows = read_csv(args.exp2_dir / "belief_rows.csv")
    categorical_rows = [row for row in belief_rows if row.get("answer_space") == "label3"]
    action_covered = sum(
        as_bool(row.get("action_logprob_all_candidates_present")) for row in action_rows
    )
    belief_covered = sum(
        as_bool(row.get("categorical_all_candidates_present"))
        for row in categorical_rows
    )
    n_logprob_queries = len(action_rows) + len(categorical_rows)
    n_covered = action_covered + belief_covered
    coverage = n_covered / n_logprob_queries if n_logprob_queries else 0.0
    report = {
        "status": "passed"
        if len(action_rows) == args.expected_prefixes
        and coverage >= args.minimum_coverage
        else "failed",
        "expected_prefixes": args.expected_prefixes,
        "action_queries": len(action_rows),
        "action_queries_with_all_four_candidates": action_covered,
        "categorical_belief_queries": len(categorical_rows),
        "categorical_queries_with_all_three_candidates": belief_covered,
        "combined_logprob_queries": n_logprob_queries,
        "combined_queries_with_all_candidates": n_covered,
        "combined_candidate_coverage": coverage,
        "minimum_coverage": args.minimum_coverage,
    }
    output = args.exp1_dir / "logprob_candidate_gate.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
