#!/usr/bin/env python3
"""Run local direct-answer action and belief readouts."""

from __future__ import annotations

import argparse
import json

from reveng.experiments.local_immediate_behavioral import run_local_immediate_behavioral


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-rows-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-mode", choices=["sentence", "environment_step"], required=True)
    parser.add_argument("--limit-positions", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = run_local_immediate_behavioral(
        candidate_rows_path=args.candidate_rows_path,
        output_dir=args.output_dir,
        analysis_mode=args.analysis_mode,
        limit_positions=args.limit_positions,
        resume=not args.no_resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
