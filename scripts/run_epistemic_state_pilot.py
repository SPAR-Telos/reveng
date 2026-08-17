#!/usr/bin/env python3
"""Run the analysis-only behavioral epistemic-state pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from reveng.experiments.epistemic_state_pilot import PilotConfig, run_pilot


DEFAULT_BELIEFS = Path(
    "outputs/experiment2_behavioral_beliefs/"
    "gpt_oss_local_sentence_matched46_v1/belief_rows.csv"
)
DEFAULT_POSITIONS = Path(
    "outputs/experiment1_activation_monitor/"
    "gpt_oss_local_sentence_matched46_v1/position_rows.csv"
)
DEFAULT_OUTPUT = Path("outputs/hypothesis_tests/epistemic_state_pilot_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify action-relevant behavioral beliefs and summarize action outcomes."
    )
    parser.add_argument("--belief-rows-path", type=Path, default=DEFAULT_BELIEFS)
    parser.add_argument("--position-rows-path", type=Path, default=DEFAULT_POSITIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--bootstrap-repeats", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_pilot(
        PilotConfig(
            belief_rows_path=args.belief_rows_path,
            position_rows_path=args.position_rows_path,
            output_dir=args.output_dir,
            confidence_threshold=args.confidence_threshold,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=args.seed,
        )
    )
    print(f"Wrote report: {paths['report']}")
    print(f"Wrote primary figure: {paths['figure']}")
    print(f"Wrote sensitivity figure: {paths['sensitivity_figure']}")


if __name__ == "__main__":
    main()
