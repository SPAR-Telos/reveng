#!/usr/bin/env python3
"""Prepare, query, and analyze the belief-replacement action pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reveng.experiments.belief_replacement_action import (
    PilotConfig,
    analyze_experiment,
    prepare_experiment,
    query_experiment,
    run_all,
)
from reveng.experiments.gpt_oss_activation_pilot import DEFAULT_MODEL_SNAPSHOT


DEFAULT_SOURCE = Path(
    "outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1"
)
DEFAULT_CANDIDATES = Path(
    "data/behavioral_probes/reasoning_belief_action_cohorts/matched_46_candidates.csv"
)
DEFAULT_OUTPUT = Path("outputs/hypothesis_tests/belief_replacement_action_pilot_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace each full reasoning trace with its final state-belief readouts and "
            "compare the resulting action with the full-CoT action."
        )
    )
    parser.add_argument(
        "--stage", choices=("prepare", "query", "analyze", "all"), default="all"
    )
    parser.add_argument("--candidate-rows-path", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--prefix-action-rows-path",
        type=Path,
        default=DEFAULT_SOURCE / "prefix_action_rows.csv",
    )
    parser.add_argument(
        "--belief-rows-path",
        type=Path,
        default=DEFAULT_SOURCE / "belief_rows.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_SNAPSHOT)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--minimum-free-gib", type=float, default=18.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _config(args: argparse.Namespace) -> PilotConfig:
    return PilotConfig(
        candidate_rows_path=args.candidate_rows_path,
        prefix_action_rows_path=args.prefix_action_rows_path,
        belief_rows_path=args.belief_rows_path,
        output_dir=args.output_dir,
        model_path=args.model_path,
        temperature=args.temperature,
        gpu_index=args.gpu_index,
        minimum_free_gib=args.minimum_free_gib,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    config = _config(args)
    if args.stage == "prepare":
        result: object = {"cohort": str(prepare_experiment(config))}
    elif args.stage == "query":
        result = query_experiment(config)
    elif args.stage == "analyze":
        result = {key: str(value) for key, value in analyze_experiment(config).items()}
    else:
        result = run_all(config)
        if "paths" in result:
            result["paths"] = {
                key: str(value) for key, value in result["paths"].items()
            }
        if "manifest" in result:
            result["manifest"] = str(result["manifest"])
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
