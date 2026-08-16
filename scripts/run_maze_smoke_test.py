#!/usr/bin/env python3
"""Prepare, run, and analyze the portable maze API smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reveng.experiments.maze_smoke_test import (
    DEFAULT_CONFIG,
    apply_api_model_overrides,
    build_schedule,
    check_configured_api_models,
    planned_call_breakdown,
    planned_call_bound,
    prepare,
    read_config,
    read_jsonl,
    run_api_schedule,
    select_schedule_models,
    together_model_availability,
    unresolved_api_model_ids,
    write_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "check-api", "query", "analyze", "all", "plan"),
        default="plan",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON config; defaults to the frozen built-in design",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("smoke_test"))
    parser.add_argument(
        "--api-model-id",
        action="append",
        default=[],
        metavar="MODEL=API_ID",
        help="Resolve an account-specific endpoint name at prepare time; repeat as needed",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Run/check only this display name; repeat to select multiple models",
    )
    parser.add_argument(
        "--replace-empty-plan",
        action="store_true",
        help="Replace a stale prepared plan only if it contains no API or local-preflight records",
    )
    parser.add_argument(
        "--confirm-api-run",
        action="store_true",
        help="Required for stages that make paid API calls",
    )
    args = parser.parse_args()

    config = read_config(args.config)
    if args.api_model_id:
        if args.stage not in {"prepare", "plan", "all"}:
            raise SystemExit(
                "Apply --api-model-id during prepare/plan so config.lock.json records it."
            )
        config = apply_api_model_overrides(config, args.api_model_id)
    if args.stage in {"prepare", "all", "plan"}:
        grids, schedule = prepare(
            args.output_dir, config, replace_empty_plan=args.replace_empty_plan
        )
    else:
        lock = args.output_dir / "config.lock.json"
        if not lock.exists():
            raise SystemExit("Run --stage prepare first.")
        config = read_config(lock)
        grids = read_jsonl(args.output_dir / "grids.jsonl")
        schedule = read_jsonl(args.output_dir / "api_schedule.jsonl")

    selected_schedule = select_schedule_models(config, schedule, args.model)

    if args.stage == "plan":
        print(
            json.dumps(
                {
                    "status": "prepared_no_api_calls",
                    "output_dir": str(args.output_dir),
                    "grids": len(grids),
                    "trajectories": len(selected_schedule),
                    "maximum_action_calls_if_every_trajectory_hits_its_step_cap": planned_call_bound(
                        selected_schedule
                    ),
                    "upper_bound_by_model": planned_call_breakdown(
                        config, selected_schedule
                    ),
                    "maximum_output_tokens_per_call": config["max_output_tokens"],
                    "models": sorted(
                        {row["api_model_id"] for row in selected_schedule}
                    ),
                    "unresolved_api_model_ids": unresolved_api_model_ids(config),
                    "next_command": f".venv-maze-api/bin/python scripts/run_maze_smoke_test.py --stage query --output-dir {args.output_dir} --confirm-api-run",
                },
                indent=2,
            )
        )
        return

    if args.stage == "check-api":
        try:
            available = together_model_availability()
        except Exception as exc:
            raise SystemExit(
                f"API model check failed: {type(exc).__name__}: {exc}"
            ) from exc
        checks = check_configured_api_models(config, available)
        if args.model:
            checks = [row for row in checks if row["model"] in set(args.model)]
        print(
            json.dumps(
                {
                    "models": checks,
                    "all_available": all(row["available"] for row in checks),
                },
                indent=2,
            )
        )
        if not checks or not all(row["available"] for row in checks):
            raise SystemExit(2)
        return

    if args.stage in {"query", "all"}:
        if not args.confirm_api_run:
            raise SystemExit(
                "Refusing paid API calls without --confirm-api-run. Run --stage plan first."
            )
        unresolved = {
            row["model"]
            for row in selected_schedule
            if str(row["api_model_id"]).startswith("REPLACE_WITH_")
        }
        if unresolved:
            raise SystemExit(
                "Unresolved API endpoint names for "
                + ", ".join(sorted(unresolved))
                + ". Re-run prepare in a fresh output directory with --api-model-id MODEL=API_ID."
            )
        run_api_schedule(
            config,
            selected_schedule,
            args.output_dir / "raw_api_calls.jsonl",
            progress_fn=print,
        )
    if args.stage in {"query", "analyze", "all", "prepare"}:
        print(json.dumps(write_analysis(args.output_dir, config, schedule), indent=2))


if __name__ == "__main__":
    main()
