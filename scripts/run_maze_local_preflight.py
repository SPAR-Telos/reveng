#!/usr/bin/env python3
"""Run the minimal local activation compatibility check on a GPU machine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reveng.experiments.maze_local_preflight import (
    inspect_local_environment,
    read_jsonl,
    run_one_local_preflight,
    write_jsonl,
    write_local_preflight_report,
)
from reveng.experiments.maze_smoke_test import read_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/maze_smoke_test.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("smoke_test"))
    parser.add_argument(
        "--model", action="append", help="Model display name; repeat to run a subset"
    )
    parser.add_argument("--grid-id", default="size11_difficulty0.6_grid1")
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 15, 23])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reasoning-setting",
        default="low",
        choices=("low", "medium", "high", "native"),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--check-environment-only",
        action="store_true",
        help="Check CUDA, Transformers version, and model classes without loading/downloading weights",
    )
    args = parser.parse_args()

    config = read_config(args.config)
    selected = [
        row for row in config["models"] if not args.model or row["name"] in args.model
    ]
    unknown = set(args.model or []) - {row["name"] for row in config["models"]}
    if unknown:
        raise SystemExit(f"Unknown model names: {sorted(unknown)}")
    if args.check_environment_only:
        report = inspect_local_environment(selected)
        print(json.dumps(report, indent=2))
        if not report["ready_without_loading_weights"]:
            raise SystemExit(2)
        return

    grids = read_jsonl(args.output_dir / "grids.jsonl")
    if not grids:
        raise SystemExit("Run scripts/run_maze_smoke_test.py --stage prepare first.")
    matches = [row for row in grids if row["grid_id"] == args.grid_id]
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one grid named {args.grid_id!r}.")

    records_path = args.output_dir / "local_preflight_records.jsonl"
    failures_path = args.output_dir / "local_preflight_failures.jsonl"
    records = read_jsonl(records_path)
    failures = read_jsonl(failures_path)
    completed = {(row["model"], row["grid_id"]) for row in records}
    for model in selected:
        key = (model["name"], args.grid_id)
        if key in completed:
            print(
                f"[local-preflight] already complete: {model['name']} / {args.grid_id}"
            )
            continue
        print(f"[local-preflight] loading {model['local_checkpoint']}", flush=True)
        try:
            record = run_one_local_preflight(
                model_config=model,
                grid=matches[0],
                layers=args.layers,
                reasoning_setting=(
                    args.reasoning_setting
                    if model.get("reasoning_control")
                    else "native"
                ),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
                output_dir=args.output_dir,
                local_files_only=args.local_files_only,
            )
            records.append(record)
            failures = [row for row in failures if row["model"] != model["name"]]
            write_jsonl(records_path, records)
            write_jsonl(failures_path, failures)
        except Exception as exc:
            failures = [row for row in failures if row["model"] != model["name"]]
            failures.append(
                {
                    "model": model["name"],
                    "grid_id": args.grid_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            write_jsonl(failures_path, failures)
            write_local_preflight_report(
                args.output_dir, config["models"], records, failures
            )
            raise
        write_local_preflight_report(
            args.output_dir, config["models"], records, failures
        )

    print(
        json.dumps(
            {
                "status": (
                    "complete"
                    if len({row["model"] for row in records}) == len(config["models"])
                    else "partial"
                ),
                "models_passed": sorted({row["model"] for row in records}),
                "report": str(args.output_dir / "local_preflight.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
