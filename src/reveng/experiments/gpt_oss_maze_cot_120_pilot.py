"""GPT-OSS-20B Maze CoT Collection: 120 Grids (7x7, m=11).

This module reuses core functions from maze_smoke_test.py but allows flexible
grid counts and difficulty levels. The pilot focuses on 7x7 grids with 120 unique
grids across 6 difficulty levels, with m=11 (10 sampled + 1 greedy) trajectories per grid.

Core reuse:
- generate_grid_rows() logic (copied to bypass rigid validation)
- build_schedule() for trajectory scheduling
- run_api_schedule() for API execution
- All output formats remain backward-compatible
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.experiments.maze_smoke_test import (
    CALL_COLUMNS,
    TRAJECTORY_COLUMNS,
    _env_layout,
    _write_csv,
    build_report,
    build_schedule,
    condition_summary,
    distance_map,
    locate,
    make_together_query,
    TokenSequenceUnavailable,
    read_jsonl,
    render_grid,
    run_api_schedule,
    trajectory_rows,
)

PILOT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "seed": 42,
    "provider": "Together AI",
    "pricing_verified_utc": "2026-08-13",
    "pricing_source_url": "https://docs.together.ai/docs/serverless/models",
    "context_mode": "current full grid only (no trajectory history)",
    "grid_sizes": [7],  # Pilot: 7x7 only
    "difficulty_levels": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # All 6 difficulty levels
    "grids_per_cell": 20,  # 20 grids per (size, difficulty) cell; total 120 grids
    "grid_candidate_pool_size": 32,  # Same as smoke test
    "grid_selection_quantiles": [0.5, 0.9],  # Median and 90th percentile
    "temperature": 0.7,  # Main sampling temperature
    "greedy_temperature": 0.0,  # Greedy (deterministic) temperature
    "top_p": 0.95,
    "max_output_tokens": 16000,
    "step_limit_multiplier": 2.0,
    "step_limit_slack": 8,
    "step_limit_hard_cap": 100,
    "max_api_attempts": 8,
    "unattended_retry_rounds": 12,
    "unattended_retry_delay_seconds": 60,
    "require_complete_token_sequence": True,
    "models": [
        {
            "name": "GPT-OSS-20B",
            "api_model_id": "openai/gpt-oss-20b",
            "local_checkpoint": "openai/gpt-oss-20b",
            "revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
            "local_model_class": "AutoModelForCausalLM",
            "local_dtype": "bfloat16",
            "reasoning_settings": ["low"],  # Pilot: start with low
            "reasoning_control": True,
            "reasoning_trace_format": "gpt_oss_channels",
            "deployment": "serverless",
            "input_usd_per_million_tokens": 0.05,
            "output_usd_per_million_tokens": 0.20,
        },
        {
            "name": "Gemma-4-31B-IT",
            "api_model_id": "google/gemma-4-31B-it",
            "local_checkpoint": "google/gemma-4-31B-it",
            "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
            "local_model_class": "AutoModelForMultimodalLM",
            "minimum_transformers_version": "5.5.0",
            "local_dtype": "bfloat16",
            "reasoning_settings": ["native"],
            "reasoning_control": False,
            "reasoning_trace_format": "before_action_json",
            "chat_template_kwargs": {"enable_thinking": True},
            "deployment": "serverless",
            "input_usd_per_million_tokens": 0.20,
            "output_usd_per_million_tokens": 0.50,
        },
    ],
    "decision_thresholds": {
        "minimum_goal_success_rate": 0.50,
        "maximum_invalid_or_failed_rate": 0.10,
        "medium_material_success_gain": 0.10,
        "token_budget_headroom": 1.20,
    },
    "pricing_note": "Per-token prices are planning values; verify Together pricing before running.",
}


def _same_collection_design(locked: dict[str, Any], current: dict[str, Any]) -> bool:
    """Ignore retry-only changes while preserving scientific design locks."""
    locked_design = json.loads(json.dumps(locked))
    current_design = json.loads(json.dumps(current))
    runtime_only_keys = {
        "max_api_attempts",
        "unattended_retry_rounds",
        "unattended_retry_delay_seconds",
    }
    for key in runtime_only_keys:
        locked_design.pop(key, None)
        current_design.pop(key, None)
    return locked_design == current_design


def _validate_or_migrate_config_lock(output_dir: Path, config: dict[str, Any]) -> None:
    config_path = output_dir / "config.lock.json"
    if not config_path.exists():
        raise ValueError("Missing config.lock.json; run plan first.")
    locked = json.loads(config_path.read_text())
    if locked == config:
        return
    if _same_collection_design(locked, config):
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        print(
            "Updated config.lock.json for retry-only runtime settings "
            "(retry counts or delays)."
        )
        return
    raise ValueError(
        "Configuration changes affect the frozen collection design; use a new "
        "output directory and run plan first."
    )


def validate_pilot_config(config: dict[str, Any]) -> None:
    """Validate pilot config; more flexible than smoke-test validation."""
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported or missing config schema_version; expected 1.")
    sizes = [int(value) for value in config.get("grid_sizes", [])]
    if not sizes or any(size < 5 or size % 2 == 0 for size in sizes):
        raise ValueError("grid_sizes must contain odd sizes >= 5.")
    difficulties = [float(value) for value in config.get("difficulty_levels", [])]
    if not difficulties or any(d < 0.0 or d > 1.0 for d in difficulties):
        raise ValueError("difficulty_levels must be in [0.0, 1.0].")
    if int(config.get("grids_per_cell", 0)) <= 0:
        raise ValueError("grids_per_cell must be > 0.")
    if int(config.get("max_output_tokens", 0)) != 16000:
        raise ValueError(
            "max_output_tokens must be 16000 for consistency with smoke test."
        )
    if not config.get("models"):
        raise ValueError("At least one model must be configured.")
    for model in config["models"]:
        if not model.get("api_model_id") or not model.get("local_checkpoint"):
            raise ValueError("Every model needs api_model_id and local_checkpoint.")
        if not re.fullmatch(r"[0-9a-f]{40}", str(model.get("revision", ""))):
            raise ValueError(
                f"{model['name']}: local checkpoint revision must be a 40-character commit SHA."
            )


def _wall_fraction(layout: list[list[str]]) -> float:
    """Calculate wall fraction in maze interior."""
    interior = [cell for row in layout[1:-1] for cell in row[1:-1]]
    return sum(cell == "#" for cell in interior) / len(interior)


def generate_grid_rows_flexible(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate grids with flexible config (not constrained by smoke-test validation).

    This is adapted from maze_smoke_test.py's generate_grid_rows() but allows
    flexible grid counts and difficulty levels.
    """
    validate_pilot_config(config)
    rows: list[dict[str, Any]] = []
    master = random.Random(int(config["seed"]))
    saved_state = random.getstate()
    try:
        for size in config["grid_sizes"]:
            for difficulty in config["difficulty_levels"]:
                candidates: list[dict[str, Any]] = []
                for candidate_index in range(int(config["grid_candidate_pool_size"])):
                    env_seed = master.randrange(2**31)
                    random.seed(env_seed)
                    env = Simple2DNavigationEnv(
                        size=int(size), complexity=float(difficulty)
                    )
                    # Suppress reset warning
                    with contextlib.redirect_stdout(io.StringIO()):
                        env.reset(seed=env_seed)
                    layout = _env_layout(env)
                    start = locate(layout, "A")
                    distances = distance_map(layout)
                    if start not in distances:
                        raise RuntimeError("Generated maze is unexpectedly unsolvable.")
                    candidates.append(
                        {
                            "grid_size": int(size),
                            "difficulty": float(difficulty),
                            "candidate_index": candidate_index,
                            "generation_seed": env_seed,
                            "layout": layout,
                            "grid_text": render_grid(layout),
                            "start_x": start[0],
                            "start_y": start[1],
                            "goal_x": locate(layout, "G")[0],
                            "goal_y": locate(layout, "G")[1],
                            "optimal_path_length": distances[start],
                            "observed_wall_fraction": _wall_fraction(layout),
                        }
                    )
                ordered = sorted(
                    candidates,
                    key=lambda row: (
                        int(row["optimal_path_length"]),
                        int(row["generation_seed"]),
                    ),
                )
                selected_indices: set[int] = set()
                # For flexible grids_per_cell, select grids evenly from quantiles
                grids_to_select = int(config["grids_per_cell"])
                for replicate in range(1, grids_to_select + 1):
                    # Distribute selection across the ordered list
                    quantile = (replicate - 0.5) / grids_to_select
                    rank = round((len(ordered) - 1) * quantile)
                    while rank in selected_indices and rank + 1 < len(ordered):
                        rank += 1
                    if rank >= len(ordered):
                        rank = len(ordered) - 1
                    selected_indices.add(rank)
                    selected = dict(ordered[rank])
                    selected.update(
                        {
                            "grid_id": f"size{int(size)}_difficulty{float(difficulty):.1f}_grid{replicate}",
                            "grid_replicate": replicate,
                            "selection_quantile": quantile,
                            "candidate_pool_size": len(ordered),
                            "candidate_path_length_rank": rank + 1,
                        }
                    )
                    rows.append(selected)
    finally:
        random.setstate(saved_state)
    return rows


def plan_stage(
    config: dict[str, Any],
    output_dir: Path,
    overwrite: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate grid manifest and schedule.

    Returns:
        (grids, schedule): grid manifest and trajectory schedule
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.lock.json"
    if config_path.exists():
        _validate_or_migrate_config_lock(output_dir, config)
    else:
        stale_artifacts = [
            name
            for name in ("trajectory_schedule.jsonl", "raw_api_calls.jsonl")
            if (output_dir / name).exists()
        ]
        if stale_artifacts:
            raise ValueError(
                "Existing collection artifacts have no config.lock.json; use a new "
                "output directory instead of mixing designs."
            )
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    grid_manifest_path = output_dir / "grid_manifest.jsonl"
    schedule_path = output_dir / "trajectory_schedule.jsonl"

    if grid_manifest_path.exists() and not overwrite:
        print(f"Loading existing grid manifest from {grid_manifest_path}")
        grids = read_jsonl(grid_manifest_path)
    else:
        print("Generating grids...")
        grids = generate_grid_rows_flexible(config)
        print(f"Generated {len(grids)} grid rows")
        # Write manifest
        grid_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with grid_manifest_path.open("w") as handle:
            for row in grids:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote grid manifest to {grid_manifest_path}")

    if schedule_path.exists() and not overwrite:
        print(f"Loading existing schedule from {schedule_path}")
        schedule = read_jsonl(schedule_path)
    else:
        print("Building schedule...")
        # Modify schedule to include multiple sampled conditions
        base_schedule = build_schedule(config, grids)
        schedule = _expand_schedule_for_m11(base_schedule, config)
        print(f"Generated {len(schedule)} trajectory jobs")
        # Write schedule
        schedule_path.parent.mkdir(parents=True, exist_ok=True)
        with schedule_path.open("w") as handle:
            for row in schedule:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote schedule to {schedule_path}")

    # Validation
    unique_grids = {row["grid_id"] for row in grids}
    print(
        f"\n✓ Unique grids: {len(unique_grids)}"
        f"\n✓ Total trajectory jobs: {len(schedule)}"
        f"\n✓ Expected complete trajectories: {len(schedule)}"
    )

    # Summary by size/difficulty
    by_cell = {}
    for grid in grids:
        key = (int(grid["grid_size"]), float(grid["difficulty"]))
        by_cell.setdefault(key, []).append(grid)
    print("\nGrids per size/difficulty cell:")
    for key in sorted(by_cell.keys()):
        print(f"  size {key[0]}, difficulty {key[1]:.1f}: {len(by_cell[key])} grids")

    # Summary by trajectory type
    sampled = [s for s in schedule if "t07" in s["trajectory_id"]]
    greedy = [
        s
        for s in schedule
        if "t0" in s["trajectory_id"] and "t07" not in s["trajectory_id"]
    ]
    print("\nTrajectory split:")
    print(f"  Sampled (T=0.7): {len(sampled)}")
    print(f"  Greedy (T=0.0): {len(greedy)}")

    return grids, schedule


def _expand_schedule_for_m11(
    base_schedule: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Build schedule with 10 sampled + 1 greedy trajectories per grid.

    Since we generate 20 grids per (size, difficulty) cell (unlike smoke-test's 2),
    we need to:
    1. Keep sampled trajectories as-is (one per grid from build_schedule)
    2. Add 9 additional sampled trajectories per grid (to get 10 sampled)
    3. Add 1 greedy trajectory per grid (only smoke-test replica 1 gets greedy by default)
    """
    expanded = []
    by_grid = {}

    # Group by grid_id
    for row in base_schedule:
        group_key = (row["model"], row["reasoning_setting"], row["grid_id"])
        by_grid.setdefault(group_key, []).append(row)

    for _group_key, grid_rows in sorted(by_grid.items()):
        # Should have exactly 1 sampled + 0 or 1 greedy from base_schedule
        sampled_rows = [r for r in grid_rows if r["sampling_condition"] == "sampled"]
        greedy_rows = [r for r in grid_rows if r["sampling_condition"] != "sampled"]

        # Keep existing sampled trajectory
        if sampled_rows:
            base_sampled = sampled_rows[0]
            expanded.append(base_sampled)

            # Add 9 more sampled trajectories (10 independent T=0.7 draws total)
            for rep_num in range(2, 11):
                new_row = dict(base_sampled)
                # Update trajectory_id to include replicate number
                base_id = re.sub(r"_t07$|_t0$", "", new_row["trajectory_id"])
                new_row["trajectory_id"] = f"{base_id}_sampled_rep{rep_num}_t07"
                new_row["sampling_condition"] = "sampled"
                expanded.append(new_row)

            # Add greedy trajectory if not already present
            if not greedy_rows:
                greedy_row = dict(base_sampled)
                base_id = re.sub(r"_t07$|_t0$", "", greedy_row["trajectory_id"])
                greedy_row["trajectory_id"] = f"{base_id}_greedy_t0"
                greedy_row["temperature"] = float(config["greedy_temperature"])
                greedy_row["sampling_condition"] = "greedy"
                expanded.append(greedy_row)
            else:
                # Use existing greedy
                expanded.extend(greedy_rows)

    return expanded


def query_stage(
    config: dict[str, Any],
    schedule: list[dict[str, Any]],
    output_dir: Path,
    api_key: str | None = None,
    model_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Run API queries and return call-level results."""
    _validate_or_migrate_config_lock(output_dir, config)
    # Match the rest of the repository's CLI behavior: commands launched from
    # the project root should pick up provider credentials from ``.env``.
    # An explicit CLI value still takes precedence.
    env_path = Path(__file__).resolve().parents[3] / ".env"
    load_dotenv(env_path, override=False)
    query_fn = make_together_query(api_key=api_key)

    if model_names:
        schedule = [row for row in schedule if row["model"] in model_names]
    raw_path = output_dir / "raw_api_calls.jsonl"
    print(f"Running API schedule ({len(schedule)} jobs)...")
    print(f"Results will be appended to {raw_path}")

    def progress_callback(msg: str) -> None:
        print(msg)

    retry_rounds = int(config.get("unattended_retry_rounds", 1))
    retry_delay = int(config.get("unattended_retry_delay_seconds", 60))
    for retry_round in range(1, retry_rounds + 1):
        try:
            return run_api_schedule(
                config,
                schedule,
                raw_path,
                query_fn=query_fn,
                progress_fn=progress_callback,
            )
        except TokenSequenceUnavailable as exc:
            if retry_round >= retry_rounds:
                raise
            print(
                f"Token-sequence validation failed in round {retry_round}/"
                f"{retry_rounds}: {exc} Resuming from the JSONL checkpoint "
                f"in {retry_delay} seconds."
            )
            time.sleep(retry_delay)
    raise AssertionError("unreachable")


def analyze_stage(
    config: dict[str, Any],
    schedule: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Aggregate call-level results into trajectory and condition summaries."""
    raw_path = output_dir / "raw_api_calls.jsonl"
    calls = read_jsonl(raw_path)

    print(f"Loaded {len(calls)} call records")

    trajectories = trajectory_rows(calls, schedule)
    print(f"Aggregated into {len(trajectories)} trajectory records")

    summaries = condition_summary(trajectories, calls)
    print(f"Generated {len(summaries)} condition summaries")

    # Write results
    call_output = output_dir / "calls.csv"
    trajectory_output = output_dir / "trajectories.csv"
    summary_output = output_dir / "summary.csv"
    report_output = output_dir / "report.md"

    _write_csv(call_output, calls, CALL_COLUMNS)
    print(f"Wrote {len(calls)} calls to {call_output}")

    _write_csv(trajectory_output, trajectories, TRAJECTORY_COLUMNS)
    print(f"Wrote {len(trajectories)} trajectories to {trajectory_output}")

    _write_csv(summary_output, summaries, summaries[0].keys() if summaries else [])
    print(f"Wrote {len(summaries)} summaries to {summary_output}")

    report = build_report(config, schedule, calls, trajectories, summaries)
    report_output.write_text(report)
    print(f"Wrote report to {report_output}")


def main():
    parser = argparse.ArgumentParser(
        description="GPT-OSS-20B Maze CoT Collection: 7x7 (m=11)"
    )
    parser.add_argument(
        "stage",
        choices=["plan", "query", "analyze", "all"],
        help="Execution stage",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/gpt_oss_maze_cot_120_pilot"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing plan/schedule",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Together API key (default: read from env)",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=[model["name"] for model in PILOT_CONFIG["models"]],
        help="Run only this model; repeat to select multiple models",
    )

    args = parser.parse_args()

    validate_pilot_config(PILOT_CONFIG)

    if args.stage in ("plan", "all"):
        print("=" * 80)
        print("PLAN STAGE: Grid Generation and Schedule Creation")
        print("=" * 80)
        _grids, schedule = plan_stage(PILOT_CONFIG, args.output_dir, args.overwrite)
        if args.stage == "plan":
            return

    if args.stage in ("query", "all"):
        print("\n" + "=" * 80)
        print("QUERY STAGE: API Execution")
        print("=" * 80)
        if args.stage == "query":
            # Load existing plan
            schedule_path = args.output_dir / "trajectory_schedule.jsonl"
            schedule = read_jsonl(schedule_path)
        query_stage(
            PILOT_CONFIG,
            schedule,
            args.output_dir,
            args.api_key,
            set(args.model) if args.model else None,
        )
        if args.stage == "query":
            return

    if args.stage in ("analyze", "all"):
        print("\n" + "=" * 80)
        print("ANALYZE STAGE: Aggregation and Reporting")
        print("=" * 80)
        if args.stage == "analyze":
            # Load existing plan
            schedule_path = args.output_dir / "trajectory_schedule.jsonl"
            schedule = read_jsonl(schedule_path)
        analyze_stage(PILOT_CONFIG, schedule, args.output_dir)

    print("\n" + "=" * 80)
    print("PILOT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import re

    main()
