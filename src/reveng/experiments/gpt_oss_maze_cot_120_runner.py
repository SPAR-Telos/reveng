"""GPT-OSS-20B Maze CoT Collection: 120 Grids × m=3 Trajectories Pilot

Extended collection runner for generating 120 deterministic regular mazes at flexible grid sizes,
difficulties, and trajectory counts. Reuses core functions from maze_smoke_test.py but with
flexible validation for collection mode (not frozen smoke-test mode).

Run stages:
  plan      - Generate grid manifest and validate schedule
  query     - Execute API calls to Together AI (resumable via JSONL checkpointing)
  analyze   - Aggregate trajectory summaries and diagnostic reports
  package   - Prepare dataset upload to Hugging Face
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

# Import core utilities from maze_smoke_test
from reveng.experiments.maze_smoke_test import (
    ACTIONS,
    ACTION_DELTAS,
    PROMPT_TEMPLATE,
    _append_jsonl,
    _env_layout,
    _fmt,
    _get,
    _usage_int,
    _wall_fraction,
    apply_action,
    build_report,
    check_configured_api_models,
    condition_summary,
    distance_map,
    estimated_cost,
    extract_response,
    layout_at_position,
    locate,
    make_together_query,
    optimal_actions,
    parse_action,
    percentile,
    read_jsonl,
    render_grid,
    stable_hash,
    together_model_availability,
    trajectory_rows,
    unresolved_api_model_ids,
)
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv


# Pilot configuration: 120 grids at 7x7 with 6 difficulties (20 per difficulty), m=3
PILOT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "collection_mode": "pilot_7x7_m3",
    "seed": 42,
    "provider": "Together AI",
    "pricing_verified_utc": "2026-08-16",
    "context_mode": "current full grid only (no trajectory history)",
    "grid_sizes": [7],  # Pilot: 7x7 only
    "difficulty_levels": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # 6 difficulty levels
    "grids_per_difficulty": 20,  # 20 grids per difficulty = 120 total grids
    "grid_candidate_pool_size": 32,
    "temperature": 0.7,
    "greedy_temperature": 0.0,
    "top_p": 0.95,
    "max_output_tokens": 16000,
    "step_limit_multiplier": 2.0,
    "step_limit_slack": 8,
    "step_limit_hard_cap": 100,
    "max_api_attempts": 3,
    # Sampling configuration: m=3 (3 sampled at T=0.7, 1 greedy at T=0)
    "trajectory_config": {
        "sampled_count": 3,
        "greedy_count": 1,
        "sampled_temperature": 0.7,
        "greedy_temperature": 0.0,
    },
    "models": [
        {
            "name": "GPT-OSS-20B",
            "api_model_id": "openai/gpt-oss-20b",
            "local_checkpoint": "openai/gpt-oss-20b",
            "revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
            "local_model_class": "AutoModelForCausalLM",
            "local_dtype": "bfloat16",
            "reasoning_settings": ["medium"],  # Use medium for pilot stability
            "reasoning_control": True,
            "reasoning_trace_format": "gpt_oss_channels",
            "deployment": "serverless",
            "input_usd_per_million_tokens": 0.05,
            "output_usd_per_million_tokens": 0.20,
        },
    ],
    "pricing_note": "Per-token prices are planning values; verify Together pricing before running.",
}


def validate_pilot_config(config: dict[str, Any]) -> None:
    """Validate collection config (more flexible than smoke-test validation)."""
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported config schema_version; expected 1.")
    
    sizes = [int(value) for value in config.get("grid_sizes", [])]
    if len(sizes) < 1 or any(size < 5 or size % 2 == 0 for size in sizes):
        raise ValueError("grid_sizes must contain odd sizes >= 5.")
    
    difficulties = [float(value) for value in config.get("difficulty_levels", [])]
    if not difficulties or any(d < 0 or d > 1 for d in difficulties):
        raise ValueError("difficulty_levels must be between 0.0 and 1.0.")
    
    grids_per_diff = int(config.get("grids_per_difficulty", 0))
    if grids_per_diff < 1:
        raise ValueError("grids_per_difficulty must be >= 1.")
    
    if int(config.get("grid_candidate_pool_size", 0)) < 10:
        raise ValueError("grid_candidate_pool_size must be at least 10.")
    
    if int(config.get("max_output_tokens", 0)) != 16000:
        raise ValueError("max_output_tokens must be 16000.")
    
    if not config.get("models"):
        raise ValueError("At least one model must be configured.")
    
    for model in config["models"]:
        if not model.get("api_model_id") or not model.get("local_checkpoint"):
            raise ValueError("Every model needs api_model_id and local_checkpoint.")
        if not isinstance(model.get("revision", ""), str) or len(model.get("revision", "")) != 40:
            raise ValueError(f"{model['name']}: revision must be a 40-character commit SHA.")


def generate_grid_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate grids_per_difficulty grids per difficulty level (no quantile selection)."""
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
                    with contextlib.redirect_stdout(open(os.devnull, "w")):
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
                
                # Select grids_per_difficulty grids by dividing the candidate pool
                ordered = sorted(
                    candidates,
                    key=lambda row: (
                        int(row["optimal_path_length"]),
                        int(row["generation_seed"]),
                    ),
                )
                
                grids_per_diff = int(config.get("grids_per_difficulty", 20))
                step = max(1, len(ordered) // grids_per_diff)
                selected_indices = list(range(0, len(ordered), step))[:grids_per_diff]
                
                for replicate, rank in enumerate(selected_indices, start=1):
                    selected = dict(ordered[rank])
                    selected.update(
                        {
                            "grid_id": f"size{int(size)}_difficulty{float(difficulty):.1f}_grid{replicate}",
                            "grid_replicate": replicate,
                            "candidate_pool_size": len(ordered),
                            "candidate_path_length_rank": rank + 1,
                        }
                    )
                    rows.append(selected)
    finally:
        random.setstate(saved_state)
    return rows


def build_schedule(
    config: dict[str, Any], grids: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build trajectory schedule with sampled + greedy trajectories."""
    schedule = []
    traj_cfg = config.get("trajectory_config", {})
    sampled_count = int(traj_cfg.get("sampled_count", 1))
    greedy_count = int(traj_cfg.get("greedy_count", 0))
    
    for model in config["models"]:
        for reasoning in model["reasoning_settings"]:
            for grid in grids:
                # Add sampled trajectories
                for sample_idx in range(sampled_count):
                    trajectory_id = "__".join(
                        [
                            model["name"].lower().replace(" ", "-"),
                            str(reasoning),
                            grid["grid_id"],
                            f"sampled_{sample_idx}",
                        ]
                    )
                    limit = min(
                        int(config["step_limit_hard_cap"]),
                        max(
                            int(grid["optimal_path_length"])
                            + int(config["step_limit_slack"]),
                            math.ceil(
                                float(config["step_limit_multiplier"])
                                * int(grid["optimal_path_length"])
                            ),
                        ),
                    )
                    schedule.append(
                        {
                            "trajectory_id": trajectory_id,
                            "model": model["name"],
                            "api_model_id": model["api_model_id"],
                            "reasoning_setting": reasoning,
                            "reasoning_control": bool(model["reasoning_control"]),
                            "temperature": float(traj_cfg.get("sampled_temperature", 0.7)),
                            "sampling_condition": "sampled",
                            "step_limit": limit,
                            **grid,
                        }
                    )
                
                # Add greedy trajectory (only on first grid replicate)
                if greedy_count > 0 and int(grid["grid_replicate"]) == 1:
                    trajectory_id = "__".join(
                        [
                            model["name"].lower().replace(" ", "-"),
                            str(reasoning),
                            grid["grid_id"],
                            "greedy",
                        ]
                    )
                    limit = min(
                        int(config["step_limit_hard_cap"]),
                        max(
                            int(grid["optimal_path_length"])
                            + int(config["step_limit_slack"]),
                            math.ceil(
                                float(config["step_limit_multiplier"])
                                * int(grid["optimal_path_length"])
                            ),
                        ),
                    )
                    schedule.append(
                        {
                            "trajectory_id": trajectory_id,
                            "model": model["name"],
                            "api_model_id": model["api_model_id"],
                            "reasoning_setting": reasoning,
                            "reasoning_control": bool(model["reasoning_control"]),
                            "temperature": float(traj_cfg.get("greedy_temperature", 0.0)),
                            "sampling_condition": "greedy",
                            "step_limit": limit,
                            **grid,
                        }
                    )
    
    return schedule


def run_plan_stage(config_path: Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate grid manifest and schedule, validate before API queries."""
    config = PILOT_CONFIG if config_path is None else json.loads(config_path.read_text())
    validate_pilot_config(config)
    
    print(f"🔧 Generating grids: {len(config['grid_sizes'])} size(s) × {len(config['difficulty_levels'])} difficulties × {config.get('grids_per_difficulty', 20)} grids/difficulty")
    grids = generate_grid_rows(config)
    print(f"✓ Generated {len(grids)} unique grids")
    
    schedule = build_schedule(config, grids)
    print(f"✓ Built schedule with {len(schedule)} trajectories")
    
    # Breakdown
    by_condition = defaultdict(int)
    for row in schedule:
        key = (row["model"], row["grid_size"], row["difficulty"], row["sampling_condition"])
        by_condition[key] += 1
    
    print("\nSchedule breakdown by model/size/difficulty/condition:")
    for (model, size, diff, cond), count in sorted(by_condition.items()):
        print(f"  {model} | size={size} | diff={diff:.1f} | {cond}: {count}")
    
    total_traj = len(schedule)
    est_tokens = total_traj * 3500  # rough estimate: ~3500 tokens per trajectory
    est_cost_low = (est_tokens / 1e6) * 0.05 * 0.2  # input 0.05, output 0.20
    
    print(f"\n📊 Estimate (very rough):")
    print(f"  Total trajectories: {total_traj}")
    print(f"  Estimated total tokens: {est_tokens:,.0f}")
    print(f"  Estimated cost: ${est_cost_low:.2f}–${est_cost_low*2:.2f}")
    print(f"  Sequential time: 10–15 hours")
    print(f"  With 2 workers: 5–8 hours (if Together allows concurrency)")
    
    return config, grids, schedule


def run_query_stage(
    output_dir: Path,
    config: dict[str, Any],
    schedule: list[dict[str, Any]],
    query_fn: Callable[..., Any] | None = None,
    model_names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Execute API calls to Together AI with resumable JSONL checkpointing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    query_fn = query_fn or make_together_query()
    
    if model_names:
        schedule = [row for row in schedule if row["model"] in set(model_names)]
    
    raw_path = output_dir / "raw_api_calls.jsonl"
    calls = __import__("reveng.experiments.maze_smoke_test", fromlist=["run_api_schedule"]).run_api_schedule(
        config, schedule, raw_path, query_fn, progress_fn=lambda msg: print(f"  {msg}")
    )
    
    return calls


def run_analyze_stage(output_dir: Path, calls: list[dict[str, Any]], schedule: list[dict[str, Any]]) -> None:
    """Aggregate trajectory summaries and diagnostic reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trajectories = trajectory_rows(calls, schedule)
    summaries = condition_summary(trajectories, calls)
    
    # Write CSVs
    call_columns = [col for col in calls[0].keys()] if calls else []
    traj_columns = [col for col in trajectories[0].keys()] if trajectories else []
    summary_columns = [col for col in summaries[0].keys()] if summaries else []
    
    from reveng.experiments.maze_smoke_test import _write_csv
    
    _write_csv(output_dir / "calls.csv", calls, call_columns)
    _write_csv(output_dir / "trajectories.csv", trajectories, traj_columns)
    _write_csv(output_dir / "condition_summary.csv", summaries, summary_columns)
    
    # Write report
    report = build_report(None, schedule, calls, trajectories, summaries)  # type: ignore
    (output_dir / "report.md").write_text(report)
    
    print(f"✓ Analysis complete: {len(trajectories)} trajectories")
    print(f"  Calls: {output_dir / 'calls.csv'}")
    print(f"  Trajectories: {output_dir / 'trajectories.csv'}")
    print(f"  Summary: {output_dir / 'condition_summary.csv'}")
    print(f"  Report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    import tyro
    
    tyro.cli(
        {
            "plan": run_plan_stage,
            "query": run_query_stage,
            "analyze": run_analyze_stage,
        }
    )
