import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from reveng.agents.llm_agent import LLMAgent
from reveng.datatypes import CustomJSONEncoder
from reveng.environment_generator.rooms_minigrid import RoomsMinigridEnv
from reveng.environment_generator.wrappers.text_obs_wrapper import (
    FullObservabilityTextWrapper,
)
from reveng.policy_inspector.policy_elicitation import generate_one_trajectory


# Default configuration parameters
DEFAULT_N_RUNS = 100
DEFAULT_TEMPLATE_NAME = "grid_full_observability_instrumental_goals.j2"
DEFAULT_OUTPUT_DIR = "instrumental_goals_results"
DEFAULT_MODELS = ["together_ai/openai/gpt-oss-20b"]
DEFAULT_NUM_WORKERS = 10
DEFAULT_MAX_STEPS = 30
DEFAULT_ROOMS_PER_SIDE = 2


def process_single_run(
    model_name: str,
    run_num: int,
    base_dir: Path,
    template_path: Path,
    max_steps: int = 30,
    rooms_per_side: int = 2,
) -> Tuple[int, Dict, Dict]:
    """Process a single complete run with door and key.

    Args:
        model_name: Name of the model to use
        run_num: Run number identifier
        base_dir: Base output directory
        template_path: Path to the prompt template
        max_steps: Maximum steps for the environment and trajectory
        rooms_per_side: Number of rooms per side

    Returns:
        Tuple of (run_num, run_result, cost_summary)
    """
    # Create a dedicated agent per run to avoid shared state across threads
    agent = LLMAgent(
        model_name=model_name, name="LLM agent", template_path=template_path
    )

    # Create directory structure: base_dir/model/run_id
    model_simplified = model_name.split("/")[-1]  # e.g., "gpt-oss-20b"
    output_dir = base_dir / model_simplified / f"run_{run_num}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create fresh environment for this run
    env = RoomsMinigridEnv(
        add_door_key=True, max_steps=max_steps, rooms_per_side=rooms_per_side
    )
    env.reset()

    # Collect trajectory
    trajectory = generate_one_trajectory(
        env=env,
        grid_id="rooms_env_with_door_key",
        agent=agent,
        max_steps_per_trajectory=max_steps,
        top_logprobs=1,
        use_logprobs=False,
        text_wrapper_cls=FullObservabilityTextWrapper,
        save_images=True,
        image_save_dir=output_dir,
    )

    # Save trajectory
    trajectory_path = output_dir / "trajectory.json"
    with open(trajectory_path, "w") as f:
        json.dump(trajectory, f, indent=2, cls=CustomJSONEncoder)

    # Build run result
    run_result = {
        "run_num": run_num,
        "steps": len(trajectory.steps),
        "final_reward": trajectory.final_reward,
        "reached_goal": trajectory.traj_metadata.get("reached_goal", False),
        "metadata": trajectory.traj_metadata,
    }

    # Get cost summary for this run
    cost_summary = agent.get_cost_summary()

    # Save individual run result immediately
    run_result_path = output_dir / "run_result.json"
    with open(run_result_path, "w") as f:
        json.dump({"run_result": run_result, "cost_summary": cost_summary}, f, indent=2)

    return run_num, run_result, cost_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run instrumental goals experiments with parallel workers"
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help=f"Number of runs per model (default: {DEFAULT_N_RUNS})",
    )
    parser.add_argument(
        "--template-name",
        type=str,
        default=DEFAULT_TEMPLATE_NAME,
        help=f"Template file name (default: {DEFAULT_TEMPLATE_NAME})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Model names to use (default: {DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of parallel workers to split runs across (default: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum steps per environment and trajectory (default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--rooms-per-side",
        type=int,
        default=DEFAULT_ROOMS_PER_SIDE,
        help=f"Number of rooms per side (default: {DEFAULT_ROOMS_PER_SIDE})",
    )

    args = parser.parse_args()

    # Base output directory with session timestamp
    session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    base_output_dir = Path(args.output_dir) / session_id
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting experiment session: {session_id}")
    print(f"Total runs per model: {args.n_runs}")
    print(
        f"Parallel workers: {args.num_workers} (runs will be distributed across workers)"
    )
    print(
        f"Total experiments: {len(args.models)} models × {args.n_runs} runs = {len(args.models) * args.n_runs}"
    )
    runs_at_once = min(args.num_workers, args.n_runs)
    print(f"Processing {runs_at_once} runs at a time")

    # Template path
    template_path = Path(__file__).parent.parent / "templates" / args.template_name

    # Store all results for final summary
    all_results = []

    # Loop over models
    for model_name in args.models:
        print(f"\n{'#' * 60}")
        print(f"# Starting experiments with model: {model_name}")
        print(f"{'#' * 60}")

        model_simplified = model_name.split("/")[-1]
        model_results = {"model": model_name, "runs": []}

        # Create list of run numbers
        run_numbers = list(range(args.n_runs))

        # Process runs in parallel using ThreadPoolExecutor
        cost_summaries = []
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all runs for this model
            futures = {
                executor.submit(
                    process_single_run,
                    model_name,
                    run_num,
                    base_output_dir,
                    template_path,
                    args.max_steps,
                    args.rooms_per_side,
                ): run_num
                for run_num in run_numbers
            }

            # Process completed runs as they finish
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Processing {model_simplified} runs",
            ):
                run_num = futures[future]
                try:
                    _, run_result, cost_summary = future.result()
                    model_results["runs"].append(run_result)
                    cost_summaries.append(cost_summary)

                    # Print brief status
                    print(
                        f"  Run {run_num}: "
                        f"Steps={run_result['steps']}, "
                        f"Reward={run_result['final_reward']}, "
                        f"Reached goal={run_result['reached_goal']}"
                    )

                except Exception as exc:
                    print(f"  Run {run_num} failed with error: {exc}")

        # Sort runs by run_num for consistent ordering
        model_results["runs"].sort(key=lambda x: x["run_num"])

        # Aggregate cost summaries
        total_cost = sum(cs.get("total_cost", 0.0) for cs in cost_summaries)
        total_calls = sum(cs.get("call_count", 0) for cs in cost_summaries)
        model_results["cost"] = {
            "total_cost": total_cost,
            "call_count": total_calls,
            "avg_cost_per_call": (total_cost / total_calls) if total_calls > 0 else 0.0,
        }

        # Count successful runs
        successful_runs = sum(
            1 for run in model_results["runs"] if run.get("reached_goal", False)
        )
        total_runs = len(model_results["runs"])
        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0.0

        model_results["success_summary"] = {
            "successful_runs": successful_runs,
            "total_runs": total_runs,
            "success_rate": success_rate,
        }

        print(
            f"\nModel {model_name} - Total cost: ${total_cost:.6f}, Calls: {total_calls}"
        )
        print(f"Successful runs: {successful_runs}/{total_runs} ({success_rate:.1f}%)")

        all_results.append(model_results)

    # Save summary to file
    summary_path = base_output_dir / "summary.json"
    summary_data = {
        "session_id": session_id,
        "n_runs": args.n_runs,
        "template_name": args.template_name,
        "num_workers": args.num_workers,
        "max_steps": args.max_steps,
        "rooms_per_side": args.rooms_per_side,
        "results": all_results,
    }

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    # Print overall success summary
    print(f"\n{'=' * 60}")
    print("OVERALL SUCCESS SUMMARY")
    print(f"{'=' * 60}")
    for model_result in all_results:
        model_name = model_result["model"]
        success_summary = model_result["success_summary"]
        print(
            f"{model_name}: {success_summary['successful_runs']}/{success_summary['total_runs']} "
            f"({success_summary['success_rate']:.1f}%)"
        )
