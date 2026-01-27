import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from reveng.agents.llm_agent import LLMAgent
from reveng.datatypes import CustomJSONEncoder, Trajectory
from reveng.environment_generator.coin_minigrid import CoinMinigridEnv
from reveng.environment_generator.wrappers.text_obs_wrapper import (
    FullObservabilityTextWrapper,
)
from reveng.policy_inspector.policy_elicitation import generate_one_trajectory
from reveng.environment_generator.utils import remove_coin, clone_env


# Default configuration parameters
DEFAULT_N_RUNS = 7
DEFAULT_TEMPLATE_NAME = "grid_full_observability_coin_only_legend.j2"
DEFAULT_OUTPUT_DIR = "coin_experiments_results_only_legend_qwen"
DEFAULT_MODELS = ["together_ai/openai/gpt-oss-20b"]
DEFAULT_NUM_WORKERS = 8
DEFAULT_ENV_SIZE = 9
DEFAULT_MAX_STEPS = 30
DEFAULT_MAX_STEPS_PER_TRAJECTORY = 50


def extract_path_from_trajectory(trajectory: Trajectory) -> List[Tuple[int, int]]:
    """Extract the sequence of agent positions from a trajectory.

    Args:
        trajectory: The trajectory object containing steps with agent positions

    Returns:
        List of (x, y) tuples representing the path taken
    """
    path = []
    for step in trajectory.steps:
        if step.agent_pos is not None:
            path.append(step.agent_pos)
    return path


def compare_path_overlap(
    path1: List[Tuple[int, int]], path2: List[Tuple[int, int]]
) -> Dict:
    """Compare two paths and calculate overlap metrics.

    Args:
        path1: First path (list of positions)
        path2: Second path (list of positions)

    Returns:
        Dictionary with overlap metrics
    """
    set1 = set(path1)
    set2 = set(path2)

    overlap = set1 & set2
    union = set1 | set2

    overlap_ratio = len(overlap) / len(union) if union else 0.0

    return {
        "overlap_positions": len(overlap),
        "overlap_ratio": overlap_ratio,
        "path1_unique_positions": len(set1),
        "path2_unique_positions": len(set2),
        "total_unique_positions": len(union),
    }


def run_single_trajectory(
    model_name: str, env_type: str, env, agent, base_dir: Path, run_num: int
):
    """Run a single trajectory experiment and save results."""
    # Create directory structure: base_dir/model/run_id/env_type
    model_simplified = model_name.split("/")[-1]  # e.g., "gpt-oss-20b"
    output_dir = base_dir / model_simplified / f"run_{run_num}" / env_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect trajectory
    grid_id = f"coin_env_{env_type}"
    trajectory = generate_one_trajectory(
        env=env,
        grid_id=grid_id,
        agent=agent,
        max_steps_per_trajectory=DEFAULT_MAX_STEPS_PER_TRAJECTORY,
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

    return trajectory


def process_single_run(
    model_name: str,
    run_num: int,
    base_dir: Path,
    template_path: Path,
    env_size: int = 9,
    max_steps: int = 100,
) -> Tuple[int, Dict, Dict]:
    """Process a single complete run (with coin + without coin).

    Args:
        model_name: Name of the model to use
        run_num: Run number identifier
        base_dir: Base output directory
        template_path: Path to the prompt template
        env_size: Size of the environment
        max_steps: Maximum steps for the environment

    Returns:
        Tuple of (run_num, run_result, cost_summary)
    """
    # Create a dedicated agent per run to avoid shared state across threads
    agent = LLMAgent(
        model_name=model_name,
        name="LLM agent",
        template_path=template_path,
        temperature=1.0,
    )

    # Create fresh environments for this run
    env_with_coin = CoinMinigridEnv(size=env_size, max_steps=max_steps)
    env_with_coin.reset()
    env_without_coin = remove_coin(clone_env(env_with_coin))

    # Run experiment with coin
    trajectory_with_coin = run_single_trajectory(
        model_name=model_name,
        env_type="with_coin",
        env=env_with_coin,
        agent=agent,
        base_dir=base_dir,
        run_num=run_num,
    )

    # Run experiment without coin
    trajectory_without_coin = run_single_trajectory(
        model_name=model_name,
        env_type="without_coin",
        env=env_without_coin,
        agent=agent,
        base_dir=base_dir,
        run_num=run_num,
    )

    # Extract coin collection info from trajectories
    coin_collected_with = trajectory_with_coin.traj_metadata.get("coin_collected", None)
    coin_collected_without = trajectory_without_coin.traj_metadata.get(
        "coin_collected", None
    )

    # Compare paths between the two trajectories
    path_with_coin = extract_path_from_trajectory(trajectory_with_coin)
    path_without_coin = extract_path_from_trajectory(trajectory_without_coin)
    path_comparison = compare_path_overlap(path_with_coin, path_without_coin)

    # Build run result
    run_result = {
        "run_num": run_num,
        "with_coin": {
            "steps": len(trajectory_with_coin.steps),
            "reward": trajectory_with_coin.final_reward,
            "coin_collected": coin_collected_with,
        },
        "without_coin": {
            "steps": len(trajectory_without_coin.steps),
            "reward": trajectory_without_coin.final_reward,
            "coin_collected": coin_collected_without,
        },
        "path_comparison": path_comparison,
    }

    # Get cost summary for this run
    cost_summary = agent.get_cost_summary()

    # Save individual run result immediately
    model_simplified = model_name.split("/")[-1]
    run_result_path = base_dir / model_simplified / f"run_{run_num}" / "run_result.json"
    with open(run_result_path, "w") as f:
        json.dump({"run_result": run_result, "cost_summary": cost_summary}, f, indent=2)

    return run_num, run_result, cost_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run coin experiments with parallel workers"
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
        help=f"Number of parallel workers (default: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "--env-size",
        type=int,
        default=DEFAULT_ENV_SIZE,
        help=f"Environment size (default: {DEFAULT_ENV_SIZE})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum steps per environment (default: {DEFAULT_MAX_STEPS})",
    )

    args = parser.parse_args()

    # Base output directory with session timestamp
    session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    base_output_dir = Path(args.output_dir) / session_id
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting experiment session: {session_id}")
    print(f"Number of runs per model: {args.n_runs}")
    print(f"Number of parallel workers: {args.num_workers}")
    print(
        f"Total experiments: {len(args.models)} models × {args.n_runs} runs × 2 environments = {len(args.models) * args.n_runs * 2}"
    )

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
                    args.env_size,
                    args.max_steps,
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
                        f"With coin (steps={run_result['with_coin']['steps']}, coin={run_result['with_coin']['coin_collected']}), "
                        f"Without coin (steps={run_result['without_coin']['steps']}, coin={run_result['without_coin']['coin_collected']}), "
                        f"Path overlap: {run_result['path_comparison']['overlap_ratio']:.2%}"
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

        print(
            f"\nModel {model_name} - Total cost: ${total_cost:.6f}, Calls: {total_calls}"
        )

        all_results.append(model_results)

    # Save summary to file
    summary_path = base_output_dir / "summary.json"
    summary_data = {
        "session_id": session_id,
        "n_runs": args.n_runs,
        "template_name": args.template_name,
        "num_workers": args.num_workers,
        "results": all_results,
    }

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
