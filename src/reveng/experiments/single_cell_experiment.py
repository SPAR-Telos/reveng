"""Single cell experiment runner.

This module provides an experiment runner that queries agents at specific positions
in generated environments and saves the results in JSONL format.
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from reveng.agents import LLMAgent
from reveng.agents.alpha_start_agent import AlphaStarAgent
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.environment_generator.wrappers.text_obs_wrapper import (
    FullObservabilityTextWrapper,
)


def _process_single_config(
    model_name: str,
    grid_size: int,
    complexity: float,
    grid_index: int,
    top_logprobs: int,
):
    """Process a single environment configuration.

    Returns a tuple of (result_dict, cost_summary_dict).
    """
    # Create dedicated agents per thread to avoid shared state
    llm_agent = LLMAgent(model_name=model_name, name="LLM agent")
    astar_agent = AlphaStarAgent(name="A* agent")

    # Generate environment
    env = Simple2DNavigationEnv(
        size=grid_size,
        complexity=complexity,
        render_mode=None,
    )
    env.reset()

    # Get agent position and goal position
    agent_position = tuple(env.agent_pos)
    goal_position = tuple(env.goal_pos)

    # Get observation as text (same as LLM agent sees it)
    text_env = FullObservabilityTextWrapper(env)
    observation = text_env.observation(None)

    # Query LLM agent
    action_llm, metadata_llm = llm_agent.select_action(
        env, return_logprobs=True, top_logprobs=top_logprobs
    )

    # Query A* agent
    action_astar, metadata_astar = astar_agent.select_action(env)

    # Create result record
    result = {
        "size": grid_size,
        "complexity": complexity,
        "agent_position": agent_position,
        "goal_position": goal_position,
        "observation": observation,
        "action": action_llm,
        "metadata": metadata_llm,
        "action_astar": action_astar,
    }

    return result, llm_agent.get_cost_summary()


def run_single_cell_experiment(
    model_name: str,
    output_file: str,
    grid_sizes: list[int],
    complexities: list[float],
    num_grids_per_cfg: int = 1,
    top_logprobs: int = 20,
    num_workers: int = 2,
):
    """Run single-cell experiment across multiple environment configurations.

    For each generated environment, queries both LLM and A* agents at the agent's
    starting position and saves the results.

    Args:
        model_name: Name of the LLM model to use
        output_file: Path to output JSONL file
        grid_sizes: List of grid sizes to test
        complexities: List of maze complexities (0.0 to 1.0)
        num_grids_per_cfg: Number of grids to generate per configuration
        top_logprobs: Number of top logprobs to return for LLM agent
        num_workers: Number of parallel workers (default: 2)
    """
    # Calculate total experiments
    total_experiments = len(grid_sizes) * len(complexities) * num_grids_per_cfg

    print("Running single-cell experiments:")
    print(f"  Model: {model_name}")
    print(f"  Grid sizes: {grid_sizes}")
    print(f"  Complexities: {complexities}")
    print(f"  Grids per config: {num_grids_per_cfg}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Workers: {num_workers}")
    print(f"  Output: {output_file}")

    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Thread-safe file writing lock
    file_lock = threading.Lock()

    # Open output file for writing
    cost_summaries = []

    with open(output_path, "w") as f:
        # Build list of all configurations to process
        configs = [
            (grid_size, complexity, i)
            for grid_size in grid_sizes
            for complexity in complexities
            for i in range(num_grids_per_cfg)
        ]

        # Process configurations in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_config,
                    model_name,
                    grid_size,
                    complexity,
                    grid_index,
                    top_logprobs,
                ): (grid_size, complexity, grid_index)
                for grid_size, complexity, grid_index in configs
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing experiments",
            ):
                grid_size, complexity, grid_index = futures[future]
                try:
                    result, cost_summary = future.result()
                    cost_summaries.append(cost_summary)

                    # Thread-safe file write
                    with file_lock:
                        f.write(json.dumps(result) + "\n")
                        f.flush()

                except Exception as exc:
                    print(
                        f"Experiment grid_size{grid_size}_complexity{complexity:.2f}_{grid_index:04d} failed: {exc}"
                    )

    # Aggregate cost summary
    total_cost = sum(cs.get("total_cost", 0.0) for cs in cost_summaries)
    total_calls = sum(cs.get("call_count", 0) for cs in cost_summaries)
    avg_cost_per_call = (total_cost / total_calls) if total_calls > 0 else 0.0

    print("\nExperiment complete!")
    print(f"Results saved to: {output_path}")
    print(
        f"Cost summary: {{'total_cost': {total_cost:.6f}, 'call_count': {total_calls}, 'avg_cost_per_call': {avg_cost_per_call:.6f}}}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run single-cell experiment: query agents at start position"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="together_ai/openai/gpt-oss-20b",
        help="Model name to use (default: together_ai/openai/gpt-oss-20b)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_interp/single_cell_results_3k.jsonl",
        help="Output JSONL file path (default: results_interp/single_cell_results.jsonl)",
    )
    parser.add_argument(
        "--grid-sizes",
        type=int,
        nargs="+",
        default=[7, 9, 11, 13, 15],
        help="List of grid sizes (default: [7, 9, 11, 13, 15])",
    )
    parser.add_argument(
        "--complexities",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="List of complexities from 0.0 to 1.0 (default: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])",
    )
    parser.add_argument(
        "--num-grids-per-cfg",
        type=int,
        default=100,
        help="Number of grids per configuration (default: 10)",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=20,
        help="Number of top logprobs to return (default: 20)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 2)",
    )

    args = parser.parse_args()

    # Validate complexities
    for c in args.complexities:
        if not 0.0 <= c <= 1.0:
            parser.error(f"Complexity must be between 0.0 and 1.0, got {c}")

    # Run experiment
    run_single_cell_experiment(
        model_name=args.model_name,
        output_file=args.output,
        grid_sizes=args.grid_sizes,
        complexities=args.complexities,
        num_grids_per_cfg=args.num_grids_per_cfg,
        top_logprobs=args.top_logprobs,
        num_workers=args.num_workers,
    )
