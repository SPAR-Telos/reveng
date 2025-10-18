"""Full policy experiment runner.

This module provides experiment runners for full policy elicitation.
"""

import argparse
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from reveng.agents import LLMAgent
from reveng.policy_inspector.extract_action_prob_utils import get_action_probs
from reveng.policy_inspector.policy_elicitation import (
    elicit_policy,
    visualize_policy_probabilities_threadsafe,
    visualize_policy_threadsafe,
)


def _process_single_environment(
    model_name: str, output_base: Path, grid_id: str, env, top_logprobs: int = 5
):
    """Run policy elicitation and visualizations for a single environment.

    Returns a tuple of (grid_id, cost_summary_dict).
    """
    # Create a dedicated agent per environment to avoid shared mutable state across threads
    llm_agent = LLMAgent(model_name=model_name, name="LLM agent")

    llm_policy, llm_policy_metadata = elicit_policy(
        env, llm_agent, top_logprobs=top_logprobs
    )

    # Save metadata
    metadata_path = output_base / f"{grid_id}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(llm_policy_metadata, f, indent=2)

    # Visualize policy
    policy_viz_path = output_base / f"{grid_id}_policy.png"
    visualize_policy_threadsafe(
        llm_policy,
        env,
        filename=str(policy_viz_path),
        title=f"LLM Agent Policy - {grid_id}",
    )

    # Process and visualize policy probabilities
    action_probabilities = get_action_probs(llm_policy_metadata)
    prob_viz_path = output_base / f"{grid_id}_probabilities.png"
    visualize_policy_probabilities_threadsafe(
        action_probabilities, env, filename=str(prob_viz_path)
    )

    return grid_id, llm_agent.get_cost_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full policy experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/baseline_grids.pkl",
        help="Path to dataset pickle file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="prob_policy_results_threads",
        help="Base output directory (default: prob_policy_results)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="together_ai/openai/gpt-oss-20b",
        help="Model name to use",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of parallel workers to use (default: 2)",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=20,
        help="Number of top logprobs to return (default: 20)",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, "rb") as f:
        dataset = pickle.load(f)

    print(f"Dataset loaded: {len(dataset)} environments")

    # Model name for worker agents
    model = args.model_name

    # Create output directory structure: results/{model_name}/
    # Sanitize model name to avoid creating nested directories
    safe_model_name = args.model_name.replace("/", "_")
    output_base = Path(args.output_dir) / safe_model_name
    output_base.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {output_base}")

    # Iterate through environments
    # environments = list(dataset.items())[::10]  # TODO: fix hardcoded 1 grid per config
    environments = list(dataset.items())  # All environments
    # Parallel processing of environments
    cost_summaries = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                _process_single_environment,
                model,
                output_base,
                grid_id,
                env,
                args.top_logprobs,
            ): grid_id
            for grid_id, env in environments
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing environments"
        ):
            grid_id = futures[future]
            try:
                _, cost_summary = future.result()
                cost_summaries.append(cost_summary)
            except Exception as exc:
                print(f"Environment {grid_id} failed with error: {exc}")

    # Aggregate and print cost summary across all workers
    total_cost = sum(cs.get("total_cost", 0.0) for cs in cost_summaries)
    total_calls = sum(cs.get("call_count", 0) for cs in cost_summaries)
    avg_cost_per_call = (total_cost / total_calls) if total_calls > 0 else 0.0

    print("\nPolicy elicitation complete!")
    print(
        f"Cost summary: {{'total_cost': {total_cost:.6f}, 'call_count': {total_calls}, 'avg_cost_per_call': {avg_cost_per_call:.6f}}}"
    )
