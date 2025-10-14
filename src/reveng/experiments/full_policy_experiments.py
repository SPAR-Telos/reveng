"""Full policy experiment runner.

This module provides experiment runners for full policy elicitation.
"""

import argparse
import json
import pickle
from pathlib import Path

from tqdm import tqdm

from reveng.agents import LLMAgent
from reveng.policy_inspector.extract_action_prob_utils import get_action_probs
from reveng.policy_inspector.policy_elicitation import (
    elicit_policy,
    visualize_policy,
    visualize_policy_probabilities,
)

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
        default="prob_policy_results",
        help="Base output directory (default: prob_policy_results)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="together_ai/openai/gpt-oss-20b",
        help="Model name to use",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, "rb") as f:
        dataset = pickle.load(f)

    print(f"Dataset loaded: {len(dataset)} environments")

    # Create agent
    model = args.model_name
    llm_agent = LLMAgent(model_name=model, name="LLM agent")

    # Create output directory structure: results/{model_name}/
    output_base = Path(args.output_dir) / args.model_name
    output_base.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {output_base}")

    # Iterate through environments
    environments = list(dataset.items())[::10][
        0:1
    ]  # TODO: fix hardcoded 1 grid per config
    pbar = tqdm(environments, desc="Processing environments")
    for grid_id, env in pbar:
        env_size = f"{env.unwrapped.width}x{env.unwrapped.height}"
        pbar.set_postfix({"grid_id": grid_id, "size": env_size})

        llm_policy, llm_policy_metadata = elicit_policy(env, llm_agent)

        # Save metadata
        metadata_path = output_base / f"{grid_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(llm_policy_metadata, f, indent=2)

        # Visualize policy
        policy_viz_path = output_base / f"{grid_id}_policy.png"
        visualize_policy(
            llm_policy,
            env,
            filename=str(policy_viz_path),
            title=f"LLM Agent Policy - {grid_id}",
        )

        # Process and visualize policy probabilities
        action_probabilities = get_action_probs(llm_policy_metadata)
        prob_viz_path = output_base / f"{grid_id}_probabilities.png"
        visualize_policy_probabilities(
            action_probabilities, env, filename=str(prob_viz_path)
        )

    print("\nPolicy elicitation complete!")
    print(f"Cost summary: {llm_agent.get_cost_summary()}")
