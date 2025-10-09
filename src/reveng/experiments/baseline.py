import argparse
import json
import pickle
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from reveng.agents.alpha_start_agent import Agent, AlphaStarAgent, RandomAgent
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.environment_generator.iso_difficulty_transformations import (
    IsoDifficultyTransformationFactory,
)
from reveng.trajectory_generator.trajectory_generator import generate_trajectories


def run_baseline(
    dataset: Dict[str, Simple2DNavigationEnv],
    agents: List[Agent],
    perturbations: Optional[
        List[Callable[[Simple2DNavigationEnv], Simple2DNavigationEnv]]
    ] = None,
    num_trajectories_per_config: int = 1,
    output_dir: str = "src/reveng/experiments/results/",
    max_steps_per_trajectory: Optional[int] = None,
) -> None:
    """
    Run multiple agents on a dataset of environments with optional perturbations.

    This function generates trajectories for all combinations of:
    - Environments (from dataset)
    - Agents (from agents list)
    - Perturbations (optional transformations applied to environments)

    Results are saved in a structured format for efficient analysis.

    Args:
        dataset: Dictionary mapping grid_id -> Simple2DNavigationEnv
        agents: List of Agent instances to evaluate
        perturbations: Optional list of environment transformation functions.
                      Each function takes an env and returns a transformed env.
                      If None, uses only the base environment.
        num_trajectories_per_config: Number of trajectories to generate per
                                     (environment, agent, perturbation) combination
        output_dir: Directory to save results
        max_steps_per_trajectory: Optional max steps per trajectory (uses env.max_steps if None)

    Output Structure:
        output_dir/
            metadata.json              # Experiment metadata
            trajectories/
                {grid_id}_{agent_name}_{perturbation_name}_{traj_idx}.json
            summary.json              # Aggregate statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trajectories_path = output_path / "trajectories"
    trajectories_path.mkdir(exist_ok=True)

    # Initialize perturbations list
    if perturbations is None:
        perturbations = []

    # Track all results for summary
    all_results = []

    # Metadata
    metadata = {
        "experiment_name": "baseline_trajectories",
        "timestamp": datetime.now().isoformat(),
        "num_environments": len(dataset),
        "num_agents": len(agents),
        "num_perturbations": len(perturbations) if perturbations else 1,
        "num_trajectories_per_config": num_trajectories_per_config,
        "agents": [agent.name for agent in agents],
        "perturbation_names": [
            getattr(p, "__name__", f"perturbation_{i}")
            for i, p in enumerate(perturbations)
        ]
        if perturbations
        else ["base"],
    }

    total_combinations = (
        len(dataset) * len(agents) * (len(perturbations) if perturbations else 1)
    )
    current_combination = 0

    print("Starting baseline experiment:")
    print(f"  Environments: {len(dataset)}")
    print(f"  Agents: {len(agents)}")
    print(
        f"  Perturbations: {len(perturbations) if perturbations else 1} (including base)"
    )
    print(f"  Trajectories per config: {num_trajectories_per_config}")
    print(f"  Total combinations: {total_combinations}")
    print(f"  Output directory: {output_path}")
    print()

    # Iterate over all combinations
    for grid_id, base_env in dataset.items():
        for agent in agents:
            # Reset agent state for each environment
            agent.reset()

            # Test base environment and all perturbations
            perturbation_list = [("base", None)] + [
                (getattr(p, "__name__", f"pert_{i}"), p)
                for i, p in enumerate(perturbations)
            ]

            for perturbation_name, perturbation_fn in perturbation_list:
                current_combination += 1

                print(
                    f"[{current_combination}/{total_combinations}] "
                    f"Grid: {grid_id}, Agent: {agent.name}, "
                    f"Perturbation: {perturbation_name}"
                )

                try:
                    # Use the environment directly from dataset
                    env = base_env

                    metadata = {
                        "grid_id": grid_id,
                        "perturbation_name": perturbation_name,
                        "agent_name": agent.name,
                    }

                    # Apply perturbation if provided
                    if perturbation_fn is not None:
                        env = perturbation_fn(env)

                    # Generate trajectories using existing infrastructure
                    trajectories = generate_trajectories(
                        env=env,
                        agent=agent,
                        num_trajectories=num_trajectories_per_config,
                        max_steps_per_trajectory=max_steps_per_trajectory
                        or env.max_steps,
                        reset_between_trajectories=True,
                        metadata=metadata,
                    )

                    # Save each trajectory
                    for traj_idx, trajectory in enumerate(trajectories):
                        # Create unique filename
                        filename = f"{grid_id}_{agent.name}_{perturbation_name}_{traj_idx:03d}.json"
                        filepath = trajectories_path / filename

                        # Save trajectory
                        with open(filepath, "w") as f:
                            json.dump(asdict(trajectory), f, indent=2)

                        # Track result
                        result_entry = {
                            "grid_id": grid_id,
                            "agent_name": agent.name,
                            "perturbation": perturbation_name,
                            "trajectory_idx": traj_idx,
                            "filename": filename,
                            "final_reward": trajectory.final_reward,
                            "num_steps": len(trajectory.steps),
                            "success": trajectory.final_reward > 0
                            if trajectory.final_reward is not None
                            else False,
                        }
                        all_results.append(result_entry)

                    print(f"  ✓ Generated {len(trajectories)} trajectories")

                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    # Log error but continue
                    error_entry = {
                        "grid_id": grid_id,
                        "agent_name": agent.name,
                        "perturbation": perturbation_name,
                        "error": str(e),
                    }
                    all_results.append(error_entry)

    # Save metadata
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Generate and save summary statistics
    summary = generate_summary(all_results, agents)
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Experiment complete!")
    print(
        f"  Total trajectories generated: {len([r for r in all_results if 'filename' in r])}"
    )
    print(f"  Results saved to: {output_path}")
    print(f"{'=' * 60}")


def generate_summary(results: List[Dict], agents: List[Agent]) -> Dict:
    """Generate summary statistics from experiment results."""
    summary = {
        "total_trajectories": len([r for r in results if "filename" in r]),
        "total_errors": len([r for r in results if "error" in r]),
        "by_agent": {},
        "by_perturbation": {},
    }

    # Compute per-agent statistics
    for agent in agents:
        agent_results = [
            r for r in results if r.get("agent_name") == agent.name and "filename" in r
        ]
        if agent_results:
            summary["by_agent"][agent.name] = {
                "num_trajectories": len(agent_results),
                "avg_steps": sum(r["num_steps"] for r in agent_results)
                / len(agent_results),
                "avg_reward": sum(r["final_reward"] or 0 for r in agent_results)
                / len(agent_results),
                "success_rate": sum(r["success"] for r in agent_results)
                / len(agent_results),
            }

    # Compute per-perturbation statistics
    perturbations = set(r.get("perturbation") for r in results if "perturbation" in r)
    for pert in perturbations:
        pert_results = [
            r for r in results if r.get("perturbation") == pert and "filename" in r
        ]
        if pert_results:
            summary["by_perturbation"][pert] = {
                "num_trajectories": len(pert_results),
                "avg_steps": sum(r["num_steps"] for r in pert_results)
                / len(pert_results),
                "avg_reward": sum(r["final_reward"] or 0 for r in pert_results)
                / len(pert_results),
                "success_rate": sum(r["success"] for r in pert_results)
                / len(pert_results),
            }

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline experiment")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to dataset pickle file"
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=1,
        help="Number of trajectories per configuration (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/reveng/experiments/baseline_results",
        help="Output directory (default: src/reveng/experiments/baseline_results)",
    )
    parser.add_argument(
        "--use-perturbations",
        action="store_true",
        help="Apply iso-difficulty transformations",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, "rb") as f:
        dataset = pickle.load(f)

    # Create agents
    agents = [
        AlphaStarAgent(name="AStar"),
        RandomAgent(name="Random"),
    ]

    # Setup perturbations if requested
    perturbations = None
    if args.use_perturbations:
        factory = IsoDifficultyTransformationFactory()
        perturbations = [
            factory.rotate_env,
            factory.reflect_env,
            factory.transpose_env,
            factory.start_goal_swap,
        ]

    # Run experiment
    run_baseline(
        dataset=dataset,
        agents=agents,
        perturbations=perturbations,
        num_trajectories_per_config=args.num_trajectories,
        output_dir=args.output_dir,
    )
