"""Command-line interface for the reveng package."""

import logging

import tyro

from reveng.commands.get_trajectory import (
    get_trajectories,
    get_trajectories_key_door_env,
    get_trajectories_multiple_per_grid,
    get_trajectory,
    get_trajectory_key_door_env,
    upload_trajectories_dir,
)
from reveng.experiments.counterfactual_artifact_builder import (
    build_counterfactual_patch_artifacts,
)
from reveng.experiments.counterfactual_activation_patching import (
    counterfactual_activation_patching,
)


def main():
    """Main entry point for the reveng-cli command-line tool.

    Configures logging and sets up the CLI with available subcommands using tyro.
    Currently supports the following subcommands:
    - get_trajectory: Generate and save agent trajectories in navigation environments
    - get_trajectories: Generate multiple agent trajectories across parameter combinations in parallel
    - get_trajectories_multiple_per_grid: Generate multiple trajectories on the same grid layout
    - upload_trajectories_dir: Upload a directory of trajectory/grid JSON files to Hugging Face
    - get_trajectory_key_door_env: Generate and save agent trajectories in rooms environments with key-door mechanics
    - get_trajectories_key_door_env: Generate multiple agent trajectories in rooms environments with key-door mechanics across parameter combinations in parallel
    - build_counterfactual_patch_artifacts: Build A/B/patched artifacts in-repo for counterfactual evaluation
    - counterfactual_activation_patching: Evaluate belief/action counterfactual outcomes from artifacts
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tyro.extras.subcommand_cli_from_dict(
        {
            "get_trajectory": get_trajectory,
            "get_trajectories": get_trajectories,
            "get_trajectories_multiple_per_grid": get_trajectories_multiple_per_grid,
            "upload_trajectories_dir": upload_trajectories_dir,
            "get_trajectory_key_door_env": get_trajectory_key_door_env,
            "get_trajectories_key_door_env": get_trajectories_key_door_env,
            "build_counterfactual_patch_artifacts": build_counterfactual_patch_artifacts,
            "counterfactual_activation_patching": counterfactual_activation_patching,
        }
    )


if __name__ == "__main__":
    main()
