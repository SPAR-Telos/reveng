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
        }
    )


if __name__ == "__main__":
    main()
