"""Command-line interface for the reveng package."""

import logging
import os
import warnings
from importlib.util import find_spec

import tyro


def _configure_runtime_preimports() -> None:
    """Apply process-level runtime guards before importing heavy subcommands."""
    # pygame emits this from an internal compatibility import at import time.
    warnings.filterwarnings(
        "ignore",
        message=r".*pkg_resources is deprecated as an API.*",
        category=UserWarning,
        module=r"pygame\.pkgdata",
    )

    # Some environments set this globally. If hf_transfer is unavailable, any
    # HF download can fail before command logic executes.
    enabled = os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "").strip().lower()
    if enabled in {"1", "true", "yes", "on"} and find_spec("hf_transfer") is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


_configure_runtime_preimports()

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
from reveng.experiments.counterfactual_expansion import (
    run_counterfactual_expansion,
)
from reveng.experiments.counterfactual_preflight import (
    validate_counterfactual_preflight,
)
from reveng.experiments.counterfactual_manifest_tools import (
    generate_counterfactual_eval_manifest,
    generate_counterfactual_grid_pairs,
    generate_counterfactual_pair_manifest,
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
    - generate_counterfactual_grid_pairs: Auto-generate reproducible A/B grid pairs
    - generate_counterfactual_pair_manifest: Auto-generate pair manifest from grid pair directories
    - generate_counterfactual_eval_manifest: Auto-generate eval manifest from pair manifest + artifacts
    - validate_counterfactual_preflight: Validate manifests/grids/runtime prerequisites before long runs
    - counterfactual_activation_patching: Evaluate belief/action counterfactual outcomes from artifacts
    - run_counterfactual_expansion: Layer sweep + threshold sensitivity with consolidated dashboard outputs
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
            "generate_counterfactual_grid_pairs": generate_counterfactual_grid_pairs,
            "generate_counterfactual_pair_manifest": generate_counterfactual_pair_manifest,
            "generate_counterfactual_eval_manifest": generate_counterfactual_eval_manifest,
            "validate_counterfactual_preflight": validate_counterfactual_preflight,
            "counterfactual_activation_patching": counterfactual_activation_patching,
            "run_counterfactual_expansion": run_counterfactual_expansion,
        }
    )


if __name__ == "__main__":
    main()
