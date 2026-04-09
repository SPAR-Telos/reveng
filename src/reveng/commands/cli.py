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
from reveng.experiments.counterfactual_signal_diagnostics import (
    diagnose_counterfactual_signal,
)
from reveng.experiments.behavioral_probe_runner import (
    run_behavioral_probe_door_semantics_ablation,
    run_behavioral_probe_prompt_ablation,
    run_behavioral_probe_smoke_test,
)
from reveng.experiments.behavioral_probe_merge import (
    merge_behavioral_probe_outputs,
)
from reveng.experiments.behavioral_probe_trajectory_data import (
    run_behavioral_probe_trajectory_eval,
)
from reveng.experiments.behavioral_probe_case_studies import (
    run_behavioral_probe_case_studies,
)
from reveng.experiments.live_patch_curve import (
    run_live_patch_curve,
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
    - diagnose_counterfactual_signal: Summarize why the surrogate counterfactual signal looks weak
    - run_live_patch_curve: Run live hidden-state patching on a local hookable model and write Mario-style layer curves
    - run_behavioral_probe_smoke_test: Run a black-box DoorKey behavioral-probe smoke test on manual 9x9 states
    - run_behavioral_probe_door_semantics_ablation: Run a focused semantics ablation for the door_open_after_right behavioral probe
    - run_behavioral_probe_prompt_ablation: Compare prompt presets on a chosen behavioral-probe subset
    - merge_behavioral_probe_outputs: Merge multiple behavioral-probe output dirs into one combined report
    - run_behavioral_probe_trajectory_eval: Mine trajectory-derived single-step probe instances from trace-viewer JSONs
    - run_behavioral_probe_case_studies: Run probe families on mined wall-hit / non-optimal trajectory slices
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
            "diagnose_counterfactual_signal": diagnose_counterfactual_signal,
            "run_live_patch_curve": run_live_patch_curve,
            "run_behavioral_probe_smoke_test": run_behavioral_probe_smoke_test,
            "run_behavioral_probe_door_semantics_ablation": run_behavioral_probe_door_semantics_ablation,
            "run_behavioral_probe_prompt_ablation": run_behavioral_probe_prompt_ablation,
            "merge_behavioral_probe_outputs": merge_behavioral_probe_outputs,
            "run_behavioral_probe_trajectory_eval": run_behavioral_probe_trajectory_eval,
            "run_behavioral_probe_case_studies": run_behavioral_probe_case_studies,
        }
    )


if __name__ == "__main__":
    main()
