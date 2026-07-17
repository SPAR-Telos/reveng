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
from reveng.experiments.behavioral_probe_alignment import (
    run_behavioral_probe_matched_eval,
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
from reveng.experiments.wall_feature_patch_experiment import (
    run_behavioral_probe_wall_feature_patch,
)
from reveng.experiments.cognitive_map_probe_reasoning_eval import (
    run_cognitive_map_probe_reasoning_eval,
)
from reveng.experiments.plan_decoder_reasoning_eval import (
    run_plan_decoder_reasoning_eval,
)
from reveng.experiments.plan_reasoning_alignment import (
    run_plan_reasoning_alignment_eval,
)
from reveng.experiments.gradual_cot_blackbox_alignment import (
    run_gradual_cot_blackbox_alignment,
)
from reveng.experiments.activation_oracle_batch import (
    preview_activation_oracle_tokens,
    run_activation_oracle_batch,
)
from reveng.experiments.activation_oracle_compare import (
    compare_activation_oracle_results,
)
from reveng.experiments.activation_oracle_paper_batches import (
    build_activation_oracle_paper_batches,
)
from reveng.experiments.activation_oracle_action_table import (
    build_activation_oracle_action_table,
)
from reveng.experiments.activation_oracle_action_plot import (
    plot_activation_oracle_action_summary,
)
from reveng.experiments.action_conditioned_belief_comparison import (
    build_action_conditioned_belief_comparison,
)
from reveng.experiments.gradual_cot_scaling_summary import (
    build_gradual_cot_scaling_summary,
)
from reveng.experiments.live_patch_curve import (
    run_live_patch_curve,
)
from reveng.experiments.step_reasoning_drift import (
    rebuild_reasoning_geometry_from_activations,
    regenerate_step_reasoning_activations,
    run_step_reasoning_drift_experiment as _run_step_reasoning_drift_experiment,
    validate_doorkey_chunking_strategy as _validate_doorkey_chunking_strategy,
)
from reveng.experiments.reasoning_belief_action import (
    prepare_reasoning_belief_action_cohorts,
    run_reasoning_belief_action_observational,
)
from reveng.experiments.intermediate_cognitive_probe_eval import (
    run_intermediate_cognitive_probe_eval,
    run_intermediate_plan_decoder_eval,
)
from reveng.experiments.position_general_belief_probe import (
    run_position_general_belief_probe_feasibility,
)
from reveng.experiments.belief_transition_indicators import (
    build_belief_transition_indicator_analysis,
)
from reveng.experiments.belief_transition_models import (
    run_belief_transition_models,
)
from reveng.experiments.experiment1_activation_monitor import (
    build_weisheng_chunk_averaged_activations,
    build_weisheng_sentence_averaged_activations,
    run_experiment1_activation_monitor,
    verify_activation_compatibility,
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
    - run_step_reasoning_drift_experiment: Run step-level prefix-action wrong-turn analysis and optional local hidden-state geometry collection
    - validate_doorkey_chunking_strategy: Validate canonical sentence-level chunking over DoorKey trajectory traces
    - regenerate_step_reasoning_activations: Regenerate non-overlapping step activations without rerunning behavioral queries
    - rebuild_reasoning_geometry_from_activations: Recompute geometry tables from corrected step activations
    - prepare_reasoning_belief_action_cohorts: Build pilot-supporting exact-matched and expanded observational cohorts
    - run_reasoning_belief_action_observational: Query prefix-conditioned beliefs and analyze belief/action transitions
    - run_intermediate_cognitive_probe_eval: Exploratorily validate released cognitive-map probes at reasoning boundaries
    - run_intermediate_plan_decoder_eval: Exploratorily validate released plan decoders at reasoning boundaries
    - run_position_general_belief_probe_feasibility: Test train-once probes across positions with held-out states
    - build_belief_transition_indicator_analysis: Summarize leading, lagging, and chained belief changes
    - run_belief_transition_models: Compare trajectory-held-out additive and interaction transition models
    - run_experiment1_activation_monitor: Build Experiment 1 activation-monitor outputs under outputs/ with chunk/activation compatibility checks
    - verify_activation_compatibility: Check whether an activation index is usable with the canonical DoorKey chunks
    - build_weisheng_chunk_averaged_activations: Map Weisheng's stride-2 packed BF16 activations onto canonical chunk-level means
    - run_behavioral_probe_smoke_test: Run a black-box DoorKey behavioral-probe smoke test on manual 9x9 states
    - run_behavioral_probe_door_semantics_ablation: Run a focused semantics ablation for the door_open_after_right behavioral probe
    - run_behavioral_probe_prompt_ablation: Compare prompt presets on a chosen behavioral-probe subset
    - run_behavioral_probe_matched_eval: Run black-box probes on white-box matched state snapshots and write a unified comparison table
    - merge_behavioral_probe_outputs: Merge multiple behavioral-probe output dirs into one combined report
    - run_behavioral_probe_trajectory_eval: Mine trajectory-derived single-step probe instances from trace-viewer JSONs
    - run_behavioral_probe_case_studies: Run probe families on mined wall-hit / non-optimal trajectory slices
    - run_behavioral_probe_wall_feature_patch: Build the minimal wall-feature patch experiment scaffold and join patched outputs when available
    - run_cognitive_map_probe_reasoning_eval: Download released public cognitive probes, verify probe/activation/trajectory compatibility, and evaluate a released pre/post slice
    - run_plan_decoder_reasoning_eval: Download published plan decoders, validate activation rows, and evaluate pre/post plan decodability when real activations are available
    - run_plan_reasoning_alignment_eval: Run the activation-side plan decoder and a textual planning counterpart on the same state x split keys
    - run_gradual_cot_blackbox_alignment: Replace BB_pre/BB_post with a revealed-CoT black-box axis on the released clean slice, the narrow failure slice, or broader trajectory-candidate failure/control slices
    - preview_activation_oracle_tokens: Preview one activation-oracle batch row with explicit token/segment ranges
    - run_activation_oracle_batch: Run Qwen3-8B activation-oracle batch queries over DoorKey-style prompts
    - compare_activation_oracle_results: Flatten AO outputs into a comparison table against behavioral labels / ground truth
    - build_activation_oracle_paper_batches: Generate AO batch JSONL files from the exact gradual-CoT paper slice rows and trajectories
    - build_activation_oracle_action_table: Summarize clean/failure AO action readouts into one reveal-level table
    - plot_activation_oracle_action_summary: Plot the merged AO clean/failure action comparison with a minimalist two-panel layout
    - build_action_conditioned_belief_comparison: Summarize which clean-slice belief readouts make the chosen action look locally consistent
    - build_gradual_cot_scaling_summary: Write a clarified summary table for the scaled revealed-CoT runs with explicit metric names
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
            "run_step_reasoning_drift_experiment": run_step_reasoning_drift_experiment,
            "validate_doorkey_chunking_strategy": validate_doorkey_chunking_strategy,
            "regenerate_step_reasoning_activations": regenerate_step_reasoning_activations,
            "rebuild_reasoning_geometry_from_activations": rebuild_reasoning_geometry_from_activations,
            "prepare_reasoning_belief_action_cohorts": prepare_reasoning_belief_action_cohorts,
            "run_reasoning_belief_action_observational": run_reasoning_belief_action_observational,
            "run_intermediate_cognitive_probe_eval": run_intermediate_cognitive_probe_eval,
            "run_intermediate_plan_decoder_eval": run_intermediate_plan_decoder_eval,
            "run_position_general_belief_probe_feasibility": run_position_general_belief_probe_feasibility,
            "build_belief_transition_indicator_analysis": build_belief_transition_indicator_analysis,
            "run_belief_transition_models": run_belief_transition_models,
            "run_experiment1_activation_monitor": run_experiment1_activation_monitor,
            "verify_activation_compatibility": verify_activation_compatibility,
            "build_weisheng_chunk_averaged_activations": build_weisheng_chunk_averaged_activations,
            "build_weisheng_sentence_averaged_activations": build_weisheng_sentence_averaged_activations,
            "run_behavioral_probe_smoke_test": run_behavioral_probe_smoke_test,
            "run_behavioral_probe_door_semantics_ablation": run_behavioral_probe_door_semantics_ablation,
            "run_behavioral_probe_prompt_ablation": run_behavioral_probe_prompt_ablation,
            "run_behavioral_probe_matched_eval": run_behavioral_probe_matched_eval,
            "merge_behavioral_probe_outputs": merge_behavioral_probe_outputs,
            "run_behavioral_probe_trajectory_eval": run_behavioral_probe_trajectory_eval,
            "run_behavioral_probe_case_studies": run_behavioral_probe_case_studies,
            "run_behavioral_probe_wall_feature_patch": run_behavioral_probe_wall_feature_patch,
            "run_cognitive_map_probe_reasoning_eval": run_cognitive_map_probe_reasoning_eval,
            "run_plan_decoder_reasoning_eval": run_plan_decoder_reasoning_eval,
            "run_plan_reasoning_alignment_eval": run_plan_reasoning_alignment_eval,
            "run_gradual_cot_blackbox_alignment": run_gradual_cot_blackbox_alignment,
            "preview_activation_oracle_tokens": preview_activation_oracle_tokens,
            "run_activation_oracle_batch": run_activation_oracle_batch,
            "compare_activation_oracle_results": compare_activation_oracle_results,
            "build_activation_oracle_paper_batches": build_activation_oracle_paper_batches,
            "build_activation_oracle_action_table": build_activation_oracle_action_table,
            "plot_activation_oracle_action_summary": plot_activation_oracle_action_summary,
            "build_action_conditioned_belief_comparison": build_action_conditioned_belief_comparison,
            "build_gradual_cot_scaling_summary": build_gradual_cot_scaling_summary,
        }
    )


def run_step_reasoning_drift_experiment(
    candidate_rows_path: str = "data/behavioral_probes/trajectory_instances_recomputed_optimal/trajectory_selection_candidates.csv",
    trajectory_dir: str = "data/hf/trajectories_key_door_100/trajectories_key_door",
    output_dir: str = "data/behavioral_probes/step_reasoning_drift",
    model_name: str = "together_ai/openai/gpt-oss-20b",
    slice_mode: str = "focused_failure",
    trajectory_slice_type: str = "short_loop",
    balanced_failure_rows_per_mode: int = 2,
    non_failure_control_rows: int = 8,
    max_examples: int | None = None,
    max_wall_hit: int = 4,
    max_avoidable_detour: int = 4,
    max_backtrack: int = 2,
    max_baseline: int = 4,
    prompt_preset: str = "cardinal_action_explicit",
    segmentation_mode: str = "paragraph_or_sentence",
    max_reasoning_steps: int | None = None,
    analysis_unit: str = "packed_chunk",
    sentence_boundaries_path: str = "data/behavioral_probes/doorkey_chunking_validation/sentences.csv",
    persistence_threshold: float = 0.5,
    collect_activations: bool = False,
    activation_source: str = "local",
    activation_prompt_mode: str = "revealed_prompt",
    local_model_name_or_path: str | None = None,
    layers: tuple[int, ...] = (15,),
    device: str = "cpu",
    device_map: str | None = None,
    torch_dtype: str = "auto",
    low_cpu_mem_usage: bool = True,
    forward_chunk_size: int = 256,
    trust_remote_code: bool = False,
    resume: bool = True,
    verbose: bool = True,
    action_mc_sample_repeats: int = 0,
    action_mc_temperature: float = 0.7,
    action_mc_max_workers: int = 1,
) -> None:
    _run_step_reasoning_drift_experiment(
        candidate_rows_path=candidate_rows_path,
        trajectory_dir=trajectory_dir,
        output_dir=output_dir,
        model_name=model_name,
        slice_mode=slice_mode,
        trajectory_slice_type=trajectory_slice_type,
        balanced_failure_rows_per_mode=balanced_failure_rows_per_mode,
        non_failure_control_rows=non_failure_control_rows,
        max_examples=max_examples,
        max_wall_hit=max_wall_hit,
        max_avoidable_detour=max_avoidable_detour,
        max_backtrack=max_backtrack,
        max_baseline=max_baseline,
        prompt_preset=prompt_preset,
        segmentation_mode=segmentation_mode,
        max_reasoning_steps=max_reasoning_steps,
        analysis_unit=analysis_unit,
        sentence_boundaries_path=sentence_boundaries_path,
        persistence_threshold=persistence_threshold,
        collect_activations=collect_activations,
        activation_source=activation_source,
        activation_prompt_mode=activation_prompt_mode,
        local_model_name_or_path=local_model_name_or_path,
        layers=layers,
        device=device,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        forward_chunk_size=forward_chunk_size,
        trust_remote_code=trust_remote_code,
        resume=resume,
        verbose=verbose,
        action_mc_sample_repeats=action_mc_sample_repeats,
        action_mc_temperature=action_mc_temperature,
        action_mc_max_workers=action_mc_max_workers,
    )


def validate_doorkey_chunking_strategy(
    trajectory_dir: str = "data/hf/trajectories_key_door_100/trajectories_key_door",
    output_dir: str = "data/behavioral_probes/doorkey_chunking_validation",
    max_analysis_chunks: int = 32,
    sample_limit: int = 25,
) -> dict:
    return _validate_doorkey_chunking_strategy(
        trajectory_dir=trajectory_dir,
        output_dir=output_dir,
        max_analysis_chunks=max_analysis_chunks,
        sample_limit=sample_limit,
    )


if __name__ == "__main__":
    main()
