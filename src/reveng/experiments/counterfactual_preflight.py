"""Preflight validation utilities for the counterfactual pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from reveng.experiments.counterfactual_activation_patching import (
    _read_manifest,
    _validate_goal_move_only as _validate_eval_goal_move_only,
)
from reveng.experiments.counterfactual_artifact_builder import (
    LAYER_KEY_DEFAULT,
    _parse_grid_text_file,
    _validate_goal_move_only as _validate_pair_goal_move_only,
)
from reveng.experiments.counterfactual_manifest_tools import (
    read_pair_manifest_strict,
)


def _check_api_key_presence() -> None:
    if os.getenv("TOGETHERAI_API_KEY") or os.getenv("TOGETHER_API_KEY"):
        return
    raise ValueError(
        "Missing Together API key. Set TOGETHERAI_API_KEY (or TOGETHER_API_KEY) before trajectory generation."
    )


def _check_eval_manifest_rows(eval_manifest_path: Path, layer_key: str) -> int:
    records = _read_manifest(eval_manifest_path)
    for record in records:
        goal_move_only_error = _validate_eval_goal_move_only(record.spec)
        if goal_move_only_error is not None:
            raise ValueError(
                f"eval manifest pair={record.spec.pair_id} violates goal-move-only constraint: {goal_move_only_error}"
            )

        # If patched trace already exists, validate metadata now so failures are early and actionable.
        if record.artifacts.patched_trace_path.exists():
            patched = json.loads(record.artifacts.patched_trace_path.read_text())
            metadata = patched.get("patch_metadata")
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"patched trace missing patch_metadata: {record.artifacts.patched_trace_path}"
                )
            if metadata.get("pre_reasoning_last_n") != 3:
                raise ValueError(
                    f"patched trace pre_reasoning_last_n must be 3: {record.artifacts.patched_trace_path}"
                )
            if metadata.get("post_reasoning_last_n") != 3:
                raise ValueError(
                    f"patched trace post_reasoning_last_n must be 3: {record.artifacts.patched_trace_path}"
                )
            if metadata.get("hook_tensor") != layer_key:
                raise ValueError(
                    f"patched trace hook_tensor must be {layer_key}: {record.artifacts.patched_trace_path}"
                )
    return len(records)


def validate_counterfactual_preflight(
    pair_manifest_path: str = "data/cf/pair_manifest.json",
    artifacts_output_dir: str = "data/cf/artifacts",
    eval_output_dir: str = "data/cf/eval_results",
    eval_manifest_path: Optional[str] = None,
    expected_k: Optional[int] = None,
    skip_trajectory_generation: bool = False,
    require_api_key: bool = True,
    layer_key: str = LAYER_KEY_DEFAULT,
) -> dict[str, Any]:
    """Validate counterfactual pipeline prerequisites without mutating tracked files.

    This validates:
    - Pair-manifest schema and coordinate parsing.
    - Referenced grid files and goal-move-only topology constraints.
    - Runtime prerequisites for trajectory generation (API key).
    - Optional eval-manifest constraints and patched-trace metadata compatibility.
    - expected_k consistency for evaluator.
    """
    pair_manifest = Path(pair_manifest_path)
    artifacts_out = Path(artifacts_output_dir)
    eval_out = Path(eval_output_dir)

    pair_specs = read_pair_manifest_strict(pair_manifest)
    for spec in pair_specs:
        layout_a = _parse_grid_text_file(spec.grid_a_path)
        layout_b = _parse_grid_text_file(spec.grid_b_path)
        _validate_pair_goal_move_only(spec, layout_a, layout_b)

    if require_api_key and not skip_trajectory_generation:
        _check_api_key_presence()

    artifacts_out.mkdir(parents=True, exist_ok=True)
    eval_out.mkdir(parents=True, exist_ok=True)

    recommended_expected_k = len(pair_specs)
    eval_manifest_rows = None
    if eval_manifest_path is not None:
        eval_manifest_rows = _check_eval_manifest_rows(Path(eval_manifest_path), layer_key=layer_key)
        recommended_expected_k = eval_manifest_rows

    if expected_k is not None and expected_k != recommended_expected_k:
        raise ValueError(
            f"expected_k mismatch: expected_k={expected_k}, recommended_expected_k={recommended_expected_k}. "
            f"Use --expected-k {recommended_expected_k}."
        )

    summary = {
        "status": "ok",
        "pair_manifest_path": str(pair_manifest),
        "n_pair_manifest_rows": len(pair_specs),
        "eval_manifest_path": eval_manifest_path,
        "n_eval_manifest_rows": eval_manifest_rows,
        "recommended_expected_k": recommended_expected_k,
        "artifacts_output_dir": str(artifacts_out),
        "eval_output_dir": str(eval_out),
        "skip_trajectory_generation": skip_trajectory_generation,
        "require_api_key": require_api_key,
        "layer_key": layer_key,
    }
    print(json.dumps(summary, indent=2))
    return summary


__all__ = ["validate_counterfactual_preflight"]
