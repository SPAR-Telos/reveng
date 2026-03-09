# Counterfactual Pipeline Constraints

This document defines enforceable constraints for the in-repo counterfactual pipeline.

## In Scope

- Reproducible grid-pair generation for multiple categories:
  - `goal_move`, `start_goal_swap`, `rotate_90`, `reflect_vertical`, `transpose`
- Pair-manifest driven counterfactual artifact generation:
  - `pair_manifest.json` -> `A.json`, `B.json`, `patched.json`
- Counterfactual evaluation from artifact manifest:
  - `manifest_for_counterfactual_activation_patching.json` -> result files

## Pair Manifest Constraints

Required keys per row:

- `pair_id`
- `category`
- `grid_a_path`
- `grid_b_path`
- `goal_a`
- `goal_b`

Backward compatibility:

- `goal_orig` is accepted as alias for `goal_a`
- `goal_new` is accepted as alias for `goal_b`

Accepted coordinate formats:

- `[x, y]`
- `(x, y)`
- `"x,y"`

## Grid Constraints

- Grid text must include a numeric header row and row indices.
- Allowed cell symbols are `#`, `_`, `A`, `G`.
- Both grids must parse into rectangular layouts.
- Exactly one `A` and one `G` must be present in each grid.

Category-specific constraints:

- `goal_move`
  - same topology ignoring `A`/`G`
  - same agent position between A/B
  - goal position changes
- `start_goal_swap`
  - same topology ignoring `A`/`G`
  - `A.agent == B.goal` and `A.goal == B.agent`
- `rotate_90`, `reflect_vertical`, `transpose`
  - `grid_b` must exactly match deterministic transform of `grid_a`

## Artifact and Metadata Constraints

Generated per pair:

- `A.json`
- `B.json`
- `patched.json`

Evaluation manifest row requires:

- `pair_id`
- `category`
- `grid_a_path`
- `grid_b_path`
- `goal_a` / `goal_b` (legacy aliases accepted)
- `a_trace_path`
- `b_trace_path`
- `patched_trace_path`

Patched trace metadata requirements:

- `patch_metadata.pre_reasoning_last_n == 3`
- `patch_metadata.post_reasoning_last_n == 3`
- `patch_metadata.hook_tensor == "model.layers.15.output"` (or CLI override value)

## Runtime Constraints

- If trajectory generation is enabled, Together API key must be set:
  - `TOGETHERAI_API_KEY` or `TOGETHER_API_KEY`
- Output directories are standardized as:
  - Artifacts: `data/cf/artifacts`
  - Evaluation: `data/cf/eval_results`

## Evaluator Cardinality Constraint

- `expected_k` must equal manifest row count.
- Recommended value is auto-derived as `len(manifest_rows)` during preflight.

## Reporting Constraints

- Invalid pairs and disruptive pairs are excluded from belief-action 2x2 tables.
- Invalid and disruptive counts are always reported separately.
- MLP and linear probe belief-action tables are reported separately.
- Probe agreement/disagreement metrics are reported when both probes are available.
