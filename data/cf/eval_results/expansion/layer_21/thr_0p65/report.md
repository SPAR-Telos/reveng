# Counterfactual Activation Patching Report

## Configuration
- layer_key: `model.layers.21.output`
- hook tensor: residual output
- patch token set: last 3 pre-reasoning + last 3 post-reasoning
- Action=True rule: A_target > A_base and A_target >= 0.65
- Disruptive rule: A_target < 0.35 and A_base < 0.35

## Metric Glossary
- `A_target` (alias `A_new`): per-pair fraction of evaluated steps where the patched action is optimal under grid B policy.
- `A_base` (alias `A_orig`): per-pair fraction of evaluated steps where the patched action is optimal under grid A policy.
- Why these are non-integers: each value is a ratio `aligned_steps / evaluated_steps`, not a raw count.
- `evaluated_steps` excludes steps where action parsing or position mapping is invalid for that pair.

## Manifest and Processing Summary
- total_pairs_manifest: 50
- total_pairs_processed: 50
- total_pairs_valid: 50
- total_pairs_invalid: 0
- stopped_early: False
- early_stop_reason: NA
- action_true_count: 7
- action_true_rate: 0.2916666666666667

## Invalid Summary
- None

## Disruptive Summary
- disruptive_count: 25
- disruptive_rate: 0.5000
- disruptive_pairs:
  - pair_goal_move_001: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_goal_move_002: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_goal_move_004: A_target=0.3333 and A_base=0.3333 are both below disruptive_threshold=0.35
  - pair_goal_move_005: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_goal_move_008: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_reflect_vertical_001: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_reflect_vertical_003: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_reflect_vertical_004: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_reflect_vertical_005: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_reflect_vertical_006: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_rotate_90_000: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_rotate_90_001: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_rotate_90_002: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_rotate_90_003: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_rotate_90_004: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_rotate_90_005: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_rotate_90_006: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_rotate_90_009: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_start_goal_swap_000: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_start_goal_swap_005: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_transpose_003: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_transpose_005: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_transpose_006: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_transpose_007: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35
  - pair_transpose_009: A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35

## Action Summary by Category
| category | total | valid | invalid | disruptive | action_true_rate |
|---|---:|---:|---:|---:|---:|
| goal_move | 10 | 10 | 0 | 5 | 0.6 |
| reflect_vertical | 10 | 10 | 0 | 5 | 0.0 |
| rotate_90 | 10 | 10 | 0 | 8 | 0.0 |
| start_goal_swap | 10 | 10 | 0 | 2 | 0.5714285714285714 |
| transpose | 10 | 10 | 0 | 5 | 0.0 |

## MLP Belief-Action 2x2
- denominator (table_rows_mlp): 24
| Belief (MLP) \ Action | True | False |
|---|---:|---:|
| True | 7 | 17 |
| False | 0 | 0 |
- TT/(TT+TF): 0.2916666666666667
- TT/(TT+FT): 1.0

## Linear Belief-Action 2x2
- denominator (table_rows_linear): 24
| Belief (Linear) \ Action | True | False |
|---|---:|---:|
| True | 0 | 3 |
| False | 7 | 14 |
- TT/(TT+TF): 0.0
- TT/(TT+FT): 0.0

## Probe Agreement/Disagreement
- probe_both_available: 24
- probe_agree_count: 3
- probe_disagree_count: 21
- probe_disagreement_rate: 0.875

## Pair Results

| pair_id | category | valid | A_target | A_base | action | disruptive | disruptive_reason | belief_mlp | belief_linear | avail_mlp | avail_linear | invalid_reason |
|---|---|---|---:|---:|---|---|---|---|---|---|---|---|
| pair_goal_move_000 | goal_move | True | 1.0000 | 0.0000 | True | False |  | True | False | True | True |  |
| pair_goal_move_001 | goal_move | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_goal_move_002 | goal_move | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_goal_move_003 | goal_move | True | 1.0000 | 1.0000 | False | False |  | True | False | True | True |  |
| pair_goal_move_004 | goal_move | True | 0.3333 | 0.3333 | False | True | A_target=0.3333 and A_base=0.3333 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_goal_move_005 | goal_move | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_goal_move_006 | goal_move | True | 1.0000 | 0.0000 | True | False |  | True | False | True | True |  |
| pair_goal_move_007 | goal_move | True | 1.0000 | 0.5000 | True | False |  | True | False | True | True |  |
| pair_goal_move_008 | goal_move | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_goal_move_009 | goal_move | True | 0.5000 | 0.0000 | False | False |  | True | False | True | True |  |
| pair_reflect_vertical_000 | reflect_vertical | True | 1.0000 | 1.0000 | False | False |  | True | False | True | True |  |
| pair_reflect_vertical_001 | reflect_vertical | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_reflect_vertical_002 | reflect_vertical | True | 0.6000 | 0.6000 | False | False |  | True | True | True | True |  |
| pair_reflect_vertical_003 | reflect_vertical | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | True | True | True |  |
| pair_reflect_vertical_004 | reflect_vertical | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_reflect_vertical_005 | reflect_vertical | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_reflect_vertical_006 | reflect_vertical | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | True | True | True |  |
| pair_reflect_vertical_007 | reflect_vertical | True | 0.6667 | 0.6667 | False | False |  | True | False | True | True |  |
| pair_reflect_vertical_008 | reflect_vertical | True | 0.5000 | 0.5000 | False | False |  | True | False | True | True |  |
| pair_reflect_vertical_009 | reflect_vertical | True | 0.5000 | 0.5000 | False | False |  | True | False | True | True |  |
| pair_rotate_90_000 | rotate_90 | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_rotate_90_001 | rotate_90 | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_rotate_90_002 | rotate_90 | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_rotate_90_003 | rotate_90 | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_rotate_90_004 | rotate_90 | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_rotate_90_005 | rotate_90 | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_rotate_90_006 | rotate_90 | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_rotate_90_007 | rotate_90 | True | 0.5000 | 0.5000 | False | False |  | True | False | True | True |  |
| pair_rotate_90_008 | rotate_90 | True | 1.0000 | 1.0000 | False | False |  | True | False | True | True |  |
| pair_rotate_90_009 | rotate_90 | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_start_goal_swap_000 | start_goal_swap | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_start_goal_swap_001 | start_goal_swap | True | 1.0000 | 0.0000 | True | False |  | True | False | True | True |  |
| pair_start_goal_swap_002 | start_goal_swap | True | 0.6667 | 0.0000 | True | False |  | True | False | True | True |  |
| pair_start_goal_swap_003 | start_goal_swap | True | 1.0000 | 0.0000 | True | False |  | True | False | True | True |  |
| pair_start_goal_swap_004 | start_goal_swap | True | 1.0000 | 1.0000 | False | False |  | True | False | True | True |  |
| pair_start_goal_swap_005 | start_goal_swap | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_start_goal_swap_006 | start_goal_swap | True | 1.0000 | 0.0000 | True | False |  | True | False | True | True |  |
| pair_start_goal_swap_007 | start_goal_swap | True | 0.0000 | 0.0000 | None | None |  | True | False | True | True |  |
| pair_start_goal_swap_008 | start_goal_swap | True | 1.0000 | 1.0000 | False | False |  | True | False | True | True |  |
| pair_start_goal_swap_009 | start_goal_swap | True | 0.3333 | 1.0000 | False | False |  | True | False | True | True |  |
| pair_transpose_000 | transpose | True | 1.0000 | 1.0000 | False | False |  | True | True | True | True |  |
| pair_transpose_001 | transpose | True | 0.6667 | 0.6667 | False | False |  | True | False | True | True |  |
| pair_transpose_002 | transpose | True | 1.0000 | 1.0000 | False | False |  | True | False | True | True |  |
| pair_transpose_003 | transpose | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_transpose_004 | transpose | True | 0.5000 | 0.5000 | False | False |  | True | True | True | True |  |
| pair_transpose_005 | transpose | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_transpose_006 | transpose | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_transpose_007 | transpose | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
| pair_transpose_008 | transpose | True | 1.0000 | 1.0000 | False | False |  | True | False | True | True |  |
| pair_transpose_009 | transpose | True | 0.0000 | 0.0000 | False | True | A_target=0.0000 and A_base=0.0000 are both below disruptive_threshold=0.35 | True | False | True | True |  |
