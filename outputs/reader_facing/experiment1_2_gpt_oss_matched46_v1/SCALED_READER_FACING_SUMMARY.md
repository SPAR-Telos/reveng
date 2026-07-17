# Scaled Matched-46 Reader-Facing Outputs

These figures use the GPT-OSS-20B matched 46-state sentence-prefix run. Action entropy is from a single temperature 0.7 candidate-logprob distribution over UP, DOWN, LEFT, and RIGHT, not repeated sampling.

## Headline Checks

- Failure states commit at mean sentence index 129.9; control states commit at 97.3.
- Optimal-to-suboptimal events have mean action-entropy change 0.040 bits with 95% bootstrap interval [-0.023, 0.102].
- The factorisation table reports descriptive next-prefix event rates, not causal effects.

## Top Descriptive Features for Optimal-to-Suboptimal Events

| Feature | n with feature | Rate with | Rate without | Difference |
|---|---:|---:|---:|---:|
| error_wall_up | 691 | 0.036 | 0.021 | 0.015 |
| error_wall_right | 1610 | 0.025 | 0.022 | 0.003 |
| error_wall_left | 1913 | 0.020 | 0.024 | -0.003 |
| any_wall_error | 4753 | 0.020 | 0.029 | -0.009 |
| any_current_state_error | 4753 | 0.020 | 0.029 | -0.009 |

## Files

- `figs/commitment_timing_by_state_group.png`
- `figs/action_entropy_change_decision_events.png`
- `figs/belief_errors_before_after_optimality_changes.png`
- `figs/belief_entropy_around_commitment_aggregate.png`
- `figs/belief_entropy_around_commitment_by_family.png`
- `figs/belief_entropy_around_commitment_by_question.png`
- `factorization_feature_outcome_table.csv`
