# Descriptive Factorized Belief-Action Analysis

This is a pilot-only descriptive analysis over complete-case prefix positions. It is not a held-out regression and not causal evidence.

Complete-case positions: 216

## Features

| Feature | Definition |
|---|---|
| `blocked_report_and_predicted_hit` | Chosen-direction wall report is yes AND chosen-action hit-wall report is yes. |
| `chosen_wall_error_and_chosen_effect_error` | Chosen-direction wall belief is wrong AND chosen-action hit-wall consequence belief is wrong. |
| `key_error_and_door_error` | Key-possession belief is wrong AND door-open belief is wrong. |
| `chosen_action_conflicts_with_reported_state` | Chosen-direction wall report is yes OR chosen-action hit-wall report is yes. |

## Largest Descriptive Risk Differences

| Outcome | Feature | n with feature | Rate with | Rate without | Difference |
|---|---|---:|---:|---:|---:|
| optimality_recovery_next | `blocked_report_and_predicted_hit` | 6 | 0.167 | 0.019 | 0.148 |
| action_change_next | `blocked_report_and_predicted_hit` | 6 | 0.167 | 0.081 | 0.086 |
| action_change_next | `chosen_wall_error_and_chosen_effect_error` | 2 | 0.000 | 0.084 | -0.084 |
| optimality_recovery_next | `chosen_action_conflicts_with_reported_state` | 66 | 0.076 | 0.000 | 0.076 |
| action_change_next | `chosen_action_conflicts_with_reported_state` | 66 | 0.121 | 0.067 | 0.055 |
| optimality_loss_next | `chosen_action_conflicts_with_reported_state` | 66 | 0.045 | 0.007 | 0.039 |
| optimality_recovery_next | `chosen_wall_error_and_chosen_effect_error` | 2 | 0.000 | 0.023 | -0.023 |
| optimality_loss_next | `blocked_report_and_predicted_hit` | 6 | 0.000 | 0.019 | -0.019 |
| optimality_loss_next | `chosen_wall_error_and_chosen_effect_error` | 2 | 0.000 | 0.019 | -0.019 |

## Interpretation

Because this run contains 8 states from one trajectory, these rows should be read as sanity checks for the factorization design. The scaled run should compare additive, action-conditioned, and factorized models with trajectory-held-out validation.
