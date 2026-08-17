# Descriptive Factorized Belief-Action Analysis

This is a pilot-only descriptive analysis over complete-case prefix positions. It is not a held-out regression and not causal evidence.

Complete-case positions: 653

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
| action_change_next | `key_error_and_door_error` | 4 | 0.250 | 0.092 | 0.158 |
| action_change_next | `chosen_wall_error_and_chosen_effect_error` | 39 | 0.026 | 0.098 | -0.072 |
| optimality_recovery_next | `blocked_report_and_predicted_hit` | 46 | 0.087 | 0.018 | 0.069 |
| optimality_recovery_next | `chosen_action_conflicts_with_reported_state` | 286 | 0.038 | 0.011 | 0.028 |
| optimality_recovery_next | `chosen_wall_error_and_chosen_effect_error` | 39 | 0.000 | 0.024 | -0.024 |
| optimality_recovery_next | `key_error_and_door_error` | 4 | 0.000 | 0.023 | -0.023 |
| optimality_loss_next | `blocked_report_and_predicted_hit` | 46 | 0.000 | 0.020 | -0.020 |
| optimality_loss_next | `chosen_wall_error_and_chosen_effect_error` | 39 | 0.000 | 0.020 | -0.020 |
| optimality_loss_next | `key_error_and_door_error` | 4 | 0.000 | 0.018 | -0.018 |
| action_change_next | `blocked_report_and_predicted_hit` | 46 | 0.109 | 0.092 | 0.016 |

## Interpretation

Because this run contains 8 states from one trajectory, these rows should be read as sanity checks for the factorization design. The scaled run should compare additive, action-conditioned, and factorized models with trajectory-held-out validation.
