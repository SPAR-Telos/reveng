# Descriptive Factorized Belief-Action Analysis

This is a pilot-only descriptive analysis over complete-case prefix positions. It is not a held-out regression and not causal evidence.

Complete-case positions: 249

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
| action_change_next | `blocked_report_and_predicted_hit` | 18 | 0.000 | 0.113 | -0.113 |
| action_change_next | `chosen_wall_error_and_chosen_effect_error` | 18 | 0.000 | 0.113 | -0.113 |
| action_change_next | `key_error_and_door_error` | 3 | 0.000 | 0.106 | -0.106 |
| action_change_next | `chosen_action_conflicts_with_reported_state` | 101 | 0.059 | 0.135 | -0.076 |
| optimality_recovery_next | `blocked_report_and_predicted_hit` | 18 | 0.000 | 0.039 | -0.039 |
| optimality_recovery_next | `chosen_wall_error_and_chosen_effect_error` | 18 | 0.000 | 0.039 | -0.039 |
| optimality_recovery_next | `key_error_and_door_error` | 3 | 0.000 | 0.037 | -0.037 |
| optimality_loss_next | `blocked_report_and_predicted_hit` | 18 | 0.000 | 0.026 | -0.026 |
| optimality_loss_next | `chosen_wall_error_and_chosen_effect_error` | 18 | 0.000 | 0.026 | -0.026 |
| optimality_loss_next | `key_error_and_door_error` | 3 | 0.000 | 0.024 | -0.024 |

## Interpretation

Because this run contains 8 states from one trajectory, these rows should be read as sanity checks for the factorization design. The scaled run should compare additive, action-conditioned, and factorized models with trajectory-held-out validation.
