# Belief-Action Transition Models

These models test whether behavioral belief readouts at reasoning position `k` predict whether the recommended action changes optimality at position `k+1`. Evaluation holds out complete trajectories. The constant prevalence baseline is a descriptive null. The results are predictive associations, not causal effects.

The primary analysis excludes `door_open` because its ground truth is constant in the pilot. The action-conditioned models test whether the model reports a wall in the direction it recommends moving and whether it predicts that move will hit a wall.

## Results

| Transition | Model | Trajectories | Events | AUROC | Brier | Log loss |
|---|---|---:|---:|---:|---:|---:|
| optimal to suboptimal | prevalence baseline | 14 | 19 | 0.500 | 0.042 | 0.179 |
| optimal to suboptimal | baseline | 14 | 19 | 0.321 | 0.044 | 0.199 |
| optimal to suboptimal | global beliefs | 14 | 19 | 0.327 | 0.043 | 0.197 |
| optimal to suboptimal | belief dynamics | 14 | 19 | 0.489 | 0.043 | 0.190 |
| optimal to suboptimal | global beliefs and dynamics | 14 | 19 | 0.361 | 0.043 | 0.240 |
| optimal to suboptimal | action conditioned | 14 | 19 | 0.288 | 0.045 | 0.207 |
| optimal to suboptimal | combined | 14 | 19 | 0.195 | 0.045 | 0.251 |
| optimal to suboptimal | belief action chain | 14 | 19 | 0.196 | 0.045 | 0.219 |
| suboptimal to optimal | prevalence baseline | 6 | 21 | 0.500 | 0.200 | 0.589 |
| suboptimal to optimal | baseline | 6 | 21 | 0.567 | 0.241 | 0.680 |
| suboptimal to optimal | global beliefs | 6 | 21 | 0.397 | 0.464 | 2.662 |
| suboptimal to optimal | belief dynamics | 6 | 21 | 0.583 | 0.253 | 0.711 |
| suboptimal to optimal | global beliefs and dynamics | 6 | 21 | 0.518 | 0.350 | 1.261 |
| suboptimal to optimal | action conditioned | 6 | 21 | 0.393 | 0.406 | 1.248 |
| suboptimal to optimal | combined | 6 | 21 | 0.575 | 0.276 | 0.892 |
| suboptimal to optimal | belief action chain | 6 | 21 | 0.403 | 0.409 | 1.378 |

## Interpretation

- For upcoming optimal-to-suboptimal transitions, the highest learned-model held-out AUROC is 0.489 from `belief_dynamics`.
- For recoveries, the highest learned-model held-out AUROC is 0.583 from `belief_dynamics`.
- If action-conditioned or belief-action-chain models do not outperform the baseline, the current behavioral probes do not yet provide evidence for a stable compositional belief chain explaining action changes.
- The constant prevalence null has AUROC 0.500. Compare learned models against it as well as against the reasoning-progress baseline.
- Because the pilot has few trajectories and transition events, model comparisons must be repeated on the matched and expanded cohorts.

## Model Definitions

- `prevalence_baseline`: intercept_only
- `baseline`: reasoning_progress
- `global_beliefs`: reasoning_progress, error_wall_left, error_wall_right, error_wall_up, error_wall_down, error_has_key
- `belief_dynamics`: reasoning_progress, n_global_belief_changes, n_global_error_onsets, n_global_error_recoveries
- `global_beliefs_and_dynamics`: reasoning_progress, error_wall_left, error_wall_right, error_wall_up, error_wall_down, error_has_key, n_global_belief_changes, n_global_error_onsets, n_global_error_recoveries
- `action_conditioned`: reasoning_progress, chosen_wall_error, chosen_wall_reports_blocked, chosen_effect_error, chosen_effect_reports_hit
- `combined`: reasoning_progress, error_wall_left, error_wall_right, error_wall_up, error_wall_down, error_has_key, chosen_wall_error, chosen_wall_reports_blocked, chosen_effect_error, chosen_effect_reports_hit
- `belief_action_chain`: reasoning_progress, error_wall_left, error_wall_right, error_wall_up, error_wall_down, error_has_key, chosen_wall_error, chosen_wall_reports_blocked, chosen_effect_error, chosen_effect_reports_hit, blocked_report*predicted_hit, chosen_wall_error*chosen_effect_error
