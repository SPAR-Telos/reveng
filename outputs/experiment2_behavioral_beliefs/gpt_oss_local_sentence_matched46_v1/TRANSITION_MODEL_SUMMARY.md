# Belief-Action Transition Models

These models test whether behavioral belief readouts at reasoning position `k` predict whether the recommended action changes optimality at position `k+1`. Evaluation holds out complete source trajectories and keeps exact-matched pairs in the same fold. Confidence intervals bootstrap those validation groups. The constant prevalence baseline is a descriptive null. The results are predictive associations, not causal effects.

The primary analysis excludes `door_open` because its ground truth is constant in the analyzed cohort. The action-conditioned models test whether the model reports a wall in the direction it recommends moving and whether it predicts that move will hit a wall.

## Results

| Transition | Model | Trajectories | Events | AUROC [95% CI] | Brier [95% CI] | Log loss [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| optimal to suboptimal | prevalence baseline | 31 | 160 | 0.500 [0.500, 0.500] | 0.029 [0.017, 0.039] | 0.134 [0.089, 0.170] |
| optimal to suboptimal | baseline | 31 | 160 | 0.603 [0.527, 0.710] | 0.029 [0.017, 0.038] | 0.132 [0.089, 0.169] |
| optimal to suboptimal | global beliefs | 31 | 160 | 0.553 [0.452, 0.706] | 0.029 [0.017, 0.039] | 0.135 [0.089, 0.175] |
| optimal to suboptimal | belief dynamics | 31 | 160 | 0.600 [0.525, 0.704] | 0.029 [0.017, 0.038] | 0.132 [0.089, 0.170] |
| optimal to suboptimal | global beliefs and dynamics | 31 | 160 | 0.552 [0.454, 0.706] | 0.029 [0.017, 0.039] | 0.135 [0.089, 0.174] |
| optimal to suboptimal | action conditioned | 31 | 160 | 0.607 [0.532, 0.711] | 0.029 [0.017, 0.039] | 0.134 [0.094, 0.171] |
| optimal to suboptimal | combined | 31 | 160 | 0.560 [0.462, 0.713] | 0.029 [0.017, 0.039] | 0.135 [0.088, 0.175] |
| optimal to suboptimal | belief action chain | 31 | 160 | 0.560 [0.462, 0.713] | 0.029 [0.017, 0.039] | 0.135 [0.088, 0.175] |
| suboptimal to optimal | prevalence baseline | 23 | 173 | 0.500 [0.500, 0.500] | 0.096 [0.075, 0.113] | 0.341 [0.284, 0.388] |
| suboptimal to optimal | baseline | 23 | 173 | 0.384 [0.257, 0.447] | 0.097 [0.076, 0.114] | 0.346 [0.288, 0.395] |
| suboptimal to optimal | global beliefs | 23 | 173 | 0.473 [0.331, 0.528] | 0.099 [0.081, 0.117] | 0.357 [0.309, 0.404] |
| suboptimal to optimal | belief dynamics | 23 | 173 | 0.375 [0.259, 0.435] | 0.097 [0.076, 0.115] | 0.348 [0.290, 0.398] |
| suboptimal to optimal | global beliefs and dynamics | 23 | 173 | 0.472 [0.340, 0.529] | 0.100 [0.081, 0.117] | 0.359 [0.310, 0.406] |
| suboptimal to optimal | action conditioned | 23 | 173 | 0.370 [0.298, 0.432] | 0.098 [0.077, 0.115] | 0.354 [0.293, 0.400] |
| suboptimal to optimal | combined | 23 | 173 | 0.447 [0.329, 0.508] | 0.100 [0.082, 0.117] | 0.358 [0.309, 0.404] |
| suboptimal to optimal | belief action chain | 23 | 173 | 0.447 [0.329, 0.509] | 0.100 [0.082, 0.117] | 0.358 [0.310, 0.404] |

## Interpretation

- For upcoming optimal-to-suboptimal transitions, the highest learned-model held-out AUROC is 0.607 from `action_conditioned`.
- For recoveries, the highest learned-model held-out AUROC is 0.473 from `global_beliefs`.
- If action-conditioned or belief-action-chain models do not outperform the baseline, the current behavioral probes do not yet provide evidence for a stable compositional belief chain explaining action changes.
- The constant prevalence null has AUROC 0.500. Compare learned models against it as well as against the reasoning-progress baseline.
- Confidence intervals are grouped-bootstrap intervals and can remain wide when few independent validation groups contain transition events.

## Paired Comparisons With Reasoning Progress

Differences below are model minus reasoning-progress baseline. Positive AUROC is better; negative Brier score and log loss are better.

| Transition | Model | AUROC difference [95% CI] | Brier difference [95% CI] | Log loss difference [95% CI] |
|---|---|---:|---:|---:|
| optimal to suboptimal | global beliefs | -0.051 [-0.112, 0.032] | 0.000 [-0.000, 0.000] | 0.003 [-0.001, 0.007] |
| optimal to suboptimal | belief dynamics | -0.003 [-0.012, 0.003] | 0.000 [-0.000, 0.000] | 0.000 [-0.000, 0.000] |
| optimal to suboptimal | global beliefs and dynamics | -0.052 [-0.112, 0.032] | 0.000 [-0.000, 0.000] | 0.003 [-0.002, 0.007] |
| optimal to suboptimal | action conditioned | 0.004 [-0.001, 0.008] | 0.000 [-0.000, 0.001] | 0.001 [-0.001, 0.005] |
| optimal to suboptimal | combined | -0.043 [-0.103, 0.038] | 0.000 [-0.000, 0.001] | 0.002 [-0.002, 0.008] |
| optimal to suboptimal | belief action chain | -0.043 [-0.103, 0.037] | 0.000 [-0.000, 0.001] | 0.002 [-0.002, 0.008] |
| suboptimal to optimal | global beliefs | 0.089 [0.025, 0.128] | 0.003 [0.000, 0.008] | 0.011 [0.004, 0.026] |
| suboptimal to optimal | belief dynamics | -0.009 [-0.021, 0.004] | 0.000 [-0.000, 0.001] | 0.002 [-0.000, 0.008] |
| suboptimal to optimal | global beliefs and dynamics | 0.088 [0.029, 0.126] | 0.003 [0.001, 0.008] | 0.013 [0.006, 0.028] |
| suboptimal to optimal | action conditioned | -0.014 [-0.047, 0.066] | 0.002 [-0.000, 0.002] | 0.008 [-0.001, 0.012] |
| suboptimal to optimal | combined | 0.063 [-0.010, 0.111] | 0.003 [0.001, 0.008] | 0.012 [0.004, 0.027] |
| suboptimal to optimal | belief action chain | 0.063 [-0.009, 0.110] | 0.003 [0.001, 0.008] | 0.012 [0.004, 0.026] |

## Model Definitions

- `prevalence_baseline`: intercept_only
- `baseline`: reasoning_progress
- `global_beliefs`: reasoning_progress, error_wall_left, error_wall_right, error_wall_up, error_wall_down, error_has_key
- `belief_dynamics`: reasoning_progress, n_global_belief_changes, n_global_error_onsets, n_global_error_recoveries
- `global_beliefs_and_dynamics`: reasoning_progress, error_wall_left, error_wall_right, error_wall_up, error_wall_down, error_has_key, n_global_belief_changes, n_global_error_onsets, n_global_error_recoveries
- `action_conditioned`: reasoning_progress, chosen_wall_error, chosen_wall_reports_blocked, chosen_effect_error, chosen_effect_reports_hit
- `combined`: reasoning_progress, error_wall_left, error_wall_right, error_wall_up, error_wall_down, error_has_key, chosen_wall_error, chosen_wall_reports_blocked, chosen_effect_error, chosen_effect_reports_hit
- `belief_action_chain`: reasoning_progress, error_wall_left, error_wall_right, error_wall_up, error_wall_down, error_has_key, chosen_wall_error, chosen_wall_reports_blocked, chosen_effect_error, chosen_effect_reports_hit, blocked_report*predicted_hit, chosen_wall_error*chosen_effect_error
