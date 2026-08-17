# Belief-Action Transition Models

These models test whether behavioral belief readouts at reasoning position `k` predict whether the recommended action changes optimality at position `k+1`. Evaluation holds out complete source trajectories and keeps exact-matched pairs in the same fold. Confidence intervals bootstrap those validation groups. The constant prevalence baseline is a descriptive null. The results are predictive associations, not causal effects.

The primary analysis excludes `door_open` because its ground truth is constant in the analyzed cohort. The action-conditioned models test whether the model reports a wall in the direction it recommends moving and whether it predicts that move will hit a wall.

## Results

| Transition | Model | Trajectories | Events | AUROC [95% CI] | Brier [95% CI] | Log loss [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| optimal to suboptimal | prevalence baseline | 31 | 80 | 0.500 [0.500, 0.500] | 0.023 [0.013, 0.035] | 0.111 [0.072, 0.157] |
| optimal to suboptimal | baseline | 31 | 80 | 0.569 [0.537, 0.637] | 0.023 [0.013, 0.034] | 0.110 [0.072, 0.156] |
| optimal to suboptimal | global beliefs | 31 | 80 | 0.597 [0.557, 0.662] | 0.023 [0.013, 0.035] | 0.111 [0.073, 0.155] |
| optimal to suboptimal | belief dynamics | 31 | 80 | 0.574 [0.531, 0.657] | 0.023 [0.013, 0.034] | 0.110 [0.072, 0.157] |
| optimal to suboptimal | global beliefs and dynamics | 31 | 80 | 0.567 [0.521, 0.649] | 0.023 [0.013, 0.035] | 0.111 [0.073, 0.158] |
| optimal to suboptimal | action conditioned | 31 | 80 | 0.565 [0.531, 0.631] | 0.023 [0.013, 0.034] | 0.111 [0.073, 0.156] |
| optimal to suboptimal | combined | 31 | 80 | 0.603 [0.556, 0.663] | 0.023 [0.013, 0.034] | 0.111 [0.074, 0.156] |
| optimal to suboptimal | belief action chain | 31 | 80 | 0.602 [0.557, 0.661] | 0.023 [0.013, 0.035] | 0.111 [0.074, 0.156] |
| suboptimal to optimal | prevalence baseline | 17 | 84 | 0.500 [0.500, 0.500] | 0.139 [0.114, 0.297] | 0.451 [0.391, 0.832] |
| suboptimal to optimal | baseline | 17 | 84 | 0.666 [0.590, 0.838] | 0.142 [0.114, 0.275] | 0.447 [0.368, 0.785] |
| suboptimal to optimal | global beliefs | 17 | 84 | 0.758 [0.681, 0.828] | 0.115 [0.095, 0.244] | 0.395 [0.347, 0.690] |
| suboptimal to optimal | belief dynamics | 17 | 84 | 0.683 [0.617, 0.844] | 0.141 [0.112, 0.270] | 0.444 [0.362, 0.767] |
| suboptimal to optimal | global beliefs and dynamics | 17 | 84 | 0.750 [0.688, 0.829] | 0.120 [0.101, 0.242] | 0.406 [0.354, 0.689] |
| suboptimal to optimal | action conditioned | 17 | 84 | 0.659 [0.570, 0.838] | 0.153 [0.111, 0.269] | 0.472 [0.360, 0.772] |
| suboptimal to optimal | combined | 17 | 84 | 0.736 [0.674, 0.818] | 0.122 [0.102, 0.248] | 0.409 [0.356, 0.711] |
| suboptimal to optimal | belief action chain | 17 | 84 | 0.732 [0.673, 0.818] | 0.123 [0.103, 0.249] | 0.415 [0.359, 0.717] |

## Interpretation

- For upcoming optimal-to-suboptimal transitions, the highest learned-model held-out AUROC is 0.603 from `combined`.
- For recoveries, the highest learned-model held-out AUROC is 0.758 from `global_beliefs`.
- If action-conditioned or belief-action-chain models do not outperform the baseline, the current behavioral probes do not yet provide evidence for a stable compositional belief chain explaining action changes.
- The constant prevalence null has AUROC 0.500. Compare learned models against it as well as against the reasoning-progress baseline.
- Confidence intervals are grouped-bootstrap intervals and can remain wide when few independent validation groups contain transition events.

## Paired Comparisons With Reasoning Progress

Differences below are model minus reasoning-progress baseline. Positive AUROC is better; negative Brier score and log loss are better.

| Transition | Model | AUROC difference [95% CI] | Brier difference [95% CI] | Log loss difference [95% CI] |
|---|---|---:|---:|---:|
| optimal to suboptimal | global beliefs | 0.027 [-0.017, 0.050] | 0.000 [-0.000, 0.000] | 0.000 [-0.001, 0.001] |
| optimal to suboptimal | belief dynamics | 0.005 [-0.031, 0.052] | 0.000 [-0.000, 0.000] | -0.000 [-0.001, 0.001] |
| optimal to suboptimal | global beliefs and dynamics | -0.003 [-0.044, 0.043] | 0.000 [-0.000, 0.000] | 0.001 [-0.000, 0.003] |
| optimal to suboptimal | action conditioned | -0.004 [-0.017, 0.006] | 0.000 [-0.000, 0.000] | 0.001 [-0.000, 0.001] |
| optimal to suboptimal | combined | 0.034 [-0.013, 0.057] | 0.000 [-0.000, 0.000] | 0.001 [-0.001, 0.002] |
| optimal to suboptimal | belief action chain | 0.033 [-0.014, 0.057] | 0.000 [0.000, 0.000] | 0.001 [-0.000, 0.002] |
| suboptimal to optimal | global beliefs | 0.093 [-0.031, 0.142] | -0.028 [-0.045, -0.005] | -0.052 [-0.106, -0.004] |
| suboptimal to optimal | belief dynamics | 0.017 [-0.001, 0.036] | -0.001 [-0.006, 0.000] | -0.004 [-0.019, 0.001] |
| suboptimal to optimal | global beliefs and dynamics | 0.085 [-0.026, 0.124] | -0.023 [-0.040, -0.005] | -0.041 [-0.099, -0.007] |
| suboptimal to optimal | action conditioned | -0.006 [-0.037, 0.029] | 0.011 [-0.009, 0.022] | 0.025 [-0.023, 0.050] |
| suboptimal to optimal | combined | 0.070 [-0.049, 0.120] | -0.021 [-0.034, -0.002] | -0.038 [-0.079, -0.000] |
| suboptimal to optimal | belief action chain | 0.066 [-0.052, 0.110] | -0.019 [-0.032, -0.003] | -0.032 [-0.072, -0.000] |

## Model Definitions

- `prevalence_baseline`: intercept_only
- `baseline`: reasoning_progress
- `global_beliefs`: reasoning_progress, error_wall_left, error_wall_right, error_wall_up, error_wall_down, error_has_key
- `belief_dynamics`: reasoning_progress, n_global_belief_changes, n_global_error_onsets, n_global_error_recoveries
- `global_beliefs_and_dynamics`: reasoning_progress, error_wall_left, error_wall_right, error_wall_up, error_wall_down, error_has_key, n_global_belief_changes, n_global_error_onsets, n_global_error_recoveries
- `action_conditioned`: reasoning_progress, chosen_wall_error, chosen_wall_reports_blocked, chosen_effect_error, chosen_effect_reports_hit
- `combined`: reasoning_progress, error_wall_left, error_wall_right, error_wall_up, error_wall_down, error_has_key, chosen_wall_error, chosen_wall_reports_blocked, chosen_effect_error, chosen_effect_reports_hit
- `belief_action_chain`: reasoning_progress, error_wall_left, error_wall_right, error_wall_up, error_wall_down, error_has_key, chosen_wall_error, chosen_wall_reports_blocked, chosen_effect_error, chosen_effect_reports_hit, blocked_report*predicted_hit, chosen_wall_error*chosen_effect_error
