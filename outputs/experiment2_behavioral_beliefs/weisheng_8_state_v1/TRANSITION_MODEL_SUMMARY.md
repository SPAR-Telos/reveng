# Belief-Action Transition Models

These models test whether behavioral belief readouts at reasoning position `k` predict whether the recommended action changes optimality at position `k+1`. Evaluation holds out complete source trajectories and keeps exact-matched pairs in the same fold. Confidence intervals bootstrap those validation groups. The constant prevalence baseline is a descriptive null. The results are predictive associations, not causal effects.

The primary analysis excludes `door_open` because its ground truth is constant in the analyzed cohort. The action-conditioned models test whether the model reports a wall in the direction it recommends moving and whether it predicts that move will hit a wall.

## Results

| Transition | Model | Trajectories | Events | AUROC [95% CI] | Brier [95% CI] | Log loss [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| optimal to suboptimal | prevalence baseline | 1 | 3 | 0.500 [0.500, 0.500] | 0.018 [0.018, 0.018] | 0.091 [0.091, 0.091] |
| suboptimal to optimal | prevalence baseline | 1 | 3 | 0.500 [0.500, 0.500] | 0.250 [0.250, 0.250] | 0.693 [0.693, 0.693] |

## Interpretation

- No optimal-to-suboptimal model was estimable.
- No recovery model was estimable.
- If action-conditioned or belief-action-chain models do not outperform the baseline, the current behavioral probes do not yet provide evidence for a stable compositional belief chain explaining action changes.
- The constant prevalence null has AUROC 0.500. Compare learned models against it as well as against the reasoning-progress baseline.
- Confidence intervals are grouped-bootstrap intervals and can remain wide when few independent validation groups contain transition events.

## Paired Comparisons With Reasoning Progress

Differences below are model minus reasoning-progress baseline. Positive AUROC is better; negative Brier score and log loss are better.

| Transition | Model | AUROC difference [95% CI] | Brier difference [95% CI] | Log loss difference [95% CI] |
|---|---|---:|---:|---:|

## Model Definitions

- `prevalence_baseline`: intercept_only
- `baseline`: reasoning_progress
- `global_beliefs`: reasoning_progress, error_wall_left, error_wall_right, error_wall_up, error_wall_down, error_has_key
- `belief_dynamics`: reasoning_progress, n_global_belief_changes, n_global_error_onsets, n_global_error_recoveries
- `global_beliefs_and_dynamics`: reasoning_progress, error_wall_left, error_wall_right, error_wall_up, error_wall_down, error_has_key, n_global_belief_changes, n_global_error_onsets, n_global_error_recoveries
- `action_conditioned`: reasoning_progress, chosen_wall_error, chosen_wall_reports_blocked, chosen_effect_error, chosen_effect_reports_hit
- `combined`: reasoning_progress, error_wall_left, error_wall_right, error_wall_up, error_wall_down, error_has_key, chosen_wall_error, chosen_wall_reports_blocked, chosen_effect_error, chosen_effect_reports_hit
- `belief_action_chain`: reasoning_progress, error_wall_left, error_wall_right, error_wall_up, error_wall_down, error_has_key, chosen_wall_error, chosen_wall_reports_blocked, chosen_effect_error, chosen_effect_reports_hit, blocked_report*predicted_hit, chosen_wall_error*chosen_effect_error
