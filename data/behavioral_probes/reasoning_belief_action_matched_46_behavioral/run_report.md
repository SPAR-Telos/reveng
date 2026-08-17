# Belief Changes and Action Changes During Reasoning

## Run Status

- Reasoning positions: 5553
- Behavioral belief queries: 72165
- Valid parse rate: 96.0%
- Action events: 1072
- Belief shifts: 7939
- Ranked candidates for later interventions: 120

## Validation Gates

- Behavioral parse-rate gate: passed
- Activation-span gate: failed (adjacent reasoning activation spans overlap; 1143 overlaps)
- Representational probe compatibility: not yet established

## Action Events

| Event type | Failure category | Events | Nearby belief shifts | Leads | Coincides | Lags |
|---|---|---:|---:|---:|---:|---:|
| action_identity_change_without_optimality_change | avoidable_detour | 5 | 0 | 0 | 0 | 0 |
| action_identity_change_without_optimality_change | backtrack | 371 | 0 | 0 | 0 | 0 |
| action_identity_change_without_optimality_change | none | 354 | 0 | 0 | 0 | 0 |
| action_identity_change_without_optimality_change | oscillation_2cycle | 9 | 0 | 0 | 0 | 0 |
| action_identity_change_without_optimality_change | short_loop | 68 | 0 | 0 | 0 | 0 |
| action_identity_change_without_optimality_change | wall_hit | 1 | 0 | 0 | 0 | 0 |
| suboptimal_to_optimal_recovery | avoidable_detour | 15 | 138 | 60 | 24 | 54 |
| suboptimal_to_optimal_recovery | backtrack | 36 | 407 | 170 | 64 | 173 |
| suboptimal_to_optimal_recovery | none | 54 | 446 | 169 | 68 | 209 |
| suboptimal_to_optimal_recovery | oscillation_2cycle | 2 | 15 | 7 | 3 | 5 |
| suboptimal_to_optimal_recovery | short_loop | 21 | 311 | 118 | 46 | 147 |
| suboptimal_to_optimal_recovery | wall_hit | 7 | 66 | 28 | 8 | 30 |
| sustained_optimal_to_suboptimal | avoidable_detour | 13 | 148 | 60 | 24 | 64 |
| sustained_optimal_to_suboptimal | backtrack | 14 | 172 | 61 | 21 | 90 |
| sustained_optimal_to_suboptimal | none | 25 | 234 | 97 | 37 | 100 |
| sustained_optimal_to_suboptimal | short_loop | 5 | 68 | 32 | 9 | 27 |
| sustained_optimal_to_suboptimal | wall_hit | 5 | 49 | 24 | 10 | 15 |
| transient_optimal_to_suboptimal | avoidable_detour | 6 | 49 | 26 | 5 | 18 |
| transient_optimal_to_suboptimal | backtrack | 21 | 231 | 97 | 36 | 98 |
| transient_optimal_to_suboptimal | none | 22 | 178 | 81 | 24 | 73 |
| transient_optimal_to_suboptimal | oscillation_2cycle | 1 | 12 | 5 | 3 | 4 |
| transient_optimal_to_suboptimal | short_loop | 15 | 230 | 93 | 28 | 109 |
| transient_optimal_to_suboptimal | wall_hit | 2 | 22 | 9 | 5 | 8 |

Feature-level associations are descriptive screening results with Benjamini-Hochberg correction.

## Matched Predictive Analysis

- Complete-case positions: 3931
- Source trajectories: 31
- Independent validation groups after keeping source trajectories and matched pairs together: 11
- Optimal-to-suboptimal prediction events: 80
- Suboptimal-to-optimal recovery events: 84

| Outcome | Best belief model | AUROC | Reasoning-progress AUROC | Paired AUROC difference, 95% CI | Calibration result |
|---|---|---:|---:|---:|---|
| Upcoming optimality loss | Combined general and action-conditioned beliefs | 0.603 | 0.569 | 0.034 [-0.013, 0.057] | No Brier-score or log-loss improvement |
| Recovery to an optimal recommendation | General belief errors | 0.758 | 0.666 | 0.093 [-0.031, 0.142] | Brier and log loss improve; paired intervals exclude zero |

The matched analysis does not establish that measured beliefs provide reliable
advance warning of optimality loss. General belief errors do contain useful
information about recovery, but action-conditioned features and the
task-motivated conjunctions add no benefit. The most common ordered
belief-change chain appears near only three events, below the prespecified
threshold of five independent trajectories.
