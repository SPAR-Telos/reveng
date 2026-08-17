# Belief Changes and Action Changes During Reasoning

## Run Status

- Reasoning positions: 719
- Behavioral belief queries: 9344
- Valid parse rate: 96.5%
- Action events: 74
- Belief shifts: 1114
- Ranked candidates for later interventions: 23

## Validation Gates

- Behavioral parse-rate gate: passed
- Activation-span gate: failed (adjacent reasoning activation spans overlap; 1647 overlaps)
- Representational probe compatibility: not yet established

## Action Events

| Event type | Failure category | Events | Nearby belief shifts | Leads | Coincides | Lags |
|---|---|---:|---:|---:|---:|---:|
| action_identity_change_without_optimality_change | backtrack | 1 | 0 | 0 | 0 | 0 |
| action_identity_change_without_optimality_change | none | 10 | 0 | 0 | 0 | 0 |
| action_identity_change_without_optimality_change | oscillation_2cycle | 9 | 0 | 0 | 0 | 0 |
| suboptimal_to_optimal_recovery | avoidable_detour | 1 | 4 | 1 | 2 | 1 |
| suboptimal_to_optimal_recovery | backtrack | 7 | 67 | 23 | 14 | 30 |
| suboptimal_to_optimal_recovery | freeze_repeat | 4 | 24 | 7 | 4 | 13 |
| suboptimal_to_optimal_recovery | none | 7 | 68 | 18 | 12 | 38 |
| suboptimal_to_optimal_recovery | oscillation_2cycle | 3 | 26 | 12 | 5 | 9 |
| suboptimal_to_optimal_recovery | short_loop | 4 | 31 | 9 | 6 | 16 |
| suboptimal_to_optimal_recovery | wall_hit | 3 | 28 | 10 | 4 | 14 |
| sustained_optimal_to_suboptimal | avoidable_detour | 1 | 4 | 3 | 0 | 1 |
| sustained_optimal_to_suboptimal | backtrack | 4 | 44 | 15 | 4 | 25 |
| sustained_optimal_to_suboptimal | freeze_repeat | 1 | 1 | 0 | 0 | 1 |
| sustained_optimal_to_suboptimal | none | 2 | 24 | 11 | 6 | 7 |
| sustained_optimal_to_suboptimal | short_loop | 1 | 7 | 2 | 0 | 5 |
| sustained_optimal_to_suboptimal | wall_hit | 4 | 30 | 12 | 1 | 17 |
| transient_optimal_to_suboptimal | avoidable_detour | 1 | 4 | 1 | 0 | 3 |
| transient_optimal_to_suboptimal | backtrack | 2 | 20 | 7 | 3 | 10 |
| transient_optimal_to_suboptimal | freeze_repeat | 3 | 19 | 5 | 4 | 10 |
| transient_optimal_to_suboptimal | none | 1 | 24 | 9 | 3 | 12 |
| transient_optimal_to_suboptimal | oscillation_2cycle | 2 | 23 | 11 | 4 | 8 |
| transient_optimal_to_suboptimal | short_loop | 2 | 16 | 8 | 1 | 7 |
| transient_optimal_to_suboptimal | wall_hit | 1 | 11 | 4 | 3 | 4 |

Feature-level associations are descriptive screening results with Benjamini-Hochberg correction. The prespecified mixed-effects model is deferred until the scaled run.
