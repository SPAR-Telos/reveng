# Belief Changes and Action Changes During Reasoning

## Run Status

- Reasoning positions: 662
- Analysis unit: sentence
- Behavioral belief queries: 8606
- Valid parse rate: 99.9%
- Action events: 61
- Belief shifts: 785
- Ranked candidates for later interventions: 4

## Validation Gates

- Behavioral parse-rate gate: passed
- Activation-span gate: failed (step_activation_rows.csv is absent; 0 overlaps)
- Representational probe compatibility: not yet established

## Action Events

| Event type | Failure category | Events | Nearby belief shifts | Leads | Coincides | Lags |
|---|---|---:|---:|---:|---:|---:|
| action_identity_change_without_optimality_change | backtrack | 10 | 0 | 0 | 0 | 0 |
| action_identity_change_without_optimality_change | none | 24 | 0 | 0 | 0 | 0 |
| suboptimal_to_optimal_recovery | backtrack | 3 | 14 | 3 | 2 | 9 |
| suboptimal_to_optimal_recovery | none | 4 | 37 | 12 | 5 | 20 |
| suboptimal_to_optimal_recovery | short_loop | 8 | 40 | 16 | 5 | 19 |
| sustained_optimal_to_suboptimal | backtrack | 2 | 4 | 2 | 0 | 2 |
| sustained_optimal_to_suboptimal | short_loop | 1 | 10 | 4 | 1 | 5 |
| transient_optimal_to_suboptimal | backtrack | 1 | 7 | 0 | 2 | 5 |
| transient_optimal_to_suboptimal | none | 2 | 26 | 7 | 4 | 15 |
| transient_optimal_to_suboptimal | short_loop | 6 | 30 | 16 | 2 | 12 |

Feature-level associations are descriptive screening results with Benjamini-Hochberg correction. The prespecified mixed-effects model is deferred until the scaled run.
