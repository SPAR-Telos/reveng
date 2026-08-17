# Belief Changes and Action Changes During Reasoning

## Run Status

- Reasoning positions: 257
- Analysis unit: packed_chunk
- Behavioral belief queries: 3341
- Valid parse rate: 99.9%
- Action events: 26
- Belief shifts: 321
- Ranked candidates for later interventions: 1

## Validation Gates

- Behavioral parse-rate gate: passed
- Activation-span gate: failed (step_activation_rows.csv is absent; 0 overlaps)
- Representational probe compatibility: not yet established

## Action Events

| Event type | Failure category | Events | Nearby belief shifts | Leads | Coincides | Lags |
|---|---|---:|---:|---:|---:|---:|
| action_identity_change_without_optimality_change | backtrack | 6 | 0 | 0 | 0 | 0 |
| action_identity_change_without_optimality_change | none | 5 | 0 | 0 | 0 | 0 |
| suboptimal_to_optimal_recovery | backtrack | 2 | 11 | 4 | 3 | 4 |
| suboptimal_to_optimal_recovery | none | 4 | 22 | 8 | 5 | 9 |
| suboptimal_to_optimal_recovery | short_loop | 3 | 20 | 8 | 5 | 7 |
| sustained_optimal_to_suboptimal | backtrack | 1 | 6 | 3 | 0 | 3 |
| transient_optimal_to_suboptimal | backtrack | 1 | 5 | 3 | 0 | 2 |
| transient_optimal_to_suboptimal | none | 2 | 19 | 5 | 4 | 10 |
| transient_optimal_to_suboptimal | short_loop | 2 | 19 | 8 | 4 | 7 |

Feature-level associations are descriptive screening results with Benjamini-Hochberg correction. The prespecified mixed-effects model is deferred until the scaled run.
