# Belief Changes and Action Changes During Reasoning

## Run Status

- Reasoning positions: 250
- Behavioral belief queries: 3250
- Valid parse rate: 96.6%
- Action events: 21
- Belief shifts: 334
- Ranked candidates for later interventions: 3

## Validation Gates

- Behavioral parse-rate gate: passed
- Activation-span gate: failed (step_activation_rows.csv is absent; 0 overlaps)
- Representational probe compatibility: not yet established

## Action Events

| Event type | Failure category | Events | Nearby belief shifts | Leads | Coincides | Lags |
|---|---|---:|---:|---:|---:|---:|
| action_identity_change_without_optimality_change | backtrack | 4 | 0 | 0 | 0 | 0 |
| action_identity_change_without_optimality_change | none | 6 | 0 | 0 | 0 | 0 |
| suboptimal_to_optimal_recovery | none | 3 | 36 | 11 | 2 | 23 |
| suboptimal_to_optimal_recovery | short_loop | 3 | 27 | 5 | 9 | 13 |
| sustained_optimal_to_suboptimal | none | 1 | 6 | 2 | 0 | 4 |
| sustained_optimal_to_suboptimal | short_loop | 1 | 5 | 3 | 1 | 1 |
| transient_optimal_to_suboptimal | none | 2 | 25 | 9 | 3 | 13 |
| transient_optimal_to_suboptimal | short_loop | 1 | 16 | 4 | 1 | 11 |

Feature-level associations are descriptive screening results with Benjamini-Hochberg correction. The prespecified mixed-effects model is deferred until the scaled run.
