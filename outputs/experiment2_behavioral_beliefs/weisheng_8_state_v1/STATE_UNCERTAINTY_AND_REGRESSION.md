# State-Belief Uncertainty And Exploratory Regression

State-belief uncertainty is computed from logprob readouts for the six primary yes/no/unknown state-belief probes: wall left, wall right, wall up, wall down, key held, and door open.
For each probe, yes/no/unknown probabilities are normalized after excluding invalid mass, then Shannon entropy is computed in bits. Per-position uncertainty is the mean entropy across available primary probes.

The regression table is exploratory. It treats repeated reasoning prefixes as rows and reports model-based logistic-regression standard errors, confidence intervals, and p-values. Because this run has 8 states from one source trajectory group, these p-values are diagnostics, not a population-level inferential claim.
In the regression, state-belief entropy is scaled in units of 0.01 bits because the logprob readouts are very confident and raw entropy values are close to zero.

## Query Summary

- Uncertainty rows requested: 1500
- Successful logprob rows: 1500
- Failed logprob rows: 0
- Approximate query cost in USD: 0.4643
- Mean probe entropy in bits: 0.002
- Mean per-position state-belief entropy in bits: 0.002

## Event-Level State-Belief Uncertainty

| event type | n | mean previous entropy | mean current entropy | mean change |
|---|---:|---:|---:|---:|
| action identity change without optimality change | 10 | 0.001 | 0.000 | -0.000 |
| suboptimal to optimal recovery | 6 | 0.002 | 0.001 | -0.001 |
| sustained optimal to suboptimal | 2 | 0.000 | 0.002 | 0.001 |
| transient optimal to suboptimal | 3 | 0.005 | 0.003 | -0.001 |

## Regression Status

| model | outcome | n | events | converged | warning |
|---|---|---:|---:|---|---|
| action_change_next_main | action_change_next | 216 | 18 | True |  |
| optimality_loss_next_compact | optimality_loss_next | 216 | 4 | True | rare event model: only 4 rows in the smaller class; coefficients and p-values are unstable |
| optimality_recovery_next_compact | optimality_recovery_next | 216 | 5 | True | rare event model: only 5 rows in the smaller class; coefficients and p-values are unstable |

## Regression Coefficients

| model | term | coefficient | SE | 95% CI | p | odds ratio |
|---|---|---:|---:|---:|---:|---:|
| action_change_next_main | Intercept | -2.698 | 0.680 | [-4.031, -1.365] | 0.0001 | 0.067 |
| action_change_next_main | reasoning_progress | -3.540 | 1.558 | [-6.594, -0.486] | 0.0231 | 0.029 |
| action_change_next_main | action_mc_entropy | 2.061 | 0.631 | [0.824, 3.298] | 0.0011 | 7.852 |
| action_change_next_main | state_belief_entropy_per_0p01_bits | 0.541 | 0.460 | [-0.360, 1.443] | 0.2394 | 1.718 |
| action_change_next_main | primary_belief_error_rate | -2.171 | 2.927 | [-7.907, 3.566] | 0.4583 | 0.114 |
| action_change_next_main | chosen_action_conflicts_with_reported_state | 1.902 | 0.755 | [0.421, 3.382] | 0.0118 | 6.697 |
| optimality_loss_next_compact | Intercept | -4.196 | 1.254 | [-6.654, -1.738] | 0.0008 | 0.015 |
| optimality_loss_next_compact | reasoning_progress | -1.844 | 2.295 | [-6.343, 2.654] | 0.4217 | 0.158 |
| optimality_loss_next_compact | action_mc_entropy | 1.345 | 1.115 | [-0.840, 3.530] | 0.2276 | 3.838 |
| optimality_loss_next_compact | state_belief_entropy_per_0p01_bits | 0.313 | 0.913 | [-1.476, 2.102] | 0.7316 | 1.368 |
| optimality_loss_next_compact | primary_belief_error_rate | 2.456 | 3.547 | [-4.496, 9.408] | 0.4887 | 11.655 |
| optimality_recovery_next_compact | Intercept | -3.372 | 0.969 | [-5.272, -1.472] | 0.0005 | 0.034 |
| optimality_recovery_next_compact | reasoning_progress | -0.270 | 1.950 | [-4.092, 3.552] | 0.8898 | 0.763 |
| optimality_recovery_next_compact | action_mc_entropy | 0.420 | 1.017 | [-1.574, 2.413] | 0.6800 | 1.521 |
| optimality_recovery_next_compact | state_belief_entropy_per_0p01_bits | 0.299 | 0.518 | [-0.716, 1.315] | 0.5633 | 1.349 |
| optimality_recovery_next_compact | primary_belief_error_rate | -7.017 | 6.518 | [-19.793, 5.758] | 0.2817 | 0.001 |
