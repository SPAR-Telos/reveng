# State-Belief Uncertainty And Exploratory Regression

State-belief uncertainty is computed from logprob readouts for the six primary yes/no/unknown state-belief probes: wall left, wall right, wall up, wall down, key held, and door open.
For each probe, yes/no/unknown probabilities are normalized after excluding invalid mass, then Shannon entropy is computed in bits. Per-position uncertainty is the mean entropy across available primary probes.
The label logprobs were requested at temperature 0.7.

The regression table is exploratory. It treats repeated reasoning prefixes as rows and reports model-based logistic-regression standard errors, confidence intervals, and p-values. Because this run has 8 states from one source trajectory group, these p-values are diagnostics, not a population-level inferential claim.
In the regression, state-belief entropy is scaled in units of 0.01 bits because the logprob readouts are very confident and raw entropy values are close to zero.

## Query Summary

- Uncertainty rows requested: 1542
- Successful logprob rows: 1542
- Failed logprob rows: 0
- Approximate query cost in USD: 0.0000
- Mean probe entropy in bits: 0.002
- Mean per-position state-belief entropy in bits: 0.002

## Event-Level State-Belief Uncertainty

| event type | n | mean previous entropy | mean current entropy | mean change |
|---|---:|---:|---:|---:|
| action identity change without optimality change | 11 | 0.000 | 0.000 | -0.000 |
| suboptimal to optimal recovery | 9 | 0.000 | 0.000 | 0.000 |
| sustained optimal to suboptimal | 1 | 0.000 | 0.000 | 0.000 |
| transient optimal to suboptimal | 5 | 0.000 | 0.000 | -0.000 |

## Regression Status

| model | outcome | n | events | converged | warning |
|---|---|---:|---:|---|---|
| action_change_next_main | action_change_next | 249 | 26 | True |  |
| optimality_loss_next_compact | optimality_loss_next | 249 | 6 | True | rare event model: only 6 rows in the smaller class; coefficients and p-values are unstable |
| optimality_recovery_next_compact | optimality_recovery_next | 249 | 9 | True | rare event model: only 9 rows in the smaller class; coefficients and p-values are unstable; near-perfect fitted probabilities; possible separation |

## Regression Coefficients

| model | term | coefficient | SE | 95% CI | p | odds ratio |
|---|---|---:|---:|---:|---:|---:|
| action_change_next_main | Intercept | -0.564 | 0.350 | [-1.250, 0.122] | 0.1069 | 0.569 |
| action_change_next_main | reasoning_progress | -4.532 | 1.261 | [-7.003, -2.061] | 0.0003 | 0.011 |
| action_change_next_main | action_entropy_bits | -208009.315 | 394654.398 | [-981531.936, 565513.305] | 0.5981 |  |
| action_change_next_main | state_belief_entropy_per_0p01_bits | -1.431 | 3.721 | [-8.724, 5.862] | 0.7005 | 0.239 |
| action_change_next_main | primary_belief_error_rate | -0.782 | 2.515 | [-5.711, 4.148] | 0.7560 | 0.458 |
| action_change_next_main | chosen_action_conflicts_with_reported_state | 0.536 | 0.645 | [-0.728, 1.800] | 0.4060 | 1.709 |
| optimality_loss_next_compact | Intercept | -2.774 | 0.685 | [-4.117, -1.431] | 0.0001 | 0.062 |
| optimality_loss_next_compact | reasoning_progress | -1.012 | 1.916 | [-4.767, 2.743] | 0.5972 | 0.363 |
| optimality_loss_next_compact | action_entropy_bits | -0.000 | 406318.721 | [-796384.693, 796384.693] | 1.0000 | 1.000 |
| optimality_loss_next_compact | state_belief_entropy_per_0p01_bits | -0.343 | 1.992 | [-4.248, 3.562] | 0.8632 | 0.709 |
| optimality_loss_next_compact | primary_belief_error_rate | -5.432 | 5.243 | [-15.707, 4.843] | 0.3001 | 0.004 |
| optimality_recovery_next_compact | Intercept | -2.113 | 0.552 | [-3.196, -1.030] | 0.0001 | 0.121 |
| optimality_recovery_next_compact | reasoning_progress | -5.364 | 1.959 | [-9.204, -1.524] | 0.0062 | 0.005 |
| optimality_recovery_next_compact | action_entropy_bits | -1245588.641 | 4761827.141 | [-10578769.837, 8087592.556] | 0.7936 |  |
| optimality_recovery_next_compact | state_belief_entropy_per_0p01_bits | -8.906 | 18.277 | [-44.728, 26.916] | 0.6261 | 0.000 |
| optimality_recovery_next_compact | primary_belief_error_rate | 6.241 | 2.961 | [0.436, 12.045] | 0.0351 | 513.127 |
