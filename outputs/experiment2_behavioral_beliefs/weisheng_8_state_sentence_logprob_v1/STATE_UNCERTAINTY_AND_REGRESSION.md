# State-Belief Uncertainty And Exploratory Regression

State-belief uncertainty is computed from logprob readouts for the six primary yes/no/unknown state-belief probes: wall left, wall right, wall up, wall down, key held, and door open.
For each probe, yes/no/unknown probabilities are normalized after excluding invalid mass, then Shannon entropy is computed in bits. Per-position uncertainty is the mean entropy across available primary probes.
The label logprobs were requested at temperature 0.7.

The regression table is exploratory. It treats repeated reasoning prefixes as rows and reports model-based logistic-regression standard errors, confidence intervals, and p-values. Because this run has 8 states from one source trajectory group, these p-values are diagnostics, not a population-level inferential claim.
In the regression, state-belief entropy is scaled in units of 0.01 bits because the logprob readouts are very confident and raw entropy values are close to zero.

## Query Summary

- Uncertainty rows requested: 3972
- Successful logprob rows: 3972
- Failed logprob rows: 0
- Approximate query cost in USD: 0.0000
- Mean probe entropy in bits: 0.002
- Mean per-position state-belief entropy in bits: 0.002

## Event-Level State-Belief Uncertainty

| event type | n | mean previous entropy | mean current entropy | mean change |
|---|---:|---:|---:|---:|
| action identity change without optimality change | 34 | 0.000 | 0.000 | 0.000 |
| suboptimal to optimal recovery | 15 | 0.001 | 0.000 | -0.000 |
| sustained optimal to suboptimal | 3 | 0.000 | 0.000 | -0.000 |
| transient optimal to suboptimal | 9 | 0.001 | 0.001 | -0.000 |

## Regression Status

| model | outcome | n | events | converged | warning |
|---|---|---:|---:|---|---|
| action_change_next_main | action_change_next | 653 | 61 | True |  |
| optimality_loss_next_compact | optimality_loss_next | 653 | 12 | True |  |
| optimality_recovery_next_compact | optimality_recovery_next | 653 | 15 | True | near-perfect fitted probabilities; possible separation |

## Regression Coefficients

| model | term | coefficient | SE | 95% CI | p | odds ratio |
|---|---|---:|---:|---:|---:|---:|
| action_change_next_main | Intercept | -1.318 | 0.237 | [-1.783, -0.853] | 0.0000 | 0.268 |
| action_change_next_main | reasoning_progress | -3.373 | 0.685 | [-4.715, -2.031] | 0.0000 | 0.034 |
| action_change_next_main | action_entropy_bits | 18482.590 | 29735.608 | [-39799.201, 76764.381] | 0.5342 |  |
| action_change_next_main | state_belief_entropy_per_0p01_bits | -0.265 | 0.501 | [-1.247, 0.718] | 0.5973 | 0.767 |
| action_change_next_main | primary_belief_error_rate | -0.369 | 1.210 | [-2.742, 2.003] | 0.7602 | 0.691 |
| action_change_next_main | chosen_action_conflicts_with_reported_state | 1.120 | 0.351 | [0.432, 1.809] | 0.0014 | 3.066 |
| optimality_loss_next_compact | Intercept | -3.809 | 0.566 | [-4.917, -2.700] | 0.0000 | 0.022 |
| optimality_loss_next_compact | reasoning_progress | -0.357 | 1.109 | [-2.530, 1.817] | 0.7478 | 0.700 |
| optimality_loss_next_compact | action_entropy_bits | -6307.481 | 57200.307 | [-118420.083, 105805.121] | 0.9122 |  |
| optimality_loss_next_compact | state_belief_entropy_per_0p01_bits | -0.050 | 0.479 | [-0.988, 0.889] | 0.9174 | 0.952 |
| optimality_loss_next_compact | primary_belief_error_rate | 0.067 | 2.216 | [-4.276, 4.410] | 0.9758 | 1.069 |
| optimality_recovery_next_compact | Intercept | -3.083 | 0.476 | [-4.015, -2.151] | 0.0000 | 0.046 |
| optimality_recovery_next_compact | reasoning_progress | -1.521 | 1.081 | [-3.641, 0.598] | 0.1594 | 0.218 |
| optimality_recovery_next_compact | action_entropy_bits | -955957.563 | 1860743.498 | [-4603014.818, 2691099.693] | 0.6074 |  |
| optimality_recovery_next_compact | state_belief_entropy_per_0p01_bits | -0.166 | 0.734 | [-1.604, 1.272] | 0.8207 | 0.847 |
| optimality_recovery_next_compact | primary_belief_error_rate | 0.745 | 2.040 | [-3.254, 4.744] | 0.7149 | 2.107 |
