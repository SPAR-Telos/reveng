# Matched-46 belief-replacement and verifier-clause experiment

## Design

The cohort contains 46 decisions from 31 trajectories. Every query retains the original grid and key status, removes the CoT, and adds one preregistered replacement block. Actions are scored from candidate-token probabilities at temperature 0.7.

| Condition | Differs from full CoT | Mean TV from full CoT | Optimal top action |
|---|---:|---:|---:|
| Full CoT | 0/46 (0.0%) | 0.000 | 38/46 (82.6%) |
| Grid only | 34/46 (73.9%) | 0.723 | 17/46 (37.0%) |
| Model reports | 31/46 (67.4%) | 0.677 | 23/46 (50.0%) |
| Verified truth | 36/46 (78.3%) | 0.767 | 11/46 (23.9%) |
| One flipped wall fact | 36/46 (78.3%) | 0.750 | 20/46 (43.5%) |
| Irrelevant control | 31/46 (67.4%) | 0.686 | 22/46 (47.8%) |

## Main comparisons

Model reports disagree with the full-CoT top action in 31/46 cases, compared with 34/46 for grid only. Verified clauses disagree with full CoT in 36/46 cases. Their top action is optimal in 11/46 cases. Relative to full CoT, verified clauses correct 2 suboptimal actions and degrade 29 optimal actions.

The verified-clause and full-CoT actions differ in 36/46 cases (mean TV 0.767). Flipping one deterministically selected wall fact changes the action relative to the verified-truth condition in 27/46 cases (mean TV 0.406).

Intervals in summary.csv and pairwise_contrasts.csv use trajectory-clustered bootstrap resampling.

## Interpretation limits

This is a prompt intervention, not evidence that the elicited clauses are the latent beliefs used during the original forward pass. Verified facts are also visible in the grid, so improvements measure whether an explicit trusted summary changes action selection. The counterfactual condition is a sensitivity test; it is not a valid task state. Direct candidate scoring is deterministic, so repeated sampling seeds would not measure generation variance.
