# Uncertainty Predictors of Action Events

Action uncertainty is Shannon entropy over 10 immediate-action samples at temperature 0.7. State-belief uncertainty is the mean Shannon entropy over the six primary yes/no/unknown logprob readouts at temperature 0.0.

| outcome | predictor | coefficient | SE | 95% CI | p | odds ratio | events |
|---|---|---:|---:|---:|---:|---:|---:|
| Action changes at next prefix | Action uncertainty (1 bit) | 2.061 | 0.631 | [0.824, 3.298] | 0.0011 | 7.852 | 18 |
| Action changes at next prefix | State-belief uncertainty (0.01 bits) | 0.541 | 0.460 | [-0.360, 1.443] | 0.2394 | 1.718 | 18 |
| Optimal action becomes suboptimal | Action uncertainty (1 bit) | 1.345 | 1.115 | [-0.840, 3.530] | 0.2276 | 3.838 | 4 |
| Optimal action becomes suboptimal | State-belief uncertainty (0.01 bits) | 0.313 | 0.913 | [-1.476, 2.102] | 0.7316 | 1.368 | 4 |
| Suboptimal action becomes optimal | Action uncertainty (1 bit) | 0.420 | 1.017 | [-1.574, 2.413] | 0.6800 | 1.521 | 5 |
| Suboptimal action becomes optimal | State-belief uncertainty (0.01 bits) | 0.299 | 0.518 | [-0.716, 1.315] | 0.5633 | 1.349 | 5 |

The action-change model has 18 events. The optimality-loss and recovery models have only 4 and 5 events, respectively, so their coefficients and p-values are unstable. All standard errors are model-based; repeated prefixes and the 8-state sample mean these are exploratory diagnostics rather than population-level inference.
