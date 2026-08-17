# Uncertainty Predictors of Action Events

Action uncertainty is Shannon entropy over the normalized UP, DOWN, LEFT, and RIGHT token probabilities from a temperature 0.7 logprob query. State-belief uncertainty is the mean Shannon entropy over the six primary yes/no/unknown logprob readouts at temperature 0.7.

| outcome | predictor | coefficient | SE | 95% CI | p | odds ratio | events |
|---|---|---:|---:|---:|---:|---:|---:|
| Action changes at next prefix | Action uncertainty (1 bit) | 18482.590 | 29735.608 | [-39799.201, 76764.381] | 0.5342 |  | 61 |
| Action changes at next prefix | State-belief uncertainty (0.01 bits) | -0.265 | 0.501 | [-1.247, 0.718] | 0.5973 | 0.767 | 61 |
| Optimal action becomes suboptimal | Action uncertainty (1 bit) | -6307.481 | 57200.307 | [-118420.083, 105805.121] | 0.9122 |  | 12 |
| Optimal action becomes suboptimal | State-belief uncertainty (0.01 bits) | -0.050 | 0.479 | [-0.988, 0.889] | 0.9174 | 0.952 | 12 |
| Suboptimal action becomes optimal | Action uncertainty (1 bit) | -955957.563 | 1860743.498 | [-4603014.818, 2691099.693] | 0.6074 |  | 15 |
| Suboptimal action becomes optimal | State-belief uncertainty (0.01 bits) | -0.166 | 0.734 | [-1.604, 1.272] | 0.8207 | 0.847 | 15 |

The action-change model has 61 events. The optimality-loss and recovery models have 12 and 15 events, respectively. Coefficients and p-values are unstable when event counts are small. All standard errors are model-based; repeated prefixes and the 8-state sample mean these are exploratory diagnostics rather than population-level inference.
