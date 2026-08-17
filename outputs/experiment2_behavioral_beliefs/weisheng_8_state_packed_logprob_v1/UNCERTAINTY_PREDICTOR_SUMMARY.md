# Uncertainty Predictors of Action Events

Action uncertainty is Shannon entropy over the normalized UP, DOWN, LEFT, and RIGHT token probabilities from a temperature 0.7 logprob query. State-belief uncertainty is the mean Shannon entropy over the six primary yes/no/unknown logprob readouts at temperature 0.7.

| outcome | predictor | coefficient | SE | 95% CI | p | odds ratio | events |
|---|---|---:|---:|---:|---:|---:|---:|
| Action changes at next prefix | Action uncertainty (1 bit) | -208009.315 | 394654.398 | [-981531.936, 565513.305] | 0.5981 |  | 26 |
| Action changes at next prefix | State-belief uncertainty (0.01 bits) | -1.431 | 3.721 | [-8.724, 5.862] | 0.7005 | 0.239 | 26 |
| Optimal action becomes suboptimal | Action uncertainty (1 bit) | -0.000 | 406318.721 | [-796384.693, 796384.693] | 1.0000 | 1.000 | 6 |
| Optimal action becomes suboptimal | State-belief uncertainty (0.01 bits) | -0.343 | 1.992 | [-4.248, 3.562] | 0.8632 | 0.709 | 6 |
| Suboptimal action becomes optimal | Action uncertainty (1 bit) | -1245588.641 | 4761827.141 | [-10578769.837, 8087592.556] | 0.7936 |  | 9 |
| Suboptimal action becomes optimal | State-belief uncertainty (0.01 bits) | -8.906 | 18.277 | [-44.728, 26.916] | 0.6261 | 0.000 | 9 |

The action-change model has 26 events. The optimality-loss and recovery models have 6 and 9 events, respectively. Coefficients and p-values are unstable when event counts are small. All standard errors are model-based; repeated prefixes and the 8-state sample mean these are exploratory diagnostics rather than population-level inference.
