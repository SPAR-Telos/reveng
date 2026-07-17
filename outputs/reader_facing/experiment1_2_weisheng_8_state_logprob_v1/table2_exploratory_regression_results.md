# Table 2. Exploratory Regression Results

Exploratory logistic regressions predicting whether the next reasoning prefix changes the recommended action or its optimality. Rows are prefixes from eight states, so p-values are diagnostics rather than population-level inference. Action confidence is scaled per +0.00001 probability because the recommended-action probability is almost always close to one.

| outcome | predictor | coefficient | SE | 95% CI | p | events | warning |
|---|---|---:|---:|---:|---:|---:|---|
| Action changes at next prefix | reasoning progress | -3.373 | 0.685 | [-4.715, -2.031] | 8.36e-07 | 61 |  |
| Action changes at next prefix | action confidence (+0.00001 probability) | -3.573 | 6.468 | [-16.251, 9.105] | 0.5807 | 61 |  |
| Action changes at next prefix | state-belief uncertainty (0.01 bits) | -0.265 | 0.501 | [-1.247, 0.718] | 0.5974 | 61 |  |
| Action changes at next prefix | belief error rate | -0.370 | 1.210 | [-2.743, 2.002] | 0.7597 | 61 |  |
| Action changes at next prefix | recommended action conflicts with reported state | 1.121 | 0.351 | [0.432, 1.809] | 0.0014 | 61 |  |
| Optimal action becomes suboptimal | reasoning progress | -0.739 | 1.232 | [-3.154, 1.676] | 0.5486 | 12 |  |
| Optimal action becomes suboptimal | action confidence (+0.00001 probability) | 1.422 | 13.107 | [-24.268, 27.112] | 0.9136 | 12 |  |
| Optimal action becomes suboptimal | state-belief uncertainty (0.01 bits) | -0.045 | 0.464 | [-0.954, 0.864] | 0.9222 | 12 |  |
| Optimal action becomes suboptimal | belief error rate | -0.401 | 2.316 | [-4.940, 4.138] | 0.8626 | 12 |  |
| Optimal action becomes suboptimal | recommended action conflicts with reported state | 0.523 | 0.711 | [-0.870, 1.916] | 0.4618 | 12 |  |
| Suboptimal action becomes optimal | reasoning progress | -3.478 | 1.212 | [-5.855, -1.102] | 0.0041 | 15 | near-perfect fitted probabilities |
| Suboptimal action becomes optimal | action confidence (+0.00001 probability) | 453.510 | 523.239 | [-572.038, 1479.059] | 0.3861 | 15 | near-perfect fitted probabilities |
| Suboptimal action becomes optimal | state-belief uncertainty (0.01 bits) | -0.168 | 0.709 | [-1.558, 1.223] | 0.8131 | 15 | near-perfect fitted probabilities |
| Suboptimal action becomes optimal | belief error rate | -1.220 | 2.208 | [-5.548, 3.107] | 0.5805 | 15 | near-perfect fitted probabilities |
| Suboptimal action becomes optimal | recommended action conflicts with reported state | 2.538 | 0.702 | [1.163, 3.914] | 0.0003 | 15 | near-perfect fitted probabilities |
