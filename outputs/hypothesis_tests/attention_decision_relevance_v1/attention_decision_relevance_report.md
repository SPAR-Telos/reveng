# Attention Decision Relevance

This analysis measures attention from the final action token to canonical reasoning sentences. It does not store full token-by-token attention matrices.

- Scope: `matched46`
- Primary metric: the share of final-action attention assigned to a seven-sentence window, averaged across attention heads
- Event window: the event sentence and 3 sentences before and after it
- Comparison: the closest non-overlapping window in the same state, at a similar reasoning position, containing no event of the same type
- Confidence intervals: computed across trajectory-level mean differences
- Event-window rows: 4,018
- Head-event rows: 257,152

## Main event-window summary

| Event | Layer | State group | Event positions | Trajectories | Attention difference (percentage points) | 95% CI | Trajectories with positive difference |
|---|---:|---|---:|---:|---:|---:|---:|
| Recommended action changes | 15 | all | 647 | 30 | 0.003 | [-0.395, 0.401] | 0.433 |
| Recommended action changes | 15 | control | 297 | 22 | -0.045 | [-0.502, 0.411] | 0.318 |
| Recommended action changes | 15 | failure | 350 | 17 | -0.026 | [-0.493, 0.440] | 0.529 |
| Reported action consequence changes | 15 | all | 842 | 29 | 1.629 | [-0.124, 3.382] | 0.793 |
| Reported action consequence changes | 15 | control | 400 | 19 | 1.871 | [-0.726, 4.469] | 0.684 |
| Reported action consequence changes | 15 | failure | 442 | 17 | 1.128 | [0.060, 2.195] | 0.765 |
| Reported wall state changes | 15 | all | 116 | 15 | 0.739 | [-0.451, 1.929] | 0.400 |
| Reported wall state changes | 15 | control | 55 | 7 | 0.593 | [-0.734, 1.921] | 0.429 |
| Reported wall state changes | 15 | failure | 61 | 9 | 0.743 | [-1.032, 2.518] | 0.333 |
| Reported key or door state changes | 15 | all | 36 | 8 | 0.570 | [-4.330, 5.470] | 0.375 |
| Reported key or door state changes | 15 | control | 7 | 3 | 2.423 | [-5.263, 10.109] | 0.333 |
| Reported key or door state changes | 15 | failure | 29 | 5 | -0.541 | [-7.349, 6.266] | 0.400 |
| Action commitment | 15 | all | 43 | 30 | 2.992 | [-0.281, 6.264] | 0.567 |
| Action commitment | 15 | control | 22 | 22 | 1.329 | [-2.058, 4.717] | 0.545 |
| Action commitment | 15 | failure | 21 | 15 | 4.966 | [0.208, 9.725] | 0.667 |
| Recommended action becomes optimal | 15 | all | 170 | 22 | -0.433 | [-1.087, 0.222] | 0.409 |
| Recommended action becomes optimal | 15 | control | 67 | 17 | 0.024 | [-0.402, 0.451] | 0.471 |
| Recommended action becomes optimal | 15 | failure | 103 | 14 | -0.954 | [-1.844, -0.064] | 0.214 |
| Sustained change to a suboptimal action | 15 | all | 112 | 17 | -0.095 | [-0.520, 0.329] | 0.294 |
| Sustained change to a suboptimal action | 15 | control | 38 | 11 | -0.149 | [-0.528, 0.231] | 0.182 |
| Sustained change to a suboptimal action | 15 | failure | 74 | 11 | -0.136 | [-0.868, 0.596] | 0.364 |
| Transient change to a suboptimal action | 15 | all | 43 | 15 | -0.008 | [-0.773, 0.757] | 0.200 |
| Transient change to a suboptimal action | 15 | control | 17 | 11 | 0.280 | [-0.727, 1.288] | 0.455 |
| Transient change to a suboptimal action | 15 | failure | 26 | 9 | -0.536 | [-0.934, -0.137] | 0.111 |

## Action commitment across usable layers

| Layer | States | Trajectories | Attention difference (percentage points) | 95% CI |
|---:|---:|---:|---:|---:|
| 15 | 43 | 30 | 2.992 | [-0.281, 6.264] |
| 23 | 43 | 30 | 0.151 | [0.021, 0.282] |

## Exploratory head screen

| Event | Layer | Head | Event positions | Attention difference (percentage points) | Positive trajectory subsets |
|---|---:|---:|---:|---:|---:|
| Action commitment | 15 | 52 | 43 | 9.253 | 5/5 |
| Action commitment | 15 | 45 | 43 | 7.470 | 4/5 |
| Action commitment | 15 | 33 | 43 | 7.398 | 5/5 |
| Action commitment | 15 | 63 | 43 | 7.324 | 5/5 |
| Action commitment | 15 | 4 | 43 | 7.222 | 5/5 |
| Action commitment | 15 | 56 | 43 | 6.727 | 4/5 |
| Action commitment | 15 | 60 | 43 | 6.582 | 4/5 |
| Action commitment | 15 | 59 | 43 | 6.467 | 5/5 |
| Action commitment | 15 | 53 | 43 | 6.100 | 5/5 |
| Action commitment | 15 | 41 | 43 | 5.685 | 5/5 |

## Interpretation note

Action commitment is the clearest current signal: the final action token assigns more attention to reasoning near commitment than to comparable reasoning elsewhere. Changes in reported action consequences show a smaller positive difference. Attention near optimality loss and recovery is close to zero. These associations do not show that the attended sentences caused or informed the final action.

Layer 8 is excluded because the stored aggregates are invalid for that layer's 128-token sliding-attention window; see `attention_analysis_checks.md`. The head screen is exploratory because heads were selected and summarized on the same pilot data.
