# Attention Decision Relevance

This analysis stores compact attention from the final action token to canonical reasoning sentences. It does not store full token-by-token attention matrices.

- Scope: `matched46`
- Metric: final-action attention `mass` aggregated over sentence windows
- Event window: +/-3 sentences
- Event-window rows: 6,474
- Head-event rows: 414,336

## Main event-window summary

| Event type | Layer | Group | Events | Mean event minus control attention | Fraction positive |
|---|---:|---|---:|---:|---:|
| action_change | 15 | all | 682 | 0.002955 | 0.501 |
| action_change | 15 | control | 313 | 0.001788 | 0.495 |
| action_change | 15 | failure | 369 | 0.003945 | 0.507 |
| belief_change:action_consequence | 15 | all | 907 | 0.007986 | 0.527 |
| belief_change:action_consequence | 15 | control | 435 | 0.010581 | 0.545 |
| belief_change:action_consequence | 15 | failure | 472 | 0.005594 | 0.511 |
| belief_change:adjacent_wall | 15 | all | 142 | 0.002964 | 0.500 |
| belief_change:adjacent_wall | 15 | control | 70 | 0.001304 | 0.571 |
| belief_change:adjacent_wall | 15 | failure | 72 | 0.004577 | 0.431 |
| belief_change:task_state | 15 | all | 39 | -0.018300 | 0.256 |
| belief_change:task_state | 15 | control | 8 | -0.002642 | 0.375 |
| belief_change:task_state | 15 | failure | 31 | -0.022341 | 0.226 |
| commitment_onset | 15 | all | 45 | 0.036739 | 0.644 |
| commitment_onset | 15 | control | 22 | 0.020512 | 0.591 |
| commitment_onset | 15 | failure | 23 | 0.052260 | 0.696 |
| suboptimal_to_optimal | 15 | all | 182 | 0.001046 | 0.456 |
| suboptimal_to_optimal | 15 | control | 75 | 0.000733 | 0.467 |
| suboptimal_to_optimal | 15 | failure | 107 | 0.001265 | 0.449 |
| sustained_optimal_to_suboptimal | 15 | all | 115 | 0.000434 | 0.443 |
| sustained_optimal_to_suboptimal | 15 | control | 40 | -0.000862 | 0.375 |
| sustained_optimal_to_suboptimal | 15 | failure | 75 | 0.001125 | 0.480 |
| transient_optimal_to_suboptimal | 15 | all | 46 | 0.002094 | 0.609 |
| transient_optimal_to_suboptimal | 15 | control | 20 | 0.004293 | 0.750 |
| transient_optimal_to_suboptimal | 15 | failure | 26 | 0.000402 | 0.500 |

## Top replicated heads

| Event type | Layer | Head | Events | Mean event minus control attention | Positive folds |
|---|---:|---:|---:|---:|---:|
| commitment_onset | 15 | 52 | 45 | 0.110284 | 5/5 |
| commitment_onset | 15 | 60 | 45 | 0.085608 | 5/5 |
| commitment_onset | 15 | 56 | 45 | 0.084274 | 5/5 |
| commitment_onset | 15 | 33 | 45 | 0.076459 | 5/5 |
| commitment_onset | 15 | 45 | 45 | 0.075125 | 5/5 |
| commitment_onset | 15 | 53 | 45 | 0.074685 | 5/5 |
| commitment_onset | 15 | 59 | 45 | 0.073574 | 5/5 |
| commitment_onset | 15 | 4 | 45 | 0.072758 | 5/5 |
| commitment_onset | 15 | 40 | 45 | 0.061273 | 5/5 |
| commitment_onset | 15 | 19 | 45 | 0.060854 | 5/5 |

## Interpretation note

This is correlational. A positive event-minus-control value means the final action token attended more to sentences near that event than to a progress-matched non-event window in the same state. This supports an information-routing hypothesis only as auxiliary evidence; it does not prove causal use.
