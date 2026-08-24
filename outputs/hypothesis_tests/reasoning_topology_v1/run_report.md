# Continuous Reasoning-Trajectory Topology Test

## Question

Do exact-matched rollout-failure states show more continuous representational recurrence, tortuosity, directional reversal, or weaker late consolidation than controls?

This test uses no PCA and no clustering. Every trace is linearly resampled to 32 reasoning-progress points, centered within trace, and RMS-normalized. The metrics are continuous trajectory summaries; `nonlocal_recurrence` measures normalized return proximity and is not a claim that a discrete graph cycle exists.

## Data

- States: 46 in 23 exact matched pairs
- Source trajectories: 31
- Independent trajectory-connected validation groups: 11
- Strict final-suboptimal failure states at layer 15: 8
- Primary representation: layer 15 sentence means
- Sensitivity layers: 8 and 23
- Intervals: bootstrap trajectory-connected groups; p-values: exact grouped sign flips

## Primary layer results

| Metric | Failure mean | Control mean | Difference | 95% interval | Two-sided p | BH q |
|---|---:|---:|---:|---:|---:|---:|
| nonlocal_recurrence | 0.3598 | 0.3593 | +0.0005 | [-0.0070, +0.0057] | 0.8398 | 0.8398 |
| path_tortuosity | 20.9721 | 21.4405 | -0.4684 | [-2.7495, +1.4171] | 0.6836 | 0.8398 |
| directional_reversal_rate | 0.9797 | 0.9362 | +0.0435 | [-0.0217, +0.1214] | 0.7656 | 0.8398 |
| late_to_early_dispersion_ratio | 1.1808 | 1.0509 | +0.1300 | [+0.0434, +0.2159] | 0.0420 | 0.1680 |

## Decision

None of the four layer-15 metrics survives within-layer false-discovery correction at q < 0.05.
Nominal directional evidence: late_to_early_dispersion_ratio

This is a matched observational test. Rollout-failure labels describe the source trajectory and are not identical to a final suboptimal recommendation. The strict final-suboptimal subset is retained in `trace_metrics.csv` but is too small for a primary independent-trajectory test.
