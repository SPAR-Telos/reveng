# Belief Indicators of Action-Optimality Transitions

Window: three reasoning positions before and after each action transition. Loss indicators are belief-error onsets; recovery indicators are belief-error recoveries.

## Optimal to Suboptimal

| Belief | Events preceded | Events coinciding | Events followed | Prospective precision | Number of error onsets |
|---|---:|---:|---:|---:|---:|
| wall_left | 0.0% | 8.0% | 0.0% | 33.3% | 6 |
| wall_right | 8.0% | 4.0% | 8.0% | 50.0% | 6 |
| wall_up | 4.0% | 4.0% | 8.0% | 14.3% | 14 |
| wall_down | 12.0% | 12.0% | 16.0% | 18.8% | 32 |
| has_key | 8.0% | 0.0% | 8.0% | 25.0% | 8 |
| door_open | 52.0% | 12.0% | 52.0% | 15.8% | 95 |

## Suboptimal to Optimal

| Belief | Events preceded | Events coinciding | Events followed | Prospective precision | Number of error recoveries |
|---|---:|---:|---:|---:|---:|
| wall_left | 3.4% | 10.3% | 0.0% | 50.0% | 8 |
| wall_right | 3.4% | 3.4% | 3.4% | 33.3% | 6 |
| wall_up | 3.4% | 6.9% | 3.4% | 18.8% | 16 |
| wall_down | 10.3% | 10.3% | 20.7% | 19.4% | 31 |
| has_key | 3.4% | 3.4% | 3.4% | 25.0% | 8 |
| door_open | 24.1% | 17.2% | 37.9% | 13.8% | 87 |

## Interpretation

- No primary belief is currently a strong standalone indicator.
- Right-wall error onset has the highest prospective precision for optimality loss, but this is based on only six onsets.
- Door-open answer changes cover many events but have low precision, occur both before and after events, and are measured against a constant closed-door truth label. Treat them as probe instability.
- Wall-down changes appear around both loss and recovery events, but their low precision makes them nonspecific.
- Exact chains are sparse. The only repeated exact chain occurs in two events; the most frequent primary-belief pair occurs in three events.

These are descriptive pilot results from 24 states, not causal effects or final statistical estimates.

Figure: `figs/belief_transition_indicator_heatmap.png`.
