# Behavioral-Only Belief and Action Pilot

The white-box transfer results are excluded. This report uses behavioral probes and prefix-conditioned action recommendations only.

## Coverage

- Independent states: 24
- Reasoning positions: 719
- Behavioral belief queries: 9,344
- Valid response parse rate: 96.5%
- Action events: 74
- Belief-answer changes: 1,114
- Invalid-answer boundaries: 387

## Action Events

| Event type | Events | With belief-error onset leading or coinciding | With persistent belief-error onset leading or coinciding |
|---|---:|---:|---:|
| sustained optimal to suboptimal | 13 | 10 | 9 |
| transient optimal to suboptimal | 12 | 10 | 8 |
| suboptimal to optimal recovery | 29 | 19 | 17 |
| action identity change without optimality change | 20 | 0 | 0 |

## Descriptive Next-Position Associations

Belief errors significantly associated with a suboptimal recommendation at the next reasoning position after false-discovery-rate correction:

| Belief | Odds ratio | Adjusted q |
|---|---:|---:|
| wall_right | 192.96 | 5.48e-20 |
| wall_left | 10.51 | 2.42e-06 |
| wall_up | 7.11 | 5.34e-06 |
| has_key | 8.39 | 0.00123 |
| wall_down | 2.97 | 0.0066 |

These are descriptive screening results. Reasoning positions are repeated measurements within only 24 states, so final inference requires the matched or expanded run with trajectory-grouped statistical analysis.

Figure: `figs/event_aligned_belief_errors.png`.
