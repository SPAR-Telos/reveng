# Sentence and Environment-Step Comparison

Sentence-prefix analysis measures changes while the grid state remains fixed. Environment-step analysis measures changes after actions update the grid state. The populations differ: the sentence analysis uses 46 matched states, whereas the environment analysis uses all 1,276 states from 95 trajectories. Values should therefore be compared as robustness descriptions, not as a controlled segmentation effect.

| Analysis unit | States | Trajectories | Positions | Mean action confidence | Mean belief entropy | Action changes | Optimality losses | Recoveries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sentence prefixes | 46 | 31 | 7,084 | 0.864 | 0.422 | 682 | 161 | 182 |
| Environment steps | 1276 | 95 | 1,276 | 1.000 | 0.235 | 511 | 20 | 22 |

Commitment onset is defined only within a fixed-state reasoning trace, so it is not reported for environment-step transitions.
