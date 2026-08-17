# Experiment 1 Activation Monitor

## Status

- Activation compatibility: `compatible`
- Usable activation rows: 2904
- Action events: 37
- Event-aligned geometry rows: 444
- Trajectories: 8

## Definitions

- Recommended action: the action elicited from the model after showing the grid state and a prefix of the existing reasoning trace.
- Action change: the recommended action differs between adjacent reasoning prefixes.
- Optimality loss: the recommendation changes from planner-optimal to planner-suboptimal.
- Sustained optimality loss: after an optimal-to-suboptimal change, at least two of the next three valid recommendations are suboptimal.
- Optimality recovery: the recommendation changes from planner-suboptimal to planner-optimal.
- Commitment onset: the first prefix where the recommendation equals the final full-trace action and all later valid prefix recommendations keep that action.
- Action entropy: Shannon entropy over repeated sampled action recommendations for the same prefix.

## Action Events

| Event type | Count |
|---|---:|
| action_change | 21 |
| commitment_onset | 5 |
| optimality_loss | 3 |
| optimality_recovery | 6 |
| sustained_optimality_loss | 2 |

## Commitment Timing

- Mean commitment progress: 0.337
- Number with commitment step: 8

## Action Entropy

- Events with action entropy delta: 37
- Mean action entropy delta at events: -0.067
