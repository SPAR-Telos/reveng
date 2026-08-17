# Experiment 2: Behavioral Beliefs and Action Optimality

The analysis contains 63,756 categorical belief readouts and 1,088 adjacent-prefix belief changes. Action and categorical belief uncertainty use candidate-token logprobs at temperature 0.7; repeated sampling is not used.

Models hold out complete trajectories and keep matched pairs in the same validation group. Results are predictive associations rather than causal effects.

| Outcome | Best learned model | AUROC | Events |
|---|---|---:|---:|
| optimal to suboptimal | action conditioned | 0.607 | 160 |
| suboptimal to optimal | global beliefs | 0.473 | 173 |

`door_open` is excluded from the fitted models because its ground truth is always `no` in this matched cohort. Consequently, the proposed key-door interaction cannot be estimated here.

Coordinate beliefs remain deferred because they require generated coordinate answers rather than the categorical candidate-logprob protocol used in this run.

See `TRANSITION_MODEL_SUMMARY.md` for held-out metrics and bootstrap intervals.
