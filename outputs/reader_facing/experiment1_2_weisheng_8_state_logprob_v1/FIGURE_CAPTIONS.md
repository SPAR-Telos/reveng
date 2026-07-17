# Figure Captions

## `figure1_action_confidence_over_reasoning_progress.png`
At each reasoning prefix, the model assigns probabilities to UP, DOWN, LEFT, and RIGHT from a temperature 0.7 logprob query. The recommended action is the highest-probability action. The line shows the median across eight states, and the band shows the interquartile range. The plot shows that action selection is usually highly confident, explaining why action entropy is close to zero.

## `figure2_belief_errors_around_action_optimality_changes.png`
Belief error rate before and after changes in whether the recommended action is optimal. Offset 0 is the first prefix where action optimality changes. Belief error is averaged over wall-left, wall-right, wall-up, wall-down, key-held, and door-open probes.

## `figure3_state_belief_uncertainty_over_reasoning_progress.png`
State-belief uncertainty is Shannon entropy over yes, no, and unknown probabilities from temperature 0.7 logprob queries. The curve shows the mean across eight state-level bin averages, and the band shows the interquartile range. Each prefix averages wall, key, and door beliefs.
