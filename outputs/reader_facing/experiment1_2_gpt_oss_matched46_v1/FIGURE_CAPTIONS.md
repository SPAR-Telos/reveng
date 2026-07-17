# Figure Captions

## commitment_timing_by_state_group.png

Action commitment timing for failure and control states in the matched 46-state GPT-OSS-20B sentence-prefix run. Commitment is the first sentence prefix where the recommended action equals the full-trace recommendation and remains stable through the rest of the trace. The left panel shows sentence index; the right panel shows the fraction of reasoning text revealed. Points are states; black markers show group means with 95 percent bootstrap confidence intervals over states. The x-axis labels include the maximum number of sentence prefixes in each group.

## action_entropy_change_decision_events.png

Change in action entropy at decision events. For each event, entropy change is entropy at the event sentence prefix minus entropy at the previous sentence prefix. Action entropy is Shannon entropy over the temperature 0.7 candidate-logprob distribution on UP, DOWN, LEFT, and RIGHT. Negative values mean the model became more confident in one action at the event. The plot does not use activations; the available activations for this run are means over sentence-token spans from the same fixed reasoning traces.

## belief_errors_before_after_optimality_changes.png

Mean belief error rates around recommendation changes and changes in whether the recommended action is optimal. Offset 0 is the first sentence prefix where the event occurs. Error rate means the fraction of relevant belief probes whose answer disagrees with the ground-truth DoorKey state label. The plot separates wall, key, and door beliefs and includes an all-current-state-belief aggregate.

## belief_entropy_around_commitment_aggregate.png

Mean current-state belief entropy around retrospective action commitment. Offset 0 is the first sentence prefix where the recommended action equals the full-trace recommendation and remains stable. Entropy is averaged over wall-left, wall-right, wall-up, wall-down, key-held, and door-open belief probes.

## belief_entropy_around_commitment_by_family.png

Mean belief entropy around retrospective action commitment, separated into wall, key, and door belief families. This checks whether post-commitment reasoning is associated with residual uncertainty about particular parts of the environment state.

## belief_entropy_around_commitment_by_question.png

Mean belief entropy around retrospective action commitment for each current-state belief probe: wall left, wall right, wall up, wall down, key held, and door open.
