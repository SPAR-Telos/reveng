# Figure Captions

## commitment_timing_by_state_group.png

Action commitment timing for failure and control states in the matched 46-state GPT-OSS-20B run. The recommendation is measured after every reasoning sentence. Commitment is the first sentence boundary where the recommended action equals the full-trace recommendation and remains stable through the rest of the trace. The left panel reports the number of reasoning sentences revealed at commitment; the right panel divides the number of reasoning characters revealed by the total number of reasoning characters in that trace. The 46 traces contain 20 to 365 reasoning sentences. Points are environment states; dark markers show group means with 95 percent bootstrap confidence intervals over states.

## action_entropy_change_decision_events.png

Change in action uncertainty when the recommendation changes or first remains stable. For each event, the plotted value is entropy after the event minus entropy at the preceding sentence boundary. Action entropy is Shannon entropy over the temperature 0.7 candidate-logprob distribution on UP, DOWN, LEFT, and RIGHT. Negative values mean the model became more confident in one action. The plot does not use activations.

## belief_errors_before_after_optimality_changes.png

Mean belief error rates around recommendation changes and changes in whether the recommended action is optimal. Position 0 is the first sentence boundary where the event occurs. Error rate is the fraction of wall, key-possession, or door-state questions whose answer disagrees with the ground-truth DoorKey state. The plot reports each belief family and an aggregate across all six current-state questions.

## belief_entropy_around_commitment_aggregate.png

Mean current-state belief uncertainty around retrospective action commitment. Position 0 is the first sentence boundary where the recommended action equals the full-trace recommendation and remains stable. Entropy is averaged over wall-left, wall-right, wall-up, wall-down, key-held, and door-open answer distributions at temperature 0.7.

## belief_entropy_around_commitment_by_family.png

Mean answer entropy around retrospective action commitment, separated into wall, key-possession, and door-state questions. This checks whether post-commitment reasoning is associated with residual uncertainty about particular parts of the environment state.

## belief_entropy_around_commitment_by_question.png

Mean belief entropy around retrospective action commitment for each current-state belief probe: wall left, wall right, wall up, wall down, key held, and door open.
