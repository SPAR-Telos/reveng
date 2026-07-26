# Results to Add to the Project Summary

## Experimental scope

The new analysis follows GPT-OSS-20B reasoning within 46 matched DoorKey
environment states, comprising 23 failure states and 23 controls from 31
on-policy trajectories. After every reasoning sentence, we elicit the model's
recommended action and its probabilities over UP, DOWN, LEFT, and RIGHT. We
also ask 63,756 questions about walls, key possession, door state, and the
predicted consequence of the recommended action. Action and categorical-belief
probabilities use candidate-token logprobs at temperature 0.7. The DoorKey BFS
planner accepts every tied optimal action.

The analysis contains 7,084 reasoning positions and 7,038 positions with
matching GPT-OSS-20B activations. Action commitment is retrospective: it is the
first sentence boundary where the recommended action equals the full-trace
recommendation and remains unchanged.

The underlying activation collection is larger than this matched analysis: it
contains sentence-mean and sentence-final GPT-OSS-20B representations at
layers 8, 15, and 23 for 153,622 reasoning sentences from 1,276 environment
states in 95 trajectories. The 46-state analysis is the matched failure-control
subset for which action and belief measurements are complete.

## Recommended order

### 1. When does the model commit to an action?

Use `figs/commitment_timing_by_state_group.png`.

Failure states commit later on average than controls, both in sentence position
and in the fraction of reasoning revealed. The means are 130 versus 97
sentences and 0.77 versus 0.68 of the reasoning text. The bootstrap intervals
overlap, so this is a descriptive tendency rather than a confirmed group
difference.

### 2. How does action uncertainty change at commitment?

Use
`../../experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1/figs/action_entropy_around_commitment.png`.

The action distribution becomes substantially sharper after commitment. At the
commitment boundary, entropy decreases by 0.245 bits relative to the preceding
sentence, with a 95% bootstrap interval from -0.386 to -0.094. Across the
post-commitment period, the recommended action remains highly confident. On
average, 28.7% of the reasoning trace remains after retrospective commitment,
so stable action selection often precedes the end of the written reasoning.

### 3. What happens to state beliefs around commitment and action changes?

Use `figs/belief_entropy_around_commitment_by_family.png`, followed by
`figs/belief_errors_before_after_optimality_changes.png`.

Wall beliefs are much more uncertain than key-possession and door-state
beliefs, and their uncertainty generally decreases after commitment. Mean
current-state belief entropy is 0.044 bits lower after commitment than before
it. Belief errors fluctuate near optimality loss and recovery, but the
trajectory-held-out models do not show that current-state or action-consequence
beliefs consistently predict those transitions better than reasoning position
alone. This is currently a null result for the strong claim that measured
belief changes reliably trigger action changes.

### 4. Can activations identify action changes?

Use
`../../experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1/figs/activation_event_monitor_auc.png`.

Simple activation metrics do not reliably identify commitment, optimality
loss, or recovery. Cosine similarity to the mean of earlier sentence
representations modestly distinguishes some events, but this is a same-position
association rather than advance prediction. Supervised activation classifiers
reach held-out AUROC values from 0.60 to 0.70, but they do not consistently
improve over a model using reasoning position, and action and belief readouts
remain stronger predictors.

### 5. Does the final action attend to reasoning near these events?

Use
`../../hypothesis_tests/attention_decision_relevance_v1/figs/attention_event_window_difference.png`.

The final action token assigns more attention to reasoning near retrospective
commitment than to a non-overlapping comparison window from the same state.
The direction is consistent at full-attention layers 15 and 23. Attention near
optimality loss, recovery, and ordinary recommendation changes is close to
zero. This result is correlational and does not establish that the attended
sentences caused or informed the final action.

## Suggested short conclusion

The scaled analysis gives a clearer account of action commitment than of
belief-driven action changes. Actions often stabilize before reasoning ends,
and action uncertainty falls after stabilization. Behavioral belief errors and
simple activation changes do not consistently predict optimality transitions
beyond reasoning progress. Final-action attention is concentrated near the
commitment boundary, providing a candidate information-routing signal for
held-out and causal tests, but not yet evidence that the model uses its state
information correctly.

## Results not recommended for this short summary

- The allocentric transition probes use a prompt formulation that may confuse
  the model and should be replaced with real-agent transition questions.
- The environment-step analysis uses a different population from the matched
  sentence analysis, so it is not a controlled comparison of analysis units.
- The factorized belief interactions and linear CKA analysis are exploratory
  null results and do not currently clarify the main story.
- The attention-head ranking uses the same pilot data for selection and effect
  estimation and belongs in an appendix until held-out confirmation.
