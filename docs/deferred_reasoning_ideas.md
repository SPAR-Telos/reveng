# Deferred Reasoning Experiments

This document keeps exploratory ideas separate from completed results and
assigns each idea to the experiment it would extend.

| Idea | Correct place | Testable claim | Required evidence |
|---|---|---|---|
| Semantic explanation of higher activation similarity | Experiment 1 follow-up | Sentences with higher similarity to preceding activations are more often restatements or decision consolidation | Blinded language-model judge labels and trajectory-held-out models; see `future_semantic_explanation_activation_similarity.md` |
| Post-commitment reasoning triggers | Experiment 2 observational extension | Particular current-state or transition-belief patterns predict whether substantial reasoning continues after action commitment | Belief errors and entropy at commitment, post-commitment sentence count, and grouped held-out prediction |
| Factorized belief triggers | Experiment 2 modelling extension | Task-motivated combinations of wall, key, door, and transition beliefs predict post-commitment length better than additive beliefs | Prespecified interactions, comparison with progress and action-confidence baselines, and validation on held-out trajectories |
| Post-commitment activation monitor | Experiment 1 representational extension | A supervised activation direction distinguishes pre-commitment reasoning, the commitment boundary, and the post-commitment tail | Position labels, train-fold-only dimensionality reduction, trajectory-held-out classification, and cross-task validation |
| Reasoning persona | Separate cross-context study | A stable model-level tendency to continue reasoning generalizes across tasks, prompts, and environments | Multiple tasks, prompt styles, models, and repeated trajectories; DoorKey alone cannot establish a persona |
| Attention to commitment and belief-change sentences | Experiment 1 mechanism follow-up | Final action tokens selectively attend to sentences associated with commitment or belief changes | New attention extraction, matched controls, head selection on training trajectories, and held-out validation |
| Activation Oracle transition readouts | Experiment 2 representational extension | Transition beliefs can be decoded from Qwen3-8B activations and agree with behavioral readouts | Matched Qwen trajectories and the supported Activation Oracle LoRA; this cannot directly validate GPT-OSS-20B |
| Counterfactual sentence replacement | Experiment 3 causal study | Changing a candidate belief-bearing sentence changes downstream beliefs or action | Original, deletion, replacement, and belief-edit conditions with repeated continuations |

## Post-Commitment Trigger Analysis

The first test should remain specific and observational. For each state, define
the outcome as the fraction or number of reasoning sentences generated after
retrospective action commitment. Compare:

1. reasoning-progress and action-confidence baseline;
2. additive current-state and transition-belief errors and entropy;
3. prespecified DoorKey interactions, such as key possession with door status
   and chosen-direction wall belief with predicted collision;
4. activation monitor features.

Evidence that a belief pattern predicts tail length would show an association
between unresolved task beliefs and continued reasoning. It would not show that
the beliefs cause the reasoning tail.

## Generalization Boundary

An activation classifier may identify a post-commitment representation within
DoorKey, but it should be called a post-commitment activation monitor, not an
“epiphenomenal reasoning vector,” until it generalizes across held-out tasks and
models. A persona claim is stronger still and requires stable behavior across
different prompts, tasks, and interaction contexts.
