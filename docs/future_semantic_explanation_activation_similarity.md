# Future Test: Semantic Explanation for Activation Similarity

## Question

Higher similarity to preceding sentence activations modestly distinguishes
optimality loss and other action events. Does this occur because the sentence
restates, summarizes, or consolidates earlier reasoning before the recommended
action changes?

The broader hypothesis is that reiteration, route rechecking, or consolidation
is more common immediately before action changes than in reasoning-progress
and length-matched non-event sentences.

This interpretation is not part of the current result. The current evidence
only establishes a geometric association.

## Proposed Judge-Based Test

Sample action-event sentences and non-event sentences matched within the same
environment state on reasoning progress and sentence length. Give a language
model judge the current sentence and all preceding reasoning, while hiding the
action-event and activation labels. Require one primary label:

| Label | Operational definition |
|---|---|
| Restatement or reiteration | Repeats a previously stated fact, action, or conclusion without adding a new inference. |
| Route rechecking | Re-evaluates whether a previously proposed route is traversable, valid, or efficient. |
| Summary or consolidation | Combines earlier claims into a plan or decision without introducing a new state fact. |
| New inference | Derives a new state fact, consequence, or route decision. |
| Correction or backtracking | Explicitly revises or rejects earlier reasoning. |
| Procedural continuation | Continues calculation or route enumeration without fitting the other categories. |

Use a fixed judge model and prompt, temperature 0, and a versioned random
sample. Run an independent second judge on a stratified 20 percent subset and
report category agreement. Judge disagreements remain an explicit limitation;
neither judge is treated as ground truth.

## Analysis

Fit trajectory-held-out models in this order:

1. action event from reasoning progress and sentence length;
2. add activation similarity and its change from the previous sentence;
3. add the judge's semantic category;
4. include both activation and semantic variables.

Report whether restatement or consolidation is more common when similarity
increases and whether adding semantic labels reduces the activation-similarity
coefficient. Such attenuation would be consistent with the semantic
explanation, but it would not establish mediation or causality.

## Connections to the Main Claim

Apply the same labels to action-distribution change points, commitment
boundaries, final-action attention windows, belief changes, and
post-commitment reasoning. This makes the following questions testable:

- whether the largest increase in full-trace action probability and later
  stable action selection have different semantic categories;
- which reasoning functions precede changes in the action distribution;
- whether optimal and suboptimal commitments have different semantic
  signatures;
- whether activation similarity and final-action attention select the same
  classes of sentences;
- whether post-commitment text consists mainly of rechecking and reiteration
  while action probabilities remain stable;
- whether correction language coincides with corrected state or transition
  beliefs.

These are correlational links. If one category consistently precedes action
changes, use that category to select sentences for deletion, replacement, and
on-policy resampling. Only those interventions can support the claim that the
identified sentence function changes the action.

## Related Work

- [Thought Anchors](https://openreview.net/forum?id=6NUtPO9PdV) studies
  sentence importance, planning, and uncertainty management using
  counterfactual resampling and attention.
- [Beyond the Commitment Boundary](https://arxiv.org/abs/2606.13603) studies
  stable answer formation and post-commitment reasoning.
- [APR](https://openreview.net/forum?id=QqliFQUVzu) studies self-restating
  reasoning after answer commitment.

These works motivate the categories but do not establish cosine similarity to
preceding activations as a semantic measure.
