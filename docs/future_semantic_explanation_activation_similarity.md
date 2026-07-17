# Future Test: Semantic Explanation for Activation Similarity

## Question

Higher similarity to preceding sentence activations modestly distinguishes
optimality loss and other action events. Does this occur because the sentence
restates, summarizes, or consolidates earlier reasoning before the recommended
action changes?

This interpretation is not part of the current result. The current evidence
only establishes a geometric association.

## Proposed Judge-Based Test

Sample action-event sentences and non-event sentences matched within the same
environment state on reasoning progress and sentence length. Give a language
model judge the current sentence and all preceding reasoning, while hiding the
action-event and activation labels. Require one primary label:

| Label | Operational definition |
|---|---|
| Restatement | Repeats a previously stated fact, action, or conclusion without adding a new inference. |
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

## Related Work

- [Thought Anchors](https://openreview.net/forum?id=VnSlfeRCaU) studies
  sentence importance, planning, and uncertainty management using
  counterfactual resampling and attention.
- [Beyond the Commitment Boundary](https://arxiv.org/abs/2606.13603) studies
  stable answer formation and post-commitment reasoning.
- [APR](https://openreview.net/forum?id=QqliFQUVzu) studies self-restating
  reasoning after answer commitment.

These works motivate the categories but do not establish cosine similarity to
preceding activations as a semantic measure.
