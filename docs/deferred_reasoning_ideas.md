# Deferred Reasoning Experiments

This document keeps exploratory ideas separate from completed results and
assigns each idea to the experiment it would extend.

## Selection Criteria

Prioritize an experiment when it:

1. can establish or falsify a central claim using held-out or causal evidence;
2. connects multiple existing results, rather than adding an isolated metric.

Runtime and implementation risk break ties. A low-cost result is not
high-priority if it cannot change the paper's conclusion.

## Recommended Central Claim

The most coherent claim available from the current work is:

> GPT-OSS-20B action selection often has two behaviorally distinguishable
> stages: a large shift in probability toward the eventual action, followed
> later by the point where that action remains the model's recommendation.
> The first stage has a candidate activation correlate, while the second is
> followed by increasing action confidence and receives elevated final-action
> attention.

This descriptive claim connects the action-distribution, commitment timing,
action uncertainty, sentence activation, final-action attention, and
post-commitment analyses. It is not yet causal. Current-state and transition
beliefs have not explained either boundary in the matched-46 pilot.

## Priority Order

| Priority | Experiment | Contribution to the central claim | Existing experiments it connects |
|---|---|---|---|
| 1 | Sentence-level action-distribution change points | Completed matched-46 test distinguishes the largest probability revision from stable action selection | Commitment timing, action entropy, optimality transitions, revealed-reasoning experiments |
| 2 | Semantic validation of both boundaries | Tests whether probability revision and stable selection have different reasoning functions | Change points, activation similarity, attention, post-commitment reasoning |
| 3 | Two-boundary truncation and sentence intervention | Tests whether reasoning after either boundary can be removed or changed without changing the action | Commitment tails, reasoning patching, epiphenomenal reasoning |
| 4 | Held-out attention confirmation followed by head ablation | Tests whether final-action information is routed through reasoning near stable selection and whether candidate heads matter causally | Final-action attention, activation monitors, reasoning patching |
| 5 | Real-agent transition beliefs around commitment | Tests whether unresolved or incorrect world-model predictions explain commitment timing or suboptimal commitment | Current-state beliefs, action-conditioned consequences, belief-action gap |
| 6 | Plan-decoder evaluation across reasoning | Tests whether immediate action information stabilizes while longer-horizon plan information changes | IRL value structure, activation collection, commitment |
| 7 | Activation-similarity stress test | Determines whether the existing geometric association adds information beyond reasoning progress and action confidence | Activation geometry and action events |

| Idea | Status | Testable claim | What remains |
|---|---|---|---|
| Sentence-level action-distribution change points | Matched-46 analysis complete | A large shift toward the full-trace action and stable recommendation are distinct boundaries | Scale only after semantic or causal validation; use repeated sampling only if candidate normalization is questioned |
| Activation-similarity stress test | Change-point test complete | Layer-15 sentence-mean similarity is associated with distribution changes and may precede the largest full-distribution change | The leading result is representation-dependent and does not predict the final-action jump or stable selection; replicate before elevating it |
| Semantic validation of action-change sentences | Not run | Reiteration, route rechecking, or consolidation occurs disproportionately before action changes and partly explains the activation-similarity and attention signals | Use blinded language-model judge labels and trajectory-held-out models; see `future_semantic_explanation_activation_similarity.md` |
| Post-commitment reasoning triggers | Partial pilot | Current-state and transition-belief uncertainty may predict how much reasoning continues after action commitment | The combined model has positive cross-validated R2 in 46 states, while either uncertainty family alone does not. Confirm on held-out trajectories with improved transition probes. |
| Factorized belief triggers | Partial null result | Task-motivated combinations of wall, key, door, and transition beliefs may explain action changes or post-commitment length better than additive beliefs | Existing simple interactions did not improve held-out action-transition prediction. Test again only after improving transition-belief measurements and adding true door-state variation. |
| Post-commitment activation monitor | Partial pilot | Activations distinguish pre-commitment reasoning, the commitment boundary, and the post-commitment tail | A binary commitment-event classifier was tested, but the prespecified three-way pre, boundary, and post classifier and cross-task validation remain. |
| Reasoning persona | Not run | A stable tendency to continue reasoning generalizes across tasks, prompts, and environments | Requires multiple tasks, prompt styles, models, and repeated trajectories. DoorKey alone cannot establish a persona. |
| Attention to commitment and belief-change sentences | First pass complete | Final action tokens selectively attend to sentences associated with commitment or belief changes | The matched-46 pilot finds a candidate commitment association. Confirm on held-out trajectories, compare with non-action output tokens, then test candidate heads causally. |
| Activation Oracle transition readouts | Not run as a matched transition test | Transition beliefs can be decoded from Qwen3-8B activations and agree with behavioral readouts | Existing Qwen Activation Oracle runs cover state and action questions, but not a scaled, matched transition-belief analysis. |
| Counterfactual sentence replacement | Not run | Changing a candidate belief-bearing sentence changes downstream beliefs or action | Compare original, deletion, replacement, and belief-edit conditions with repeated continuations. |
| Plan-decoder reasoning hypothesis | Newly feasible, not run | Reasoning sharpens immediate-action information while degrading longer-horizon plan information | Evaluate the released pre-reasoning and post-reasoning plan decoders on the newly collected compatible GPT-OSS activations; see `plan_decoder_reasoning_hypothesis.md`. |
| Exact MI-peaks analysis | Not run | Sparse reasoning positions carry unusually high dependence on the final answer representation | The current linear CKA analysis is not this method. Implement the published token-level estimator before making an MI claim. |
| D2H or activation-manifold analysis | Not run | Correct and failed reasoning traces occupy different low-dimensional trajectories | Defer until simple held-out activation monitors generalize; otherwise the added geometric complexity is unlikely to clarify the mechanism. |

## Sentence-Level Action-Distribution Change Points

The script-level implementation plan is in
`docs/action_distribution_change_point_script_plan.md`.

The current action elicitation is already sufficient for the primary test. At
each sentence boundary \(t\), candidate-token logprobs define:

\[
q_t(a) = P(a \mid \text{prompt and reasoning through sentence }t),
\quad a \in \{\mathrm{UP},\mathrm{DOWN},\mathrm{LEFT},\mathrm{RIGHT}\}.
\]

Use three complementary boundaries:

| Boundary | Definition | Question answered |
|---|---|---|
| Largest final-action probability increase | \(t_{\mathrm{jump}}=\arg\max_t[q_t(a_T)-q_{t-1}(a_T)]\), where \(a_T\) is the full-trace action | Which sentence most sharply increases support for the final action? |
| Stable-action boundary | Earliest \(t\) for which \(\arg\max q_t=a_T\) and the argmax remains \(a_T\) through the trace | When does the recommended action stop changing? |
| Distribution change point | A sentence with high Jensen-Shannon divergence \(JS(q_{t-1},q_t)\), optionally selected by an offline change-point algorithm | Where does the full action distribution change, including changes that preserve the argmax? |

The action probability vector measures decision change, not semantic change.
If semantic change is needed, separately compute a text-embedding distance or
use blinded reasoning-function labels. Do not describe a change in the action
histogram as semantic drift.

Sentence resolution is the correct first test because the behavioral probes,
activations, commitment events, and attention aggregates already share those
boundaries. Token-level analysis should be attempted only if sentence-level
boundaries miss sharp within-sentence changes.

Repeated sampling is not required for the primary analysis because the stored
temperature-0.7 candidate logprobs already yield \(q_t\). A sensitivity check
can sample 30 actions at a stratified set of positions around detected
boundaries and compare the empirical histogram with the logprob distribution.
Sampling 30 times at all 7,084 matched-46 sentence positions would require
212,520 generations and is not justified before this gate. Sampling at every
token position would require about 3.26 million generations for the matched-46
set (108,522 reasoning tokens), or about 71.1 million for the full activation
corpus (2.37 million reasoning tokens).

Adding a sentence and forcing an immediate action is a truncation-based
marginal-effect test, but it does not isolate the sentence's semantic content.
Deleting, replacing, or resampling that sentence is required for a stronger
causal claim.

## Activation-Similarity Stress Test

Use the existing layer-15 sentence-mean cosine similarity as one prespecified
feature. For each target, fit trajectory-held-out models:

1. reasoning progress and action confidence;
2. the same baseline plus activation similarity;
3. the same model with the event at the next sentence as the target.

Targets are recommendation change, optimal-to-suboptimal transition,
suboptimal-to-optimal transition, and commitment onset. Report AUROC, log loss,
and Brier score. Use progress-matched non-event positions and a
within-trajectory permutation test. Exclude positions within three sentences
of another event in a sensitivity analysis.

This test is minor but useful: if activation similarity improves next-sentence
prediction, it becomes an early internal signal that connects activation
geometry to behavioral changes. If it only distinguishes the current event, it
should remain a descriptive contemporaneous association.

## Semantic Validation Across Existing Experiments

The semantic classification should not be treated as a standalone annotation
exercise. Use one shared sentence taxonomy to test the following links:

| Existing or planned result | What semantic labels make testable | Valid conclusion if supported |
|---|---|---|
| Action-distribution change points | Whether large probability shifts occur after planning, reiteration, route rechecking, consolidation, correction, or backtracking sentences | Particular reasoning functions are associated with action-distribution changes |
| Activation similarity | Whether high similarity is explained by restatement, rechecking, or consolidation rather than a generic position effect | The activation signal has an interpretable linguistic correlate |
| Final-action attention | Whether the final action token preferentially attends to planning or consolidation sentences rather than arbitrary nearby text | Final-action attention is selective for particular reasoning functions |
| Action commitment | Which sentence function most often occurs at the stable-action boundary, separately for optimal and suboptimal commitments | Commitment has a reproducible semantic signature |
| Post-commitment reasoning | Whether the tail primarily contains checking, repetition, or correction language despite a stable action distribution | Apparently deliberative post-commitment text can be behaviorally inert |
| Belief changes | Whether correction or backtracking sentences coincide with corrected state or transition reports | Verbal correction aligns, or fails to align, with measured belief correction |
| Sentence interventions | Which semantic categories should be deleted, replaced, or resampled first | Category-conditioned interventions can test whether the identified sentences cause action changes |

The primary semantic hypothesis is:

> Sentences classified as reiteration, route rechecking, or consolidation are
> more common immediately before action changes than among sentences matched
> on trajectory, reasoning progress, and length.

For the matched-46 analysis, label all optimality losses, recoveries, and
commitment boundaries after deduplication, plus a stratified sample of ordinary
recommendation changes. Match each event sentence to a non-event sentence from
the same trajectory on reasoning progress and sentence length. Use a fixed
temperature-zero judge, hide action and activation labels, and use a second
judge on 20 percent of examples.

Classification can validate a semantic association and can explain why
activation or attention signals occur. It cannot show that the sentence caused
the action change. That requires the planned deletion, replacement, or
on-policy resampling experiment.

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
