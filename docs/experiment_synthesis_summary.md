# Synthesis of the repo’s experiments and claims

## One-sentence unifying claim

Across the paper, posters, and planned follow-ups, the central claim is that language-model agents often encode task-relevant state and value information but do not reliably convert that information into the final action; the key missing link is the reasoning stage where action commitment, confidence, and semantic function change.

## What the completed work is really about

The main empirical story in [belief_action_gap.tex](../belief_action_gap.tex) and [mapping_belief_action_gap_full_report.tex](../mapping_belief_action_gap_full_report.tex) is not simply “the model is wrong sometimes.” It is closer to:

- the model often knows useful state facts;
- it often has a coherent internal preference structure for what the task-optimal action should be;
- yet its emitted action can still be inconsistent with that information;
- this is best described as an action-selection or commitment failure rather than a pure belief failure.

This is the core tension behind the work.

## Main experiments already run

### 1. Belief and action decomposition

The main paper studies three linked measurements:

- representational belief: state variables decoded from hidden activations;
- behavioral belief: explicit answers to local state questions;
- action value: one-step inverse reinforcement learning over exact next states.

The main empirical result is that the model’s internal representations are often more predictive of the task-optimal action than of the action it actually emits.

### 2. Reasoning-stage diagnostics

The repo also studies how the gap changes during reasoning:

- pre- vs post-reasoning belief precision;
- action probability changes over reasoning steps;
- commitment timing, when the model’s recommended action first becomes stable;
- post-commitment reasoning and whether it is behaviorally meaningful or mostly epiphenomenal.

This connects the belief-action-gap paper to a more mechanistic story about how reasoning changes action selection.

### 3. Behavioral probes and failure slices

The behavioral-probe pipeline in [README.md](../README.md) and the supporting notes in [docs/behavioral_probe_pathologies.md](./behavioral_probe_pathologies.md) show a large amount of work on:

- directional wall questions;
- object-location questions;
- action-consequence questions;
- failure-state slices where the model reports a local fact correctly but still acts inconsistently.

These experiments are important because they show that the contradiction is not only hidden in activations; it can also be observed from the model’s own explicit statements.

## Semantic classification work: completed, with a qualified result

The semantic classification bridge is now complete under [outputs/hypothesis_tests/semantic_reasoning_classification_v1](../outputs/hypothesis_tests/semantic_reasoning_classification_v1):

- matched change-point sentences versus matched control sentences;
- a blinded annotation guide;
- all 320 sentences labeled by a deterministic local judge;
- explicit adjudication of all 18 low-confidence outputs;
- calibration, hidden-duplicate, raw-label sensitivity, and representative-example audits;
- matched tests and links to exact action events, activation geometry, attention, and behavioral beliefs.

The primary hypothesis was:

- the sentences immediately before action-distribution changes or commitment boundaries are more likely to be functions such as verification, consolidation, restatement, or correction than ordinary reasoning sentences.

The answer is null-to-modest. Route deliberation is 5.3 percentage points more common at retrospective BEAST change points than at same-state, progress/length-matched sentences, but the trajectory-bootstrap interval includes zero. The four-way omnibus test is not significant (p = 0.219), and semantic function does not reliably improve held-out change-point discrimination by itself. The conclusion is unchanged without the low-confidence adjudication overrides.

This sharpens the paper: action-distribution changes are meaningful routing and timing boundaries, but they do not have one stable verbal discourse-function signature. Behavioral-belief entropy does vary across semantic functions in an exploratory corrected analysis, with the highest entropy during route deliberation; this should be replicated rather than promoted to the main claim.

BEAST itself is an offline, full-trace detector. Its posterior for a candidate location uses observations after that location, and the reported event-proximity analysis uses a symmetric ±3-sentence window. These results therefore localize and interpret changes retrospectively; they are not real-time predictions. The separate practical action-event monitor is the prospective analysis and uses only information available at the current sentence.

## Current paper and poster materials

The main .tex documents point to the same scientific agenda:

- [belief_action_gap.tex](../belief_action_gap.tex): the core paper draft focused on the belief-action-gap framing and the claim that the model can know and value the right thing while still acting inconsistently.
- [report_branch_behav_probes.tex](../report_branch_behav_probes.tex): a more applied progress report that emphasizes behavioral probes, agreement/disagreement analyses, and the gap between what the model reports and what it does.
- [mapping_belief_action_gap_full_report.tex](../mapping_belief_action_gap_full_report.tex): a fuller report version that formalizes the framework more explicitly and connects it to inverse reinforcement learning, probes, and action-selection failure.
- [docs/slides/counterfactual_progress_slides.tex](./slides/counterfactual_progress_slides.tex): the presentation-oriented version that highlights the intervention logic and the distinction between belief and action.

These are not separate projects. They are different presentations of the same underlying thesis.

## Planned but not yet completed experiments

The repo also contains several high-value follow-ups that are not yet fully run:

1. Causal sentence interventions
   - Delete, replace, or resample candidate reasoning sentences near action changes.
   - This would move from correlation to causal evidence.

2. Attention and mechanistic follow-ups
   - Test whether the final action token selectively attends to commitment-related or belief-change-related sentences.
   - Follow with head-level or token-level ablation if the signal is robust.

3. Transition-belief and post-commitment analyses
   - Test whether unresolved state or transition beliefs predict continued reasoning after commitment.

4. Plan-decoder reasoning hypothesis
   - Test whether reasoning sharpens local next-action information while degrading longer-horizon plan structure.
   - This is a strong and elegant follow-up if the activation artifacts are compatible.

## The most coherent next framing

If we want a simple, field-standard story for the next paper or poster, I would frame it as:

> Language-model agents often contain enough state and value information to choose the right action, but that information is not always faithfully converted into the emitted action. Reasoning changes this conversion process through a commitment stage that can be studied with action-probability change points, semantic labels of reasoning sentences, and intervention-based tests.

That framing unifies the papers, the probes, the commitment analyses, and the semantic-classification work.
