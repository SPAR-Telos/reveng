# Step-Level Reasoning Diagnostic: Balanced Run v1

## Research Question

This experiment tests two related questions:

1. As progressively more of a reasoning trace is revealed, does the model's
   newly elicited next-action recommendation change from a planner-optimal
   action to a planner-suboptimal action?
2. When such a transition occurs, do hidden-state geometry measurements change
   near the same reasoning chunk?

This is an observational diagnostic. It does not establish that a reasoning
chunk caused the original trajectory's action or failure.

## Definitions

- **Failure state**: a DoorKey state selected because the original trajectory
  exhibits a behavioral failure category. This label refers to the original
  trajectory, not to every prefix-action query.
- **Prefix-elicited action**: a new next-action recommendation obtained by
  giving TogetherAI the unchanged grid state and the first `k` reasoning
  chunks.
- **Planner-optimal action**: an elicited action contained in the stored
  `optimal_actions_json` set for that grid state.
- **Sustained optimal-to-suboptimal recommendation transition**: the first
  prefix where the preceding valid prefix elicits a planner-optimal action,
  the current prefix elicits a planner-suboptimal action, and at least 50% of
  valid later prefixes also elicit planner-suboptimal actions.
- **Mixed or transient suboptimal recommendations**: at least one prefix
  elicits a planner-suboptimal action, but no transition satisfies the
  sustained-transition rule.
- **All valid prefix-elicited actions planner-optimal**: every valid prefix
  query elicits a planner-optimal action. One failure state in this category
  has one unavailable TogetherAI response; all of its other responses are
  planner-optimal.

The report does not call reasoning text itself "optimal." Optimality applies
only to the action elicited after a reasoning prefix.

## Design

- Model: `openai/gpt-oss-20b`, MXFP4 on one NVIDIA L4.
- States: 24 previously evaluated DoorKey states: 12 failure states and 12
  non-failure controls.
- Failure categories: backtrack, short loop, two-cycle oscillation, wall
  collision, repeated action without progress, and avoidable detour.
- Reasoning segmentation: contiguous natural-language chunks, capped at 32
  chunks per state. These are punctuation/newline-based textual units, not
  semantically annotated reasoning steps.
- Activations: pre-reasoning boundary plus every reasoning chunk, layers 8,
  15, and 23; last-token and mean-pooled representations.

## Heuristic Choices

The sustained-transition label uses a `0.5` persistence threshold: at least
half of valid prefixes from the transition onward must recommend a
planner-suboptimal action. A stricter all-later-prefixes-suboptimal label is
also stored, but is not used for the primary event-centered geometry result.

At each sustained transition, the geometry comparison uses the mean of the
preceding three chunks as its reference. The 32-chunk segmentation cap, the
`0.5` persistence threshold, and the three-chunk reference window are
pre-analysis heuristic choices. No threshold is applied to the cosine-based
geometry metrics; their values are reported continuously.

The chunks have not been classified as semantic reasoning moves such as state
reconstruction, route evaluation, correction, backtracking, or commitment.
The detected transition can occur inside a merged chunk. See
`method_examples.md` and `docs/step_reasoning_drift.md` for the exact chunking
and action-query procedures.

## Planner-Optimality Audit

The stored `optimal_actions_json` values come from a DoorKey-aware BFS over
actual environment transitions, not a goal-only distance calculation. The
planner state tracks the rendered grid and key-carrying bit, so shortest paths
account for acquiring/removing the key and opening/removing a required door.
Every tied minimum-distance first action is accepted as planner-optimal.
The run does not use the source dataset's `legacy_optimal_actions_json`, which
does not model these DoorKey transitions.

Fresh recomputation exactly matches the stored action set for all 24 states.
Three states have two tied optimal first actions, and none of the five
sustained transitions occurs in those states. Therefore, tied shortest routes
do not explain the reported sustained transitions. The simplified legacy
action set differs from the DoorKey-aware set in 11/24 states, confirming that
using the correct stored field matters.

The audit also exposes a sampling confound rather than a planner error:

- all 12 states without a held key require acquiring it on every shortest path
- all 22 states with a visible closed door require opening and stepping through
  it on every shortest path
- failure states are mostly post-key (`9/12` carrying), while controls are
  mostly pre-key (`9/12` not carrying)

The current sample has no key-visible-but-unnecessary or closed-door-bypassable
controls. Future scaling should match or stratify states by task phase,
required-door status, shortest distance, and number of tied optimal actions.
Per-state details are in `planner_optimality_audit.csv`.
A focused explanation and aggregate table are in
`planner_optimality_audit.md`.

## Data Quality And Complete State Accounting

- Prefix evaluations: 719 total; 718 valid (99.9%); 1 unavailable TogetherAI
  response.
- Activation-index rows: 2,157; every indexed tensor path exists.
- Sustained transitions: 5/24 states, including 4/12 failure states and 1/12
  non-failure controls.
- Token-boundary audit: adjacent activation spans overlap at 549/695 chunk
  boundaries (79.0%); see the limitation below.

| State group | N | Sustained optimal-to-suboptimal transition | Mixed/transient suboptimal recommendations | All valid prefix-elicited actions planner-optimal |
|---|---:|---:|---:|---:|
| Failure states | 12 | 4/12 (33.3%) | 6/12 (50.0%) | 2/12 (16.7%) |
| Non-failure controls | 12 | 1/12 (8.3%) | 4/12 (33.3%) | 7/12 (58.3%) |

The eight failure states without a sustained transition are fully accounted
for: six have mixed or transient suboptimal recommendations, and two have
planner-optimal recommendations for every valid prefix. They do not contribute
to the event-centered geometry analysis because that analysis requires an
identified transition step.

Per-state outcomes are recorded in `state_outcome_accounting.csv`.

## Behavioral Interpretation

Sustained optimal-to-suboptimal recommendation transitions occur more often in
failure states than controls (4/12 versus 1/12). Conversely, every valid
prefix elicits a planner-optimal action more often in controls (7/12 versus
2/12). This supports the feasibility of locating some behaviorally meaningful
changes within reasoning traces, but it does not yield a transition for every
failure state. With only 12 states per group, these results are descriptive.

## Geometry Interpretation

The event-centered analysis compares each geometry metric at the sustained
transition against its mean over the preceding three chunks. For the four
failure-state transitions, mean-pooled overall-direction agreement decreases on average at
layers 8, 15, and 23. Individual states vary substantially, and there is only
one control transition. The current run therefore does not establish a
reliable geometric detector or a preferred layer.

The outcome-class comparison aggregates each activation metric to one mean per
state and layer, then compares states with sustained transitions, transient
suboptimal recommendations, and planner-optimal recommendations throughout.
The three classes overlap substantially across overall-direction agreement, adjacent-step
cosine, optimal-action anchor cosine, and update norm. The current activation
metrics therefore do not clearly distinguish these behavioral outcomes.

Overall-direction agreement is the cosine similarity between the current
chunk-to-chunk activation update and the trajectory's overall first-to-final
activation direction. A negative transition-minus-reference difference means
that the activation update at the transition points less strongly along that
overall direction than updates in the preceding three chunks. It is not, by
itself, a classification of a reasoning chunk as wrong. The corresponding
internal CSV field is `aligned_change`.

### Activation Token-Boundary Limitation

The completed run maps each character chunk to every token whose offset
overlaps that chunk. Because GPT-style tokens can combine leading whitespace
with the next word, adjacent token spans overlap at 549/695 boundaries
(79.0%). In the concrete transition shown in `method_examples.md`, the saved
last token for chunk 13 decodes to ` But`, which begins chunk 14, and both
chunks include that token.

The existing last-token geometry is therefore not a clean measurement at the
end of each textual chunk. Mean-pooled geometry is less affected by one
boundary token but is also not strictly non-overlapping. The activation
extraction should be corrected and regenerated before making substantive
geometry claims. The behavioral prefix-action results are unaffected.

The optimal-trajectory anchor cosine is available for every geometry row,
using representations from trajectories whose valid prefix actions are all
planner-optimal. It remains a secondary exploratory metric because anchors and
evaluated states come from the same small run.

## Figures

- `figs/balanced_geometry_overview.png` shows the complete behavioral outcome
  distribution and mean-pooled overall-direction-agreement differences at the four
  failure-state sustained transitions.
- `figs/aligned_change_curves.png` shows overall-direction-agreement trajectories.
- `figs/activation_metrics_by_outcome_class.png` compares state-level
  activation metrics across sustained transitions, transient suboptimality,
  and planner-optimal recommendations throughout.
- `figs/wrong_turn_histogram.png` is retained as a legacy diagnostic whose
  internal "wrong turn" wording refers to sustained recommendation
  transitions.
- `method_examples.md` shows a concrete chunk transition, raw prefix-elicited
  actions, prompt shape, and decoded activation tokens.
- `planner_optimality_audit.csv` verifies stored action sets and records
  key/door requirements across shortest paths.
- `planner_optimality_audit.md` summarizes the planner semantics, audit
  findings, and remaining sampling confounds.
- `state_class_geometry_rows.csv` contains one mean activation-metric row per
  state and layer.
- `class_geometry_summary.csv` summarizes those state-level means by outcome
  class and layer.

## Activation Storage

- Tensor root: `data/behavioral_probes/step_reasoning_drift_balanced_v1/activations/`
- Activation index: `data/behavioral_probes/step_reasoning_drift_balanced_v1/step_activation_rows.csv`
- Prompt and analysis text: `data/behavioral_probes/step_reasoning_drift_balanced_v1/prompts/`
- Per-state resumable checkpoints: `data/behavioral_probes/step_reasoning_drift_balanced_v1/example_checkpoints.jsonl`
- Each tensor is a compact `(2880,)` BF16 vector. Paths encode trajectory,
  environment step, reasoning chunk, layer, and pooling type.

## Next Scaling Decision

Scale both failure states and matched non-failure controls. The immediate goal
is to estimate sustained-transition rates by failure category and determine
whether any geometry measurement consistently changes before or at those
transitions. Mixed/transient cases should also be analyzed directly rather
than discarded by the current persistence threshold.

The concrete follow-up specification for belief lead/lag analysis, semantic
reasoning-move annotation, and counterfactual sentence replacement is in
`docs/reasoning_belief_shift_experiment_proposal.md`.
