# Do Internal-State and Belief Shifts Track Action Changes During Reasoning?

## Research Question

Can we detect when GPT-OSS-20B's reasoning stops supporting an optimal action,
and can that change be explained by shifts in its state beliefs?

We answer this in three stages:

1. test whether activation geometry monitors action-optimality transitions;
2. test whether interpretable behavioral belief shifts explain or predict the
   same transitions;
3. intervene on supported belief shifts or reasoning chunks to test causality.

The first two experiments are observational. They identify predictive signals
and candidate mechanisms, but do not establish that an activation or belief
change causes an action change.

## Experimental Unit

Each example is one fixed DoorKey environment state taken from an existing
GPT-OSS-20B trajectory step, paired with the reasoning trace originally
generated for that state. The environment does not advance during this
experiment.

The trace is deterministically divided at paragraph and sentence boundaries.
Position zero contains no revealed reasoning. Position `k` contains the first
`k` reasoning chunks. At every position, the model receives:

- the same grid and key-possession status;
- the reasoning prefix available at that position;
- a request for either its recommended next action or one belief report.

Thus, action and belief changes across positions reflect changes induced by
revealing more of the same reasoning trace, while the correct environment state
remains fixed.

These chunks are sentence-like text units, not manually annotated reasoning
operations. Planning, checking, backtracking, and correction labels are an
exploratory follow-up.

## Shared Data

The study has three cohorts:

| Cohort | States | Composition | Realized or estimated reasoning-prefix positions | Planned belief queries |
|---|---:|---|---:|---:|
| Pipeline pilot | 24 | balanced failures and controls | 719 | 9,344 completed |
| Exact-matched analysis | 46 | 23 failure states and 23 controls | 5,553 realized | 72,165 |
| Expanded analysis | 185 | 124 failure states and 61 controls | 28,874 | up to 375,362 |

The exact-matched controls match failure states on key possession, visibility
of a closed door, number of tied optimal actions, and remaining optimal
distance in bins of width five. The expanded cohort contains backtrack,
short-loop, avoidable-detour, two-cycle oscillation, wall-hit, and repeated
inaction failures.

Failure categories describe what happened in the original rollout from the
selected state. They do not determine whether a separately elicited
prefix-conditioned recommendation is optimal.

| Failure category | Exact-matched states | Expanded states |
|---|---:|---:|
| No recorded failure | 23 | 61 |
| Backtrack | 13 | 57 |
| Short loop | 3 | 49 |
| Avoidable detour | 4 | 9 |
| Two-cycle oscillation | 1 | 4 |
| Wall hit | 1 | 3 |
| Repeated inaction | 1 | 2 |

All equally short actions returned by the DoorKey-aware planner are accepted as
optimal. The planner accounts for key acquisition and door passage.

## Experiment 1: Activation-Level Monitor of Action-Optimality Transitions

### Method

This experiment asks whether generic hidden-state geometry changes when a
prefix-elicited recommendation changes optimality. It does not decode a named
belief from the activations.

The completed pilot used:

- 24 DoorKey states: 12 failure states and 12 controls;
- 719 prefix-action evaluations, of which 718 parsed successfully;
- reasoning traces divided into at most 32 paragraph or sentence-like chunks;
- local teacher-forced inference with `openai/gpt-oss-20b`;
- decoder layers 8, 15, and 23;
- the activation at the last token of each chunk and the mean activation over
  the chunk;
- corrected token spans with zero overlap between adjacent chunks.

For each layer and representation type, measure:

- update norm between adjacent reasoning chunks;
- cosine similarity between adjacent chunks;
- overall-direction agreement: cosine between the current activation update
  and the trace's overall first-to-final activation direction;
- cosine similarity to an empirical anchor built from traces whose
  prefix-elicited actions remain optimal.

Overall-direction agreement near `1` means the current activation update
continues along the trace's net first-to-final activation direction; a value
near `0` means it moves orthogonally to that direction; and a value near `-1`
means it reverses direction. It measures activation-trajectory continuity, not
reasoning correctness. The corresponding internal CSV field is
`aligned_change`.

The primary event is a sustained optimal-to-suboptimal recommendation
transition. At each event, compare the metric at the transition with its mean
over the preceding three chunks. No threshold is applied to the continuous
activation metrics. The sustained-transition definition uses the heuristic
that at least half of later valid recommendations remain suboptimal.

### Completed Result

Five of 24 states contain a sustained transition: four failure states and one
control. In the original event-centered analysis, mean-pooled
overall-direction agreement decreases at the four failure-state transitions
across layers 8, 15, and 23, but individual states vary substantially. That
event-centered result used the original activation artifact, whose adjacent
chunk spans overlap, and must be rebuilt on the corrected artifact before
substantive interpretation.

The clean corrected-artifact comparison finds that state-level
overall-direction agreement, adjacent-step cosine, optimality-anchor cosine,
and update norm overlap strongly among sustained transitions, transient
suboptimal recommendations, and traces that remain optimal. The pilot
therefore establishes feasibility but not a reliable activation-level monitor
or preferred layer.

### Scaling Proposal

After the matched prefix-action labels are complete, collect local activations
for those exact 46 states without rerunning behavioral queries. Test whether
each activation metric at position `k` predicts an optimality transition at
`k+1`, compared with constant-rate and reasoning-progress baselines. Evaluate
with grouped cross-validation, keeping source trajectories and matched pairs
together. Report AUROC, Brier score, log loss, event-aligned curves, and
grouped-bootstrap uncertainty intervals.

Before scaling, rebuild the event-centered pilot table using the corrected
non-overlapping activation artifact.

Treat this as a monitor only if a metric improves held-out discrimination
without worsening calibration relative to the null. Do not interpret an
activation-geometry change as a belief change without a validated decoder.

## Experiment 2: Behavioral Belief Shifts and Action Changes

### Measurements at Each Reasoning Position

First, greedily query the model for one recommended action from `UP`, `DOWN`,
`LEFT`, and `RIGHT`. Evaluate it against the stored set of planner-optimal
actions.

Then issue up to 13 separate greedy behavioral belief queries:

- four adjacent-wall beliefs: left, right, up, and down;
- current key possession;
- whether any door is currently open;
- coordinates of the agent, goal, key, and door;
- when the recommended action parses successfully, three predicted
  consequences of that action: hitting a wall,
  holding the key afterward, and leaving a door open afterward.

Each belief answer is compared with ground truth from the fixed environment
state. Every prompt, raw response, parsed answer, and parse error is retained.

The primary belief set is the four wall beliefs and key possession. Door-open
belief is excluded from the primary predictive models because its ground truth
is constant in the pilot. Coordinates and action consequences are exploratory,
except for the action-conditioned model described below.

### Outcomes and Events

The primary outcome is:

`Does an optimal recommendation at position k become suboptimal at position k+1?`

The secondary outcome is:

`Does a suboptimal recommendation at position k become optimal at position k+1?`

Event-level analyses additionally distinguish:

- sustained optimality loss: at least two of the next three valid
  recommendations are suboptimal;
- transient optimality loss: an optimal recommendation becomes suboptimal but
  does not satisfy the sustained rule;
- recovery: a suboptimal recommendation becomes optimal;
- belief shift: a probe answer changes between adjacent positions;
- belief-error onset: a correct belief becomes incorrect;
- belief-error recovery: an incorrect belief becomes correct;
- persistent belief error: the error is present in at least two of the next
  three valid measurements.

The lead or lag of a belief event is:

`belief-event position - action-event position`

The primary event window is three reasoning positions before or after an
action event. The analysis also reports the stricter definition where an
optimality loss persists to the end of the trace. No learned score is converted
to an event using a manually selected threshold.

### Prespecified Hypotheses

1. Belief-error onsets and increases in the number of changing beliefs predict
   optimality loss at the next reasoning position.
2. Belief-error recoveries predict action recovery.
3. Beliefs directly relevant to the recommended action are more predictive
   than unrelated state-belief errors.
4. Persistent belief errors are more strongly associated with sustained than
   transient optimality loss.
5. If beliefs compose into a useful action model, task-motivated
   belief-action conjunctions improve held-out prediction beyond additive
   belief errors.

### Predictive Models

Fit the following regularized logistic models separately for optimality loss
and recovery:

1. Constant event-rate null.
2. Reasoning-progress baseline.
3. General belief errors: reasoning progress plus errors in four wall beliefs
   and key possession.
4. Belief dynamics: reasoning progress plus the counts of belief changes,
   error onsets, and error recoveries at position `k`.
5. General beliefs and belief dynamics.
6. Action-conditioned beliefs: whether the wall belief in the recommended
   direction is wrong or reports blockage, and whether the predicted
   consequence of that action is wrong or predicts a wall hit.
7. Combined general and action-conditioned beliefs.
8. Belief-action chain: the combined model plus two task-motivated
   conjunctions linking reported blockage and predicted wall hit.

Do not multiply raw categorical answers or search every possible interaction.
Evaluate learned models with grouped cross-validation. Keep all selected states
from the same source trajectory in one fold; in the exact-matched analysis,
also keep both members of each matched pair in one fold. Report AUROC, Brier
score, and log loss, with grouped-bootstrap uncertainty intervals in the
confirmatory analyses. The analysis unit is a selected fixed state and its
complete reasoning trace, not an individual reasoning position.

For individual belief events, report event coverage, precision, false-alarm
rate, lead-lag distributions, and false-discovery-rate-corrected exploratory
tests. Only describe an ordered belief chain as recurring if it appears in at
least five independent trajectories.

### Completed Pilot

The 24-state pilot passed the response-parsing gate:

- 719 reasoning positions;
- 9,344 belief queries;
- 96.5 percent valid belief responses;
- 74 action-change events;
- 1,114 belief-answer changes.

Descriptive position-level tests found large associations between several wall
and key-possession errors and a suboptimal next recommendation. These estimates
are not treated as confirmatory because positions repeat within only 24 states.

The stricter trajectory-held-out predictive analysis did not find evidence that
the measured beliefs reliably predict optimality loss:

- constant-rate null AUROC: 0.500;
- strongest learned model, belief dynamics: 0.489;
- action-conditioned model: 0.288;
- belief-action-chain model: 0.196.

For recovery, belief dynamics reached AUROC 0.583 but had worse Brier score and
log loss than the constant-rate null. The pilot therefore does not support a
stable compositional belief-chain claim. Complete-case prediction used 14
trajectories with 19 optimality-loss events and six trajectories with 21
recovery events, which is too small for a definitive negative conclusion.

### Completed Exact-Matched Analysis

The exact-matched behavioral run completed all 72,165 planned belief queries
over 46 states and 5,553 reasoning-prefix positions, with no failed queries and
a 96.0 percent valid-answer rate. It contains 62 sustained and 67 transient
optimality losses, plus 135 recoveries.

The predictive analysis keeps complete source trajectories and both members of
each matched pair in one cross-validation fold. These dependencies reduce the
46 states to 11 independent validation groups. Grouped-bootstrap intervals
therefore remain important.

For upcoming optimality loss, the combined model reaches AUROC 0.603. Its
improvement over the reasoning-progress baseline is 0.034, with a 95 percent
grouped-bootstrap interval from -0.013 to 0.057. It does not improve Brier
score or log loss. The matched run therefore does not establish that measured
beliefs reliably predict when an optimal recommendation will become
suboptimal.

For recovery, the general belief-error model reaches AUROC 0.758 and improves
Brier score by -0.028 and log loss by -0.053 relative to reasoning progress.
The grouped-bootstrap intervals for both calibration improvements exclude
zero. Action-conditioned features and the task-motivated conjunctions do not
improve on general belief errors.

No ordered belief-change chain satisfies the prespecified recurrence threshold:
the most common candidate occurs near only three events. No chain should
advance to causal testing from this run.

The expanded cohort is useful for testing failure-category heterogeneity,
validating the recovery signal, and increasing the number of independent
source trajectories. It should not be framed as simple power scaling intended
to turn the unsupported optimality-loss hypothesis into a positive result.

## Relationship Between Activation Monitoring and Representational Probes

Experiment 1 uses model-agnostic activation geometry. It can indicate that the
hidden-state trajectory changed, but cannot identify which state belief
changed.

Representational belief probes are excluded from the primary study. Released
white-box probes do not transfer reliably from their fixed pre-reasoning and
post-reasoning positions to intermediate reasoning boundaries. Corrected
non-overlapping step activations exist for the pilot, and position-general
linear probes show preliminary feasibility for key possession and some wall
directions, but they are not yet validated measurements.

A later representational analysis must train position-general probes on the
expanded activation collection, split complete trajectories or canonical
states across train and test sets, include open-door states, and report
performance by reasoning progress before comparing white-box and behavioral
belief shifts.

Activation Oracle is also excluded from the primary analysis because the
available Oracle adapter targets Qwen3-8B rather than GPT-OSS-20B.

## Joint Analysis of Experiments 1 and 2

After collecting activation geometry and behavioral beliefs for the same
matched states and reasoning positions, compare four next-position transition
models:

1. reasoning-progress baseline;
2. activation metrics only;
3. behavioral beliefs only;
4. activation metrics and behavioral beliefs together.

This tests whether activation geometry provides an earlier but less
interpretable warning signal, whether measured beliefs explain transition
risk beyond generic geometry, or whether both capture the same variation.
Report held-out predictive performance and the lead or lag of each signal
relative to sustained transitions and recoveries.

Treat this as an incremental-prediction analysis, not causal mediation. A
causal claim requires the interventions in Experiment 3.

## Experiment 3: Deferred Causal Study

Only after the observational gate is passed, select approximately 30 reasoning
chunks associated with recurring, predictive belief changes. For each chunk,
compare:

- the original chunk;
- deletion;
- a semantically different replacement;
- insertion of a correct belief statement;
- insertion of an incorrect counterfactual belief statement.

Generate ten continuations per condition and measure downstream belief
answers, recommended-action identity, and action optimality. For a supported
belief pair, intervene on each belief separately and jointly to test whether
the joint effect exceeds the sum of individual effects.

## Reproducible Execution

Run the matched analysis with:

```bash
bash scripts/run_matched_belief_transition_pipeline.sh
```

After the matched decision gate, run the expanded analysis with:

```bash
bash scripts/run_expanded_belief_transition_pipeline.sh
```

The pipelines checkpoint every prefix-action query, completed state, and
belief query. Analysis outputs include per-position measurements, event and
lead-lag tables, trajectory-held-out model predictions, summary tables, and
event-aligned figures.
