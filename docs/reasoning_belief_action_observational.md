# Does Belief Shifts Change Action Along Reasoning

This experiment tests whether changes in measured state beliefs precede, coincide
with, or follow changes in the action recommended by GPT-OSS-20B during a
reasoning trace. The first run is observational and contains no interventions.

## Measurements

At every sentence-delimited reasoning prefix, the experiment uses the action
already elicited by the step-reasoning-drift run and asks separate
prefix-conditioned behavioral questions.

Primary beliefs:

- walls immediately left, right, up, and down;
- whether the agent holds the key;
- whether any door is open.

Exploratory beliefs:

- agent, goal, key, and door coordinates;
- whether the currently recommended action hits a wall, acquires the key, or
  leaves a door open.

Each prompt includes the fixed grid state, key status, and the same revealed
reasoning prefix used to elicit the action. Ground truth is derived from the
stored DoorKey state. The runner stores every prompt and raw response.

## Event Definitions

- Sustained optimal-to-suboptimal transition: an optimal action becomes
  suboptimal and at least two of the next three valid actions are suboptimal.
- Transient optimal-to-suboptimal transition: an optimal action becomes
  suboptimal without satisfying the sustained rule.
- Recovery: a suboptimal action becomes optimal.
- Belief shift: the answer to a belief question changes between adjacent
  reasoning positions.
- Persistent belief error: an error begins and is present at least twice among
  the next three measurements.

The output also reports the stricter criterion where all remaining valid actions
are suboptimal. There are no manually chosen score thresholds.

## Commands

Prepare the exact-matched 46-state cohort and full 185-state cohort:

```bash
uv run reveng-cli prepare_reasoning_belief_action_cohorts
```

Run the observational analysis on an existing step-drift run:

```bash
uv run reveng-cli run_reasoning_belief_action_observational \
  --drift-run-dir data/behavioral_probes/step_reasoning_drift_balanced_v1 \
  --output-dir data/behavioral_probes/reasoning_belief_action_balanced_v1
```

The run is resumable through `belief_rows.jsonl`. Restart the same command with
the same output directory and the default `--resume` setting.

- Each attempted query is appended immediately to `belief_rows.jsonl`.
- Successful queries are skipped after restart.
- Failed queries are retried after restart.
- A truncated final checkpoint line from an abrupt stop is ignored safely.
- `run_status.json` reports completed, remaining, and failed queries.
- `run_config.json` prevents accidentally resuming with a different model,
  prompt preset, candidate dataset, or prefix-action dataset.
- Checkpoint data is flushed after every query and synced to disk every ten
  queries by default. Set `--checkpoint-fsync-every 1` for maximum durability
  at the cost of additional filesystem overhead.

Use `--no-resume` only when intentionally restarting the output directory from
the beginning.

Use `--max-positions 2` for a cheap real-API smoke test. The existing 24-state
drift run contains 719 positions and schedules approximately 9,300 separate
greedy belief queries under the default panel.

## Representational Probe Gate

The original 24-state activation artifact should not be used for the
representational comparison because adjacent sentence ranges share boundary
tokens. Corrected activations have now been generated at:

`data/behavioral_probes/step_reasoning_drift_balanced_v1_corrected_activations`

The corrected artifact assigns shared leading-whitespace tokens to the later
sentence and has zero adjacent-span overlaps.

The released cognitive-map probe is not directly compatible with this
per-position artifact:

- Each corrected last-token or mean-pooled step representation has 2,880
  features.
- The released layer-15 MLP probe expects 8,642 features: three 2,880-feature
  token representations concatenated with two query-coordinate features.
- The released probe was trained for fixed pre-reasoning and post-reasoning
  token positions, not arbitrary sentence boundaries within reasoning.

Therefore, zero overlap establishes that the corrected activations are valid
for step-level geometry, but it does not establish that the released probe can
be applied at intermediate reasoning positions. A representational comparison
requires either:

1. collecting the released probe's expected three-token feature windows at
   each reasoning boundary and validating their use at intermediate positions;
2. or training new per-position probes on the current last-token or mean-pooled
   representation format.

We collected the required three-token boundary windows and ran an exploratory
transfer test on the 24-state artifact. Both released cognitive-map probes and
both released plan decoders failed the validation gate at reasoning
boundaries. The cognitive-map probes had near-chance intermediate directional
wall accuracy and zero agent-location exact accuracy. The plan decoders were
also substantially below their released fixed-position reference results.
The same-state pre-boundary window underperformed as well, so the failure
cannot be attributed only to intermediate-position shift; prompt-format and
task-distribution transfer also matter.

The exploratory predictions are retained under:

`data/behavioral_probes/step_reasoning_drift_balanced_v1_corrected_boundary_windows`

They must not be used as primary white-box belief measurements. The 24-state
artifact is suitable for pipeline validation but is too small for reliable new
probe training. Probe training should use the expanded state collection and
keep trajectories separated across train and test sets.

### Position-General Probe Feasibility

A grouped-state pilot tested whether one linear probe per belief can be
trained across all reasoning positions and then applied throughout reasoning.
All positions from a state were assigned to the same cross-validation fold.
Across three seeds, the best balanced accuracies were 0.695 for key possession,
0.644 for wall-left, 0.607 for wall-right, 0.546 for wall-down, and 0.493 for
wall-up. Door-open was not trainable because every available state had the same
closed-door label.

This supports position-general probing as a feasible direction for some
beliefs, but not as a validated measurement yet. The full run should collect
expanded-state activations, add open-door states, split by trajectory or
canonical grid state, and evaluate both linear and MLP probes by reasoning
progress.

## Outputs

The observational runner writes per-belief and per-position rows, action
transitions, belief shifts, event lead-lag rows, false-discovery-rate-corrected
feature associations, ranked candidates for later interventions, and
event-aligned belief and activation figures.

The mixed-effects model is deliberately deferred until the scaled run because
the 24-state pilot is intended to validate measurement and parsing rather than
support a stable multivariable estimate.
