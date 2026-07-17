# Behavioral-Only Scaling

The completed 24-state pilot contains 719 reasoning positions and 9,344 belief
queries. It cost approximately 4.77 USD including prefix-action queries.

The next runs should exclude activation collection and white-box outputs.

## Matched 46-State Run

Measured workload from deterministic reasoning segmentation:

- 7,217 prefix-action queries;
- up to 93,821 belief queries;
- 21 to 376 positions per state, with a median of 143.

As of June 7, 2026, 29 states and 4,276 prefix positions are checkpointed.
Belief probing has not started.

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/reveng-cli run_step_reasoning_drift_experiment \
  --candidate-rows-path data/behavioral_probes/reasoning_belief_action_cohorts/matched_46_candidates.csv \
  --output-dir data/behavioral_probes/step_reasoning_drift_matched_46_behavioral \
  --slice-mode trajectory_candidates \
  --trajectory-slice-type selection_candidates \
  --no-collect-activations

MPLCONFIGDIR=/tmp/matplotlib .venv/bin/reveng-cli run_reasoning_belief_action_observational \
  --drift-run-dir data/behavioral_probes/step_reasoning_drift_matched_46_behavioral \
  --candidate-rows-path data/behavioral_probes/reasoning_belief_action_cohorts/matched_46_candidates.csv \
  --output-dir data/behavioral_probes/reasoning_belief_action_matched_46_behavioral
```

## Expanded 185-State Run

Measured workload from deterministic reasoning segmentation:

- 28,874 prefix-action queries;
- up to 375,362 belief queries;
- 19 to 456 positions per state, with a median of 145.

Use the same commands with
`expanded_185_candidates.csv`,
`step_reasoning_drift_expanded_185_behavioral`, and
`reasoning_belief_action_expanded_185_behavioral`.

Both commands are resumable. The prefix-action runner checkpoints after each
prefix query and each completed state, and the belief runner checkpoints after
each query. Because trace lengths vary substantially, state count alone is not
a reliable progress or runtime estimate.
