# Experiment 2 Behavioral Belief Probes, 8-State Pilot

## Scope

This run uses the same 8 GPT-OSS-20B DoorKey states used by Experiment 1 with Weisheng's activation-backed chunks. It is behavioral only. No linear probes were trained or applied, and no Activation Oracle run was attempted.

## Inputs

- Prefix actions: `outputs/experiment1_activation_monitor/weisheng_8_state_canonical_prefix_actions/prefix_action_rows.csv`
- State metadata and labels: `outputs/experiment1_activation_monitor/weisheng_8_state_candidates/candidate_rows.csv`
- Model queried: `together_ai/openai/gpt-oss-20b`
- Prefix positions: 250
- Belief queries: 3,250

Each prompt contains the fixed grid state, whether the agent holds the key, and the revealed reasoning prefix used to elicit the action at that position.

## Probe Targets

| Target family | Questions |
|---|---|
| Current nearby walls | Is there a wall immediately left, right, up, and down of the agent? |
| Current task state | Does the agent have the key? Is any door open? |
| Object locations | What are the agent, goal, key, and door coordinates? |
| Action effects | For the currently recommended action, will it hit a wall, have the key afterward, and leave any door open afterward? |

## Ground Truth

Labels are computed from the fixed DoorKey grid state and transition model, not by human annotation. For example, if the grid has `A` at row 3, column 5 and `#` at row 3, column 4, then `wall_left = yes`. If the recommended action is `DOWN`, the transition model simulates one DoorKey move down and labels whether that move hits a wall, acquires the key, or leaves a door open.

## Results

| Measure | Value |
|---|---:|
| Valid response parse rate | 96.6% |
| Belief shifts | 334 |
| Action events | 21 |
| Optimal to suboptimal transitions | 5 |
| Suboptimal to optimal recoveries | 6 |
| Ranked candidates for later intervention | 3 |

By target family, parse rates were 100.0% for coordinates, 98.3% for action effects, 97.8% for key and door state, and 91.2% for wall questions.

## Interpretation

The pilot supports the measurement pipeline: prefix-conditioned belief probes can be run at every reasoning chunk, parsed reliably, and aligned to action changes. Descriptively, door-open errors and action-effect errors are the most visible candidates near optimality changes. Three persistent belief-error onsets occurred before or at sustained optimal-to-suboptimal transitions, making them candidates for the later intervention study.

The held-out predictive models are not interpretable in this 8-state run because all states come from one source trajectory. Trajectory-held-out validation therefore has only one validation group. The next behavioral run needs multiple independent trajectories before comparing reasoning-progress, general belief-error, and action-conditioned models.

## Main Artifacts

- `belief_rows.csv`: one row per prefix and belief question.
- `belief_shift_rows.csv`: adjacent changes in belief answers.
- `belief_action_lead_lag_rows.csv`: belief shifts aligned to action events.
- `belief_transition_indicator_summary.csv`: leading, coincident, and lagging indicator summaries.
- `ranked_intervention_candidates.csv`: candidate belief shifts for a later causal intervention.
- `figs/event_aligned_belief_errors.png`: belief-error rate around action events.
- `figs/belief_transition_indicator_heatmap.png`: belief-change coverage of transition events.

## Regression Outputs

The pilot writes `transition_model_full_data_coefficients.csv`, but these are standardized coefficients from full-data exploratory logistic models. They are not a conventional inferential regression table with standard errors and p-values. Because the 8 states all come from one source trajectory, trajectory-held-out regression is not estimable beyond the constant prevalence baseline. A conventional regression table should be produced only after scaling to multiple independent trajectories.

## Counterintuitive Points in the Pilot

- Door-open reports have the largest error rate among current-state probes, even though the door state is visually available in the grid. This suggests the model may under-use or misread the door symbol under prefix-conditioned reasoning.
- For sustained optimal-to-suboptimal transitions, the mean primary belief error rate does not clearly rise before the event. It rises after the event in this pilot, so it is better interpreted as a lagging diagnostic than a real-time leading indicator.
- Coordinate probes parse well and are mostly correct, while nearby wall probes parse worse. This is counterintuitive if local geometry should be easier than absolute coordinates, and it may reflect prompt-format sensitivity for yes/no wall questions.
