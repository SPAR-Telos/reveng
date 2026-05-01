# Behavioral Probe Failure-Mode Selection

This note documents how trajectory-derived behavioral-probe candidates are
selected from the downloaded DoorKey trajectory JSONs. It is intentionally
separate from `behavioral_probe_pathologies.md`, which records prompting and
logprob-readout failure modes.

## Source Artifacts

- Input trajectories:
  `data/hf/trajectories_key_door_100/trajectories_key_door/`
- Mined output directory:
  `data/behavioral_probes/trajectory_instances_recomputed_optimal/`
- Main candidate table:
  `data/behavioral_probes/trajectory_instances_recomputed_optimal/trajectory_selection_candidates.csv`
- Full per-step table:
  `data/behavioral_probes/trajectory_instances_recomputed_optimal/trajectory_failure_modes.csv`
- Trajectory manifest:
  `data/behavioral_probes/trajectory_instances_recomputed_optimal/trajectory_manifest.csv`

## Selection Unit

The behavioral-probe unit remains one rendered state from one trajectory step:

- `grid_text`
- `carrying_key`
- `observed_action`
- optional proposed action only through the question wording

No full trajectory history is inserted into probe prompts. Trajectory-level
labels only decide which trajectories and steps are useful to sample.

Audience-facing row labels:

| Display label | Internal field | Meaning |
|---|---|---|
| state just before tagged failure | `is_pre_failure_context=True` | The immediately preceding distinct state before a selected failure step. It is included only for comparison; it is not itself a failure label. |
| tagged failure state | `selection_stage=failure` | The selected step that matched a deterministic failure-mode rule, such as wall hit, backtrack, short loop, freeze repeat, or avoidable detour. |

## Trajectory-Level Judge Criteria

Trajectory class is judged against a recomputed DoorKey shortest-path solver,
not the stored `astar_distance` field. In this dataset, `stored_astar_distance`
is suspicious for most trajectories, so the recomputed value is the safer
selection signal.

| Class | Criterion |
|---|---|
| `optimal_success` | Reached goal and `actual_length == recomputed_optimal_length` |
| `suboptimal_success` | Reached goal and `actual_length != recomputed_optimal_length` |
| `failed` | Did not reach goal, or no recomputed optimal length is available |

Current manifest counts:

| Quantity | Count |
|---|---:|
| trajectories analyzed | 95 |
| `optimal_success` trajectories | 79 |
| `suboptimal_success` trajectories | 16 |
| `failed` trajectories | 0 |
| stored `astar_distance` suspicious | 94 |
| mean actual length | 13.432 |
| mean recomputed optimal length | 13.000 |
| mean length delta among successful trajectories | 0.432 |

## Step-Level Failure Modes

Each trajectory step is replayed under the reconstructed environment state and
tagged deterministically.

| Failure mode | Judge criterion |
|---|---|
| `wall_hit` | Observed action leaves the agent position unchanged. |
| `backtrack` | The action returns the agent to the previous position. |
| `oscillation_2cycle` | Positions follow an `A-B-A-B` pattern. |
| `short_loop` | The next position was visited within the last 6 positions, excluding backtrack and strict 2-cycle cases. |
| `freeze_repeat` | The same full rendered state recurs for at least 2 consecutive steps with no decrease in recomputed optimal distance. |
| `avoidable_detour` | Observed action is not in the DoorKey-aware optimal action set, and it is not a wall hit. |
| `terminal_failure_tail` | Last 3 steps of a failed trajectory. This is a selection label, not a causal failure mode. |
| `pre_failure_context` | The immediately previous distinct state before a selected failure step. Audience-facing label: "previous state before failure." This is comparative context, not a failure mode. |

Full step-level failure counts from the manifest:

| Failure mode | Total step count |
|---|---:|
| `wall_hit` | 3 |
| `backtrack` | 70 |
| `oscillation_2cycle` | 10 |
| `short_loop` | 54 |
| `freeze_repeat` | 3 |
| `avoidable_detour` | 19 |

## Selection Policy

The candidate selector is designed to keep informative steps without letting
long loops dominate the probe set.

| Rule | Selection behavior |
|---|---|
| `wall_hit`, `backtrack`, `avoidable_detour` | Select the trigger step. |
| `oscillation_2cycle`, `short_loop`, `freeze_repeat` | Select the onset step and at most one representative repeated step. |
| `suboptimal_success` trajectory | Select tagged failure steps, capped per trajectory. |
| `failed` trajectory | Select failure onset plus terminal tail steps. There are no failed trajectories in the current 95-trajectory set. |
| `pre_failure_context` | Add `step_index - 1` for selected failure steps when it exists, is distinct, and is not already selected. In plots/reports, display this as "previous state before failure." |
| Deduplication | If adjacent candidates have the same full rendered state and same primary failure mode, keep the earliest. |
| Per-trajectory cap | Keep at most 4 selected rows per trajectory, allowing one context state plus failure rows. |

## Current Candidate Set

`trajectory_selection_candidates.csv` currently contains 185 selected probe
candidates from 60 trajectories.

| Category | Count |
|---|---:|
| selected candidate rows | 185 |
| selected trajectories | 60 |
| failure rows | 124 |
| previous-state-before-failure context rows | 61 |
| failure-onset rows | 60 |
| terminal-tail rows | 0 |
| wall-hit rows | 3 |
| non-optimal-action rows | 18 |

Candidate rows by trajectory class:

| Trajectory class | Selected rows |
|---|---:|
| `optimal_success` | 127 |
| `suboptimal_success` | 58 |

Candidate rows by primary step label:

| Primary step label | Selected rows |
|---|---:|
| `none` / previous-state-before-failure context | 61 |
| `backtrack` | 57 |
| `short_loop` | 49 |
| `avoidable_detour` | 9 |
| `oscillation_2cycle` | 4 |
| `wall_hit` | 3 |
| `freeze_repeat` | 2 |

Candidate rows by selection reason:

| Selection reason | Selected rows |
|---|---:|
| `pre_failure_context` / previous state before failure | 61 |
| `backtrack` | 37 |
| `short_loop_onset` | 29 |
| `short_loop_representative` | 15 |
| `backtrack,suboptimal_success_failure_step` | 15 |
| `avoidable_detour,suboptimal_success_failure_step` | 9 |
| `avoidable_detour,backtrack,suboptimal_success_failure_step` | 5 |
| `backtrack,oscillation_2cycle_onset,suboptimal_success_failure_step` | 3 |
| `suboptimal_success_failure_step,wall_hit` | 3 |
| `short_loop_onset,suboptimal_success_failure_step` | 3 |
| `short_loop_representative,suboptimal_success_failure_step` | 2 |
| `freeze_repeat_onset,short_loop_onset,suboptimal_success_failure_step` | 1 |
| `freeze_repeat_onset,suboptimal_success_failure_step` | 1 |
| `avoidable_detour,backtrack,oscillation_2cycle_onset,suboptimal_success_failure_step` | 1 |

## Balanced Subset Categories

The balanced case-study run samples a smaller set of states so that every
available failure category appears at least once. These are the categories in
`data/behavioral_probes/case_studies_balanced_failure_modes/`.

| Category | Count | Audience-facing definition / selection criterion |
|---|---:|---|
| state just before tagged failure | 4 | A comparison state: the immediately preceding distinct state before a selected failure state. It is not itself counted as a failure. |
| backtrack | 5 | The observed action moves the agent back to its previous position. This can be a mistake, but can also be the optimal recovery action after an earlier bad move. |
| short loop | 3 | The observed action moves to a position visited recently, within a 6-step window, excluding direct backtracks and strict 2-cycles. |
| avoidable detour | 2 | The observed action is not in the A* optimal action set from the current state, and it is not a wall hit. |
| 2-cycle oscillation | 2 | The trajectory follows an `A-B-A-B` positional pattern. |
| wall hit | 2 | The observed action leaves the agent in the same position because the move is blocked by a wall. |
| freeze repeat | 2 | The same full rendered state recurs for at least two consecutive steps without reducing the recomputed A* distance to goal. |

Important: A* optimal actions are recomputed from the state at each step. If
the model makes a bad move at step `t`, the optimal action at step `t+1` is the
best recovery action from that new state, not the action that would have been
optimal from the original trajectory start.

## Why A Table Is Appropriate

The candidate set is produced by deterministic rules, so a table is the right
summary format:

- it makes the judge criteria auditable
- it shows the class imbalance directly
- it separates context rows from actual failure rows
- it makes clear that wall hits are rare in the current pool

For reporting to the team, use the candidate-count tables above plus 2-4
rendered examples from `trajectory_selection_candidates.csv`.

The scaled balanced run also writes a focused wall-probe gap artifact:

- `data/behavioral_probes/case_studies_balanced_failure_modes/belief_action_gap_summary.csv`
- `data/behavioral_probes/case_studies_balanced_failure_modes/figs/belief_action_gap_summary.png`
- `data/behavioral_probes/case_studies_balanced_failure_modes/failure_mode_by_gap.csv`
- `data/behavioral_probes/case_studies_balanced_failure_modes/figs/failure_mode_by_gap.png`

This is the primary table/figure for the current question: whether action-
relevant wall beliefs align with observed actions and A* optimality.

The balanced case-study output now separates:

- main case-selection figure:
  `figs/failure_mode_case_selection.png`
  This shows tagged failure states only.
- appendix case-selection figure:
  `figs/failure_mode_case_selection_with_context.png`
  This adds the comparison states immediately before tagged failures.

Suggested caption for the case-selection figure:

> Trajectory states selected for behavioral-probe case studies. Each row is one
> rendered state that is later queried independently. Rows marked "state just
> before tagged failure" are comparison states immediately preceding a linked
> failure state; they are not themselves counted as failures. Rows marked
> "tagged failure state" are the selected steps that match deterministic
> failure-mode rules such as wall hit, backtrack, loop, freeze, or avoidable
> detour. The observed action is the model's action at that state, and optimal
> actions are recomputed with the DoorKey-aware A* solver.

Suggested caption:

> Wall-probe accuracy and belief-action mismatch on failure-mode-balanced
> trajectory states. The top panel checks whether the model can correctly
> report whether there is a wall in each direction. The bottom panel asks two
> action-facing questions: whether the model moves into a direction it reports
> as walled, and whether it takes a non-A* action despite correctly reporting
> the relevant wall fact. Fractions above bars show gap cases over eligible
> observed-action single-step states in the plotted subset, not trajectories.

Suggested caption for the failure-mode-by-gap figure:

> Belief-action mismatch by failure mode on the balanced trajectory subset.
> Each group corresponds to tagged failure states only, excluding comparison
> context rows. The top panel shows how accurately the model reports whether
> the observed-action direction is blocked by a wall within each failure mode.
> The bottom panel shows two gap rates: whether the model moves into a
> direction it reports as walled, and whether it takes a non-A* action despite
> correctly reporting the relevant wall fact. Fractions above bars are gap
> cases over eligible tagged failure states in the plotted subset, not
> trajectories.

## Probe Accuracy vs Gap Metrics

Probe accuracy and belief-action gap are reported separately.

| Metric | Definition | Interpretation |
|---|---|---|
| probe accuracy | `mean(probe_answer == ground_truth_label)` | Whether the elicited belief matches the environment truth. This should approach 100% when prompts are semantically unambiguous. |
| local belief-action gap rate | `inconsistent / (inconsistent + potentially_consistent)` | Whether the model takes the action that its own wall/action-conditioned belief says is blocked. This uses only local action/belief consistency. |
| A*-conditioned gap rate | `gap / (gap + consistent)` after filtering to correct, valid probe answers about the observed action | Whether the observed action is non-A* even when the probe correctly captures the relevant local fact. |

The A*-conditioned denominator is intentionally conservative. A row contributes
only when:

- the probe question targets the observed action
- A* optimality metadata is available
- the probe answer is valid
- the probe answer matches the ground-truth label

Then:

- `gap`: observed action is not in the recomputed A* optimal action set
- `consistent`: observed action is in the recomputed A* optimal action set

This keeps incorrect probe answers out of the belief-action-gap denominator.
