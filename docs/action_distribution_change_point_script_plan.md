# Script Plan: Sentence-Level Action-Distribution Change Points

## Goal

Test whether GPT-OSS-20B reaches its final recommended action through a sharp
sentence-level change in the probability distribution over `UP`, `DOWN`,
`LEFT`, and `RIGHT`, and whether that change agrees with the existing stable
action boundary.

The primary run is offline. It uses the 7,084 stored action distributions from
46 matched DoorKey states across 31 trajectories. It requires no model
inference and no new activations.

## Claims and Tests

| Claim | Test | Evidence required |
|---|---|---|
| Commitment is abrupt | Compare the largest increase in final-action probability with the second-largest increase in each state | Median largest-to-second-largest ratio above 1, with a trajectory-bootstrap interval |
| The probability jump and stable-action boundary identify the same transition | Compare their sentence and normalized-character positions | Median absolute offset and fraction within 1, 3, and 5 sentences |
| Reasoning after the boundary changes the action distribution less | Compare distance to the full-trace distribution before and after the boundary | Lower post-boundary Jensen-Shannon divergence and total variation |
| Failure and control states commit differently | Compare boundary position, abruptness, and post-boundary stability in matched groups | Matched-pair estimates with trajectory-bootstrap intervals |
| The boundary has an internal correlate | Compare activation metrics and final-action attention near the probability jump with progress-matched non-event positions | Trajectory-bootstrap differences using existing activations and attention |

Do not use action-distribution divergence to “predict” recommendation changes:
both are derived from the same action probabilities. Recommendation-change
alignment is a descriptive validity check, not independent evidence.

## Definitions

For state \(i\) and sentence boundary \(t\), parse the stored normalized vector:

\[
q_{i,t}(a),\quad
a \in \{\mathrm{UP},\mathrm{DOWN},\mathrm{LEFT},\mathrm{RIGHT}\}.
\]

Let \(a_i^*=\arg\max_a q_{i,T}(a)\) be the full-trace recommended action.

Compute:

- final-action probability: \(p_{i,t}=q_{i,t}(a_i^*)\);
- final-action probability change:
  \(\Delta p_{i,t}=p_{i,t}-p_{i,t-1}\);
- adjacent Jensen-Shannon divergence:
  \(JS(q_{i,t-1},q_{i,t})\), using base-2 logarithms;
- adjacent total variation:
  \(\frac{1}{2}\sum_a|q_{i,t}(a)-q_{i,t-1}(a)|\);
- distance to the full-trace distribution:
  \(JS(q_{i,t},q_{i,T})\) and total variation;
- action entropy, confidence, and their adjacent changes.

Define three boundaries:

1. `final_action_jump`: sentence with maximum \(\Delta p_{i,t}\);
2. `largest_distribution_change`: sentence with maximum adjacent
   Jensen-Shannon divergence;
3. `stable_action_boundary`: existing earliest sentence whose argmax equals
   \(a_i^*\) and remains equal through the trace.

Use deterministic earliest-position tie breaking. Do not introduce a
hand-tuned threshold in the primary analysis.

Report two abruptness measures:

- largest positive \(\Delta p\) divided by the second-largest positive
  \(\Delta p\);
- largest adjacent Jensen-Shannon divergence divided by the second-largest.

If fewer than two positive changes exist, record the ratio as missing and
retain the raw values.

## Source Data

Primary:

`outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1/position_rows.csv`

Required columns:

- `example_id`, `trajectory_id`, `matched_pair_id`, `matched_role`;
- `position_index`, `reasoning_step_idx`, `reasoning_progress`;
- `action_probabilities_json`, `action_label`, `action_is_optimal`;
- `final_full_trace_action`, `commitment_onset`;
- `primary_step_failure_mode`, `trajectory_class`.

Integration inputs:

- action events:
  `outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1/event_rows.csv`;
- activation metrics:
  `outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1/geometry_rows.csv`;
- belief measurements:
  `outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1/belief_rows.csv`;
- canonical sentence text:
  `data/behavioral_probes/doorkey_chunking_validation/sentences.csv`;
- compact final-action attention:
  `outputs/hypothesis_tests/attention_decision_relevance_v1/attention_index.csv`
  and its safetensors shards.

The stored action vectors are normalized over the four action candidates.
They do not measure probability assigned to non-action text. The report must
state this limitation.

## Code Changes

### 1. Add a pure analysis module

Create:

`src/reveng/experiments/action_distribution_change_points.py`

Functions:

- `parse_action_distribution(value, actions)`;
- `validate_action_distribution(distribution, tolerance=1e-6)`;
- `jensen_shannon_bits(left, right)`;
- `total_variation(left, right)`;
- `compute_position_metrics(rows)`;
- `select_change_points(position_metrics)`;
- `compute_boundary_offsets(change_points)`;
- `build_event_aligned_rows(position_metrics, change_points, window=10)`;
- `build_progress_matched_controls(...)`;
- `summarize_by_state_group(...)`;
- `trajectory_bootstrap_ci(...)`;
- `build_semantic_judge_candidates(...)`.

Keep this module independent of plotting and model loading.

### 2. Add the main offline command

Create:

`scripts/build_action_distribution_change_points.py`

Arguments:

- `--position-rows`;
- `--event-rows`;
- `--geometry-rows`;
- `--belief-rows`;
- `--sentences`;
- `--attention-index`;
- `--output-dir`;
- `--event-window`, default `10`;
- `--control-progress-tolerance`, default `0.03`;
- `--bootstrap-repeats`, default `2000`;
- `--seed`, default `42`;
- `--skip-attention`;

Default output:

`outputs/hypothesis_tests/action_distribution_change_points_v1/`

Execution order:

1. validate source rows and four-action distributions;
2. compute per-position action-distribution metrics;
3. identify the three boundaries per state;
4. reproduce the existing stable-action boundaries exactly;
5. compute within-state boundary offsets and post-boundary stability;
6. compare failure and control states;
7. join activation, belief, and attention measurements;
8. export semantic-judge candidates;
9. write tables, figures, captions, report, and manifest.

### 3. Reuse attention without extracting it again

Refactor the event-window comparison from
`scripts/run_attention_decision_relevance.py` into:

`src/reveng/experiments/attention_event_windows.py`

Allow the new command to pass `final_action_jump` and
`largest_distribution_change` as event positions. Use the existing final-action
attention tensors. Compare each seven-sentence event window with a
non-overlapping same-state window at similar reasoning progress.

Do not regenerate attention.

### 4. Add an optional sampling sensitivity command

Create only after the offline result passes validation:

`scripts/run_action_distribution_sampling_sensitivity.py`

Select, per state:

- the `final_action_jump`;
- one same-state progress-matched control;
- the stable-action boundary when it differs from the jump by more than three
  sentences.

Run 30 action samples per selected position at temperature `0.7`, top-p `0.95`,
with seeds derived deterministically from base seed `42`. Save checkpoints
after every position and support `--resume`.

Compare empirical action frequencies with the stored logprob distribution
using Jensen-Shannon divergence. This is a sensitivity analysis, not the
primary estimator.

Expected maximum is 138 selected positions and 4,140 sampled actions, rather
than 212,520 samples over every sentence.

## Outputs

Write under:

`outputs/hypothesis_tests/action_distribution_change_points_v1/`

Data:

- `position_distribution_metrics.csv`: one row per sentence position;
- `state_change_points.csv`: three boundaries and abruptness statistics per
  state;
- `boundary_alignment_rows.csv`: offsets among boundary definitions;
- `event_aligned_action_distribution.csv`: probability trajectories around
  each boundary;
- `post_boundary_stability.csv`;
- `state_group_summary.csv`;
- `activation_boundary_summary.csv`;
- `belief_boundary_summary.csv`;
- `attention_boundary_summary.csv`;
- `semantic_judge_candidates.jsonl`;
- `run_manifest.json`.

Reports:

- `run_report.md`;
- `FIGURE_CAPTIONS.md`;
- `VALIDATION_REPORT.md`.

Reader-facing figures:

1. `figs/final_action_probability_around_jump.png`
   - x-axis: `Sentence position relative to largest increase in final-action probability`;
   - y-axis: `Probability assigned to the full-trace action`;
   - median and interquartile range across states;
   - failure and control panels;
   - state counts shown.
2. `figs/probability_jump_vs_stable_commitment.png`
   - x-axis: `Largest probability-jump position (fraction of reasoning characters)`;
   - y-axis: `Stable-action position (fraction of reasoning characters)`;
   - diagonal equality line;
   - points colored by failure or control status.

Diagnostics only:

- abruptness-ratio histogram;
- adjacent Jensen-Shannon divergence traces for selected cases;
- boundary-offset histogram.

Do not create a figure merely showing that Jensen-Shannon divergence is high
when the argmax action changes.

## Semantic-Validation Handoff

For each unique boundary, export:

- current and previous exact sentence text;
- preceding reasoning context reference;
- boundary type and score;
- action probabilities before and after;
- action and optimality change;
- failure taxonomy;
- a same-state progress- and length-matched non-event sentence.

Hide boundary, action, activation, attention, and outcome labels in the judge
input. The later semantic experiment will classify planning, reiteration,
route rechecking, consolidation, correction, backtracking, new inference, and
procedural continuation.

This permits tests of whether the same semantic categories:

- mark probability change points;
- explain activation similarity;
- receive elevated final-action attention;
- differ between optimal and suboptimal commitments;
- dominate post-commitment reasoning.

## Validation

Unit tests in:

`tests/test_action_distribution_change_points.py`

Test:

- identical distributions produce zero divergence;
- disjoint distributions produce one bit of Jensen-Shannon divergence;
- total variation is within `[0, 1]`;
- maximum-jump and maximum-divergence boundaries are correct on synthetic
  trajectories;
- ties select the earliest position;
- stable-action boundary reproduces the existing definition;
- malformed, negative, missing, or non-normalized distributions fail clearly;
- one-sentence and constant-action traces do not crash;
- bootstrap results are deterministic under seed `42`.

Integration acceptance:

- exactly 46 states, 31 trajectories, and 7,084 positions;
- exactly four finite probabilities per position, summing to one within
  tolerance;
- exactly one boundary of each type per eligible state;
- all 46 stored stable-action boundaries reproduced;
- existing event counts reproduced: 682 recommendation changes, 115 sustained
  optimality losses, 46 transient optimality losses, and 182 recoveries;
- every nonzero sentence position joins to canonical text;
- activation and attention joins report explicit coverage rather than silently
  dropping states;
- every figure has explicit axes, sample sizes, and a standalone caption.

## Decision Rules

Proceed to post-boundary truncation if:

- probability jumps are concentrated in one sentence rather than diffuse;
- the probability-jump and stable-action boundaries usually occur within
  three sentences;
- action distributions remain closer to the full-trace distribution after the
  boundary.

Do not claim a discrete commitment mechanism if:

- largest and second-largest probability changes are similar;
- boundary definitions disagree substantially;
- post-boundary action distributions continue to drift.

Proceed to semantic judging if boundary sentences can be matched to suitable
same-state controls. Proceed to sampling sensitivity only if the stored
logprob result is promising or candidate normalization is questioned.

## Estimated Cost

| Stage | Time | Model inference | Additional disk |
|---|---:|---:|---:|
| Module, command, tests, reports | 3–5 hours engineering | None | Less than 100 MB |
| Matched-46 offline analysis | Under 5 minutes | None | Approximately 10–30 MB |
| Existing activation and attention joins | Under 5 minutes | None | Approximately 10 MB |
| Optional 30-sample sensitivity | Benchmark first; likely 1–3 hours locally | At most 4,140 short action generations | Less than 50 MB |

Primary-run API cost is zero.

## Execution Status

The matched-46 offline analysis is complete at:

`outputs/hypothesis_tests/action_distribution_change_points_v1/`

The run used 46 states from 31 trajectories and 7,084 sentence positions. It
required no model inference. The final robustness run used 5,000
state-length-preserving random-position comparisons, 2,000 grouped bootstrap
repetitions, and 200 within-state circular-shift permutations.

The original single-boundary decision rule did not pass. The largest increase
in full-trace action probability and stable action selection are within three
sentences in 39 percent of states, with a median absolute separation of 16
sentences. Their agreement is nevertheless stronger than a random valid
sentence baseline. The revised interpretation is a two-stage descriptive
account: a large action-probability revision often precedes stable action
selection.

The current-state belief-error comparison is null. Layer-15 sentence-mean
activation similarity has a contemporaneous association with the probability
revision. It does not improve prediction of the next final-action jump or
stable-selection boundary. It does improve AUROC for predicting the next
sentence's largest change in the complete action distribution, but this result
is representation-dependent and has negligible log-loss improvement.

No repeated-sampling sensitivity or semantic judging was run. The former
would require at most 4,140 new action generations. The latter has 137 blinded
boundary-control sentence pairs ready in `semantic_judge_candidates.jsonl`.
