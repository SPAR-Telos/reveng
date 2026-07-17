# Matched White-Box / Black-Box Probe Schema

This file defines the input table for the matched behavioural-probe run.

The goal is simple:

- the white-box team and the black-box team should talk about the **same state**
- they should answer the **same question**
- we should then compare those answers with the **environment truth** and the **observed action**

## What one row means

One input row means:

> At this one state, for this one question, on this one reasoning split (`pre` or `post`), what was the ground truth, and what did the white-box probe predict?

So:

- **one row is not a whole trajectory**
- **one row is not a whole grid with all questions**
- **one row is one state + one question**

If you have:

- one state
- four questions (`wall_left`, `wall_right`, `wall_up`, `wall_down`)

then you should have **four rows**.

If you have both:

- `pre`
- `post`

for the same state and same question, then you should have **two rows** for that question.

## Concrete example

Suppose we look at trajectory `traj_07`, step `12`.

At that step, we want to evaluate:

- `wall_right`
- `wall_up`

and we have only the `pre` split.

Then the input table should contain these two rows:

| example_id | trajectory_id | step_index | reasoning_split | question_id | ground_truth_label | whitebox_prediction |
|---|---|---:|---|---|---|---|
| ex_007_s12 | traj_07 | 12 | pre | wall_right | yes | yes |
| ex_007_s12 | traj_07 | 12 | pre | wall_up | no | no |

These two rows refer to the **same state**.
They differ only in the **question**.

If we also have the `post` split for `wall_right`, we add one more row:

| example_id | trajectory_id | step_index | reasoning_split | question_id | ground_truth_label | whitebox_prediction |
|---|---|---:|---|---|---|---|
| ex_007_s12 | traj_07 | 12 | post | wall_right | yes | yes |

That third row refers to:

- the same state
- the same question as the first row
- but a different reasoning split

## What the behavioural pipeline does with these rows

Internally, the pipeline groups rows by:

- `example_id`
- `reasoning_split`

That grouped object is one **behavioural probe instance**.

So in the example above:

- the first two rows become one black-box run on state `ex_007_s12`, split `pre`
- the third row becomes one separate black-box run on state `ex_007_s12`, split `post`

After the black-box run finishes, the pipeline expands back out to one joined row per:

- `example_id`
- `reasoning_split`
- `question_id`

This final joined table is the main output:

- `probe_alignment_rows.csv`

## Required columns

These are the minimum columns we need.

| Column | What it means | Example |
|---|---|---|
| `example_id` | Unique id for the state snapshot | `ex_007_s12` |
| `question_id` | Which question is being probed | `wall_right` |
| `ground_truth_label` | Environment truth for that question | `yes` |
| `reasoning_split` | Whether this row is `pre` or `post` reasoning | `pre` |
| `grid_text` or `state_description_text` | The state description the black-box probe will see | 9x9 grid text |

Notes:

- `question_id` may also be provided under the alias `target_name` or `target_variable`
- `ground_truth_label` may also be provided as `ground_truth`
- at least one of `grid_text` or `state_description_text` must be present

## Recommended columns

These are not strictly required for the run, but they make the final analysis much better.

| Column | What it means | Why it matters |
|---|---|---|
| `trajectory_id` | Which trajectory this state came from | lets us trace back to the source run |
| `step_index` | Which step in that trajectory | lets us align with actions and next states |
| `carrying_key` | Whether the agent is carrying the key at this step | needed for comparable state descriptions |
| `observed_action` | The action the model actually took at this step | needed for belief-action gap analysis |
| `optimal_actions_json` | The A*-optimal action set from this state | needed for optimality comparisons |
| `is_optimal_action` | Whether `observed_action` is in the optimal set | quick filter for non-optimal actions |
| `wall_hit` | Whether the observed action leaves the agent in the same position | useful for wall-hit analysis |
| `whitebox_prediction` | The white-box probe prediction for this row | needed for white-box / black-box comparison |
| `whitebox_score` | Confidence or score from the white-box probe | useful but optional |

## Allowed values

Keep these fields simple and consistent.

| Column | Expected format |
|---|---|
| `question_id` | behavioural registry name such as `wall_right`, `hit_wall_after_right`, `is_goal_up` |
| `reasoning_split` | `pre` or `post` |
| `ground_truth_label` | `yes`, `no`, or `unknown` |
| `whitebox_prediction` | ideally `yes`, `no`, or `unknown` |
| `optimal_actions_json` | JSON list such as `["LEFT", "UP"]` |
| `is_optimal_action` | `true` or `false` |
| `wall_hit` | `true` or `false` |

## CSV example

This is a valid small CSV:

```csv
example_id,trajectory_id,step_index,reasoning_split,question_id,ground_truth_label,grid_text,carrying_key,observed_action,optimal_actions_json,is_optimal_action,wall_hit,whitebox_prediction,whitebox_score
ex_007_s12,traj_07,12,pre,wall_right,yes,"# # # ...",false,RIGHT,"[\"UP\"]",false,true,yes,0.91
ex_007_s12,traj_07,12,pre,wall_up,no,"# # # ...",false,RIGHT,"[\"UP\"]",false,true,no,0.88
ex_007_s12,traj_07,12,post,wall_right,yes,"# # # ...",false,RIGHT,"[\"UP\"]",false,true,yes,0.95
```

## Which output file should colleagues read first?

Start with:

- `probe_alignment_rows.csv`

Each row in that file means:

> one state, one question, one reasoning split, with both white-box and black-box answers attached

That is the table to use for:

- white-box / black-box agreement
- finding rows where both probes are correct
- checking whether the observed action was still non-optimal or hit a wall

## Plain-language summary

If you want to explain this in one sentence to a colleague:

> The input table is a long-form table where each row is one state-question pair, and the output table adds the black-box answer so we can compare environment truth, white-box probe, black-box probe, and observed action on exactly the same state.
