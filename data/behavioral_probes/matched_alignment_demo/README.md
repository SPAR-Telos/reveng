# Matched Alignment Demo

This folder contains a minimal matched-input demo for the white-box / black-box
alignment pipeline.

## What is in the demo

File:

- `matched_probe_rows_demo.jsonl`

This file contains:

- `2` state snapshots
- `2` questions per state
  - `wall_right`
  - `wall_up`
- `2` reasoning splits
  - `pre`
  - `post`

Total:

- `8` input rows

## What one row means

One row means:

> one state, one question, one reasoning split

For example:

- `demo_state_1`
- `wall_right`
- `pre`

is one row.

The same state with:

- `wall_up`

is a second row.

The same state and question with:

- `post`

is another row again.

## Where the states come from

Both states are derived from the same 9x9 DoorKey layout sent by the white-box team.

Base layout:

```text
#########
#___#G__#
#___#___#
#_______#
#D#######
#___#K__#
#___#___#
#_______#
#########
```

### State 1

Agent at row `1`, col `3`:

```text
#########
#__A#G__#
#___#___#
#_______#
#D#######
#___#K__#
#___#___#
#_______#
#########
```

Labels used:

- `wall_right = yes`
- `wall_up = yes`

Metadata used:

- `observed_action = RIGHT`
- `wall_hit = true`
- `is_optimal_action = false`

### State 2

Agent at row `3`, col `3`:

```text
#########
#___#G__#
#___#___#
#__A____#
#D#######
#___#K__#
#___#___#
#_______#
#########
```

Labels used:

- `wall_right = no`
- `wall_up = no`

Metadata used:

- `observed_action = RIGHT`
- `wall_hit = false`
- `is_optimal_action = true`

## What this demo is for

This demo is only for checking:

- schema clarity
- row grouping
- `pre` / `post` split handling
- question alignment
- white-box / black-box join format

It is not meant to be a scientific result.

## Mocked end-to-end artefacts

This folder also contains mocked output files:

- `mock_behavioral_probe_rows.csv`
- `mock_probe_alignment_rows.csv`
- `mock_probe_alignment_summary.csv`
- `mock_probe_alignment_gap_cases.csv`
- `mock_artifact_summary.json`

These files were generated without calling the model API.

They are only meant to show colleagues:

- what the joined output table looks like
- what the summary table looks like
- how disagreement flags appear in practice

One row is intentionally set up as a disagreement example:

- `example_id = demo_state_2`
- `reasoning_split = post`
- `question_id = wall_up`

For that row:

- `ground_truth_label = no`
- `whitebox_prediction = no`
- mocked black-box answer = `yes`

This gives one concrete example of a `whitebox_correct_blackbox_wrong` case.

## Actual model-produced artefacts

The following directories were produced by the chosen model:

- `../matched_alignment_demo_actual/pre/`
- `../matched_alignment_demo_actual/post/`

Model used:

- `together_ai/openai/gpt-oss-20b`

Commands used:

```bash
uv run reveng-cli run_behavioral_probe_matched_eval \
  --matched-rows-path data/behavioral_probes/matched_alignment_demo/matched_probe_rows_demo.jsonl \
  --model-name together_ai/openai/gpt-oss-20b \
  --output-dir data/behavioral_probes/matched_alignment_demo_actual/pre \
  --reasoning-split pre \
  --question-family wall_directional \
  --greedy-repeats 1 \
  --mc-sample-repeats 1 \
  --logprob-temperatures 0.0 0.7 1.0

uv run reveng-cli run_behavioral_probe_matched_eval \
  --matched-rows-path data/behavioral_probes/matched_alignment_demo/matched_probe_rows_demo.jsonl \
  --model-name together_ai/openai/gpt-oss-20b \
  --output-dir data/behavioral_probes/matched_alignment_demo_actual/post \
  --reasoning-split post \
  --question-family wall_directional \
  --greedy-repeats 1 \
  --mc-sample-repeats 1 \
  --logprob-temperatures 0.0 0.7 1.0
```

These runs are still only a tiny pipeline check. They are not the real matched
experiment with white-box exports yet.

## Local validation

This local command checks that the file loads and groups correctly without calling the model API:

```bash
uv run python -c "from reveng.experiments.behavioral_probe_alignment import load_matched_probe_rows, build_behavioral_probe_instances_from_matched_rows; rows = load_matched_probe_rows('data/behavioral_probes/matched_alignment_demo/matched_probe_rows_demo.jsonl', reasoning_split=None, question_family='wall_directional'); instances = build_behavioral_probe_instances_from_matched_rows(rows); print({'n_rows': len(rows), 'n_instances': len(instances), 'instances': [(x['example_id'], x['reasoning_split'], x['allowed_question_ids']) for x in instances]})"
```

Expected result:

- `n_rows = 8`
- `n_instances = 4`

Those `4` instances are:

- `demo_state_1`, `pre`
- `demo_state_1`, `post`
- `demo_state_2`, `pre`
- `demo_state_2`, `post`
