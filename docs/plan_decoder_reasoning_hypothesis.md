# Plan-Decoder Reasoning Hypothesis

This note saves a future experiment direction that is scientifically useful but
currently blocked on activation availability.

## Hypothesis

Reasoning degrades longer-horizon planning information while sharpening local
action-readiness.

Operationally:

- `post` activations should make the **next action** more decodable
- `pre` activations should make **longer action prefixes** more decodable

This is a representational hypothesis about planning information, not a claim
about cognitive maps or grid-state knowledge.

## Why this is useful

This hypothesis fits the observed pattern that some agents appear locally
plausible but globally unstable, including back-and-forth behaviour. If the
effect is real, it would support a "local above global" account of some
trajectory failures:

- reasoning may help choose an immediate move
- but may reduce decodable long-horizon plan structure

## What probes we actually have

Published Hugging Face checkpoints currently available:

- `decoder_probe_layer15_pre_reasoning.pt`
- `decoder_probe_layer15_post_reasoning.pt`

These look like **plan decoders**, not cognitive-map probes.

Observed checkpoint signature:

- input width `2880`
- `10` learned queries
- `5`-way output head

This matches the paper's plan-decoder description much more closely than the
paper's cognitive-map probe description.

## What the paper says the plan decoder does

The paper describes a decoder trained on:

- three extracted token activations at a chosen layer/stage
- `pre`: final prompt tokens before reasoning starts
- `post`: final reasoning tokens before the first action
- target: a fixed `10`-step executed action sequence

The published metric is prefix accuracy:

- `prefix@N = 1` is immediate next-action decodability
- larger `prefix@N` measures longer-horizon plan decodability

## Valid experiment with current probe access

If real activations become available, a valid activation-level experiment is:

1. run the published `pre` decoder on matched `pre` activations
2. run the published `post` decoder on matched `post` activations
3. compare:
   - next-action accuracy
   - prefix accuracy at `N = 2, 3, 5, 10`
4. split results by:
   - clean states
   - failure states
   - backtrack / loop / detour slices where available

Expected pattern:

- `post > pre` at `prefix@1`
- `pre > post` for longer prefixes

## Ideal outputs

Main figure:

- prefix accuracy curves for `pre` and `post` across `N = 1..10`

Supporting figure:

- `post - pre` delta by prefix length

Main tables:

- overall next-action and prefix accuracies
- failure-mode split with:
  - `pre@1`, `post@1`
  - `pre@5`, `post@5`
  - deltas

Case studies:

- a few grids with observed action, optimal action, and decoded pre/post plan
  prefixes

## Current blockers

This experiment is not blocked on probe checkpoints anymore. It is blocked on
which public activation/probe releases are actually usable on this machine.

Current blockers:

- "The broader public \"with cognitive probes\" release is too large for this machine."
- the broader public `trajectories_test_full_with_cognitive_map_probes` release
  is too large for this machine to use directly:
  - file: `trajectories_test_full_with_probes.tar`
  - size: about `41.9 GB`
  - free disk at attempted download time: about `23.3 GB`
- so if we later want failure-focused white-box slices from that tar, we need
  either:
  - more disk, or
  - a narrower release / direct subset from colleagues
- local `openai/gpt-oss-20b` cache is still incomplete for local extraction
- local loading stack is still incomplete for a practical 20B run:
  - `accelerate` missing
  - `bitsandbytes` missing

So the current repo can:

- download and validate the published plan decoders
- use the released `activations_test_full` dataset where compatible
- evaluate them when real activation tensors are provided

But it cannot yet:

- extract those activations locally from GPT-OSS-20B on this machine
- rely on the oversized `with_cognitive_map_probes` tar for failure-focused
  slices on this machine

## Confounders to keep in mind

The empty-grid scaling artefact is still relevant.

For this plan-decoder hypothesis, it is less damaging than for a behavioural
claim if the main comparison is within-state `pre` vs `post`, but it still
matters when comparing across:

- grid sizes
- obstacle densities
- failure categories

Controls:

- stratify by grid size
- keep empty/open grids separate from obstacle grids
- prioritise matched within-state comparisons over pooled cross-size averages

## Status

Saved for future use.

Reusable code already exists in the repo for this direction:

- published plan-decoder loader/eval pipeline
- published probe inspection gate

What is still needed for a real run:

- a valid activation source table, or
- a full local model setup capable of extracting the required activations
