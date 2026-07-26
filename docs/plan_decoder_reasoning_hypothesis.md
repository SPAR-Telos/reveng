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

## Current status

The earlier activation and local-model blockers are resolved. The repository
now contains GPT-OSS-20B activations for 1,276 environment states from 95
trajectories at layers 8, 15, and 23. The saved representations include
sentence means and final-token activations, complete-reasoning summaries,
three-token PRE and POST windows, and the final action token.

The remaining work is a compatibility and evaluation task:

- verify that the released decoder's expected three activation positions match
  the stored PRE and POST window tensors exactly;
- verify checkpoint model revision, layer, hidden width, normalization, and
  action-label vocabulary;
- join each state to its executed ten-action target sequence;
- evaluate prefix accuracy without fitting or selecting on the test states.

The oversized public cognitive-probe archive is still unnecessary for this
specific plan-decoder test because the required boundary activations have now
been generated locally.

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

- the decoder-to-activation compatibility check described above;
- a target table containing the next ten executed actions for each state.
