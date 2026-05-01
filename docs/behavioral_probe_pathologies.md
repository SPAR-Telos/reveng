# Behavioral Probe Pathologies

This note records behavioral-probe failure modes that are worth preserving
across refactors. The main goal is to avoid re-discovering the same issues
after the extraction or prompting code changes.

For trajectory-derived candidate selection rules and counts, see
`docs/behavioral_probe_failure_mode_selection.md`. This file is about
prompting, parsing, and readout pathologies rather than dataset mining.

## Logprobs vs `top_logprobs`

In the OpenAI-compatible API shape used in this repo:

- `logprobs` means per-output-token log probability information is returned.
- `top_logprobs` means the API should also return a bounded list of likely
  alternative tokens for each output token position.

In our behavioral-probe code, we request both on the logprob path:

- [behavioral_probe_runner.py#L277](/root/reveng/src/reveng/experiments/behavioral_probe_runner.py#L277)
- [behavioral_probe_runner.py#L285](/root/reveng/src/reveng/experiments/behavioral_probe_runner.py#L285)
- [behavioral_probe_runner.py#L296](/root/reveng/src/reveng/experiments/behavioral_probe_runner.py#L296)
- [behavioral_probe_runner.py#L304](/root/reveng/src/reveng/experiments/behavioral_probe_runner.py#L304)

The returned structure is a token stream over output content, not a dedicated
"final answer probability" object. That distinction matters for reasoning
models and constrained outputs.

Useful references:

- Together logprobs docs: `https://docs.together.ai/docs/logprobs`
- Together GPT-OSS docs: `https://docs.together.ai/docs/gpt-oss`
- OpenAI logprobs cookbook: `https://developers.openai.com/cookbook/examples/using_logprobs`

## Current Logprob Readout Definition

Current `logprob sampled` is a surfaced-answer-anchored single-token readout:
we pick the final answer-like token in the exposed `logprobs.content` that
matches the visible answer, use that token's `top_logprobs` to form a local
`A/B/C` distribution, and then map it to `yes/no/unknown`.

More explicitly, for `label3` probes the implementation is:

1. run a constrained `A/B/C` query with `logprobs=True` and `top_logprobs`
2. parse the visible surfaced answer from `raw_text`
3. scan `logprobs.content` for answer-like tokens
4. select the answer token by:
   - preferring the **last** answer-like token whose semantic meaning matches
     the surfaced answer
   - otherwise falling back to the **last** answer-like token in the stream
5. build raw local probabilities from that one token position using:
   - the chosen token's own `logprob`
   - plus its `top_logprobs`
6. normalize those raw candidates
7. map:
   - `A -> yes`
   - `B -> no`
   - `C -> unknown`
   - direct `yes/no/unknown` tokens map directly

This is **not**:

- Monte Carlo sampling
- raw model logits
- a full-sequence probability
- a provider-guaranteed "final answer probability" object

Relevant code:

- token selection: [behavioral_probe_runner.py#L422](/root/reveng/src/reveng/experiments/behavioral_probe_runner.py#L422)
- raw probability extraction: [behavioral_probe_runner.py#L447](/root/reveng/src/reveng/experiments/behavioral_probe_runner.py#L447)
- semantic mapping: [behavioral_probe_runner.py#L459](/root/reveng/src/reveng/experiments/behavioral_probe_runner.py#L459)
- stored answer-token logprobs: [behavioral_probe_runner.py#L481](/root/reveng/src/reveng/experiments/behavioral_probe_runner.py#L481)
- readout assembly: [behavioral_probe_runner.py#L657](/root/reveng/src/reveng/experiments/behavioral_probe_runner.py#L657)

## Pathology 1: Wrong Answer Token Extraction

### Symptom

The visible output can be `B` (`no`) while the extracted logprob answer token
is `A` (`yes`), producing a hard contradiction:

- `greedy_answer = no`
- `mc_yes_no_answer = no`
- `logprob_yes_no_answer_t0 = yes`

### Root Cause

The current extraction rule takes the **first answer-like token** in
`logprobs.content`, not the token that corresponds to the final surfaced
answer:

- [behavioral_probe_runner.py#L427](/root/reveng/src/reveng/experiments/behavioral_probe_runner.py#L427)
- [behavioral_probe_runner.py#L488](/root/reveng/src/reveng/experiments/behavioral_probe_runner.py#L488)

That rule is too weak for reasoning-model/provider stacks, because the token
stream can contain answer-like material before the final visible label.

### Strong Example

From `data/behavioral_probes/greedy_repeated_slice/behavioral_probe_raw.json`:

- example: `smoke_010_open_door_elsewhere`
- question: `hit_wall_after_right`
- visible raw output: `B`
- repeated greedy raw outputs: all `B`
- MC raw outputs: all `B`
- extracted `logprob_t0_answer_token_logprobs[0].token`: `' A'`

So the surfaced answer is stable, but the extracted answer token is wrong.

### Interpretation

This is a logprob-stream/readout problem, not a prompt mismatch and not a Monte
Carlo artifact.

### Fix

The extractor now uses a visible-answer-anchored rule:

1. parse the surfaced answer from `raw_text`
2. scan `logprobs.content` from the end
3. select the last answer-like token that matches the surfaced answer
4. only if no such token exists, fall back to the last answer-like token

After this fix, the previously stable contradiction case
`smoke_010_open_door_elsewhere / hit_wall_after_right` now aligns:

- `greedy_answer = no`
- `greedy_modal_answer = no`
- `mc_yes_no_answer = no`
- `logprob_yes_no_answer_t0 = no`

## Pathology 2: Real `T=0` Instability

### Symptom

Repeated `T=0` calls can disagree on the same prompt, even when the prompt and
seed are fixed.

### Strong Example

From `data/behavioral_probes/greedy_repeated_slice/behavioral_probe_rows.csv`:

- example: `smoke_009_wall_right_no_door`
- question: `hit_wall_after_right`
- `greedy_answer = yes`
- `greedy_modal_answer = no`
- `greedy_repeated_agreement_rate_for_row = 0.4`

From the raw artifact:

- first visible output: `A`
- repeated `T=0` outputs: `['B', 'B', 'B', 'B', 'B', 'A', 'A', 'B', 'A']`

### Interpretation

Hosted `T=0` inference is not guaranteed to be end-to-end deterministic.
Backend batching / runtime effects can produce real instability.

This pathology is real, but it does **not** explain the strongest logprob
contradictions by itself.

## Pathology 3: MC Wording Needed Tightening

Earlier plots used wording like `MC yes>no`, which was too strong.

What MC actually does:

- sample full outputs repeatedly
- parse them into `yes/no/unknown/invalid`
- form an empirical answer distribution from counts
- derive a yes/no decision from that empirical distribution

The current cleaner phrasing is:

- `MC sampled yes/no`

This is implemented from:

- [behavioral_probe_runner.py#L889](/root/reveng/src/reveng/experiments/behavioral_probe_runner.py#L889)
- [behavioral_probe_runner.py#L859](/root/reveng/src/reveng/experiments/behavioral_probe_runner.py#L859)
- [behavioral_probe_metrics.py#L46](/root/reveng/src/reveng/experiments/behavioral_probe_metrics.py#L46)

## Pathology 4: Earlier CSV Writer Failure on Mixed Schemas

When `--question-family all` mixed `label3` and `coord_json` rows in a single
run, the CSV writer originally used the first row's keys as the schema. Later
coordinate rows then crashed the write with extra fields.

This is fixed by taking the union of row keys in first-seen order before
writing.

## Extraction Refactor: What Can Be Deterministic vs Heuristic

### Deterministic for current `label3`

For current `label3` probes, the visible answer is constrained to one of:

- `A`
- `B`
- `C`

and is interpreted as:

- `A -> yes`
- `B -> no`
- `C -> unknown`

Because the surfaced answer is supposed to be exactly one label, we can do a
better extraction than "first answer-like token":

1. Parse the visible surfaced answer from `raw_text`.
2. Normalize it into the constrained label or semantic class.
3. Scan `logprobs.content` from the **end** rather than the start.
4. Select the last answer-like token that matches the surfaced answer.

For the current constrained `label3` setup, that matching rule is close to
deterministic.

### Still Heuristic

There are still edge cases where a fallback heuristic is needed:

- the provider returns direct semantic tokens (`yes`, `no`, `unknown`) instead
  of `A/B/C`
- the visible output contains punctuation or extra formatting
- the provider tokenization splits or prefixes the label with whitespace
- no answer-like token in `logprobs.content` cleanly matches the surfaced answer

So the refactor target should be:

- **primary rule**: deterministic visible-answer-anchored extraction
- **fallback rule**: heuristic last answer-like token

### What We Should Avoid

We should not continue to interpret the first answer-like token in
`logprobs.content` as the answer token. That rule is the known failure mode.
