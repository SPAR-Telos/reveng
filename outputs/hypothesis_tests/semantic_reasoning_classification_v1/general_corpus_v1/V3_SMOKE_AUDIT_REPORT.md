# V3 multi-label smoke audit

## Run and scope

- Output: `annotations_gpt_oss_20b_multilabel_v3_smoke.csv`
- Judge: local `openai/gpt-oss-20b`, low reasoning effort
- Taxonomy: `general_multilabel_v3_human_calibrated`
- Rows: 44 total = 40 unique sentences + 4 hidden duplicates
- Completion: 44/44, with no row errors
- Runtime: approximately 4 minutes 21 seconds

This is a smoke-test audit, not an accuracy estimate. Forty unique items are
enough to expose recurrent failure modes but not to estimate corpus-wide label
rates or human agreement precisely.

## Duplicate consistency

“Exact-label consistency” means that both copies received the same complete
set of semantic labels. “Primary-label consistency” means that the derived
single grouping label was also the same. The primary label is not a separate
judge decision; code derives it from the multi-label set using a fixed
precedence rule.

| Measure | Agreement |
|---|---:|
| Exact semantic-label set | 4/4 (100%) |
| Derived primary label | 4/4 (100%) |
| Annotation status | 4/4 (100%) |
| Four cue fields | 16/16 (100%) |
| Confidence | 4/4 (100%) |
| Rationale text | 4/4 (100%) |

| Duplicated target | Labels on both copies | Primary label |
|---|---|---|
| “That might be shorter.” | `new_inference` | `new_inference` |
| “We need path.” | `new_inference` | `new_inference` |
| “Up from (5,5) blocked.” | `verification`, `state_readout` | `verification` |
| “So left to (5,3) goal?” | `action_commitment`, `state_readout` | `action_commitment` |

This establishes deterministic repeatability at temperature zero for these
four items. It does not establish that their labels are correct.

## Descriptive label profile

| Statistic | Result |
|---|---:|
| Mean labels per unique sentence | 1.55 |
| One-label sentences | 19/40 (47.5%) |
| Two-label sentences | 20/40 (50.0%) |
| Three-label sentences | 1/40 (2.5%) |
| `new_inference` anywhere | 27/40 (67.5%) |
| `new_inference` as primary | 12/40 (30.0%) |
| `verification` anywhere | 12/40 (30.0%) |
| `state_readout` anywhere | 9/40 (22.5%) |
| Low-confidence annotations | 0/40 |
| `unclear` annotations | 0/40 |

The most common overlap was `verification` + `new_inference`: 9 instances,
or 75% of all sentences carrying `verification`. The next most common was the
intended `action_commitment` + `state_readout` overlap: 3 instances.

## Insights and examples

### 1. `new_inference` is probably over-applied

The label appears on 67.5% of unique items and frequently accompanies a more
specific function. Several examples look like state reporting, checking, or
route narration rather than a distinct derived conclusion:

| Target | Output | Concern |
|---|---|---|
| “Row5: … col6 '_' open.” | `state_readout`, `new_inference` | “Open” is a direct decoding of `_`; a second inference label adds little. |
| “So (6,5) is open.” | `state_readout`, `new_inference` | Likely a state readout unless the context contains a genuinely separate consequence. |
| “Goal G at row2 col3?” | `new_inference` | A tentative location readout/check is not clearly a derived conclusion. |
| “That's 8 steps to (5,5).” | `new_inference` | Could be arithmetic bookkeeping or consolidation, depending on preceding context. |
| “Need to open door adjacent to door.” | `new_inference` | May state a task constraint, but the fragment does not show derivation by itself. |

Proposed boundary: apply `new_inference` only when the sentence asserts a
consequence not already explicit in the state readout, verification, plan, or
summary. Do not add it merely because understanding a symbol or coordinate
requires trivial interpretation.

### 2. Confidence is not calibrated to ambiguity

Every annotation was `complete` and high confidence. Yet the sample contains
questions and fragments such as “Goal G at row2 col3?”, “Did we miss
something?”, and “So left to (5,3) goal?”. At least some should plausibly be
medium confidence or `unclear`.

This means the current confidence/status fields should not be treated as
empirical uncertainty measures. A future prompt revision should include a
medium-confidence demonstration and explicitly say that grammatical fragments
or unresolved questions can be `unclear`.

### 3. Rationales sometimes claim more than the target sentence performs

Examples include:

- For “But the goal is at (2,6), which is to the right of column4,” the
  rationale says the goal is unreachable. The target states a spatial relation,
  not unreachability.
- For “Did we miss something?”, the rationale says it confirms route length.
  The target initiates a check but does not itself confirm anything.
- Some neighbor-listing rationales claim that route feasibility was confirmed
  even when the target only enumerates coordinates.

The labeler is occasionally explaining the surrounding argument rather than
the semantic function of the target sentence. Rationales are therefore useful
for auditing but should not be analyzed as faithful generated explanations.

### 4. `verification` + `new_inference` needs a stricter overlap test

This overlap can be legitimate: a sentence can check a route and derive that it
is blocked. Its 9/12 frequency here suggests the judge may add
`new_inference` whenever a verification has an outcome. A stricter rule would
reserve the overlap for a separately stated consequence; “cell blocked” alone
is verification/state readout, while “cell blocked, therefore the lower route
is impossible” performs both functions.

### 5. Questions are being converted into commitments or conclusions

“So left to (5,3) goal?” received `action_commitment` + `state_readout` with
high confidence. The question mark may instead signal a tentative check. The
taxonomy needs an explicit rule: interrogative or hedged movement is not an
`action_commitment` unless preceding context clearly establishes that the route
has already been selected.

## What the smoke run supports

It supports the engineering pipeline: model loading, schema validation,
checkpoint output, canonical category names, and deterministic duplicate
handling all worked. It does not yet support a claim that the label set is
accurate enough for a formal full-corpus interpretation. The next useful gate
is a targeted human audit of approximately 20 v3 items enriched for
`new_inference`, `verification` + `new_inference`, questions, and high-confidence
fragments, followed by a fresh 400-item pilot if the prompt changes.
