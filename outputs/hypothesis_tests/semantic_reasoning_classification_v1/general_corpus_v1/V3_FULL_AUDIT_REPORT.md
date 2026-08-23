# V3 full-corpus semantic-label audit

## Run status

- File: `annotations_gpt_oss_20b_multilabel_v3_full.csv`
- Inventory coverage: 7,038/7,038 sentence IDs
- Successful annotations: 7,036
- Failed annotations: 2
- Status: `incomplete` because of the two failures
- Judge: Together `openai/gpt-oss-20b`, low reasoning effort
- Approximate runtime: 23 minutes 33 seconds
- Tokens: 16,189,949 prompt; 1,092,530 completion
- Estimated Together charge at $0.05/M input and $0.20/M output: $1.03

The run is corpus-complete in row coverage but not formally complete in valid
labels: two rows contain errors rather than annotations.

## Duplicate consistency and primary-label agreement

Neither metric can be calculated for this full run.

| Requested measure | Result | Reason |
|---|---|---|
| Duplicate consistency | Not measurable | All 7,038 `duplicate_of_annotation_id` fields are empty; no audit item was deliberately submitted twice. |
| Primary-label agreement | Not measurable | Each sentence has only one judgment. `primary_label` is deterministically derived from that judgment's multi-label set, not supplied by a second judge. |

Therefore this report makes no duplicate-agreement or inter-judge-agreement
claim for the 7,038-row run. Measuring either requires a second annotation of
the same sentences. Naturally recurring boilerplate and the separate 40-row
smoke run are not counted as duplicates of this full run.

## Full-corpus label profile

Among 7,036 successful annotations:

| Labels per sentence | Count | Percentage |
|---:|---:|---:|
| 1 | 4,435 | 63.0% |
| 2 | 2,152 | 30.6% |
| 3 | 440 | 6.3% |
| 4 | 9 | 0.1% |

Mean label count is 1.435.

| Label anywhere | Count | Percentage |
|---|---:|---:|
| `new_inference` | 2,795 | 39.7% |
| `state_readout` | 2,598 | 36.9% |
| `verification` | 1,510 | 21.5% |
| `route_planning` | 1,506 | 21.4% |
| `restatement` | 806 | 11.5% |
| `action_commitment` | 394 | 5.6% |
| `procedural_continuation` | 356 | 5.1% |
| `consolidation` | 99 | 1.4% |
| `correction` | 31 | 0.4% |

The most common primary labels are `state_readout` (2,005), `verification`
(1,493), `route_planning` (1,283), and `new_inference` (1,079).

## Statistical and qualitative insights

### 1. `new_inference` remains a broad add-on, although less extreme than the smoke suggested

The full rate is 39.7%, compared with 67.5% in the 40-item v3 smoke and 48.4%
in the earlier v2 pilot. The small smoke sample overstated its corpus rate, but
overlap remains common:

- 893/1,510 verification sentences (59.1%) also have `new_inference`.
- 674/1,506 route-planning sentences (44.8%) also have `new_inference`.
- 392/2,598 state-readout sentences (15.1%) also have `new_inference`.

The boundary is therefore still permissive, especially for verification and
planning outcomes.

### 2. Selected route step versus possible route remains a review boundary

The full run labels “From (7,3) up to (6,3)” as `route_planning`, while phrases
that explicitly say “next action” generally receive `action_commitment`.
Hedged forms such as “we could go” sometimes still receive commitment. These
examples should be included in any later human audit.

### 3. Questions are more often treated as tentative in the full run, but exceptions remain

“So left to (5,3) goal?” changed from local `action_commitment` +
`state_readout` to Together `route_planning` + `new_inference`. “Could there be
a shorter path?” changed from route planning to verification. Across the full
corpus, five hedged or interrogative sentences still received
`action_commitment`, generally because they also contain phrases such as “next
move” or “should go.”

### 4. State readout versus inference still overlaps

The full run labels “Goal G at row2 col3?”, “So (6,5) is open,” and a Row5 open
cell statement as `state_readout` alone. However, 392 state readouts also carry
`new_inference`, including “Goal G at row 7 col 7?”. This boundary remains worth
targeted human review.

### 5. Confidence remains severely concentrated

- High: 7,003/7,036 (99.5%)
- Medium: 26/7,036 (0.37%)
- Low: 7/7,036 (0.10%)
- `unclear`: 9/7,036 (0.13%)

This distribution is implausibly certain for a corpus containing fragments,
questions, and hedged statements. Confidence and `annotation_status` should not
be treated as calibrated uncertainty measures.

### 6. Rationales are audit aids, not stable measurements

Some rationales imply more than the target itself states:

- “But the goal is at (2,6)…” receives the rationale “Records the goal location
  relative to a known wall,” although the target only states its relation to a
  column, not explicitly to a wall.
- “Did we miss something?” was stable as `verification`, but a rationale that
  says it confirms a route length would overstate what the target performs.
- “Agent can open door automatically…” alternates between a newly derived rule
  and a restated rule depending on how preceding context is interpreted.

Rationales are useful for finding boundary errors but should not become an
analysis variable without separate validation.

## Two failed rows

Both failures exhausted four attempts because the model combined
`procedural_continuation` with substantive labels, which the schema forbids:

- `keepdoor_21__step_009__sentence_011`: “Good. Row3: …”
- `keepdoor_33__step_001__sentence_045`: “Ok. From agent at (2,6).”

These mixed sentences expose a segmentation/taxonomy issue: a procedural
discourse marker and substantive state content occur in one segmented unit.
The current exclusive procedural rule rejects the whole response instead of
retaining the substantive label.

## Bottom line

The full run is operationally successful (99.97% valid rows) and useful for
exploratory corpus interpretation. It provides label frequencies and examples,
but no duplicate consistency or primary-label agreement estimate because it
contains only one judgment per sentence. Confidence is highly concentrated,
and the route/commitment and verification/inference boundaries warrant human
review. Any reported experiment should include label-sensitivity analyses or
collapse unstable fine labels into broader human-readable groups.
