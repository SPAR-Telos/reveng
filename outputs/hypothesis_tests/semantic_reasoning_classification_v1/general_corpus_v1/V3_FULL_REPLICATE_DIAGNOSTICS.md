# V3 full-run replicate diagnostics

## Comparison

This report compares two independent Together AI annotations of the complete
7,038-sentence inventory using the same model, v3 taxonomy, few-shot examples,
prompt hash, seed, temperature 0, and low reasoning effort.

- Original: `annotations_gpt_oss_20b_multilabel_v3_full.csv`
- Replicate: `annotations_gpt_oss_20b_multilabel_v3_full_replicate.csv`
- IDs present in both: 7,038/7,038
- Valid judgments in both: 7,036
- Original failures: 2
- Replicate failures: 1
- Unique IDs excluded from agreement: 2; the replicate failure is one of the
  original failures

Label-set comparison parses `semantic_labels` as sets, so JSON ordering cannot
create a disagreement.

## Requested measures

| Measure | Agreement |
|---|---:|
| Exact multi-label set | 5,271/7,036 = **74.9%** |
| Derived primary label | 6,012/7,036 = **85.4%** |

Trajectory-clustered bootstrap intervals, resampling the 31 source
environments/trajectories, are 73.8%–75.9% for exact-set agreement and
84.4%–86.5% for primary-label agreement. These intervals describe variation
across the present trajectory collection; the observed corpus rates above are
exact for the 7,036 valid pairs.

Primary-label Cohen's kappa is 0.820. Mean per-sentence label-set Jaccard
similarity is 0.854, and micro label F1 between runs is 0.877, treating neither
run as ground truth.

There are 1,765 exact-set disagreements but only 1,024 primary-label
disagreements. Thus the derived primary label hides 741 disagreements in
secondary labels.

## Per-label repeatability

Positive-label Dice agreement measures how often the two runs select the same
label when at least one run selects it. Kappa additionally accounts for
agreement expected from each label's prevalence.

| Label | Original positive | Replicate positive | Positive Dice | Kappa |
|---|---:|---:|---:|---:|
| `action_commitment` | 394 | 392 | 0.934 | 0.930 |
| `state_readout` | 2,598 | 2,626 | 0.923 | 0.877 |
| `route_planning` | 1,506 | 1,533 | 0.918 | 0.895 |
| `procedural_continuation` | 356 | 354 | 0.915 | 0.911 |
| `new_inference` | 2,795 | 2,825 | 0.872 | 0.787 |
| `verification` | 1,510 | 1,521 | 0.823 | 0.774 |
| `restatement` | 806 | 814 | 0.777 | 0.747 |
| `correction` | 31 | 26 | 0.632 | 0.630 |
| `consolidation` | 99 | 109 | 0.558 | 0.551 |

The overall label counts are very similar between runs, so the disagreement is
mainly sentence-level boundary instability rather than systematic frequency
drift. `correction` and `consolidation` are both rare and substantially less
repeatable.

## Primary-label stability by original category

| Original primary label | Rows | Exact set | Same primary |
|---|---:|---:|---:|
| `procedural_continuation` | 356 | 91.3% | 91.3% |
| `action_commitment` | 386 | 86.0% | 92.2% |
| `state_readout` | 2,005 | 84.1% | 91.5% |
| `route_planning` | 1,283 | 74.4% | 88.4% |
| `restatement` | 391 | 73.7% | 73.7% |
| `new_inference` | 1,079 | 72.5% | 76.6% |
| `verification` | 1,493 | 59.6% | 82.2% |
| `correction` | 31 | 32.3% | 58.1% |
| `consolidation` | 12 | 25.0% | 25.0% |

Verification has reasonably stable primary assignment but unstable secondary
labels. Correction and consolidation are too rare and unstable for confident
fine-grained comparisons without human review or broader grouping.

## Most frequent primary-label changes

The largest reciprocal boundaries are:

| Original → replicate | Count |
|---|---:|
| `new_inference` → `verification` | 107 |
| `verification` → `new_inference` | 100 |
| `verification` → `state_readout` | 69 |
| `state_readout` → `verification` | 65 |
| `route_planning` → `verification` | 64 |
| `verification` → `route_planning` | 51 |
| `new_inference` → `route_planning` | 51 |
| `restatement` → `state_readout` | 48 |
| `new_inference` → `state_readout` | 47 |

The near-symmetric directions show boundary ambiguity rather than one run
consistently preferring a category.

## Representative generated rationales

### Inference versus verification

Target: “That path avoids door.”

- Original `new_inference`: “Derives that the described route does not pass
  through the door.”
- Replicate `verification`, `new_inference`: “States a property of a previously
  described route, verifying it.”

The proposition is stable; whether relating it to a prior route counts as a
separate verification function is not.

### Planning versus commitment

Target: “Let's pick down, down, down, right?”

- Original `action_commitment`, `route_planning`: “Proposes and selects a
  specific movement sequence.”
- Replicate `route_planning`: “Proposes a concrete sequence without
  committing.”

The interrogative wording leaves selection versus proposal unresolved.

### Restatement versus state readout

Target: “The door is at (4,5).”

- Original `restatement`: “Repeats a previously stated location of the door.”
- Replicate `state_readout`, `restatement`: “Repeats a previously stated
  location and records it as environment state.”

This illustrates a secondary-label disagreement that the same primary label
would not necessarily reveal in other examples.

### Consolidation versus inference

Target: “So no passage.”

- Original `consolidation`: “Summarizes earlier wall checks into a single
  conclusion.”
- Replicate `new_inference`: “Concludes that a passage does not exist based on
  earlier wall checks.”

The distinction depends on whether the conclusion is treated as combining
earlier results or adding a new consequence.

### Same primary, different secondary function

Target: “Goal G at (5,7).”

- Original `state_readout`, `new_inference`
- Replicate `state_readout`

Both runs agree on the main function, but disagree on whether reading the goal
coordinate is also an inference. This is the recurring source of the gap
between 74.9% exact-set and 85.4% primary-label agreement.

## Other output-field consistency

| Field | Agreement |
|---|---:|
| Annotation status | 99.87% |
| `explicitly_revises_prior_reasoning` | 99.40% |
| `evaluates_prior_route_or_claim` | 90.65% |
| `repeats_prior_content` | 90.09% |
| `introduces_new_information_or_plan` | 94.14% |
| Confidence | 99.50% |
| Exact rationale wording | 27.60% |

Exact rationale wording is not an agreement target; the low value shows that
rationales are generated explanations rather than stable categorical data.
The two weakest cue fields correspond directly to the unstable verification
and restatement boundaries.

## Interpretation

The newest v3 pipeline now has defensible corpus-wide repeatability numbers:
74.9% for the complete multi-label set and 85.4% for its derived primary label.
These measure same-prompt production reproducibility, not correctness or human
agreement. State readout, route planning, procedural continuation, and action
commitment are comparatively stable; verification, new inference,
restatement, correction, and consolidation should be grouped, sensitivity
checked, or manually reviewed when they support substantive experimental
claims.
