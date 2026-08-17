# Sentence Label Table Guide

## Recommended table

`sentence_label_table.csv` is the compact, semantics-only product. Each row is one
reasoning sentence with its preceding context, semantic function, confidence,
rationale, and four observable semantic cues. It intentionally contains no
change-point role, matched-pair key, action outcome, BEAST score, activation feature,
attention feature, or behavioral-belief measurement.

Use this table to inspect or reuse the semantic annotations without depending on the
change-point experiment. Its readable `sentence_id` is composed from the environment,
environment step, and one-based sentence number. The opaque annotation hash is retained
only in the full experimental ledger for compatibility with the original annotation run.

## All sentences in the matched cohort

`all_reasoning_sentence_inventory.csv` contains all 7,038 reasoning sentences in the
46-state matched cohort. The 320 classified sentences have populated semantic fields;
the remaining 6,718 rows have `semantic_label_available = false` and blank label fields.
This inventory makes the sampling denominator explicit but does not claim that unreviewed
sentences have been classified.

## Full experimental ledger

`sentence_label_cpd_analysis_table.csv` preserves the original full joined table used
to reproduce the matched CPD, action-event, activation, attention, and behavioral-
belief analyses. Use it only when those experimental fields are needed.

The classifications were produced from `context_before` and `target_sentence` while
the judge was blinded to CPD status and all downstream measurements. However, the 320
sentences were originally sampled as 160 CPD/matched-control pairs; the compact table
is therefore a reusable annotation table, not a population-representative corpus of
all reasoning sentences.

## Compact columns

| Column | Meaning |
|---|---|
| `sentence_id` | Readable compound identifier, such as `keepdoor_33__step_002__sentence_038`. |
| `environment_id` | DoorKey environment trajectory identifier, such as `keepdoor_33`. |
| `environment_step` | Environment step represented by this reasoning trace. |
| `sentence_number` | One-based sentence number within the trace. |
| `context_before` | Reasoning text preceding the target sentence. |
| `target_sentence` | Sentence whose discourse function was classified. |
| `target_sentence_characters` | Character length of the target sentence. |
| `primary_label` | Fine-grained semantic-function label used for examples and sensitivity analyses. |
| `broad_label` | Calibrated four-way label used for primary inference. |
| `confidence` | Judge/adjudicator confidence: high, medium, or low. |
| `rationale` | Short explanation for the final label. |
| `explicitly_revises_prior_reasoning` | Whether the sentence rejects or changes earlier reasoning. |
| `evaluates_prior_route_or_claim` | Whether it checks an earlier route or claim. |
| `repeats_prior_content` | Whether substantive content already appears in the context. |
| `introduces_new_information_or_plan` | Whether it adds a new fact, inference, route, or decision. |
| `adjudication_applied` | Whether a low-confidence model judgment was explicitly reviewed and replaced. |

These are calibrated AI-assisted labels, not independent human ground truth. Prefer
`broad_label` for aggregate analysis and inspect the context and rationale before
quoting a fine-label example.
