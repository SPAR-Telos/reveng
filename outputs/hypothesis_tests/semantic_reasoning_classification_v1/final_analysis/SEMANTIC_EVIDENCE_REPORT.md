# Audited Semantic Reasoning Classification

## Scope and reliability

- Fully labeled blinded items: 320 (160 matched pairs).
- Low-confidence labels explicitly adjudicated: 18.
- Primary close matches: 152 pairs.
- Apparent fine-label calibration agreement: 52.5%; Cohen's kappa 0.417.
- Apparent four-way calibration agreement: 77.5%; Cohen's kappa 0.534.
- Hidden-duplicate agreement: fine 100.0%; four-way 100.0%.

The nine-way labels are retained for transparency and examples, but the
four-way grouping is the primary inferential taxonomy. The reliability
sample is small and AI-adjudicated, and the prompt was calibrated after
pilot inspection. These are reproducible judge labels, not an independent
human reliability estimate or human ground truth.

The primary conclusion is unchanged without the 18 low-confidence
overrides: the raw-judge four-way omnibus test gives p=0.241 versus p=0.219 after adjudication, and the semantics-only AUROC change is +0.026 versus +0.029.

## Matched change-point comparison

The omnibus paired permutation test gives total-variation distance 0.072 across the four semantic categories (p=0.2190; 152 pairs).

| Semantic function | Change points | Matched sentences | Paired difference | 95% trajectory-bootstrap interval | McNemar q |
|---|---:|---:|---:|---:|---:|
| route deliberation | 74.3% | 69.1% | +5.3% | [-2.1%, +12.0%] | 0.8993 |
| summary or restatement | 15.1% | 13.2% | +2.0% | [-5.0%, +9.2%] | 0.8993 |
| other | 1.3% | 4.6% | -3.3% | [-6.7%, +0.0%] | 0.8993 |
| state readout | 9.2% | 13.2% | -3.9% | [-9.7%, +1.7%] | 0.8993 |

## Incremental association

Adding four-way semantic labels to progress and sentence length changes trajectory-held-out AUROC by +0.029 (95% trajectory-bootstrap interval [-0.005, +0.066]).
Adding both semantics and activation geometry changes AUROC by +0.051 [-0.001, +0.111].

This is contemporaneous classification of retrospectively selected
change points. It is not real-time prediction and cannot establish that
a semantic function caused the action distribution to change.

## Connections to existing analyses

- `semantic_event_composition.csv` relates sentence functions to exact
  recommendation changes, optimality loss, recovery, adjacent action-
  distribution divergence, activation geometry, and behavioral-belief
  changes, with trajectory-bootstrap intervals.
- `semantic_outcome_omnibus_tests.csv` gives exploratory, within-
  trajectory permutation tests for those outcome profiles.
- `attention_by_semantic_function.csv` tests whether the existing
  immediate-action attention effect is concentrated in a semantic class;
  `attention_semantic_omnibus_test.json` gives its trajectory-preserving
  permutation test.
- `matched_semantic_comparisons.csv` contains both the prespecified
  verification/consolidation/restatement composite and all fine-label
  sensitivity analyses.

## Interpretation for the central claim

The primary semantic result is null-to-modest, not a clean discourse-
function signature. Route deliberation is numerically more common at
change points, but its interval includes zero; the four-way omnibus test
is not significant; and semantics alone neither reliably improves AUROC
nor held-out log loss. None of the exact recommendation-change,
optimality-loss, recovery, or commitment associations survives correction
(smallest q=0.406).

The one corrected exploratory association is behavioral-belief entropy
(within-trajectory permutation q=0.008). Mean entropy is highest during route deliberation (0.493 bits), versus state readout (0.312), summary or restatement (0.362), and other functions (0.333). This is a secondary association and should be replicated.

The immediate-action attention difference is numerically largest for
summary or restatement sentences, but its semantic heterogeneity test is
not reliable (p=0.053).

Together, these results sharpen rather than replace the central claim.
Action-distribution changes are behaviorally meaningful routing/timing
boundaries, but they are not tied to one stable kind of verbal sentence.
The paper should therefore emphasize imperfect conversion of available
state/value information into action and the timing of commitment—not claim
that one named discourse operation causes the action change.

See `representative_examples.csv` and `EXAMPLE_AUDIT.md` before quoting
individual cases.
