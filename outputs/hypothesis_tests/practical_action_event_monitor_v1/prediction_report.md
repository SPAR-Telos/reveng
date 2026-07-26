# Practical Prediction of Action Changes and Optimality Loss

This analysis forecasts events after the current reasoning sentence. It does not classify an event using the activation or belief readout from the event sentence itself.

- Environment states: 46
- Trajectories: 31
- Sentence-backed positions before horizon filtering: 6,992
- Validation: five grouped outer folds by trajectory, with elastic-net tuning in grouped inner folds.
- Primary metric: area under the precision-recall curve (AUPRC).

## Targets

| Target | Horizon | At-risk positions | Positive windows | Rate |
|---|---:|---:|---:|---:|
| Recommendation change | 1 | 6,946 | 658 | 0.095 |
| Optimality loss | 1 | 5,359 | 156 | 0.029 |
| Recovery | 1 | 1,587 | 173 | 0.109 |
| Recommendation change | 3 | 6,854 | 1,416 | 0.207 |
| Optimality loss | 3 | 5,283 | 347 | 0.066 |
| Recovery | 3 | 1,571 | 367 | 0.234 |

An optimality-loss prediction is evaluated only while the current recommendation is planner-optimal. A recovery prediction is evaluated only while it is suboptimal. The three-sentence labels indicate whether an event occurs at any of the next three sentence boundaries.

## Held-Out Results

| Target | Horizon | Best direct model | AUPRC | Baseline AUPRC | AUPRC change | Log-loss improvement |
|---|---:|---|---:|---:|---:|---:|
| Recommendation change | 1 | activation | 0.334 | 0.335 | -0.001 | -0.0009 |
| Recommendation change | 3 | verified beliefs | 0.532 | 0.531 | +0.001 | -0.0001 |
| Optimality loss | 1 | observable beliefs | 0.181 | 0.166 | +0.015 | +0.0033 |
| Optimality loss | 3 | observable beliefs | 0.334 | 0.273 | +0.061 | +0.0112 |
| Recovery | 1 | activation | 0.252 | 0.244 | +0.009 | +0.0003 |
| Recovery | 3 | text semantic | 0.392 | 0.383 | +0.009 | -0.0001 |

A positive AUPRC change is not sufficient by itself. The result is treated as useful only when held-out log loss also improves and the trajectory-bootstrap interval supports the same direction.

## Headline

Observable belief readouts provide the only clear improvement for the primary failure target. They increase three-sentence optimality-loss AUPRC by +0.061 (trajectory-bootstrap 95% interval +0.014 to +0.123) and improve log loss by +0.0112 (+0.0003 to +0.0243).

The effect is not conclusive at the one-sentence horizon. Activation features do not improve optimality-loss prediction beyond the behavioral baseline, and the combined model does not outperform observable beliefs alone. This supports a short-horizon association between explicit belief readouts and upcoming optimality loss, not a general claim that all internal signals improve failure prediction.

## Hierarchical Prediction

| Target | Horizon | Direct combined-observable AUPRC | Hierarchical AUPRC | Direct log loss | Hierarchical log loss |
|---|---:|---:|---:|---:|---:|
| Optimality loss | 1 | 0.180 | 0.200 | 0.1044 | 0.1029 |
| Optimality loss | 3 | 0.330 | 0.333 | 0.1922 | 0.1912 |
| Recovery | 1 | 0.186 | 0.250 | 0.3317 | 0.2992 |
| Recovery | 3 | 0.385 | 0.402 | 0.5036 | 0.5007 |

The hierarchical score multiplies the predicted probability of a recommendation change by the predicted probability that the change is an optimality loss or recovery. It does not consistently outperform direct prediction.

## Alert Utility for Three-Sentence Optimality Loss

| Model | Precision at top 5% alerts | Unique transitions detected | False alerts per 100 sentences | Median warning |
|---|---:|---:|---:|---:|
| baseline | 0.362 | 0.449 | 3.20 | 2.0 sentences |
| observable beliefs | 0.423 | 0.442 | 2.90 | 2.0 sentences |
| combined observable hierarchical | 0.434 | 0.455 | 2.84 | 2.0 sentences |

At a fixed 5% alert budget, observable beliefs increase alert precision and reduce false alerts, but they do not clearly increase the fraction of unique optimality-loss transitions detected relative to the behavioral baseline.

## Large Action-Distribution Changes

The text-semantic model improves next-sentence large-distribution-change AUPRC by +0.008 (+0.002 to +0.013) and log loss by +0.0011 (+0.0000 to +0.0024). This is a small but consistent association with changes in the full action distribution, not evidence that text semantics predicts planner failure.

## Predictor Definitions

- The behavioral baseline uses reasoning progress, current action confidence, and the Jensen-Shannon divergence from the preceding action distribution.
- Observable belief signals use probe probabilities, entropy, answer changes, and conflicts between the recommended action and reported walls or consequences. They do not require simulator truth.
- Verified belief errors compare those answers with the fixed DoorKey state and transition model.
- Activation signals use GPT-OSS-20B layer-15 sentence means plus sparse cross-layer comparisons at layers 8, 15, and 23.
- Text semantics use the REFRAIN `all-MiniLM-L6-v2` maximum similarity to preceding reasoning sentences and related sentence-embedding novelty measures.

## Representation-Dynamics Baseline

The coarse representation baseline averages training-fold-standardized rolling sentence dispersion and sparse cross-layer change. It is not D²H. Exact D²H requires token-level states at every layer and attention-guided token selection, which are unavailable in the stored sentence aggregates.

## Commitment

Retrospective commitment onset is not used as a deployable prediction target because it is defined by checking that the selected action remains unchanged through the rest of the trace. It remains a descriptive boundary.
