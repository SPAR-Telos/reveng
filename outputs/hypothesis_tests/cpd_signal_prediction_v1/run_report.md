# Can Belief Changes Anticipate BEAST Change Points?

## Natural-language summary

This analysis asks whether changes in the model's explicit state beliefs at the current sentence help identify a change point that offline BEAST later places in the next sentence or next three sentences. It then asks a separate question: whether activation dynamics, belief changes, and offline BEAST output—alone and in combination—help predict future recommendation changes, commitment, loss of optimality, or recovery.

Every model feature is taken from the current or an earlier sentence. Models without BEAST can therefore compute a score at the current sentence if the action, activation, and belief readouts are available. BEAST itself is retrospective because it uses the rest of the time series to decide whether the current location is a change point, so models containing BEAST are retrospective comparisons only. Commitment is also a retrospective evaluation target, as explained below.

## Data and validation

- Environment states: 46
- Trajectories: 31
- Sentence positions: 6,992
- Held-out evaluation: five folds, keeping every sentence from a trajectory in the same fold.
- Primary metric: area under the precision-recall curve (AUPRC); positive log-loss improvement means better probability estimates.
- Frozen BEAST outcomes: 160 original change points; 127 repeatedly found in at least 12 of 16 reruns.

## Question 1: do belief changes anticipate offline BEAST locations?

The reference model uses the current sentence number, current action confidence, and the change in the action distribution at the preceding sentence boundary. The candidate adds four plain summaries of belief change: average and largest probability shift, average entropy shift, and the number of belief answers that changed.

| BEAST outcome | Future window | Positive windows | Reference AUPRC | With belief changes | AUPRC change (95% interval) | Log-loss improvement |
|---|---:|---:|---:|---:|---:|---:|
| All frozen points | 1 sentence | 160 | 0.061 | 0.068 | +0.007 (-0.003 to +0.015) | +0.000 |
| All frozen points | 3 sentences | 474 | 0.165 | 0.159 | -0.006 (-0.011 to -0.001) | -0.000 |
| Repeatedly found subset | 1 sentence | 127 | 0.050 | 0.051 | +0.001 (-0.003 to +0.007) | -0.001 |
| Repeatedly found subset | 3 sentences | 375 | 0.127 | 0.124 | -0.003 (-0.008 to -0.001) | -0.001 |

### Result

At the three-sentence horizon, the evidence does not show a clear improvement from belief changes beyond recent action history.

## Question 2: which current signals predict future action events?

This table first compares the current-and-past activation-plus-belief model with the behavioral baseline. It then measures what offline BEAST adds to that same model. The BEAST comparison is diagnostic, not a deployable predictor.

| Event | Window | Activation + belief: AUPRC change | Log-loss improvement | Additional AUPRC change from offline BEAST | Additional log-loss improvement |
|---|---:|---:|---:|---:|---:|
| Commitment onset | 1 sentence | +0.003 (-0.006 to +0.032) | -0.000 | -0.001 (-0.006 to +0.002) | -0.000 |
| Commitment onset | 3 sentences | -0.004 (-0.011 to +0.003) | -0.001 | -0.001 (-0.006 to +0.017) | -0.000 |
| Loss of optimality | 1 sentence | -0.007 (-0.023 to +0.003) | -0.000 | +0.000 (-0.009 to +0.009) | -0.000 |
| Loss of optimality | 3 sentences | +0.015 (-0.018 to +0.050) | -0.000 | +0.000 (-0.002 to +0.004) | +0.000 |
| Recommendation change | 1 sentence | -0.002 (-0.005 to +0.001) | -0.000 | +0.000 (-0.001 to +0.001) | +0.000 |
| Recommendation change | 3 sentences | -0.000 (-0.004 to +0.003) | -0.000 | -0.001 (-0.002 to +0.000) | +0.000 |
| Recovery of optimality | 1 sentence | +0.005 (-0.018 to +0.019) | -0.000 | -0.002 (-0.006 to +0.001) | -0.000 |
| Recovery of optimality | 3 sentences | -0.007 (-0.025 to +0.013) | -0.002 | -0.002 (-0.004 to +0.001) | -0.000 |

### All signal combinations at the three-sentence horizon

Each cell is held-out AUPRC, followed in parentheses by its change from the behavioral baseline. The belief signal here means the four belief-change summaries defined above; it is narrower than the full belief-state model in the earlier practical-monitor analysis.

| Current-sentence inputs | Recommendation change | Commitment onset | Loss of optimality | Recovery |
|---|---:|---:|---:|---:|
| Behavioral baseline | 0.525 (+0.000) | 0.055 (+0.000) | 0.271 (+0.000) | 0.401 (+0.000) |
| Baseline + activation | 0.526 (+0.001) | 0.059 (+0.004) | 0.261 (-0.011) | 0.402 (+0.001) |
| Baseline + belief changes | 0.521 (-0.004) | 0.052 (-0.003) | 0.294 (+0.022) | 0.392 (-0.009) |
| Baseline + offline BEAST | 0.525 (-0.000) | 0.058 (+0.002) | 0.270 (-0.001) | 0.409 (+0.008) |
| Baseline + activation + belief changes | 0.524 (-0.000) | 0.052 (-0.004) | 0.286 (+0.015) | 0.393 (-0.007) |
| Baseline + activation + offline BEAST | 0.525 (+0.001) | 0.061 (+0.005) | 0.265 (-0.007) | 0.399 (-0.002) |
| Baseline + belief changes + offline BEAST | 0.521 (-0.004) | 0.053 (-0.003) | 0.291 (+0.020) | 0.392 (-0.009) |
| All signals, including offline BEAST | 0.523 (-0.001) | 0.051 (-0.004) | 0.286 (+0.015) | 0.392 (-0.009) |

### Result

The preplanned activation-plus-belief comparison does not clearly improve any action event over the behavioral baseline: every AUPRC interval crosses zero, and most log-loss changes are negative or effectively zero. Adding offline BEAST to those current-and-past signals also does not produce a clear incremental improvement. Individual combinations are retained in the table and CSV as exploratory comparisons, not selected as new headline results.

Commitment onset is also a retrospective evaluation label: its location is defined by checking that the recommendation remains stable through the rest of the trace. Predictors use no future features, but the target cannot be confirmed at the moment it occurs.


## Report-ready figures

In these figures, action distribution means the probability distribution over `RIGHT`, `LEFT`, `UP`, and `DOWN`.

Across 6,946 sentence boundaries, the recommended action changed 658 times. Euclidean distance from the previous sentence's distribution identified these changes with AUPRC 0.806; distance from the pre-reasoning distribution had AUPRC 0.074.

![Distribution distance and action changes](figs/distribution_distance_action_changes.png)

Adjacent changes in the six behavioural-probe readouts had only a weak association with adjacent changes in the action distribution (Spearman r = 0.066). Adding these belief-change features changed held-out AUPRC for identifying offline BEAST locations from 0.252 to 0.250.

![Belief changes and distribution outcomes](figs/belief_changes_distribution_outcomes.png)

The first association is partly structural: both distribution distance and the recommended-action label are derived from the same four probabilities. The BEAST outcome is also derived from the full action-distribution series. These are correlational identification results, not independent causal predictions.

## Interpretation rules

- AUPRC change measures whether the added signals rank true future events above non-events more effectively.
- Log-loss improvement measures whether the predicted probabilities become more accurate. A positive value is better.
- We call an addition useful only when AUPRC improves, its trajectory-resampling interval supports the same direction, and log loss also improves.
- Correlated signals may share information. A weak incremental result does not imply that a signal has no association on its own.
- Offline BEAST results cannot be described as real-time prediction.

## Files

- `prediction_summary.csv`: held-out metrics for every target and model.
- `planned_comparisons.csv`: the direct signal-addition questions and uncertainty intervals.
- `out_of_fold_predictions.parquet`: one held-out probability per row, target, and model.
- `analysis_rows.parquet`: aligned features and future labels.
- `figs/distribution_distance_action_changes.png`: how distribution distances identify recommendation changes.
- `figs/belief_changes_distribution_outcomes.png`: belief changes versus distribution changes and offline BEAST locations.
- `../action_distribution_cpd_robustness_v1/MEASUREMENT_FREEZE.md`: frozen BEAST outcome definition.
