# Behavioral epistemic-state pilot

## Question and method

This analysis asks whether three beliefs relevant to the model's currently recommended action form useful behavioral states: confidently correct, unresolved, or confidently wrong. At every reasoning sentence, it checks beliefs about a wall in the chosen direction, whether that action will hit a wall, and whether the model will have the key afterward. A factual answer counts as confident at probability 0.80. A confident error takes priority over unresolved evidence, so a position that contains both remains classified as confidently wrong while both component flags remain in the row-level table.

The analysis uses 7,084 existing reasoning positions from 31 trajectories. It makes no new model calls and does not use change-point detection, semantic labels, or activation probes. Loss means that an action that is currently optimal becomes suboptimal at least once in the next three sentences. Recovery is the reverse. Both outcomes require three later sentences in the same example.

## Counts and result

At the primary threshold, 1,454 positions were confidently correct, 3,698 were unresolved, and 1,932 were confidently wrong. Confidently correct, Unresolved, Confidently wrong met the planned minimum of 20 positions across five trajectories.

The primary comparison is optimality loss after confidently wrong versus confidently correct beliefs. The observed rates were 11.0% and 7.5%, a difference of +3.5 percentage points (95% trajectory-bootstrap interval -3.3 percentage points to +10.1 percentage points). This comparison passes the planned population gate.

The primary interval includes zero, so this pilot does not provide clear evidence that confidently wrong beliefs have a different near-term loss rate. For unresolved versus confidently correct positions, the loss-rate difference was -2.8 percentage points (95% interval -8.3 percentage points to +2.9 percentage points). The strongest descriptive pattern was instead current suboptimality: 43.9% for confidently correct, 12.8% for unresolved, and 27.5% for confidently wrong positions. Recovery rates were 21.1%, 26.4%, and 22.8%, respectively. The high suboptimal-action rate in the confidently correct group identifies belief-utilization failure candidates, but it could also mean these three local beliefs omit other information needed to choose the globally optimal action.

## Why 0.80, and how sensitive is the result?

The 0.80 boundary is a prespecified operational definition of high factual confidence, not a uniquely correct or literature-derived cutoff. It requires at least 80% probability on one factual answer, leaving at most 20% for the other factual answer and `unknown` together. It was fixed before examining these action outcomes. Because any hard boundary can move near-threshold positions between categories, the analysis repeats everything at 0.70 (lenient) and 0.90 (strict), changing no data or outcome definition.

At 0.70, there were 1,923 confidently correct, 2,497 unresolved, and 2,664 confidently wrong positions. At 0.80, there were 1,454 confidently correct, 3,698 unresolved, and 1,932 confidently wrong positions. At 0.90, there were 816 confidently correct, 5,221 unresolved, and 1,047 confidently wrong positions. The qualitative ordering was stable: confidently correct positions had the highest current-suboptimality rate, and confidently wrong positions had the highest subsequent-loss rate, at all three boundaries. The wrong-minus-correct loss difference was +3.0 percentage points, +3.5 percentage points, and +5.3 percentage points at 0.70, 0.80, and 0.90. However, category membership changed substantially, so the exact rates and labels remain threshold-dependent. `figures/threshold_sensitivity.png` shows both the outcome rates and this reclassification directly.

Across all three required beliefs, explicit `unknown` was the top-probability answer in 1,013 of 21,252 valid-or-attempted measurements (4.8%). There were 0 invalid positions. A position is marked invalid when at least one required belief row is missing or has malformed probabilities.

## Limitations

These are descriptive associations, not causal effects. The categories use simulator truth and a chosen confidence cutoff; the threshold table shows how much the result depends on that cutoff. Repeated positions are correlated, so intervals resample whole trajectories. End-of-example positions without three later sentences remain in the state table but are not used for loss or recovery. Explicit `unknown` is only a measurement diagnostic here; the analysis makes no claim about honest or dishonest reporting.

## Files

- `epistemic_state_rows.csv`: auditable sentence-level states, belief probabilities, and outcomes.
- `epistemic_state_summary.csv`: primary-threshold rates and trajectory-bootstrap intervals.
- `state_rate_contrasts.csv`: state-versus-confidently-correct rate differences.
- `threshold_sensitivity.csv`: the same summaries at 0.70, 0.80, and 0.90.
- `measurement_audit.csv`: missingness, explicit-unknown frequency, and state coverage.
- `figures/epistemic_state_action_outcomes.png`: primary-threshold conditional outcome rates, with each panel's denominator stated explicitly.
- `figures/threshold_sensitivity.png`: outcome rates and category membership at the 0.70, 0.80, and 0.90 boundaries.
