# Reader-Facing Figures and Tables

Use these outputs for the pilot write-up. They are selected because each answers a specific question with field-standard terms.

## Recommended Tables

- `table1_run_coverage_event_counts.md`: what was measured and how many events were observed.
- `table2_exploratory_regression_results.md`: exploratory regression coefficients for action-change outcomes.

## Recommended Figures

- `figs/figure1_action_confidence_over_reasoning_progress.png`: whether recommended actions are high-confidence across reasoning.
- `figs/figure2_belief_errors_around_action_optimality_changes.png`: whether belief errors change near optimality changes.
- `figs/figure3_state_belief_uncertainty_over_reasoning_progress.png`: how state-belief uncertainty changes as more reasoning is revealed.

## Diagnostics Not Recommended for Main Text

- `action_entropy_events.png`: action entropy is near zero throughout, so this is a diagnostic rather than a main result.
- the previous activation-metric diagnostic figure: the current activation metrics are hard to interpret and should remain exploratory.
- `sentence_vs_packed_boundaries.png`: useful as a methods appendix figure, but Table 1 is clearer for the main result.
- action-transition heatmaps: useful diagnostics, but too dependent on this specific grid path for a reader-facing result.
