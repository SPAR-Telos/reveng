# Where Action Probability Moves During Reasoning

## Question

Do sentence-to-sentence changes move probability toward or away from optimal actions, and do failure states differ from matched control states? We also ask whether retrospective BEAST change points identify more directional movement than ordinary positions at the same stage of reasoning.

## Method

The analysis uses 46 states in 23 matched failure–control pairs. At each sentence, **optimal-action probability** is the sum of the readout probabilities assigned to all actions that are optimal in the current environment state. The primary transition measure is its signed sentence-to-sentence change: positive values move probability toward optimal actions and negative values move it away. No threshold is used for the primary continuous comparison.

For the change-point comparison, each valid BEAST point is paired with a non-change-point position from the same state at similar reasoning progress. Windowed change is the mean optimal-action probability in the 3 sentences after the position minus the mean in the 3 sentences before it. BEAST remains retrospective.

## Results

Failure and control states began with similar optimal-action probability: 0.454 versus 0.410; the paired difference was +4.4 percentage points (95% interval -11.5 to +21.1). Across the full reasoning trace, failure states assigned less probability to optimal actions: 0.656 versus 0.797; difference -14.1 percentage points (95% interval -26.2 to -3.8). At the final readout, the corresponding probabilities were 0.652 and 1.000; difference -34.8 percentage points (95% interval -56.5 to -17.4).

From the initial to the final readout, optimal-action probability increased by 19.9 percentage points in failure states and 59.0 in controls; paired difference -39.2 percentage points (95% interval -67.9 to -13.5). A sentence reduced optimal-action probability in 47.5% of failure-state transitions and 45.6% of control-state transitions; difference +2.0 percentage points (95% interval -1.4 to +5.1). Thus, the groups differ clearly in net accumulation, not in the frequency of individual decreases. Threshold sensitivity gives the same qualitative result.

The change-point comparison retained 160 BEAST points with complete windows, drawn from 30 trajectories. Their absolute windowed change was 26.4 percentage points, compared with 6.2 at progress-matched positions; difference +20.2 percentage points (95% interval +14.4 to +26.0). Across all states, signed change was +7.0 percentage points at BEAST points and -0.0 at matched positions; difference +7.1 percentage points (95% interval +1.0 to +13.1).

At BEAST points, optimal-action probability changed by +12.9 percentage points in controls and +1.5 in failure states. Across 21 complete state pairs, the control-minus-failure contrast was +14.6 percentage points (95% interval -0.6 to +29.1); its interval includes zero. For the points recovered in at least 12 of 16 robustness reruns, the contrast was +20.9 percentage points (95% interval +1.2 to +40.4). This subset result is exploratory.

## Interpretation

Failure and control states start similarly, but controls accumulate substantially more probability on optimal actions. Failure states do not show a clearly higher frequency of sentence-level decreases, so the result should not be described simply as failures taking more wrong-way steps. The analysis does not establish that a particular sentence causes a later action.

BEAST points identify substantially larger probability movements than matched positions. Their average direction is toward optimal actions in control states but close to zero in failure states. The direct state-group contrast is uncertain for all detected points and positive for the repeatedly detected subset. The supported conclusion is therefore that CPD identifies large revisions; differential consolidation on optimal actions is promising but requires replication. BEAST is retrospective, and these associations are not causal.

## Figures

- `figs/action_probability_flow_by_progress.png`
- `figs/change_point_probability_flow.png`

## Output tables

- `probability_flow_rows.csv`: one row per reasoning position.
- `state_probability_flow_summary.csv`: one row per state.
- `matched_failure_control_summary.csv`: paired state comparisons.
- `threshold_sensitivity.csv`: minimum-change sensitivity analysis.
- `progress_summary.csv`: optimal-action probability over normalized reasoning progress.
- `change_point_control_pairs.csv`: BEAST points and their progress-matched positions.
- `change_point_flow_summary.csv`: change-point comparisons with trajectory-bootstrap intervals.
- `change_point_state_group_comparison.csv`: matched failure–control contrasts at change points.
