# Reader-Facing Evidence

## Recommended figures

1. `figs/belief_readouts_forecast_optimality_loss.png`
   - Main prospective result: observable belief readouts provide modest warning of
     optimality loss within the next three reasoning sentences.
2. `figs/action_distribution_changes_near_events.png`
   - Behavioral structure: abrupt changes in action probabilities cluster around
     changes in the recommended action.
3. `figs/attention_changes_when_recommended_action_changes.png`
   - Mechanistic clue: the immediate action readout increases attention to the newest
     reasoning sentence when its recommendation changes.
4. `figs/activation_monitor_baseline_comparison.png`
   - Negative comparison: supervised activation monitors do not outperform behavioral
     action and belief readouts.
5. `figs/semantic_labels_across_analyses.png`
   - Interpretive synthesis: semantic labels constrain simple discourse-function
     stories, identify a high-belief-uncertainty regime, and prioritize causal
     intervention candidates.

## Semantic synthesis

Use `SEMANTIC_CROSS_ANALYSIS_SYNTHESIS.md` for the claims that can be carried into the
paper and `semantic_cross_analysis_decision_table.csv` for their numerical provenance.
`SEMANTIC_PAPER_INSERT.tex` contains a copy-ready results paragraph, figure block, and
compact table.
The main semantic conclusion is not that one reasoning function causes action changes.
It is that action-routing boundaries are semantically heterogeneous, while route
deliberation marks elevated belief uncertainty.

## Main paper context

The existing paper results remain necessary context: state facts are decodable before
reasoning, inferred action values favor optimal actions more often than realized
behavior, reasoning-trace patching changes final actions, and correct wall reports can
coexist with wall-hit actions. The new figures do not replace those results.

## Appendix or diagnostic only

- Generic activation-distance curves: associations are difficult to interpret and do
  not prospectively predict optimality loss.
- State-belief entropy over reasoning progress: failure and control curves are similar
  in this pilot.
- Event-aligned belief-error curves: useful for inspection, but uncertainty is omitted
  and the main averages are mostly flat.
- Commitment timing and post-commitment length: descriptive and based on a
  retrospective boundary.
- Final-action attention to event windows: replaced by immediate action-readout
  attention with progress-matched controls.
