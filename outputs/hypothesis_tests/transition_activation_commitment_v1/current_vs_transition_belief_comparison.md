# Current-State Versus Transition-Belief Comparison

This first-pass comparison uses the transition-belief probes already present in the matched-46 run: the three action-consequence probes for the action recommended at each prefix. These probes are action-conditioned but not allocentric. Separately collected allocentric readouts are not included in this model comparison.

| Target | Best learned model | AUROC | Events | Interpretation |
|---|---|---:|---:|---|
| action change | transition beliefs | 0.545 | 682 | not decisive |
| optimal to suboptimal | progress baseline | 0.605 | 161 | not decisive |
| suboptimal to optimal | progress baseline | 0.621 | 182 | not decisive |
| commitment onset | current state beliefs | 0.736 | 45 | not decisive |

Success criterion for the planned allocentric test: transition-belief AUROC must beat both progress and current-state baselines by at least 0.03 with a grouped-bootstrap lower bound above zero.
