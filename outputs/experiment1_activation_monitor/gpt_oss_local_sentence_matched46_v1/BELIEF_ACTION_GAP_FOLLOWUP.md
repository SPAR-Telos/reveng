# Belief-Action Gap Follow-Up

Belief-action conflict is defined as recommending an action while the model's own elicited belief says that action is blocked: either the chosen-direction wall probe answers yes, or the chosen-action hit-wall consequence probe answers yes. This is a simple empirical gap and does not assume a full planner.

## Descriptive Summary

| Group | States | Prefixes | Conflict rate | Mean belief error | Mean belief entropy |
|---|---:|---:|---:|---:|---:|
| all | 46 | 7038 | 0.003 | 0.160 | 0.423 |
| control | 23 | 3373 | 0.000 | 0.141 | 0.424 |
| failure | 23 | 3665 | 0.005 | 0.177 | 0.423 |

## Predictive Summary

| Target | Best learned model | AUROC | Events |
|---|---|---:|---:|
| action change | belief action conflict | 0.534 | 682 |
| optimal to suboptimal | belief action conflict | 0.605 | 161 |
| suboptimal to optimal | belief action conflict | 0.622 | 182 |
| commitment onset | belief error entropy | 0.736 | 45 |
