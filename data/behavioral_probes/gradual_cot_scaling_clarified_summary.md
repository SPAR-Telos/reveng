# Clarified Gradual-CoT Scaling Summary

Columns
- `adjacent_wall_belief_accuracy`: accuracy of the black-box wall-state probe against the ground-truth adjacent-wall label.
- `probed_next_action_in_optimal_action_set_rate`: the revealed-CoT next-action query asks the model what action it would take; this column reports whether that probed next action is in the state's optimal action set.
- `chosen_direction_wall_belief_contradiction_rate`: among rows where the wall question targets the same direction as the model's chosen action and the wall answer is a valid `yes/no`, the rate at which the probe says that chosen direction is blocked (`yes`).
- `chosen_direction_wall_belief_correct_but_action_suboptimal_rate`: among rows where the wall question targets the model's chosen direction and the wall answer is both valid and correct, the rate at which the model's chosen action is still suboptimal.

Not represented in this table
- `probed optimal action, evaluated against optimal action`
- `probed optimal action, evaluated against model action`
- `probed next/model action, evaluated against an external model-action label`

| Slice | Reveal % | Wall belief accuracy | Probed next action in optimal set | Probe says chosen direction is blocked | Probe is correct about chosen direction but action is suboptimal |
|---|---:|---:|---:|---:|---:|
| non_failure_controls | 0 | 98.8% | 88.5% | 3.3% | 11.5% |
| non_failure_controls | 25 | 97.5% | 96.7% | 0.0% | 1.7% |
| non_failure_controls | 50 | 97.5% | 98.4% | 1.6% | 1.7% |
| non_failure_controls | 75 | 98.8% | 96.7% | 4.9% | 1.7% |
| non_failure_controls | 100 | 99.6% | 100.0% | 1.6% | 0.0% |
| balanced_failure_modes | 0 | 93.8% | 87.5% | 6.2% | 12.5% |
| balanced_failure_modes | 25 | 98.4% | 93.8% | 0.0% | 6.2% |
| balanced_failure_modes | 50 | 100.0% | 81.2% | 6.2% | 18.8% |
| balanced_failure_modes | 75 | 98.4% | 75.0% | 12.5% | 25.0% |
| balanced_failure_modes | 100 | 100.0% | 75.0% | 12.5% | 25.0% |
| short_loop | 0 | 99.5% | 63.3% | 12.2% | 35.4% |
| short_loop | 25 | 98.5% | 89.8% | 6.1% | 10.2% |
| short_loop | 50 | 97.4% | 98.0% | 2.0% | 2.1% |
| short_loop | 75 | 97.4% | 95.9% | 4.1% | 4.3% |
| short_loop | 100 | 100.0% | 100.0% | 0.0% | 0.0% |
