# Activation-Oracle Action Comparison Summary

This table aggregates only the action-oriented activation-oracle prompts.

Setup
- Base model: `Qwen/Qwen3-8B`
- Activation-oracle adapter: `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B`
- `False`: the matched non-failure slice used for the clean revealed-CoT analysis in the paper.
- `True`: the failure slice used for the revealed-CoT analysis in the paper.

- `Ask next action, compare with model's chosen action`: the oracle is asked what action the model would take, and that answer is checked against the model's actual emitted action.
- `Ask next action, compare with optimal action set`: the same next-action readout is instead checked against the DoorKey-aware optimal action set for that state.
- `Ask optimal action, compare with optimal action set`: the oracle is asked what action is shortest-path optimal, and that answer is checked against the optimal action set.
- `State × reveal pairs`: the number of distinct `(state, revealed reasoning fraction)` cases contributing to that row.

| Failure Case | Revealed reasoning fraction | State × reveal pairs | Ask next action, compare with model's chosen action | Ask next action, compare with optimal action set | Ask optimal action, compare with optimal action set |
| --- | --- | --- | --- | --- | --- |
| False | 0% | 15 | 9/15 (60.0%) | 12/15 (80.0%) | 12/15 (80.0%) |
| False | 25% | 15 | 7/15 (46.7%) | 12/15 (80.0%) | 12/15 (80.0%) |
| False | 50% | 15 | 10/15 (66.7%) | 14/15 (93.3%) | 13/15 (86.7%) |
| False | 75% | 15 | 11/15 (73.3%) | 15/15 (100.0%) | 15/15 (100.0%) |
| False | 100% | 15 | 15/15 (100.0%) | 15/15 (100.0%) | 15/15 (100.0%) |
| False | Overall | 75 | 52/75 (69.3%) | 68/75 (90.7%) | 67/75 (89.3%) |
| True | 0% | 13 | 1/13 (7.7%) | 3/13 (23.1%) | 3/13 (23.1%) |
| True | 25% | 13 | 9/13 (69.2%) | 8/13 (61.5%) | 8/13 (61.5%) |
| True | 50% | 13 | 9/13 (69.2%) | 7/13 (53.8%) | 6/13 (46.2%) |
| True | 75% | 13 | 10/13 (76.9%) | 9/13 (69.2%) | 9/13 (69.2%) |
| True | 100% | 13 | 13/13 (100.0%) | 6/13 (46.2%) | 5/13 (38.5%) |
| True | Overall | 65 | 42/65 (64.6%) | 33/65 (50.8%) | 31/65 (47.7%) |
