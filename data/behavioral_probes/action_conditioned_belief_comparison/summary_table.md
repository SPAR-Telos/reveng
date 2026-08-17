# Action-Conditioned White-Box vs Black-Box Belief Comparison

This summary uses the released clean repeat-10 slice only.

Methods
- White-box released probe
- Black-box repeated greedy modal
- Black-box MC yes/no
- Black-box logprob yes/no (T=0.7)

Interpretation
- `Directional belief accuracy`: on the direction actually taken by the agent, how often does the readout recover the correct local wall belief?
- `Consistency profile`: on the direction actually taken by the agent, does the readout make that action look locally blocked (`inconsistent`), locally free (`potentially_consistent`), or undecided (`unknown_based`)?

Failure slice note: Failure-slice white-box comparison not included: the exact paper failure slice currently has zero non-empty `whitebox_prediction_pre/post` values, so a comparable white-box analysis requires backfilling those predictions first.

## Split: all

### Directional Belief Accuracy

| Subset | Method | n | Accuracy | Fraction unknown |
| --- | --- | --- | --- | --- |
| non_optimal_action_rows | White-box released probe | 0 | n/a | n/a |
| non_optimal_action_rows | Black-box repeated greedy modal | 0 | n/a | n/a |
| non_optimal_action_rows | Black-box MC yes/no | 0 | n/a | n/a |
| non_optimal_action_rows | Black-box logprob yes/no (T=0.7) | 0 | n/a | n/a |
| all_clean_rows | White-box released probe | 30 | 29/30 (96.7%) | 0/30 (0.0%) |
| all_clean_rows | Black-box repeated greedy modal | 30 | 30/30 (100.0%) | 0/30 (0.0%) |
| all_clean_rows | Black-box MC yes/no | 30 | 30/30 (100.0%) | 0/30 (0.0%) |
| all_clean_rows | Black-box logprob yes/no (T=0.7) | 30 | 30/30 (100.0%) | 0/30 (0.0%) |

### Consistency Profile

| Subset | Method | n | Inconsistent | Potentially consistent | Unknown based |
| --- | --- | --- | --- | --- | --- |
| non_optimal_action_rows | White-box released probe | 0 | n/a | n/a | n/a |
| non_optimal_action_rows | Black-box repeated greedy modal | 0 | n/a | n/a | n/a |
| non_optimal_action_rows | Black-box MC yes/no | 0 | n/a | n/a | n/a |
| non_optimal_action_rows | Black-box logprob yes/no (T=0.7) | 0 | n/a | n/a | n/a |
| all_clean_rows | White-box released probe | 30 | 1/30 (3.3%) | 29/30 (96.7%) | 0/30 (0.0%) |
| all_clean_rows | Black-box repeated greedy modal | 30 | 0/30 (0.0%) | 30/30 (100.0%) | 0/30 (0.0%) |
| all_clean_rows | Black-box MC yes/no | 30 | 0/30 (0.0%) | 30/30 (100.0%) | 0/30 (0.0%) |
| all_clean_rows | Black-box logprob yes/no (T=0.7) | 30 | 0/30 (0.0%) | 30/30 (100.0%) | 0/30 (0.0%) |

## Split: pre

### Directional Belief Accuracy

| Subset | Method | n | Accuracy | Fraction unknown |
| --- | --- | --- | --- | --- |
| non_optimal_action_rows | White-box released probe | 0 | n/a | n/a |
| non_optimal_action_rows | Black-box repeated greedy modal | 0 | n/a | n/a |
| non_optimal_action_rows | Black-box MC yes/no | 0 | n/a | n/a |
| non_optimal_action_rows | Black-box logprob yes/no (T=0.7) | 0 | n/a | n/a |
| all_clean_rows | White-box released probe | 15 | 14/15 (93.3%) | 0/15 (0.0%) |
| all_clean_rows | Black-box repeated greedy modal | 15 | 15/15 (100.0%) | 0/15 (0.0%) |
| all_clean_rows | Black-box MC yes/no | 15 | 15/15 (100.0%) | 0/15 (0.0%) |
| all_clean_rows | Black-box logprob yes/no (T=0.7) | 15 | 15/15 (100.0%) | 0/15 (0.0%) |

### Consistency Profile

| Subset | Method | n | Inconsistent | Potentially consistent | Unknown based |
| --- | --- | --- | --- | --- | --- |
| non_optimal_action_rows | White-box released probe | 0 | n/a | n/a | n/a |
| non_optimal_action_rows | Black-box repeated greedy modal | 0 | n/a | n/a | n/a |
| non_optimal_action_rows | Black-box MC yes/no | 0 | n/a | n/a | n/a |
| non_optimal_action_rows | Black-box logprob yes/no (T=0.7) | 0 | n/a | n/a | n/a |
| all_clean_rows | White-box released probe | 15 | 1/15 (6.7%) | 14/15 (93.3%) | 0/15 (0.0%) |
| all_clean_rows | Black-box repeated greedy modal | 15 | 0/15 (0.0%) | 15/15 (100.0%) | 0/15 (0.0%) |
| all_clean_rows | Black-box MC yes/no | 15 | 0/15 (0.0%) | 15/15 (100.0%) | 0/15 (0.0%) |
| all_clean_rows | Black-box logprob yes/no (T=0.7) | 15 | 0/15 (0.0%) | 15/15 (100.0%) | 0/15 (0.0%) |

## Split: post

### Directional Belief Accuracy

| Subset | Method | n | Accuracy | Fraction unknown |
| --- | --- | --- | --- | --- |
| non_optimal_action_rows | White-box released probe | 0 | n/a | n/a |
| non_optimal_action_rows | Black-box repeated greedy modal | 0 | n/a | n/a |
| non_optimal_action_rows | Black-box MC yes/no | 0 | n/a | n/a |
| non_optimal_action_rows | Black-box logprob yes/no (T=0.7) | 0 | n/a | n/a |
| all_clean_rows | White-box released probe | 15 | 15/15 (100.0%) | 0/15 (0.0%) |
| all_clean_rows | Black-box repeated greedy modal | 15 | 15/15 (100.0%) | 0/15 (0.0%) |
| all_clean_rows | Black-box MC yes/no | 15 | 15/15 (100.0%) | 0/15 (0.0%) |
| all_clean_rows | Black-box logprob yes/no (T=0.7) | 15 | 15/15 (100.0%) | 0/15 (0.0%) |

### Consistency Profile

| Subset | Method | n | Inconsistent | Potentially consistent | Unknown based |
| --- | --- | --- | --- | --- | --- |
| non_optimal_action_rows | White-box released probe | 0 | n/a | n/a | n/a |
| non_optimal_action_rows | Black-box repeated greedy modal | 0 | n/a | n/a | n/a |
| non_optimal_action_rows | Black-box MC yes/no | 0 | n/a | n/a | n/a |
| non_optimal_action_rows | Black-box logprob yes/no (T=0.7) | 0 | n/a | n/a | n/a |
| all_clean_rows | White-box released probe | 15 | 0/15 (0.0%) | 15/15 (100.0%) | 0/15 (0.0%) |
| all_clean_rows | Black-box repeated greedy modal | 15 | 0/15 (0.0%) | 15/15 (100.0%) | 0/15 (0.0%) |
| all_clean_rows | Black-box MC yes/no | 15 | 0/15 (0.0%) | 15/15 (100.0%) | 0/15 (0.0%) |
| all_clean_rows | Black-box logprob yes/no (T=0.7) | 15 | 0/15 (0.0%) | 15/15 (100.0%) | 0/15 (0.0%) |
