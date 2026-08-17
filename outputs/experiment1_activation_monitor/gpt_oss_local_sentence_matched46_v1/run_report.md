# Experiment 1: Activation Monitor of Action Events

The matched analysis contains 46 DoorKey states from 31 trajectories, 7,084 sentence prefixes, and 42,228 activation rows.
Activations are GPT-OSS-20B layer 15 sentence means for the prespecified primary analysis. AUROC measures whether each activation metric distinguishes event positions from other sentence positions. Intervals bootstrap complete trajectories and are descriptive, not causal.

A retrospective commitment boundary was identifiable for all 46 states. It is the first prefix whose recommended action equals the full-trace recommendation and remains unchanged thereafter.

| Event | Positions | Change-magnitude AUROC | 95% interval |
|---|---:|---:|---:|
| Recommendation change | 666 | 0.483 | [0.455, 0.506] |
| Commitment onset | 45 | 0.493 | [0.400, 0.583] |
| Optimal to suboptimal | 160 | 0.454 | [0.405, 0.496] |
| Suboptimal to optimal | 173 | 0.472 | [0.423, 0.506] |

The complete metric table is `activation_event_monitor_summary.csv`. The figure is `figs/activation_event_monitor_auc.png`.

The empirical optimal-trace anchor from the original proposal is not included yet; these results use only adjacent change magnitude, distance from the preceding sentence, and similarity to the mean of preceding reasoning sentences.
