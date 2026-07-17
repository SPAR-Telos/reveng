# Information-Use Test

This analysis uses existing GPT-OSS-20B sentence activations and existing matched-46 action-event labels. It does not collect attention tensors.

- Sentence positions: 7,038
- Environment states: 46
- Trajectories: 31
- Validation groups: 23
- Primary activation representation: layer 15 sentence mean.
- Predictive models use grouped cross-validation by matched pair or trajectory.
- Dependence score is linear CKA between PCA-reduced activations and the binary event label; it is used as a low-cost HSIC-style proxy, not as a direct mutual-information estimate.
- The event label is at the same sentence position as the activation. This is an event-classification test, not a leading-indicator test.
- The permutation null shuffles labels within validation groups to break feature-label alignment while preserving each group's event count.

## Difference From the MI-Peaks Paper

The MI-peaks paper measures token-level HSIC/MI between each generated token representation and the gold answer representation, then searches for sparse peaks during reasoning. This run instead measures whether sentence-level GPT-OSS-20B activations are statistically dependent on action-event labels and whether they improve held-out event prediction. Therefore, the numbers here should be read as activation-event dependence scores, not as reproduced MI-peaks values.

## Headline Predictive Improvement

| Target | Best added signal | AUROC change | Log-loss improvement | Interpretation |
|---|---|---:|---:|---|
| action change | baseline plus beliefs and primary activation | 0.030 | 0.0085 | positive held-out improvement |
| optimal to suboptimal | baseline plus beliefs | -0.011 | -0.0009 | no held-out improvement |
| suboptimal to optimal | baseline plus beliefs | -0.011 | -0.0008 | no held-out improvement |
| commitment onset | baseline plus beliefs | -0.006 | 0.0000 | small log-loss improvement only |

## Strongest Dependence Scores

| Target | Layer | Representation | Score | Null mean | Permutation p |
|---|---:|---|---:|---:|---:|
| action change | 15 | sentence_mean | 0.00437 | 0.00062 | 0.005 |
| optimal to suboptimal | 15 | sentence_mean | 0.00241 | 0.00058 | 0.005 |
| suboptimal to optimal | 23 | sentence_final | 0.00257 | 0.00060 | 0.005 |
| commitment onset | 15 | sentence_final | 0.00180 | 0.00062 | 0.005 |

## Interpretation Rules

- High activation-event dependence with no predictive improvement means the activation is associated with events but does not clearly improve action-event prediction.
- Low dependence with no predictive improvement means there is no current evidence for this information-use claim.
- Predictive improvement from belief features is stronger evidence for a belief-action link than raw activation dependence alone.
- Attention extraction should remain deferred unless a follow-up targets the positive recommendation-change signal.
