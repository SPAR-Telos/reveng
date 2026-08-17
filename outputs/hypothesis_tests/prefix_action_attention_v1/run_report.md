# Immediate-Action Attention Analysis

## Question

Does attention at the immediate action readout change when the model's recommended action changes?

## Method

- Query position: the last prompt token immediately before the action label. The logits at this position score UP, DOWN, LEFT, and RIGHT.
- Comparison: change in attention from sentence t-1 to t where the replayed recommended action changes, minus the same change at a progress-matched sentence from the same environment state where the replayed action is stable.
- Primary layer: full-attention layer 15. Full-attention layer 23 is a prespecified sensitivity analysis.
- Primary measure: attention per token, averaged across all attention heads.
- Uncertainty: 95% bootstrap intervals over trajectories.
- Interpretation: positive values mean attention increased more when the replayed recommended action changed than at its matched sentence. Attention is correlational and does not establish causal information use.

## Coverage

- Extracted prompt positions: 634
- Environment states: 44
- Trajectories: 30
- Reproduced action argmax: 597/634
- Previously detected pairs with a replayed recommendation change: 102/160
- Pairs with a replayed recommendation change and a stable matched control: 84
- Primary pairs after requiring eager-attention argmax agreement at all four positions: 73
- Main reported pairs after also requiring a high-quality progress match: 67

## Primary estimates

| Prompt region | Matched pairs | Additional attention change (percentage points per token) | 95% bootstrap interval |
|---|---:|---:|---:|
| Grid | 67 | -0.0002 | [-0.0006, 0.0001] |
| Key possession statement | 67 | 0.0004 | [-0.0016, 0.0025] |
| Earlier reasoning sentences | 67 | -0.0072 | [-0.0187, 0.0015] |
| Newest reasoning sentence | 67 | 0.0838 | [0.0442, 0.1314] |
| Action question | 67 | 0.0121 | [-0.0104, 0.0326] |
| Answer-format prompt | 67 | 0.0151 | [-0.0231, 0.0529] |
| Agent and chosen destination cells | 67 | -0.0019 | [-0.0102, 0.0063] |
| Key, door, and goal cells | 67 | -0.0014 | [-0.0031, 0.0002] |

## Limitations

- Candidate positions come from an offline Bayesian change-point detector applied to an earlier action-probability series. The primary subset requires a recommendation change under the current replay and a stable matched control.
- The earlier probability vectors are not exactly reproducible in the current software environment: 597 of 634 action argmaxes agree. Primary event labels therefore use the current replay rather than the earlier labels.
- Explicit attention requires GPT-OSS's eager attention implementation. The primary subset excludes a pair if eager attention changes the normal-kernel action argmax at any compared position.
- The boundary is retrospective: its detection uses the full probability series, although each attention measurement uses only the reasoning revealed up to that sentence.
- Mean attention per token controls for region length but not for all prompt composition effects.
- Attention weight is not proof that the attended information affected the recommended action.
