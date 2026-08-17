# Position-General White-Box Probe Feasibility

Question: can one probe be trained across reasoning positions and then applied at every reasoning position?

Pilot protocol: layer 15 linear logistic probes; all positions used for training; entire states held out in five grouped folds; state and class weighted loss; three seeds; 24 independent states and 719 positions.

| Belief | Best representation | Mean balanced accuracy | Seed standard deviation | Assessment |
|---|---|---:|---:|---|
| has_key | sentence_mean | 0.695 | 0.007 | promising |
| wall_down | sentence_mean | 0.546 | 0.024 | not demonstrated |
| wall_left | sentence_mean | 0.644 | 0.004 | weak signal |
| wall_right | sentence_mean | 0.607 | 0.029 | weak signal |
| wall_up | last_token | 0.493 | 0.017 | not demonstrated |

`door_open` was not trainable because every state in both the 24-state pilot and expanded 185-state cohort has the same closed-door label.

## Interpretation

A single position-general probe is feasible in principle, but it must be trained on activations sampled across reasoning positions. It cannot be assumed that a probe trained only before or after reasoning transfers to arbitrary positions. The pilot provides promising evidence for key possession and one wall direction, but other wall directions remain weak or unsupported.

Activation Oracle is analogous in being trained to answer questions from activations across varied contexts and positions. Its flexibility comes from broad position-distributed training, not from a guarantee that one trained readout is calibrated at every token. The available Activation Oracle LoRA targets Qwen3-8B, so it cannot validate GPT-OSS-20B representations directly.

## Recommended Full Run

1. Collect layer-15 boundary activations for the expanded 185-state cohort.
2. Balance or augment each belief label, especially wall directions, and add open-door states.
3. Split by trajectory or canonical grid state before training.
4. Train one position-general linear probe and one MLP probe per belief.
5. Report held-out-state performance overall and by reasoning progress.
6. Use a probe only where held-out balanced accuracy remains stable across progress buckets.

The expanded activation footprint is estimated at under 1 GB using the current compact format.
