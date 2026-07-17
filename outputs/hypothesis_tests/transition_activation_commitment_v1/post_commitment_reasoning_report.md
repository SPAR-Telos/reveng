# Post-Commitment Reasoning

States with a commitment boundary: 46.
Commitment is retrospective: the first prefix where the recommended action equals the full-trace action and remains stable.
Post-commitment reasoning is action-epiphenomenal when the action distribution changes little after commitment, but this does not imply that the text is useless for belief calibration.

- Mean post-commitment tail: 39.4 sentences (0.287 of the trace).
- Mean max post-commitment action-distribution drift: 0.218 total variation.
- Mean post-commitment current-state entropy: 0.335 bits.
- Mean post-commitment transition-belief entropy: 0.575 bits.

## Predicting Post-Commitment Length

| Model | MAE | RMSE | Cross-validated R2 |
|---|---:|---:|---:|
| action confidence only | 0.228 | 0.282 | -0.077 |
| current state uncertainty | 0.223 | 0.278 | -0.050 |
| transition belief uncertainty | 0.216 | 0.279 | -0.056 |
| combined | 0.204 | 0.254 | 0.123 |
