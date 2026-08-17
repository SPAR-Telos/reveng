# Intermediate White-Box Probe Validation

The released white-box probes were applied exploratorily to three-token windows ending at each corrected reasoning boundary. The validation gate failed, so these predictions must not be used as primary belief measurements.

| Probe | Released fixed-position reference | All reasoning boundaries | Same-state pre-boundary window | Decision |
|---|---:|---:|---:|---|
| Pre cognitive-map, overall grid accuracy | 96.6% | 67.1% | 65.9% | Reject |
| Post cognitive-map, overall grid accuracy | 90.7% | 63.3% | 65.3% | Reject |
| Pre cognitive-map, directional wall accuracy | 79.5% | 47.5% | 72.9% | Reject for intermediate positions |
| Post cognitive-map, directional wall accuracy | 84.1% | 51.9% | 72.9% | Reject for intermediate positions |
| Pre plan decoder, next-action accuracy | 53.3% | 26.7% | 25.0% | Reject |
| Post plan decoder, next-action accuracy | 60.0% | 14.0% | 25.0% | Reject |

The cognitive-map checkpoint supports agent, wall, goal, open-cell, and padding classes. It does not probe key possession, door-open status, or predicted action consequences; those outputs remain missing.

The low same-state pre-boundary performance means the failure is not attributable only to arbitrary reasoning positions. Prompt-format and task-distribution transfer also matter. Primary per-position white-box analysis requires probes trained and held-out validated on this activation and prompt distribution.

Artifacts:

- `intermediate_cognitive_probe_eval/` contains exploratory cognitive-map predictions.
- `intermediate_plan_decoder_eval/` contains exploratory plan-decoder predictions.
- `whitebox_intermediate_validation_summary.csv` contains the consolidated validation metrics.
