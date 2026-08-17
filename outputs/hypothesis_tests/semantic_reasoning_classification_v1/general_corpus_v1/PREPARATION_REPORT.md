# General semantic-labelling preparation

This is an analysis-independent labelling dataset. Model inputs contain only a target sentence and its preceding reasoning context. They contain no change-point status, matched-pair role, action outcome, activation, attention, or belief measurement.

- Full sentence inventory: 7,038
- Previously labelled sentences: 320
- Previously unlabelled sentences: 6,718
- Random general-corpus calibration sentences: 240
- Existing-reference stress sentences: 80
- Hidden repeated items: 12
- Few-shot demonstrations: 12
- Broad-label few-shot demonstrations: 16 (four per broad label)

The random calibration sample covers all 31 trajectories represented by the cohort and is balanced across reasoning-progress and sentence-length strata. The existing-reference items are a stress test, not independent human ground truth: most references originated from GPT-OSS-20B, with low-confidence cases AI-adjudicated.

Do not give `calibration_key.csv` to a label model. It contains sampling roles and reference labels. `calibration_items.csv` is the blinded model input.
