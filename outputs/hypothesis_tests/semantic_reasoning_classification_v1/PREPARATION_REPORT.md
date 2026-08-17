# Semantic Reasoning Classification Preparation

- Detected change-point sentences: 160
- Same-state matched comparison sentences: 160
- Matched pairs: 160
- Fixed environment states: 44
- Trajectories: 30
- Median absolute reasoning-progress difference: 0.0292
- Maximum absolute reasoning-progress difference: 0.2105
- Primary matched pairs within 0.10 reasoning progress: 152
- Sensitivity-only pairs above 0.10 reasoning progress: 8
- Human pilot: 44 rows, including 4 hidden duplicates

Controls are from the same fixed environment state, are outside a plus-or-minus
3-sentence window around every detected change point, and
are matched without replacement on reasoning progress and sentence length
using global minimum-cost assignment. The primary analysis uses pairs
within 0.10 reasoning progress; all pairs remain available for sensitivity
analysis. Outcome, action, activation, and failure labels are absent from the
annotation template.

Start with `human_pilot_annotation.csv` and
`SEMANTIC_ANNOTATION_GUIDE.md`. Do not share `annotation_key.csv` or
`pilot_duplicate_key.csv` with annotators.
