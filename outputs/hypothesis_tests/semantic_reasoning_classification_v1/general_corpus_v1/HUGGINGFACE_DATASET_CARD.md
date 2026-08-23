---
pretty_name: GPT-OSS DoorKey Sentence-Level Semantic Labels v3
task_categories:
  - text-classification
language:
  - en
tags:
  - reasoning
  - chain-of-thought
  - semantic-labeling
  - multi-label-classification
  - minigrid
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/annotations_gpt_oss_20b_multilabel_v3_full.csv
---

# GPT-OSS DoorKey sentence-level semantic labels v3

This dataset contains sentence-level semantic-function annotations for GPT-OSS-20B reasoning traces from DoorKey environments. Labels are multi-valued and intended to make existing behavioral, activation, attention, and trajectory results easier to interpret.

## Contents

- 7,038 sentence rows covering the full inventory
- 7,036 successful annotations and 2 rows retaining annotation errors
- Preceding reasoning context and target sentence
- Multi-label semantic functions and a deterministically derived primary label
- Observable cue fields, confidence, short rationale, judge metadata, token usage, and raw response

The canonical taxonomy is documented in `docs/MULTILABEL_TAXONOMY.md`. Labels include `state_readout`, `route_planning`, `verification`, `correction`, `new_inference`, `consolidation`, `restatement`, `procedural_continuation`, and `action_commitment`.

## Suggested uses

- Stratify activation, attention, uncertainty, or behavioral results by reasoning function.
- Test whether semantic functions or transitions predict action changes, errors, recovery, or commitment.
- Select interpretable sentence subsets for causal interventions or human review.

## Loading

```python
from datasets import load_dataset

dataset = load_dataset("YOUR_USERNAME/YOUR_DATASET_REPO", split="train")
```

`semantic_labels` is stored in the CSV as a JSON array string and should be parsed before multi-label analysis.

## Annotation procedure

The judge was Together-hosted `openai/gpt-oss-20b` with low reasoning effort and temperature zero. It received the preceding reasoning, target sentence, taxonomy instructions, and 12 few-shot examples; later reasoning was hidden. See `few_shot_examples_multilabel.json`, the run manifest, and the taxonomy document for exact provenance.

## Limitations

- The full run contains one judgment per sentence and no deliberately repeated audit items. Duplicate consistency and inter-judge primary-label agreement are therefore not measurable from the full run.
- The output should be treated as exploratory annotation rather than human-validated ground truth.
- `new_inference` is broad and frequently overlaps verification and route planning.
- Confidence is highly concentrated at `high` and should not be interpreted as calibrated uncertainty.
- Fine boundaries such as route planning versus action commitment remain ambiguous.
- Two rows failed schema validation and are retained with their error fields populated.

See `docs/V3_FULL_AUDIT_REPORT.md` for descriptive statistics, examples, and additional caveats.
