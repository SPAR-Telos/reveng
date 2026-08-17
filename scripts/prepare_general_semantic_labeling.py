#!/usr/bin/env python3
"""Prepare blinded, analysis-independent semantic labelling inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from reveng.experiments.general_semantic_labeling import (
    broad_few_shots_to_json,
    build_general_inventory,
    few_shots_to_json,
    prepare_calibration_items,
    select_broad_few_shot_examples,
    select_few_shot_examples,
    write_json,
)


DEFAULT_ROOT = Path(
    "outputs/hypothesis_tests/semantic_reasoning_classification_v1/general_corpus_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-inventory",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/semantic_reasoning_classification_v1/"
            "final_analysis/all_reasoning_sentence_inventory.csv"
        ),
    )
    parser.add_argument(
        "--canonical-sentences",
        type=Path,
        default=Path("data/behavioral_probes/doorkey_chunking_validation/sentences.csv"),
    )
    parser.add_argument(
        "--existing-labels",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/semantic_reasoning_classification_v1/"
            "final_analysis/sentence_label_table.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--general-sample-size", type=int, default=240)
    parser.add_argument("--reference-sample-size", type=int, default=80)
    parser.add_argument("--duplicate-items", type=int, default=12)
    parser.add_argument("--few-shot-examples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_inventory = pd.read_csv(args.source_inventory)
    canonical = pd.read_csv(args.canonical_sentences)
    existing = pd.read_csv(args.existing_labels)
    inventory = build_general_inventory(source_inventory, canonical)
    few_shots = select_few_shot_examples(
        existing, n_examples=args.few_shot_examples, seed=args.seed
    )
    broad_few_shots = select_broad_few_shot_examples(existing, seed=args.seed)
    blinded, key = prepare_calibration_items(
        inventory,
        existing,
        few_shots,
        general_sample_size=args.general_sample_size,
        reference_sample_size=args.reference_sample_size,
        duplicate_items=args.duplicate_items,
        seed=args.seed,
    )

    paths = {
        "inventory": args.output_dir / "sentence_inventory.csv",
        "few_shots": args.output_dir / "few_shot_examples.json",
        "broad_few_shots": args.output_dir / "broad_few_shot_examples.json",
        "calibration_items": args.output_dir / "calibration_items.csv",
        "calibration_key": args.output_dir / "calibration_key.csv",
        "report": args.output_dir / "PREPARATION_REPORT.md",
        "runbook": args.output_dir / "RUNBOOK.md",
        "manifest": args.output_dir / "preparation_manifest.json",
    }
    inventory.to_csv(paths["inventory"], index=False)
    write_json(paths["few_shots"], few_shots_to_json(few_shots))
    write_json(paths["broad_few_shots"], broad_few_shots_to_json(broad_few_shots))
    blinded.to_csv(paths["calibration_items"], index=False)
    key.to_csv(paths["calibration_key"], index=False)
    non_duplicates = key["duplicate_of_annotation_id"].fillna("").eq("")
    source_counts = key.loc[non_duplicates, "calibration_source"].value_counts()
    paths["report"].write_text(
        f"""# General semantic-labelling preparation

This is an analysis-independent labelling dataset. Model inputs contain only a target sentence and its preceding reasoning context. They contain no change-point status, matched-pair role, action outcome, activation, attention, or belief measurement.

- Full sentence inventory: {len(inventory):,}
- Previously labelled sentences: {int(inventory['previously_labelled'].sum()):,}
- Previously unlabelled sentences: {int((~inventory['previously_labelled']).sum()):,}
- Random general-corpus calibration sentences: {int(source_counts.get('general_random', 0)):,}
- Existing-reference stress sentences: {int(source_counts.get('matched_reference_stress', 0)):,}
- Hidden repeated items: {int(key['duplicate_of_annotation_id'].fillna('').ne('').sum()):,}
- Few-shot demonstrations: {len(few_shots):,}
- Broad-label few-shot demonstrations: {len(broad_few_shots):,} (four per broad label)

The random calibration sample covers all {key.loc[key['calibration_source'].eq('general_random'), 'environment_id'].nunique()} trajectories represented by the cohort and is balanced across reasoning-progress and sentence-length strata. The existing-reference items are a stress test, not independent human ground truth: most references originated from GPT-OSS-20B, with low-confidence cases AI-adjudicated.

Do not give `calibration_key.csv` to a label model. It contains sampling roles and reference labels. `calibration_items.csv` is the blinded model input.
"""
    )
    paths["runbook"].write_text(
        """# General semantic-labelling runbook

Run from the repository root. Both judges are local and require CUDA; neither silently falls back to CPU.

## 1. Smaller few-shot candidate

```bash
.venv/bin/python scripts/run_general_semantic_labeler.py --resume
```

## 2. GPT-OSS-20B continuity judge

```bash
.venv/bin/python scripts/run_general_semantic_labeler.py \\
  --model-path /root/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee \\
  --model-name openai/gpt-oss-20b \\
  --minimum-free-gib 16 \\
  --output outputs/hypothesis_tests/semantic_reasoning_classification_v1/general_corpus_v1/annotations_gpt_oss_20b.csv \\
  --resume
```

## 3. Compare and decide

```bash
.venv/bin/python scripts/analyze_general_semantic_labeling_pilot.py
```

Do not start the full-corpus pass unless `decision_gate.json` passes and the disagreement review is acceptable.

The cached GPT-OSS-20B snapshot uses native MXFP4 weights and was verified to fit on the 23-GiB L4 with the 16-GiB pre-load guard.
"""
    )
    write_json(
        paths["manifest"],
        {
            "analysis": "general_semantic_labeling_calibration_v1",
            "seed": args.seed,
            "context_sentences": 8,
            "context_character_limit": 1800,
            "inputs": {
                "source_inventory": {
                    "path": str(args.source_inventory),
                    "sha256": sha256(args.source_inventory),
                },
                "canonical_sentences": {
                    "path": str(args.canonical_sentences),
                    "sha256": sha256(args.canonical_sentences),
                },
                "existing_labels": {
                    "path": str(args.existing_labels),
                    "sha256": sha256(args.existing_labels),
                },
            },
            "outputs": {name: str(path) for name, path in paths.items()},
            "full_inventory_rows": len(inventory),
            "general_calibration_rows": int(source_counts.get("general_random", 0)),
            "reference_stress_rows": int(
                source_counts.get("matched_reference_stress", 0)
            ),
            "duplicate_rows": int(
                key["duplicate_of_annotation_id"].fillna("").ne("").sum()
            ),
            "model_input_is_blinded": True,
        },
    )
    print(f"Wrote {paths['inventory']}")
    print(f"Wrote blinded calibration items: {paths['calibration_items']}")


if __name__ == "__main__":
    main()
