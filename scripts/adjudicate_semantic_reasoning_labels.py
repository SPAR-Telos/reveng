#!/usr/bin/env python3
"""Combine two blinded AI judges and prepare disagreements for adjudication."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from reveng.experiments.semantic_reasoning_classification import ANNOTATION_COLUMNS


ROOT = Path("outputs/hypothesis_tests/semantic_reasoning_classification_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-a", type=Path, required=True)
    parser.add_argument("--judge-b", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "judge_agreement"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(args.judge_a).fillna("")
    b = pd.read_csv(args.judge_b).fillna("")
    keep = ["annotation_id", "context_before", "target_sentence", *ANNOTATION_COLUMNS]
    merged = a[keep].merge(
        b[keep],
        on="annotation_id",
        suffixes=("_judge_a", "_judge_b"),
        validate="one_to_one",
    )
    merged["primary_label_agrees"] = (
        merged["primary_label_judge_a"] == merged["primary_label_judge_b"]
    )
    agreements = merged[merged["primary_label_agrees"]].copy()
    disagreements = merged[~merged["primary_label_agrees"]].copy()
    agreements.to_csv(args.output_dir / "agreements.csv", index=False)
    disagreements.to_csv(args.output_dir / "disagreements_for_adjudication.csv", index=False)

    summary_rows = []
    for column in [*ANNOTATION_COLUMNS[:4], "primary_label"]:
        left = merged[f"{column}_judge_a"]
        right = merged[f"{column}_judge_b"]
        summary_rows.append(
            {
                "field": column,
                "n": len(merged),
                "exact_agreement": float((left == right).mean()),
                "cohen_kappa": float(cohen_kappa_score(left, right)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "judge_agreement_summary.csv", index=False)
    print(
        f"rows={len(merged)} label_agreement={merged.primary_label_agrees.mean():.3f} "
        f"disagreements={len(disagreements)}"
    )


if __name__ == "__main__":
    main()
