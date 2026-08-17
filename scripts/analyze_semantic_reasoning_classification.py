#!/usr/bin/env python3
"""Validate semantic labels and compare detected points with matched sentences."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

from reveng.experiments.semantic_reasoning_classification import (
    SEMANTIC_LABELS,
    validate_annotation_rows,
)


PRIMARY_COMPOSITE = {"verification", "consolidation", "restatement"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/semantic_reasoning_classification_v1/"
            "human_pilot_annotation.csv"
        ),
    )
    parser.add_argument(
        "--annotation-key",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/semantic_reasoning_classification_v1/"
            "annotation_key.csv"
        ),
    )
    parser.add_argument(
        "--duplicate-key",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/semantic_reasoning_classification_v1/"
            "pilot_duplicate_key.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/semantic_reasoning_classification_v1/"
            "human_label_analysis"
        ),
    )
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def normalized_annotations(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].fillna("").astype(str).str.strip()
    return result


def duplicate_agreement(
    annotations: pd.DataFrame,
    duplicate_key: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    lookup = annotations.set_index("annotation_id")
    output: list[dict[str, object]] = []
    for row in duplicate_key.itertuples():
        original_id = str(row.duplicate_of_annotation_id)
        duplicate_id = str(row.duplicate_annotation_id)
        if original_id not in lookup.index or duplicate_id not in lookup.index:
            continue
        original = lookup.loc[original_id]
        duplicate = lookup.loc[duplicate_id]
        if not original["primary_label"] or not duplicate["primary_label"]:
            continue
        output.append(
            {
                "original_annotation_id": original_id,
                "duplicate_annotation_id": duplicate_id,
                "original_label": original["primary_label"],
                "duplicate_label": duplicate["primary_label"],
                "primary_label_agrees": (
                    original["primary_label"] == duplicate["primary_label"]
                ),
            }
        )
    rows = pd.DataFrame(output)
    summary = {
        "n_labeled_duplicate_pairs": len(rows),
        "n_agreeing_duplicate_pairs": (
            int(rows["primary_label_agrees"].sum()) if len(rows) else 0
        ),
        "duplicate_primary_label_agreement": (
            float(rows["primary_label_agrees"].mean()) if len(rows) else float("nan")
        ),
    }
    return rows, summary


def paired_category_summary(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    labels: list[tuple[str, set[str]]] = [
        (label, {label}) for label in SEMANTIC_LABELS
    ] + [("verification_or_consolidation_or_restatement", PRIMARY_COMPOSITE)]
    for analysis_set, subset in (
        ("primary_matches", rows[rows["match_quality"].eq("primary")]),
        ("all_matches", rows),
    ):
        for label, accepted in labels:
            values = subset.copy()
            values["has_feature"] = values["primary_label"].isin(accepted)
            paired = values.pivot(
                index="pair_id", columns="item_role", values="has_feature"
            ).dropna()
            event = paired["detected_change_point"].astype(bool)
            control = paired["matched_non_change_sentence"].astype(bool)
            event_only = int((event & ~control).sum())
            control_only = int((~event & control).sum())
            discordant = event_only + control_only
            p_value = (
                float(binomtest(event_only, discordant, 0.5).pvalue)
                if discordant
                else 1.0
            )
            output.append(
                {
                    "analysis_set": analysis_set,
                    "semantic_feature": label,
                    "n_pairs": len(paired),
                    "rate_at_detected_change_points": event.mean(),
                    "rate_at_matched_non_change_sentences": control.mean(),
                    "paired_rate_difference": event.mean() - control.mean(),
                    "pairs_feature_only_at_change_point": event_only,
                    "pairs_feature_only_at_comparison": control_only,
                    "mcnemar_exact_p": p_value,
                }
            )
    return pd.DataFrame(output)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations = normalized_annotations(pd.read_csv(args.annotations))
    validation = validate_annotation_rows(
        annotations,
        allow_partial=args.allow_partial,
    )
    labeled = annotations[annotations["primary_label"].ne("")].copy()
    if labeled.empty:
        raise ValueError("No completed semantic labels were found")

    duplicate_key = pd.read_csv(args.duplicate_key)
    duplicate_rows, duplicate_summary = duplicate_agreement(
        labeled, duplicate_key
    )
    duplicate_ids = set(duplicate_key["duplicate_annotation_id"].astype(str))
    unique_labels = labeled[~labeled["annotation_id"].isin(duplicate_ids)]
    key = pd.read_csv(args.annotation_key)
    merged = unique_labels.merge(
        key,
        on="annotation_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(unique_labels):
        raise ValueError("Some labeled annotation IDs do not match the hidden key")

    category_summary = paired_category_summary(merged)
    duplicate_rows.to_csv(args.output_dir / "duplicate_agreement.csv", index=False)
    merged.to_csv(args.output_dir / "labeled_semantic_rows.csv", index=False)
    category_summary.to_csv(
        args.output_dir / "semantic_category_paired_comparison.csv", index=False
    )

    primary = category_summary[
        (category_summary["analysis_set"] == "primary_matches")
        & (
            category_summary["semantic_feature"]
            == "verification_or_consolidation_or_restatement"
        )
    ]
    primary_text = (
        "The primary comparison is unavailable until both members of at least "
        "one close matched pair are labeled."
        if primary.empty or int(primary.iloc[0]["n_pairs"]) == 0
        else (
            f"Among {int(primary.iloc[0]['n_pairs'])} completely labeled close "
            "pairs, verification, consolidation, or restatement occurs in "
            f"{primary.iloc[0]['rate_at_detected_change_points']:.1%} of detected "
            "change-point sentences and "
            f"{primary.iloc[0]['rate_at_matched_non_change_sentences']:.1%} of "
            "matched comparison sentences "
            f"(paired difference {primary.iloc[0]['paired_rate_difference']:+.1%}; "
            f"exact McNemar p={primary.iloc[0]['mcnemar_exact_p']:.4f})."
        )
    )
    report = f"""# Semantic Label Analysis

## Coverage

- Annotation rows supplied: {len(annotations)}
- Completed rows: {len(labeled)}
- Unique completed sentences after removing hidden duplicates: {len(merged)}
- Completely labeled matched pairs: {merged.groupby('pair_id').filter(lambda x: len(x) == 2).pair_id.nunique()}
- Labeled duplicate pairs: {duplicate_summary['n_labeled_duplicate_pairs']}
- Exact duplicate-label agreement: {duplicate_summary['duplicate_primary_label_agreement']}

## Primary Test

{primary_text}

The comparison is observational. A category difference can identify the
reasoning functions associated with action-distribution changes, but cannot
show that those sentences caused the recommendation to change.

## Validation

{chr(10).join(f'- {line}' for line in validation)}
"""
    (args.output_dir / "semantic_label_analysis_report.md").write_text(report)


if __name__ == "__main__":
    main()
