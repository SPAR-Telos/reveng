#!/usr/bin/env python3
"""Train and evaluate a trajectory-held-out semantic sentence classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from reveng.experiments.semantic_reasoning_classification import (
    validate_annotation_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/semantic_reasoning_classification_v1/"
            "full_annotation_template.csv"
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
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/semantic_reasoning_classification_v1/"
            "supervised_classifier"
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-labeled-rows", type=int, default=100)
    return parser.parse_args()


def make_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=seed,
                ),
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    annotations = pd.read_csv(args.annotations)
    validate_annotation_rows(annotations)
    key = pd.read_csv(args.annotation_key)
    rows = annotations.merge(
        key[["annotation_id", "trajectory_id"]],
        on="annotation_id",
        how="inner",
        validate="one_to_one",
    )
    if len(rows) < args.minimum_labeled_rows:
        raise ValueError(
            f"Only {len(rows)} labeled rows; require at least "
            f"{args.minimum_labeled_rows} before supervised training"
        )
    rows["model_text"] = (
        "PRECEDING REASONING:\n"
        + rows["context_before"].fillna("").astype(str)
        + "\nTARGET SENTENCE:\n"
        + rows["target_sentence"].fillna("").astype(str)
    )
    groups = rows["trajectory_id"].astype(str)
    n_groups = groups.nunique()
    folds = min(args.folds, n_groups)
    if folds < 2:
        raise ValueError("Need at least two trajectories for held-out evaluation")

    splitter = GroupKFold(n_splits=folds)
    predictions = np.empty(len(rows), dtype=object)
    fold_ids = np.full(len(rows), -1, dtype=int)
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(rows, rows["primary_label"], groups)
    ):
        pipeline = make_pipeline(args.seed + fold)
        pipeline.fit(
            rows.iloc[train_indices]["model_text"],
            rows.iloc[train_indices]["primary_label"],
        )
        predictions[test_indices] = pipeline.predict(
            rows.iloc[test_indices]["model_text"]
        )
        fold_ids[test_indices] = fold

    rows["predicted_primary_label"] = predictions
    rows["fold"] = fold_ids
    accuracy = float(accuracy_score(rows["primary_label"], predictions))
    macro_f1 = float(
        f1_score(rows["primary_label"], predictions, average="macro")
    )
    weighted_f1 = float(
        f1_score(rows["primary_label"], predictions, average="weighted")
    )
    class_report = pd.DataFrame(
        classification_report(
            rows["primary_label"],
            predictions,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "held_out_predictions.csv", index=False)
    class_report.to_csv(args.output_dir / "classification_metrics_by_label.csv")
    final_pipeline = make_pipeline(args.seed)
    final_pipeline.fit(rows["model_text"], rows["primary_label"])
    joblib.dump(final_pipeline, args.output_dir / "semantic_classifier.joblib")

    summary = {
        "labeled_rows": len(rows),
        "trajectories": n_groups,
        "grouped_folds": folds,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "model": "TF-IDF unigram-bigram logistic regression",
        "regularization_C": 1.0,
        "class_weight": "balanced",
        "seed": args.seed,
    }
    (args.output_dir / "classifier_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    report = f"""# Supervised Semantic Reasoning Classifier

- Labeled sentences: {len(rows)}
- Held-out trajectories: {n_groups} across {folds} grouped folds
- Accuracy: {accuracy:.3f}
- Macro F1: {macro_f1:.3f}
- Weighted F1: {weighted_f1:.3f}

The model uses TF-IDF unigram and bigram features from the target sentence and
the preceding eight-sentence context, followed by class-balanced logistic
regression with `C=1.0`. Every sentence from a trajectory remains in the same
fold. This classifier should be used to scale a calibrated human or judge
taxonomy, not as independent semantic evidence.
"""
    (args.output_dir / "classifier_report.md").write_text(report)


if __name__ == "__main__":
    main()
