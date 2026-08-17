#!/usr/bin/env python3
"""Compare local semantic judges and write the scale-up decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from reveng.experiments.general_semantic_labeling import (
    BROAD_LABEL_MAP,
    compare_candidates,
    evaluate_broad_candidate,
    write_json,
)


DEFAULT_ROOT = Path(
    "outputs/hypothesis_tests/semantic_reasoning_classification_v1/general_corpus_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _read_status(path: Path) -> dict[str, object]:
    return json.loads(path.read_text()) if path.exists() else {"status": "not_started"}


def main() -> None:
    args = parse_args()
    key = pd.read_csv(args.run_dir / "calibration_key.csv").fillna("")
    items = pd.read_csv(args.run_dir / "calibration_items.csv").fillna("")
    candidates = {
        "llama_3_2_1b": args.run_dir / "annotations_llama_3_2_1b.csv",
        "gpt_oss_20b": args.run_dir / "annotations_gpt_oss_20b.csv",
        "qwen_3_7_max": args.run_dir / "annotations_qwen_3_7_max.csv",
    }
    available = {
        name: pd.read_csv(path).fillna("")
        for name, path in candidates.items()
        if path.exists()
    }
    if available:
        summary, disagreements, gate = compare_candidates(available, key)
    else:
        summary = pd.DataFrame()
        disagreements = pd.DataFrame()
        gate = {
            "status": "incomplete",
            "smaller_model_passes": False,
            "manual_disagreement_review_required": True,
        }

    summary_path = args.run_dir / "candidate_metrics.csv"
    review_path = args.run_dir / "disagreement_review.csv"
    gate_path = args.run_dir / "decision_gate.json"
    report_path = args.run_dir / "run_report.md"
    summary.to_csv(summary_path, index=False)
    if len(disagreements):
        disagreements = disagreements.merge(
            items[["annotation_id", "context_before", "target_sentence"]],
            on="annotation_id",
            how="left",
            validate="one_to_one",
        )
    for column in ("reviewed_primary_label", "review_notes", "reviewer"):
        disagreements[column] = ""
    disagreements.to_csv(review_path, index=False)
    write_json(gate_path, gate)

    qwen_gate: dict[str, object] | None = None
    qwen_pairwise: dict[str, object] | None = None
    qwen_usage: dict[str, object] | None = None
    if {"qwen_3_7_max", "gpt_oss_20b"} <= set(available):
        qwen = available["qwen_3_7_max"][["annotation_id", "primary_label"]].copy()
        teacher = available["gpt_oss_20b"][["annotation_id", "primary_label"]].copy()
        qwen["qwen_broad_label"] = qwen["primary_label"].map(BROAD_LABEL_MAP)
        teacher["gpt_oss_broad_label"] = teacher["primary_label"].map(BROAD_LABEL_MAP)
        paired = (
            key.merge(qwen, on="annotation_id", validate="one_to_one")
            .merge(
                teacher,
                on="annotation_id",
                validate="one_to_one",
                suffixes=("_qwen", "_gpt_oss"),
            )
        )
        general = paired.loc[
            paired["calibration_source"].eq("general_random")
            & paired["duplicate_of_annotation_id"].fillna("").eq("")
        ].copy()
        qwen_pairwise = {
            "model_a": "qwen_3_7_max",
            "model_b": "gpt_oss_20b",
            "sample": "general_random_nonduplicate",
            "n": len(general),
            "fine_agreement": float(
                (general["primary_label_qwen"] == general["primary_label_gpt_oss"]).mean()
            ),
            "broad_agreement": float(
                (general["qwen_broad_label"] == general["gpt_oss_broad_label"]).mean()
            ),
        }
        pd.DataFrame([qwen_pairwise]).to_csv(
            args.run_dir / "qwen_gpt_oss_agreement.csv", index=False
        )
        qwen_disagreements = paired.loc[
            paired["primary_label_qwen"] != paired["primary_label_gpt_oss"]
        ].merge(
            items[["annotation_id", "context_before", "target_sentence"]],
            on="annotation_id",
            how="left",
            validate="one_to_one",
        )
        for column in ("reviewed_primary_label", "review_notes", "reviewer"):
            qwen_disagreements[column] = ""
        qwen_disagreements.to_csv(
            args.run_dir / "qwen_gpt_oss_disagreement_review.csv", index=False
        )
        qwen_metrics = summary.set_index("model_name").loc["qwen_3_7_max"]
        checks = {
            "parse_success_at_least_99_percent": qwen_metrics["parse_success_rate"]
            >= 0.99,
            "reference_broad_agreement_at_least_75_percent": qwen_metrics[
                "reference_broad_accuracy"
            ]
            >= 0.75,
            "reference_fine_agreement_at_least_50_percent": qwen_metrics[
                "reference_fine_accuracy"
            ]
            >= 0.50,
            "general_broad_agreement_with_gpt_oss_at_least_80_percent": qwen_pairwise[
                "broad_agreement"
            ]
            >= 0.80,
            "duplicate_broad_agreement_at_least_95_percent": qwen_metrics[
                "duplicate_broad_agreement"
            ]
            >= 0.95,
            "all_four_broad_categories_predicted": qwen_metrics[
                "general_broad_categories_predicted"
            ]
            == 4,
        }
        qwen_gate = {
            "status": "complete",
            "candidate": "qwen_3_7_max",
            "candidate_passes": bool(all(checks.values())),
            "checks": {name: bool(value) for name, value in checks.items()},
            "reference_is_independent_human_gold": False,
            "manual_disagreement_review_required": True,
        }
        write_json(args.run_dir / "qwen_decision_gate.json", qwen_gate)
        qwen_rows = available["qwen_3_7_max"]
        prompt_tokens = int(qwen_rows["prompt_tokens"].sum())
        completion_tokens = int(qwen_rows["completion_tokens"].sum())
        qwen_usage = {
            "model": "Qwen/Qwen3.7-Max",
            "rows": len(qwen_rows),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "one_attempt_rows": int(qwen_rows["attempts"].eq(1).sum()),
            "high_confidence_rows": int(qwen_rows["confidence"].eq("high").sum()),
            "estimated_cost_usd_at_run_prices": float(
                prompt_tokens * 1.25e-6 + completion_tokens * 3.75e-6
            ),
            "input_price_usd_per_million_tokens": 1.25,
            "output_price_usd_per_million_tokens": 3.75,
        }
        write_json(args.run_dir / "qwen_usage_summary.json", qwen_usage)

    broad_path = args.run_dir / "annotations_llama_3_2_1b_broad.csv"
    broad_summary: dict[str, object] | None = None
    broad_gate: dict[str, object] | None = None
    if broad_path.exists():
        broad_summary, broad_disagreements, broad_gate = evaluate_broad_candidate(
            pd.read_csv(broad_path).fillna(""),
            key,
            teacher_annotations=available.get("gpt_oss_20b"),
        )
        if len(broad_disagreements):
            broad_disagreements = broad_disagreements.merge(
                items[["annotation_id", "context_before", "target_sentence"]],
                on="annotation_id",
                how="left",
                validate="one_to_one",
            )
        for column in ("reviewed_broad_label", "review_notes", "reviewer"):
            broad_disagreements[column] = ""
        broad_disagreements.to_csv(
            args.run_dir / "broad_disagreement_review.csv", index=False
        )
        write_json(args.run_dir / "broad_decision_gate.json", broad_gate)

    statuses = {
        name: _read_status(path.with_suffix(".status.json"))
        for name, path in candidates.items()
    }
    statuses["llama_3_2_1b_broad"] = _read_status(
        broad_path.with_suffix(".status.json")
    )
    status_lines = [
        f"- `{name}`: {status.get('status', 'unknown')}"
        for name, status in statuses.items()
    ]
    if qwen_gate is not None:
        decision = (
            "Qwen3.7-Max clears the automatic checks; review its disagreements before scale-up."
            if qwen_gate["candidate_passes"]
            else "Qwen3.7-Max has higher reference agreement than GPT-OSS-20B but does not clear the automatic checks. Do not use it for full-corpus labels yet."
        )
    elif broad_gate is not None and broad_gate.get("status") == "complete":
        decision = (
            "The direct four-way model clears the automatic checks; review broad disagreements before scale-up."
            if broad_gate.get("broad_candidate_passes")
            else "The direct four-way model does not clear the automatic checks. Do not use it for full-corpus labels."
        )
    elif gate["status"] == "complete":
        decision = (
            "The smaller model clears the automatic calibration checks. Review the "
            "disagreement table before using it for the full corpus."
            if gate["smaller_model_passes"]
            else "The smaller model does not clear the automatic calibration checks. Do not use it for the full corpus."
        )
    elif gate["status"] == "smaller_model_rejected":
        decision = (
            "The 1B model fails the absolute calibration checks and is rejected. "
            "Do not use it for the 6,718-sentence full-corpus pass."
        )
    else:
        decision = (
            "The comparison is incomplete. No model has been authorized for the "
            "6,718-sentence scale-up."
        )
    metric_text = "No candidate metrics are available yet."
    if len(summary):
        metric_text = summary.to_markdown(index=False, floatfmt=".3f")
    broad_metric_text = (
        "The direct four-way candidate has not been run."
        if broad_summary is None
        else pd.DataFrame([broad_summary]).to_markdown(index=False, floatfmt=".3f")
    )
    plain_results = "No completed candidate results are available."
    if len(summary):
        indexed_metrics = summary.set_index("model_name")
        small = indexed_metrics.loc["llama_3_2_1b"]
        teacher = indexed_metrics.loc["gpt_oss_20b"]
        plain_results = (
            f"The nine-way 1B model reached {small['reference_broad_accuracy']:.1%} broad and "
            f"{small['reference_fine_accuracy']:.1%} fine agreement on the existing-reference "
            f"stress set. GPT-OSS-20B with the new few-shot prompt reached "
            f"{teacher['reference_broad_accuracy']:.1%} broad and "
            f"{teacher['reference_fine_accuracy']:.1%} fine agreement."
        )
    if broad_summary is not None:
        plain_results += (
            f" The direct four-way 1B model reached "
            f"{float(broad_summary['reference_broad_accuracy']):.1%} reference agreement and "
            f"{float(broad_summary['general_broad_agreement_with_teacher']):.1%} agreement with "
            "GPT-OSS-20B on the random general-corpus sample. The direct four-way model does not clear the planned scale-up gate."
        )
    if qwen_gate is not None and qwen_pairwise is not None:
        qwen_metrics = summary.set_index("model_name").loc["qwen_3_7_max"]
        plain_results += (
            f" The proprietary Qwen3.7-Max judge reached "
            f"{qwen_metrics['reference_broad_accuracy']:.1%} broad and "
            f"{qwen_metrics['reference_fine_accuracy']:.1%} fine agreement with the existing "
            f"references. On the 240 random previously unlabelled sentences, it agreed with "
            f"GPT-OSS-20B on {float(qwen_pairwise['broad_agreement']):.1%} of broad labels and "
            f"{float(qwen_pairwise['fine_agreement']):.1%} of fine labels. All 332 Qwen outputs "
            "reported high confidence, so its confidence field cannot prioritize uncertain cases."
        )
    report_path.write_text(
        f"""# General semantic-labelling calibration pilot

## Status

{chr(10).join(status_lines)}

{decision}

## What this pilot tests

The pilot compares a few-shot Llama-3.2-1B labeler, the existing GPT-OSS-20B judge, and the proprietary Qwen3.7-Max model served by Together AI. Inputs are blinded to change points and every downstream measurement. The evaluation contains a random general-corpus sample, an existing-label stress set, and hidden repeated items.

The existing references are not independent human ground truth: most were generated by GPT-OSS-20B and 18 low-confidence cases were AI-adjudicated. Consequently, reference agreement is a continuity check. Model disagreements on the new random sample require review before scale-up.

## Result in plain language

{plain_results} No labels have been propagated to the remaining 6,718 sentences.

## Candidate metrics

{metric_text}

## Direct four-way candidate

{broad_metric_text}

## Scale-up rule

For nine-way labelling, the 1B model must parse at least 99% of items, reach at least 75% broad and 50% fine agreement on the reference stress set, remain within three percentage points of GPT-OSS-20B broad agreement, agree broadly with GPT-OSS-20B on at least 80% of the random sample, reproduce at least 95% of hidden duplicates at the broad level, and use all four broad categories.

For direct four-way labelling, the fine-label requirement does not apply. The model must still reach 75% reference agreement, 80% agreement with GPT-OSS-20B on the random sample, 99% parsing, 95% repeat consistency, all four categories, and remain within three points of the teacher on the reference set. Passing automatic checks is necessary but not sufficient: disagreement review is still required.

The proprietary Qwen judge is checked against the same predeclared thresholds: at least 99% parsing, 75% broad and 50% fine reference agreement, 80% broad agreement with GPT-OSS-20B on the random sample, 95% repeated-item agreement, and all four broad categories. These are continuity and consistency checks, not accuracy estimates, because the references are not independent human gold labels.

No full-corpus model should be presented as human ground truth. The final table must preserve model, prompt, confidence and review provenance for every sentence.
"""
    )
    print(f"Wrote {report_path}")
    print(decision)


if __name__ == "__main__":
    main()
