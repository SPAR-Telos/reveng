#!/usr/bin/env python3
"""Prepare blinded semantic reasoning annotation items and matched controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from reveng.experiments.semantic_reasoning_classification import (
    build_blinded_annotation_rows,
    build_human_pilot,
    judge_item_json,
    match_change_points_to_controls,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--change-points",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/action_distribution_cpd_beast_v1/"
            "detected_change_points.csv"
        ),
    )
    parser.add_argument(
        "--positions",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/action_distribution_cpd_beast_v1/"
            "position_change_probabilities.csv"
        ),
    )
    parser.add_argument(
        "--sentences",
        type=Path,
        default=Path(
            "data/behavioral_probes/doorkey_chunking_validation/sentences.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/semantic_reasoning_classification_v1"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclusion-window", type=int, default=3)
    parser.add_argument("--context-sentences", type=int, default=8)
    parser.add_argument("--pilot-pairs", type=int, default=20)
    parser.add_argument("--pilot-duplicates", type=int, default=4)
    return parser.parse_args()


def full_preceding_context(
    sentences: pd.DataFrame,
    example_id: str,
    position_index: int,
) -> str:
    trace = sentences[
        sentences["trace_id"].eq(example_id)
        & sentences["kind"].eq("reasoning")
        & sentences["sentence_id"].astype(int).lt(position_index - 1)
    ].sort_values("sentence_id")
    return "\n".join(
        f"[{int(row.sentence_id) + 1}] {row.text}" for row in trace.itertuples()
    )


def write_guide(path: Path) -> None:
    path.write_text(
        """# Semantic Reasoning Annotation Guide

## Question

What function does the target sentence perform relative to the reasoning that
precedes it? Judge the sentence's discourse function, not whether its content
is factually correct or whether the model's action is optimal.

The annotation file is blinded: it does not reveal whether a sentence is a
detected action-distribution change point or a matched comparison sentence.

## Primary Labels

Choose exactly one label.

| Label | Use when |
|---|---|
| `correction` | The sentence explicitly rejects, revises, or reverses an earlier fact, route, or conclusion. Cues such as “wait”, “actually”, or “no” count only when they perform a real revision. |
| `verification` | The sentence checks whether an earlier fact or proposed route is valid, traversable, or efficient without yet replacing it. |
| `state_reconstruction` | The sentence directly reads, records, or locates an environment feature from the grid, such as an object coordinate, wall, or open cell. |
| `route_planning` | The sentence proposes a new action, route, subgoal, or action sequence. |
| `new_inference` | The sentence derives a new consequence, constraint, comparison, or conclusion that is not primarily a route proposal. |
| `consolidation` | The sentence combines multiple earlier facts or conclusions into a summary, plan, or decision without adding a materially new premise. |
| `restatement` | The sentence repeats one earlier fact, route, or conclusion without materially checking, combining, or changing it. |
| `procedural_continuation` | The sentence continues enumeration, arithmetic, bookkeeping, or connective narration and does not fit a more specific category. |
| `unclear` | The available context is insufficient or two labels remain equally plausible after applying the rules below. |

## Decision Order

Apply these questions in order:

1. Does it explicitly revise or reject earlier reasoning? Use `correction`.
2. Does it evaluate an earlier claim or route? Use `verification`.
3. Does it directly transcribe the current grid? Use `state_reconstruction`.
4. Does it construct a new route or action sequence? Use `route_planning`.
5. Does it derive a new consequence or conclusion? Use `new_inference`.
6. Does it combine several earlier results? Use `consolidation`.
7. Does it repeat one earlier result? Use `restatement`.
8. Otherwise use `procedural_continuation` or, if context is inadequate, `unclear`.

This precedence makes the labels mutually exclusive. For example, a sentence
that repeats a route in order to test it is `verification`, not `restatement`.
A sentence that says “Actually, that route is blocked” is `correction`, not
`verification`.

## Observable Cue Columns

Fill each with `yes`, `no`, or `unsure`.

| Column | Criterion |
|---|---|
| `explicitly_revises_prior_reasoning` | The sentence rejects or changes an earlier claim, route, or decision. |
| `evaluates_prior_route_or_claim` | The sentence tests an earlier proposal or factual claim. |
| `repeats_prior_content` | Substantive content already appears in the preceding context. |
| `introduces_new_information_or_plan` | The sentence adds a new fact, inference, route, or decision. |

The cue columns are not labels. A correction may both repeat prior content and
introduce a revised conclusion.

## Confidence

- `high`: the primary label follows directly from the wording and context.
- `medium`: one label is preferable but a nearby category is plausible.
- `low`: the label is tentative; use a short rationale explaining the ambiguity.

## Consistency Rules

- Do not use action correctness, failure status, or change-point status; these
  fields are intentionally hidden.
- Treat “then”, “wait”, and “therefore” as cues, not labels.
- A list of grid cells is `state_reconstruction` unless it evaluates a route.
- A proposed sequence of movements is `route_planning`.
- “This path works” after checking several constraints is `consolidation`.
- Use only preceding context. Do not infer intent from later sentences.
- Give the same label to semantically identical duplicate items. Four hidden
  duplicates in the pilot measure within-annotator consistency.

## Pilot Procedure

1. Label `human_pilot_annotation.csv` without opening the key file.
2. Review all `low`-confidence and `unclear` rows once.
3. Return the completed CSV.
4. We will report duplicate agreement and compare the labels with an independent
   temperature-zero judge before labeling the full set.
5. After the full 320-row template is adjudicated, train the supervised
   classifier with `scripts/train_semantic_reasoning_classifier.py`. The pilot
   is too small for reported classifier performance.
"""
    )


def write_judge_prompt(path: Path) -> None:
    path.write_text(
        """You are annotating the discourse function of one sentence in a
reasoning trace. Use only the preceding reasoning and target sentence. Do not
judge factual correctness, action optimality, or task success.

Apply this precedence:
correction; verification; state_reconstruction; route_planning; new_inference;
consolidation; restatement; procedural_continuation; unclear.

Definitions and output fields must follow SEMANTIC_ANNOTATION_GUIDE.md exactly.
Return only one JSON object matching required_output. Do not add markdown.
"""
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points = pd.read_csv(args.change_points)
    positions = pd.read_csv(args.positions)
    sentences = pd.read_csv(args.sentences)

    matched = match_change_points_to_controls(
        positions,
        points,
        sentences,
        exclusion_window=args.exclusion_window,
    )
    blinded, key = build_blinded_annotation_rows(
        matched,
        sentences,
        seed=args.seed,
        context_sentences=args.context_sentences,
    )
    pilot, duplicates = build_human_pilot(
        blinded,
        key,
        pairs=args.pilot_pairs,
        duplicate_items=args.pilot_duplicates,
        seed=args.seed,
    )

    matched.to_csv(args.output_dir / "matched_sentence_pairs.csv", index=False)
    blinded.to_csv(args.output_dir / "full_annotation_template.csv", index=False)
    key.to_csv(args.output_dir / "annotation_key.csv", index=False)
    pilot.to_csv(args.output_dir / "human_pilot_annotation.csv", index=False)
    duplicates.to_csv(args.output_dir / "pilot_duplicate_key.csv", index=False)

    annotation_lookup = blinded.set_index("annotation_id")
    judge_lines: list[str] = []
    for row in key.itertuples():
        annotation = annotation_lookup.loc[row.annotation_id]
        context = full_preceding_context(
            sentences,
            str(row.example_id),
            int(row.position_index),
        )
        judge_lines.append(judge_item_json(annotation, context))
    (args.output_dir / "judge_items.jsonl").write_text(
        "\n".join(judge_lines) + "\n"
    )
    write_guide(args.output_dir / "SEMANTIC_ANNOTATION_GUIDE.md")
    write_judge_prompt(args.output_dir / "JUDGE_PROMPT.txt")

    pair_rows = matched.drop_duplicates("pair_id")
    primary_pairs = int(pair_rows["match_quality"].eq("primary").sum())
    sensitivity_pairs = int(
        pair_rows["match_quality"].eq("sensitivity_only").sum()
    )
    report = f"""# Semantic Reasoning Classification Preparation

- Detected change-point sentences: {int((matched.item_role == "detected_change_point").sum())}
- Same-state matched comparison sentences: {int((matched.item_role == "matched_non_change_sentence").sum())}
- Matched pairs: {matched.pair_id.nunique()}
- Fixed environment states: {matched.example_id.nunique()}
- Trajectories: {matched.trajectory_id.nunique()}
- Median absolute reasoning-progress difference: {pair_rows.absolute_progress_difference.median():.4f}
- Maximum absolute reasoning-progress difference: {pair_rows.absolute_progress_difference.max():.4f}
- Primary matched pairs within 0.10 reasoning progress: {primary_pairs}
- Sensitivity-only pairs above 0.10 reasoning progress: {sensitivity_pairs}
- Human pilot: {len(pilot)} rows, including {len(duplicates)} hidden duplicates

Controls are from the same fixed environment state, are outside a plus-or-minus
{args.exclusion_window}-sentence window around every detected change point, and
are matched without replacement on reasoning progress and sentence length
using global minimum-cost assignment. The primary analysis uses pairs
within 0.10 reasoning progress; all pairs remain available for sensitivity
analysis. Outcome, action, activation, and failure labels are absent from the
annotation template.

Start with `human_pilot_annotation.csv` and
`SEMANTIC_ANNOTATION_GUIDE.md`. Do not share `annotation_key.csv` or
`pilot_duplicate_key.csv` with annotators.
"""
    (args.output_dir / "PREPARATION_REPORT.md").write_text(report)
    manifest = {
        "seed": args.seed,
        "exclusion_window_sentences": args.exclusion_window,
        "context_sentences_shown_to_human": args.context_sentences,
        "detected_change_points": len(points),
        "matched_pairs": matched["pair_id"].nunique(),
        "full_annotation_rows": len(blinded),
        "pilot_rows": len(pilot),
        "pilot_hidden_duplicates": len(duplicates),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
