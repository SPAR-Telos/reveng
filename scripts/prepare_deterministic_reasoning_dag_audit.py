#!/usr/bin/env python3
"""Prepare a small matched-trace audit for typed reasoning-DAG edges."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path("outputs/hypothesis_tests")
POSITIONS = ROOT / "reasoning_regime_coarse_graining_v1/regime_positions.csv"
CHANGE_POINTS = ROOT / "action_distribution_cpd_beast_v1/detected_change_points.csv"
ANNOTATIONS = ROOT / (
    "semantic_reasoning_classification_v1/general_corpus_v1/"
    "annotations_gpt_oss_20b_multilabel_v3_full.csv"
)
DEFAULT_OUTPUT = ROOT / "deterministic_reasoning_dag_audit_v1"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_review = args.output_dir / "edge_review.csv"
    if existing_review.exists() and not args.force:
        existing = pd.read_csv(existing_review, keep_default_na=False)
        review_columns = [
            "reviewer_a_edges_json", "reviewer_b_edges_json",
            "adjudicated_edges_json", "review_notes",
        ]
        if any(
            existing[column].astype(str).str.strip().ne("").any()
            for column in review_columns if column in existing
        ):
            raise FileExistsError(
                f"{existing_review} contains review data; use --force to replace it"
            )
    positions = pd.read_csv(POSITIONS)
    annotations = pd.read_csv(ANNOTATIONS)
    change_points = pd.read_csv(CHANGE_POINTS)

    sizes = (
        positions.groupby(["matched_pair_id", "matched_role", "example_id"])
        .size().reset_index(name="n_positions")
    )
    eligible = sizes.groupby("matched_pair_id").agg(
        n_roles=("matched_role", "nunique"), total_positions=("n_positions", "sum")
    )
    selected_pairs = eligible[eligible.n_roles.eq(2)].nsmallest(
        args.n_pairs, "total_positions"
    ).index.tolist()
    selected = sizes[sizes.matched_pair_id.isin(selected_pairs)].copy()

    node_columns = [
        "example_id", "position_index", "matched_pair_id", "matched_role",
        "reasoning_progress", "regime", "is_optimal", "is_stable",
        "prob_up", "prob_down", "prob_left", "prob_right",
        "computed_action_entropy_bits",
    ]
    nodes = positions[node_columns].merge(
        annotations[[
            "example_id", "sentence_number", "target_sentence", "semantic_labels",
            "primary_label", "explicitly_revises_prior_reasoning",
            "evaluates_prior_route_or_claim", "repeats_prior_content",
            "introduces_new_information_or_plan",
        ]],
        left_on=["example_id", "position_index"],
        right_on=["example_id", "sentence_number"], how="inner", validate="one_to_one",
    )
    nodes = nodes[nodes.matched_pair_id.isin(selected_pairs)].copy()
    cp_keys = set(zip(change_points.example_id, change_points.position_index, strict=True))
    nodes["beast_change_point"] = [
        (example_id, position) in cp_keys
        for example_id, position in zip(nodes.example_id, nodes.position_index, strict=True)
    ]
    nodes["node_id"] = [
        f"{example_id}__sentence_{int(position):03d}"
        for example_id, position in zip(nodes.example_id, nodes.position_index, strict=True)
    ]
    nodes = nodes.sort_values(["matched_pair_id", "matched_role", "position_index"])

    review = nodes[[
        "matched_pair_id", "matched_role", "example_id", "node_id",
        "position_index", "target_sentence", "semantic_labels",
    ]].copy()
    review["reviewer_a_edges_json"] = ""
    review["reviewer_b_edges_json"] = ""
    review["adjudicated_edges_json"] = ""
    review["review_notes"] = ""

    selected.to_csv(args.output_dir / "selected_traces.csv", index=False)
    nodes.to_csv(args.output_dir / "nodes.csv", index=False)
    review.to_csv(args.output_dir / "edge_review.csv", index=False)
    (args.output_dir / "ANNOTATION_GUIDE.md").write_text(
        """# Typed reasoning-DAG edge audit

For each destination node, identify the **minimal complete set** of earlier nodes needed to interpret its contribution. Every edge must point left-to-right. Enter a JSON list such as:

```json
[{"source_node_id":"...__sentence_004","edge_label":"infer"}]
```

Allowed labels:

- `infer`: premise or state information supports a derived claim;
- `restate`: destination repeats/paraphrases the source;
- `support` / `attack`: destination validates or contradicts source content;
- `proceed`: source motivates the next plan;
- `execute`: destination carries out a source plan;
- `decompose`: destination is a subplan of source;
- `verify`: destination initiates checking of source;
- `backtrack`: destination replaces/abandons a source plan;
- `positive` / `negative` / `uncertain`: destination evaluates source;
- `state_update`: destination revises a DoorKey world-state representation;
- `route_update`: destination revises a route or subgoal from source;
- `action_commit`: destination commits to an action derived from source.

Do not use chronological proximity alone. Use `[]` when a reviewed node introduces an independent fact/plan and therefore has no predecessors. A blank cell means **not yet reviewed**. Reviewers A and B annotate independently before adjudication. The BEAST flag and behavioral fields are intentionally absent from `edge_review.csv`; they are joined only after edge reliability is measured.
"""
    )
    (args.output_dir / "RUNBOOK.md").write_text(
        """# Runbook

1. Annotate `edge_review.csv` independently in the A and B columns using `nodes.csv` as the complete trace.
2. Compare edge detection, source selection, and label agreement before adjudication.
3. Do not inspect behavioral outcomes while annotating edges.
4. Populate `adjudicated_edges_json` only after independent review is complete.
"""
    )


if __name__ == "__main__":
    main()
