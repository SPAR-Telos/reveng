from __future__ import annotations

import json

import pandas as pd
import pytest

from reveng.experiments.reasoningflow_transfer import (
    REASONINGFLOW_LABELS,
    analyze_pilot,
    extract_json_object,
    gpu_status,
    normalize_nodes,
    prepare_pilot_items,
    select_review_sentences,
    structure_score,
    write_example_splits,
)
from scripts.run_reasoningflow_transfer_pilot import FEW_SHOTS, SYSTEM_PROMPT


def _inventory(
    n_trajectories: int = 10, sentences_per_trajectory: int = 30
) -> pd.DataFrame:
    rows = []
    for trajectory in range(n_trajectories):
        for sentence in range(sentences_per_trajectory):
            target = (
                f"Cell {sentence} is open, so inspect it and then consider the route."
                if sentence % 3 == 0
                else f"Inspect cell {sentence}."
            )
            rows.append(
                {
                    "sentence_id": f"t{trajectory}__sentence_{sentence:03d}",
                    "environment_id": f"t{trajectory}",
                    "environment_step": 1,
                    "sentence_number": sentence + 1,
                    "context_before": "A preceding sentence.",
                    "target_sentence": target,
                    "target_sentence_characters": len(target),
                    "previously_labelled": sentence == 1,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_pilot_is_deterministic_balanced_and_blinded() -> None:
    inventory = _inventory()
    first = prepare_pilot_items(inventory, sample_size=40, seed=42)
    second = prepare_pilot_items(inventory, sample_size=40, seed=42)
    assert first["sentence_id"].tolist() == second["sentence_id"].tolist()
    assert first["sample_stratum"].value_counts().to_dict() == {
        "random": 20,
        "structure_enriched": 20,
    }
    assert first["sentence_id"].nunique() == 40
    assert first["environment_id"].nunique() == 10
    assert (
        not first["sentence_id"]
        .isin(inventory.loc[inventory["previously_labelled"], "sentence_id"])
        .any()
    )
    assert not {
        "change_point",
        "action_probability",
        "optimality",
        "commitment",
    } & set(first.columns)


def test_structure_score_uses_surface_complexity() -> None:
    assert structure_score(
        "Inspect it, then turn left because the door is open."
    ) > structure_score("Move left.")


def test_json_and_node_validation_require_an_exact_partition() -> None:
    target = "Suppose X; then Y."
    payload = {
        "nodes": [
            {
                "start_character": 0,
                "end_character": 11,
                "node_text": "Suppose X; ",
                "reasoningflow_label": "assumption",
                "confidence": "high",
                "short_rationale": "It introduces a hypothetical.",
            },
            {
                "start_character": 11,
                "end_character": len(target),
                "node_text": "then Y.",
                "reasoningflow_label": "reasoning",
                "confidence": "high",
                "short_rationale": "It derives a consequence.",
            },
        ]
    }
    parsed = extract_json_object("Result:\n```json\n" + json.dumps(payload) + "\n```")
    nodes = normalize_nodes(parsed, target_sentence=target)
    assert [node["reasoningflow_label"] for node in nodes] == [
        "assumption",
        "reasoning",
    ]
    with pytest.raises(ValueError, match="contiguous"):
        normalize_nodes(
            {
                "nodes": [
                    {**payload["nodes"][0]},
                    {**payload["nodes"][1], "start_character": 12},
                ]
            },
            target_sentence=target,
        )
    with pytest.raises(ValueError, match="invalid label"):
        normalize_nodes(
            {
                "nodes": [
                    {
                        "start_character": 0,
                        "end_character": len(target),
                        "node_text": target,
                        "reasoningflow_label": "action_commitment",
                        "confidence": "high",
                        "short_rationale": "Not part of ReasoningFlow.",
                    }
                ]
            },
            target_sentence=target,
        )


def test_built_in_examples_have_valid_offsets_and_labels() -> None:
    for example in FEW_SHOTS:
        nodes = normalize_nodes(
            {"nodes": example["nodes"]}, target_sentence=example["target_sentence"]
        )
        assert all(
            node["reasoningflow_label"] in REASONINGFLOW_LABELS for node in nodes
        )
    assert "explicit-action-commitment timing" in SYSTEM_PROMPT
    assert "action-commitment label" in SYSTEM_PROMPT


def _pilot_and_nodes(n_sentences: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    items = _inventory(n_trajectories=5, sentences_per_trajectory=12).head(n_sentences)
    items = items.rename(columns={"sentence_id": "source_sentence_id"})
    items["pilot_id"] = [f"p{index:03d}" for index in range(len(items))]
    items["sentence_id"] = items["source_sentence_id"]
    items["sample_stratum"] = [
        "random" if index % 2 else "structure_enriched" for index in range(len(items))
    ]
    items["structure_score"] = items["target_sentence"].map(structure_score)
    node_rows = []
    for index, row in enumerate(items.itertuples(index=False)):
        split = (
            len(row.target_sentence) // 2 if index < 25 else len(row.target_sentence)
        )
        spans = (
            [(0, split), (split, len(row.target_sentence))]
            if index < 25
            else [(0, split)]
        )
        for node_number, (start, end) in enumerate(spans, start=1):
            node_rows.append(
                {
                    "pilot_id": row.pilot_id,
                    "sentence_id": row.sentence_id,
                    "node_number": node_number,
                    "start_character": start,
                    "end_character": end,
                    "node_text": row.target_sentence[start:end],
                    "reasoningflow_label": (
                        "reasoning" if node_number == 1 else "planning"
                    ),
                    "confidence": "low" if index < 5 else "high",
                    "short_rationale": "Synthetic test node.",
                }
            )
    return items, pd.DataFrame(node_rows)


def test_review_selection_has_requested_priority_mix() -> None:
    items, nodes = _pilot_and_nodes()
    review = select_review_sentences(
        items,
        nodes,
        review_size=20,
        priority_size=10,
        seed=42,
    )
    assert len(review) == 20
    assert review["pilot_id"].nunique() == 20
    assert review["priority_review"].sum() >= 10
    assert set(review["boundary_reasonable"]) == {""}
    assert set(review["labels_reasonable"]) == {""}


def test_gpu_check_does_not_require_a_gpu() -> None:
    assert gpu_status(0, 16.0)["status"] in {
        "ready",
        "blocked_gpu_busy",
        "blocked_gpu_unavailable",
    }


def test_analysis_reports_review_rates_and_schema_gaps(tmp_path) -> None:
    items, nodes = _pilot_and_nodes()
    review = select_review_sentences(items, nodes, review_size=20, priority_size=10)
    review.loc[:9, "boundary_reasonable"] = "yes"
    review.loc[10:19, "boundary_reasonable"] = "no"
    review.loc[:14, "labels_reasonable"] = "yes"
    review.loc[15:19, "labels_reasonable"] = "no"
    review.loc[0, "missing_doorkey_function"] = "direct state observation"
    result = analyze_pilot(items, nodes, review)
    assert result["pilot_sentences"] == 50
    assert result["nodes"] == 75
    assert result["reviewed_metrics"]["boundary_reasonable"]["rate"] == 0.5
    assert result["reviewed_metrics"]["labels_reasonable"]["rate"] == 0.75
    assert len(result["schema_gaps"]) == 1
    output = tmp_path / "examples.md"
    write_example_splits(output, items, nodes, maximum_examples=3)
    assert output.read_text().count("## ") == 3
