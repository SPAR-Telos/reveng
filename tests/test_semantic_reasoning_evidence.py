import numpy as np
import pandas as pd

from scripts.build_semantic_reasoning_evidence import (
    BROAD_LABEL_MAP,
    BROAD_LABELS,
    SEMANTIC_ONLY_COLUMNS,
    SEMANTIC_OUTCOMES,
    add_readable_sentence_identifiers,
    benjamini_hochberg,
    build_all_sentence_inventory,
    cohen_kappa,
    omnibus_paired_permutation,
    semantic_only_table,
    semantic_outcome_permutation_tests,
)
from reveng.experiments.semantic_reasoning_classification import SEMANTIC_LABELS


def test_broad_taxonomy_covers_every_fine_label() -> None:
    assert set(BROAD_LABEL_MAP) == set(SEMANTIC_LABELS)
    assert set(BROAD_LABEL_MAP.values()) == set(BROAD_LABELS)


def test_compact_semantic_table_excludes_experiment_columns() -> None:
    row = {column: "value" for column in SEMANTIC_ONLY_COLUMNS}
    row["trajectory_id"] = "model_rooms2_doorkey_keepdoor_7"
    row["step_index"] = 2
    row["position_index"] = 11
    row["pair_id"] = "pair_1"
    row["posterior_change_probability"] = 0.9
    row["action_label"] = "UP"
    compact = semantic_only_table(pd.DataFrame([row]))

    assert list(compact.columns) == list(SEMANTIC_ONLY_COLUMNS)
    assert "pair_id" not in compact
    assert "posterior_change_probability" not in compact
    assert "action_label" not in compact
    assert compact.iloc[0]["sentence_id"] == (
        "keepdoor_7__step_002__sentence_011"
    )


def test_readable_sentence_identifier_components_are_explicit() -> None:
    rows = pd.DataFrame(
        {
            "trajectory_id": ["model_rooms2_doorkey_keepdoor_12"],
            "step_index": [3],
            "position_index": [9],
        }
    )
    result = add_readable_sentence_identifiers(rows)
    assert result.iloc[0]["environment_id"] == "keepdoor_12"
    assert result.iloc[0]["environment_step"] == 3
    assert result.iloc[0]["sentence_number"] == 9
    assert result.iloc[0]["sentence_id"] == (
        "keepdoor_12__step_003__sentence_009"
    )


def test_all_sentence_inventory_marks_only_available_labels() -> None:
    positions = pd.DataFrame(
        {
            "example_id": ["trace", "trace", "trace"],
            "trajectory_id": [
                "model_rooms2_doorkey_keepdoor_4",
                "model_rooms2_doorkey_keepdoor_4",
                "model_rooms2_doorkey_keepdoor_4",
            ],
            "step_index": [1, 1, 1],
            "position_index": [0, 1, 2],
        }
    )
    sentences = pd.DataFrame(
        {
            "trace_id": ["trace", "trace"],
            "sentence_id": [0, 1],
            "kind": ["reasoning", "reasoning"],
            "text": ["First.", "Second."],
        }
    )
    labels = pd.DataFrame(
        {
            "sentence_id": ["keepdoor_4__step_001__sentence_002"],
            "primary_label": ["verification"],
            "broad_label": ["route_deliberation"],
            "confidence": ["high"],
        }
    )
    result = build_all_sentence_inventory(positions, sentences, labels)
    assert len(result) == 2
    assert result["semantic_label_available"].tolist() == [False, True]
    assert pd.isna(result.iloc[0]["primary_label"])


def test_cohen_kappa_is_one_for_identical_labels() -> None:
    labels = pd.Series(["state_readout", "route_deliberation", "other"])
    assert cohen_kappa(labels, labels) == 1.0


def test_benjamini_hochberg_is_monotone_in_ranked_order() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.8])
    assert all(0 <= value <= 1 for value in adjusted)
    assert adjusted[0] <= adjusted[2] <= adjusted[1] <= adjusted[3]


def test_omnibus_pair_permutation_detects_complete_role_separation() -> None:
    rows = []
    for index in range(40):
        rows.extend(
            [
                {
                    "pair_id": f"pair_{index}",
                    "item_role": "detected_change_point",
                    "broad_label": "route_deliberation",
                },
                {
                    "pair_id": f"pair_{index}",
                    "item_role": "matched_non_change_sentence",
                    "broad_label": "state_readout",
                },
            ]
        )
    result = omnibus_paired_permutation(
        pd.DataFrame(rows),
        repeats=1000,
        seed=42,
    )
    assert result["total_variation_distance"] == 1.0
    assert result["paired_permutation_p"] < 0.01


def test_semantic_outcome_test_handles_nonconsecutive_indices() -> None:
    rows = pd.DataFrame(
        {
            "item_role": ["detected_change_point"] * 6,
            "match_quality": ["primary"] * 6,
            "trajectory_id": ["a", "a", "a", "b", "b", "b"],
            "broad_label": [
                "state_readout",
                "route_deliberation",
                "summary_or_restatement",
                "state_readout",
                "route_deliberation",
                "summary_or_restatement",
            ],
            **{
                outcome: [0.0, 1.0, np.nan, 0.0, 1.0, 0.5]
                for outcome in SEMANTIC_OUTCOMES
            },
        }
    )
    result = semantic_outcome_permutation_tests(rows, repeats=20, seed=2)
    assert set(result["outcome"]) == set(SEMANTIC_OUTCOMES)
    assert result["within_trajectory_permutation_p"].between(0, 1).all()
