import pandas as pd
import pytest

from reveng.experiments.semantic_reasoning_classification import (
    build_blinded_annotation_rows,
    build_human_pilot,
    match_change_points_to_controls,
    validate_annotation_rows,
)


def synthetic_sentences() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trace_id": "state",
                "sentence_id": index,
                "kind": "reasoning",
                "text": f"Sentence {index}.",
                "char_start": index * 12,
                "char_end": (index + 1) * 12,
            }
            for index in range(12)
        ]
    )


def test_matching_excludes_change_point_windows_and_blinds_metadata() -> None:
    positions = pd.DataFrame(
        [
            {
                "example_id": "state",
                "trajectory_id": "trajectory",
                "step_index": 1,
                "position_index": index,
                "reasoning_progress": index / 12,
            }
            for index in range(1, 13)
        ]
    )
    points = pd.DataFrame(
        [
            {
                "example_id": "state",
                "trajectory_id": "trajectory",
                "step_index": 1,
                "position_index": 5,
                "reasoning_progress": 5 / 12,
                "posterior_change_probability_at_position": 0.9,
                "matched_role": "failure",
                "primary_step_failure_mode": "wall_hit",
            }
        ]
    )
    matched = match_change_points_to_controls(
        positions,
        points,
        synthetic_sentences(),
        exclusion_window=2,
    )
    control = matched[matched["item_role"] == "matched_non_change_sentence"].iloc[0]
    assert abs(int(control["position_index"]) - 5) > 2

    blinded, key = build_blinded_annotation_rows(
        matched, synthetic_sentences(), seed=42
    )
    assert "item_role" not in blinded
    assert "matched_role" not in blinded
    assert "item_role" in key
    assert set(blinded["annotation_id"]) == set(key["annotation_id"])


def test_annotation_validation_rejects_invalid_labels() -> None:
    row = {
        "annotation_id": "item",
        "explicitly_revises_prior_reasoning": "no",
        "evaluates_prior_route_or_claim": "no",
        "repeats_prior_content": "no",
        "introduces_new_information_or_plan": "yes",
        "primary_label": "not_a_label",
        "confidence": "high",
        "rationale": "test",
    }
    with pytest.raises(ValueError, match="Invalid primary labels"):
        validate_annotation_rows(pd.DataFrame([row]))


def test_pilot_duplicate_identifiers_do_not_reveal_duplicates() -> None:
    rows = []
    key_rows = []
    for pair_index in range(4):
        pair_id = f"pair_{pair_index}"
        for role in ("detected_change_point", "matched_non_change_sentence"):
            annotation_id = f"sem_{pair_index}_{role}"
            rows.append(
                {
                    "annotation_id": annotation_id,
                    "context_before": "Context.",
                    "target_sentence": f"Sentence {pair_index}.",
                    "target_sentence_characters": 11,
                }
            )
            key_rows.append(
                {
                    "annotation_id": annotation_id,
                    "pair_id": pair_id,
                    "matched_role": "failure",
                    "change_point_progress": 0.5,
                    "posterior_change_probability": 0.9,
                }
            )
    pilot, duplicate_key = build_human_pilot(
        pd.DataFrame(rows),
        pd.DataFrame(key_rows),
        pairs=4,
        duplicate_items=2,
        seed=42,
    )
    duplicate_ids = set(duplicate_key["duplicate_annotation_id"])
    assert duplicate_ids
    assert duplicate_ids.issubset(set(pilot["annotation_id"]))
    assert all(value.startswith("sem_") for value in duplicate_ids)
    assert all("dup" not in value for value in duplicate_ids)
