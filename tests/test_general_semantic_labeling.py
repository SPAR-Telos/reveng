from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from reveng.experiments.general_semantic_labeling import (
    BROAD_LABEL_MAP,
    build_general_inventory,
    compare_candidates,
    evaluate_broad_candidate,
    prepare_calibration_items,
    select_few_shot_examples,
    select_broad_few_shot_examples,
    stratified_general_sample,
)
from reveng.experiments.semantic_reasoning_classification import (
    ANNOTATION_COLUMNS,
)
from scripts.run_general_semantic_labeler import (
    build_messages,
    extract_json_object,
    gpu_status,
    normalize_payload,
)
from scripts.run_general_semantic_broad_labeler import (
    normalize_broad_payload,
    recover_truncated_broad_payload,
)


def _canonical_and_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical_rows = []
    inventory_rows = []
    for trajectory in range(2):
        environment = f"keepdoor_{trajectory}"
        trace = f"prefix_doorkey_{environment}_step_001"
        for sentence_index, text in enumerate(("Map grid.", "Cell open.", "Go right.")):
            canonical_rows.append(
                {
                    "trace_id": trace,
                    "sentence_id": sentence_index,
                    "kind": "reasoning",
                    "text": text,
                }
            )
            inventory_rows.append(
                {
                    "sentence_id": f"{environment}__step_001__sentence_{sentence_index + 1:03d}",
                    "environment_id": environment,
                    "environment_step": 1,
                    "sentence_number": sentence_index + 1,
                    "target_sentence": text,
                    "semantic_label_available": sentence_index == 0,
                }
            )
    return pd.DataFrame(canonical_rows), pd.DataFrame(inventory_rows)


def test_general_inventory_uses_only_preceding_context() -> None:
    canonical, source = _canonical_and_inventory()
    result = build_general_inventory(source, canonical, context_sentences=2)
    first = result.loc[result["sentence_number"].eq(1)].iloc[0]
    third = result.loc[
        result["environment_id"].eq("keepdoor_0") & result["sentence_number"].eq(3)
    ].iloc[0]
    assert first["context_before"] == ""
    assert "[1] Map grid." in third["context_before"]
    assert "[2] Cell open." in third["context_before"]
    assert "Go right." not in third["context_before"]
    assert len(result) == len(source)


def test_general_sample_is_deterministic_and_covers_trajectories() -> None:
    rows = []
    for trajectory in range(4):
        for sentence in range(20):
            rows.append(
                {
                    "sentence_id": f"t{trajectory}-s{sentence}",
                    "environment_id": f"t{trajectory}",
                    "reasoning_progress": (sentence + 0.5) / 20,
                    "target_sentence_characters": ((sentence * 7 + trajectory * 3) % 20) + 3,
                    "previously_labelled": False,
                }
            )
    inventory = pd.DataFrame(rows)
    first = stratified_general_sample(inventory, sample_size=24, seed=42)
    second = stratified_general_sample(inventory, sample_size=24, seed=42)
    assert first["sentence_id"].tolist() == second["sentence_id"].tolist()
    assert first["environment_id"].nunique() == 4
    assert first["sampling_stratum"].nunique() == 12


def _existing_labels(n_per_label: int = 3) -> pd.DataFrame:
    labels = [
        "state_reconstruction",
        "route_planning",
        "verification",
        "correction",
        "new_inference",
        "restatement",
        "procedural_continuation",
    ]
    rows = []
    for label in labels:
        for index in range(n_per_label):
            rows.append(
                {
                    "sentence_id": f"{label}-{index}",
                    "environment_id": f"trajectory-{index}",
                    "environment_step": 1,
                    "sentence_number": index + 1,
                    "example_id": f"example-{label}-{index}",
                    "reasoning_progress": 0.5,
                    "context_before": "Prior sentence.",
                    "target_sentence": f"Target {label} {index}.",
                    "target_sentence_characters": 20,
                    "primary_label": label,
                    "broad_label": BROAD_LABEL_MAP[label],
                    "confidence": "high",
                    "rationale": "The discourse function is explicit.",
                    "explicitly_revises_prior_reasoning": "no",
                    "evaluates_prior_route_or_claim": "no",
                    "repeats_prior_content": "no",
                    "introduces_new_information_or_plan": "yes",
                }
            )
    return pd.DataFrame(rows)


def test_few_shots_cover_available_labels_before_repeating() -> None:
    labels = _existing_labels()
    selected = select_few_shot_examples(labels, n_examples=12, seed=42)
    counts = selected["primary_label"].value_counts()
    assert set(counts.index) == set(labels["primary_label"])
    assert counts.max() <= 2


def test_broad_few_shots_are_balanced() -> None:
    selected = select_broad_few_shot_examples(_existing_labels(n_per_label=12))
    assert len(selected) == 16
    assert set(selected["broad_label"].value_counts()) == {4}


def test_calibration_items_are_blinded_and_repeat_ids_are_hidden() -> None:
    labels = _existing_labels(n_per_label=12)
    few_shots = select_few_shot_examples(labels, n_examples=12, seed=42)
    inventory = labels[
        [
            "sentence_id",
            "environment_id",
            "environment_step",
            "sentence_number",
            "example_id",
            "reasoning_progress",
            "context_before",
            "target_sentence",
            "target_sentence_characters",
        ]
    ].copy()
    inventory["previously_labelled"] = False
    for index in range(100):
        inventory.loc[len(inventory)] = {
            "sentence_id": f"general-{index}",
            "environment_id": f"trajectory-{index % 5}",
            "environment_step": 2,
            "sentence_number": index + 1,
            "example_id": f"general-example-{index}",
            "reasoning_progress": (index + 0.5) / 100,
            "context_before": "Context.",
            "target_sentence": f"General target {index}.",
            "target_sentence_characters": 10 + index,
            "previously_labelled": False,
        }
    blinded, key = prepare_calibration_items(
        inventory,
        labels,
        few_shots,
        general_sample_size=40,
        reference_sample_size=20,
        duplicate_items=4,
        seed=42,
    )
    assert len(blinded) == 64
    assert list(blinded.columns) == [
        "annotation_id",
        "context_before",
        "target_sentence",
        "target_sentence_characters",
    ]
    assert key["duplicate_of_annotation_id"].fillna("").ne("").sum() == 4
    assert key["annotation_id"].str.startswith("cal_").all()
    assert "reference_primary_label" not in blinded.columns


def test_response_parsing_and_few_shot_messages() -> None:
    payload = {
        "explicitly_revises_prior_reasoning": "no",
        "evaluates_prior_route_or_claim": "yes",
        "repeats_prior_content": "yes",
        "introduces_new_information_or_plan": "no",
        "primary_label": "verification",
        "confidence": "high",
        "rationale": "It evaluates the route.",
    }
    parsed = extract_json_object("Answer:\n```json\n" + __import__("json").dumps(payload) + "\n```")
    assert normalize_payload(parsed)["primary_label"] == "verification"
    example = {"context_before": "Prior.", "target_sentence": "Check route.", **payload}
    messages = build_messages(SimpleNamespace(context_before="", target_sentence="Go."), [example])
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    with pytest.raises(ValueError):
        normalize_payload({**payload, "primary_label": "made_up"})
    assert normalize_broad_payload(
        {
            "broad_label": "route_deliberation",
            "confidence": "medium",
            "rationale": "It checks a proposed route.",
        }
    )["broad_label"] == "route_deliberation"
    recovered = recover_truncated_broad_payload(
        '{"broad_label":"route_deliberation","confidence":"high",'
        '"rationale":"It checks a route and was truncated'
    )
    assert recovered["broad_label"] == "route_deliberation"
    normalized = normalize_broad_payload(
        {
            "broad_label": "state_reconstruction",
            "confidence": "high",
            "rationale": "It directly records a visible cell.",
        }
    )
    assert normalized["broad_label"] == "state_readout"
    assert normalized["label_normalization_applied"] == "true"


def test_gpu_check_never_requires_a_gpu() -> None:
    status = gpu_status(0, 3.0)
    assert status["status"] in {"ready", "blocked_gpu_unavailable", "blocked_gpu_busy"}


def test_candidate_gate_with_identical_synthetic_predictions() -> None:
    primary = [
        "state_reconstruction",
        "route_planning",
        "restatement",
        "procedural_continuation",
    ]
    key_rows = []
    annotation_rows = []
    for index in range(12):
        label = primary[index % 4]
        annotation_id = f"a-{index}"
        source = "matched_reference_stress" if index < 4 else "general_random"
        key_rows.append(
            {
                "annotation_id": annotation_id,
                "sentence_id": annotation_id,
                "calibration_source": source,
                "reference_primary_label": label if index < 4 else "",
                "reference_broad_label": BROAD_LABEL_MAP[label] if index < 4 else "",
                "duplicate_of_annotation_id": "",
            }
        )
        annotation_rows.append(
            {
                "annotation_id": annotation_id,
                "primary_label": label,
                "confidence": "high",
            }
        )
    for index in range(8, 12):
        original = annotation_rows[index]
        annotation_id = f"repeat-{index}"
        key_rows.append(
            {
                "annotation_id": annotation_id,
                "sentence_id": f"repeat-sentence-{index}",
                "calibration_source": "general_random",
                "reference_primary_label": "",
                "reference_broad_label": "",
                "duplicate_of_annotation_id": original["annotation_id"],
            }
        )
        annotation_rows.append(
            {
                "annotation_id": annotation_id,
                "primary_label": original["primary_label"],
                "confidence": "high",
            }
        )
    key = pd.DataFrame(key_rows)
    predictions = pd.DataFrame(annotation_rows)
    summary, disagreements, gate = compare_candidates(
        {"llama_3_2_1b": predictions, "gpt_oss_20b": predictions}, key
    )
    assert len(summary) == 2
    assert len(disagreements) == 0
    assert gate["status"] == "complete"
    assert gate["smaller_model_passes"] is True


def test_candidate_can_be_rejected_without_teacher_output() -> None:
    key = pd.DataFrame(
        [
            {
                "annotation_id": f"a-{index}",
                "sentence_id": f"a-{index}",
                "calibration_source": "matched_reference_stress",
                "reference_primary_label": "verification",
                "reference_broad_label": "route_deliberation",
                "duplicate_of_annotation_id": "",
            }
            for index in range(4)
        ]
    )
    predictions = pd.DataFrame(
        {
            "annotation_id": [f"a-{index}" for index in range(4)],
            "primary_label": ["state_reconstruction"] * 4,
            "confidence": ["high"] * 4,
        }
    )
    _, _, gate = compare_candidates({"llama_3_2_1b": predictions}, key)
    assert gate["status"] == "smaller_model_rejected"
    assert gate["smaller_model_passes"] is False


def test_direct_broad_candidate_gate() -> None:
    broad_labels = [
        "state_readout",
        "route_deliberation",
        "summary_or_restatement",
        "other",
    ]
    fine_for_broad = {
        "state_readout": "state_reconstruction",
        "route_deliberation": "verification",
        "summary_or_restatement": "restatement",
        "other": "procedural_continuation",
    }
    key_rows = []
    broad_rows = []
    teacher_rows = []
    for index in range(12):
        broad = broad_labels[index % 4]
        annotation_id = f"b-{index}"
        key_rows.append(
            {
                "annotation_id": annotation_id,
                "calibration_source": (
                    "matched_reference_stress" if index < 4 else "general_random"
                ),
                "reference_broad_label": broad if index < 4 else "",
                "duplicate_of_annotation_id": "",
            }
        )
        broad_rows.append(
            {"annotation_id": annotation_id, "broad_label": broad, "confidence": "high"}
        )
        teacher_rows.append(
            {
                "annotation_id": annotation_id,
                "primary_label": fine_for_broad[broad],
            }
        )
    for index in range(8, 12):
        broad = broad_labels[index % 4]
        annotation_id = f"b-repeat-{index}"
        key_rows.append(
            {
                "annotation_id": annotation_id,
                "calibration_source": "general_random",
                "reference_broad_label": "",
                "duplicate_of_annotation_id": f"b-{index}",
            }
        )
        broad_rows.append(
            {"annotation_id": annotation_id, "broad_label": broad, "confidence": "high"}
        )
        teacher_rows.append(
            {
                "annotation_id": annotation_id,
                "primary_label": fine_for_broad[broad],
            }
        )
    summary, disagreements, gate = evaluate_broad_candidate(
        pd.DataFrame(broad_rows),
        pd.DataFrame(key_rows),
        teacher_annotations=pd.DataFrame(teacher_rows),
    )
    assert summary["reference_broad_accuracy"] == 1
    assert len(disagreements) == 0
    assert gate["broad_candidate_passes"] is True
