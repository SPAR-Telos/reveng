"""Preparation and validation utilities for semantic reasoning labels."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


SEMANTIC_LABELS = (
    "state_reconstruction",
    "route_planning",
    "verification",
    "correction",
    "new_inference",
    "consolidation",
    "restatement",
    "procedural_continuation",
    "unclear",
)

CONFIDENCE_LABELS = ("high", "medium", "low")
BOOLEAN_LABELS = ("yes", "no", "unsure")
ANNOTATION_COLUMNS = (
    "explicitly_revises_prior_reasoning",
    "evaluates_prior_route_or_claim",
    "repeats_prior_content",
    "introduces_new_information_or_plan",
    "primary_label",
    "confidence",
    "rationale",
)


def stable_id(*values: object, length: int = 16) -> str:
    serialized = "\x1f".join(str(value) for value in values)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]


def build_sentence_lookup(sentences: pd.DataFrame) -> pd.DataFrame:
    required = {"trace_id", "sentence_id", "kind", "text", "char_start", "char_end"}
    missing = sorted(required - set(sentences.columns))
    if missing:
        raise ValueError(f"Sentence rows missing required columns: {missing}")
    frame = sentences[sentences["kind"].eq("reasoning")].copy()
    frame["position_index"] = frame["sentence_id"].astype(int) + 1
    if frame.duplicated(["trace_id", "position_index"]).any():
        raise ValueError("Sentence table contains duplicate trace positions")
    return frame


def add_sentence_context(
    items: pd.DataFrame,
    sentences: pd.DataFrame,
    *,
    context_sentences: int = 8,
    context_character_limit: int = 1800,
) -> pd.DataFrame:
    """Attach target text and bounded preceding context without future text."""
    lookup = build_sentence_lookup(sentences)
    traces = {
        str(trace_id): group.sort_values("position_index")
        for trace_id, group in lookup.groupby("trace_id")
    }
    output: list[dict[str, Any]] = []
    for row in items.to_dict("records"):
        example_id = str(row["example_id"])
        position = int(row["position_index"])
        if position <= 0:
            raise ValueError(f"{example_id}: position zero has no reasoning sentence")
        trace = traces.get(example_id)
        if trace is None:
            raise ValueError(f"{example_id}: no canonical reasoning sentences")
        target = trace[trace["position_index"].astype(int).eq(position)]
        if len(target) != 1:
            raise ValueError(
                f"{example_id}: expected one sentence at position {position}"
            )
        preceding = trace[trace["position_index"].astype(int).lt(position)].tail(
            context_sentences
        )
        context = "\n".join(
            f"[{int(source.position_index)}] {source.text}"
            for source in preceding.itertuples()
        )
        if len(context) > context_character_limit:
            context = context[-context_character_limit:]
            first_newline = context.find("\n")
            if first_newline >= 0:
                context = "[Earlier context omitted]\n" + context[first_newline + 1 :]
        target_row = target.iloc[0]
        output.append(
            {
                **row,
                "context_before": context,
                "target_sentence": str(target_row["text"]),
                "target_sentence_characters": len(str(target_row["text"])),
                "target_char_start": int(target_row["char_start"]),
                "target_char_end": int(target_row["char_end"]),
            }
        )
    return pd.DataFrame(output)


def _match_cost(
    event_progress: float,
    event_length: int,
    control_progress: float,
    control_length: int,
) -> float:
    progress_cost = abs(event_progress - control_progress) / 0.10
    length_cost = abs(math.log((control_length + 1) / (event_length + 1)))
    return float(progress_cost + length_cost)


def match_change_points_to_controls(
    positions: pd.DataFrame,
    change_points: pd.DataFrame,
    sentences: pd.DataFrame,
    *,
    exclusion_window: int = 3,
) -> pd.DataFrame:
    """Match each detected point to a same-state non-change sentence."""
    sentence_lookup = build_sentence_lookup(sentences)[
        ["trace_id", "position_index", "text"]
    ].copy()
    sentence_lookup["sentence_characters"] = sentence_lookup["text"].astype(str).str.len()
    position_frame = positions.merge(
        sentence_lookup,
        left_on=["example_id", "position_index"],
        right_on=["trace_id", "position_index"],
        how="inner",
        validate="one_to_one",
    )
    if position_frame["example_id"].nunique() != positions["example_id"].nunique():
        raise ValueError("Not every state has canonical sentence text")

    output: list[dict[str, Any]] = []
    for example_id, points in change_points.groupby("example_id", sort=True):
        state = position_frame[position_frame["example_id"].eq(example_id)].copy()
        detected_positions = set(
            change_points.loc[
                change_points["example_id"].eq(example_id), "position_index"
            ].astype(int)
        )
        excluded = {
            candidate
            for position in detected_positions
            for candidate in range(
                position - exclusion_window, position + exclusion_window + 1
            )
        }
        controls = state[
            state["position_index"].astype(int).gt(0)
            & ~state["position_index"].astype(int).isin(excluded)
        ].copy()
        if len(controls) < len(points):
            raise ValueError(
                f"{example_id}: only {len(controls)} controls for {len(points)} points"
            )

        ordered_points = points.sort_values("position_index").reset_index(drop=True)
        controls = controls.sort_values("position_index").reset_index(drop=True)
        event_sentences = [
            state[
                state["position_index"].astype(int).eq(int(point.position_index))
            ].iloc[0]
            for point in ordered_points.itertuples()
        ]
        cost_matrix = np.asarray(
            [
                [
                    _match_cost(
                        float(point.reasoning_progress),
                        int(event_sentence["sentence_characters"]),
                        float(control.reasoning_progress),
                        int(control.sentence_characters),
                    )
                    for control in controls.itertuples()
                ]
                for point, event_sentence in zip(
                    ordered_points.itertuples(), event_sentences, strict=True
                )
            ]
        )
        point_indices, control_indices = linear_sum_assignment(cost_matrix)
        assignments = dict(zip(point_indices, control_indices, strict=True))
        for point_index, point in enumerate(ordered_points.itertuples()):
            event_sentence = event_sentences[point_index]
            selected_index = assignments[point_index]
            control = controls.iloc[selected_index]
            cost = float(cost_matrix[point_index, selected_index])
            pair_id = stable_id(example_id, point.position_index, selected_index)
            common = {
                "pair_id": pair_id,
                "example_id": example_id,
                "trajectory_id": point.trajectory_id,
                "step_index": int(point.step_index),
                "matched_role": point.matched_role,
                "primary_step_failure_mode": point.primary_step_failure_mode,
                "match_cost": cost,
                "change_point_position": int(point.position_index),
                "control_position": int(control["position_index"]),
                "change_point_progress": float(point.reasoning_progress),
                "control_progress": float(control["reasoning_progress"]),
                "absolute_progress_difference": abs(
                    float(point.reasoning_progress)
                    - float(control["reasoning_progress"])
                ),
                "change_point_sentence_characters": int(
                    event_sentence["sentence_characters"]
                ),
                "control_sentence_characters": int(
                    control["sentence_characters"]
                ),
                "posterior_change_probability": float(
                    point.posterior_change_probability_at_position
                ),
            }
            common["match_quality"] = (
                "primary" if common["absolute_progress_difference"] <= 0.10
                else "sensitivity_only"
            )
            output.extend(
                [
                    {
                        **common,
                        "item_role": "detected_change_point",
                        "position_index": int(point.position_index),
                    },
                    {
                        **common,
                        "item_role": "matched_non_change_sentence",
                        "position_index": int(control["position_index"]),
                    },
                ]
            )
    result = pd.DataFrame(output)
    if result.duplicated(["example_id", "item_role", "position_index"]).any():
        raise ValueError("Matching reused a sentence within the same item role")
    return result.sort_values(["pair_id", "item_role"]).reset_index(drop=True)


def build_blinded_annotation_rows(
    matched_items: pd.DataFrame,
    sentences: pd.DataFrame,
    *,
    seed: int = 42,
    context_sentences: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contextualized = add_sentence_context(
        matched_items,
        sentences,
        context_sentences=context_sentences,
    )
    contextualized["annotation_id"] = [
        "sem_" + stable_id(row.pair_id, row.item_role)
        for row in contextualized.itertuples()
    ]
    shuffled = contextualized.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    blinded = shuffled[
        [
            "annotation_id",
            "context_before",
            "target_sentence",
            "target_sentence_characters",
        ]
    ].copy()
    for column in ANNOTATION_COLUMNS:
        blinded[column] = ""
    key = contextualized.drop(
        columns=["context_before", "target_sentence"], errors="ignore"
    ).copy()
    return blinded, key


def build_human_pilot(
    blinded: pd.DataFrame,
    key: pd.DataFrame,
    *,
    pairs: int = 20,
    duplicate_items: int = 4,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_metadata = (
        key[
            [
                "pair_id",
                "matched_role",
                "change_point_progress",
                "posterior_change_probability",
            ]
        ]
        .drop_duplicates("pair_id")
        .copy()
    )
    pair_metadata["progress_quartile"] = pd.cut(
        pair_metadata["change_point_progress"],
        bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=False,
    )
    selected: list[str] = []
    per_stratum = max(1, pairs // 8)
    for (_, _), group in pair_metadata.groupby(
        ["matched_role", "progress_quartile"], dropna=False
    ):
        take = min(per_stratum, len(group))
        selected.extend(
            group.sample(n=take, random_state=seed)["pair_id"].astype(str).tolist()
        )
    remaining = pair_metadata[~pair_metadata["pair_id"].isin(selected)]
    if len(selected) < pairs:
        selected.extend(
            remaining.sample(
                n=min(pairs - len(selected), len(remaining)),
                random_state=seed + 1,
            )["pair_id"]
            .astype(str)
            .tolist()
        )
    selected = selected[:pairs]

    pilot_key = key[key["pair_id"].astype(str).isin(selected)].copy()
    pilot = blinded[
        blinded["annotation_id"].isin(pilot_key["annotation_id"])
    ].copy()
    duplicate_source = pilot.sample(
        n=min(duplicate_items, len(pilot)),
        random_state=seed + 2,
    )
    duplicate_rows = duplicate_source.copy()
    duplicate_rows["annotation_id"] = [
        "sem_" + stable_id("reliability_item", value, seed)
        for value in duplicate_source["annotation_id"]
    ]
    duplicate_key = duplicate_source[["annotation_id"]].copy()
    duplicate_key["duplicate_annotation_id"] = duplicate_rows["annotation_id"].values
    duplicate_key = duplicate_key.rename(
        columns={"annotation_id": "duplicate_of_annotation_id"}
    )
    pilot = pd.concat([pilot, duplicate_rows], ignore_index=True).sample(
        frac=1.0, random_state=seed + 3
    )
    pilot = pilot.reset_index(drop=True)
    return pilot, duplicate_key


def validate_annotation_rows(
    annotations: pd.DataFrame,
    *,
    allow_partial: bool = False,
) -> list[str]:
    required = {"annotation_id", *ANNOTATION_COLUMNS}
    missing = sorted(required - set(annotations.columns))
    if missing:
        raise ValueError(f"Annotation file missing columns: {missing}")
    if annotations["annotation_id"].duplicated().any():
        raise ValueError("Annotation IDs must be unique")

    normalized = annotations.copy()
    for column in ANNOTATION_COLUMNS:
        normalized[column] = normalized[column].fillna("").astype(str).str.strip()
    labeled = normalized["primary_label"].ne("")
    if not allow_partial and not labeled.all():
        raise ValueError(f"{int((~labeled).sum())} rows do not have a primary label")
    invalid_labels = sorted(
        set(normalized.loc[labeled, "primary_label"]) - set(SEMANTIC_LABELS)
    )
    if invalid_labels:
        raise ValueError(f"Invalid primary labels: {invalid_labels}")
    invalid_confidence = sorted(
        set(normalized.loc[labeled, "confidence"]) - set(CONFIDENCE_LABELS)
    )
    if invalid_confidence:
        raise ValueError(f"Invalid confidence labels: {invalid_confidence}")
    for column in ANNOTATION_COLUMNS[:4]:
        invalid = sorted(
            set(normalized.loc[labeled, column]) - set(BOOLEAN_LABELS)
        )
        if invalid:
            raise ValueError(f"Invalid values for {column}: {invalid}")
    return [
        f"rows={len(normalized)}",
        f"labeled_rows={int(labeled.sum())}",
        f"unlabeled_rows={int((~labeled).sum())}",
    ]


def judge_item_json(row: pd.Series, full_context: str) -> str:
    annotation_id = row.get("annotation_id", row.name)
    payload = {
        "annotation_id": annotation_id,
        "judge_input": {
            "preceding_reasoning": full_context,
            "target_sentence": row["target_sentence"],
        },
        "required_output": {
            "explicitly_revises_prior_reasoning": "yes|no|unsure",
            "evaluates_prior_route_or_claim": "yes|no|unsure",
            "repeats_prior_content": "yes|no|unsure",
            "introduces_new_information_or_plan": "yes|no|unsure",
            "primary_label": "|".join(SEMANTIC_LABELS),
            "confidence": "high|medium|low",
            "rationale": "one short sentence",
        },
    }
    return json.dumps(payload, ensure_ascii=True)
