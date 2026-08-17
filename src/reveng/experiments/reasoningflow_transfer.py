"""Preparation and analysis helpers for a ReasoningFlow transfer pilot."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


REASONINGFLOW_LABELS = (
    "planning",
    "fact",
    "reasoning",
    "restatement",
    "assumption",
    "example",
    "reflection",
    "conclusion",
)
CONFIDENCE_LABELS = ("high", "medium", "low")

NODE_COLUMNS = (
    "pilot_id",
    "sentence_id",
    "node_number",
    "start_character",
    "end_character",
    "node_text",
    "reasoningflow_label",
    "confidence",
    "short_rationale",
)

STRUCTURE_CUE_RE = re.compile(
    r"\b(?:and|but|so|then|because|however|actually|instead|wait|"
    r"therefore|although|while|whereas|thus)\b",
    flags=re.IGNORECASE,
)


def gpu_status(gpu_index: int, minimum_free_gib: float) -> dict[str, Any]:
    """Report whether a CUDA device has enough free memory for local inference."""
    import torch

    if not torch.cuda.is_available():
        return {
            "status": "blocked_gpu_unavailable",
            "gpu_index": gpu_index,
            "minimum_free_gib": minimum_free_gib,
            "message": "CUDA is unavailable; no CPU fallback is permitted.",
        }
    try:
        with torch.cuda.device(gpu_index):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            name = torch.cuda.get_device_name(gpu_index)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "blocked_gpu_unavailable",
            "gpu_index": gpu_index,
            "minimum_free_gib": minimum_free_gib,
            "message": f"{type(exc).__name__}: {exc}",
        }
    free_gib = free_bytes / 2**30
    status = "ready" if free_gib >= minimum_free_gib else "blocked_gpu_busy"
    result: dict[str, Any] = {
        "status": status,
        "gpu_index": gpu_index,
        "gpu_name": name,
        "free_gib": free_gib,
        "total_gib": total_bytes / 2**30,
        "minimum_free_gib": minimum_free_gib,
    }
    if status == "blocked_gpu_busy":
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        result["processes"] = [
            line.strip() for line in query.stdout.splitlines() if line.strip()
        ]
    return result


def stable_hash(value: str, seed: int = 42, length: int = 32) -> str:
    """Return a deterministic hexadecimal rank."""
    return hashlib.sha256(f"{seed}\x1f{value}".encode()).hexdigest()[:length]


def structure_score(text: str) -> float:
    """Score surface structure without using experimental or outcome fields."""
    value = str(text)
    lexical_cues = min(len(STRUCTURE_CUE_RE.findall(value)), 5)
    punctuation_cues = min(value.count(","), 2) + min(
        value.count(";") + value.count(":"), 2
    )
    length_component = min(len(value), 300) / 100.0
    return float(2 * lexical_cues + 0.5 * punctuation_cues + length_component)


def _round_robin_sample(
    candidates: pd.DataFrame,
    *,
    sample_size: int,
    rank_column: str,
) -> pd.DataFrame:
    """Select deterministically while distributing rows across trajectories."""
    if sample_size > len(candidates):
        raise ValueError("sample size exceeds available candidates")
    grouped = {
        str(name): list(group.sort_values(rank_column).index)
        for name, group in candidates.groupby("environment_id", sort=True)
    }
    selected: list[int] = []
    depth = 0
    while len(selected) < sample_size:
        added = False
        for name in sorted(grouped):
            indices = grouped[name]
            if depth < len(indices):
                selected.append(int(indices[depth]))
                added = True
                if len(selected) == sample_size:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != sample_size:
        raise ValueError("could not construct the requested trajectory-balanced sample")
    return candidates.loc[selected].copy()


def prepare_pilot_items(
    inventory: pd.DataFrame,
    *,
    sample_size: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    """Select equal random and structure-enriched samples from unlabelled rows."""
    required = {
        "sentence_id",
        "environment_id",
        "environment_step",
        "sentence_number",
        "context_before",
        "target_sentence",
        "target_sentence_characters",
        "previously_labelled",
    }
    if missing := required - set(inventory.columns):
        raise ValueError(f"inventory is missing columns: {sorted(missing)}")
    if sample_size <= 0 or sample_size % 2:
        raise ValueError("sample size must be a positive even integer")

    candidates = inventory.loc[~inventory["previously_labelled"].astype(bool)].copy()
    if sample_size > len(candidates):
        raise ValueError("sample size exceeds the number of unlabelled sentences")
    candidates["context_before"] = candidates["context_before"].fillna("")
    candidates["target_sentence"] = candidates["target_sentence"].fillna("")
    candidates["structure_score"] = candidates["target_sentence"].map(structure_score)
    candidates["random_rank"] = candidates["sentence_id"].map(
        lambda value: stable_hash(str(value), seed)
    )

    half = sample_size // 2
    random_rows = _round_robin_sample(
        candidates,
        sample_size=half,
        rank_column="random_rank",
    )
    random_rows["sample_stratum"] = "random"

    remaining = candidates.loc[~candidates.index.isin(random_rows.index)].copy()
    remaining["structure_rank"] = remaining.apply(
        lambda row: (
            f"{999999 - int(round(float(row['structure_score']) * 1000)):06d}__"
            f"{stable_hash(str(row['sentence_id']), seed + 1)}"
        ),
        axis=1,
    )
    enriched_rows = _round_robin_sample(
        remaining,
        sample_size=half,
        rank_column="structure_rank",
    )
    enriched_rows["sample_stratum"] = "structure_enriched"

    result = pd.concat([random_rows, enriched_rows], ignore_index=True)
    result["pilot_id"] = result["sentence_id"].map(
        lambda value: f"rf_{stable_hash(str(value), seed, length=16)}"
    )
    result = result.sort_values(["sample_stratum", "pilot_id"]).reset_index(drop=True)
    columns = [
        "pilot_id",
        "sentence_id",
        "environment_id",
        "environment_step",
        "sentence_number",
        "context_before",
        "target_sentence",
        "target_sentence_characters",
        "sample_stratum",
        "structure_score",
    ]
    result = result[columns]
    if result["sentence_id"].duplicated().any():
        raise ValueError("pilot sample contains duplicate sentences")
    if result["pilot_id"].duplicated().any():
        raise ValueError("pilot sample contains duplicate pilot IDs")
    return result


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first decodable JSON object from a model response."""
    value = str(text).strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    decoder = json.JSONDecoder()
    for position, character in enumerate(value):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("response does not contain a JSON object")


def normalize_nodes(
    payload: Mapping[str, Any],
    *,
    target_sentence: str,
) -> list[dict[str, Any]]:
    """Validate a complete, ordered partition of a target sentence into nodes."""
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("payload must contain a non-empty nodes list")
    target = str(target_sentence)
    normalized: list[dict[str, Any]] = []
    cursor = 0
    for index, raw in enumerate(raw_nodes, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"node {index} is not an object")
        try:
            start = int(raw["start_character"])
            end = int(raw["end_character"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"node {index} has invalid character offsets") from exc
        if start != cursor:
            raise ValueError(
                f"node {index} starts at {start}, expected contiguous offset {cursor}"
            )
        if end <= start or end > len(target):
            raise ValueError(f"node {index} has invalid span [{start}, {end})")
        node_text = str(raw.get("node_text", ""))
        if target[start:end] != node_text:
            raise ValueError(f"node {index} text does not match its character span")
        label = str(raw.get("reasoningflow_label", raw.get("label", ""))).lower()
        if label not in REASONINGFLOW_LABELS:
            raise ValueError(f"node {index} has invalid label {label!r}")
        confidence = str(raw.get("confidence", "")).lower()
        if confidence not in CONFIDENCE_LABELS:
            raise ValueError(f"node {index} has invalid confidence {confidence!r}")
        rationale = str(raw.get("short_rationale", raw.get("rationale", ""))).strip()
        if not rationale:
            raise ValueError(f"node {index} is missing a rationale")
        normalized.append(
            {
                "node_number": index,
                "start_character": start,
                "end_character": end,
                "node_text": node_text,
                "reasoningflow_label": label,
                "confidence": confidence,
                "short_rationale": rationale,
            }
        )
        cursor = end
    if cursor != len(target):
        raise ValueError(
            f"nodes end at character {cursor}, but target has {len(target)} characters"
        )
    return normalized


def select_review_sentences(
    pilot_items: pd.DataFrame,
    node_rows: pd.DataFrame,
    *,
    review_size: int = 40,
    priority_size: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Choose a fixed mix of challenging and ordinary sentences for review."""
    if review_size <= 0 or priority_size < 0 or priority_size > review_size:
        raise ValueError("invalid review or priority size")
    if review_size > len(pilot_items):
        raise ValueError("review size exceeds pilot size")
    required = {"pilot_id", "node_number", "confidence", "reasoningflow_label"}
    if missing := required - set(node_rows.columns):
        raise ValueError(f"node rows are missing columns: {sorted(missing)}")

    grouped = node_rows.groupby("pilot_id", sort=False).agg(
        predicted_node_count=("node_number", "count"),
        has_low_confidence=("confidence", lambda values: (values == "low").any()),
    )
    candidates = pilot_items.merge(grouped, on="pilot_id", how="left")
    candidates["predicted_node_count"] = (
        candidates["predicted_node_count"].fillna(0).astype(int)
    )
    candidates["has_low_confidence"] = candidates["has_low_confidence"].fillna(True)
    candidates["priority_review"] = (
        candidates["predicted_node_count"].gt(1) | candidates["has_low_confidence"]
    )
    candidates["review_rank"] = candidates["pilot_id"].map(
        lambda value: stable_hash(str(value), seed + 10)
    )
    priority = candidates.loc[candidates["priority_review"]].sort_values("review_rank")
    selected_priority = priority.head(priority_size)
    remaining = candidates.loc[
        ~candidates.index.isin(selected_priority.index)
    ].sort_values("review_rank")
    selected = pd.concat(
        [selected_priority, remaining.head(review_size - len(selected_priority))]
    ).copy()
    if len(selected) != review_size:
        raise ValueError("could not select the requested review sentences")

    nodes_json = {
        pilot_id: json.dumps(
            group[
                [
                    "node_number",
                    "start_character",
                    "end_character",
                    "node_text",
                    "reasoningflow_label",
                    "confidence",
                    "short_rationale",
                ]
            ].to_dict("records"),
            ensure_ascii=False,
        )
        for pilot_id, group in node_rows.sort_values(
            ["pilot_id", "node_number"]
        ).groupby("pilot_id", sort=False)
    }
    selected["model_nodes_json"] = selected["pilot_id"].map(nodes_json).fillna("[]")
    selected["boundary_reasonable"] = ""
    selected["labels_reasonable"] = ""
    selected["missing_doorkey_function"] = ""
    selected["manual_nodes_json"] = ""
    selected["reviewer"] = ""
    selected["review_notes"] = ""
    return selected[
        [
            "pilot_id",
            "sentence_id",
            "sample_stratum",
            "context_before",
            "target_sentence",
            "predicted_node_count",
            "priority_review",
            "model_nodes_json",
            "boundary_reasonable",
            "labels_reasonable",
            "missing_doorkey_function",
            "manual_nodes_json",
            "reviewer",
            "review_notes",
        ]
    ].reset_index(drop=True)


def _review_rate(review: pd.DataFrame, column: str) -> tuple[int, int, float | None]:
    values = review[column].fillna("").astype(str).str.strip().str.lower()
    reviewed = values.isin({"yes", "no"})
    denominator = int(reviewed.sum())
    numerator = int(values.eq("yes").sum())
    rate = numerator / denominator if denominator else None
    return numerator, denominator, rate


def analyze_pilot(
    pilot_items: pd.DataFrame,
    node_rows: pd.DataFrame,
    review: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build compact tables and metrics for the transfer pilot."""
    complete_ids = set(node_rows["pilot_id"].astype(str))
    sentence_summary = pilot_items.copy()
    node_counts = node_rows.groupby("pilot_id").size().rename("node_count")
    sentence_summary = sentence_summary.merge(node_counts, on="pilot_id", how="left")
    sentence_summary["node_count"] = (
        sentence_summary["node_count"].fillna(0).astype(int)
    )
    sentence_summary["label_complete"] = sentence_summary["pilot_id"].isin(complete_ids)
    sentence_summary["segmentation_group"] = sentence_summary["node_count"].map(
        lambda value: (
            "unlabelled"
            if value == 0
            else ("one_node" if value == 1 else "multiple_nodes")
        )
    )
    segmentation = (
        sentence_summary.groupby(["sample_stratum", "segmentation_group"], dropna=False)
        .agg(
            sentences=("pilot_id", "count"), trajectories=("environment_id", "nunique")
        )
        .reset_index()
    )

    label_summary = (
        node_rows.groupby("reasoningflow_label", dropna=False)
        .agg(
            nodes=("pilot_id", "count"),
            sentences=("pilot_id", "nunique"),
            low_confidence_nodes=("confidence", lambda values: (values == "low").sum()),
        )
        .reset_index()
    )
    reviewed_metrics: dict[str, Any] = {}
    schema_gaps = pd.DataFrame(
        columns=[
            "pilot_id",
            "sentence_id",
            "target_sentence",
            "missing_doorkey_function",
            "review_notes",
        ]
    )
    if review is not None and len(review):
        for column in ("boundary_reasonable", "labels_reasonable"):
            numerator, denominator, rate = _review_rate(review, column)
            reviewed_metrics[column] = {
                "yes": numerator,
                "reviewed": denominator,
                "rate": rate,
            }
        gap_mask = (
            review["missing_doorkey_function"].fillna("").astype(str).str.strip().ne("")
        )
        schema_gaps = review.loc[
            gap_mask,
            [
                "pilot_id",
                "sentence_id",
                "target_sentence",
                "missing_doorkey_function",
                "review_notes",
            ],
        ].copy()

    return {
        "sentence_summary": sentence_summary,
        "segmentation_summary": segmentation,
        "label_summary": label_summary,
        "schema_gaps": schema_gaps,
        "reviewed_metrics": reviewed_metrics,
        "pilot_sentences": len(pilot_items),
        "labelled_sentences": int(sentence_summary["label_complete"].sum()),
        "nodes": len(node_rows),
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_example_splits(
    path: Path,
    pilot_items: pd.DataFrame,
    node_rows: pd.DataFrame,
    *,
    maximum_examples: int = 10,
) -> None:
    counts = node_rows.groupby("pilot_id").size().rename("node_count")
    examples = pilot_items.merge(counts, on="pilot_id", how="inner").sort_values(
        ["node_count", "structure_score", "pilot_id"], ascending=[False, False, True]
    )
    blocks = ["# Example ReasoningFlow splits", ""]
    for row in examples.head(maximum_examples).itertuples(index=False):
        blocks.extend([f"## {row.sentence_id}", "", f"> {row.target_sentence}", ""])
        subset = node_rows.loc[node_rows["pilot_id"].eq(row.pilot_id)].sort_values(
            "node_number"
        )
        for node in subset.itertuples(index=False):
            blocks.append(
                f"- `{node.reasoningflow_label}` [{node.start_character}:{node.end_character}]: "
                f"{node.node_text!r}"
            )
        blocks.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(blocks))


def validate_output_columns(frame: pd.DataFrame, required: Sequence[str]) -> None:
    if missing := set(required) - set(frame.columns):
        raise ValueError(f"output is missing columns: {sorted(missing)}")
