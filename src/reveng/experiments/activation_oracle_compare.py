from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def _canonicalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonicalize_json(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_canonicalize_json(item) for item in value]
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _extract_json_blob(text: str) -> str | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else None


def normalize_oracle_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(_canonicalize_json(value), sort_keys=True, separators=(",", ":"))

    text = str(value).strip()
    if not text:
        return ""

    candidate = text
    json_blob = _extract_json_blob(text)
    if json_blob is not None:
        candidate = json_blob

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        if normalized in {"up", "down", "left", "right", "unknown", "yes", "no", "blocked", "free"}:
            return normalized
        return normalized
    return json.dumps(_canonicalize_json(parsed), sort_keys=True, separators=(",", ":"))


def compare_activation_oracle_results(
    results_path: str,
    output_csv_path: str | None = None,
    summary_json_path: str | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(results_path)
    if output_csv_path is None:
        output_csv_path = str(Path(results_path).with_name("ao_behavioral_comparison.csv"))
    if summary_json_path is None:
        summary_json_path = str(Path(results_path).with_name("ao_behavioral_comparison_summary.json"))

    comparison_rows: list[dict[str, Any]] = []
    summary = {
        "results_path": str(results_path),
        "rows": len(rows),
        "rows_with_behavioral_label": 0,
        "rows_with_representational_label": 0,
        "rows_with_ground_truth": 0,
        "segment_matches_behavioral": 0,
        "full_sequence_matches_behavioral": 0,
        "segment_matches_representational": 0,
        "full_sequence_matches_representational": 0,
        "segment_matches_ground_truth": 0,
        "full_sequence_matches_ground_truth": 0,
    }

    for row in rows:
        segment_response = (row.get("segment_responses") or [""])[0] if row.get("segment_responses") else ""
        full_sequence_response = (row.get("full_sequence_responses") or [""])[0] if row.get("full_sequence_responses") else ""
        behavioral_label = row.get("behavioral_label", "")
        ground_truth = row.get("ground_truth", "")

        normalized_behavioral = normalize_oracle_value(behavioral_label)
        representational_label = row.get("representational_label", "")
        normalized_representational = normalize_oracle_value(representational_label)
        normalized_ground_truth = normalize_oracle_value(ground_truth)
        normalized_segment = normalize_oracle_value(segment_response)
        normalized_full_sequence = normalize_oracle_value(full_sequence_response)

        segment_matches_behavioral = bool(normalized_behavioral) and normalized_segment == normalized_behavioral
        full_matches_behavioral = bool(normalized_behavioral) and normalized_full_sequence == normalized_behavioral
        segment_matches_representational = bool(normalized_representational) and normalized_segment == normalized_representational
        full_matches_representational = bool(normalized_representational) and normalized_full_sequence == normalized_representational
        segment_matches_ground_truth = bool(normalized_ground_truth) and normalized_segment == normalized_ground_truth
        full_matches_ground_truth = bool(normalized_ground_truth) and normalized_full_sequence == normalized_ground_truth

        summary["rows_with_behavioral_label"] += int(bool(normalized_behavioral))
        summary["rows_with_representational_label"] += int(bool(normalized_representational))
        summary["rows_with_ground_truth"] += int(bool(normalized_ground_truth))
        summary["segment_matches_behavioral"] += int(segment_matches_behavioral)
        summary["full_sequence_matches_behavioral"] += int(full_matches_behavioral)
        summary["segment_matches_representational"] += int(segment_matches_representational)
        summary["full_sequence_matches_representational"] += int(full_matches_representational)
        summary["segment_matches_ground_truth"] += int(segment_matches_ground_truth)
        summary["full_sequence_matches_ground_truth"] += int(full_matches_ground_truth)

        comparison_rows.append(
            {
                "row_id": row.get("row_id", ""),
                "example_id": row.get("example_id", ""),
                "trajectory_id": row.get("trajectory_id", ""),
                "step_index": row.get("step_index", ""),
                "reasoning_reveal_pct": row.get("reasoning_reveal_pct", ""),
                "question_id": row.get("question_id", ""),
                "oracle_prompt": row.get("oracle_prompt", ""),
                "behavioral_label": behavioral_label,
                "representational_label": representational_label,
                "ground_truth": ground_truth,
                "segment_response": segment_response,
                "full_sequence_response": full_sequence_response,
                "normalized_behavioral_label": normalized_behavioral,
                "normalized_representational_label": normalized_representational,
                "normalized_ground_truth": normalized_ground_truth,
                "normalized_segment_response": normalized_segment,
                "normalized_full_sequence_response": normalized_full_sequence,
                "segment_matches_behavioral_label": segment_matches_behavioral,
                "full_sequence_matches_behavioral_label": full_matches_behavioral,
                "segment_matches_representational_label": segment_matches_representational,
                "full_sequence_matches_representational_label": full_matches_representational,
                "segment_matches_ground_truth": segment_matches_ground_truth,
                "full_sequence_matches_ground_truth": full_matches_ground_truth,
            }
        )

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()) if comparison_rows else [
            "row_id",
            "oracle_prompt",
            "behavioral_label",
            "representational_label",
            "ground_truth",
            "segment_response",
            "full_sequence_response",
            "normalized_behavioral_label",
            "normalized_representational_label",
            "normalized_ground_truth",
            "normalized_segment_response",
            "normalized_full_sequence_response",
            "segment_matches_behavioral_label",
            "full_sequence_matches_behavioral_label",
            "segment_matches_representational_label",
            "full_sequence_matches_representational_label",
            "segment_matches_ground_truth",
            "full_sequence_matches_ground_truth",
        ])
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary_path = Path(summary_json_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    return summary


__all__ = ["compare_activation_oracle_results", "normalize_oracle_value"]
