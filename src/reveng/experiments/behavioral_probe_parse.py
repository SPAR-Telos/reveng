"""Parsing utilities for behavioral-probe black-box answers."""

from __future__ import annotations

import json
import re
from typing import Any

VALID_BEHAVIORAL_PROBE_ANSWERS = ("yes", "no", "unknown", "invalid")
BEHAVIORAL_PROBE_LABEL_MAP = {
    "a": "yes",
    "b": "no",
    "c": "unknown",
    "yes": "yes",
    "no": "no",
    "unknown": "unknown",
}
COORD_MISSING = {"row": -1, "col": -1}


def parse_behavioral_probe_answer(raw_text: str | None) -> str:
    """Parse a raw model string into yes/no/unknown/invalid."""
    if raw_text is None:
        return "invalid"

    cleaned = raw_text.strip().lower()
    if not cleaned:
        return "invalid"

    while cleaned and cleaned[-1] in {".", "!", "?", '"', "'", " "}:
        cleaned = cleaned[:-1].rstrip()
    while cleaned and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:].lstrip()

    if cleaned in BEHAVIORAL_PROBE_LABEL_MAP:
        return BEHAVIORAL_PROBE_LABEL_MAP[cleaned]
    return "invalid"


def parse_coordinate_answer(raw_text: str | None) -> dict[str, int] | None:
    """Parse a JSON coordinate response of the form {"row": int, "col": int}."""
    if raw_text is None:
        return None
    cleaned = raw_text.strip()
    if not cleaned:
        return None

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        payload = json.loads(cleaned)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    row = payload.get("row")
    col = payload.get("col")
    if not isinstance(row, int) or not isinstance(col, int):
        return None
    return {"row": row, "col": col}


def coordinate_to_key(coord: dict[str, int] | None) -> str:
    if coord is None:
        return "invalid"
    return f"{coord['row']},{coord['col']}"


def coordinate_manhattan_distance(
    pred: dict[str, int] | None,
    target: dict[str, int] | None,
) -> int | None:
    if pred is None or target is None:
        return None
    return abs(int(pred["row"]) - int(target["row"])) + abs(int(pred["col"]) - int(target["col"]))


__all__ = [
    "BEHAVIORAL_PROBE_LABEL_MAP",
    "COORD_MISSING",
    "VALID_BEHAVIORAL_PROBE_ANSWERS",
    "coordinate_manhattan_distance",
    "coordinate_to_key",
    "parse_behavioral_probe_answer",
    "parse_coordinate_answer",
]
