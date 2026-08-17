"""Utilities for immediate-action attention at reasoning sentence boundaries."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from reveng.experiments.local_immediate_behavioral import (
    ACTION_LABELS,
    _action_question,
    _analysis_text,
    _belief_question,
    _bool,
    _context_text,
    _longest_common_prefix,
    _trajectory_step,
)


REGION_NAMES = (
    "grid",
    "agent_status",
    "earlier_reasoning",
    "newest_reasoning_sentence",
    "action_question",
    "answer_scaffold",
    "chosen_move_cells",
    "task_objects",
)

REGION_LABELS = {
    "grid": "Grid",
    "agent_status": "Key possession statement",
    "earlier_reasoning": "Earlier reasoning sentences",
    "newest_reasoning_sentence": "Newest reasoning sentence",
    "action_question": "Action question",
    "answer_scaffold": "Answer-format prompt",
    "chosen_move_cells": "Agent and chosen destination cells",
    "task_objects": "Key, door, and goal cells",
}

ACTION_DELTAS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def trace_id(trajectory_id: str, step_index: int) -> str:
    return f"{trajectory_id}_step_{int(step_index):03d}"


def load_sentence_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        if row.get("kind", "reasoning") != "reasoning":
            continue
        grouped[row["trace_id"]].append(row)
    return {
        key: sorted(value, key=lambda row: int(row["sentence_id"]))
        for key, value in grouped.items()
    }


def grid_cell_character_spans(grid_text: str) -> dict[tuple[int, int], tuple[int, int, str]]:
    """Map grid row and column to the character span of its cell marker."""
    cells: dict[tuple[int, int], tuple[int, int, str]] = {}
    columns: list[int] | None = None
    offset = 0
    for line in grid_text.splitlines(keepends=True):
        matches = list(re.finditer(r"\S+", line))
        if columns is None and matches and all(match.group().isdigit() for match in matches):
            columns = [int(match.group()) for match in matches]
        elif (
            columns is not None
            and len(matches) >= len(columns) + 1
            and matches[0].group().isdigit()
        ):
            row = int(matches[0].group())
            for col, match in zip(columns, matches[1 : len(columns) + 1]):
                cells[(row, col)] = (
                    offset + match.start(),
                    offset + match.end(),
                    match.group(),
                )
        offset += len(line)
    return cells


def _find_unique(text: str, value: str, *, start: int = 0) -> tuple[int, int]:
    first = text.find(value, start)
    if first < 0:
        raise ValueError(f"Could not locate prompt substring: {value[:80]!r}")
    return first, first + len(value)


def _overlapping_token_indices(
    offsets: Sequence[tuple[int, int]],
    spans: Iterable[tuple[int, int]],
    *,
    exclude: set[int] | None = None,
) -> list[int]:
    excluded = exclude or set()
    spans = [(start, end) for start, end in spans if start < end]
    return [
        token_index
        for token_index, (token_start, token_end) in enumerate(offsets)
        if token_index not in excluded
        and token_start < token_end
        and any(token_start < end and token_end > start for start, end in spans)
    ]


def render_action_prompt(
    *,
    tokenizer: Any,
    grid_text: str,
    carrying_key: bool,
    analysis: str,
    sentence_rows: Sequence[dict[str, str]],
    position_index: int,
    expected_action: str,
) -> dict[str, Any]:
    """Render the exact immediate-action query and map its semantic regions."""
    if position_index < 0 or position_index > len(sentence_rows):
        raise ValueError(f"Invalid sentence position {position_index}.")
    char_end = 0 if position_index == 0 else int(sentence_rows[position_index - 1]["char_end"])
    reasoning_raw = analysis[:char_end]
    reasoning = reasoning_raw.strip()
    context = _context_text(
        grid_text=grid_text,
        carrying_key=carrying_key,
        reasoning_prefix=reasoning_raw,
    )
    question = _action_question()[1]
    chat_rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": context + question}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )
    scaffold = '<|channel|>final<|message|>{"action":"'
    rendered = chat_rendered + scaffold
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    if not input_ids:
        raise ValueError("Rendered action prompt has no tokens.")
    query_index = len(input_ids) - 1
    # The original behavioral runner caches the longest prefix shared by the
    # action and belief questions, then scores each complete question suffix.
    # Preserve that partition for exact action-probability reproduction.
    belief_question = _belief_question("wall_left")[1]
    belief_rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": context + belief_question}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )
    belief_ids = list(
        tokenizer(belief_rendered, add_special_tokens=False)["input_ids"]
    )
    scoring_common_length = _longest_common_prefix((input_ids, belief_ids))
    if scoring_common_length <= 0 or scoring_common_length >= len(input_ids):
        raise ValueError("Could not recover the original behavioral cache boundary.")

    grid_start, grid_end = _find_unique(rendered, grid_text)
    agent_start, _ = _find_unique(rendered, "Agent status:")
    reasoning_header_start, reasoning_header_end = _find_unique(
        rendered, "Reasoning trace available so far:"
    )
    question_start, question_end = _find_unique(rendered, question)
    scaffold_start = len(chat_rendered)

    # Context always inserts one newline after this header.
    reasoning_start = reasoning_header_end
    if rendered[reasoning_start : reasoning_start + 1] == "\n":
        reasoning_start += 1
    reasoning_end = reasoning_start + len(reasoning)
    if rendered[reasoning_start:reasoning_end] != reasoning:
        raise ValueError("Reasoning text does not match the rendered prompt.")

    latest_span: tuple[int, int] | None = None
    earlier_span: tuple[int, int] | None = None
    if position_index > 0 and reasoning:
        latest = sentence_rows[position_index - 1]
        leading_trim = len(reasoning_raw) - len(reasoning_raw.lstrip())
        latest_start = max(0, int(latest["char_start"]) - leading_trim)
        latest_end = min(len(reasoning), int(latest["char_end"]) - leading_trim)
        if latest_start < latest_end:
            latest_span = (reasoning_start + latest_start, reasoning_start + latest_end)
            if latest_start > 0:
                earlier_span = (reasoning_start, reasoning_start + latest_start)

    cells = grid_cell_character_spans(grid_text)

    region_spans: dict[str, list[tuple[int, int]]] = {
        "grid": [(grid_start, grid_end)],
        "agent_status": [(agent_start, reasoning_header_start)],
        "earlier_reasoning": [earlier_span] if earlier_span else [],
        "newest_reasoning_sentence": [latest_span] if latest_span else [],
        "action_question": [(question_start, question_end)],
        "answer_scaffold": [(scaffold_start, len(rendered))],
        "chosen_move_cells": [],
        "task_objects": [],
    }
    region_tokens = {
        name: _overlapping_token_indices(
            offsets,
            spans,
            exclude={query_index},
        )
        for name, spans in region_spans.items()
    }
    return {
        "rendered": rendered,
        "input_ids": input_ids,
        "offsets": offsets,
        "query_index": query_index,
        "scoring_common_length": scoring_common_length,
        "char_end": char_end,
        "region_spans": region_spans,
        "region_token_indices": region_tokens,
        "grid_start": grid_start,
        "grid_cells": cells,
        "reasoning_start": reasoning_start,
        "reasoning_length": len(reasoning),
        "reasoning_leading_trim": len(reasoning_raw) - len(reasoning_raw.lstrip()),
        "expected_action": expected_action,
    }


def add_grid_action_regions(
    prepared: dict[str, Any],
    *,
    current_position: str,
    action: str,
) -> None:
    """Add token spans for the current cell, chosen destination, and K/D/G cells."""
    current_col, current_row = (int(value.strip()) for value in current_position.split(","))
    delta_row, delta_col = ACTION_DELTAS[action]
    destination = (current_row + delta_row, current_col + delta_col)
    current = (current_row, current_col)
    cells = prepared["grid_cells"]
    chosen_spans = []
    for coordinate in (current, destination):
        if coordinate in cells:
            start, end, _marker = cells[coordinate]
            chosen_spans.append((prepared["grid_start"] + start, prepared["grid_start"] + end))
    object_spans = [
        (prepared["grid_start"] + start, prepared["grid_start"] + end)
        for start, end, marker in cells.values()
        if marker in {"K", "D", "G"}
    ]
    prepared["region_spans"]["chosen_move_cells"] = chosen_spans
    prepared["region_spans"]["task_objects"] = object_spans
    for name in ("chosen_move_cells", "task_objects"):
        prepared["region_token_indices"][name] = _overlapping_token_indices(
            prepared["offsets"],
            prepared["region_spans"][name],
            exclude={prepared["query_index"]},
        )


def build_prompt_specs(
    *,
    pair_rows_path: Path,
    candidate_rows_path: Path,
    position_rows_path: Path,
    sentence_rows_path: Path,
    trajectory_dir: Path,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    pairs = read_csv(pair_rows_path)
    candidates = {row["example_id"]: row for row in read_csv(candidate_rows_path)}
    positions = {
        (row["example_id"], int(row["position_index"])): row
        for row in read_csv(position_rows_path)
    }
    sentences = load_sentence_rows(sentence_rows_path)
    required = {
        (row["example_id"], max(0, int(row["position_index"]) + delta))
        for row in pairs
        for delta in (-1, 0)
    }
    payloads: dict[str, dict[str, Any]] = {}
    specs: list[dict[str, Any]] = []
    for example_id, position_index in sorted(required):
        candidate = candidates[example_id]
        trajectory_id = candidate["trajectory_id"]
        if trajectory_id not in payloads:
            payloads[trajectory_id] = json.loads(
                (trajectory_dir / f"{trajectory_id}.json").read_text()
            )
        step_index = int(candidate["step_index"])
        analysis = _analysis_text(
            _trajectory_step(payloads[trajectory_id], step_index)["output_text"]
        )
        canonical = sentences[trace_id(trajectory_id, step_index)]
        expected = positions[(example_id, position_index)]
        prepared = render_action_prompt(
            tokenizer=tokenizer,
            grid_text=candidate["grid_text"],
            carrying_key=_bool(candidate["carrying_key"]),
            analysis=analysis,
            sentence_rows=canonical,
            position_index=position_index,
            expected_action=expected["action_label"],
        )
        if int(expected["analysis_char_end"]) != int(prepared["char_end"]):
            raise ValueError(f"Character boundary mismatch for {example_id}:{position_index}.")
        add_grid_action_regions(
            prepared,
            current_position=candidate["current_position"],
            action=expected["action_label"],
        )
        specs.append(
            {
                "example_id": example_id,
                "trajectory_id": trajectory_id,
                "step_index": step_index,
                "position_index": position_index,
                "reasoning_progress": float(expected["reasoning_progress"]),
                "expected_action": expected["action_label"],
                "expected_probabilities": json.loads(expected["action_probabilities_json"]),
                "current_position": candidate["current_position"],
                "primary_step_failure_mode": expected.get("primary_step_failure_mode", ""),
                "matched_role": expected.get("matched_role", ""),
                **prepared,
            }
        )
    return specs, pairs


def probability_error(
    observed: dict[str, float],
    expected: dict[str, float],
) -> float:
    return max(abs(float(observed[label]) - float(expected[label])) for label in ACTION_LABELS)


def aggregate_attention(
    attention: Any,
    *,
    sentence_token_indices: Sequence[Sequence[int]],
    region_token_indices: dict[str, Sequence[int]],
) -> dict[str, Any]:
    """Aggregate a heads-by-source-token tensor without discarding head identity."""
    import torch

    if attention.ndim != 2:
        raise ValueError("Attention must have shape [heads, source_tokens].")
    sentence_mass = torch.zeros(
        (attention.shape[0], len(sentence_token_indices)), dtype=torch.float32
    )
    sentence_mean = torch.zeros_like(sentence_mass)
    sentence_counts = torch.zeros(len(sentence_token_indices), dtype=torch.int64)
    for index, token_indices in enumerate(sentence_token_indices):
        valid = [token for token in token_indices if 0 <= token < attention.shape[-1]]
        sentence_counts[index] = len(valid)
        if valid:
            values = attention[:, valid]
            sentence_mass[:, index] = values.sum(dim=-1)
            sentence_mean[:, index] = values.mean(dim=-1)
    region_mass = torch.zeros(
        (attention.shape[0], len(REGION_NAMES)), dtype=torch.float32
    )
    region_mean = torch.zeros_like(region_mass)
    region_counts = torch.zeros(len(REGION_NAMES), dtype=torch.int64)
    for index, name in enumerate(REGION_NAMES):
        valid = [
            token
            for token in region_token_indices.get(name, ())
            if 0 <= token < attention.shape[-1]
        ]
        region_counts[index] = len(valid)
        if valid:
            values = attention[:, valid]
            region_mass[:, index] = values.sum(dim=-1)
            region_mean[:, index] = values.mean(dim=-1)
    return {
        "sentence_mass": sentence_mass,
        "sentence_mean": sentence_mean,
        "sentence_counts": sentence_counts,
        "region_mass": region_mass,
        "region_mean": region_mean,
        "region_counts": region_counts,
    }


def paired_attention_differences(
    region_rows: Sequence[dict[str, Any]],
    pair_rows: Sequence[dict[str, str]],
    *,
    metric: str,
) -> list[dict[str, Any]]:
    values = {
        (
            str(row["example_id"]),
            int(row["position_index"]),
            int(row["layer"]),
            str(row["region"]),
        ): float(row[metric])
        for row in region_rows
        if str(row.get("available", "True")).lower() == "true"
        and str(row.get(metric, "")) not in {"", "nan"}
    }
    by_pair: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in pair_rows:
        by_pair[row["pair_id"]][row["item_role"]] = row
    output: list[dict[str, Any]] = []
    layers = sorted({int(row["layer"]) for row in region_rows})
    regions = sorted({str(row["region"]) for row in region_rows})
    for pair_id, roles in sorted(by_pair.items()):
        change = roles.get("detected_change_point")
        control = roles.get("matched_non_change_sentence")
        if not change or not control:
            continue
        for layer in layers:
            for region in regions:
                def get(item: dict[str, str], delta: int) -> float | None:
                    key = (
                        item["example_id"],
                        max(0, int(item["position_index"]) + delta),
                        layer,
                        region,
                    )
                    return values.get(key)

                change_before, change_at = get(change, -1), get(change, 0)
                control_before, control_at = get(control, -1), get(control, 0)
                if None in {change_before, change_at, control_before, control_at}:
                    continue
                change_delta = float(change_at) - float(change_before)
                control_delta = float(control_at) - float(control_before)
                output.append(
                    {
                        "pair_id": pair_id,
                        "example_id": change["example_id"],
                        "trajectory_id": change["trajectory_id"],
                        "step_index": int(change["step_index"]),
                        "match_quality": change["match_quality"],
                        "change_point_position": int(change["position_index"]),
                        "control_position": int(control["position_index"]),
                        "layer": layer,
                        "region": region,
                        "metric": metric,
                        "change_point_before": change_before,
                        "change_point_at": change_at,
                        "matched_control_before": control_before,
                        "matched_control_at": control_at,
                        "change_point_delta": change_delta,
                        "matched_control_delta": control_delta,
                        "difference_in_differences": change_delta - control_delta,
                    }
                )
    return output


def trajectory_bootstrap_ci(
    rows: Sequence[dict[str, Any]],
    *,
    value_key: str,
    repeats: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["trajectory_id"])].append(float(row[value_key]))
    trajectory_values = np.asarray(
        [float(np.mean(values)) for values in grouped.values()], dtype=float
    )
    if not len(trajectory_values):
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [
            float(np.mean(rng.choice(trajectory_values, len(trajectory_values), replace=True)))
            for _ in range(repeats)
        ]
    )
    return (
        float(np.mean(trajectory_values)),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )
