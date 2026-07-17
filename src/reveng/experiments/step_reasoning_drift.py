"""Step-level reasoning-drift experiment on DoorKey reasoning traces.

This module has two tiers:

1. Behavioral-only prefix evaluation on existing reasoning traces.
2. Optional local hidden-state collection for per-step geometry metrics.

The public released activation artifacts are intentionally not reused here:
they provide pre/post reasoning snapshots, not step-indexed hidden states.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import re
import threading
from bisect import bisect_left
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Sequence

from reveng.experiments.behavioral_probe_runner import (
    BehavioralProbeLLM,
    _behavioral_probe_preamble,
    _collect_action4_logprob_readout,
    _empty_usage_bucket,
    _query_action,
)
from reveng.experiments.gradual_cot_blackbox_alignment import (
    _extract_analysis_and_final,
    _load_local_trajectory_payload,
    _select_candidate_slice_rows,
)
from reveng.experiments.wall_feature_patch_experiment import (
    load_wall_feature_patch_candidates,
    select_wall_feature_patch_states,
)

DEFAULT_CANDIDATE_ROWS_PATH = (
    "data/behavioral_probes/trajectory_instances_recomputed_optimal/trajectory_selection_candidates.csv"
)
DEFAULT_TRAJECTORY_DIR = "data/hf/trajectories_key_door_100/trajectories_key_door"
DEFAULT_OUTPUT_DIR = "data/behavioral_probes/step_reasoning_drift"
DEFAULT_MODEL_NAME = "together_ai/openai/gpt-oss-20b"
DEFAULT_PROMPT_PRESET = "cardinal_action_explicit"
DEFAULT_LAYERS = (15,)
DEFAULT_SENTENCE_BOUNDARIES_PATH = (
    "data/behavioral_probes/doorkey_chunking_validation/sentences.csv"
)
ACTION_LABELS = ("UP", "DOWN", "LEFT", "RIGHT")
ACTIVATION_SCHEMA_VERSION = "step_reasoning_drift_v1"
CHUNKER_VERSION = "doorkey_sentence_v1"
PACKER_VERSION = "balanced_contiguous_word_count_v2"

try:  # Optional in unit tests and behavior-only runs.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


@dataclass(frozen=True)
class ReasoningStep:
    step_idx: int
    start_char: int
    end_char: int
    text: str
    segmentation_mode: str
    canonical_sentence_id: int | None = None


@dataclass(frozen=True)
class SentenceChunk:
    trace_id: str
    chunk_id: int
    kind: str
    text: str
    char_start: int
    char_end: int
    contains_action_json_mention: bool
    chunker_version: str = CHUNKER_VERSION
    token_start: int | None = None
    token_end: int | None = None


@dataclass(frozen=True)
class AnalysisChunk:
    analysis_id: int
    sentence_start: int
    sentence_end: int
    text: str
    char_start: int
    char_end: int

@dataclass(frozen=True)
class DriftExample:
    example_id: str
    trajectory_id: str
    step_index: int
    grid_text: str
    carrying_key: bool
    observed_action: str
    optimal_actions: tuple[str, ...]
    is_optimal_action: bool | None
    failure_category: str
    source_dataset: str
    selection_stage: str


@dataclass(frozen=True)
class PrefixStepEvaluation:
    reasoning_step_idx: int
    revealed_step_count: int
    revealed_analysis_chars: int
    action_label: str
    action_is_optimal: bool | None
    action_matches_final: bool | None
    revealed_analysis_text: str


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value)).strip("_") or "unnamed"


def _safe_relative_path(value: str) -> Path:
    return Path(*[_safe_path_component(part) for part in str(value).split("/") if part])


def _parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).upper() for item in value]
    if value is None:
        return []
    stripped = str(value).strip()
    if not stripped:
        return []
    try:
        parsed = json.loads(stripped)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [str(item).upper() for item in parsed]
    return []


def _blank_line_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for chunk in re.split(r"(\n\s*\n+)", text):
        if not chunk:
            continue
        start = cursor
        cursor += len(chunk)
        if chunk.strip():
            spans.append((start, cursor))
    return spans


_ACTION_PATTERN = r"UP|DOWN|LEFT|RIGHT"
_ACTION_JSON_RE = re.compile(rf'\{{\s*"action"\s*:\s*"(?:{_ACTION_PATTERN})"\s*,?\s*\}}')
_PROTECTED_SPLIT_RES = (
    re.compile(rf"(?<!\w)\d+\.(?=\s+(?:{_ACTION_PATTERN})\b)"),
    re.compile(r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)"),
    re.compile(r"\b\d+\.\d+\b"),
    re.compile(r"\b(?:e\.g|i\.e|etc|vs)\.", flags=re.IGNORECASE),
    _ACTION_JSON_RE,
)


def find_action_json_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in _ACTION_JSON_RE.finditer(text)]


def find_terminal_action_json_span(text: str) -> tuple[int, int] | None:
    spans = find_action_json_spans(text)
    for start, end in reversed(spans):
        if re.fullmatch(r"[\s\.。]*", text[end:]):
            return start, end
    return None


def _protected_split_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pattern in _PROTECTED_SPLIT_RES:
        spans.extend(match.span() for match in pattern.finditer(text))
    return sorted(spans)


def _in_span(index: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in spans)


def _canonical_sentence_spans(text: str) -> list[tuple[int, int]]:
    """Sentence-level DoorKey chunk spans with guards for false boundaries."""
    spans: list[tuple[int, int]] = []
    protected = _protected_split_spans(text)
    cursor = 0
    split_re = re.compile(r"(?:\n\s*\n)|(?<=[.!?])\s+")
    for match in split_re.finditer(text):
        is_blank_break = "\n" in match.group(0) and re.fullmatch(r"\n\s*\n", match.group(0))
        if not is_blank_break:
            punctuation_idx = match.start() - 1
            if punctuation_idx >= 0 and _in_span(punctuation_idx, protected):
                continue
        end = match.start()
        if text[cursor:end].strip():
            start = cursor + len(text[cursor:end]) - len(text[cursor:end].lstrip())
            trimmed_end = end - len(text[cursor:end]) + len(text[cursor:end].rstrip())
            spans.append((start, trimmed_end))
        cursor = match.end()
    if text[cursor:].strip():
        start = cursor + len(text[cursor:]) - len(text[cursor:].lstrip())
        end = len(text) - len(text[cursor:]) + len(text[cursor:].rstrip())
        spans.append((start, end))

    merged: list[tuple[int, int]] = []
    buffer: tuple[int, int] | None = None
    for start, end in spans:
        if len(text[start:end].strip()) < 10:
            buffer = (buffer[0], end) if buffer is not None else (start, end)
            continue
        if buffer is not None:
            start = buffer[0]
            buffer = None
        merged.append((start, end))
    if buffer is not None:
        if merged:
            merged[-1] = (merged[-1][0], buffer[1])
        else:
            merged.append(buffer)
    return merged


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    return _canonical_sentence_spans(text)


def chunk_doorkey_reasoning_trace(
    raw_reasoning_trace: str,
    *,
    trace_id: str = "",
) -> list[SentenceChunk]:
    """Return immutable sentence spans for a DoorKey reasoning trace."""
    terminal_span = find_terminal_action_json_span(raw_reasoning_trace)
    if terminal_span is None:
        reasoning_text = raw_reasoning_trace
        final_text = None
    else:
        reasoning_text = raw_reasoning_trace[: terminal_span[0]]
        final_text = raw_reasoning_trace[terminal_span[0] : terminal_span[1]]

    chunks: list[SentenceChunk] = []
    for start, end in _canonical_sentence_spans(reasoning_text):
        text = raw_reasoning_trace[start:end]
        chunks.append(
            SentenceChunk(
                trace_id=trace_id,
                chunk_id=len(chunks),
                kind="reasoning",
                text=text,
                char_start=start,
                char_end=end,
                contains_action_json_mention=bool(find_action_json_spans(text)),
            )
        )
    if final_text is not None:
        chunks.append(
            SentenceChunk(
                trace_id=trace_id,
                chunk_id=len(chunks),
                kind="final_action",
                text=final_text,
                char_start=terminal_span[0],
                char_end=terminal_span[1],
                contains_action_json_mention=True,
            )
        )
    return chunks


def pack_to_analysis_chunks(
    sentence_chunks: Sequence[SentenceChunk],
    *,
    max_chunks: int = 32,
    raw_text: str | None = None,
) -> list[AnalysisChunk]:
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive")
    reasoning_chunks = [chunk for chunk in sentence_chunks if chunk.kind == "reasoning"]
    if len(reasoning_chunks) <= max_chunks:
        return [
            AnalysisChunk(
                analysis_id=idx,
                sentence_start=chunk.chunk_id,
                sentence_end=chunk.chunk_id,
                text=chunk.text,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
            )
            for idx, chunk in enumerate(reasoning_chunks)
        ]

    # Place boundaries near equal cumulative word-count targets while retaining
    # sentence order. This prevents the final chunk from absorbing the tail.
    lengths = [max(1, len(chunk.text.split())) for chunk in reasoning_chunks]
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    boundaries = [0]
    start = 0
    n_sentences = len(reasoning_chunks)
    for group_index in range(1, max_chunks):
        minimum_end = start + 1
        maximum_end = n_sentences - (max_chunks - group_index)
        target = cumulative[-1] * group_index / max_chunks
        insertion = bisect_left(
            cumulative,
            target,
            lo=minimum_end,
            hi=maximum_end + 1,
        )
        candidates = {
            max(minimum_end, min(maximum_end, insertion)),
            max(minimum_end, min(maximum_end, insertion - 1)),
        }
        end = min(candidates, key=lambda idx: (abs(cumulative[idx] - target), idx))
        boundaries.append(end)
        start = end
    boundaries.append(n_sentences)
    groups = [
        reasoning_chunks[start:end]
        for start, end in zip(boundaries, boundaries[1:])
    ]

    packed: list[AnalysisChunk] = []
    for idx, group in enumerate(groups):
        start, end = group[0].char_start, group[-1].char_end
        text = raw_text[start:end] if raw_text is not None else "\n".join(chunk.text for chunk in group)
        packed.append(
            AnalysisChunk(
                analysis_id=idx,
                sentence_start=group[0].chunk_id,
                sentence_end=group[-1].chunk_id,
                text=text,
                char_start=start,
                char_end=end,
            )
        )
    return packed


def verify_doorkey_chunks(
    raw_text: str,
    sentence_chunks: Sequence[SentenceChunk],
    analysis_chunks: Sequence[AnalysisChunk],
    *,
    max_analysis_chunks: int = 32,
) -> list[tuple[Any, ...]]:
    errors: list[tuple[Any, ...]] = []
    for chunk in sentence_chunks:
        if not chunk.text.strip():
            errors.append(("empty_chunk", chunk.chunk_id))
        if raw_text[chunk.char_start : chunk.char_end] != chunk.text:
            errors.append(("span_text_mismatch", chunk.chunk_id))
        stripped = chunk.text.strip()
        if re.fullmatch(r"\d+\.", stripped):
            errors.append(("bad_numbered_split", chunk.chunk_id, chunk.text))
        if chunk.kind == "reasoning" and re.fullmatch(r"[\W_]+", stripped):
            errors.append(("punctuation_only_fragment", chunk.chunk_id, chunk.text))
    spans = sorted((chunk.char_start, chunk.char_end, chunk.chunk_id) for chunk in sentence_chunks)
    for (start_1, end_1, idx_1), (start_2, _end_2, idx_2) in zip(spans, spans[1:], strict=False):
        if end_1 > start_2:
            errors.append(("overlap", idx_1, idx_2))
    if len(analysis_chunks) > max_analysis_chunks:
        errors.append(("too_many_analysis_chunks", len(analysis_chunks)))
    for chunk in analysis_chunks:
        if raw_text[chunk.char_start : chunk.char_end] != chunk.text:
            errors.append(("analysis_span_text_mismatch", chunk.analysis_id))
    return errors


def segment_reasoning_trace(
    text: str,
    *,
    segmentation_mode: str = "paragraph_or_sentence",
    max_steps: int | None = None,
) -> list[ReasoningStep]:
    """Segment a reasoning trace into packed DoorKey analysis chunks.

    `paragraph_or_sentence` now follows the canonical DoorKey sentence-level
    splitter and returns an at-most-`max_steps` packed view over immutable
    sentence spans.
    """
    normalized = text.strip()
    if not normalized:
        return []

    if segmentation_mode == "newline":
        spans = _blank_line_spans(text)
        sentence_chunks = [
            SentenceChunk(
                trace_id="",
                chunk_id=idx,
                kind="reasoning",
                text=text[start:end],
                char_start=start,
                char_end=end,
                contains_action_json_mention=bool(find_action_json_spans(text[start:end])),
            )
            for idx, (start, end) in enumerate(spans)
        ]
        analysis_chunks = pack_to_analysis_chunks(
            sentence_chunks,
            max_chunks=max_steps if max_steps is not None and max_steps > 0 else len(sentence_chunks),
            raw_text=text,
        )
    elif segmentation_mode in {"sentence", "paragraph_or_sentence"}:
        sentence_chunks = chunk_doorkey_reasoning_trace(text)
        analysis_chunks = pack_to_analysis_chunks(
            sentence_chunks,
            max_chunks=max_steps if max_steps is not None and max_steps > 0 else len(sentence_chunks),
            raw_text=text,
        )
    else:
        raise ValueError(f"Unsupported segmentation_mode: {segmentation_mode}")

    steps: list[ReasoningStep] = []
    for idx, chunk in enumerate(analysis_chunks, start=1):
        if not chunk.text.strip():
            continue
        steps.append(
            ReasoningStep(
                step_idx=idx,
                start_char=chunk.char_start,
                end_char=chunk.char_end,
                text=chunk.text,
                segmentation_mode=segmentation_mode,
            )
        )
    return steps


def load_canonical_sentence_boundaries(
    path: str | Path,
) -> dict[tuple[str, int], list[dict[str, str]]]:
    """Load reasoning-only sentence spans keyed by trajectory and environment step."""
    boundary_path = Path(path)
    if not boundary_path.exists():
        raise FileNotFoundError(f"Canonical sentence boundary table not found: {boundary_path}")
    index: dict[tuple[str, int], list[dict[str, str]]] = {}
    with boundary_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("kind", "reasoning")) != "reasoning":
                continue
            trajectory_id = Path(str(row["path"])).stem
            key = (trajectory_id, int(row["step_id"]))
            index.setdefault(key, []).append(row)
    for rows in index.values():
        rows.sort(key=lambda row: int(row["sentence_id"]))
    return index


def canonical_sentence_steps(
    *,
    boundary_index: dict[tuple[str, int], list[dict[str, str]]],
    trajectory_id: str,
    step_index: int,
    analysis_text: str,
) -> list[ReasoningStep]:
    """Return sentence units only after exact span and text validation."""
    key = (trajectory_id, int(step_index))
    rows = boundary_index.get(key)
    if not rows:
        raise ValueError(
            "No canonical reasoning-sentence boundaries for "
            f"trajectory_id={trajectory_id!r}, step_index={step_index}."
        )
    steps: list[ReasoningStep] = []
    previous_end = -1
    seen_sentence_ids: set[int] = set()
    for unit_index, row in enumerate(rows, start=1):
        sentence_id = int(row["sentence_id"])
        start = int(row["char_start"])
        end = int(row["char_end"])
        if sentence_id in seen_sentence_ids:
            raise ValueError(f"Duplicate canonical sentence_id={sentence_id} for {key}.")
        if start < 0 or end <= start or end > len(analysis_text):
            raise ValueError(
                f"Invalid canonical sentence span {start}:{end} for {key}; "
                f"analysis length is {len(analysis_text)}."
            )
        if start < previous_end:
            raise ValueError(f"Overlapping canonical sentence spans for {key} at sentence {sentence_id}.")
        expected_text = str(row["text"])
        actual_text = analysis_text[start:end]
        if actual_text != expected_text:
            raise ValueError(
                f"Canonical sentence text mismatch for {key}, sentence_id={sentence_id}."
            )
        steps.append(
            ReasoningStep(
                step_idx=unit_index,
                start_char=start,
                end_char=end,
                text=actual_text,
                segmentation_mode="canonical_sentence",
                canonical_sentence_id=sentence_id,
            )
        )
        previous_end = end
        seen_sentence_ids.add(sentence_id)
    return steps


def _state_prefix_before_reasoning(*, grid_text: str, carrying_key: bool, prompt_preset: str) -> str:
    return (
        _behavioral_probe_preamble(prompt_preset)
        + "\n"
        + "Choose the next move that best advances toward the goal while respecting the DoorKey "
        + "mechanics in the current state.\n\n"
        + "The key is automatically picked up when the agent moves onto K. "
        + "A closed door D can only be opened if the agent already has the key.\n\n"
        + "# Inputs\n\n"
        + "Current grid state:\n\n"
        + grid_text
        + "\n\n"
        + f"Agent status:\n- Carrying key: {str(bool(carrying_key)).lower()}\n\n"
        + "Reasoning trace available so far:\n"
    )


def _state_suffix_after_reasoning() -> str:
    return (
        "\n\n"
        + 'Respond with exactly a JSON object of the form {"action": "<UP|DOWN|LEFT|RIGHT>"}.\n'
        + "Do not include any extra text before or after the JSON."
    )


def build_revealed_action_prompt(
    *,
    grid_text: str,
    carrying_key: bool,
    revealed_analysis: str,
    prompt_preset: str = DEFAULT_PROMPT_PRESET,
) -> str:
    return (
        _state_prefix_before_reasoning(
            grid_text=grid_text,
            carrying_key=carrying_key,
            prompt_preset=prompt_preset,
        )
        + revealed_analysis.strip()
        + _state_suffix_after_reasoning()
    )


def _render_original_trajectory_prompt(payload: dict[str, Any], step_payload: dict[str, Any]) -> str:
    template = payload.get("prompt", {}).get("prompt_template")
    if not isinstance(template, str):
        raise ValueError("Trajectory payload is missing prompt.prompt_template.")
    grid_state = step_payload.get("grid_state")
    if not isinstance(grid_state, list) or not all(isinstance(item, str) for item in grid_state):
        raise ValueError("Trajectory step is missing grid_state text.")
    return template.replace("{{grid_state}}", "\n".join(grid_state))


def _build_step_prefix_prompt_parts(
    *,
    grid_text: str,
    carrying_key: bool,
    prompt_preset: str,
) -> tuple[str, str]:
    return (
        _state_prefix_before_reasoning(
            grid_text=grid_text,
            carrying_key=carrying_key,
            prompt_preset=prompt_preset,
        ),
        _state_suffix_after_reasoning(),
    )


def _extract_final_action_from_output(output_text: str) -> str:
    match = re.search(r'"action"\s*:\s*"(UP|DOWN|LEFT|RIGHT)"', output_text)
    if match:
        return match.group(1)
    raise ValueError("Unable to recover final action from step output_text.")


def _normalize_action_label(label: str) -> str:
    normalized = str(label).strip().upper()
    return normalized if normalized in ACTION_LABELS else "INVALID"


def action_probabilities_from_samples(actions: Sequence[str]) -> dict[str, float]:
    valid_actions = [_normalize_action_label(action) for action in actions]
    valid_actions = [action for action in valid_actions if action in ACTION_LABELS]
    if not valid_actions:
        return {}
    counts = Counter(valid_actions)
    total = sum(counts.values())
    return {action: counts[action] / total for action in ACTION_LABELS if counts[action]}


def action_entropy_from_probabilities(probabilities: dict[str, float]) -> float | None:
    if not probabilities:
        return None
    entropy = 0.0
    for probability in probabilities.values():
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return entropy


def summarize_action_samples(actions: Sequence[str]) -> dict[str, Any]:
    normalized = [_normalize_action_label(action) for action in actions]
    valid = [action for action in normalized if action in ACTION_LABELS]
    probabilities = action_probabilities_from_samples(valid)
    return {
        "action_mc_actions_json": json.dumps(normalized),
        "action_mc_valid_actions_json": json.dumps(valid),
        "action_mc_probs_json": json.dumps(probabilities, sort_keys=True),
        "action_mc_entropy": action_entropy_from_probabilities(probabilities),
        "action_mc_valid_count": len(valid),
        "action_mc_invalid_count": len(normalized) - len(valid),
    }


def _load_drift_examples(
    *,
    candidate_rows_path: str,
    slice_mode: str,
    trajectory_slice_type: str,
    balanced_failure_rows_per_mode: int,
    non_failure_control_rows: int,
    max_examples: int | None,
    max_wall_hit: int,
    max_avoidable_detour: int,
    max_backtrack: int,
    max_baseline: int,
) -> list[DriftExample]:
    rows = load_wall_feature_patch_candidates(candidate_rows_path)
    if slice_mode == "focused_failure":
        selected_rows = select_wall_feature_patch_states(
            rows,
            max_wall_hit=max_wall_hit,
            max_avoidable_detour=max_avoidable_detour,
            max_backtrack=max_backtrack,
            max_baseline=max_baseline,
        )
        source_dataset = "focused_failure"
    elif slice_mode == "trajectory_candidates":
        if trajectory_slice_type == "balanced_failure_modes_with_controls":
            balanced_rows = _select_candidate_slice_rows(
                rows,
                slice_type="balanced_failure_modes",
                rows_per_mode=balanced_failure_rows_per_mode,
            )
            selected_keys = {
                (str(row.get("trajectory_id", "")), int(row.get("step_index") or 0))
                for row in balanced_rows
            }
            independent_controls = [
                row
                for row in _select_candidate_slice_rows(
                    rows,
                    slice_type="non_failure_controls",
                    rows_per_mode=balanced_failure_rows_per_mode,
                )
                if (str(row.get("trajectory_id", "")), int(row.get("step_index") or 0))
                not in selected_keys
            ][: max(0, int(non_failure_control_rows))]
            selected_rows = [*balanced_rows, *independent_controls]
        else:
            selected_rows = _select_candidate_slice_rows(
                rows,
                slice_type=trajectory_slice_type,
                rows_per_mode=balanced_failure_rows_per_mode,
            )
        source_dataset = f"trajectory_candidates:{trajectory_slice_type}"
    else:
        raise ValueError(f"Unsupported slice_mode: {slice_mode}")

    if max_examples is not None:
        selected_rows = selected_rows[: max(0, int(max_examples))]

    examples: list[DriftExample] = []
    for row in selected_rows:
        examples.append(
            DriftExample(
                example_id=str(row["example_id"]),
                trajectory_id=str(row["trajectory_id"]),
                step_index=int(row["step_index"]),
                grid_text=str(row["grid_text"]),
                carrying_key=bool(row.get("carrying_key") or False),
                observed_action=_normalize_action_label(str(row.get("observed_action", ""))),
                optimal_actions=tuple(_parse_json_list(row.get("optimal_actions_json", row.get("optimal_actions", [])))),
                is_optimal_action=row.get("is_optimal_action"),
                failure_category=str(row.get("primary_step_failure_mode", "none") or "none"),
                source_dataset=source_dataset,
                selection_stage=str(row.get("selection_stage", "")),
            )
        )
    return sorted(examples, key=lambda item: (item.trajectory_id, item.step_index, item.example_id))


def _last_stable_match_step(actions: Sequence[str], target: str) -> int | None:
    if not actions:
        return None
    for idx in range(len(actions)):
        if actions[idx] != target:
            continue
        if all(action == target for action in actions[idx:]):
            return idx
    return None


def _first_recovery_step(optimal_flags: Sequence[bool | None]) -> int | None:
    seen_non_optimal = False
    for idx, flag in enumerate(optimal_flags):
        if flag is False:
            seen_non_optimal = True
        elif seen_non_optimal and flag is True:
            return idx
    return None


def _first_wrong_turn_step_strict(optimal_flags: Sequence[bool | None]) -> int | None:
    if not optimal_flags:
        return None
    for idx in range(1, len(optimal_flags)):
        if optimal_flags[idx - 1] is True and optimal_flags[idx] is False:
            if all(flag is False for flag in optimal_flags[idx:]):
                return idx
    return None


def _first_wrong_turn_step_majority(
    optimal_flags: Sequence[bool | None],
    *,
    persistence_threshold: float,
) -> int | None:
    if not optimal_flags:
        return None
    for idx in range(1, len(optimal_flags)):
        if optimal_flags[idx - 1] is not True or optimal_flags[idx] is not False:
            continue
        tail = [flag for flag in optimal_flags[idx:] if flag is not None]
        if not tail:
            continue
        non_optimal_rate = sum(flag is False for flag in tail) / len(tail)
        if non_optimal_rate >= persistence_threshold:
            return idx
    return None


def summarize_prefix_behavior(
    prefix_evals: Sequence[PrefixStepEvaluation],
    *,
    final_action: str,
    persistence_threshold: float,
) -> dict[str, Any]:
    actions = [row.action_label for row in prefix_evals]
    optimal_flags = [row.action_is_optimal for row in prefix_evals]
    commitment_idx = _last_stable_match_step(actions, final_action)
    recovery_idx = _first_recovery_step(optimal_flags)
    wrong_turn_strict = _first_wrong_turn_step_strict(optimal_flags)
    wrong_turn_majority = _first_wrong_turn_step_majority(
        optimal_flags,
        persistence_threshold=persistence_threshold,
    )
    return {
        "n_reasoning_prefixes": len(prefix_evals),
        "n_invalid_prefixes": sum(flag is None for flag in optimal_flags),
        "valid_prefix_fraction": (
            sum(flag is not None for flag in optimal_flags) / len(optimal_flags)
            if optimal_flags else 0.0
        ),
        "trajectory_remains_optimal": bool(optimal_flags) and all(
            flag is True for flag in optimal_flags
        ),
        "commitment_step": commitment_idx,
        "recovery_step": recovery_idx,
        "wrong_turn_step_strict": wrong_turn_strict,
        "wrong_turn_step_majority": wrong_turn_majority,
        "wrong_turn_detected": wrong_turn_strict is not None or wrong_turn_majority is not None,
    }


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in a))
    norm_b = math.sqrt(sum(float(y) * float(y) for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return dot / (norm_a * norm_b)


def _subtract(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in a))


def compute_reasoning_geometry_metrics(
    vectors: Sequence[Sequence[float]],
    *,
    anchor_vectors_by_bucket: dict[int, Sequence[float]] | None = None,
    step_indices: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    if not vectors:
        return []
    resolved_step_indices = list(step_indices) if step_indices is not None else list(range(1, len(vectors) + 1))
    if len(resolved_step_indices) != len(vectors):
        raise ValueError("step_indices must have the same length as vectors.")
    rows: list[dict[str, Any]] = []
    drift_vector = _subtract(vectors[-1], vectors[0]) if len(vectors) >= 2 else [0.0 for _ in vectors[0]]
    cumulative_change = 0.0
    for position, (step_idx, vec) in enumerate(zip(resolved_step_indices, vectors)):
        net_change = _norm(_subtract(vec, vectors[0]))
        adjacent_cos = None
        aligned_change = None
        update_norm = None
        if position >= 1:
            update = _subtract(vec, vectors[position - 1])
            update_norm = _norm(update)
            cumulative_change += update_norm
            adjacent_cos = _cosine_similarity(vec, vectors[position - 1])
            aligned_change = _cosine_similarity(update, drift_vector)
        anchor_cos = None
        if anchor_vectors_by_bucket and step_idx in anchor_vectors_by_bucket:
            anchor_cos = _cosine_similarity(vec, anchor_vectors_by_bucket[step_idx])
        rows.append(
            {
                "reasoning_step_idx": step_idx,
                "net_change": net_change,
                "cumulative_change": cumulative_change,
                "adjacent_step_cosine": adjacent_cos,
                "aligned_change": aligned_change,
                "optimality_anchor_cosine": anchor_cos,
                "update_norm": update_norm,
            }
        )
    return rows


class LocalHiddenStateCollector:
    """Collect per-step hidden states from a local hookable HF causal LM."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cpu",
        device_map: str | None = None,
        torch_dtype: str = "auto",
        low_cpu_mem_usage: bool = True,
        forward_chunk_size: int = 256,
        trust_remote_code: bool = False,
    ) -> None:
        if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ModuleNotFoundError(
                "Per-step activation collection requires local torch + transformers."
            )
        if (device.startswith("cuda") or device_map in {"auto", "cuda"}) and not torch.cuda.is_available():
            dev_nodes = sorted(Path("/dev").glob("nvidia*"))
            node_text = ", ".join(str(path) for path in dev_nodes) if dev_nodes else "none"
            raise RuntimeError(
                "CUDA activation collection was requested, but PyTorch cannot see a CUDA GPU. "
                "This is a driver/runtime exposure problem rather than a model-size limit. "
                f"Visible NVIDIA device nodes: {node_text}. "
                "Check that the runtime exposes /dev/nvidia0, /dev/nvidiactl, and /dev/nvidia-uvm, "
                "or restart the instance/container with GPU passthrough enabled."
            )
        self.device = torch.device(device) if not device_map else None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": low_cpu_mem_usage,
        }
        if torch_dtype:
            load_kwargs["dtype"] = torch_dtype
        if device_map:
            load_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        if self.device is not None:
            self.model.to(self.device)
        self.model.eval()
        self.has_fast_offsets = bool(getattr(self.tokenizer, "is_fast", False))
        self.forward_chunk_size = max(1, int(forward_chunk_size))

    def collect(
        self,
        *,
        prompt_text: str,
        step_spans_in_prompt: Sequence[tuple[int, int, int]],
        layers: Sequence[int],
        output_dir: Path,
        record_prefix: str,
    ) -> list[dict[str, Any]]:
        encoded = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_offsets_mapping=self.has_fast_offsets,
            return_tensors="pt",
        )
        if self.device is not None:
            input_ids = encoded["input_ids"].to(self.device)
        else:
            first_param = next(self.model.parameters(), None)
            input_device = first_param.device if first_param is not None else torch.device("cpu")
            input_ids = encoded["input_ids"].to(input_device)
        offset_mapping = encoded.get("offset_mapping")
        if offset_mapping is not None:
            offset_mapping = offset_mapping[0].tolist()

        base_model = getattr(self.model, "model", self.model)
        decoder_layers = getattr(base_model, "layers", None)
        if decoder_layers is None:
            raise ValueError("Could not locate decoder layers on the local model.")
        requested_layers = sorted({int(layer) for layer in layers})
        for layer in requested_layers:
            if layer < 0 or layer >= len(decoder_layers):
                raise ValueError(f"Layer {layer} is out of range for {len(decoder_layers)} decoder layers.")

        captured: dict[int, list[Any]] = {layer: [] for layer in requested_layers}
        handles = []

        def _capture_layer(layer: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer].append(hidden.detach().cpu())

            return hook

        for layer in requested_layers:
            handles.append(decoder_layers[layer].register_forward_hook(_capture_layer(layer)))

        past_key_values = None
        try:
            with torch.no_grad():
                for start in range(0, input_ids.shape[1], self.forward_chunk_size):
                    chunk = input_ids[:, start : start + self.forward_chunk_size]
                    outputs = base_model(
                        input_ids=chunk,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_key_values = outputs.past_key_values
                    del outputs
        finally:
            for handle in handles:
                handle.remove()
        layer_hidden_states = {
            layer: torch.cat(chunks, dim=1)[0]
            for layer, chunks in captured.items()
            if chunks
        }
        if len(layer_hidden_states) != len(requested_layers):
            raise ValueError("Model hooks did not capture every requested layer.")

        activation_dir = output_dir / "activations"
        activation_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []

        def _token_span_from_chars(start_char: int, end_char: int) -> tuple[int, int]:
            if offset_mapping:
                covered = [
                    idx for idx, (tok_start, tok_end) in enumerate(offset_mapping)
                    if tok_end > start_char and tok_start < end_char
                ]
                if covered:
                    return covered[0], covered[-1]
            start_ids = self.tokenizer(prompt_text[:start_char], add_special_tokens=False)["input_ids"]
            end_ids = self.tokenizer(prompt_text[:end_char], add_special_tokens=False)["input_ids"]
            return len(start_ids), max(len(end_ids) - 1, len(start_ids))

        resolved_spans: list[tuple[int, int, int]] = []
        for reasoning_step_idx, start_char, end_char in sorted(
            step_spans_in_prompt,
            key=lambda item: (item[1], item[2], item[0]),
        ):
            token_start, token_end = _token_span_from_chars(start_char, end_char)
            if token_end < token_start:
                continue
            resolved_spans.append((reasoning_step_idx, token_start, token_end))
        # GPT-style tokens can combine leading whitespace with the first word
        # of the later sentence, causing adjacent character spans to claim the
        # same token. Assign shared boundary tokens to the later sentence so
        # each earlier span's last token remains semantically inside that span.
        for idx in range(len(resolved_spans) - 1):
            step_idx, token_start, token_end = resolved_spans[idx]
            next_start = resolved_spans[idx + 1][1]
            resolved_spans[idx] = (step_idx, token_start, min(token_end, next_start - 1))

        for reasoning_step_idx, token_start, token_end in resolved_spans:
            if token_end < token_start:
                continue
            for layer in layers:
                layer_tensor = layer_hidden_states[int(layer)]
                step_tensor = layer_tensor[token_start : token_end + 1]
                # Clone compact vectors so torch.save does not serialize the full
                # sequence storage backing these views.
                last_vec = step_tensor[-1].detach().cpu().clone()
                mean_vec = step_tensor.mean(dim=0).detach().cpu().clone()
                boundary_start = max(0, token_end - 2)
                boundary_window = layer_tensor[boundary_start : token_end + 1].detach().cpu().clone()
                step_dir = (
                    activation_dir
                    / _safe_relative_path(record_prefix)
                    / f"reasoning_step_{reasoning_step_idx:03d}"
                    / f"layer_{layer}"
                )
                step_dir.mkdir(parents=True, exist_ok=True)
                last_path = step_dir / "last_token.pt"
                mean_path = step_dir / "mean_pool.pt"
                boundary_window_path = step_dir / "boundary_window_3.pt"
                torch.save(last_vec, last_path)
                torch.save(mean_vec, mean_path)
                torch.save(boundary_window, boundary_window_path)
                rows.append(
                    {
                        "reasoning_step_idx": reasoning_step_idx,
                        "layer": layer,
                        "step_start_token": token_start,
                        "step_end_token": token_end,
                        "step_last_token_activation_path": str(last_path),
                        "step_mean_activation_path": str(mean_path),
                        "step_boundary_window_start_token": boundary_start,
                        "step_boundary_window_activation_path": str(boundary_window_path),
                        "step_boundary_window_n_tokens": int(boundary_window.shape[0]),
                    }
                )
        del past_key_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return rows


def _mean_vectors(vectors: Iterable[Sequence[float]]) -> list[float]:
    vectors = list(vectors)
    if not vectors:
        return []
    width = len(vectors[0])
    sums = [0.0] * width
    for vec in vectors:
        for idx, value in enumerate(vec):
            sums[idx] += float(value)
    return [value / len(vectors) for value in sums]


def _build_anchor_vectors(
    geometry_input_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
) -> dict[tuple[int, str, int], list[float]]:
    if torch is None:
        return {}
    by_key: dict[tuple[int, str, int], list[list[float]]] = {}
    optimal_example_ids = {
        str(row["example_id"])
        for row in trajectory_rows
        if row.get("trajectory_remains_optimal") is True
    }
    for row in geometry_input_rows:
        if str(row["example_id"]) not in optimal_example_ids:
            continue
        path = row["activation_path"]
        tensor = torch.load(path, map_location="cpu")
        if hasattr(tensor, "tolist"):
            vector = tensor.tolist()
        else:  # pragma: no cover
            vector = list(tensor)
        key = (
            int(row["layer"]),
            str(row["representation_kind"]),
            int(row["progress_bucket_10"]),
        )
        by_key.setdefault(key, []).append(vector)
    return {key: _mean_vectors(vectors) for key, vectors in by_key.items()}


def _plot_wrong_turn_histogram(rows: list[dict[str, Any]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    values = [int(row["wrong_turn_step_majority"]) for row in rows if row.get("wrong_turn_step_majority") not in {"", None}]
    if not values:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = range(1, max(values) + 2)
    ax.hist(values, bins=bins, color="#2B6CB0", edgecolor="white", align="left")
    ax.set_xlabel("Wrong-turn reasoning step")
    ax.set_ylabel("Trajectory count")
    ax.set_title("Wrong-turn step histogram")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_mean_aligned_change(rows: list[dict[str, Any]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    groups = {
        "stay_optimal": [row for row in rows if row.get("trajectory_group") == "stay_optimal"],
        "go_wrong": [row for row in rows if row.get("trajectory_group") == "go_wrong"],
        "recover": [row for row in rows if row.get("trajectory_group") == "recover"],
    }
    if not any(groups.values()):
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    palette = {
        "stay_optimal": "#1D4ED8",
        "go_wrong": "#60A5FA",
        "recover": "#93C5FD",
    }
    for label, subset in groups.items():
        by_step: dict[int, list[float]] = {}
        for row in subset:
            value = row.get("aligned_change")
            if value in {None, ""}:
                continue
            by_step.setdefault(int(row["reasoning_step_idx"]), []).append(float(value))
        if not by_step:
            continue
        xs = sorted(by_step)
        ys = [sum(by_step[idx]) / len(by_step[idx]) for idx in xs]
        ax.plot(xs, ys, marker="o", color=palette[label], label=label.replace("_", " "))
    ax.set_xlabel("Reasoning step")
    ax.set_ylabel("Mean aligned change")
    ax.set_title("Aligned change by reasoning step")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def regenerate_step_reasoning_activations(
    *,
    source_run_dir: str = "data/behavioral_probes/step_reasoning_drift_balanced_v1",
    output_dir: str = "data/behavioral_probes/step_reasoning_drift_balanced_v1_corrected_activations",
    local_model_name_or_path: str | None = None,
    device: str = "cpu",
    device_map: str | None = "auto",
    torch_dtype: str = "auto",
    low_cpu_mem_usage: bool = True,
    forward_chunk_size: int = 256,
    trust_remote_code: bool = False,
    resume: bool = True,
    verbose: bool = True,
) -> None:
    """Regenerate corrected activation spans from a completed drift run.

    This reuses the exact saved prompts and character spans, avoiding any new
    prefix-action API queries.
    """
    source_dir = Path(source_run_dir)
    out_dir = Path(output_dir)
    source_rows_path = source_dir / "step_activation_rows.csv"
    if not source_rows_path.exists():
        raise FileNotFoundError(f"Missing source activation index: {source_rows_path}")
    source_rows: list[dict[str, Any]] = []
    with source_rows_path.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError("Source activation index is empty.")
    if local_model_name_or_path is None:
        local_model_name_or_path = str(source_rows[0]["local_model_name_or_path"])

    collector = LocalHiddenStateCollector(
        local_model_name_or_path,
        device=device,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        forward_chunk_size=forward_chunk_size,
        trust_remote_code=trust_remote_code,
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in source_rows:
        grouped.setdefault(str(row["example_id"]), []).append(row)
    checkpoint_path = out_dir / "completed_examples.jsonl"
    completed: set[str] = set()
    corrected_rows: list[dict[str, Any]] = []
    if resume and checkpoint_path.exists():
        for line in checkpoint_path.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            completed.add(str(payload["example_id"]))
            corrected_rows.extend(payload["activation_rows"])
    elif checkpoint_path.exists():
        checkpoint_path.write_text("")

    out_dir.mkdir(parents=True, exist_ok=True)
    for example_idx, (example_id, rows) in enumerate(sorted(grouped.items()), start=1):
        if example_id in completed:
            if verbose:
                print(f"Corrected activation example already completed: {example_id}")
            continue
        if verbose:
            print(f"Corrected activation example {example_idx}/{len(grouped)}: {example_id}")
        template = rows[0]
        prompt_path = Path(str(template["prompt_text_path"]))
        if not prompt_path.is_absolute():
            prompt_path = Path(prompt_path)
        prompt_text = prompt_path.read_text()
        layers = sorted({int(row["layer"]) for row in rows})
        span_by_step: dict[int, tuple[int, int, int]] = {}
        metadata_by_key: dict[tuple[int, int], dict[str, Any]] = {}
        for row in rows:
            reasoning_step_idx = int(row["reasoning_step_idx"])
            span_by_step[reasoning_step_idx] = (
                reasoning_step_idx,
                int(row["step_start_char_in_prompt"]),
                int(row["step_end_char_in_prompt"]),
            )
            metadata_by_key[(reasoning_step_idx, int(row["layer"]))] = row
        collected = collector.collect(
            prompt_text=prompt_text,
            step_spans_in_prompt=list(span_by_step.values()),
            layers=layers,
            output_dir=out_dir,
            record_prefix=f"{template['trajectory_id']}/step_{int(template['step_index']):03d}",
        )
        example_rows: list[dict[str, Any]] = []
        for result in collected:
            key = (int(result["reasoning_step_idx"]), int(result["layer"]))
            source = metadata_by_key[key]
            example_rows.append(
                {
                    **source,
                    **result,
                    "activation_schema_version": "step_reasoning_drift_v1_nonoverlap",
                    "local_model_name_or_path": local_model_name_or_path,
                }
            )
        corrected_rows.extend(example_rows)
        with checkpoint_path.open("a") as handle:
            handle.write(json.dumps({"example_id": example_id, "activation_rows": example_rows}) + "\n")

    _write_csv(out_dir / "step_activation_rows.csv", corrected_rows)
    overlaps = 0
    by_example_layer: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in corrected_rows:
        by_example_layer.setdefault((str(row["example_id"]), int(row["layer"])), []).append(row)
    for rows in by_example_layer.values():
        ordered = sorted(rows, key=lambda row: int(row["reasoning_step_idx"]))
        overlaps += sum(
            int(current["step_start_token"]) <= int(previous["step_end_token"])
            for previous, current in zip(ordered, ordered[1:])
        )
    _write_json(
        out_dir / "manifest.json",
        {
            "status": "completed",
            "source_run_dir": source_run_dir,
            "local_model_name_or_path": local_model_name_or_path,
            "n_examples": len(grouped),
            "n_activation_rows": len(corrected_rows),
            "layers": sorted({int(row["layer"]) for row in corrected_rows}),
            "adjacent_span_overlaps": overlaps,
            "activation_spans_valid": overlaps == 0,
            "behavioral_queries_rerun": False,
        },
    )
    rebuild_reasoning_geometry_from_activations(
        source_run_dir=source_run_dir,
        activation_run_dir=output_dir,
    )


def rebuild_reasoning_geometry_from_activations(
    *,
    source_run_dir: str = "data/behavioral_probes/step_reasoning_drift_balanced_v1",
    activation_run_dir: str = "data/behavioral_probes/step_reasoning_drift_balanced_v1_corrected_activations",
) -> None:
    """Recompute geometry metrics from a corrected activation index."""
    if torch is None:
        raise ModuleNotFoundError("Geometry rebuilding requires torch.")
    source_dir = Path(source_run_dir)
    activation_dir = Path(activation_run_dir)
    with (activation_dir / "step_activation_rows.csv").open(newline="") as handle:
        activation_rows = list(csv.DictReader(handle))
    with (source_dir / "trajectory_wrong_turn_summary.csv").open(newline="") as handle:
        trajectory_rows = list(csv.DictReader(handle))
    geometry_inputs: list[dict[str, Any]] = []
    for row in activation_rows:
        for representation_kind, path_key in (
            ("last_token", "step_last_token_activation_path"),
            ("mean_pool", "step_mean_activation_path"),
        ):
            geometry_inputs.append(
                {
                    **row,
                    "layer": int(row["layer"]),
                    "reasoning_step_idx": int(row["reasoning_step_idx"]),
                    "progress_bucket_10": int(row["progress_bucket_10"]),
                    "representation_kind": representation_kind,
                    "activation_path": row[path_key],
                }
            )
    normalized_trajectories = [
        {
            **row,
            "trajectory_remains_optimal": str(row.get("trajectory_remains_optimal", "")).lower() == "true",
        }
        for row in trajectory_rows
    ]
    anchor_vectors = _build_anchor_vectors(geometry_inputs, normalized_trajectories)
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in geometry_inputs:
        grouped.setdefault(
            (str(row["example_id"]), int(row["layer"]), str(row["representation_kind"])),
            [],
        ).append(row)
    geometry_rows: list[dict[str, Any]] = []
    for (example_id, layer, representation_kind), rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["reasoning_step_idx"]))
        vectors = [
            torch.load(row["activation_path"], map_location="cpu").tolist()
            for row in ordered
        ]
        anchors = {
            int(row["reasoning_step_idx"]): anchor_vectors[
                (layer, representation_kind, int(row["progress_bucket_10"]))
            ]
            for row in ordered
            if (layer, representation_kind, int(row["progress_bucket_10"])) in anchor_vectors
        }
        metrics = compute_reasoning_geometry_metrics(
            vectors,
            anchor_vectors_by_bucket=anchors,
            step_indices=[int(row["reasoning_step_idx"]) for row in ordered],
        )
        for row, metric in zip(ordered, metrics):
            geometry_rows.append(
                {
                    "example_id": example_id,
                    "trajectory_id": row["trajectory_id"],
                    "step_index": row["step_index"],
                    "failure_category": row["failure_category"],
                    "trajectory_group": row["trajectory_group"],
                    "layer": layer,
                    "representation_kind": representation_kind,
                    "reasoning_progress": row["reasoning_progress"],
                    "progress_bucket_10": row["progress_bucket_10"],
                    "activation_path": row["activation_path"],
                    **metric,
                }
            )
    _write_csv(activation_dir / "geometry_rows.csv", geometry_rows)
    for filename in (
        "prefix_action_rows.csv",
        "trajectory_wrong_turn_summary.csv",
        "state_outcome_accounting.csv",
        "planner_optimality_audit.csv",
    ):
        source_path = source_dir / filename
        if source_path.exists():
            with source_path.open(newline="") as handle:
                _write_csv(activation_dir / filename, list(csv.DictReader(handle)))
    state_outcomes = activation_dir / "state_outcome_accounting.csv"
    if state_outcomes.exists():
        from reveng.experiments.step_reasoning_drift_class_plot import build_class_geometry_figure

        build_class_geometry_figure(run_dir=activation_dir)


def run_step_reasoning_drift_experiment(
    *,
    candidate_rows_path: str = DEFAULT_CANDIDATE_ROWS_PATH,
    trajectory_dir: str = DEFAULT_TRAJECTORY_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    slice_mode: str = "focused_failure",
    trajectory_slice_type: str = "short_loop",
    balanced_failure_rows_per_mode: int = 2,
    non_failure_control_rows: int = 8,
    max_examples: int | None = None,
    max_wall_hit: int = 4,
    max_avoidable_detour: int = 4,
    max_backtrack: int = 2,
    max_baseline: int = 4,
    prompt_preset: str = DEFAULT_PROMPT_PRESET,
    segmentation_mode: str = "paragraph_or_sentence",
    max_reasoning_steps: int | None = None,
    analysis_unit: str = "packed_chunk",
    sentence_boundaries_path: str = DEFAULT_SENTENCE_BOUNDARIES_PATH,
    persistence_threshold: float = 0.5,
    collect_activations: bool = False,
    activation_source: str = "local",
    activation_prompt_mode: str = "revealed_prompt",
    local_model_name_or_path: str | None = None,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
    device: str = "cpu",
    device_map: str | None = None,
    torch_dtype: str = "auto",
    low_cpu_mem_usage: bool = True,
    forward_chunk_size: int = 256,
    trust_remote_code: bool = False,
    resume: bool = True,
    verbose: bool = True,
    action_query_fn: Callable[[str, int], tuple[str, str, dict[str, Any]]] | None = None,
    action_mc_sample_repeats: int = 0,
    action_mc_temperature: float = 0.7,
    action_mc_max_workers: int = 1,
    action_mc_query_fn: Callable[[str, int], tuple[str, str, dict[str, Any]]] | None = None,
    action_logprob_temperature: float | None = None,
    action_top_logprobs: int = 20,
    max_prefix_positions_per_example: int | None = None,
    action_logprob_max_workers: int = 1,
) -> None:
    if analysis_unit not in {"packed_chunk", "sentence"}:
        raise ValueError("analysis_unit must be 'packed_chunk' or 'sentence'.")
    if analysis_unit == "sentence" and max_reasoning_steps is not None:
        raise ValueError(
            "Sentence analysis uses every canonical sentence boundary; "
            "max_reasoning_steps must be None."
        )
    if activation_source != "local":
        raise ValueError(
            "Per-step mode does not support the released public activation artifacts: "
            "those artifacts only provide pre/post reasoning snapshots, not step-indexed hidden states. "
            "Use activation_source='local' with a hookable local model."
        )
    if collect_activations and not local_model_name_or_path:
        raise ValueError("collect_activations=True requires local_model_name_or_path.")
    if activation_prompt_mode not in {"revealed_prompt", "original_trace"}:
        raise ValueError(f"Unsupported activation_prompt_mode: {activation_prompt_mode}")
    if action_logprob_temperature is not None and action_mc_sample_repeats:
        raise ValueError(
            "Action logprob entropy and MC action sampling are separate protocols; "
            "set action_mc_sample_repeats=0 for a logprob run."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_lock_handle = (out_dir / ".prefix_action_run.lock").open("a+")
    try:
        fcntl.flock(run_lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        run_lock_handle.close()
        raise RuntimeError(
            f"Another prefix-action runner is already using {out_dir}."
        ) from exc
    trajectory_root = Path(trajectory_dir)
    sentence_boundary_index = (
        load_canonical_sentence_boundaries(sentence_boundaries_path)
        if analysis_unit == "sentence"
        else None
    )
    examples = _load_drift_examples(
        candidate_rows_path=candidate_rows_path,
        slice_mode=slice_mode,
        trajectory_slice_type=trajectory_slice_type,
        balanced_failure_rows_per_mode=balanced_failure_rows_per_mode,
        non_failure_control_rows=non_failure_control_rows,
        max_examples=max_examples,
        max_wall_hit=max_wall_hit,
        max_avoidable_detour=max_avoidable_detour,
        max_backtrack=max_backtrack,
        max_baseline=max_baseline,
    )

    usage_summary: dict[str, Any] = {
        "model_name": model_name,
        "by_phase": {},
        "total": _empty_usage_bucket(),
    }
    client: BehavioralProbeLLM | None = None
    if action_query_fn is None:
        client = BehavioralProbeLLM(model_name=model_name, temperature=0.0)

        def action_query_fn(prompt: str, seed: int) -> tuple[str, str, dict[str, Any]]:
            assert client is not None
            return _query_action(client, prompt, seed=seed)
    action_logprob_client: BehavioralProbeLLM | None = None
    if action_logprob_temperature is not None:
        action_logprob_client = BehavioralProbeLLM(
            model_name=model_name,
            temperature=action_logprob_temperature,
        )
    mc_client: BehavioralProbeLLM | None = None
    if action_mc_sample_repeats > 0 and action_mc_query_fn is None:
        mc_client = BehavioralProbeLLM(model_name=model_name, temperature=action_mc_temperature)

        def action_mc_query_fn(prompt: str, seed: int) -> tuple[str, str, dict[str, Any]]:
            assert mc_client is not None
            return _query_action(mc_client, prompt, seed=seed)

    collector: LocalHiddenStateCollector | None = None
    if collect_activations:
        collector = LocalHiddenStateCollector(
            str(local_model_name_or_path),
            device=device,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            forward_chunk_size=forward_chunk_size,
            trust_remote_code=trust_remote_code,
        )

    step_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    activation_index_rows: list[dict[str, Any]] = []
    geometry_input_rows: list[dict[str, Any]] = []
    checkpoint_path = out_dir / "example_checkpoints.jsonl"
    prefix_checkpoint_path = out_dir / "prefix_query_checkpoints.jsonl"
    prefix_query_lookup: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    if resume and prefix_checkpoint_path.exists():
        for line in prefix_checkpoint_path.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            prefix_query_lookup[
                (
                    str(payload["example_id"]),
                    str(payload.get("analysis_unit", "packed_chunk")),
                    int(payload["reasoning_step_idx"]),
                    str(
                        payload.get(
                            "checkpoint_identity_sha256", payload["prompt_sha256"]
                        )
                    ),
                )
            ] = payload
    completed_example_ids: set[str] = set()
    if resume and checkpoint_path.exists():
        checkpoints: dict[str, dict[str, Any]] = {}
        for line in checkpoint_path.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            checkpoints[str(payload["example_id"])] = payload
        for example_id, payload in checkpoints.items():
            completed_example_ids.add(example_id)
            restored_step_rows = payload.get("step_rows", [])
            step_rows.extend(restored_step_rows)
            restored_trajectory_row = payload["trajectory_row"]
            if restored_step_rows:
                restored_prefix_evals = [
                    PrefixStepEvaluation(
                        reasoning_step_idx=int(row["reasoning_step_idx"]),
                        revealed_step_count=int(row["revealed_step_count"]),
                        revealed_analysis_chars=int(row["revealed_analysis_chars"]),
                        action_label=str(row["action_label"]),
                        action_is_optimal=row.get("action_is_optimal"),
                        action_matches_final=row.get("action_matches_final"),
                        revealed_analysis_text=str(row.get("revealed_analysis_text", "")),
                    )
                    for row in restored_step_rows
                ]
                restored_summary = summarize_prefix_behavior(
                    restored_prefix_evals,
                    final_action=str(restored_trajectory_row["final_action"]),
                    persistence_threshold=persistence_threshold,
                )
                restored_trajectory_row.update(restored_summary)
                n_reasoning_steps = int(restored_trajectory_row["n_reasoning_steps"])
                for key in ("wrong_turn_step_strict", "wrong_turn_step_majority", "commitment_step", "recovery_step"):
                    value = restored_trajectory_row.get(key)
                    restored_trajectory_row[f"{key}_progress"] = (
                        float(value) / n_reasoning_steps
                        if value not in {None, ""} and n_reasoning_steps
                        else ""
                    )
                restored_trajectory_row["trajectory_group"] = (
                    "stay_optimal" if restored_summary["trajectory_remains_optimal"]
                    else "recover" if restored_summary["recovery_step"] is not None
                    else "go_wrong"
                )
            trajectory_rows.append(restored_trajectory_row)
            restored_activation_rows = payload.get("activation_rows", [])
            activation_index_rows.extend(restored_activation_rows)
            for row in restored_activation_rows:
                geometry_input_rows.append(
                    {
                        **row,
                        "representation_kind": "last_token",
                        "activation_path": row["step_last_token_activation_path"],
                    }
                )
                geometry_input_rows.append(
                    {
                        **row,
                        "representation_kind": "mean_pool",
                        "activation_path": row["step_mean_activation_path"],
                    }
                )
            usage_summary = payload.get("usage_summary", usage_summary)

    for example in examples:
        if example.example_id in completed_example_ids:
            if verbose:
                print(f"Step-drift example already checkpointed: {example.example_id}")
            continue
        if verbose:
            print(f"Step-drift example: {example.example_id}")
        step_row_start = len(step_rows)
        activation_row_start = len(activation_index_rows)
        payload = _load_local_trajectory_payload(example.trajectory_id, trajectory_root)
        step_payload = payload["steps"][example.step_index]
        analysis_text, _final_text = _extract_analysis_and_final(step_payload["output_text"])
        final_action = _extract_final_action_from_output(step_payload["output_text"])
        if analysis_unit == "sentence":
            assert sentence_boundary_index is not None
            reasoning_steps = canonical_sentence_steps(
                boundary_index=sentence_boundary_index,
                trajectory_id=example.trajectory_id,
                step_index=example.step_index,
                analysis_text=analysis_text,
            )
        else:
            reasoning_steps = segment_reasoning_trace(
                analysis_text,
                segmentation_mode=segmentation_mode,
                max_steps=max_reasoning_steps,
            )
        prefix_evals: list[PrefixStepEvaluation] = []
        before_reasoning, after_reasoning = _build_step_prefix_prompt_parts(
            grid_text=example.grid_text,
            carrying_key=example.carrying_key,
            prompt_preset=prompt_preset,
        )
        prompt_spans: list[tuple[int, int, int]] = []

        n_prefix_positions = len(reasoning_steps) + 1
        if max_prefix_positions_per_example is not None:
            n_prefix_positions = min(
                n_prefix_positions, max(0, int(max_prefix_positions_per_example))
            )

        def make_prefix_spec(revealed_count: int) -> dict[str, Any]:
            boundary = reasoning_steps[revealed_count - 1] if revealed_count > 0 else None
            revealed = analysis_text[: boundary.end_char] if boundary is not None else ""
            rendered_prompt = before_reasoning + revealed.strip() + after_reasoning
            rendered_prompt_sha256 = hashlib.sha256(rendered_prompt.encode()).hexdigest()
            identity_payload = {
                "prompt_sha256": rendered_prompt_sha256,
                "model_name": model_name,
                "action_logprob_temperature": action_logprob_temperature,
                "action_top_logprobs": action_top_logprobs,
                "candidate_labels": list(ACTION_LABELS),
            }
            identity_sha256 = (
                hashlib.sha256(
                    json.dumps(identity_payload, sort_keys=True).encode()
                ).hexdigest()
                if action_logprob_temperature is not None
                else rendered_prompt_sha256
            )
            return {
                "revealed_count": revealed_count,
                "boundary_step": boundary,
                "revealed_text": revealed,
                "prompt": rendered_prompt,
                "prompt_sha256": rendered_prompt_sha256,
                "checkpoint_identity_payload": identity_payload,
                "checkpoint_identity_sha256": identity_sha256,
            }

        prefix_specs = [make_prefix_spec(idx) for idx in range(n_prefix_positions)]
        if action_logprob_temperature is not None and action_logprob_max_workers > 1:
            pending_specs = [
                spec
                for spec in prefix_specs
                if (
                    example.example_id,
                    analysis_unit,
                    spec["revealed_count"],
                    spec["checkpoint_identity_sha256"],
                )
                not in prefix_query_lookup
            ]
            worker_local = threading.local()

            def query_logprob_prefix(spec: dict[str, Any]) -> dict[str, Any]:
                worker_client = getattr(worker_local, "client", None)
                if worker_client is None:
                    worker_client = BehavioralProbeLLM(
                        model_name=model_name,
                        temperature=float(action_logprob_temperature),
                    )
                    worker_local.client = worker_client
                readout = _collect_action4_logprob_readout(
                    client=worker_client,
                    prompt=spec["prompt"],
                    seed=0,
                    top_logprobs=action_top_logprobs,
                )
                fields = {
                    "action_visible_sample": readout["visible_action"],
                    "action_logprob_probs_json": json.dumps(
                        readout["probabilities"], sort_keys=True
                    ),
                    "action_logprob_entropy_bits": readout["entropy"],
                    "action_logprob_candidate_probability_mass": readout[
                        "candidate_probability_mass"
                    ],
                    "action_logprob_top_probability_mass": readout[
                        "top_logprob_probability_mass"
                    ],
                    "action_logprob_excluded_probability_mass_lower_bound": readout[
                        "excluded_probability_mass_lower_bound"
                    ],
                    "action_logprob_missing_candidates_json": json.dumps(
                        readout["missing_candidates"]
                    ),
                    "action_logprob_raw_top_logprobs_json": json.dumps(
                        readout["raw_top_logprobs"], sort_keys=True
                    ),
                    "action_logprob_all_candidates_present": not readout[
                        "missing_candidates"
                    ],
                }
                boundary = spec["boundary_step"]
                return {
                    "example_id": example.example_id,
                    "analysis_unit": analysis_unit,
                    "reasoning_step_idx": spec["revealed_count"],
                    "analysis_unit_char_end": boundary.end_char if boundary else 0,
                    "canonical_sentence_id": (
                        boundary.canonical_sentence_id if boundary else None
                    ),
                    "prompt_sha256": spec["prompt_sha256"],
                    "checkpoint_identity_sha256": spec[
                        "checkpoint_identity_sha256"
                    ],
                    "checkpoint_identity_payload": spec[
                        "checkpoint_identity_payload"
                    ],
                    "action_label": readout["recommended_action"],
                    "raw_text": readout["raw_output"],
                    "usage": readout["usage"],
                    "action_mc_actions": [],
                    "action_mc_raw_outputs": [],
                    "action_mc_sample_repeats": 0,
                    "action_mc_temperature": action_mc_temperature,
                    "action_mc_max_workers": action_mc_max_workers,
                    "action_logprob_fields": fields,
                }

            if pending_specs and verbose:
                print(
                    f"Running {len(pending_specs)} pending action-logprob prefixes "
                    f"with max_workers={action_logprob_max_workers}"
                )
            with ThreadPoolExecutor(
                max_workers=max(1, int(action_logprob_max_workers))
            ) as pool, prefix_checkpoint_path.open("a") as checkpoint_file:
                futures = {
                    pool.submit(query_logprob_prefix, spec): spec for spec in pending_specs
                }
                for future in as_completed(futures):
                    spec = futures[future]
                    try:
                        checkpoint_payload = future.result()
                    except Exception as exc:
                        if verbose:
                            print(
                                "Action-logprob prefetch failed and will be retried "
                                f"serially: {example.example_id} "
                                f"step={spec['revealed_count']}: {type(exc).__name__}: {exc}"
                            )
                        continue
                    checkpoint_file.write(
                        json.dumps(checkpoint_payload, sort_keys=True) + "\n"
                    )
                    checkpoint_file.flush()
                    prefix_query_lookup[
                        (
                            example.example_id,
                            analysis_unit,
                            int(spec["revealed_count"]),
                            str(spec["checkpoint_identity_sha256"]),
                        )
                    ] = checkpoint_payload

        for spec in prefix_specs:
            revealed_count = int(spec["revealed_count"])
            boundary_step = spec["boundary_step"]
            revealed_text = str(spec["revealed_text"])
            prompt = str(spec["prompt"])
            prompt_sha256 = str(spec["prompt_sha256"])
            checkpoint_identity_payload = dict(spec["checkpoint_identity_payload"])
            checkpoint_identity_sha256 = str(spec["checkpoint_identity_sha256"])
            restored_query = prefix_query_lookup.get(
                (
                    example.example_id,
                    analysis_unit,
                    revealed_count,
                    checkpoint_identity_sha256,
                )
            )
            query_error = ""
            action_logprob_fields: dict[str, Any] = {}
            if restored_query is not None:
                action_label = str(restored_query["action_label"])
                raw_text = str(restored_query["raw_text"])
                usage = dict(restored_query.get("usage", _empty_usage_bucket()))
                mc_actions = list(restored_query.get("action_mc_actions", []))
                mc_raw_outputs = list(restored_query.get("action_mc_raw_outputs", []))
                action_logprob_fields = dict(
                    restored_query.get("action_logprob_fields", {})
                )
                if verbose:
                    print(
                        f"Prefix query already checkpointed: {example.example_id} "
                        f"reasoning_step_idx={revealed_count}"
                    )
            else:
                mc_actions = []
                mc_raw_outputs = []
                try:
                    if action_logprob_temperature is not None:
                        assert action_logprob_client is not None
                        readout = _collect_action4_logprob_readout(
                            client=action_logprob_client,
                            prompt=prompt,
                            seed=0,
                            top_logprobs=action_top_logprobs,
                        )
                        action_label = str(readout["recommended_action"])
                        raw_text = str(readout["raw_output"])
                        usage = dict(readout["usage"])
                        action_logprob_fields = {
                            "action_visible_sample": readout["visible_action"],
                            "action_logprob_probs_json": json.dumps(
                                readout["probabilities"], sort_keys=True
                            ),
                            "action_logprob_entropy_bits": readout["entropy"],
                            "action_logprob_candidate_probability_mass": readout[
                                "candidate_probability_mass"
                            ],
                            "action_logprob_top_probability_mass": readout[
                                "top_logprob_probability_mass"
                            ],
                            "action_logprob_excluded_probability_mass_lower_bound": readout[
                                "excluded_probability_mass_lower_bound"
                            ],
                            "action_logprob_missing_candidates_json": json.dumps(
                                readout["missing_candidates"]
                            ),
                            "action_logprob_raw_top_logprobs_json": json.dumps(
                                readout["raw_top_logprobs"], sort_keys=True
                            ),
                            "action_logprob_all_candidates_present": not readout[
                                "missing_candidates"
                            ],
                        }
                    else:
                        action_label, raw_text, usage = action_query_fn(prompt, 0)
                except Exception as exc:
                    action_label = "INVALID"
                    raw_text = ""
                    usage = _empty_usage_bucket()
                    query_error = f"{type(exc).__name__}: {exc}"
                    if verbose:
                        print(
                            f"Prefix query failed after retries: {example.example_id} "
                            f"reasoning_step_idx={revealed_count}: {query_error}"
                        )
            mc_query_error = ""
            if not query_error and action_mc_sample_repeats > 0 and len(mc_actions) < action_mc_sample_repeats:
                def run_mc_sample(sample_idx: int) -> tuple[int, str, str, dict[str, Any], str]:
                    try:
                        assert action_mc_query_fn is not None
                        mc_action, mc_raw_text, mc_usage = action_mc_query_fn(prompt, sample_idx)
                        return sample_idx, mc_action, mc_raw_text, mc_usage, ""
                    except Exception as exc:
                        return sample_idx, "INVALID", "", _empty_usage_bucket(), f"{type(exc).__name__}: {exc}"

                mc_results: list[tuple[int, str, str, dict[str, Any], str]] = []
                max_workers = max(1, min(int(action_mc_max_workers), int(action_mc_sample_repeats)))
                if max_workers == 1:
                    mc_results = [run_mc_sample(sample_idx) for sample_idx in range(action_mc_sample_repeats)]
                else:
                    with ThreadPoolExecutor(max_workers=max_workers) as pool:
                        future_to_idx = {
                            pool.submit(run_mc_sample, sample_idx): sample_idx
                            for sample_idx in range(action_mc_sample_repeats)
                        }
                        for future in as_completed(future_to_idx):
                            mc_results.append(future.result())
                mc_actions = []
                mc_raw_outputs = []
                for _sample_idx, mc_action, mc_raw_text, mc_usage, sample_error in sorted(mc_results, key=lambda item: item[0]):
                    if sample_error and not mc_query_error:
                        mc_query_error = sample_error
                    mc_actions.append(_normalize_action_label(mc_action))
                    mc_raw_outputs.append(mc_raw_text)
                    mc_usage_bucket = usage_summary["by_phase"].setdefault("prefix_action_mc", _empty_usage_bucket())
                    mc_usage_bucket["requests"] += 1
                    usage_summary["total"]["requests"] += 1
                    for key in ("cost_usd", "prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens", "retry_count", "requests_with_retry", "retry_sleep_seconds"):
                        value = mc_usage.get(key, 0)
                        mc_usage_bucket[key] += value
                        usage_summary["total"][key] += value
            action_mc_summary = summarize_action_samples(mc_actions)
            if restored_query is None and not query_error:
                checkpoint_payload = {
                    "example_id": example.example_id,
                    "analysis_unit": analysis_unit,
                    "reasoning_step_idx": revealed_count,
                    "analysis_unit_char_end": boundary_step.end_char if boundary_step else 0,
                    "canonical_sentence_id": (
                        boundary_step.canonical_sentence_id if boundary_step else None
                    ),
                    "prompt_sha256": prompt_sha256,
                    "checkpoint_identity_sha256": checkpoint_identity_sha256,
                    "checkpoint_identity_payload": checkpoint_identity_payload,
                    "action_label": action_label,
                    "raw_text": raw_text,
                    "usage": usage,
                    "action_mc_actions": mc_actions,
                    "action_mc_raw_outputs": mc_raw_outputs,
                    "action_mc_sample_repeats": action_mc_sample_repeats,
                    "action_mc_temperature": action_mc_temperature,
                    "action_mc_max_workers": action_mc_max_workers,
                    "action_logprob_fields": action_logprob_fields,
                }
                with prefix_checkpoint_path.open("a") as prefix_checkpoint_file:
                    prefix_checkpoint_file.write(
                        json.dumps(checkpoint_payload, sort_keys=True) + "\n"
                    )
                prefix_query_lookup[
                    (
                        example.example_id,
                        analysis_unit,
                        revealed_count,
                        checkpoint_identity_sha256,
                    )
                ] = checkpoint_payload
            action_label = _normalize_action_label(action_label)
            prefix_eval = PrefixStepEvaluation(
                reasoning_step_idx=revealed_count,
                revealed_step_count=revealed_count,
                revealed_analysis_chars=len(revealed_text),
                action_label=action_label,
                action_is_optimal=(action_label in example.optimal_actions) if action_label in ACTION_LABELS else None,
                action_matches_final=(action_label == final_action) if action_label in ACTION_LABELS else None,
                revealed_analysis_text=revealed_text,
            )
            prefix_evals.append(prefix_eval)
            usage_bucket = usage_summary["by_phase"].setdefault("prefix_action", _empty_usage_bucket())
            usage_bucket["requests"] += 1
            usage_summary["total"]["requests"] += 1
            for key in ("cost_usd", "prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens", "retry_count", "requests_with_retry", "retry_sleep_seconds"):
                value = usage.get(key, 0)
                usage_bucket[key] += value
                usage_summary["total"][key] += value
            step_rows.append(
                {
                    "example_id": example.example_id,
                    "trajectory_id": example.trajectory_id,
                    "step_index": example.step_index,
                    "failure_category": example.failure_category,
                    "selection_stage": example.selection_stage,
                    "source_dataset": example.source_dataset,
                    "final_action": final_action,
                    "observed_action": example.observed_action,
                    "optimal_actions_json": json.dumps(example.optimal_actions),
                    "analysis_unit": analysis_unit,
                    "analysis_unit_index": revealed_count,
                    "canonical_sentence_id": (
                        boundary_step.canonical_sentence_id if boundary_step else ""
                    ),
                    "analysis_unit_char_start": boundary_step.start_char if boundary_step else 0,
                    "analysis_unit_char_end": boundary_step.end_char if boundary_step else 0,
                    "reasoning_step_idx": revealed_count,
                    "reasoning_progress": (
                        revealed_count / len(reasoning_steps) if reasoning_steps else 0.0
                    ),
                    "revealed_step_count": revealed_count,
                    "revealed_analysis_chars": len(revealed_text),
                    "reasoning_character_progress": (
                        len(revealed_text) / len(analysis_text) if analysis_text else 0.0
                    ),
                    "action_label": action_label,
                    "action_is_optimal": prefix_eval.action_is_optimal,
                    "action_matches_final": prefix_eval.action_matches_final,
                    "action_logprob_temperature": (
                        action_logprob_temperature
                        if action_logprob_temperature is not None
                        else ""
                    ),
                    "action_top_logprobs": (
                        action_top_logprobs
                        if action_logprob_temperature is not None
                        else ""
                    ),
                    **action_logprob_fields,
                    "action_mc_sample_repeats": action_mc_sample_repeats,
                    "action_mc_temperature": action_mc_temperature if action_mc_sample_repeats else "",
                    **action_mc_summary,
                    "action_mc_raw_outputs_json": json.dumps(mc_raw_outputs),
                    "action_mc_query_error": mc_query_error,
                    "revealed_analysis_text": revealed_text,
                    "raw_action_text": raw_text,
                    "query_error": query_error,
                }
            )

        summary = summarize_prefix_behavior(
            prefix_evals,
            final_action=final_action,
            persistence_threshold=persistence_threshold,
        )
        trajectory_group = (
            "stay_optimal" if summary["trajectory_remains_optimal"]
            else "recover" if summary["recovery_step"] is not None
            else "go_wrong"
        )
        trajectory_row = {
            "example_id": example.example_id,
            "trajectory_id": example.trajectory_id,
            "step_index": example.step_index,
            "failure_category": example.failure_category,
            "selection_stage": example.selection_stage,
            "source_dataset": example.source_dataset,
            "n_reasoning_steps": len(reasoning_steps),
            "final_action": final_action,
            "observed_action": example.observed_action,
            "optimal_actions_json": json.dumps(example.optimal_actions),
            "analysis_unit": analysis_unit,
            "trajectory_group": trajectory_group,
            **summary,
        }
        for key in ("wrong_turn_step_strict", "wrong_turn_step_majority", "commitment_step", "recovery_step"):
            value = trajectory_row.get(key)
            trajectory_row[f"{key}_progress"] = (
                float(value) / len(reasoning_steps)
                if value not in {None, ""} and reasoning_steps
                else ""
            )
        trajectory_rows.append(trajectory_row)

        if collector is not None and reasoning_steps:
            if activation_prompt_mode == "original_trace":
                original_prompt = _render_original_trajectory_prompt(payload, step_payload)
                output_text = str(step_payload.get("output_text", ""))
                analysis_offset = output_text.find(analysis_text)
                if analysis_offset < 0:
                    raise ValueError(
                        f"Could not align analysis text inside output_text for {example.example_id}"
                    )
                full_prompt = original_prompt + output_text
                char_base = len(original_prompt) + analysis_offset
            else:
                full_prompt = before_reasoning + analysis_text.strip() + after_reasoning
                char_base = len(before_reasoning)
            prompt_dir = out_dir / "prompts" / _safe_path_component(example.trajectory_id)
            prompt_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = prompt_dir / f"step_{example.step_index:03d}_{activation_prompt_mode}_prompt.txt"
            analysis_path = prompt_dir / f"step_{example.step_index:03d}_analysis.txt"
            prompt_path.write_text(full_prompt)
            analysis_path.write_text(analysis_text)
            step_by_idx = {step.step_idx: step for step in reasoning_steps}
            prompt_spans = [
                (0, max(0, char_base - 1), char_base),
                *[
                (
                    step.step_idx,
                    char_base + step.start_char,
                    char_base + step.end_char,
                )
                for step in reasoning_steps
                ],
            ]
            record_prefix = f"{example.trajectory_id}/step_{example.step_index:03d}"
            collected_rows = collector.collect(
                prompt_text=full_prompt,
                step_spans_in_prompt=prompt_spans,
                layers=layers,
                output_dir=out_dir,
                record_prefix=record_prefix,
            )
            for row in collected_rows:
                reasoning_step_idx = int(row["reasoning_step_idx"])
                step = step_by_idx.get(reasoning_step_idx)
                char_start_prompt = char_base + step.start_char if step else max(0, char_base - 1)
                char_end_prompt = char_base + step.end_char if step else char_base
                row_base = {
                    "activation_schema_version": ACTIVATION_SCHEMA_VERSION,
                    "analysis_unit": analysis_unit,
                    "activation_prompt_mode": activation_prompt_mode,
                    "example_id": example.example_id,
                    "trajectory_id": example.trajectory_id,
                    "step_index": example.step_index,
                    "failure_category": example.failure_category,
                    "selection_stage": example.selection_stage,
                    "source_dataset": example.source_dataset,
                    "source_trajectory_path": str(trajectory_root / f"{example.trajectory_id}.json"),
                    "local_model_name_or_path": str(local_model_name_or_path),
                    "prompt_text_path": str(prompt_path),
                    "analysis_text_path": str(analysis_path),
                    "step_text": step.text if step else "[pre-reasoning boundary]",
                    "step_start_char_in_analysis": step.start_char if step else "",
                    "step_end_char_in_analysis": step.end_char if step else "",
                    "step_start_char_in_prompt": char_start_prompt,
                    "step_end_char_in_prompt": char_end_prompt,
                    "reasoning_progress": (
                        reasoning_step_idx / len(reasoning_steps) if reasoning_steps else 0.0
                    ),
                    "progress_bucket_10": (
                        min(10, round(10 * reasoning_step_idx / len(reasoning_steps)))
                        if reasoning_steps else 0
                    ),
                    "trajectory_group": trajectory_group,
                    **row,
                }
                activation_index_rows.append(row_base)
                geometry_input_rows.append(
                    {
                        **row_base,
                        "representation_kind": "last_token",
                        "activation_path": row["step_last_token_activation_path"],
                    }
                )
                geometry_input_rows.append(
                    {
                        **row_base,
                        "representation_kind": "mean_pool",
                        "activation_path": row["step_mean_activation_path"],
                    }
                )

        with checkpoint_path.open("a") as checkpoint_file:
            checkpoint_file.write(
                json.dumps(
                    {
                        "example_id": example.example_id,
                        "step_rows": step_rows[step_row_start:],
                        "trajectory_row": trajectory_row,
                        "activation_rows": activation_index_rows[activation_row_start:],
                        "usage_summary": usage_summary,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    geometry_rows: list[dict[str, Any]] = []
    geometry_summary_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    if geometry_input_rows and torch is not None:
        anchor_vectors = _build_anchor_vectors(geometry_input_rows, trajectory_rows)
        grouped_vectors: dict[tuple[str, int, str], list[tuple[int, list[float], dict[str, Any]]]] = {}
        for row in geometry_input_rows:
            tensor = torch.load(row["activation_path"], map_location="cpu")
            vector = tensor.tolist() if hasattr(tensor, "tolist") else list(tensor)
            grouped_vectors.setdefault(
                (
                    str(row["example_id"]),
                    int(row["layer"]),
                    str(row["representation_kind"]),
                ),
                [],
            ).append((int(row["reasoning_step_idx"]), vector, row))

        for (example_id, layer, rep_kind), items in grouped_vectors.items():
            ordered = sorted(items, key=lambda item: item[0])
            anchor_subset = {
                int(item_meta["reasoning_step_idx"]): anchor_vectors[
                    (layer, rep_kind, int(item_meta["progress_bucket_10"]))
                ]
                for _, _, item_meta in ordered
                if (layer, rep_kind, int(item_meta["progress_bucket_10"])) in anchor_vectors
            }
            metrics = compute_reasoning_geometry_metrics(
                [vec for _, vec, _ in ordered],
                anchor_vectors_by_bucket=anchor_subset,
                step_indices=[step_idx for step_idx, _, _ in ordered],
            )
            traj_row = next(row for row in trajectory_rows if row["example_id"] == example_id)
            wrong_turn_idx = traj_row.get("wrong_turn_step_majority")
            for metric_row, (_, _vec, meta) in zip(metrics, ordered):
                geometry_rows.append(
                    {
                        "example_id": example_id,
                        "trajectory_id": meta["trajectory_id"],
                        "step_index": meta["step_index"],
                        "failure_category": meta["failure_category"],
                        "trajectory_group": meta["trajectory_group"],
                        "layer": layer,
                        "representation_kind": rep_kind,
                        "wrong_turn_step_majority": wrong_turn_idx,
                        "reasoning_progress": meta["reasoning_progress"],
                        "progress_bucket_10": meta["progress_bucket_10"],
                        "activation_path": meta["activation_path"],
                        **metric_row,
                    }
                )

        grouped_geometry: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
        for row in geometry_rows:
            grouped_geometry.setdefault(
                (str(row["example_id"]), int(row["layer"]), str(row["representation_kind"])),
                [],
            ).append(row)
        trajectory_by_example = {str(row["example_id"]): row for row in trajectory_rows}
        event_metrics = ("aligned_change", "update_norm", "adjacent_step_cosine", "optimality_anchor_cosine")
        for (example_id, layer, representation_kind), rows in grouped_geometry.items():
            trajectory = trajectory_by_example[example_id]
            wrong_turn = trajectory.get("wrong_turn_step_majority")
            if wrong_turn in {None, ""}:
                continue
            wrong_turn = int(wrong_turn)
            ordered_rows = sorted(rows, key=lambda row: int(row["reasoning_step_idx"]))
            event_row = next(
                (row for row in ordered_rows if int(row["reasoning_step_idx"]) == wrong_turn),
                None,
            )
            if event_row is None:
                continue
            previous_rows = [
                row for row in ordered_rows if int(row["reasoning_step_idx"]) < wrong_turn
            ][-3:]
            output_row: dict[str, Any] = {
                "example_id": example_id,
                "trajectory_id": trajectory["trajectory_id"],
                "step_index": trajectory["step_index"],
                "failure_category": trajectory["failure_category"],
                "trajectory_group": trajectory["trajectory_group"],
                "layer": layer,
                "representation_kind": representation_kind,
                "wrong_turn_step_majority": wrong_turn,
                "wrong_turn_progress_majority": trajectory["wrong_turn_step_majority_progress"],
                "pre_window_size": len(previous_rows),
            }
            for metric in event_metrics:
                event_value = event_row.get(metric)
                previous_values = [
                    float(row[metric]) for row in previous_rows if row.get(metric) not in {None, ""}
                ]
                previous_mean = sum(previous_values) / len(previous_values) if previous_values else None
                output_row[f"{metric}_at_wrong_turn"] = event_value
                output_row[f"{metric}_previous_3_mean"] = previous_mean
                output_row[f"{metric}_delta_at_wrong_turn"] = (
                    float(event_value) - previous_mean
                    if event_value not in {None, ""} and previous_mean is not None
                    else None
                )
            event_rows.append(output_row)

        summary_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
        for row in event_rows:
            summary_groups.setdefault(
                (str(row["failure_category"]), int(row["layer"]), str(row["representation_kind"])),
                [],
            ).append(row)
        category_counts = {
            category: len(rows)
            for category, rows in {
                category: [row for row in trajectory_rows if str(row["failure_category"]) == category]
                for category in {str(row["failure_category"]) for row in trajectory_rows}
            }.items()
        }
        for (category, layer, representation_kind), rows in sorted(summary_groups.items()):
            aligned_at = [
                float(row["aligned_change_at_wrong_turn"])
                for row in rows if row.get("aligned_change_at_wrong_turn") not in {None, ""}
            ]
            aligned_delta = [
                float(row["aligned_change_delta_at_wrong_turn"])
                for row in rows if row.get("aligned_change_delta_at_wrong_turn") not in {None, ""}
            ]
            update_delta = [
                float(row["update_norm_delta_at_wrong_turn"])
                for row in rows if row.get("update_norm_delta_at_wrong_turn") not in {None, ""}
            ]
            wrong_turn_progress = [
                float(row["wrong_turn_progress_majority"])
                for row in rows if row.get("wrong_turn_progress_majority") not in {None, ""}
            ]
            geometry_summary_rows.append(
                {
                    "failure_category": category,
                    "layer": layer,
                    "representation_kind": representation_kind,
                    "n_trajectories": category_counts.get(category, 0),
                    "n_with_identifiable_wrong_turn_and_geometry": len(rows),
                    "median_wrong_turn_progress": median(wrong_turn_progress) if wrong_turn_progress else "",
                    "mean_aligned_change_at_wrong_turn": sum(aligned_at) / len(aligned_at) if aligned_at else "",
                    "mean_aligned_change_delta_vs_previous_3": sum(aligned_delta) / len(aligned_delta) if aligned_delta else "",
                    "mean_update_norm_delta_vs_previous_3": sum(update_delta) / len(update_delta) if update_delta else "",
                }
            )
        _plot_wrong_turn_histogram(trajectory_rows, out_dir / "figs" / "wrong_turn_histogram.png")
        _plot_mean_aligned_change(geometry_rows, out_dir / "figs" / "aligned_change_curves.png")

    _write_csv(out_dir / "prefix_action_rows.csv", step_rows)
    _write_csv(out_dir / "trajectory_wrong_turn_summary.csv", trajectory_rows)
    if activation_index_rows:
        _write_csv(out_dir / "step_activation_rows.csv", activation_index_rows)
    if geometry_rows:
        _write_csv(out_dir / "geometry_rows.csv", geometry_rows)
    if geometry_input_rows and event_rows:
        _write_csv(out_dir / "wrong_turn_geometry_events.csv", event_rows)
    if geometry_summary_rows:
        _write_csv(out_dir / "geometry_summary.csv", geometry_summary_rows)
    _write_json(
        out_dir / "manifest.json",
        {
            "status": "completed",
            "slice_mode": slice_mode,
            "trajectory_slice_type": trajectory_slice_type if slice_mode == "trajectory_candidates" else "",
            "balanced_failure_rows_per_mode": balanced_failure_rows_per_mode,
            "non_failure_control_rows": non_failure_control_rows,
            "n_examples": len(examples),
            "collect_activations": collect_activations,
            "activation_source": activation_source if collect_activations else "",
            "activation_prompt_mode": activation_prompt_mode if collect_activations else "",
            "local_model_name_or_path": local_model_name_or_path or "",
            "device_map": device_map or "",
            "torch_dtype": torch_dtype if collect_activations else "",
            "low_cpu_mem_usage": low_cpu_mem_usage if collect_activations else "",
            "forward_chunk_size": forward_chunk_size if collect_activations else "",
            "resume": resume,
            "layers": list(layers),
            "segmentation_mode": segmentation_mode,
            "max_reasoning_steps": max_reasoning_steps,
            "analysis_unit": analysis_unit,
            "sentence_boundaries_path": (
                sentence_boundaries_path if analysis_unit == "sentence" else ""
            ),
            "sentence_boundaries_sha256": (
                hashlib.sha256(Path(sentence_boundaries_path).read_bytes()).hexdigest()
                if analysis_unit == "sentence"
                else ""
            ),
            "persistence_threshold": persistence_threshold,
            "action_mc_sample_repeats": action_mc_sample_repeats,
            "action_mc_temperature": action_mc_temperature,
            "action_mc_max_workers": action_mc_max_workers,
            "action_logprob_temperature": action_logprob_temperature,
            "action_top_logprobs": (
                action_top_logprobs if action_logprob_temperature is not None else None
            ),
            "action_logprob_max_workers": action_logprob_max_workers,
            "action_logprob_candidate_labels": (
                list(ACTION_LABELS) if action_logprob_temperature is not None else []
            ),
            "max_prefix_positions_per_example": max_prefix_positions_per_example,
            "candidate_rows_path": candidate_rows_path,
            "trajectory_dir": trajectory_dir,
            "usage_summary": usage_summary,
        },
    )


def _iter_trajectory_analysis_texts(trajectory_dir: Path) -> Iterable[tuple[Path, int, str]]:
    for path in sorted(trajectory_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for step in payload.get("steps", []):
            output_text = str(step.get("output_text", ""))
            analysis_text, _final_text = _extract_analysis_and_final(output_text)
            if analysis_text.strip():
                yield path, int(step.get("step_id", step.get("step_index", 0))), analysis_text


def validate_doorkey_chunking_strategy(
    *,
    trajectory_dir: str = DEFAULT_TRAJECTORY_DIR,
    output_dir: str = "data/behavioral_probes/doorkey_chunking_validation",
    max_analysis_chunks: int = 32,
    sample_limit: int = 25,
) -> dict[str, Any]:
    """Validate canonical DoorKey chunking over local trajectory JSON files."""
    trajectory_root = Path(trajectory_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    sentence_counts: list[int] = []
    analysis_counts: list[int] = []
    error_rows: list[dict[str, Any]] = []
    sentence_rows: list[dict[str, Any]] = []
    analysis_rows: list[dict[str, Any]] = []
    sample_sentence_rows: list[dict[str, Any]] = []
    sample_analysis_rows: list[dict[str, Any]] = []
    capped_chunk_word_counts: list[list[int]] = []
    files_seen: set[str] = set()
    n_traces = 0
    n_traces_over_chunk_cap = 0
    n_internal_action_json = 0
    n_internal_standalone_action_json = 0
    n_terminal_action_json = 0

    for path, step_id, analysis_text in _iter_trajectory_analysis_texts(trajectory_root):
        files_seen.add(str(path))
        n_traces += 1
        trace_id = f"{path.stem}_step_{step_id:03d}"
        sentence_chunks = chunk_doorkey_reasoning_trace(analysis_text, trace_id=trace_id)
        analysis_chunks = pack_to_analysis_chunks(
            sentence_chunks,
            max_chunks=max_analysis_chunks,
            raw_text=analysis_text,
        )
        errors = verify_doorkey_chunks(
            analysis_text,
            sentence_chunks,
            analysis_chunks,
            max_analysis_chunks=max_analysis_chunks,
        )
        reasoning_sentences = [chunk for chunk in sentence_chunks if chunk.kind == "reasoning"]
        sentence_counts.append(len(reasoning_sentences))
        analysis_counts.append(len(analysis_chunks))
        n_traces_over_chunk_cap += int(len(reasoning_sentences) > max_analysis_chunks)
        if len(analysis_chunks) == max_analysis_chunks:
            capped_chunk_word_counts.append(
                [max(1, len(chunk.text.split())) for chunk in analysis_chunks]
            )
        n_internal_action_json += sum(
            int(chunk.kind == "reasoning" and chunk.contains_action_json_mention)
            for chunk in sentence_chunks
        )
        n_internal_standalone_action_json += sum(
            int(chunk.kind == "reasoning" and bool(_ACTION_JSON_RE.fullmatch(chunk.text.strip())))
            for chunk in sentence_chunks
        )
        n_terminal_action_json += sum(int(chunk.kind == "final_action") for chunk in sentence_chunks)
        for error in errors:
            error_rows.append(
                {
                    "trace_id": trace_id,
                    "path": str(path),
                    "step_id": step_id,
                    "error_type": error[0],
                    "error_payload": json.dumps(error[1:]),
                }
            )
        for chunk in sentence_chunks:
            sentence_rows.append(
                {
                    "trace_id": trace_id,
                    "path": str(path),
                    "step_id": step_id,
                    "sentence_id": chunk.chunk_id,
                    "kind": chunk.kind,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "token_start": chunk.token_start if chunk.token_start is not None else "",
                    "token_end": chunk.token_end if chunk.token_end is not None else "",
                    "contains_action_json_mention": chunk.contains_action_json_mention,
                    "chunker_version": chunk.chunker_version,
                    "text": chunk.text,
                }
            )
        for chunk in analysis_chunks:
            analysis_rows.append(
                {
                    "trace_id": trace_id,
                    "path": str(path),
                    "step_id": step_id,
                    "analysis_id": chunk.analysis_id,
                    "sentence_start": chunk.sentence_start,
                    "sentence_end": chunk.sentence_end,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "packing_version": PACKER_VERSION,
                    "text": chunk.text,
                }
            )
        if len(sample_sentence_rows) < sample_limit:
            for chunk in sentence_chunks[: max(0, sample_limit - len(sample_sentence_rows))]:
                sample_sentence_rows.append(
                    {
                        "trace_id": trace_id,
                        "sentence_id": chunk.chunk_id,
                        "kind": chunk.kind,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "contains_action_json_mention": chunk.contains_action_json_mention,
                        "text": chunk.text,
                    }
                )
        if len(sample_analysis_rows) < sample_limit:
            for chunk in analysis_chunks[: max(0, sample_limit - len(sample_analysis_rows))]:
                sample_analysis_rows.append(
                    {
                        "trace_id": trace_id,
                        "analysis_id": chunk.analysis_id,
                        "sentence_start": chunk.sentence_start,
                        "sentence_end": chunk.sentence_end,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "packing_version": PACKER_VERSION,
                        "text": chunk.text,
                    }
                )

    def _mean(values: list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    first_chunk_words = [values[0] for values in capped_chunk_word_counts]
    penultimate_chunk_words = [values[-2] for values in capped_chunk_word_counts]
    final_chunk_words = [values[-1] for values in capped_chunk_word_counts]
    final_to_other_ratios = [
        values[-1] / max(1.0, float(median(values[:-1])))
        for values in capped_chunk_word_counts
    ]
    error_counts = Counter(row["error_type"] for row in error_rows)
    report = {
        "chunker_version": CHUNKER_VERSION,
        "packing_version": PACKER_VERSION,
        "trajectory_dir": str(trajectory_root),
        "n_files": len(files_seen),
        "n_reasoning_traces": n_traces,
        "mean_sentences": _mean(sentence_counts),
        "median_sentences": float(median(sentence_counts)) if sentence_counts else 0.0,
        "max_sentences": max(sentence_counts) if sentence_counts else 0,
        "mean_analysis_chunks": _mean(analysis_counts),
        "mean_chunked_trajectory_chunks": _mean(analysis_counts),
        "median_analysis_chunks": float(median(analysis_counts)) if analysis_counts else 0.0,
        "median_chunked_trajectory_chunks": (
            float(median(analysis_counts)) if analysis_counts else 0.0
        ),
        "max_analysis_chunks": max(analysis_counts) if analysis_counts else 0,
        "max_chunked_trajectory_chunks": max(analysis_counts) if analysis_counts else 0,
        "pct_traces_over_chunk_cap": (
            n_traces_over_chunk_cap / n_traces if n_traces else 0.0
        ),
        "n_traces_at_chunk_cap": len(capped_chunk_word_counts),
        "median_words_first_chunk_at_cap": (
            float(median(first_chunk_words)) if first_chunk_words else 0.0
        ),
        "median_words_penultimate_chunk_at_cap": (
            float(median(penultimate_chunk_words)) if penultimate_chunk_words else 0.0
        ),
        "median_words_final_chunk_at_cap": (
            float(median(final_chunk_words)) if final_chunk_words else 0.0
        ),
        "median_final_to_other_chunk_word_ratio_at_cap": (
            float(median(final_to_other_ratios)) if final_to_other_ratios else 0.0
        ),
        "n_internal_action_json_mentions": n_internal_action_json,
        "n_internal_standalone_action_json_mentions": n_internal_standalone_action_json,
        "n_terminal_action_json_chunks": n_terminal_action_json,
        "n_validation_errors": len(error_rows),
        "n_empty_chunk_errors": error_counts.get("empty_chunk", 0),
        "n_span_errors": error_counts.get("span_text_mismatch", 0)
        + error_counts.get("analysis_span_text_mismatch", 0),
        "n_overlap_errors": error_counts.get("overlap", 0),
        "n_action_json_errors": error_counts.get("terminal_action_json_leaked_into_reasoning", 0),
        "n_bad_numbered_splits": error_counts.get("bad_numbered_split", 0),
        "n_punctuation_only_fragment_errors": error_counts.get("punctuation_only_fragment", 0),
        "n_analysis_chunk_overflow": error_counts.get("too_many_analysis_chunks", 0),
        "error_counts": dict(error_counts),
    }
    _write_json(output_root / "chunking_report.json", report)
    _write_csv_rows(output_root / "chunking_errors.csv", error_rows)
    _write_csv_rows(output_root / "sentences.csv", sentence_rows)
    _write_csv_rows(output_root / "chunked_trajectories.csv", analysis_rows)
    _write_csv_rows(output_root / "sample_sentences.csv", sample_sentence_rows)
    _write_csv_rows(output_root / "sample_chunked_trajectories.csv", sample_analysis_rows)
    (output_root / "chunking_report.md").write_text(
        "\n".join(
            [
                "# DoorKey Chunking Validation",
                "",
                f"- Chunker version: `{CHUNKER_VERSION}`",
                f"- Packing version: `{PACKER_VERSION}`",
                f"- Trajectory files: {report['n_files']}",
                f"- Reasoning traces: {report['n_reasoning_traces']}",
                f"- Mean sentences: {report['mean_sentences']:.2f}",
                f"- Median sentences: {report['median_sentences']:.1f}",
                f"- Max sentences: {report['max_sentences']}",
                f"- Traces over {max_analysis_chunks} sentences: "
                f"{report['pct_traces_over_chunk_cap']:.1%}",
                f"- Max chunks per row in `chunked_trajectories.csv`: "
                f"{report['max_chunked_trajectory_chunks']}",
                f"- Validation errors: {report['n_validation_errors']}",
                "",
                "## Validation",
                "",
                "The chunks are usable because every exported sentence is tied to a fixed "
                "character span in the original reasoning trace, and every packed trajectory "
                "chunk is a deterministic view over adjacent sentence spans. Traces exceeding "
                f"{max_analysis_chunks} sentences are divided at approximately equal cumulative "
                "word-count boundaries, while preserving sentence order. The same span IDs "
                "can therefore be reused for activation extraction, behavioral belief queries, "
                "action-prefix evaluation, semantic labels, and later interventions.",
                "",
                "| Check | Count | Interpretation |",
                "|---|---:|---|",
                f"| Empty chunks | {report['n_empty_chunk_errors']} | No empty sentence rows were emitted. |",
                f"| Span mismatches | {report['n_span_errors']} | Exported text matches `raw_trace[char_start:char_end]`. |",
                f"| Character-span overlaps | {report['n_overlap_errors']} | No overlapping sentence spans were found. |",
                f"| Terminal action JSON in reasoning | {report['n_action_json_errors']} | Terminal action JSON is not leaked into reasoning chunks. |",
                f"| Isolated numbered-list fragments | {report['n_bad_numbered_splits']} | No false split such as `1.` or `2.` was found. |",
                f"| Punctuation-only fragments | {report['n_punctuation_only_fragment_errors']} | No meaningless punctuation-only chunks were found. |",
                f"| Analysis chunks over cap | {report['n_analysis_chunk_overflow']} | Every packed trajectory has at most {max_analysis_chunks} chunks. |",
                "",
                "## Chunk-Length Balance",
                "",
                f"For the {report['n_traces_at_chunk_cap']} traces packed into exactly "
                f"{max_analysis_chunks} chunks:",
                "",
                "| Position | Median words |",
                "|---|---:|",
                f"| First chunk | {report['median_words_first_chunk_at_cap']:.1f} |",
                f"| Penultimate chunk | {report['median_words_penultimate_chunk_at_cap']:.1f} |",
                f"| Final chunk | {report['median_words_final_chunk_at_cap']:.1f} |",
                "",
                "The median final-chunk length divided by the median length of the preceding "
                f"chunks is {report['median_final_to_other_chunk_word_ratio_at_cap']:.2f}. "
                "The previous greedy packer produced a much longer final chunk by forcing all "
                "remaining sentences into chunk 32; that legacy table is retained only for "
                "reproducing the existing 8-state activation run.",
                "",
                f"Internal action JSON mentions are allowed when reasoning continues after them; "
                f"{report['n_internal_action_json_mentions']} such mentions were flagged, including "
                f"{report['n_internal_standalone_action_json_mentions']} standalone JSON snippets.",
                "",
                "Primary outputs:",
                "",
                "- `sentences.csv`",
                "- `chunked_trajectories.csv`",
                "- `chunking_errors.csv`",
            ]
        )
        + "\n"
    )
    return report


__all__ = [
    "AnalysisChunk",
    "CHUNKER_VERSION",
    "PACKER_VERSION",
    "DEFAULT_CANDIDATE_ROWS_PATH",
    "DEFAULT_TRAJECTORY_DIR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_LAYERS",
    "DriftExample",
    "LocalHiddenStateCollector",
    "PrefixStepEvaluation",
    "ReasoningStep",
    "SentenceChunk",
    "action_entropy_from_probabilities",
    "action_probabilities_from_samples",
    "build_revealed_action_prompt",
    "chunk_doorkey_reasoning_trace",
    "compute_reasoning_geometry_metrics",
    "pack_to_analysis_chunks",
    "regenerate_step_reasoning_activations",
    "rebuild_reasoning_geometry_from_activations",
    "run_step_reasoning_drift_experiment",
    "segment_reasoning_trace",
    "summarize_action_samples",
    "summarize_prefix_behavior",
    "validate_doorkey_chunking_strategy",
    "verify_doorkey_chunks",
]
