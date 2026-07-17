"""Gradual-CoT black-box alignment on clean matched and failure-focused slices."""

from __future__ import annotations

import csv
import json
import math
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from reveng.experiments.behavioral_probe_metrics import (
    belief_action_consistency,
    sampled_answer_from_probabilities,
    shannon_entropy,
)
from reveng.experiments.behavioral_probe_parse import parse_behavioral_probe_answer
from reveng.experiments.behavioral_probe_questions import BEHAVIORAL_PROBE_QUESTION_BY_ID
from reveng.experiments.behavioral_probe_runner import (
    BehavioralProbeLLM,
    _add_usage,
    _behavioral_probe_preamble,
    _belief_prompt,
    _collect_label3_logprob_readout,
    _empty_usage_bucket,
    _observed_action_prompt_with_state_text,
    _probabilities_from_mc_answers,
    _query_action,
    _query_text,
)
from reveng.experiments.wall_feature_patch_experiment import (
    load_wall_feature_patch_candidates,
    select_wall_feature_patch_states,
)

DEFAULT_MATCHED_WALL_ROWS_PATH = (
    "data/cognitive_map_probe_reasoning_eval_public_slice_repeat10/matched_wall_rows.csv"
)
DEFAULT_FAILURE_CANDIDATE_ROWS_PATH = (
    "data/behavioral_probes/trajectory_instances_recomputed_optimal/trajectory_selection_candidates.csv"
)
DEFAULT_FAILURE_TRAJECTORY_DIR = "data/hf/trajectories_key_door_100/trajectories_key_door"
DEFAULT_OUTPUT_DIR = "data/behavioral_probes/gradual_cot_alignment_public_slice"
DEFAULT_MODEL_NAME = "together_ai/openai/gpt-oss-20b"
REVEAL_PCTS = (0, 25, 50, 75, 100)
WALL_SAMPLING_MODES = ("full", "logprob_only")
PUBLIC_TRAJECTORIES_REPO_ID = "project-telos/trajectories_test_full"
ANALYSIS_START = "<|channel|>analysis<|message|>"
FINAL_START = "<|end|><|start|>assistant<|channel|>final<|message|>"
TOKENIZER_MODEL_ID = "openai/gpt-oss-20b"
WALL_QUESTION_IDS = ("wall_left", "wall_right", "wall_up", "wall_down")
ACTION_LABELS = ("UP", "DOWN", "LEFT", "RIGHT", "INVALID")
CORE_FAILURE_MODES = (
    "wall_hit",
    "backtrack",
    "oscillation_2cycle",
    "short_loop",
    "freeze_repeat",
    "avoidable_detour",
)
_THREAD_LOCAL = threading.local()

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover
    AutoTokenizer = None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
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


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


class _TokenizerHelper:
    def __init__(self, model_id: str = TOKENIZER_MODEL_ID) -> None:
        self.method = "char_fallback"
        self.tokenizer = None
        if AutoTokenizer is None:
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            self.method = "hf_tokenizer_local"
            return
        except Exception:
            self.tokenizer = None
            self.method = "char_fallback"

    def total_units(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def reveal_prefix(self, text: str, pct: int) -> tuple[str, int, int]:
        total_units = self.total_units(text)
        if pct <= 0 or total_units == 0:
            return "", 0, total_units
        if pct >= 100:
            return text, total_units, total_units
        reveal_units = max(0, math.floor(total_units * (pct / 100.0)))
        if self.tokenizer is None:
            return text[:reveal_units], reveal_units, total_units
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        snippet = self.tokenizer.decode(token_ids[:reveal_units], skip_special_tokens=False)
        return snippet, reveal_units, total_units


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r".*?(?:[.!?](?:\s+|$)|\n+|$)", text, flags=re.DOTALL):
        segment = match.group(0)
        if not segment:
            continue
        end = start + len(segment)
        if segment.strip():
            spans.append((start, end))
        start = end
    return spans


def _sentence_boundary_metadata(full_text: str, revealed_text: str) -> tuple[int, float]:
    spans = _sentence_spans(full_text)
    if not spans or not revealed_text:
        return 0, 0.0
    revealed_len = len(revealed_text)
    boundary_index = 0
    coverage = 0.0
    for idx, (_start, end) in enumerate(spans, start=1):
        if end <= revealed_len:
            boundary_index = idx
            coverage = end / len(full_text) if full_text else 0.0
        else:
            break
    return boundary_index, coverage


def _extract_analysis_and_final(output_text: str) -> tuple[str, str]:
    if ANALYSIS_START not in output_text or FINAL_START not in output_text:
        return "", output_text
    analysis = output_text.split(ANALYSIS_START, 1)[1].split(FINAL_START, 1)[0]
    final_text = output_text.split(FINAL_START, 1)[1]
    final_text = final_text.split("<|return|>", 1)[0]
    return analysis.strip(), final_text.strip()


def _trajectory_filename_from_stem(trajectory_stem: str) -> str:
    size_match = re.search(r"_size(\d+)_", trajectory_stem)
    if size_match is None:
        raise ValueError(f"Unable to recover grid size from trajectory stem: {trajectory_stem}")
    size = size_match.group(1)
    return f"size{size}/{trajectory_stem}.json"


def _load_hf_trajectory_payload(trajectory_stem: str, cache_dir: Path) -> dict[str, Any]:
    filename = _trajectory_filename_from_stem(trajectory_stem)
    path = hf_hub_download(
        repo_id=PUBLIC_TRAJECTORIES_REPO_ID,
        repo_type="dataset",
        filename=filename,
        cache_dir=str(cache_dir),
    )
    return json.load(open(path))


def _load_local_trajectory_payload(trajectory_id: str, trajectory_dir: Path) -> dict[str, Any]:
    path = trajectory_dir / f"{trajectory_id}.json"
    if not path.exists():
        hits = list(trajectory_dir.rglob(f"{trajectory_id}.json"))
        if not hits:
            raise FileNotFoundError(f"Could not find local trajectory JSON for {trajectory_id} under {trajectory_dir}")
        path = hits[0]
    return json.load(open(path))


def _load_pair_rows_from_matched_wall_rows(path: str | Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _read_csv(path):
        key = (str(row["example_id"]), str(row["question_id"]))
        base = grouped.setdefault(
            key,
            {
                "example_id": str(row["example_id"]),
                "trajectory_id": str(row["trajectory_id"]),
                "step_index": str(row["step_index"]),
                "question_id": str(row["question_id"]),
                "grid_text": str(row["grid_text"]),
                "carrying_key": bool(_normalize_bool(row.get("carrying_key")) or False),
                "ground_truth_label": str(row["ground_truth_label"]),
                "observed_action": str(row.get("observed_action", "")).upper(),
                "optimal_actions_json": str(row.get("optimal_actions_json", "[]")),
                "is_optimal_action": _normalize_bool(row.get("is_optimal_action")),
                "wall_hit": _normalize_bool(row.get("wall_hit")),
                "primary_step_failure_mode": str(row.get("primary_step_failure_mode", "none") or "none"),
                "source_dataset": str(row.get("source_dataset", "clean_matched_public_slice")),
                "whitebox_prediction_pre": "",
                "whitebox_prediction_post": "",
            },
        )
        split = str(row["reasoning_split"])
        if split == "pre":
            base["whitebox_prediction_pre"] = str(row["whitebox_prediction"])
        elif split == "post":
            base["whitebox_prediction_post"] = str(row["whitebox_prediction"])
    return sorted(grouped.values(), key=lambda row: (row["trajectory_id"], int(row["step_index"]), row["question_id"]))


def _load_pair_rows_from_failure_candidates(
    path: str | Path,
    *,
    max_wall_hit: int,
    max_avoidable_detour: int,
    max_backtrack: int,
    max_baseline: int,
) -> list[dict[str, Any]]:
    candidates = load_wall_feature_patch_candidates(str(path))
    selected = select_wall_feature_patch_states(
        candidates,
        max_wall_hit=max_wall_hit,
        max_avoidable_detour=max_avoidable_detour,
        max_backtrack=max_backtrack,
        max_baseline=max_baseline,
    )
    pair_rows: list[dict[str, Any]] = []
    for row in selected:
        probe_truths = row["probe_truths"]
        for question_id in WALL_QUESTION_IDS:
            pair_rows.append(
                {
                    "example_id": str(row["example_id"]),
                    "trajectory_id": str(row["trajectory_id"]),
                    "step_index": str(row["step_index"]),
                    "question_id": question_id,
                    "grid_text": str(row["grid_text"]),
                    "carrying_key": bool(row.get("carrying_key") or False),
                    "ground_truth_label": str(probe_truths.get(question_id, "")),
                    "observed_action": str(row.get("observed_action", "")).upper(),
                    "optimal_actions_json": json.dumps(row.get("optimal_actions", [])),
                    "is_optimal_action": row.get("is_optimal_action"),
                    "wall_hit": row.get("wall_hit"),
                    "primary_step_failure_mode": str(row.get("primary_step_failure_mode", "none") or "none"),
                    "source_dataset": "trajectory_failure_candidates",
                    "whitebox_prediction_pre": "",
                    "whitebox_prediction_post": "",
                }
            )
    return sorted(pair_rows, key=lambda row: (row["trajectory_id"], int(row["step_index"]), row["question_id"]))


def _parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
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
        return [str(item) for item in parsed]
    return []


def _candidate_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("trajectory_id", "")),
        int(row.get("step_index") or 0),
        str(row.get("example_id", "")),
    )


def _row_matches_failure_mode(row: dict[str, Any], failure_mode: str) -> bool:
    if failure_mode == "terminal_failure_tail":
        return bool(_normalize_bool(row.get("is_terminal_failure_tail")))
    failure_modes = set(_parse_json_list(row.get("failure_modes_json", row.get("failure_modes", []))))
    primary_mode = str(row.get("primary_step_failure_mode", "none") or "none")
    return failure_mode in failure_modes or primary_mode == failure_mode


def _select_balanced_failure_mode_states(
    rows: list[dict[str, Any]],
    *,
    rows_per_mode: int,
) -> list[dict[str, Any]]:
    selected_rows = [row for row in rows if _normalize_bool(row.get("selected_for_probe")) is True]
    by_step = {
        (str(row.get("trajectory_id", "")), int(row.get("step_index") or 0)): row
        for row in selected_rows
    }
    chosen: dict[tuple[str, int], dict[str, Any]] = {}
    for mode in CORE_FAILURE_MODES:
        mode_rows = [
            row
            for row in selected_rows
            if str(row.get("selection_stage", "")) == "failure" and _row_matches_failure_mode(row, mode)
        ]
        for row in sorted(mode_rows, key=_candidate_sort_key)[:rows_per_mode]:
            key = (str(row.get("trajectory_id", "")), int(row.get("step_index") or 0))
            chosen[key] = row
            context_key = (key[0], key[1] - 1)
            context_row = by_step.get(context_key)
            if (
                context_row is not None
                and _normalize_bool(context_row.get("is_pre_failure_context")) is True
                and int(context_row.get("pre_failure_for_step_index") or -1) == key[1]
            ):
                chosen[context_key] = context_row
    ordered_keys = sorted(
        chosen.keys(),
        key=lambda key: (
            str(chosen[key].get("trajectory_id", "")),
            key[1],
        ),
    )
    return [chosen[key] for key in ordered_keys]


def _select_candidate_slice_rows(
    rows: list[dict[str, Any]],
    *,
    slice_type: str,
    rows_per_mode: int,
) -> list[dict[str, Any]]:
    selected_for_probe = [row for row in rows if _normalize_bool(row.get("selected_for_probe")) is True]
    if slice_type == "all_rows":
        return rows
    if slice_type == "selection_candidates":
        return selected_for_probe
    if slice_type == "tagged_failure_states":
        return [row for row in selected_for_probe if str(row.get("selection_stage", "")) == "failure"]
    if slice_type == "balanced_failure_modes":
        return _select_balanced_failure_mode_states(rows, rows_per_mode=rows_per_mode)
    if slice_type == "context_rows":
        return [row for row in selected_for_probe if str(row.get("selection_stage", "")) == "context"]
    if slice_type == "baseline_optimal":
        return [
            row
            for row in selected_for_probe
            if str(row.get("primary_step_failure_mode", "none") or "none") in {"", "none"}
            and _normalize_bool(row.get("is_optimal_action")) is True
            and _normalize_bool(row.get("wall_hit")) is False
        ]
    if slice_type == "non_failure_controls":
        return [
            row
            for row in selected_for_probe
            if str(row.get("selection_stage", "")) == "context"
            and str(row.get("primary_step_failure_mode", "none") or "none") in {"", "none"}
        ]
    if slice_type == "non_optimal_action":
        return [
            row
            for row in rows
            if str(row.get("observed_action", "")).upper() in {"UP", "DOWN", "LEFT", "RIGHT"}
            and _normalize_bool(row.get("is_optimal_action")) is False
        ]
    if slice_type == "suboptimal_trajectory":
        return [
            row
            for row in rows
            if row.get("trajectory_length_delta") not in {"", None}
            and float(row["trajectory_length_delta"]) > 0
        ]
    if slice_type == "failed_trajectory":
        return [row for row in rows if _normalize_bool(row.get("trajectory_reached_goal")) is False]
    if slice_type.startswith("primary_failure_mode:"):
        failure_mode = slice_type.split(":", 1)[1]
        return [
            row
            for row in rows
            if str(row.get("primary_step_failure_mode", "none") or "none") == failure_mode
        ]
    if slice_type.startswith("failure_mode:"):
        failure_mode = slice_type.split(":", 1)[1]
        return [row for row in rows if _row_matches_failure_mode(row, failure_mode)]
    raise ValueError(f"Unknown trajectory_slice_type: {slice_type}")


def _load_pair_rows_from_trajectory_candidates(
    path: str | Path,
    *,
    slice_type: str,
    rows_per_mode: int,
    max_examples: int | None = None,
) -> list[dict[str, Any]]:
    candidate_rows = load_wall_feature_patch_candidates(str(path))
    selected_rows = _select_candidate_slice_rows(
        candidate_rows,
        slice_type=slice_type,
        rows_per_mode=rows_per_mode,
    )
    if max_examples is not None:
        selected_rows = selected_rows[:max(0, int(max_examples))]
    pair_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        probe_truths = row["probe_truths"]
        for question_id in WALL_QUESTION_IDS:
            pair_rows.append(
                {
                    "example_id": str(row["example_id"]),
                    "trajectory_id": str(row["trajectory_id"]),
                    "step_index": str(row["step_index"]),
                    "question_id": question_id,
                    "grid_text": str(row["grid_text"]),
                    "carrying_key": bool(row.get("carrying_key") or False),
                    "ground_truth_label": str(probe_truths.get(question_id, "")),
                    "observed_action": str(row.get("observed_action", "")).upper(),
                    "optimal_actions_json": json.dumps(row.get("optimal_actions", [])),
                    "is_optimal_action": row.get("is_optimal_action"),
                    "wall_hit": row.get("wall_hit"),
                    "primary_step_failure_mode": str(row.get("primary_step_failure_mode", "none") or "none"),
                    "selection_stage": str(row.get("selection_stage", "")),
                    "source_dataset": f"trajectory_candidates:{slice_type}",
                    "whitebox_prediction_pre": "",
                    "whitebox_prediction_post": "",
                }
            )
    return sorted(pair_rows, key=lambda row: (row["trajectory_id"], int(row["step_index"]), row["question_id"]))


def _reasoning_state_block(grid_text: str, carrying_key: bool, revealed_analysis: str) -> str:
    block = "Current grid state:\n\n" + grid_text + "\n\n"
    block += f"Agent status:\n- Carrying key: {str(bool(carrying_key)).lower()}\n\n"
    if revealed_analysis.strip():
        block += "Reasoning trace available so far:\n" + revealed_analysis.strip() + "\n\n"
    return block


def _clean_question_prompt_text(prompt_text: str) -> str:
    return re.sub(
        r"\s*Answer with exactly one of:\s*yes,\s*no,\s*unknown\.\s*$",
        "",
        prompt_text,
        flags=re.IGNORECASE,
    ).rstrip()


def _wall_logprob_prompt(*, grid_text: str, carrying_key: bool, question_id: str, revealed_analysis: str) -> str:
    question = BEHAVIORAL_PROBE_QUESTION_BY_ID[question_id]
    state_description_text = _reasoning_state_block(grid_text, carrying_key, revealed_analysis)
    return _belief_prompt(
        grid_text=grid_text,
        carrying_key=carrying_key,
        question=question,
        prompt_preset="cardinal_action_explicit",
        state_description_text=state_description_text,
    )


def _wall_json_prompt(*, grid_text: str, carrying_key: bool, question_id: str, revealed_analysis: str) -> str:
    question = BEHAVIORAL_PROBE_QUESTION_BY_ID[question_id]
    state_description_text = _reasoning_state_block(grid_text, carrying_key, revealed_analysis)
    return (
        _behavioral_probe_preamble("cardinal_action_explicit")
        + "\n# Inputs\n\n"
        + state_description_text
        + _clean_question_prompt_text(question.prompt_text)
        + "\n\n"
        + 'Return exactly one JSON object of the form {"label": "<yes|no|unknown>"} and nothing else.\n'
    )


def _action_prompt(*, grid_text: str, carrying_key: bool, revealed_analysis: str) -> str:
    return _observed_action_prompt_with_state_text(
        grid_text=grid_text,
        carrying_key=carrying_key,
        prompt_preset="cardinal_action_explicit",
        state_description_text=_reasoning_state_block(grid_text, carrying_key, revealed_analysis),
    )


def _parse_optimal_actions(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item).upper() for item in parsed]
    except Exception:
        pass
    return []


def _parse_wall_json_label(raw_text: str | None) -> str:
    if raw_text is None:
        return "invalid"
    cleaned = raw_text.strip()
    if not cleaned:
        return "invalid"
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict) and isinstance(payload.get("label"), str):
            return parse_behavioral_probe_answer(payload["label"])
    except Exception:
        pass
    return parse_behavioral_probe_answer(raw_text)


def _action_probabilities(sampled_actions: list[str]) -> dict[str, float]:
    probs = {label: 0.0 for label in ACTION_LABELS}
    if not sampled_actions:
        probs["INVALID"] = 1.0
        return probs
    total = len(sampled_actions)
    for label in sampled_actions:
        normalized = label if label in probs else "INVALID"
        probs[normalized] += 1.0 / total
    return probs


def _sampled_action_from_probabilities(probabilities: dict[str, float]) -> str:
    if not probabilities:
        return "INVALID"
    order = {label: idx for idx, label in enumerate(ACTION_LABELS)}
    return max(probabilities.items(), key=lambda item: (item[1], -order[item[0]]))[0]


def _action_entropy(probabilities: dict[str, float]) -> float:
    return shannon_entropy(probabilities)


def _action_agreement_rate(sampled_actions: list[str], modal_action: str) -> float:
    if not sampled_actions:
        return 0.0
    return sum(action == modal_action for action in sampled_actions) / len(sampled_actions)


def _mean_present(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
    return sum(vals) / len(vals) if vals else float("nan")


def _bool_rate_present(rows: list[dict[str, Any]], key: str) -> float:
    vals = [row[key] for row in rows if row.get(key) is not None and row.get(key) != ""]
    return sum(1 for value in vals if value is True) / len(vals) if vals else float("nan")


def _optimality_rate(rows: list[dict[str, Any]], key: str) -> float:
    vals = [row.get(key) for row in rows if row.get(key) is not None]
    if not vals:
        return 0.0
    return sum(1 for value in vals if value is True) / len(vals)


def _local_gap_rate(rows: list[dict[str, Any]], consistency_key: str = "blackbox_belief_action_consistency") -> tuple[int, int, float]:
    counts: Counter[str] = Counter()
    for row in rows:
        status = row.get(consistency_key)
        if status in {"inconsistent", "potentially_consistent"}:
            counts[str(status)] += 1
    gap = counts.get("inconsistent", 0)
    consistent = counts.get("potentially_consistent", 0)
    denom = gap + consistent
    return gap, denom, (gap / denom if denom else 0.0)


def _optimality_gap_rate(
    rows: list[dict[str, Any]],
    *,
    action_key: str,
    optimal_key: str,
    correct_key: str = "blackbox_correct",
) -> tuple[int, int, float]:
    gap = 0
    consistent = 0
    for row in rows:
        target_action = row["question_id"].removeprefix("wall_").upper()
        if row.get(action_key) != target_action:
            continue
        if row.get(correct_key) is not True:
            continue
        if row.get(optimal_key) is None:
            continue
        if row.get(optimal_key) is True:
            consistent += 1
        else:
            gap += 1
    denom = gap + consistent
    return gap, denom, (gap / denom if denom else 0.0)


def _plot_gradual_metrics(summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    xs = [int(row["reasoning_reveal_pct"]) for row in summary_rows]
    acc = [float(row["blackbox_wall_accuracy"]) for row in summary_rows]
    ent = [float(row["mean_blackbox_entropy"]) for row in summary_rows]
    local = [float(row["local_belief_action_gap_rate"]) for row in summary_rows]
    action_mc_ent = [float(row.get("action_mc_mean_entropy", 0.0)) for row in summary_rows]
    fig, axes = plt.subplots(4, 1, figsize=(8, 13), constrained_layout=True)
    axes[0].plot(xs, acc, marker="o")
    axes[0].set_title("Adjacent-wall belief accuracy vs revealed CoT")
    axes[0].set_xlabel("Revealed CoT (%)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].plot(xs, ent, marker="o")
    axes[1].set_title("Adjacent-wall belief entropy vs revealed CoT")
    axes[1].set_xlabel("Revealed CoT (%)")
    axes[1].set_ylabel("Mean entropy")
    axes[2].plot(xs, local, marker="o")
    axes[2].set_title("Chosen-direction wall-belief contradiction rate vs revealed CoT")
    axes[2].set_xlabel("Revealed CoT (%)")
    axes[2].set_ylabel("Gap rate")
    axes[2].set_ylim(0.0, 1.0)
    axes[3].plot(xs, action_mc_ent, marker="o")
    axes[3].set_title("Action MC entropy vs revealed CoT")
    axes[3].set_xlabel("Revealed CoT (%)")
    axes[3].set_ylabel("Mean entropy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_gradual_cot_alignment_rows(
    rows: list[dict[str, Any]],
    *,
    out_dir: Path | None = None,
    reveal_pcts: tuple[int, ...] = REVEAL_PCTS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_question_reveal: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_question_reveal[(str(row["question_id"]), int(row["reasoning_reveal_pct"]))].append(row)

    revised_rows: list[dict[str, Any]] = []
    has_whitebox = any(str(row.get("whitebox_prediction_pre", "")) or str(row.get("whitebox_prediction_post", "")) for row in rows)
    if has_whitebox:
        for question_id in sorted({str(row["question_id"]) for row in rows}):
            rows0 = by_question_reveal[(question_id, 0)]
            rows100 = by_question_reveal[(question_id, 100)]
            n = len(rows0)
            wb_pre = sum(1 for row in rows0 if row["whitebox_prediction_pre"] == row["ground_truth_label"]) / n if n else 0.0
            bb_0 = sum(1 for row in rows0 if row["blackbox_correct"] is True) / n if n else 0.0
            agr_pre_0 = sum(1 for row in rows0 if row["whitebox_prediction_pre"] == row["blackbox_answer"]) / n if n else 0.0
            wb_post = sum(1 for row in rows100 if row["whitebox_prediction_post"] == row["ground_truth_label"]) / n if n else 0.0
            bb_100 = sum(1 for row in rows100 if row["blackbox_correct"] is True) / n if n else 0.0
            agr_post_100 = sum(1 for row in rows100 if row["whitebox_prediction_post"] == row["blackbox_answer"]) / n if n else 0.0
            gap_0 = sum(1 for row in rows0 if row["blackbox_belief_action_consistency"] == "inconsistent")
            gap_100 = sum(1 for row in rows100 if row["blackbox_belief_action_consistency"] == "inconsistent")
            revised_rows.append(
                {
                    "question_id": question_id,
                    "n": n,
                    "WB_pre": wb_pre,
                    "BB_0": bb_0,
                    "Agr_pre_0": agr_pre_0,
                    "WB_post": wb_post,
                    "BB_100": bb_100,
                    "Agr_post_100": agr_post_100,
                    "ΔWB": wb_post - wb_pre,
                    "ΔBB": bb_100 - bb_0,
                    "Gap_0": gap_0,
                    "Gap_100": gap_100,
                }
            )

    gradual_rows: list[dict[str, Any]] = []
    for pct in reveal_pcts:
        rows_pct = [row for row in rows if int(row["reasoning_reveal_pct"]) == pct]
        n = len(rows_pct)
        gap_count, gap_denom, gap_rate = _local_gap_rate(rows_pct)
        a_gap_count, a_gap_denom, a_gap_rate = _optimality_gap_rate(
            rows_pct,
            action_key="blackbox_action",
            optimal_key="blackbox_action_is_optimal",
        )
        mc_gap_count, mc_gap_denom, mc_gap_rate = _local_gap_rate(rows_pct, "blackbox_belief_action_consistency_mc")
        mc_a_gap_count, mc_a_gap_denom, mc_a_gap_rate = _optimality_gap_rate(
            rows_pct,
            action_key="action_mc_modal_action",
            optimal_key="action_mc_modal_is_optimal",
        )
        gradual_rows.append(
            {
                "reasoning_reveal_pct": pct,
                "n_rows": n,
                "blackbox_wall_accuracy": sum(1 for row in rows_pct if row["blackbox_correct"] is True) / n if n else 0.0,
                "mean_blackbox_entropy": sum(float(row["blackbox_entropy"]) for row in rows_pct) / n if n else 0.0,
                "wall_greedy_modal_accuracy": _bool_rate_present(rows_pct, "wall_greedy_modal_correct"),
                "wall_greedy_mean_agreement_rate": _mean_present(rows_pct, "wall_greedy_agreement_rate"),
                "wall_mc_modal_accuracy": _bool_rate_present(rows_pct, "wall_mc_modal_correct"),
                "wall_mc_mean_entropy": _mean_present(rows_pct, "wall_mc_entropy"),
                "wall_mc_mean_agreement_rate": _mean_present(rows_pct, "wall_mc_agreement_rate"),
                "action_greedy_modal_optimality_rate": _optimality_rate(rows_pct, "blackbox_action_is_optimal"),
                "action_greedy_mean_agreement_rate": _mean_present(rows_pct, "action_greedy_agreement_rate"),
                "action_mc_modal_optimality_rate": _optimality_rate(rows_pct, "action_mc_modal_is_optimal"),
                "action_mc_mean_entropy": _mean_present(rows_pct, "action_mc_entropy"),
                "action_mc_mean_agreement_rate": _mean_present(rows_pct, "action_mc_agreement_rate"),
                "local_belief_action_gap_count": gap_count,
                "local_belief_action_gap_denominator": gap_denom,
                "local_belief_action_gap_rate": gap_rate,
                "astar_conditioned_gap_count": a_gap_count,
                "astar_conditioned_gap_denominator": a_gap_denom,
                "astar_conditioned_gap_rate": a_gap_rate,
                "mc_local_belief_action_gap_count": mc_gap_count,
                "mc_local_belief_action_gap_denominator": mc_gap_denom,
                "mc_local_belief_action_gap_rate": mc_gap_rate,
                "mc_astar_conditioned_gap_count": mc_a_gap_count,
                "mc_astar_conditioned_gap_denominator": mc_a_gap_denom,
                "mc_astar_conditioned_gap_rate": mc_a_gap_rate,
            }
        )

    failure_breakdown_rows: list[dict[str, Any]] = []
    grouped_failure_modes = {
        str(row.get("primary_step_failure_mode", "none"))
        for row in rows
        if str(row.get("primary_step_failure_mode", "none")) not in {"", "none"}
    }
    for mode in sorted(grouped_failure_modes):
        for pct in reveal_pcts:
            subset = [
                row for row in rows
                if str(row.get("primary_step_failure_mode", "none")) == mode
                and int(row["reasoning_reveal_pct"]) == pct
            ]
            if not subset:
                continue
            gap_count, gap_denom, gap_rate = _local_gap_rate(subset)
            a_gap_count, a_gap_denom, a_gap_rate = _optimality_gap_rate(
                subset,
                action_key="blackbox_action",
                optimal_key="blackbox_action_is_optimal",
            )
            failure_breakdown_rows.append(
                {
                    "failure_mode": mode,
                    "reasoning_reveal_pct": pct,
                    "n_rows": len(subset),
                    "blackbox_wall_accuracy": sum(1 for row in subset if row["blackbox_correct"] is True) / len(subset),
                    "mean_blackbox_entropy": sum(float(row["blackbox_entropy"]) for row in subset) / len(subset),
                    "action_greedy_modal_optimality_rate": _optimality_rate(subset, "blackbox_action_is_optimal"),
                    "local_belief_action_gap_rate": gap_rate,
                    "local_belief_action_gap_denominator": gap_denom,
                    "astar_conditioned_gap_rate": a_gap_rate,
                    "astar_conditioned_gap_denominator": a_gap_denom,
                }
            )

    if out_dir is not None:
        if revised_rows:
            _write_csv(out_dir / "revised_clean_alignment_table.csv", revised_rows)
            md_lines = [
                "| question_id | n | WB_pre | BB_0 | Agr_pre_0 | WB_post | BB_100 | Agr_post_100 | ΔWB | ΔBB | Gap_0 | Gap_100 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for row in revised_rows:
                md_lines.append(
                    "| {question_id} | {n} | {WB_pre:.3f} | {BB_0:.3f} | {Agr_pre_0:.3f} | {WB_post:.3f} | {BB_100:.3f} | {Agr_post_100:.3f} | {ΔWB:+.3f} | {ΔBB:+.3f} | {Gap_0} | {Gap_100} |".format(**row)
                )
            (out_dir / "revised_clean_alignment_table.md").write_text("\n".join(md_lines) + "\n")
        _write_csv(out_dir / "gradual_blackbox_reasoning_table.csv", gradual_rows)
        if failure_breakdown_rows:
            _write_csv(out_dir / "failure_mode_breakdown.csv", failure_breakdown_rows)
        _plot_gradual_metrics(gradual_rows, out_dir / "figs" / "gradual_blackbox_reasoning.png")

    return revised_rows, gradual_rows, failure_breakdown_rows


def _normalize_loaded_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    float_keys = {
        "blackbox_entropy",
        "wall_greedy_entropy",
        "wall_greedy_agreement_rate",
        "wall_mc_entropy",
        "wall_mc_agreement_rate",
        "action_greedy_entropy",
        "action_greedy_agreement_rate",
        "action_mc_entropy",
        "action_mc_agreement_rate",
        "sentence_boundary_coverage_pct",
    }
    int_keys = {"reasoning_reveal_pct", "analysis_total_tokens", "revealed_analysis_tokens", "sentence_boundary_index"}
    bool_keys = {
        "blackbox_parse_valid",
        "blackbox_correct",
        "wall_greedy_modal_correct",
        "wall_mc_modal_correct",
        "blackbox_action_is_optimal",
        "action_mc_modal_is_optimal",
        "blackbox_action_hits_wall",
        "action_mc_modal_hits_wall",
        "is_optimal_action",
        "wall_hit",
    }
    for row in rows:
        normalized = dict(row)
        for key in float_keys:
            if key in normalized and normalized[key] not in {"", None}:
                normalized[key] = float(normalized[key])
        for key in int_keys:
            if key in normalized and normalized[key] not in {"", None}:
                normalized[key] = int(float(normalized[key]))
        for key in bool_keys:
            if key in normalized:
                normalized[key] = _normalize_bool(normalized[key])
        normalized_rows.append(normalized)
    return normalized_rows


def load_and_resummarize_gradual_cot_alignment(
    *,
    rows_path: str | Path,
    output_dir: str | Path,
    reveal_pcts: tuple[int, ...] = REVEAL_PCTS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return summarize_gradual_cot_alignment_rows(
        _normalize_loaded_rows(_read_csv(rows_path)),
        out_dir=Path(output_dir),
        reveal_pcts=reveal_pcts,
    )


def _maybe_write_checkpoint(
    *,
    out_dir: Path,
    long_rows: list[dict[str, Any]],
    usage_summary: dict[str, Any],
    slice_mode: str,
    total_examples: int,
    completed_examples: int,
    reveal_pcts: tuple[int, ...],
    model_name: str,
    greedy_repeats: int,
    mc_sample_repeats: int,
    mc_temperature: float,
    tokenization_method: str,
    extra_manifest: dict[str, Any],
) -> None:
    _write_csv(out_dir / "gradual_cot_alignment_rows.csv", long_rows)
    _write_json(
        out_dir / "gradual_cot_alignment_status.json",
        {
            "status": "running" if completed_examples < total_examples else "completed",
            "slice_mode": slice_mode,
            "n_examples": total_examples,
            "completed_examples": completed_examples,
            "n_long_rows": len(long_rows),
            "reveal_pcts": list(reveal_pcts),
            "greedy_repeats": greedy_repeats,
            "mc_sample_repeats": mc_sample_repeats,
            "mc_temperature": mc_temperature,
            "tokenization_method": tokenization_method,
            "model_name": model_name,
            "usage_summary": usage_summary,
            **extra_manifest,
        },
    )


def _get_thread_client(*, model_name: str, temperature: float) -> BehavioralProbeLLM:
    cache = getattr(_THREAD_LOCAL, "llm_clients", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL.llm_clients = cache
    key = (model_name, float(temperature))
    client = cache.get(key)
    if client is None:
        client = BehavioralProbeLLM(model_name=model_name, temperature=float(temperature))
        cache[key] = client
    return client


def _request_action_once(*, model_name: str, temperature: float, prompt: str, seed: int) -> tuple[str, str, dict[str, Any]]:
    client = _get_thread_client(model_name=model_name, temperature=temperature)
    return _query_action(client, prompt, seed=seed)


def _request_wall_logprob_once(*, model_name: str, prompt: str, seed: int, top_logprobs: int) -> dict[str, Any]:
    client = _get_thread_client(model_name=model_name, temperature=0.0)
    return _collect_label3_logprob_readout(client=client, prompt=prompt, seed=seed, top_logprobs=top_logprobs)


def _request_wall_json_once(*, model_name: str, temperature: float, prompt: str, seed: int) -> tuple[str, str, dict[str, Any]]:
    client = _get_thread_client(model_name=model_name, temperature=temperature)
    raw_text, usage = _query_text(client, prompt, seed=seed)
    return _parse_wall_json_label(raw_text), raw_text, usage


def _completed_key_set(rows: list[dict[str, Any]]) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    for row in rows:
        try:
            keys.add((str(row["example_id"]), str(row["question_id"]), int(row["reasoning_reveal_pct"])))
        except Exception:
            continue
    return keys


def run_gradual_cot_blackbox_alignment(
    *,
    matched_wall_rows_path: str = DEFAULT_MATCHED_WALL_ROWS_PATH,
    failure_candidate_rows_path: str = DEFAULT_FAILURE_CANDIDATE_ROWS_PATH,
    trajectory_dir: str = DEFAULT_FAILURE_TRAJECTORY_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    top_logprobs: int = 20,
    reveal_pcts: tuple[int, ...] = REVEAL_PCTS,
    cache_dir: str = "data/hf/cache",
    slice_mode: str = "clean_matched",
    greedy_repeats: int = 10,
    mc_sample_repeats: int = 10,
    mc_temperature: float = 0.7,
    max_wall_hit: int = 4,
    max_avoidable_detour: int = 4,
    max_backtrack: int = 2,
    max_baseline: int = 4,
    trajectory_slice_type: str = "balanced_failure_modes",
    balanced_failure_rows_per_mode: int = 2,
    max_examples: int | None = None,
    max_concurrency: int = 8,
    wall_sampling_mode: str = "full",
    resume: bool = True,
    verbose: bool = True,
) -> None:
    if slice_mode not in {"clean_matched", "failure_wall_core", "trajectory_candidates"}:
        raise ValueError(f"Unsupported slice_mode: {slice_mode}")
    if wall_sampling_mode not in WALL_SAMPLING_MODES:
        raise ValueError(f"Unsupported wall_sampling_mode: {wall_sampling_mode}")

    if slice_mode == "clean_matched":
        pair_rows = _load_pair_rows_from_matched_wall_rows(matched_wall_rows_path)
    elif slice_mode == "failure_wall_core":
        pair_rows = _load_pair_rows_from_failure_candidates(
            failure_candidate_rows_path,
            max_wall_hit=max_wall_hit,
            max_avoidable_detour=max_avoidable_detour,
            max_backtrack=max_backtrack,
            max_baseline=max_baseline,
        )
    else:
        pair_rows = _load_pair_rows_from_trajectory_candidates(
            failure_candidate_rows_path,
            slice_type=trajectory_slice_type,
            rows_per_mode=balanced_failure_rows_per_mode,
            max_examples=max_examples,
        )

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    trajectory_root = Path(trajectory_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_helper = _TokenizerHelper()
    usage_summary: dict[str, Any] = {
        "model_name": model_name,
        "tokenization_method": tokenizer_helper.method,
        "by_phase": {},
        "total": _empty_usage_bucket(),
    }

    grouped_examples: dict[str, dict[str, Any]] = {}
    for pair in pair_rows:
        example = grouped_examples.setdefault(
            pair["example_id"],
            {
                "example_id": pair["example_id"],
                "trajectory_id": pair["trajectory_id"],
                "step_index": pair["step_index"],
                "grid_text": pair["grid_text"],
                "carrying_key": bool(pair.get("carrying_key") or False),
                "observed_action": pair.get("observed_action", ""),
                "optimal_actions_json": pair.get("optimal_actions_json", "[]"),
                "is_optimal_action": pair.get("is_optimal_action"),
                "wall_hit": pair.get("wall_hit"),
                "primary_step_failure_mode": pair.get("primary_step_failure_mode", "none"),
                "source_dataset": pair.get("source_dataset", ""),
                "questions": [],
            },
        )
        example["questions"].append(pair)

    rows_path = out_dir / "gradual_cot_alignment_rows.csv"
    long_rows: list[dict[str, Any]] = []
    completed_keys: set[tuple[str, str, int]] = set()
    if resume and rows_path.exists():
        long_rows = _normalize_loaded_rows(_read_csv(rows_path))
        completed_keys = _completed_key_set(long_rows)
        if verbose:
            print(f"Resuming from existing rows: {len(long_rows)}", flush=True)

    _maybe_write_checkpoint(
        out_dir=out_dir,
        long_rows=long_rows,
        usage_summary=usage_summary,
        slice_mode=slice_mode,
        total_examples=len(grouped_examples),
        completed_examples=len(
            {
                str(row["example_id"])
                for row in long_rows
            }
        ),
        reveal_pcts=reveal_pcts,
        model_name=model_name,
        greedy_repeats=greedy_repeats,
        mc_sample_repeats=mc_sample_repeats,
        mc_temperature=mc_temperature,
        tokenization_method=tokenizer_helper.method,
        extra_manifest={
            "matched_wall_rows_path": matched_wall_rows_path if slice_mode == "clean_matched" else "",
            "failure_candidate_rows_path": failure_candidate_rows_path if slice_mode in {"failure_wall_core", "trajectory_candidates"} else "",
            "trajectory_dir": trajectory_dir if slice_mode in {"failure_wall_core", "trajectory_candidates"} else "",
            "trajectory_slice_type": trajectory_slice_type if slice_mode == "trajectory_candidates" else "",
            "balanced_failure_rows_per_mode": balanced_failure_rows_per_mode if slice_mode == "trajectory_candidates" else "",
            "max_examples": max_examples if slice_mode == "trajectory_candidates" else "",
            "max_concurrency": max_concurrency,
            "wall_sampling_mode": wall_sampling_mode,
            "resume": resume,
        },
    )

    trajectory_cache: dict[str, dict[str, Any]] = {}
    total_examples = len(grouped_examples)
    processed_examples = 0

    for example_idx, (example_id, example_meta) in enumerate(sorted(grouped_examples.items()), start=1):
        example_expected_keys = {
            (example_id, pair["question_id"], pct)
            for pair in example_meta["questions"]
            for pct in reveal_pcts
        }
        if resume and example_expected_keys.issubset(completed_keys):
            processed_examples += 1
            if verbose:
                print(
                    f"Skipping completed example {processed_examples}/{total_examples}: {example_id}",
                    flush=True,
                )
            continue

        if verbose:
            print(f"Gradual-CoT example {example_idx}/{total_examples}: {example_id}", flush=True)
        trajectory_id = example_meta["trajectory_id"]
        if trajectory_id not in trajectory_cache:
            if slice_mode == "clean_matched":
                trajectory_cache[trajectory_id] = _load_hf_trajectory_payload(trajectory_id, cache_root)
            else:
                trajectory_cache[trajectory_id] = _load_local_trajectory_payload(trajectory_id, trajectory_root)
        payload = trajectory_cache[trajectory_id]
        step_index = int(example_meta["step_index"])
        step = payload["steps"][step_index]
        analysis_text, final_text = _extract_analysis_and_final(step["output_text"])
        optimal_actions = _parse_optimal_actions(example_meta["optimal_actions_json"])

        reveal_meta: dict[int, dict[str, Any]] = {}
        for pct in reveal_pcts:
            revealed_analysis, revealed_units, total_units = tokenizer_helper.reveal_prefix(analysis_text, pct)
            sentence_idx, sentence_cov = _sentence_boundary_metadata(analysis_text, revealed_analysis)
            reveal_meta[pct] = {
                "revealed_analysis": revealed_analysis,
                "analysis_total_tokens": total_units,
                "revealed_analysis_tokens": revealed_units,
                "sentence_boundary_index": sentence_idx,
                "sentence_boundary_coverage_pct": sentence_cov * 100.0,
                "final_text": final_text,
            }

        task_specs: list[dict[str, Any]] = []
        for pct in reveal_pcts:
            action_prompt = _action_prompt(
                grid_text=example_meta["grid_text"],
                carrying_key=bool(example_meta["carrying_key"]),
                revealed_analysis=reveal_meta[pct]["revealed_analysis"],
            )
            for idx in range(greedy_repeats):
                task_specs.append({
                    "kind": "action_greedy",
                    "pct": pct,
                    "index": idx,
                    "prompt": action_prompt,
                    "seed": 0,
                    "temperature": 0.0,
                })
            for idx in range(mc_sample_repeats):
                task_specs.append({
                    "kind": "action_mc",
                    "pct": pct,
                    "index": idx,
                    "prompt": action_prompt,
                    "seed": idx + 1,
                    "temperature": mc_temperature,
                })
            for pair in example_meta["questions"]:
                qid = pair["question_id"]
                task_specs.append({
                    "kind": "wall_logprob",
                    "pct": pct,
                    "question_id": qid,
                    "prompt": _wall_logprob_prompt(
                        grid_text=pair["grid_text"],
                        carrying_key=bool(pair.get("carrying_key") or False),
                        question_id=qid,
                        revealed_analysis=reveal_meta[pct]["revealed_analysis"],
                    ),
                    "seed": 0,
                })
                if wall_sampling_mode == "full":
                    wall_json_prompt = _wall_json_prompt(
                        grid_text=pair["grid_text"],
                        carrying_key=bool(pair.get("carrying_key") or False),
                        question_id=qid,
                        revealed_analysis=reveal_meta[pct]["revealed_analysis"],
                    )
                    for idx in range(greedy_repeats):
                        task_specs.append({
                            "kind": "wall_greedy",
                            "pct": pct,
                            "question_id": qid,
                            "index": idx,
                            "prompt": wall_json_prompt,
                            "seed": 0,
                            "temperature": 0.0,
                        })
                    for idx in range(mc_sample_repeats):
                        task_specs.append({
                            "kind": "wall_mc",
                            "pct": pct,
                            "question_id": qid,
                            "index": idx,
                            "prompt": wall_json_prompt,
                            "seed": idx + 1,
                            "temperature": mc_temperature,
                        })

        action_greedy_results: dict[int, list[tuple[str, str]]] = defaultdict(list)
        action_mc_results: dict[int, list[tuple[str, str]]] = defaultdict(list)
        wall_logprob_results: dict[tuple[int, str], dict[str, Any]] = {}
        wall_greedy_results: dict[tuple[int, str], list[tuple[str, str]]] = defaultdict(list)
        wall_mc_results: dict[tuple[int, str], list[tuple[str, str]]] = defaultdict(list)

        with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as pool:
            future_to_spec = {}
            for spec in task_specs:
                if spec["kind"] in {"action_greedy", "action_mc"}:
                    future = pool.submit(
                        _request_action_once,
                        model_name=model_name,
                        temperature=float(spec["temperature"]),
                        prompt=str(spec["prompt"]),
                        seed=int(spec["seed"]),
                    )
                elif spec["kind"] == "wall_logprob":
                    future = pool.submit(
                        _request_wall_logprob_once,
                        model_name=model_name,
                        prompt=str(spec["prompt"]),
                        seed=int(spec["seed"]),
                        top_logprobs=top_logprobs,
                    )
                else:
                    future = pool.submit(
                        _request_wall_json_once,
                        model_name=model_name,
                        temperature=float(spec["temperature"]),
                        prompt=str(spec["prompt"]),
                        seed=int(spec["seed"]),
                    )
                future_to_spec[future] = spec

            for future in as_completed(future_to_spec):
                spec = future_to_spec[future]
                if spec["kind"] == "wall_logprob":
                    readout = future.result()
                    wall_logprob_results[(int(spec["pct"]), str(spec["question_id"]))] = readout
                    _add_usage(usage_summary, "wall_logprob", readout.get("usage"))
                elif spec["kind"] in {"action_greedy", "action_mc"}:
                    label, raw_text, usage = future.result()
                    _add_usage(usage_summary, spec["kind"], usage)
                    target = action_greedy_results if spec["kind"] == "action_greedy" else action_mc_results
                    target[int(spec["pct"])] .append((label, raw_text))
                else:
                    label, raw_text, usage = future.result()
                    _add_usage(usage_summary, spec["kind"], usage)
                    target = wall_greedy_results if spec["kind"] == "wall_greedy" else wall_mc_results
                    target[(int(spec["pct"]), str(spec["question_id"]))].append((label, raw_text))

        reveal_action_cache: dict[int, dict[str, Any]] = {}
        for pct in reveal_pcts:
            action_greedy_pairs = action_greedy_results[pct]
            action_greedy_labels = [label for label, _ in action_greedy_pairs]
            action_greedy_raws = [raw for _, raw in action_greedy_pairs]
            action_greedy_probs = _action_probabilities(action_greedy_labels)
            action_greedy_modal = _sampled_action_from_probabilities(action_greedy_probs)
            action_mc_pairs = action_mc_results[pct]
            action_mc_labels = [label for label, _ in action_mc_pairs]
            action_mc_raws = [raw for _, raw in action_mc_pairs]
            action_mc_probs = _action_probabilities(action_mc_labels)
            action_mc_modal = _sampled_action_from_probabilities(action_mc_probs)
            reveal_action_cache[pct] = {
                **reveal_meta[pct],
                "blackbox_action": action_greedy_modal,
                "blackbox_action_raw": action_greedy_raws[0] if action_greedy_raws else "",
                "blackbox_action_is_optimal": action_greedy_modal in optimal_actions if optimal_actions else None,
                "action_greedy_actions_json": json.dumps(action_greedy_labels),
                "action_greedy_raw_outputs_json": json.dumps(action_greedy_raws),
                "action_greedy_probs_json": json.dumps(action_greedy_probs, sort_keys=True),
                "action_greedy_entropy": _action_entropy(action_greedy_probs),
                "action_greedy_agreement_rate": _action_agreement_rate(action_greedy_labels, action_greedy_modal),
                "action_mc_modal_action": action_mc_modal,
                "action_mc_modal_is_optimal": action_mc_modal in optimal_actions if optimal_actions else None,
                "action_mc_actions_json": json.dumps(action_mc_labels),
                "action_mc_raw_outputs_json": json.dumps(action_mc_raws),
                "action_mc_probs_json": json.dumps(action_mc_probs, sort_keys=True),
                "action_mc_entropy": _action_entropy(action_mc_probs),
                "action_mc_agreement_rate": _action_agreement_rate(action_mc_labels, action_mc_modal),
            }

        new_rows_for_example: list[dict[str, Any]] = []
        for pair in sorted(example_meta["questions"], key=lambda item: item["question_id"]):
            for pct in reveal_pcts:
                action_meta = reveal_action_cache[pct]
                readout = wall_logprob_results[(pct, pair["question_id"])]
                wall_greedy_pairs = wall_greedy_results.get((pct, pair["question_id"]), [])
                wall_greedy_labels = [label for label, _ in wall_greedy_pairs]
                wall_greedy_raws = [raw for _, raw in wall_greedy_pairs]
                wall_greedy_probs = _probabilities_from_mc_answers(wall_greedy_labels)
                wall_greedy_modal = sampled_answer_from_probabilities(wall_greedy_probs)
                wall_mc_pairs = wall_mc_results.get((pct, pair["question_id"]), [])
                wall_mc_labels = [label for label, _ in wall_mc_pairs]
                wall_mc_raws = [raw for _, raw in wall_mc_pairs]
                wall_mc_probs = _probabilities_from_mc_answers(wall_mc_labels)
                wall_mc_modal = sampled_answer_from_probabilities(wall_mc_probs)

                blackbox_visible_answer = str(readout["visible_answer"])
                blackbox_answer = str(readout["answer"])
                blackbox_correct = blackbox_answer == pair["ground_truth_label"]
                blackbox_action = action_meta["blackbox_action"]
                question_for_action = f"wall_{blackbox_action.lower()}" if blackbox_action != "INVALID" else ""
                wall_truth_for_action = pair["ground_truth_label"] if pair["question_id"] == question_for_action else None
                blackbox_action_hits_wall = wall_truth_for_action == "yes" if question_for_action == pair["question_id"] else None
                action_mc_question = f"wall_{action_meta['action_mc_modal_action'].lower()}" if action_meta["action_mc_modal_action"] != "INVALID" else ""
                wall_truth_for_mc_action = pair["ground_truth_label"] if pair["question_id"] == action_mc_question else None
                action_mc_modal_hits_wall = wall_truth_for_mc_action == "yes" if action_mc_question == pair["question_id"] else None

                new_rows_for_example.append(
                    {
                        "example_id": pair["example_id"],
                        "trajectory_id": pair["trajectory_id"],
                        "step_index": pair["step_index"],
                        "question_id": pair["question_id"],
                        "reasoning_reveal_pct": pct,
                        "analysis_total_tokens": action_meta["analysis_total_tokens"],
                        "revealed_analysis_tokens": action_meta["revealed_analysis_tokens"],
                        "analysis_tokenization_method": tokenizer_helper.method,
                        "sentence_boundary_index": action_meta["sentence_boundary_index"],
                        "sentence_boundary_coverage_pct": action_meta["sentence_boundary_coverage_pct"],
                        "grid_text": pair["grid_text"],
                        "carrying_key": bool(pair.get("carrying_key") or False),
                        "ground_truth_label": pair["ground_truth_label"],
                        "whitebox_prediction_pre": pair.get("whitebox_prediction_pre", ""),
                        "whitebox_prediction_post": pair.get("whitebox_prediction_post", ""),
                        "blackbox_answer": blackbox_answer,
                        "blackbox_visible_answer": blackbox_visible_answer,
                        "blackbox_parse_valid": blackbox_visible_answer in {"yes", "no", "unknown"},
                        "blackbox_answer_from_probs": readout["answer"],
                        "blackbox_correct": blackbox_correct,
                        "blackbox_entropy": readout["entropy"],
                        "blackbox_probs_json": json.dumps(readout["semantic_probs"], sort_keys=True),
                        "wall_greedy_answers_json": json.dumps(wall_greedy_labels) if wall_sampling_mode == "full" else "",
                        "wall_greedy_raw_outputs_json": json.dumps(wall_greedy_raws) if wall_sampling_mode == "full" else "",
                        "wall_greedy_modal_answer": wall_greedy_modal if wall_sampling_mode == "full" else "",
                        "wall_greedy_modal_correct": (wall_greedy_modal == pair["ground_truth_label"]) if wall_sampling_mode == "full" else None,
                        "wall_greedy_probs_json": json.dumps(wall_greedy_probs, sort_keys=True) if wall_sampling_mode == "full" else "",
                        "wall_greedy_entropy": shannon_entropy(wall_greedy_probs) if wall_sampling_mode == "full" else None,
                        "wall_greedy_agreement_rate": (sum(answer == wall_greedy_modal for answer in wall_greedy_labels) / len(wall_greedy_labels)) if wall_sampling_mode == "full" and wall_greedy_labels else None,
                        "wall_mc_answers_json": json.dumps(wall_mc_labels) if wall_sampling_mode == "full" else "",
                        "wall_mc_raw_outputs_json": json.dumps(wall_mc_raws) if wall_sampling_mode == "full" else "",
                        "wall_mc_modal_answer": wall_mc_modal if wall_sampling_mode == "full" else "",
                        "wall_mc_modal_correct": (wall_mc_modal == pair["ground_truth_label"]) if wall_sampling_mode == "full" else None,
                        "wall_mc_probs_json": json.dumps(wall_mc_probs, sort_keys=True) if wall_sampling_mode == "full" else "",
                        "wall_mc_entropy": shannon_entropy(wall_mc_probs) if wall_sampling_mode == "full" else None,
                        "wall_mc_agreement_rate": (sum(answer == wall_mc_modal for answer in wall_mc_labels) / len(wall_mc_labels)) if wall_sampling_mode == "full" and wall_mc_labels else None,
                        "blackbox_action": blackbox_action,
                        "blackbox_action_raw": action_meta["blackbox_action_raw"],
                        "blackbox_action_is_optimal": action_meta["blackbox_action_is_optimal"],
                        "blackbox_action_hits_wall": blackbox_action_hits_wall,
                        "action_greedy_actions_json": action_meta["action_greedy_actions_json"],
                        "action_greedy_raw_outputs_json": action_meta["action_greedy_raw_outputs_json"],
                        "action_greedy_probs_json": action_meta["action_greedy_probs_json"],
                        "action_greedy_entropy": action_meta["action_greedy_entropy"],
                        "action_greedy_agreement_rate": action_meta["action_greedy_agreement_rate"],
                        "action_mc_modal_action": action_meta["action_mc_modal_action"],
                        "action_mc_modal_is_optimal": action_meta["action_mc_modal_is_optimal"],
                        "action_mc_modal_hits_wall": action_mc_modal_hits_wall,
                        "action_mc_actions_json": action_meta["action_mc_actions_json"],
                        "action_mc_raw_outputs_json": action_meta["action_mc_raw_outputs_json"],
                        "action_mc_probs_json": action_meta["action_mc_probs_json"],
                        "action_mc_entropy": action_meta["action_mc_entropy"],
                        "action_mc_agreement_rate": action_meta["action_mc_agreement_rate"],
                        "blackbox_belief_action_consistency": belief_action_consistency(blackbox_answer, blackbox_action, pair["question_id"]),
                        "blackbox_belief_action_consistency_mc": belief_action_consistency(blackbox_answer, action_meta["action_mc_modal_action"], pair["question_id"]),
                        "observed_action": pair["observed_action"],
                        "optimal_actions_json": pair["optimal_actions_json"],
                        "is_optimal_action": pair["is_optimal_action"],
                        "wall_hit": pair["wall_hit"],
                        "primary_step_failure_mode": pair.get("primary_step_failure_mode", "none"),
                        "source_dataset": pair.get("source_dataset", ""),
                    }
                )

        long_rows = [
            row for row in long_rows
            if str(row["example_id"]) != example_id
        ]
        long_rows.extend(new_rows_for_example)
        completed_keys.update(_completed_key_set(new_rows_for_example))
        processed_examples += 1
        _maybe_write_checkpoint(
            out_dir=out_dir,
            long_rows=long_rows,
            usage_summary=usage_summary,
            slice_mode=slice_mode,
            total_examples=total_examples,
            completed_examples=processed_examples,
            reveal_pcts=reveal_pcts,
            model_name=model_name,
            greedy_repeats=greedy_repeats,
            mc_sample_repeats=mc_sample_repeats,
            mc_temperature=mc_temperature,
            tokenization_method=tokenizer_helper.method,
            extra_manifest={
                "matched_wall_rows_path": matched_wall_rows_path if slice_mode == "clean_matched" else "",
                "failure_candidate_rows_path": failure_candidate_rows_path if slice_mode in {"failure_wall_core", "trajectory_candidates"} else "",
                "trajectory_dir": trajectory_dir if slice_mode in {"failure_wall_core", "trajectory_candidates"} else "",
                "trajectory_slice_type": trajectory_slice_type if slice_mode == "trajectory_candidates" else "",
                "balanced_failure_rows_per_mode": balanced_failure_rows_per_mode if slice_mode == "trajectory_candidates" else "",
                "max_examples": max_examples if slice_mode == "trajectory_candidates" else "",
                "max_concurrency": max_concurrency,
                "wall_sampling_mode": wall_sampling_mode,
                "resume": resume,
            },
        )

    revised_rows, gradual_rows, failure_breakdown_rows = summarize_gradual_cot_alignment_rows(
        long_rows,
        out_dir=out_dir,
        reveal_pcts=reveal_pcts,
    )

    _write_json(
        out_dir / "gradual_cot_alignment_status.json",
        {
            "status": "completed",
            "slice_mode": slice_mode,
            "matched_wall_rows_path": matched_wall_rows_path if slice_mode == "clean_matched" else "",
            "failure_candidate_rows_path": failure_candidate_rows_path if slice_mode in {"failure_wall_core", "trajectory_candidates"} else "",
            "trajectory_dir": trajectory_dir if slice_mode in {"failure_wall_core", "trajectory_candidates"} else "",
            "trajectory_slice_type": trajectory_slice_type if slice_mode == "trajectory_candidates" else "",
            "balanced_failure_rows_per_mode": balanced_failure_rows_per_mode if slice_mode == "trajectory_candidates" else "",
            "max_examples": max_examples if slice_mode == "trajectory_candidates" else "",
            "model_name": model_name,
            "reveal_pcts": list(reveal_pcts),
            "greedy_repeats": greedy_repeats,
            "mc_sample_repeats": mc_sample_repeats,
            "mc_temperature": mc_temperature,
            "n_examples": len(grouped_examples),
            "completed_examples": len(grouped_examples),
            "n_long_rows": len(long_rows),
            "n_revised_rows": len(revised_rows),
            "n_gradual_rows": len(gradual_rows),
            "n_failure_breakdown_rows": len(failure_breakdown_rows),
            "tokenization_method": tokenizer_helper.method,
            "usage_summary": usage_summary,
            "max_concurrency": max_concurrency,
            "wall_sampling_mode": wall_sampling_mode,
            "resume": resume,
        },
    )


__all__ = [
    "_extract_analysis_and_final",
    "_sentence_boundary_metadata",
    "load_and_resummarize_gradual_cot_alignment",
    "run_gradual_cot_blackbox_alignment",
    "summarize_gradual_cot_alignment_rows",
]
