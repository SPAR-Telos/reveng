"""Observational analysis of belief changes and action changes during reasoning."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Sequence

from scipy.stats import fisher_exact

from reveng.experiments.behavioral_probe_parse import (
    coordinate_manhattan_distance,
    coordinate_to_key,
    parse_behavioral_probe_answer,
    parse_coordinate_answer,
)
from reveng.experiments.behavioral_probe_questions import (
    BehavioralProbeQuestion,
    get_behavioral_probe_questions,
)
from reveng.experiments.behavioral_probe_runner import (
    BehavioralProbeLLM,
    _belief_prompt,
    _collect_label3_logprob_readout,
    _empty_usage_bucket,
    _query_text,
)

DEFAULT_CANDIDATE_ROWS = (
    "data/behavioral_probes/trajectory_instances_recomputed_optimal/"
    "trajectory_selection_candidates.csv"
)
DEFAULT_DRIFT_RUN_DIR = "data/behavioral_probes/step_reasoning_drift_balanced_v1"
DEFAULT_OUTPUT_DIR = "data/behavioral_probes/reasoning_belief_action_balanced_v1"
DEFAULT_MODEL_NAME = "together_ai/openai/gpt-oss-20b"
PRIMARY_QUESTION_IDS = (
    "wall_left",
    "wall_right",
    "wall_up",
    "wall_down",
    "has_key",
    "door_open",
)
COORDINATE_QUESTION_IDS = (
    "agent_location",
    "goal_location",
    "key_location",
    "door_location",
)
ACTION_EFFECT_PREFIXES = ("hit_wall_after_", "has_key_after_", "door_open_after_")
VALID_ACTIONS = {"UP", "DOWN", "LEFT", "RIGHT"}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint_rows(path: Path) -> tuple[dict[tuple[str, int, str], dict[str, Any]], int]:
    """Load successful rows preferentially and tolerate an interrupted final line."""
    latest: dict[tuple[str, int, str], dict[str, Any]] = {}
    malformed_lines = 0
    if not path.exists():
        return latest, malformed_lines
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = (str(row["example_id"]), int(row["reasoning_step_idx"]), str(row["question_id"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            malformed_lines += 1
            continue
        if not row.get("query_error"):
            latest[key] = row
        elif key not in latest or latest[key].get("query_error"):
            latest[key] = row
    return latest, malformed_lines


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _json(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _cohort_key(row: dict[str, Any]) -> tuple[bool, bool, int, int]:
    distance = int(float(row.get("current_optimal_distance") or 0))
    return (
        _as_bool(row.get("carrying_key")),
        "D" in str(row.get("grid_text", "")),
        len(_json(row.get("optimal_actions_json"), [])),
        distance // 5,
    )


def prepare_reasoning_belief_action_cohorts(
    *,
    candidate_rows_path: str = DEFAULT_CANDIDATE_ROWS,
    output_dir: str = "data/behavioral_probes/reasoning_belief_action_cohorts",
) -> dict[str, Any]:
    """Write pilot, exact-matched, and expanded candidate CSVs."""
    rows = [row for row in _read_csv(candidate_rows_path) if _as_bool(row.get("selected_for_probe"))]
    failures = [row for row in rows if row.get("selection_stage") == "failure"]
    controls = [
        row for row in rows
        if row.get("selection_stage") == "context"
        and str(row.get("primary_step_failure_mode", "none") or "none") == "none"
    ]
    by_key_failure: dict[tuple[bool, bool, int, int], list[dict[str, Any]]] = defaultdict(list)
    by_key_control: dict[tuple[bool, bool, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in failures:
        by_key_failure[_cohort_key(row)].append(row)
    for row in controls:
        by_key_control[_cohort_key(row)].append(row)
    matched: list[dict[str, Any]] = []
    pair_idx = 0
    for key in sorted(set(by_key_failure) & set(by_key_control)):
        left = sorted(by_key_failure[key], key=lambda row: row["example_id"])
        right = sorted(by_key_control[key], key=lambda row: row["example_id"])
        for failure, control in zip(left, right):
            pair_idx += 1
            for role, row in (("failure", failure), ("control", control)):
                matched.append(
                    {
                        **row,
                        "matched_pair_id": f"pair_{pair_idx:03d}",
                        "matched_role": role,
                        "match_carrying_key": key[0],
                        "match_closed_door_visible": key[1],
                        "match_n_optimal_actions": key[2],
                        "match_distance_bin_5": key[3],
                    }
                )
    out = Path(output_dir)
    _write_csv(out / "matched_46_candidates.csv", matched)
    _write_csv(out / "expanded_185_candidates.csv", rows)
    manifest = {
        "candidate_rows_path": candidate_rows_path,
        "n_selected_candidates": len(rows),
        "n_failure_states": len(failures),
        "n_control_states": len(controls),
        "n_exact_matched_pairs": pair_idx,
        "n_exact_matched_states": len(matched),
        "matching_variables": [
            "carrying_key",
            "closed_door_visible",
            "number_of_optimal_actions",
            "current_optimal_distance_bin_width_5",
        ],
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


def _selected_questions(action: str) -> list[BehavioralProbeQuestion]:
    questions = get_behavioral_probe_questions(question_family="all")
    allowed = set(PRIMARY_QUESTION_IDS) | set(COORDINATE_QUESTION_IDS)
    if action in VALID_ACTIONS:
        allowed |= {f"{prefix}{action.lower()}" for prefix in ACTION_EFFECT_PREFIXES}
    return [question for question in questions if question.question_id in allowed]


def _state_description(meta: dict[str, Any], revealed_analysis: str) -> str:
    return (
        "Current grid state:\n\n"
        + str(meta["grid_text"])
        + "\n\nAgent status:\n- Carrying key: "
        + str(_as_bool(meta.get("carrying_key"))).lower()
        + "\n\nReasoning trace available so far:\n"
        + revealed_analysis.strip()
    )


def _answer_key(question: BehavioralProbeQuestion, raw_text: str) -> tuple[str, bool, int | str]:
    if question.answer_space == "coord_json":
        parsed = parse_coordinate_answer(raw_text)
        return coordinate_to_key(parsed), parsed is not None, ""
    parsed = parse_behavioral_probe_answer(raw_text)
    return parsed, parsed != "invalid", ""


def _truth_key(question: BehavioralProbeQuestion, truth: Any) -> str:
    if question.answer_space == "coord_json":
        return coordinate_to_key(truth)
    return str(truth)


def _is_persistent(flags: Sequence[bool | None], index: int) -> bool:
    window = [value for value in flags[index:] if value is not None][:3]
    return len(window) >= 2 and sum(value is True for value in window) >= 2


def classify_action_events(position_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in position_rows:
        by_example[str(row["example_id"])].append(row)
    for example_id, rows in by_example.items():
        ordered = sorted(rows, key=lambda row: int(row["reasoning_step_idx"]))
        flags = [row.get("action_is_optimal") for row in ordered]
        valid_flags = [flag if isinstance(flag, bool) else None for flag in flags]
        for idx in range(1, len(ordered)):
            previous, current = valid_flags[idx - 1], valid_flags[idx]
            event_type = ""
            if previous is True and current is False:
                event_type = (
                    "sustained_optimal_to_suboptimal"
                    if _is_persistent(
                        [None if flag is None else flag is False for flag in valid_flags],
                        idx,
                    )
                    else "transient_optimal_to_suboptimal"
                )
            elif previous is False and current is True:
                event_type = "suboptimal_to_optimal_recovery"
            elif (
                previous is not None
                and current is not None
                and ordered[idx - 1]["action_label"] != ordered[idx]["action_label"]
            ):
                event_type = "action_identity_change_without_optimality_change"
            if not event_type:
                continue
            events.append(
                {
                    "event_id": f"{example_id}:action:{ordered[idx]['reasoning_step_idx']}",
                    "example_id": example_id,
                    "trajectory_id": ordered[idx]["trajectory_id"],
                    "failure_category": ordered[idx]["failure_category"],
                    "reasoning_step_idx": ordered[idx]["reasoning_step_idx"],
                    "reasoning_progress": ordered[idx]["reasoning_progress"],
                    "reasoning_character_progress": ordered[idx].get(
                        "reasoning_character_progress", ""
                    ),
                    "analysis_unit": ordered[idx].get("analysis_unit", "packed_chunk"),
                    "event_type": event_type,
                    "previous_action": ordered[idx - 1]["action_label"],
                    "current_action": ordered[idx]["action_label"],
                    "strictly_sustained_to_trace_end": (
                        current is False
                        and all(flag is False for flag in valid_flags[idx:] if flag is not None)
                    ),
                }
            )
    return events


def classify_belief_shifts(belief_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shifts: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in belief_rows:
        grouped[(str(row["example_id"]), str(row["question_id"]))].append(row)
    for (example_id, question_id), rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["reasoning_step_idx"]))
        errors = [
            bool(row["belief_is_error"]) if bool(row["answer_valid"]) else None
            for row in ordered
        ]
        for idx in range(1, len(ordered)):
            previous, current = ordered[idx - 1], ordered[idx]
            if previous["answer_key"] == current["answer_key"]:
                continue
            if not previous["answer_valid"] or not current["answer_valid"]:
                shift_type = "invalid_boundary"
            elif not previous["belief_is_error"] and current["belief_is_error"]:
                shift_type = "belief_error_onset"
            elif previous["belief_is_error"] and not current["belief_is_error"]:
                shift_type = "belief_error_recovery"
            else:
                shift_type = "answer_change_without_error_change"
            shifts.append(
                {
                    "shift_id": f"{example_id}:{question_id}:{current['reasoning_step_idx']}",
                    "example_id": example_id,
                    "trajectory_id": current["trajectory_id"],
                    "failure_category": current["failure_category"],
                    "reasoning_step_idx": current["reasoning_step_idx"],
                    "reasoning_progress": current["reasoning_progress"],
                    "reasoning_character_progress": current.get(
                        "reasoning_character_progress", ""
                    ),
                    "analysis_unit": current.get("analysis_unit", "packed_chunk"),
                    "question_id": question_id,
                    "question_family": current["question_family"],
                    "previous_answer": previous["answer_key"],
                    "current_answer": current["answer_key"],
                    "ground_truth": current["ground_truth_key"],
                    "shift_type": shift_type,
                    "persistent_belief_error": (
                        shift_type == "belief_error_onset" and _is_persistent(errors, idx)
                    ),
                }
            )
    return shifts


def _build_lead_lag(
    action_events: list[dict[str, Any]],
    belief_shifts: list[dict[str, Any]],
    *,
    event_window: int,
) -> list[dict[str, Any]]:
    shifts_by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in belief_shifts:
        shifts_by_example[str(row["example_id"])].append(row)
    rows: list[dict[str, Any]] = []
    for event in action_events:
        if event["event_type"] == "action_identity_change_without_optimality_change":
            continue
        for shift in shifts_by_example.get(str(event["example_id"]), []):
            lag = int(shift["reasoning_step_idx"]) - int(event["reasoning_step_idx"])
            if abs(lag) > event_window:
                continue
            rows.append(
                {
                    **event,
                    "belief_shift_id": shift["shift_id"],
                    "question_id": shift["question_id"],
                    "question_family": shift["question_family"],
                    "shift_type": shift["shift_type"],
                    "persistent_belief_error": shift["persistent_belief_error"],
                    "belief_shift_step_idx": shift["reasoning_step_idx"],
                    "belief_shift_minus_action_event_steps": lag,
                    "temporal_relation": "leads" if lag < 0 else "coincides" if lag == 0 else "lags",
                }
            )
    return rows


def _bh_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=lambda idx: p_values[idx])
    adjusted = [1.0] * len(p_values)
    running = 1.0
    for rank_idx in range(len(order) - 1, -1, -1):
        original_idx = order[rank_idx]
        rank = rank_idx + 1
        running = min(running, p_values[original_idx] * len(order) / rank)
        adjusted[original_idx] = min(1.0, running)
    return adjusted


def _feature_associations(
    position_rows: list[dict[str, Any]],
    belief_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    next_suboptimal: dict[tuple[str, int], bool] = {}
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in position_rows:
        by_example[str(row["example_id"])].append(row)
    for example_id, rows in by_example.items():
        ordered = sorted(rows, key=lambda row: int(row["reasoning_step_idx"]))
        for current, following in zip(ordered, ordered[1:]):
            next_suboptimal[(example_id, int(current["reasoning_step_idx"]))] = (
                following.get("action_is_optimal") is False
            )
    results: list[dict[str, Any]] = []
    for question_id in sorted({str(row["question_id"]) for row in belief_rows}):
        rows = [
            row for row in belief_rows
            if row["question_id"] == question_id
            and bool(row["answer_valid"])
            and (str(row["example_id"]), int(row["reasoning_step_idx"])) in next_suboptimal
        ]
        a = sum(
            bool(row["belief_is_error"])
            and next_suboptimal[(str(row["example_id"]), int(row["reasoning_step_idx"]))]
            for row in rows
        )
        b = sum(bool(row["belief_is_error"]) for row in rows) - a
        c = sum(
            not bool(row["belief_is_error"])
            and next_suboptimal[(str(row["example_id"]), int(row["reasoning_step_idx"]))]
            for row in rows
        )
        d = len(rows) - a - b - c
        odds_ratio, p_value = fisher_exact([[a, b], [c, d]])
        results.append(
            {
                "question_id": question_id,
                "question_family": rows[0]["question_family"] if rows else "",
                "n_positions": len(rows),
                "error_and_next_suboptimal": a,
                "error_and_next_optimal": b,
                "correct_and_next_suboptimal": c,
                "correct_and_next_optimal": d,
                "odds_ratio": odds_ratio,
                "fisher_exact_p": p_value,
            }
        )
    adjusted = _bh_adjust([float(row["fisher_exact_p"]) for row in results])
    for row, value in zip(results, adjusted):
        row["fdr_bh_q"] = value
    return results


def _plot_outputs(
    *,
    out_dir: Path,
    position_rows: list[dict[str, Any]],
    belief_rows: list[dict[str, Any]],
    action_events: list[dict[str, Any]],
    geometry_rows: list[dict[str, str]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figs = out_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    analysis_units = {
        str(row.get("analysis_unit", "packed_chunk")) for row in position_rows
    }
    analysis_unit = next(iter(analysis_units)) if len(analysis_units) == 1 else "analysis unit"
    unit_label = "sentence" if analysis_unit == "sentence" else "packed reasoning unit"
    error_counts: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for row in belief_rows:
        if row["question_id"] in PRIMARY_QUESTION_IDS:
            error_counts[(str(row["example_id"]), int(row["reasoning_step_idx"]))].append(
                _as_bool(row["belief_is_error"])
            )
    event_types = (
        "sustained_optimal_to_suboptimal",
        "transient_optimal_to_suboptimal",
        "suboptimal_to_optimal_recovery",
    )
    event_labels = {
        "sustained_optimal_to_suboptimal": "Sustained: optimal to suboptimal",
        "transient_optimal_to_suboptimal": "Transient: optimal to suboptimal",
        "suboptimal_to_optimal_recovery": "Recovery: suboptimal to optimal",
    }
    event_counts = Counter(event["event_type"] for event in action_events)
    palette = ("#0B4F8A", "#5C9DCE", "#B6D7EE")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for event_type, color in zip(event_types, palette):
        by_offset: dict[int, list[float]] = defaultdict(list)
        for event in action_events:
            if event["event_type"] != event_type:
                continue
            center = int(event["reasoning_step_idx"])
            for offset in range(-3, 4):
                values = error_counts.get((str(event["example_id"]), center + offset), [])
                if values:
                    by_offset[offset].append(sum(values) / len(values))
        xs = sorted(by_offset)
        if xs:
            ax.plot(
                xs,
                [mean(by_offset[x]) for x in xs],
                marker="o",
                color=color,
                label=f"{event_labels[event_type]} (n={event_counts[event_type]})",
            )
    ax.axvline(0, color="#15324B", linewidth=1, linestyle="--")
    ax.set(
        xlabel=f"{unit_label.capitalize()} offset (0 = first prefix with changed action optimality)",
        ylabel="Mean error rate across wall, key, and door beliefs",
        title="Wall, Key, and Door Belief Errors Before and After Optimality Changes",
    )
    ax.grid(axis="y", color="#d7e3f0", linewidth=0.8)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)
    fig.tight_layout()
    fig.savefig(figs / "event_aligned_belief_errors.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if not geometry_rows:
        return
    event_class: dict[str, str] = {}
    for event in action_events:
        if event["event_type"] == "sustained_optimal_to_suboptimal":
            event_class[str(event["example_id"])] = "sustained transition"
        elif event["event_type"] == "transient_optimal_to_suboptimal" and str(event["example_id"]) not in event_class:
            event_class[str(event["example_id"])] = "transient transition"
        elif event["event_type"] == "suboptimal_to_optimal_recovery" and str(event["example_id"]) not in event_class:
            event_class[str(event["example_id"])] = "recovery"
    for row in position_rows:
        event_class.setdefault(str(row["example_id"]), "no optimality loss")
    selected = [
        row for row in geometry_rows
        if row.get("representation_kind") == "mean_pool"
        and row.get("layer") == "15"
        and row.get("aligned_change") not in {"", None}
    ]
    classes = ("sustained transition", "transient transition", "recovery", "no optimality loss")
    values = [
        [
            float(row["aligned_change"]) for row in selected
            if event_class.get(str(row["example_id"])) == label
        ]
        for label in classes
    ]
    if any(values):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.boxplot(values, tick_labels=classes, patch_artist=True)
        for patch, color in zip(ax.patches, ("#0B4F8A", "#5C9DCE", "#8EC1DF", "#D9ECF7")):
            patch.set_facecolor(color)
        ax.set_ylabel("Aligned activation change, layer 15 mean-pooled")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(figs / "activation_geometry_by_action_event.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def run_reasoning_belief_action_observational(
    *,
    drift_run_dir: str = DEFAULT_DRIFT_RUN_DIR,
    candidate_rows_path: str = DEFAULT_CANDIDATE_ROWS,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    prompt_preset: str = "cardinal_action_explicit",
    event_window: int = 3,
    max_positions: int | None = None,
    checkpoint_fsync_every: int = 10,
    resume: bool = True,
    verbose: bool = True,
    max_workers: int = 1,
    belief_query_fn: Callable[[str, int], tuple[str, dict[str, Any]]] | None = None,
    categorical_logprob_temperature: float | None = None,
    categorical_top_logprobs: int = 20,
) -> dict[str, Any]:
    """Query prefix-conditioned beliefs and analyze their relationship to actions."""
    drift = Path(drift_run_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix_rows = _read_csv(drift / "prefix_action_rows.csv")
    if max_positions is not None:
        prefix_rows = prefix_rows[: max(0, int(max_positions))]
    candidate_by_id = {row["example_id"]: row for row in _read_csv(candidate_rows_path)}
    analysis_units = {str(row.get("analysis_unit", "packed_chunk")) for row in prefix_rows}
    if len(analysis_units) != 1:
        raise ValueError(f"Expected one analysis unit in prefix rows; found {sorted(analysis_units)}")
    analysis_unit = next(iter(analysis_units))
    thread_local = threading.local()
    if belief_query_fn is None:
        def belief_query_fn(prompt: str, seed: int) -> tuple[str, dict[str, Any]]:
            client = getattr(thread_local, "coordinate_client", None)
            if client is None:
                client = BehavioralProbeLLM(model_name=model_name, temperature=0.0)
                thread_local.coordinate_client = client
            return _query_text(client, prompt, seed=seed)

    checkpoint = out / "belief_rows.jsonl"
    status_path = out / "run_status.json"
    config_path = out / "run_config.json"
    run_config = {
        "drift_run_dir": str(drift.resolve()),
        "prefix_action_rows_sha256": _file_sha256(drift / "prefix_action_rows.csv"),
        "candidate_rows_path": str(Path(candidate_rows_path).resolve()),
        "candidate_rows_sha256": _file_sha256(candidate_rows_path),
        "model_name": model_name,
        "prompt_preset": prompt_preset,
        "question_panel": "core_six_coordinates_and_recommended_action_effects_v1",
        "analysis_unit": analysis_unit,
        "categorical_logprob_temperature": categorical_logprob_temperature,
        "categorical_top_logprobs": (
            categorical_top_logprobs
            if categorical_logprob_temperature is not None
            else None
        ),
        "categorical_candidate_labels": (
            ["A", "B", "C"] if categorical_logprob_temperature is not None else []
        ),
    }
    if resume and config_path.exists():
        existing_config = json.loads(config_path.read_text())
        if existing_config != run_config:
            raise ValueError(
                "Cannot resume because the saved run configuration differs from the requested run. "
                "Use a new output directory or pass --no-resume."
            )
    else:
        _write_json_atomic(config_path, run_config)

    def build_prompt_and_identity(
        position: dict[str, str], question: BehavioralProbeQuestion
    ) -> tuple[str, str]:
        meta = candidate_by_id.get(position["example_id"])
        if meta is None:
            raise ValueError(f"Missing candidate metadata for {position['example_id']}")
        prompt = _belief_prompt(
            grid_text=meta["grid_text"],
            carrying_key=_as_bool(meta.get("carrying_key")),
            question=question,
            prompt_preset=prompt_preset,
            state_description_text=_state_description(
                meta, position["revealed_analysis_text"]
            ),
        )
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        identity = hashlib.sha256(
            json.dumps(
                {
                    "prompt_sha256": prompt_sha256,
                    "model_name": model_name,
                    "temperature": (
                        categorical_logprob_temperature
                        if question.answer_space == "label3"
                        else 0.0
                    ),
                    "top_logprobs": (
                        categorical_top_logprobs
                        if question.answer_space == "label3"
                        and categorical_logprob_temperature is not None
                        else None
                    ),
                    "candidate_labels": (
                        ["A", "B", "C"]
                        if question.answer_space == "label3"
                        else []
                    ),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        return prompt, identity

    planned_identity: dict[tuple[str, int, str], str] = {}
    for position in prefix_rows:
        for question in _selected_questions(str(position["action_label"]).upper()):
            key = (
                str(position["example_id"]),
                int(position["reasoning_step_idx"]),
                question.question_id,
            )
            _prompt, identity = build_prompt_and_identity(position, question)
            planned_identity[key] = identity
    planned_keys = set(planned_identity)
    latest_rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    malformed_checkpoint_lines = 0
    if resume:
        latest_rows, malformed_checkpoint_lines = _load_checkpoint_rows(checkpoint)
        latest_rows = {
            key: row
            for key, row in latest_rows.items()
            if key in planned_keys
            and row.get("checkpoint_identity_sha256") == planned_identity[key]
        }
    elif checkpoint.exists():
        checkpoint.write_text("")
    if checkpoint.exists() and checkpoint.stat().st_size:
        with checkpoint.open("rb+") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
    completed = {key for key, row in latest_rows.items() if not row.get("query_error")}
    total_planned_queries = len(planned_keys)
    checkpoint_fsync_every = max(1, int(checkpoint_fsync_every))
    max_workers = max(1, int(max_workers))
    usage = _empty_usage_bucket()
    attempted_this_invocation = 0

    def write_status(status: str, *, current_key: tuple[str, int, str] | None = None) -> None:
        _write_json_atomic(
            status_path,
            {
                "status": status,
                "total_planned_queries": total_planned_queries,
                "successful_queries": len(completed),
                "remaining_queries": total_planned_queries - len(completed),
                "failed_queries_recorded": sum(bool(row.get("query_error")) for row in latest_rows.values()),
                "malformed_checkpoint_lines_ignored": malformed_checkpoint_lines,
                "attempted_this_invocation": attempted_this_invocation,
                "current_key": list(current_key) if current_key else None,
                "resume_enabled": resume,
            },
        )

    write_status("running")
    current_key: tuple[str, int, str] | None = None

    def run_single_query(
        position: dict[str, str],
        question: BehavioralProbeQuestion,
    ) -> tuple[tuple[str, int, str], dict[str, Any], dict[str, Any]]:
        example_id = position["example_id"]
        meta = candidate_by_id.get(example_id)
        if meta is None:
            raise ValueError(f"Missing candidate metadata for {example_id}")
        truths = _json(meta.get("probe_truths_json"), {})
        action = str(position["action_label"]).upper()
        key = (example_id, int(position["reasoning_step_idx"]), question.question_id)
        prompt, checkpoint_identity_sha256 = build_prompt_and_identity(
            position, question
        )
        raw_text = ""
        query_error = ""
        query_usage: dict[str, Any] = {}
        categorical_fields: dict[str, Any] = {}
        try:
            if (
                question.answer_space == "label3"
                and categorical_logprob_temperature is not None
            ):
                client = getattr(thread_local, "categorical_client", None)
                if client is None:
                    client = BehavioralProbeLLM(
                        model_name=model_name,
                        temperature=categorical_logprob_temperature,
                    )
                    thread_local.categorical_client = client
                readout = _collect_label3_logprob_readout(
                    client=client,
                    prompt=prompt,
                    seed=0,
                    top_logprobs=categorical_top_logprobs,
                )
                raw_text = str(readout["raw_output"])
                query_usage = dict(readout["usage"])
                categorical_fields = {
                    "visible_answer": readout["visible_answer"],
                    "categorical_probs_json": json.dumps(
                        readout["semantic_probs"], sort_keys=True
                    ),
                    "categorical_entropy_bits": readout["entropy"],
                    "categorical_candidate_probability_mass": readout[
                        "candidate_probability_mass"
                    ],
                    "categorical_top_probability_mass": readout[
                        "top_logprob_probability_mass"
                    ],
                    "categorical_excluded_probability_mass_lower_bound": readout[
                        "excluded_probability_mass_lower_bound"
                    ],
                    "categorical_missing_candidates_json": json.dumps(
                        readout["missing_candidates"]
                    ),
                    "categorical_raw_top_logprobs_json": json.dumps(
                        readout["raw_top_logprobs"], sort_keys=True
                    ),
                    "categorical_all_candidates_present": not readout[
                        "missing_candidates"
                    ],
                    "categorical_argmax_answer": readout["answer"],
                }
            else:
                raw_text, query_usage = belief_query_fn(prompt, 0)
        except Exception as exc:
            query_error = f"{type(exc).__name__}: {exc}"
        if categorical_fields:
            answer_key = str(categorical_fields["categorical_argmax_answer"])
            answer_valid = answer_key in {"yes", "no", "unknown"}
        else:
            answer_key, answer_valid, _ = _answer_key(question, raw_text)
        truth = truths[question.target_variable]
        truth_key = _truth_key(question, truth)
        manhattan: int | str = ""
        if question.answer_space == "coord_json":
            manhattan = coordinate_manhattan_distance(parse_coordinate_answer(raw_text), truth) or 0
        row = {
            "example_id": example_id,
            "trajectory_id": position["trajectory_id"],
            "step_index": position["step_index"],
            "failure_category": position["failure_category"],
            "reasoning_step_idx": int(position["reasoning_step_idx"]),
            "reasoning_progress": float(position["reasoning_progress"]),
            "reasoning_character_progress": position.get("reasoning_character_progress", ""),
            "analysis_unit": analysis_unit,
            "analysis_unit_index": position.get(
                "analysis_unit_index", position["reasoning_step_idx"]
            ),
            "analysis_unit_char_start": position.get("analysis_unit_char_start", ""),
            "analysis_unit_char_end": position.get("analysis_unit_char_end", ""),
            "canonical_sentence_id": position.get("canonical_sentence_id", ""),
            "action_label": action,
            "action_is_optimal": position["action_is_optimal"] == "True" if position["action_is_optimal"] else None,
            "question_id": question.question_id,
            "question_family": question.family,
            "answer_space": question.answer_space,
            "answer_key": answer_key,
            "answer_valid": answer_valid,
            "ground_truth_key": truth_key,
            "belief_is_error": answer_key != truth_key,
            "coordinate_manhattan_distance": manhattan,
            "raw_belief_text": raw_text,
            "query_error": query_error,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "checkpoint_identity_sha256": checkpoint_identity_sha256,
            "categorical_logprob_temperature": (
                categorical_logprob_temperature
                if question.answer_space == "label3"
                and categorical_logprob_temperature is not None
                else ""
            ),
            "categorical_top_logprobs": (
                categorical_top_logprobs
                if question.answer_space == "label3"
                and categorical_logprob_temperature is not None
                else ""
            ),
            **categorical_fields,
        }
        return key, row, query_usage

    def record_result(
        handle: Any,
        key: tuple[str, int, str],
        row: dict[str, Any],
        query_usage: dict[str, Any],
    ) -> None:
        nonlocal attempted_this_invocation
        latest_rows[key] = row
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        attempted_this_invocation += 1
        if attempted_this_invocation % checkpoint_fsync_every == 0:
            os.fsync(handle.fileno())
        if not row.get("query_error"):
            completed.add(key)
        usage["requests"] += 1
        for usage_key in usage:
            if usage_key != "requests":
                usage[usage_key] += query_usage.get(usage_key, 0)
        write_status("running", current_key=key)

    pending_tasks: list[tuple[int, dict[str, str], BehavioralProbeQuestion]] = []
    for position_index, position in enumerate(prefix_rows, start=1):
        action = str(position["action_label"]).upper()
        for question in _selected_questions(action):
            key = (position["example_id"], int(position["reasoning_step_idx"]), question.question_id)
            if key not in completed:
                pending_tasks.append((position_index, position, question))
    try:
        with checkpoint.open("a") as handle:
            if max_workers == 1:
                for position_index, position, question in pending_tasks:
                    current_key = (position["example_id"], int(position["reasoning_step_idx"]), question.question_id)
                    if verbose:
                        print(
                            f"Belief position {position_index}/{len(prefix_rows)}: "
                            f"{position['example_id']} step={position['reasoning_step_idx']} "
                            f"question={question.question_id}"
                        )
                    key, row, query_usage = run_single_query(position, question)
                    record_result(handle, key, row, query_usage)
            else:
                if verbose:
                    print(f"Running {len(pending_tasks)} pending belief queries with max_workers={max_workers}")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(run_single_query, position, question): (
                            position["example_id"],
                            int(position["reasoning_step_idx"]),
                            question.question_id,
                        )
                        for _position_index, position, question in pending_tasks
                    }
                    for future in as_completed(futures):
                        current_key = futures[future]
                        key, row, query_usage = future.result()
                        record_result(handle, key, row, query_usage)
            os.fsync(handle.fileno())
    except BaseException:
        write_status("interrupted", current_key=current_key)
        raise

    belief_rows = list(latest_rows.values())
    position_rows: list[dict[str, Any]] = []
    beliefs_by_position: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in belief_rows:
        beliefs_by_position[(str(row["example_id"]), int(row["reasoning_step_idx"]))].append(row)
    for position in prefix_rows:
        rows = beliefs_by_position[(position["example_id"], int(position["reasoning_step_idx"]))]
        primary = [row for row in rows if row["question_id"] in PRIMARY_QUESTION_IDS]
        position_rows.append(
            {
                **position,
                "action_is_optimal": position["action_is_optimal"] == "True" if position["action_is_optimal"] else None,
                "n_beliefs": len(rows),
                "n_valid_beliefs": sum(bool(row["answer_valid"]) for row in rows),
                "n_primary_belief_errors": sum(bool(row["belief_is_error"]) for row in primary),
                "primary_belief_error_rate": sum(bool(row["belief_is_error"]) for row in primary) / len(primary) if primary else "",
            }
        )
    action_events = classify_action_events(position_rows)
    belief_shifts = classify_belief_shifts(belief_rows)
    lead_lag = _build_lead_lag(action_events, belief_shifts, event_window=event_window)
    associations = _feature_associations(position_rows, belief_rows)
    ranked = sorted(
        [
            row for row in lead_lag
            if row["event_type"] == "sustained_optimal_to_suboptimal"
            and row["shift_type"] == "belief_error_onset"
            and int(row["belief_shift_minus_action_event_steps"]) <= 0
        ],
        key=lambda row: (
            not bool(row["persistent_belief_error"]),
            -int(row["belief_shift_minus_action_event_steps"]),
            row["example_id"],
        ),
    )
    summary: list[dict[str, Any]] = []
    summary_counts = Counter(
        (str(row["event_type"]), str(row["failure_category"])) for row in action_events
    )
    for (event_type, failure_category), count in sorted(summary_counts.items()):
        relevant = [
            row for row in lead_lag
            if row["event_type"] == event_type and row["failure_category"] == failure_category
        ]
        summary.append(
            {
                "event_type": event_type,
                "failure_category": failure_category,
                "n_events": count,
                "n_nearby_belief_shifts": len(relevant),
                "n_belief_shifts_leading": sum(row["temporal_relation"] == "leads" for row in relevant),
                "n_belief_shifts_coinciding": sum(row["temporal_relation"] == "coincides" for row in relevant),
                "n_belief_shifts_lagging": sum(row["temporal_relation"] == "lags" for row in relevant),
                "n_persistent_belief_error_onsets": sum(
                    row["shift_type"] == "belief_error_onset" and bool(row["persistent_belief_error"])
                    for row in relevant
                ),
            }
        )
    _write_csv(out / "belief_rows.csv", belief_rows)
    _write_csv(out / "position_rows.csv", position_rows)
    _write_csv(out / "action_transition_rows.csv", action_events)
    _write_csv(out / "belief_shift_rows.csv", belief_shifts)
    _write_csv(out / "belief_action_lead_lag_rows.csv", lead_lag)
    _write_csv(out / "feature_association_fdr.csv", associations)
    _write_csv(out / "ranked_intervention_candidates.csv", ranked)
    _write_csv(out / "summary_by_event_type.csv", summary)
    audit_candidates = [
        {"audit_type": "action_transition", **row} for row in action_events[:50]
    ] + [{"audit_type": "belief_shift", **row} for row in belief_shifts[:50]]
    _write_csv(out / "manual_audit_candidates.csv", audit_candidates)
    span_validation = validate_activation_spans(drift / "step_activation_rows.csv")
    geometry_path = drift / "geometry_rows.csv"
    geometry_rows = _read_csv(geometry_path) if geometry_path.exists() and span_validation["valid"] else []
    _plot_outputs(
        out_dir=out,
        position_rows=position_rows,
        belief_rows=belief_rows,
        action_events=action_events,
        geometry_rows=geometry_rows,
    )
    parse_rate = sum(bool(row["answer_valid"]) for row in belief_rows) / len(belief_rows) if belief_rows else 0.0
    manifest = {
        "status": "completed",
        "drift_run_dir": drift_run_dir,
        "candidate_rows_path": candidate_rows_path,
        "model_name": model_name,
        "analysis_unit": analysis_unit,
        "categorical_logprob_temperature": categorical_logprob_temperature,
        "categorical_top_logprobs": (
            categorical_top_logprobs
            if categorical_logprob_temperature is not None
            else None
        ),
        "categorical_candidate_labels": (
            ["A", "B", "C"] if categorical_logprob_temperature is not None else []
        ),
        "max_positions": max_positions,
        "n_positions": len(position_rows),
        "n_belief_queries": len(belief_rows),
        "maximum_belief_queries": len(position_rows) * 13,
        "n_action_events": len(action_events),
        "n_belief_shifts": len(belief_shifts),
        "belief_valid_parse_rate": parse_rate,
        "minimum_required_parse_rate": 0.95,
        "parse_rate_gate_passed": parse_rate >= 0.95,
        "activation_span_validation": span_validation,
        "representational_probe_gate": (
            "blocked_until_activation_spans_are_regenerated"
            if not span_validation["valid"]
            else "activation_spans_valid_probe_compatibility_not_yet_established"
        ),
        "mixed_effects_model_status": (
            "not_fit_in_pilot; use feature_association_fdr.csv for trajectory-position descriptive screening "
            "and fit the prespecified mixed-effects model only after scaling"
        ),
        "usage_summary": usage,
    }
    _write_json(out / "manifest.json", manifest)
    report_lines = [
        "# Belief Changes and Action Changes During Reasoning",
        "",
        "## Run Status",
        "",
        f"- Reasoning positions: {len(position_rows)}",
        f"- Analysis unit: {analysis_unit}",
        f"- Behavioral belief queries: {len(belief_rows)}",
        f"- Valid parse rate: {parse_rate:.1%}",
        f"- Action events: {len(action_events)}",
        f"- Belief shifts: {len(belief_shifts)}",
        f"- Ranked candidates for later interventions: {len(ranked)}",
        "",
        "## Validation Gates",
        "",
        f"- Behavioral parse-rate gate: {'passed' if parse_rate >= 0.95 else 'failed'}",
        (
            "- Activation-span gate: passed"
            if span_validation["valid"]
            else f"- Activation-span gate: failed ({span_validation['reason']}; "
            f"{span_validation['n_overlaps']} overlaps)"
        ),
        "- Representational probe compatibility: not yet established",
        "",
        "## Action Events",
        "",
        "| Event type | Failure category | Events | Nearby belief shifts | Leads | Coincides | Lags |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    report_lines.extend(
        "| {event_type} | {failure_category} | {n_events} | {n_nearby_belief_shifts} | "
        "{n_belief_shifts_leading} | {n_belief_shifts_coinciding} | {n_belief_shifts_lagging} |".format(
            **row
        )
        for row in summary
    )
    report_lines.extend(
        [
            "",
            "Feature-level associations are descriptive screening results with "
            "Benjamini-Hochberg correction. The prespecified mixed-effects model is deferred "
            "until the scaled run.",
        ]
    )
    (out / "run_report.md").write_text("\n".join(report_lines) + "\n")
    write_status("completed")
    return manifest


def validate_activation_spans(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"valid": False, "reason": "step_activation_rows.csv is absent", "n_overlaps": 0}
    rows = _read_csv(path)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["example_id"], row["layer"])].append(row)
    overlaps = 0
    for group_rows in grouped.values():
        ordered = sorted(group_rows, key=lambda row: int(row["reasoning_step_idx"]))
        overlaps += sum(
            int(current["step_start_token"]) <= int(previous["step_end_token"])
            for previous, current in zip(ordered, ordered[1:])
        )
    return {
        "valid": overlaps == 0,
        "reason": "" if overlaps == 0 else "adjacent reasoning activation spans overlap",
        "n_rows": len(rows),
        "n_overlaps": overlaps,
    }


__all__ = [
    "PRIMARY_QUESTION_IDS",
    "classify_action_events",
    "classify_belief_shifts",
    "prepare_reasoning_belief_action_cohorts",
    "run_reasoning_belief_action_observational",
    "validate_activation_spans",
]
