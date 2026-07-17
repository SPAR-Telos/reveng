"""Minimal wall-feature patch experiment scaffold.

This experiment bridges white-box wall-feature interventions with black-box
behavioral and action readouts on the same selected failure states.

It deliberately stops at the intervention boundary when no patched outputs are
available. In that case it still writes:
- selected states
- donor/intervention templates
- unpatched wall/action readouts (optional)
- joined rows with pending patch slots
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from reveng.experiments.behavioral_probe_questions import BEHAVIORAL_PROBE_QUESTION_BY_ID
from reveng.experiments.behavioral_probe_runner import (
    BehavioralProbeLLM,
    _belief_prompt,
    _empty_usage_bucket,
    _observed_action_prompt_with_state_text,
    _query_action,
    _query_text,
    _write_csv,
)
from reveng.experiments.behavioral_probe_parse import parse_behavioral_probe_answer
from reveng.experiments.behavioral_probe_smoke_data import derive_probe_truths_from_state
from reveng.experiments.behavioral_probe_trajectory_data import DoorKeyStateSolver

PATCH_TYPES = ("none", "pre_to_pre", "post_to_post", "pre_to_post")
PATCH_STAGE_MAP = {
    "none": ("", ""),
    "pre_to_pre": ("pre", "pre"),
    "post_to_post": ("post", "post"),
    "pre_to_post": ("pre", "post"),
}
FAILURE_MODE_PRIORITY = ("wall_hit", "avoidable_detour", "backtrack")
VALID_ACTIONS = {"LEFT", "RIGHT", "UP", "DOWN"}


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return list(payload["rows"])
    raise ValueError(f"Unsupported file format: {path}")


def _parse_bool(value: Any) -> bool | None:
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


def _parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    stripped = str(value).strip()
    if not stripped:
        return []
    parsed = json.loads(stripped)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def _normalize_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["example_id"] = str(row.get("example_id", ""))
    normalized["trajectory_id"] = str(row.get("trajectory_id", ""))
    normalized["step_index"] = str(row.get("step_index", ""))
    normalized["grid_text"] = str(row.get("grid_text", ""))
    normalized["carrying_key"] = bool(_parse_bool(row.get("carrying_key")) or False)
    normalized["observed_action"] = str(row.get("observed_action", "")).upper()
    normalized["is_optimal_action"] = _parse_bool(row.get("is_optimal_action"))
    normalized["wall_hit"] = _parse_bool(row.get("wall_hit"))
    normalized["primary_step_failure_mode"] = str(row.get("primary_step_failure_mode", "none") or "none")
    normalized["selection_reason"] = str(row.get("selection_reason", ""))
    normalized["selected_for_probe"] = _parse_bool(row.get("selected_for_probe"))
    normalized["optimal_actions"] = _parse_json_list(row.get("optimal_actions_json", row.get("optimal_actions", [])))
    probe_truths_raw = row.get("probe_truths_json")
    if probe_truths_raw:
        probe_truths = json.loads(probe_truths_raw) if isinstance(probe_truths_raw, str) else probe_truths_raw
    else:
        probe_truths = derive_probe_truths_from_state(normalized["grid_text"], normalized["carrying_key"])
    normalized["probe_truths"] = probe_truths
    normalized["target_wall_question"] = _action_to_wall_question_id(normalized["observed_action"])
    target_question = normalized["target_wall_question"]
    normalized["ground_truth_label"] = probe_truths.get(target_question, "") if target_question else ""
    return normalized


def _action_to_wall_question_id(action: str) -> str | None:
    action = str(action).upper().strip()
    if action not in VALID_ACTIONS:
        return None
    return f"wall_{action.lower()}"


def load_wall_feature_patch_candidates(path: str) -> list[dict[str, Any]]:
    return [_normalize_candidate_row(row) for row in _load_rows(path)]


def _is_action_relevant(row: dict[str, Any]) -> bool:
    return (
        row.get("observed_action") in VALID_ACTIONS
        and row.get("target_wall_question") in BEHAVIORAL_PROBE_QUESTION_BY_ID
        and row.get("ground_truth_label") in {"yes", "no", "unknown"}
    )


def select_wall_feature_patch_states(
    rows: list[dict[str, Any]],
    *,
    max_wall_hit: int = 4,
    max_avoidable_detour: int = 4,
    max_backtrack: int = 2,
    max_baseline: int = 4,
) -> list[dict[str, Any]]:
    quotas = {
        "wall_hit": max_wall_hit,
        "avoidable_detour": max_avoidable_detour,
        "backtrack": max_backtrack,
    }
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("trajectory_id", ""),
            int(row.get("step_index") or 0),
            row.get("example_id", ""),
        )

    for mode in FAILURE_MODE_PRIORITY:
        limit = quotas[mode]
        if limit <= 0:
            continue
        candidates = [
            row for row in rows
            if _is_action_relevant(row)
            and row.get("primary_step_failure_mode") == mode
            and row.get("example_id") not in seen_ids
        ]
        for row in sorted(candidates, key=_rank_key)[:limit]:
            selected.append(row)
            seen_ids.add(row["example_id"])

    baseline_candidates = [
        row for row in rows
        if _is_action_relevant(row)
        and row.get("primary_step_failure_mode") in {"", "none"}
        and row.get("is_optimal_action") is True
        and row.get("wall_hit") is False
        and row.get("example_id") not in seen_ids
    ]
    for row in sorted(baseline_candidates, key=_rank_key)[:max_baseline]:
        baseline_row = dict(row)
        baseline_row["primary_step_failure_mode"] = "baseline_optimal"
        selected.append(baseline_row)
        seen_ids.add(row["example_id"])

    return selected


def _donor_sort_key(recipient: dict[str, Any], donor: dict[str, Any]) -> tuple[Any, ...]:
    same_carry = donor.get("carrying_key") == recipient.get("carrying_key")
    same_action = donor.get("observed_action") == recipient.get("observed_action")
    same_failure = donor.get("primary_step_failure_mode") == recipient.get("primary_step_failure_mode")
    return (
        0 if same_carry else 1,
        0 if same_action else 1,
        0 if same_failure else 1,
        donor.get("trajectory_id", ""),
        int(donor.get("step_index") or 0),
        donor.get("example_id", ""),
    )


def choose_wall_feature_donor(recipient: dict[str, Any], pool_rows: list[dict[str, Any]]) -> dict[str, Any]:
    question_id = recipient["target_wall_question"]
    label = recipient["ground_truth_label"]
    opposite = "no" if label == "yes" else "yes" if label == "no" else None
    if opposite is None:
        raise ValueError(f"Recipient {recipient['example_id']} has non-binary wall label: {label}")
    candidates = [
        row for row in pool_rows
        if _is_action_relevant(row)
        and row.get("example_id") != recipient.get("example_id")
        and row.get("target_wall_question") == question_id
        and row.get("ground_truth_label") == opposite
    ]
    if not candidates:
        raise ValueError(
            f"No opposite-label donor found for {recipient['example_id']} / {question_id} / {label}"
        )
    return sorted(candidates, key=lambda donor: _donor_sort_key(recipient, donor))[0]


def filter_rows_with_wall_feature_donors(
    selected_rows: list[dict[str, Any]],
    pool_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in selected_rows:
        try:
            choose_wall_feature_donor(row, pool_rows)
        except ValueError:
            dropped.append(row)
            continue
        kept.append(row)
    return kept, dropped


def build_wall_feature_patch_template(selected_rows: list[dict[str, Any]], pool_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    template_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        donor = choose_wall_feature_donor(row, pool_rows)
        for patch_type in PATCH_TYPES:
            source_stage, target_stage = PATCH_STAGE_MAP[patch_type]
            template_rows.append(
                {
                    "example_id": row["example_id"],
                    "trajectory_id": row["trajectory_id"],
                    "step_index": row["step_index"],
                    "failure_mode": row["primary_step_failure_mode"],
                    "target_wall_question": row["target_wall_question"],
                    "ground_truth_label": row["ground_truth_label"],
                    "patch_type": patch_type,
                    "reasoning_stage_source": source_stage,
                    "reasoning_stage_target": target_stage,
                    "donor_example_id": "" if patch_type == "none" else donor["example_id"],
                    "donor_trajectory_id": "" if patch_type == "none" else donor["trajectory_id"],
                    "donor_step_index": "" if patch_type == "none" else donor["step_index"],
                    "donor_label": row["ground_truth_label"] if patch_type == "none" else donor["ground_truth_label"],
                    "patched_wall_answer": row["ground_truth_label"] if patch_type == "none" else "",
                    "patched_action": row.get("observed_action", "") if patch_type == "none" else "",
                    "patch_backend": "none" if patch_type == "none" else "",
                    "patch_artifact_path": "",
                    "notes": "fill patched_* columns after running wall-feature intervention" if patch_type != "none" else "unpatched baseline row",
                }
            )
    return template_rows


def _build_usage_summary(model_name: str) -> dict[str, Any]:
    return {"model_name": model_name, "by_phase": {}, "total": _empty_usage_bucket()}


def _add_usage(usage_summary: dict[str, Any], phase: str, usage: dict[str, Any]) -> None:
    bucket = usage_summary["by_phase"].setdefault(phase, _empty_usage_bucket())
    bucket["requests"] += 1
    usage_summary["total"]["requests"] += 1
    for key in ("cost_usd", "prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens", "retry_count", "requests_with_retry", "retry_sleep_seconds"):
        value = usage.get(key, 0)
        bucket[key] += value
        usage_summary["total"][key] += value


def collect_unpatched_readouts(
    selected_rows: list[dict[str, Any]],
    *,
    model_name: str,
    prompt_preset: str = "cardinal_action_explicit",
    belief_client: Any | None = None,
    action_client: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    belief_client = belief_client or BehavioralProbeLLM(model_name, temperature=0.0)
    action_client = action_client or BehavioralProbeLLM(model_name, temperature=0.0)
    usage_summary = _build_usage_summary(model_name)
    readouts: list[dict[str, Any]] = []
    for row in selected_rows:
        question = BEHAVIORAL_PROBE_QUESTION_BY_ID[row["target_wall_question"]]
        belief_prompt = _belief_prompt(
            grid_text=row["grid_text"],
            carrying_key=row["carrying_key"],
            question=question,
            prompt_preset=prompt_preset,
            state_description_text=None,
        )
        wall_raw, wall_usage = _query_text(belief_client, belief_prompt, seed=0)
        _add_usage(usage_summary, "unpatched_wall", wall_usage)
        wall_answer = parse_behavioral_probe_answer(wall_raw)

        action_prompt = _observed_action_prompt_with_state_text(
            grid_text=row["grid_text"],
            carrying_key=row["carrying_key"],
            prompt_preset=prompt_preset,
            state_description_text=None,
        )
        action_label, action_raw, action_usage = _query_action(action_client, action_prompt, seed=0)
        _add_usage(usage_summary, "unpatched_action", action_usage)

        readouts.append(
            {
                "example_id": row["example_id"],
                "unpatched_wall_answer": wall_answer,
                "unpatched_wall_raw": wall_raw,
                "unpatched_action": action_label,
                "unpatched_action_raw": action_raw,
            }
        )
    return readouts, usage_summary


def _load_patch_results(path: str | Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    rows = _load_rows(path)
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        example_id = str(row.get("example_id", ""))
        patch_type = str(row.get("patch_type", ""))
        if not example_id or not patch_type:
            continue
        index[(example_id, patch_type)] = dict(row)
    return index


def _patched_action_hits_wall(row: dict[str, Any], patched_action: str) -> bool | None:
    action = str(patched_action).upper().strip()
    if action not in VALID_ACTIONS:
        return None
    solver = DoorKeyStateSolver()
    state = (row["grid_text"], row["carrying_key"])
    return bool(solver.step(state, action)["hit_wall"])


def build_wall_feature_patch_rows(
    *,
    selected_rows: list[dict[str, Any]],
    template_rows: list[dict[str, Any]],
    unpatched_readouts: list[dict[str, Any]] | None,
    patch_results_index: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_index = {row["example_id"]: row for row in selected_rows}
    unpatched_index = {row["example_id"]: row for row in (unpatched_readouts or [])}
    output_rows: list[dict[str, Any]] = []
    for template in template_rows:
        selected = selected_index[template["example_id"]]
        unpatched = unpatched_index.get(template["example_id"], {})
        patch_result = patch_results_index.get((template["example_id"], template["patch_type"]), {})
        if template["patch_type"] == "none":
            patched_wall_answer = unpatched.get(
                "unpatched_wall_answer",
                template.get("patched_wall_answer", selected.get("ground_truth_label", "")),
            )
            patched_action = unpatched.get(
                "unpatched_action",
                template.get("patched_action", selected.get("observed_action", "")),
            )
        else:
            patched_wall_answer = patch_result.get("patched_wall_answer", template.get("patched_wall_answer", ""))
            patched_action = patch_result.get("patched_action", template.get("patched_action", ""))
        patched_action = str(patched_action).upper().strip() if patched_action else ""
        patched_wall_answer = parse_behavioral_probe_answer(patched_wall_answer) if patched_wall_answer else ""
        patched_action_is_optimal = None
        if patched_action:
            patched_action_is_optimal = patched_action in selected.get("optimal_actions", [])
        patched_action_hits_wall = _patched_action_hits_wall(selected, patched_action) if patched_action else None
        unpatched_wall = str(unpatched.get("unpatched_wall_answer", ""))
        unpatched_action = str(unpatched.get("unpatched_action", "")).upper().strip()
        row = {
            "example_id": selected["example_id"],
            "trajectory_id": selected["trajectory_id"],
            "step_index": selected["step_index"],
            "failure_mode": selected["primary_step_failure_mode"],
            "target_wall_question": selected["target_wall_question"],
            "reasoning_stage_source": template["reasoning_stage_source"],
            "reasoning_stage_target": template["reasoning_stage_target"],
            "patch_type": template["patch_type"],
            "donor_example_id": patch_result.get("donor_example_id", template["donor_example_id"]),
            "ground_truth_label": selected["ground_truth_label"],
            "unpatched_wall_answer": unpatched_wall,
            "patched_wall_answer": patched_wall_answer,
            "unpatched_action": unpatched_action,
            "patched_action": patched_action,
            "observed_action": selected["observed_action"],
            "is_optimal_action": selected["is_optimal_action"],
            "wall_hit": selected["wall_hit"],
            "wall_answer_flipped": (patched_wall_answer != unpatched_wall) if patched_wall_answer and unpatched_wall else None,
            "action_flipped": (patched_action != unpatched_action) if patched_action and unpatched_action else None,
            "patched_action_is_optimal": patched_action_is_optimal,
            "patched_action_hits_wall": patched_action_hits_wall,
            "patch_backend": patch_result.get("patch_backend", template.get("patch_backend", "")),
            "patch_artifact_path": patch_result.get("patch_artifact_path", template.get("patch_artifact_path", "")),
            "status": "ready" if template["patch_type"] == "none" or patch_result else "pending_patch_results",
            "unpatched_wall_raw": unpatched.get("unpatched_wall_raw", ""),
            "unpatched_action_raw": unpatched.get("unpatched_action_raw", ""),
            "grid_text": selected["grid_text"],
            "carrying_key": selected["carrying_key"],
        }
        output_rows.append(row)
    return output_rows


def summarize_wall_feature_patch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for patch_type in PATCH_TYPES:
        patch_rows = [row for row in rows if row["patch_type"] == patch_type]
        if not patch_rows:
            continue
        ready_rows = [row for row in patch_rows if row.get("status") == "ready"]
        wall_flip_values = [row["wall_answer_flipped"] for row in ready_rows if isinstance(row.get("wall_answer_flipped"), bool)]
        action_flip_values = [row["action_flipped"] for row in ready_rows if isinstance(row.get("action_flipped"), bool)]
        toward_truth_values = [
            (row.get("patched_wall_answer") == row.get("ground_truth_label") and row.get("patched_wall_answer") != row.get("unpatched_wall_answer"))
            for row in ready_rows
            if row.get("patched_wall_answer")
        ]
        base_opt = [row["is_optimal_action"] for row in ready_rows if isinstance(row.get("is_optimal_action"), bool)]
        patch_opt = [row["patched_action_is_optimal"] for row in ready_rows if isinstance(row.get("patched_action_is_optimal"), bool)]
        base_wall = [row["wall_hit"] for row in ready_rows if isinstance(row.get("wall_hit"), bool)]
        patch_wall = [row["patched_action_hits_wall"] for row in ready_rows if isinstance(row.get("patched_action_hits_wall"), bool)]

        def _mean(values: list[bool]) -> float | None:
            if not values:
                return None
            return sum(1.0 if value else 0.0 for value in values) / len(values)

        patched_opt_change = None
        if base_opt and patch_opt and len(base_opt) == len(patch_opt):
            patched_opt_change = _mean(patch_opt) - _mean(base_opt)
        patched_wall_change = None
        if base_wall and patch_wall and len(base_wall) == len(patch_wall):
            patched_wall_change = _mean(patch_wall) - _mean(base_wall)

        summary_rows.append(
            {
                "patch_type": patch_type,
                "n_states": len(patch_rows),
                "n_ready_states": len(ready_rows),
                "wall_answer_flip_rate": _mean(wall_flip_values),
                "action_flip_rate": _mean(action_flip_values),
                "flip_toward_ground_truth_rate": _mean(toward_truth_values),
                "patched_optimality_change": patched_opt_change,
                "patched_wall_hit_change": patched_wall_change,
                "status": "ready" if len(ready_rows) == len(patch_rows) else "pending_patch_results",
            }
        )
    return summary_rows


def write_wall_feature_patch_examples(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    ready_changed = [
        row for row in rows
        if row.get("status") == "ready"
        and (row.get("wall_answer_flipped") is True or row.get("action_flipped") is True)
        and row.get("patch_type") != "none"
    ]
    lines = ["# Wall Feature Patch Examples", ""]
    if not ready_changed:
        lines.append("No ready changed examples available yet.")
    else:
        for row in ready_changed[:8]:
            lines.extend(
                [
                    f"## {row['example_id']} / {row['patch_type']}",
                    "",
                    f"- failure mode: `{row['failure_mode']}`",
                    f"- target wall question: `{row['target_wall_question']}`",
                    f"- donor: `{row['donor_example_id']}`",
                    f"- ground truth: `{row['ground_truth_label']}`",
                    f"- unpatched wall answer: `{row['unpatched_wall_answer']}`",
                    f"- patched wall answer: `{row['patched_wall_answer']}`",
                    f"- unpatched action: `{row['unpatched_action']}`",
                    f"- patched action: `{row['patched_action']}`",
                    "",
                    "```text",
                    row['grid_text'].rstrip(),
                    "```",
                    "",
                ]
            )
    path.write_text("\n".join(lines) + "\n")


def run_behavioral_probe_wall_feature_patch(
    candidate_rows_path: str = "data/behavioral_probes/trajectory_instances_recomputed_optimal/trajectory_selection_candidates.csv",
    model_name: str = "together_ai/openai/gpt-oss-20b",
    output_dir: str = "data/behavioral_probes/wall_feature_patch_core",
    *,
    patch_results_path: str | None = None,
    max_wall_hit: int = 4,
    max_avoidable_detour: int = 4,
    max_backtrack: int = 2,
    max_baseline: int = 4,
    prompt_preset: str = "cardinal_action_explicit",
    run_unpatched_queries: bool = True,
    verbose: bool = True,
) -> None:
    rows = load_wall_feature_patch_candidates(candidate_rows_path)
    selected_rows = select_wall_feature_patch_states(
        rows,
        max_wall_hit=max_wall_hit,
        max_avoidable_detour=max_avoidable_detour,
        max_backtrack=max_backtrack,
        max_baseline=max_baseline,
    )
    selected_rows, dropped_rows = filter_rows_with_wall_feature_donors(selected_rows, rows)
    if len(selected_rows) < 4:
        raise ValueError("Need at least 4 selected states for the wall-feature patch experiment scaffold.")
    template_rows = build_wall_feature_patch_template(selected_rows, rows)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "selected_states.csv", selected_rows)
    _write_csv(out_dir / "wall_feature_patch_interventions_template.csv", template_rows)

    unpatched_readouts: list[dict[str, Any]] | None = None
    usage_summary = _build_usage_summary(model_name)
    if run_unpatched_queries:
        unpatched_readouts, usage_summary = collect_unpatched_readouts(
            selected_rows,
            model_name=model_name,
            prompt_preset=prompt_preset,
        )
        _write_csv(out_dir / "unpatched_readouts.csv", unpatched_readouts)
        (out_dir / "usage_summary.json").write_text(json.dumps(usage_summary, indent=2))

    patch_results_index = _load_patch_results(patch_results_path)
    joined_rows = build_wall_feature_patch_rows(
        selected_rows=selected_rows,
        template_rows=template_rows,
        unpatched_readouts=unpatched_readouts,
        patch_results_index=patch_results_index,
    )
    summary_rows = summarize_wall_feature_patch_rows(joined_rows)
    _write_csv(out_dir / "wall_feature_patch_rows.csv", joined_rows)
    _write_csv(out_dir / "wall_feature_patch_summary.csv", summary_rows)
    write_wall_feature_patch_examples(out_dir / "wall_feature_patch_examples.md", joined_rows)

    status = {
        "status": "ready" if patch_results_index else "pending_patch_results",
        "n_candidate_rows": len(rows),
        "n_selected_states": len(selected_rows),
        "n_dropped_no_donor_states": len(dropped_rows),
        "n_template_rows": len(template_rows),
        "n_patch_result_rows": len(patch_results_index),
        "patch_results_path": patch_results_path,
        "run_unpatched_queries": run_unpatched_queries,
        "dropped_no_donor_example_ids": [row["example_id"] for row in dropped_rows],
    }
    (out_dir / "wall_feature_patch_status.json").write_text(json.dumps(status, indent=2))
    if verbose:
        print(json.dumps(status, indent=2))


__all__ = [
    "PATCH_TYPES",
    "build_wall_feature_patch_rows",
    "build_wall_feature_patch_template",
    "choose_wall_feature_donor",
    "collect_unpatched_readouts",
    "filter_rows_with_wall_feature_donors",
    "load_wall_feature_patch_candidates",
    "run_behavioral_probe_wall_feature_patch",
    "select_wall_feature_patch_states",
    "summarize_wall_feature_patch_rows",
]
