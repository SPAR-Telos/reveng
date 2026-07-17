from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from reveng.experiments.activation_oracle_batch import DEFAULT_MODEL_NAME
from reveng.experiments.activation_oracle_prompts import (
    LOCAL_GRID_STATE_FAMILY,
    MODEL_ACTION_FAMILY,
    OPTIMAL_ACTION_FAMILY,
    model_action_oracle_prompt,
    optimal_action_oracle_prompt,
    wall_state_oracle_prompt,
)
from reveng.experiments.gradual_cot_blackbox_alignment import (
    REVEAL_PCTS,
    TOKENIZER_MODEL_ID,
    _action_prompt,
    _extract_analysis_and_final,
    _normalize_loaded_rows,
)

DEFAULT_PUBLIC_ROWS_PATH = "data/behavioral_probes/gradual_cot_alignment_public_slice/gradual_cot_alignment_rows.csv"
DEFAULT_FAILURE_ROWS_PATH = "data/behavioral_probes/gradual_cot_alignment_failure_wall_core_light/gradual_cot_alignment_rows.csv"
DEFAULT_FAILURE_TRAJECTORY_DIR = "data/hf/trajectories_key_door_100/trajectories_key_door"
DEFAULT_PUBLIC_TRAJECTORY_CACHE = "data/hf/cache/datasets--project-telos--trajectories_test_full/snapshots"
DEFAULT_OUTPUT_CLEAN_PATH = "data/activation_oracle/paper_clean_revealed_alignment_qwen_batch_v2.jsonl"
DEFAULT_OUTPUT_FAILURE_PATH = "data/activation_oracle/paper_failure_revealed_alignment_qwen_batch_v2.jsonl"
DEFAULT_CLEAN_REVEAL_PCTS = REVEAL_PCTS


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    return json.load(path.open())


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _load_public_trajectory_payload(trajectory_id: str, cache_root: Path) -> dict[str, Any]:
    matches = sorted(cache_root.rglob(f"{trajectory_id}.json"))
    if not matches:
        raise FileNotFoundError(f"Could not find cached public trajectory for {trajectory_id} under {cache_root}")
    return _load_json(matches[0])


def _load_failure_trajectory_payload(trajectory_id: str, trajectory_root: Path) -> dict[str, Any]:
    path = trajectory_root / f"{trajectory_id}.json"
    if not path.exists():
        matches = sorted(trajectory_root.rglob(f"{trajectory_id}.json"))
        if not matches:
            raise FileNotFoundError(f"Could not find local failure trajectory for {trajectory_id} under {trajectory_root}")
        path = matches[0]
    return _load_json(path)


def _reveal_analysis_text(
    *,
    analysis_text: str,
    analysis_tokenization_method: str,
    revealed_analysis_tokens: int,
    tokenizer: Any | None,
) -> str:
    if revealed_analysis_tokens <= 0:
        return ""
    if analysis_tokenization_method == "char_fallback":
        return analysis_text[:revealed_analysis_tokens]
    if analysis_tokenization_method == "hf_tokenizer_local":
        if tokenizer is None:
            raise RuntimeError("Tokenizer required for hf_tokenizer_local reveal reconstruction")
        token_ids = tokenizer.encode(analysis_text, add_special_tokens=False)
        return tokenizer.decode(token_ids[:revealed_analysis_tokens], skip_special_tokens=False)
    raise ValueError(f"Unsupported analysis_tokenization_method: {analysis_tokenization_method}")


def _format_action_prompt(
    *,
    grid_text: str,
    carrying_key: bool,
    revealed_analysis: str,
) -> str:
    return _action_prompt(
        grid_text=grid_text,
        carrying_key=carrying_key,
        revealed_analysis=revealed_analysis,
    )


def _wall_row_to_ao_row(
    *,
    row: dict[str, Any],
    target_prompt: str,
    representational_label: str = "",
) -> dict[str, Any]:
    return {
        "row_id": f"ao_{row['example_id']}_{row['question_id']}_r{row['reasoning_reveal_pct']}",
        "target_messages_json": [{"role": "user", "content": target_prompt}],
        "oracle_prompt": wall_state_oracle_prompt(str(row["question_id"])),
        "ground_truth": str(row.get("ground_truth_label", "")),
        "behavioral_label": str(row.get("blackbox_answer", "")),
        "representational_label": representational_label,
        "example_id": str(row.get("example_id", "")),
        "trajectory_id": str(row.get("trajectory_id", "")),
        "question_id": str(row.get("question_id", "")),
        "oracle_prompt_family": LOCAL_GRID_STATE_FAMILY,
        "source_dataset": str(row.get("source_dataset", "")),
        "observed_action": str(row.get("observed_action", "")),
        "blackbox_action": str(row.get("blackbox_action", "")),
        "optimal_actions_json": str(row.get("optimal_actions_json", "[]")),
        "blackbox_action_is_optimal": row.get("blackbox_action_is_optimal"),
        "is_optimal_action": row.get("is_optimal_action"),
        "whitebox_prediction_pre": str(row.get("whitebox_prediction_pre", "")),
        "whitebox_prediction_post": str(row.get("whitebox_prediction_post", "")),
        "reasoning_reveal_pct": int(row["reasoning_reveal_pct"]),
        "step_index": int(row["step_index"]),
        "oracle_input_types": ["full_seq"],
        "notes": "Generated from the exact gradual-CoT paper slice row and trajectory prompt.",
    }


def _action_row_to_ao_row(
    *,
    row: dict[str, Any],
    target_prompt: str,
) -> dict[str, Any]:
    blackbox_action = str(row.get("blackbox_action", "")).upper()
    return {
        "row_id": f"ao_{row['example_id']}_model_action_r{row['reasoning_reveal_pct']}",
        "target_messages_json": [{"role": "user", "content": target_prompt}],
        "oracle_prompt": model_action_oracle_prompt(),
        "ground_truth": blackbox_action,
        "behavioral_label": blackbox_action,
        "representational_label": "",
        "example_id": str(row.get("example_id", "")),
        "trajectory_id": str(row.get("trajectory_id", "")),
        "question_id": "model_action",
        "oracle_prompt_family": MODEL_ACTION_FAMILY,
        "source_dataset": str(row.get("source_dataset", "")),
        "observed_action": str(row.get("observed_action", "")),
        "blackbox_action": blackbox_action,
        "optimal_actions_json": str(row.get("optimal_actions_json", "[]")),
        "blackbox_action_is_optimal": row.get("blackbox_action_is_optimal"),
        "is_optimal_action": row.get("is_optimal_action"),
        "whitebox_prediction_pre": str(row.get("whitebox_prediction_pre", "")),
        "whitebox_prediction_post": str(row.get("whitebox_prediction_post", "")),
        "reasoning_reveal_pct": int(row["reasoning_reveal_pct"]),
        "step_index": int(row["step_index"]),
        "oracle_input_types": ["full_seq"],
        "notes": "Generated from the exact gradual-CoT paper slice row and trajectory prompt; compares AO forced action to the model's own revealed-CoT action.",
    }


def _optimal_action_row_to_ao_row(
    *,
    row: dict[str, Any],
    target_prompt: str,
) -> dict[str, Any]:
    return {
        "row_id": f"ao_{row['example_id']}_optimal_action_r{row['reasoning_reveal_pct']}",
        "target_messages_json": [{"role": "user", "content": target_prompt}],
        "oracle_prompt": optimal_action_oracle_prompt(),
        "ground_truth": "",
        "behavioral_label": "",
        "representational_label": "",
        "example_id": str(row.get("example_id", "")),
        "trajectory_id": str(row.get("trajectory_id", "")),
        "question_id": "optimal_action",
        "oracle_prompt_family": OPTIMAL_ACTION_FAMILY,
        "source_dataset": str(row.get("source_dataset", "")),
        "observed_action": str(row.get("observed_action", "")),
        "blackbox_action": str(row.get("blackbox_action", "")).upper(),
        "optimal_actions_json": str(row.get("optimal_actions_json", "[]")),
        "blackbox_action_is_optimal": row.get("blackbox_action_is_optimal"),
        "is_optimal_action": row.get("is_optimal_action"),
        "whitebox_prediction_pre": str(row.get("whitebox_prediction_pre", "")),
        "whitebox_prediction_post": str(row.get("whitebox_prediction_post", "")),
        "reasoning_reveal_pct": int(row["reasoning_reveal_pct"]),
        "step_index": int(row["step_index"]),
        "oracle_input_types": ["full_seq"],
        "notes": "Generated from the exact gradual-CoT paper slice row and trajectory prompt; compares AO optimal-action readout to the stored optimal action set.",
    }


def build_activation_oracle_paper_batches(
    *,
    public_rows_path: str = DEFAULT_PUBLIC_ROWS_PATH,
    failure_rows_path: str = DEFAULT_FAILURE_ROWS_PATH,
    public_trajectory_cache: str = DEFAULT_PUBLIC_TRAJECTORY_CACHE,
    failure_trajectory_dir: str = DEFAULT_FAILURE_TRAJECTORY_DIR,
    output_clean_path: str = DEFAULT_OUTPUT_CLEAN_PATH,
    output_failure_path: str = DEFAULT_OUTPUT_FAILURE_PATH,
    clean_reveal_pcts: tuple[int, ...] = DEFAULT_CLEAN_REVEAL_PCTS,
    failure_reveal_pcts: tuple[int, ...] = REVEAL_PCTS,
    include_clean_action_rows: bool = True,
    include_clean_optimal_action_rows: bool = True,
    include_failure_action_rows: bool = True,
    include_failure_optimal_action_rows: bool = True,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID, local_files_only=True)
    public_rows = _normalize_loaded_rows(_read_csv(public_rows_path))
    failure_rows = _normalize_loaded_rows(_read_csv(failure_rows_path))

    public_rows = [row for row in public_rows if int(row["reasoning_reveal_pct"]) in clean_reveal_pcts]
    failure_rows = [row for row in failure_rows if int(row["reasoning_reveal_pct"]) in failure_reveal_pcts]

    public_cache_root = Path(public_trajectory_cache)
    failure_root = Path(failure_trajectory_dir)
    public_payloads: dict[str, dict[str, Any]] = {}
    failure_payloads: dict[str, dict[str, Any]] = {}

    clean_batch_rows: list[dict[str, Any]] = []
    clean_action_keys_seen: set[tuple[str, int]] = set()
    clean_optimal_keys_seen: set[tuple[str, int]] = set()
    for row in public_rows:
        trajectory_id = str(row["trajectory_id"])
        payload = public_payloads.get(trajectory_id)
        if payload is None:
            payload = _load_public_trajectory_payload(trajectory_id, public_cache_root)
            public_payloads[trajectory_id] = payload
        step = payload["steps"][int(row["step_index"])]
        analysis_text, _final_text = _extract_analysis_and_final(step["output_text"])
        revealed_analysis = _reveal_analysis_text(
            analysis_text=analysis_text,
            analysis_tokenization_method=str(row["analysis_tokenization_method"]),
            revealed_analysis_tokens=int(row["revealed_analysis_tokens"]),
            tokenizer=tokenizer,
        )
        target_prompt = _format_action_prompt(
            grid_text=str(row["grid_text"]),
            carrying_key=_normalize_bool(row.get("carrying_key", False)),
            revealed_analysis=revealed_analysis,
        )
        pct = int(row["reasoning_reveal_pct"])
        representational_label = ""
        if pct == 0:
            representational_label = str(row.get("whitebox_prediction_pre", ""))
        elif pct == 100:
            representational_label = str(row.get("whitebox_prediction_post", ""))
        clean_batch_rows.append(
            _wall_row_to_ao_row(
                row=row,
                target_prompt=target_prompt,
                representational_label=representational_label,
            )
        )
        action_key = (str(row["example_id"]), int(row["reasoning_reveal_pct"]))
        if include_clean_action_rows and action_key not in clean_action_keys_seen:
            clean_action_keys_seen.add(action_key)
            clean_batch_rows.append(_action_row_to_ao_row(row=row, target_prompt=target_prompt))
        if include_clean_optimal_action_rows and action_key not in clean_optimal_keys_seen:
            clean_optimal_keys_seen.add(action_key)
            clean_batch_rows.append(_optimal_action_row_to_ao_row(row=row, target_prompt=target_prompt))

    failure_batch_rows: list[dict[str, Any]] = []
    action_keys_seen: set[tuple[str, int]] = set()
    optimal_keys_seen: set[tuple[str, int]] = set()
    for row in failure_rows:
        trajectory_id = str(row["trajectory_id"])
        payload = failure_payloads.get(trajectory_id)
        if payload is None:
            payload = _load_failure_trajectory_payload(trajectory_id, failure_root)
            failure_payloads[trajectory_id] = payload
        step = payload["steps"][int(row["step_index"])]
        analysis_text, _final_text = _extract_analysis_and_final(step["output_text"])
        revealed_analysis = _reveal_analysis_text(
            analysis_text=analysis_text,
            analysis_tokenization_method=str(row["analysis_tokenization_method"]),
            revealed_analysis_tokens=int(row["revealed_analysis_tokens"]),
            tokenizer=tokenizer,
        )
        target_prompt = _format_action_prompt(
            grid_text=str(row["grid_text"]),
            carrying_key=_normalize_bool(row.get("carrying_key", False)),
            revealed_analysis=revealed_analysis,
        )
        failure_batch_rows.append(
            _wall_row_to_ao_row(
                row=row,
                target_prompt=target_prompt,
                representational_label="",
            )
        )
        if include_failure_action_rows:
            action_key = (str(row["example_id"]), int(row["reasoning_reveal_pct"]))
            if action_key not in action_keys_seen:
                action_keys_seen.add(action_key)
                failure_batch_rows.append(_action_row_to_ao_row(row=row, target_prompt=target_prompt))
        if include_failure_optimal_action_rows:
            optimal_key = (str(row["example_id"]), int(row["reasoning_reveal_pct"]))
            if optimal_key not in optimal_keys_seen:
                optimal_keys_seen.add(optimal_key)
                failure_batch_rows.append(_optimal_action_row_to_ao_row(row=row, target_prompt=target_prompt))

    _write_jsonl(output_clean_path, clean_batch_rows)
    _write_jsonl(output_failure_path, failure_batch_rows)

    manifest = {
        "model_name": DEFAULT_MODEL_NAME,
        "public_rows_path": public_rows_path,
        "failure_rows_path": failure_rows_path,
        "output_clean_path": output_clean_path,
        "output_failure_path": output_failure_path,
        "clean_reveal_pcts": list(clean_reveal_pcts),
        "failure_reveal_pcts": list(failure_reveal_pcts),
        "include_clean_action_rows": include_clean_action_rows,
        "include_clean_optimal_action_rows": include_clean_optimal_action_rows,
        "include_failure_action_rows": include_failure_action_rows,
        "include_failure_optimal_action_rows": include_failure_optimal_action_rows,
        "n_clean_rows": len(clean_batch_rows),
        "n_clean_action_rows": len(clean_action_keys_seen),
        "n_clean_optimal_action_rows": len(clean_optimal_keys_seen),
        "n_failure_rows": len(failure_batch_rows),
        "n_failure_wall_rows": len(failure_rows),
        "n_failure_action_rows": len(action_keys_seen),
        "n_failure_optimal_action_rows": len(optimal_keys_seen),
    }
    print(json.dumps(manifest, indent=2))
    return manifest


__all__ = [
    "build_activation_oracle_paper_batches",
    "DEFAULT_OUTPUT_CLEAN_PATH",
    "DEFAULT_OUTPUT_FAILURE_PATH",
]
