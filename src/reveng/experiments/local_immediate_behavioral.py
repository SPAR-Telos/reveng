"""Local direct-answer action and belief readouts for reasoning prefixes."""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reveng.experiments.behavioral_probe_questions import get_behavioral_probe_questions
from reveng.experiments.behavioral_probe_runner import _behavioral_probe_preamble
from reveng.experiments.gpt_oss_activation_pilot import DEFAULT_MODEL_SNAPSHOT
from reveng.experiments.gradual_cot_blackbox_alignment import ANALYSIS_START, FINAL_START


ACTION_LABELS = ("UP", "DOWN", "LEFT", "RIGHT")
STATE_QUESTION_IDS = (
    "wall_left",
    "wall_right",
    "wall_up",
    "wall_down",
    "has_key",
    "door_open",
)
ACTION_EFFECT_PREFIXES = ("hit_wall_after_", "has_key_after_", "door_open_after_")
RAW_TO_SEMANTIC = {"A": "yes", "B": "no", "C": "unknown"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _json(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _entropy(probabilities: dict[str, float]) -> float:
    return -sum(value * math.log2(value) for value in probabilities.values() if value > 0)


def _longest_common_prefix(sequences: Sequence[Sequence[int]]) -> int:
    if not sequences:
        return 0
    limit = min(map(len, sequences))
    for index in range(limit):
        value = sequences[0][index]
        if any(sequence[index] != value for sequence in sequences[1:]):
            return index
    return limit


def _cacheable_prefix_length(sequences: Sequence[Sequence[int]]) -> int:
    """Retain at least one branch token, including when scoring one question."""
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("candidate sequences must be non-empty")
    return min(
        _longest_common_prefix(sequences),
        min(len(sequence) - 1 for sequence in sequences),
    )


def _analysis_text(output_text: str) -> str:
    if ANALYSIS_START not in output_text or FINAL_START not in output_text:
        raise ValueError("Trajectory output lacks analysis/final channel markers.")
    return output_text.split(ANALYSIS_START, 1)[1].split(FINAL_START, 1)[0]


def _trajectory_step(payload: dict[str, Any], step_index: int) -> dict[str, Any]:
    for index, step in enumerate(payload["steps"]):
        if int(step.get("step_id", index)) == step_index:
            return step
    raise ValueError(f"Missing environment step {step_index}.")


def _load_sentence_ends(path: Path) -> dict[str, list[tuple[int, int]]]:
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in _read_csv(path):
        if row.get("kind") not in {None, "", "reasoning"}:
            continue
        grouped[row["trace_id"]].append((int(row["sentence_id"]), int(row["char_end"])))
    return {key: sorted(value) for key, value in grouped.items()}


def _context_text(*, grid_text: str, carrying_key: bool, reasoning_prefix: str) -> str:
    return (
        _behavioral_probe_preamble("cardinal_action_explicit")
        + "\n# Inputs\n\nCurrent grid state:\n\n"
        + grid_text
        + "\n\nAgent status:\n- Carrying key: "
        + str(carrying_key).lower()
        + "\n\nReasoning trace available so far:\n"
        + reasoning_prefix.strip()
        + "\n\n"
    )


def _action_question() -> tuple[str, str, tuple[str, ...]]:
    return (
        "action",
        "# Question\nChoose the next move that best advances toward the goal while respecting "
        "DoorKey mechanics. Return only the requested JSON action.",
        ACTION_LABELS,
    )


def _belief_question(question_id: str) -> tuple[str, str, tuple[str, ...]]:
    questions = {q.question_id: q for q in get_behavioral_probe_questions(question_family="all")}
    question = questions[question_id]
    return (
        question_id,
        "# Question\n"
        + question.prompt_text.replace(
            " Answer with exactly one of: yes, no, unknown.", ""
        )
        + "\nReturn one label only: A = yes, B = no, C = unknown.",
        ("A", "B", "C"),
    )


class LocalImmediateReadout:
    def __init__(
        self,
        model_path: str | Path,
        *,
        temperature: float = 0.7,
        forward_chunk_size: int = 512,
    ) -> None:
        self.model_path = Path(model_path)
        self.temperature = float(temperature)
        self.forward_chunk_size = int(forward_chunk_size)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        ).eval()
        self.device = self.model.model.embed_tokens.weight.device

    def _render_ids(
        self,
        context: str,
        question: str,
        *,
        answer_prefix: str,
    ) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": context + question}],
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="low",
        )
        rendered += "<|channel|>final<|message|>" + answer_prefix
        return list(self.tokenizer(rendered, add_special_tokens=False)["input_ids"])

    def score_questions(
        self,
        *,
        context: str,
        questions: Sequence[tuple[str, str, tuple[str, ...]]],
        skip_unchosen_action_effects: bool = False,
    ) -> dict[str, dict[str, Any]]:
        sequences: list[list[int]] = []
        candidate_ids: list[list[int]] = []
        for question_id, question, labels in questions:
            answer_prefix = '{"action":"' if question_id == "action" else ""
            sequences.append(
                self._render_ids(context, question, answer_prefix=answer_prefix)
            )
            ids = [
                int(self.tokenizer(label, add_special_tokens=False)["input_ids"][0])
                for label in labels
            ]
            if any(len(self.tokenizer(label, add_special_tokens=False)["input_ids"]) != 1 for label in labels):
                raise ValueError(f"Candidate labels are not single tokens: {labels}")
            candidate_ids.append(ids)

        common_length = _cacheable_prefix_length(sequences)
        if common_length == 0:
            raise ValueError("Questions have no common token prefix.")
        common_ids = torch.tensor([sequences[0][:common_length]], device=self.device)
        cache = None
        with torch.inference_mode():
            for start in range(0, common_ids.shape[1], self.forward_chunk_size):
                output = self.model(
                    input_ids=common_ids[:, start : start + self.forward_chunk_size],
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                cache = output.past_key_values
                del output
        assert cache is not None
        results: dict[str, dict[str, Any]] = {}
        selected_action: str | None = None
        with torch.inference_mode():
            for (question_id, _question, labels), sequence, ids in zip(
                questions, sequences, candidate_ids
            ):
                if (
                    skip_unchosen_action_effects
                    and question_id.startswith(ACTION_EFFECT_PREFIXES)
                    and selected_action is not None
                    and not question_id.endswith(selected_action.lower())
                ):
                    continue
                suffix = torch.tensor([sequence[common_length:]], device=self.device, dtype=torch.long)
                # Cache layers append by assigning newly concatenated tensors. A
                # shallow structural fork therefore shares the immutable prefix
                # tensors without mutating the reusable base cache.
                branch_cache = copy.copy(cache)
                branch_cache.layers = [copy.copy(layer) for layer in cache.layers]
                output = self.model(
                    input_ids=suffix,
                    past_key_values=branch_cache,
                    use_cache=True,
                    return_dict=True,
                )
                logits = output.logits[0, -1, ids].float() / self.temperature
                probabilities_tensor = torch.softmax(logits, dim=0).cpu()
                probabilities = {
                    label: float(probabilities_tensor[index])
                    for index, label in enumerate(labels)
                }
                answer = max(probabilities, key=probabilities.get)
                results[question_id] = {
                    "answer": answer,
                    "probabilities": probabilities,
                    "entropy_bits": _entropy(probabilities),
                    "candidate_logprobs": {
                        label: float(torch.log_softmax(logits, dim=0)[index].cpu())
                        for index, label in enumerate(labels)
                    },
                    "common_tokens": common_length,
                    "suffix_tokens": len(sequence) - common_length,
                }
                if question_id == "action":
                    selected_action = answer
                del output
                del branch_cache
        del cache
        return results


def run_local_immediate_behavioral(
    *,
    candidate_rows_path: str | Path,
    output_dir: str | Path,
    analysis_mode: str,
    model_path: str | Path = DEFAULT_MODEL_SNAPSHOT,
    trajectory_dir: str | Path = "data/hf/trajectories_key_door_100/trajectories_key_door",
    sentences_path: str | Path = "data/behavioral_probes/doorkey_chunking_validation/sentences.csv",
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int = 42,
    limit_positions: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    if analysis_mode not in {"sentence", "environment_step"}:
        raise ValueError("analysis_mode must be sentence or environment_step")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "position_readouts.jsonl"
    candidates = _read_csv(Path(candidate_rows_path))
    sentence_ends = _load_sentence_ends(Path(sentences_path))
    config = {
        "protocol": "local_direct_final_answer_logprobs_v1",
        "model_path": str(Path(model_path).resolve()),
        "model_revision": Path(model_path).name,
        "candidate_rows_path": str(Path(candidate_rows_path).resolve()),
        "analysis_mode": analysis_mode,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "entropy_source": "candidate_token_logprobs; no repeated sampling",
        "coordinate_queries": "deferred stratified subset",
    }
    config_path = output / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing output configuration differs from this run.")
    _atomic_json(config_path, config)

    completed: dict[tuple[str, int], dict[str, Any]] = {}
    if resume and checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed[(row["example_id"], int(row["position_index"]))] = row

    planned: list[tuple[dict[str, str], int, int, str]] = []
    payload_cache: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        trajectory_id = candidate["trajectory_id"]
        payload = payload_cache.get(trajectory_id)
        if payload is None:
            payload = json.loads(
                (Path(trajectory_dir) / f"{trajectory_id}.json").read_text()
            )
            payload_cache[trajectory_id] = payload
        step_index = int(candidate["step_index"])
        analysis = _analysis_text(_trajectory_step(payload, step_index)["output_text"])
        trace_id = f"{trajectory_id}_step_{step_index:03d}"
        boundaries = sentence_ends[trace_id]
        if analysis_mode == "sentence":
            planned.append((candidate, 0, 0, ""))
            planned.extend(
                (candidate, sentence_id + 1, char_end, analysis[:char_end])
                for sentence_id, char_end in boundaries
            )
        else:
            sentence_id, char_end = boundaries[-1]
            planned.append((candidate, sentence_id + 1, char_end, analysis[:char_end]))
    if limit_positions is not None:
        planned = planned[: int(limit_positions)]

    pending = [item for item in planned if (item[0]["example_id"], item[1]) not in completed]
    manifest = {
        **config,
        "status": "running",
        "planned_positions": len(planned),
        "completed_positions": len(completed),
        "pending_positions": len(pending),
    }
    _atomic_json(output / "run_manifest.json", manifest)
    if pending:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        reader = LocalImmediateReadout(model_path, temperature=temperature)
        questions_by_id = {
            q.question_id: q for q in get_behavioral_probe_questions(question_family="all")
        }
        started = time.perf_counter()
        with checkpoint.open("a") as handle:
            for pending_index, (candidate, position_index, char_end, reasoning) in enumerate(
                pending, start=1
            ):
                context = _context_text(
                    grid_text=candidate["grid_text"],
                    carrying_key=_bool(candidate["carrying_key"]),
                    reasoning_prefix=reasoning,
                )
                all_effect_ids = [
                    f"{prefix}{direction}"
                    for direction in ("up", "down", "left", "right")
                    for prefix in ACTION_EFFECT_PREFIXES
                ]
                base_questions = [_action_question()] + [
                    _belief_question(question_id) for question_id in STATE_QUESTION_IDS
                ] + [_belief_question(question_id) for question_id in all_effect_ids]
                readouts = reader.score_questions(
                    context=context,
                    questions=base_questions,
                    skip_unchosen_action_effects=True,
                )
                action = str(readouts["action"]["answer"])
                effect_ids = [f"{prefix}{action.lower()}" for prefix in ACTION_EFFECT_PREFIXES]
                truths = _json(candidate["probe_truths_json"], {})
                beliefs = []
                for question_id in (*STATE_QUESTION_IDS, *effect_ids):
                    result = readouts[question_id]
                    semantic = RAW_TO_SEMANTIC[result["answer"]]
                    truth = str(truths[questions_by_id[question_id].target_variable])
                    beliefs.append(
                        {
                            "question_id": question_id,
                            "answer_key": semantic,
                            "ground_truth_key": truth,
                            "belief_is_error": semantic != truth,
                            "probabilities": {
                                RAW_TO_SEMANTIC[key]: value
                                for key, value in result["probabilities"].items()
                            },
                            "entropy_bits": result["entropy_bits"],
                        }
                    )
                optimal_actions = set(_json(candidate["optimal_actions_json"], []))
                row = {
                    "example_id": candidate["example_id"],
                    "trajectory_id": candidate["trajectory_id"],
                    "step_index": int(candidate["step_index"]),
                    "position_index": position_index,
                    "analysis_unit": analysis_mode,
                    "analysis_char_end": char_end,
                    "reasoning_progress": (
                        char_end / max(1, len(_analysis_text(_trajectory_step(payload_cache[candidate['trajectory_id']], int(candidate['step_index']))['output_text'])))
                    ),
                    "action_label": action,
                    "action_is_optimal": action in optimal_actions,
                    "action_probabilities": readouts["action"]["probabilities"],
                    "action_entropy_bits": readouts["action"]["entropy_bits"],
                    "beliefs": beliefs,
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                completed[(row["example_id"], position_index)] = row
                elapsed = time.perf_counter() - started
                manifest.update(
                    {
                        "completed_positions": len(completed),
                        "pending_positions": len(planned) - len(completed),
                        "elapsed_seconds_this_invocation": elapsed,
                        "seconds_per_position": elapsed / pending_index,
                    }
                )
                _atomic_json(output / "run_manifest.json", manifest)
                if pending_index == 1 or pending_index % 25 == 0:
                    remaining_seconds = (
                        manifest["seconds_per_position"]
                        * (len(pending) - pending_index)
                    )
                    print(
                        f"[local-readout] {pending_index}/{len(pending)} new positions; "
                        f"{manifest['seconds_per_position']:.2f} s/position; "
                        f"estimated {remaining_seconds / 3600:.2f} h remaining",
                        flush=True,
                    )

    rows = sorted(completed.values(), key=lambda row: (row["example_id"], row["position_index"]))
    action_rows = []
    belief_rows = []
    for row in rows:
        action_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"beliefs", "action_probabilities"}
            }
            | {"action_probabilities_json": json.dumps(row["action_probabilities"], sort_keys=True)}
        )
        for belief in row["beliefs"]:
            belief_rows.append(
                {
                    "example_id": row["example_id"],
                    "trajectory_id": row["trajectory_id"],
                    "step_index": row["step_index"],
                    "position_index": row["position_index"],
                    "analysis_unit": row["analysis_unit"],
                    "reasoning_progress": row["reasoning_progress"],
                    "action_label": row["action_label"],
                    **{key: value for key, value in belief.items() if key != "probabilities"},
                    "probabilities_json": json.dumps(belief["probabilities"], sort_keys=True),
                }
            )
    _write_csv(output / "prefix_action_rows.csv", action_rows)
    _write_csv(output / "belief_rows.csv", belief_rows)
    manifest.update(
        {
            "status": "completed" if len(rows) == len(planned) else "partial",
            "completed_positions": len(rows),
            "pending_positions": len(planned) - len(rows),
            "belief_rows": len(belief_rows),
        }
    )
    _atomic_json(output / "run_manifest.json", manifest)
    return manifest


__all__ = ["LocalImmediateReadout", "run_local_immediate_behavioral"]
