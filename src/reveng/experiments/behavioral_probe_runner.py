"""Black-box behavioral-probe experiments for DoorKey states."""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from reveng.agents.llm_templates import ActionResponse
from reveng.experiments.behavioral_probe_metrics import (
    belief_action_consistency,
    sampled_answer_from_probabilities,
    shannon_entropy,
    summarize_coordinate_samples,
    summarize_probe_rows,
    valid_parse_rate,
    valid_parse_rate_from_answers,
    yes_no_answer_from_probabilities,
)
from reveng.experiments.behavioral_probe_parse import (
    BEHAVIORAL_PROBE_LABEL_MAP,
    VALID_BEHAVIORAL_PROBE_ANSWERS,
    coordinate_manhattan_distance,
    parse_behavioral_probe_answer,
    parse_coordinate_answer,
)
from reveng.experiments.behavioral_probe_plots import (
    plot_behavioral_probe_summary,
    plot_coordinate_probe_summary,
    plot_directional_probe_heatmap,
)
from reveng.experiments.behavioral_probe_questions import (
    BehavioralProbeQuestion,
    DOOR_OPEN_AFTER_ACTION_VARIANTS,
    get_behavioral_probe_questions,
)
from reveng.experiments.behavioral_probe_smoke_data import (
    SMOKE_TEST_STATES,
    derive_probe_truths,
    validate_smoke_examples,
)
from reveng.llm_interface import BaseLLMInterface

logger = logging.getLogger(__name__)

RAW_LABEL_TOKENS = ("A", "B", "C")
DIRECT_SEMANTIC_TOKENS = ("yes", "no", "unknown")
SUPPORTED_LOGPROB_TEMPS = (0.0, 0.7, 1.0)
PROMPT_PRESETS: dict[str, str] = {
    "default_observable_state": "Answer using only the current observable state shown below.",
    "cardinal_action_explicit": (
        "Answer using only the current observable state shown below. "
        "Interpret RIGHT/LEFT/UP/DOWN as cardinal directions in the rendered grid."
    ),
    "coordinate_output_explicit": (
        "Answer using only the current observable state shown below. "
        "For coordinate questions, coordinates are grid indices with row first, then col."
    ),
}


def _temp_suffix(temperature: float) -> str:
    if math.isclose(temperature, 0.0):
        return "t0"
    if math.isclose(temperature, 0.7):
        return "t07"
    if math.isclose(temperature, 1.0):
        return "t1"
    return f"t{str(temperature).replace('.', '')}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _behavioral_probe_preamble(prompt_preset: str) -> str:
    extra = PROMPT_PRESETS.get(prompt_preset)
    if extra is None:
        raise ValueError(f"Unknown prompt preset: {prompt_preset}")
    return (
        "# Instructions\n\n"
        "You are controlling an agent in a 9x9 fully observable DoorKey GridWorld. "
        "The environment contains walls, open spaces, a goal, a key, and doors.\n\n"
        "Legend:\n"
        "---------------\n"
        "#: Wall\n"
        "_: Open Space\n"
        "G: Goal\n"
        "A: Current agent position\n"
        "D: Door (closed/locked)\n"
        "O: Door (open)\n"
        "K: Key\n"
        "---------------\n\n"
        "Interpret RIGHT as the cell one column to the right (east), LEFT as one column to the left "
        "(west), UP as one row up (north), and DOWN as one row down (south) in the rendered grid.\n"
        f"{extra}\n"
    )


def _observed_action_prompt(grid_text: str, carrying_key: bool, prompt_preset: str) -> str:
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
        + f"Agent status:\n- Carrying key: {str(carrying_key).lower()}\n\n"
        + 'Respond with exactly a JSON object of the form {"action": "<UP|DOWN|LEFT|RIGHT>"}.\n'
        + "Do not include any extra text before or after the JSON."
    )


def _belief_prompt(
    *,
    grid_text: str,
    carrying_key: bool,
    question: BehavioralProbeQuestion,
    prompt_preset: str,
) -> str:
    prompt = (
        _behavioral_probe_preamble(prompt_preset)
        + "\n"
        + "# Inputs\n\n"
        + "Current grid state:\n\n"
        + grid_text
        + "\n\n"
    )
    if question.show_carrying_key:
        prompt += f"Agent status:\n- Carrying key: {str(carrying_key).lower()}\n\n"
    prompt += question.prompt_text + "\n\n"
    if question.answer_space == "coord_json":
        prompt += (
            'Return exactly one JSON object of the form {"row": <int>, "col": <int>} and nothing else.\n'
        )
    else:
        prompt += (
            "Return exactly one label and nothing else:\n"
            "A = yes\n"
            "B = no\n"
            "C = unknown\n"
        )
    return prompt


def _clean_json_response(response: str) -> str:
    cleaned = re.sub(r",(\s*[}\]])", r"\1", response)
    last_brace = cleaned.rfind("}")
    if last_brace == -1:
        return cleaned
    brace_count = 0
    start_pos: Optional[int] = None
    for idx in range(last_brace, -1, -1):
        if cleaned[idx] == "}":
            brace_count += 1
        elif cleaned[idx] == "{":
            brace_count -= 1
            if brace_count == 0:
                start_pos = idx
                break
    if start_pos is None:
        return cleaned
    return cleaned[start_pos : last_brace + 1]


def _extract_action_label(raw_text: str) -> str:
    cleaned = _clean_json_response(raw_text)
    try:
        parsed = ActionResponse.model_validate_json(cleaned)
        return parsed.action.to_str()
    except Exception:
        match = re.search(r"\b(UP|DOWN|LEFT|RIGHT)\b", raw_text.upper())
        if match:
            return match.group(1)
        raise ValueError("Unable to parse observed action from model output.")


class BehavioralProbeLLM(BaseLLMInterface):
    """Minimal black-box client for behavioral probe questions."""

    def __init__(self, model_name: str, temperature: float) -> None:
        super().__init__(model_name=model_name, temperature=temperature, template_path=None)

    def ask_text(self, prompt: str, *, seed: int) -> str:
        response, _, _ = self._make_completion_request(
            prompt,
            response_format=None,
            seed=seed,
            reasoning_effort="low",
            allowed_openai_params=["seed", "reasoning_effort"],
        )
        assert isinstance(response, str)
        return response

    def ask_text_with_logprobs(
        self,
        prompt: str,
        *,
        seed: int,
        top_logprobs: int = 20,
    ) -> tuple[str, Any]:
        response, _, raw_response = self._make_completion_request(
            prompt,
            response_format=None,
            seed=seed,
            reasoning_effort="low",
            logprobs=True,
            top_logprobs=top_logprobs,
            allowed_openai_params=["seed", "reasoning_effort", "logprobs", "top_logprobs"],
        )
        assert isinstance(response, str)
        return response, raw_response

    def ask_action(self, prompt: str, *, seed: int) -> tuple[str, str]:
        raw_text = self.ask_text(prompt, seed=seed)
        return _extract_action_label(raw_text), raw_text


def _normalize_raw_answer_token(token_text: str | None) -> str | None:
    if token_text is None:
        return None
    cleaned = token_text.strip().lower()
    if not cleaned:
        return None
    while cleaned and cleaned[-1] in {".", "!", "?", '"', "'", " "}:
        cleaned = cleaned[:-1].rstrip()
    while cleaned and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:].lstrip()
    if cleaned in {"a", "b", "c"}:
        return cleaned.upper()
    if cleaned in DIRECT_SEMANTIC_TOKENS:
        return cleaned
    return None


def _semantic_answer_from_raw_token(raw_token: str | None) -> str:
    if raw_token is None:
        return "invalid"
    if raw_token in RAW_LABEL_TOKENS:
        return BEHAVIORAL_PROBE_LABEL_MAP[raw_token.lower()]
    if raw_token in DIRECT_SEMANTIC_TOKENS:
        return raw_token
    return "invalid"


def extract_raw_answer_candidates_from_logprobs(logprobs: Any) -> dict[str, float]:
    probabilities = {token: 0.0 for token in [*RAW_LABEL_TOKENS, *DIRECT_SEMANTIC_TOKENS]}
    if logprobs is None or getattr(logprobs, "content", None) is None:
        return probabilities

    answer_entry = None
    for entry in logprobs.content:
        token_text = _normalize_raw_answer_token(getattr(entry, "token", None))
        if token_text is not None:
            answer_entry = entry
            break

    if answer_entry is None:
        return probabilities

    candidates: dict[str, float] = {}

    def add_candidate(token_text: str | None, logprob_value: float | None) -> None:
        if token_text is None or logprob_value is None or not math.isfinite(logprob_value):
            return
        normalized = _normalize_raw_answer_token(token_text)
        if normalized is None:
            return
        candidates[normalized] = candidates.get(normalized, 0.0) + math.exp(logprob_value)

    add_candidate(getattr(answer_entry, "token", None), getattr(answer_entry, "logprob", None))
    for top in getattr(answer_entry, "top_logprobs", []) or []:
        add_candidate(getattr(top, "token", None), getattr(top, "logprob", None))

    total = sum(candidates.values())
    if total <= 0.0:
        return probabilities
    for token, value in candidates.items():
        probabilities[token] = value / total
    return probabilities


def map_raw_candidates_to_semantic_probs(raw_probs: dict[str, float]) -> dict[str, float]:
    probabilities = {answer: 0.0 for answer in VALID_BEHAVIORAL_PROBE_ANSWERS}
    if not raw_probs:
        probabilities["invalid"] = 1.0
        return probabilities

    total = 0.0
    for raw_token, prob in raw_probs.items():
        semantic = _semantic_answer_from_raw_token(raw_token)
        if semantic == "invalid":
            continue
        probabilities[semantic] += float(prob)
        total += float(prob)
    if total <= 0.0:
        probabilities["invalid"] = 1.0
        return probabilities
    semantic_total = sum(probabilities.values())
    for answer in probabilities:
        probabilities[answer] = probabilities[answer] / semantic_total if semantic_total else 0.0
    return probabilities


def _probabilities_from_choice_logprobs(logprobs: Any) -> dict[str, float]:
    return map_raw_candidates_to_semantic_probs(extract_raw_answer_candidates_from_logprobs(logprobs))


def _extract_answer_token_logprobs(logprobs: Any) -> list[dict[str, Any]] | None:
    if logprobs is None or getattr(logprobs, "content", None) is None:
        return None
    for entry in logprobs.content:
        token_text = _normalize_raw_answer_token(getattr(entry, "token", None))
        if token_text is None:
            continue
        return [
            {
                "token": getattr(entry, "token", None),
                "logprob": getattr(entry, "logprob", None),
                "top_logprobs": [
                    {
                        "token": getattr(top, "token", None),
                        "logprob": getattr(top, "logprob", None),
                    }
                    for top in (getattr(entry, "top_logprobs", []) or [])
                ],
            }
        ]
    return None


def _probabilities_from_mc_answers(sampled_answers: list[str]) -> dict[str, float]:
    probabilities = {answer: 0.0 for answer in VALID_BEHAVIORAL_PROBE_ANSWERS}
    if not sampled_answers:
        probabilities["invalid"] = 1.0
        return probabilities
    total = len(sampled_answers)
    for answer in sampled_answers:
        probabilities[answer] = probabilities.get(answer, 0.0) + 1.0 / total
    return probabilities


def _argmax_label(probabilities: dict[str, float]) -> str:
    if not probabilities:
        return "invalid"
    return max(probabilities.items(), key=lambda item: item[1])[0]


def _build_probability_diagnostics_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        if row["answer_space"] != "label3":
            continue
        raw_probs_t0 = json.loads(row["raw_answer_probs_t0_json"])
        raw_probs_t07 = json.loads(row["raw_answer_probs_t07_json"])
        raw_probs_t1 = json.loads(row["raw_answer_probs_t1_json"])
        probs_t0 = json.loads(row["answer_probs_t0_json"])
        probs_t07 = json.loads(row["answer_probs_t07_json"])
        probs_t1 = json.loads(row["answer_probs_t1_json"])
        mc_probs = json.loads(row["mc_answer_probs_json"])
        diagnostics.append(
            {
                "example_id": row["example_id"],
                "question_id": row["question_id"],
                "ground_truth_label": row["ground_truth_label"],
                "greedy_answer": row["greedy_answer"],
                "raw_argmax_t0": _argmax_label(raw_probs_t0),
                "raw_argmax_t07": _argmax_label(raw_probs_t07),
                "raw_argmax_t1": _argmax_label(raw_probs_t1),
                "semantic_argmax_t0": _argmax_label(probs_t0),
                "semantic_argmax_t07": _argmax_label(probs_t07),
                "semantic_argmax_t1": _argmax_label(probs_t1),
                "p_yes_t0": probs_t0.get("yes", 0.0),
                "p_no_t0": probs_t0.get("no", 0.0),
                "yes_gt_no_t0": probs_t0.get("yes", 0.0) > probs_t0.get("no", 0.0),
                "p_yes_t07": probs_t07.get("yes", 0.0),
                "p_no_t07": probs_t07.get("no", 0.0),
                "yes_gt_no_t07": probs_t07.get("yes", 0.0) > probs_t07.get("no", 0.0),
                "p_yes_t1": probs_t1.get("yes", 0.0),
                "p_no_t1": probs_t1.get("no", 0.0),
                "yes_gt_no_t1": probs_t1.get("yes", 0.0) > probs_t1.get("no", 0.0),
                "mc_answer": row["mc_answer"],
                "mc_yes_no_answer": row.get(
                    "mc_yes_no_answer",
                    "yes" if mc_probs.get("yes", 0.0) > mc_probs.get("no", 0.0) else "no",
                ),
                "mc_p_yes": mc_probs.get("yes", 0.0),
                "mc_p_no": mc_probs.get("no", 0.0),
                "mc_yes_gt_no": mc_probs.get("yes", 0.0) > mc_probs.get("no", 0.0),
                "raw_answer_probs_t0_json": row["raw_answer_probs_t0_json"],
                "raw_answer_probs_t07_json": row["raw_answer_probs_t07_json"],
                "raw_answer_probs_t1_json": row["raw_answer_probs_t1_json"],
                "answer_probs_t0_json": row["answer_probs_t0_json"],
                "answer_probs_t07_json": row["answer_probs_t07_json"],
                "answer_probs_t1_json": row["answer_probs_t1_json"],
                "mc_answer_probs_json": row["mc_answer_probs_json"],
            }
        )
    return diagnostics


def _smoke_instances() -> list[dict[str, Any]]:
    validate_smoke_examples()
    instances: list[dict[str, Any]] = []
    for example in SMOKE_TEST_STATES:
        instances.append(
            {
                "example_id": example.example_id,
                "grid_text": example.grid_text,
                "carrying_key": example.carrying_key,
                "observed_action": example.observed_action,
                "probe_truths": derive_probe_truths(example),
                "notes": example.notes,
                "source": "smoke_test",
            }
        )
    return instances


def _collect_label3_logprob_readout(
    *,
    client: BehavioralProbeLLM,
    prompt: str,
    seed: int,
    top_logprobs: int,
) -> dict[str, Any]:
    raw_text, raw_response = client.ask_text_with_logprobs(prompt, seed=seed, top_logprobs=top_logprobs)
    logprobs = raw_response.choices[0].logprobs if raw_response is not None else None
    raw_probs = extract_raw_answer_candidates_from_logprobs(logprobs)
    semantic_probs = map_raw_candidates_to_semantic_probs(raw_probs)
    return {
        "raw_output": raw_text,
        "visible_answer": parse_behavioral_probe_answer(raw_text),
        "raw_probs": raw_probs,
        "semantic_probs": semantic_probs,
        "answer": sampled_answer_from_probabilities(semantic_probs),
        "entropy": shannon_entropy(semantic_probs),
        "answer_token_logprobs": _extract_answer_token_logprobs(logprobs),
    }


def _run_probe_rows(
    *,
    model_name: str,
    examples: list[dict[str, Any]],
    questions: list[BehavioralProbeQuestion],
    mc_sample_repeats: int,
    mc_temperature: float,
    prompt_preset: str,
    logprob_temperatures: tuple[float, ...],
    action_client: BehavioralProbeLLM | None = None,
    logprob_clients: dict[str, BehavioralProbeLLM] | None = None,
    mc_client: BehavioralProbeLLM | None = None,
    top_logprobs: int = 20,
    verbose: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if prompt_preset not in PROMPT_PRESETS:
        raise ValueError(f"Unknown prompt preset: {prompt_preset}")
    if any(temp not in SUPPORTED_LOGPROB_TEMPS for temp in logprob_temperatures):
        raise ValueError(f"Unsupported logprob temperature requested: {logprob_temperatures}")

    action_client = action_client or BehavioralProbeLLM(model_name, temperature=0.0)
    logprob_clients = logprob_clients or {
        _temp_suffix(temp): BehavioralProbeLLM(model_name, temperature=temp)
        for temp in logprob_temperatures
    }
    mc_client = mc_client or BehavioralProbeLLM(model_name, temperature=mc_temperature)

    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    total_examples = len(examples)
    total_questions = len(questions)

    for example_idx, example in enumerate(examples, start=1):
        if verbose:
            logger.info("Behavioral probe example %s/%s: %s", example_idx, total_examples, example["example_id"])
        observed_action_prompt = _observed_action_prompt(
            example["grid_text"],
            bool(example["carrying_key"]),
            prompt_preset,
        )
        if example.get("observed_action") is not None:
            observed_action = example["observed_action"]
            observed_action_raw = None
            observed_action_source = "dataset"
        else:
            try:
                observed_action, observed_action_raw = action_client.ask_action(observed_action_prompt, seed=0)
                observed_action_source = "model_query"
            except Exception as exc:
                observed_action = "INVALID"
                observed_action_raw = f"ERROR: {exc}"
                observed_action_source = "model_query_error"
                logger.error(
                    "Observed action query failed for %s: %s\n%s",
                    example["example_id"],
                    exc,
                    traceback.format_exc(),
                )

        for question_idx, question in enumerate(questions, start=1):
            if verbose:
                logger.info("  Question %s/%s: %s", question_idx, total_questions, question.question_id)
            prompt = _belief_prompt(
                grid_text=example["grid_text"],
                carrying_key=bool(example["carrying_key"]),
                question=question,
                prompt_preset=prompt_preset,
            )
            ground_truth = example["probe_truths"][question.target_variable]

            if question.answer_space == "coord_json":
                greedy_raw = ""
                greedy_coord: dict[str, int] | None = None
                try:
                    greedy_raw = logprob_clients["t0"].ask_text(prompt, seed=0)
                    greedy_coord = parse_coordinate_answer(greedy_raw)
                except Exception as exc:
                    greedy_raw = f"ERROR: {exc}"
                    logger.error(
                        "Greedy coordinate query failed for %s/%s: %s\n%s",
                        example["example_id"],
                        question.question_id,
                        exc,
                        traceback.format_exc(),
                    )

                mc_raw_outputs: list[str] = []
                mc_coords: list[dict[str, int] | None] = []
                for seed in range(1, mc_sample_repeats + 1):
                    try:
                        mc_raw = mc_client.ask_text(prompt, seed=seed)
                    except Exception as exc:
                        mc_raw = f"ERROR: {exc}"
                    mc_raw_outputs.append(mc_raw)
                    mc_coords.append(parse_coordinate_answer(mc_raw))
                mc_summary = summarize_coordinate_samples(mc_coords, ground_truth)
                row = {
                    "example_id": example["example_id"],
                    "question_id": question.question_id,
                    "target_variable": question.target_variable,
                    "question_family": question.family,
                    "answer_space": question.answer_space,
                    "prompt_preset": prompt_preset,
                    "grid_text": example["grid_text"],
                    "carrying_key": example["carrying_key"],
                    "observed_action": observed_action,
                    "greedy_raw_output": greedy_raw,
                    "greedy_coordinate_json": json.dumps(greedy_coord or {"row": -1, "col": -1}, sort_keys=True),
                    "greedy_coordinate_key": "invalid" if greedy_coord is None else f"{greedy_coord['row']},{greedy_coord['col']}",
                    "ground_truth_coordinate_json": json.dumps(ground_truth, sort_keys=True),
                    "ground_truth_coordinate_key": f"{ground_truth['row']},{ground_truth['col']}",
                    "greedy_exact_match": 1.0 if greedy_coord == ground_truth else 0.0,
                    "greedy_manhattan_distance": (
                        "" if greedy_coord is None else coordinate_manhattan_distance(greedy_coord, ground_truth)
                    ),
                    "mc_modal_coordinate_json": json.dumps(mc_summary["modal_coordinate"], sort_keys=True),
                    "mc_coordinates_json": json.dumps(
                        [coord if coord is not None else {"row": -1, "col": -1} for coord in mc_coords],
                        sort_keys=True,
                    ),
                    "mc_modal_exact_match": 1.0 if mc_summary["modal_coordinate"] == ground_truth else 0.0,
                    "mc_modal_manhattan_distance": coordinate_manhattan_distance(
                        mc_summary["modal_coordinate"], ground_truth
                    ),
                    "mc_mean_manhattan_distance": mc_summary["mean_manhattan_distance"],
                    "mc_unique_coordinate_count": mc_summary["unique_coordinate_count"],
                    "mc_valid_parse_rate_for_row": mc_summary["valid_parse_rate"],
                }
                rows.append(row)
                raw_rows.append(
                    {
                        "example_id": example["example_id"],
                        "question_id": question.question_id,
                        "prompt": prompt,
                        "observed_action_prompt": observed_action_prompt,
                        "observed_action_raw": observed_action_raw,
                        "observed_action_source": observed_action_source,
                        "mc_raw_outputs": mc_raw_outputs,
                        "parsed_row": row,
                    }
                )
                continue

            readouts: dict[str, dict[str, Any]] = {}
            for temp in logprob_temperatures:
                suffix = _temp_suffix(temp)
                try:
                    readouts[suffix] = _collect_label3_logprob_readout(
                        client=logprob_clients[suffix],
                        prompt=prompt,
                        seed=0,
                        top_logprobs=top_logprobs,
                    )
                except Exception as exc:
                    logger.error(
                        "Logprob belief query failed for %s/%s at %s: %s\n%s",
                        example["example_id"],
                        question.question_id,
                        suffix,
                        exc,
                        traceback.format_exc(),
                    )
                    invalid_probs = {token: 0.0 for token in [*RAW_LABEL_TOKENS, *DIRECT_SEMANTIC_TOKENS]}
                    invalid_semantic = {answer: 0.0 for answer in VALID_BEHAVIORAL_PROBE_ANSWERS}
                    invalid_semantic["invalid"] = 1.0
                    readouts[suffix] = {
                        "raw_output": f"ERROR: {exc}",
                        "visible_answer": "invalid",
                        "raw_probs": invalid_probs,
                        "semantic_probs": invalid_semantic,
                        "answer": "invalid",
                        "entropy": 0.0,
                        "answer_token_logprobs": None,
                    }

            greedy_raw = readouts["t0"]["raw_output"]
            greedy_answer = readouts["t0"]["visible_answer"]

            mc_raw_outputs: list[str] = []
            mc_answers: list[str] = []
            for seed in range(1, mc_sample_repeats + 1):
                try:
                    mc_raw = mc_client.ask_text(prompt, seed=seed)
                except Exception as exc:
                    mc_raw = f"ERROR: {exc}"
                mc_raw_outputs.append(mc_raw)
                mc_answers.append(parse_behavioral_probe_answer(mc_raw))
            mc_probs = _probabilities_from_mc_answers(mc_answers)
            mc_answer = sampled_answer_from_probabilities(mc_probs)
            mc_yes_no_answer = yes_no_answer_from_probabilities(mc_probs)

            row = {
                "example_id": example["example_id"],
                "question_id": question.question_id,
                "target_variable": question.target_variable,
                "question_family": question.family,
                "answer_space": question.answer_space,
                "variant_id": question.variant_id or "",
                "prompt_preset": prompt_preset,
                "grid_text": example["grid_text"],
                "carrying_key": example["carrying_key"],
                "observed_action": observed_action,
                "greedy_answer": greedy_answer,
                "ground_truth_label": ground_truth,
                "belief_action_consistency": belief_action_consistency(
                    readouts["t0"]["answer"], observed_action, question.question_id
                ),
                "mc_belief_action_consistency": belief_action_consistency(
                    mc_answer, observed_action, question.question_id
                ),
                "mc_answer": mc_answer,
                "mc_yes_no_answer": mc_yes_no_answer,
                "mc_answers_json": json.dumps(mc_answers),
                "mc_answer_probs_json": json.dumps(mc_probs, sort_keys=True),
                "mc_entropy": shannon_entropy(mc_probs),
                "mc_valid_parse_rate_for_row": valid_parse_rate_from_answers(mc_answers),
                # Backward-compatible aliases
                "sampled_answer": readouts["t1"]["answer"],
                "answer_probs_json": json.dumps(readouts["t1"]["semantic_probs"], sort_keys=True),
                "entropy": readouts["t1"]["entropy"],
            }
            for suffix in ("t0", "t07", "t1"):
                readout = readouts[suffix]
                row[f"logprob_{suffix}_raw_output"] = readout["raw_output"]
                row[f"raw_answer_probs_{suffix}_json"] = json.dumps(readout["raw_probs"], sort_keys=True)
                row[f"answer_probs_{suffix}_json"] = json.dumps(readout["semantic_probs"], sort_keys=True)
                row[f"logprob_answer_{suffix}"] = readout["answer"]
                row[f"logprob_yes_no_answer_{suffix}"] = yes_no_answer_from_probabilities(
                    readout["semantic_probs"]
                )
                row[f"entropy_{suffix}"] = readout["entropy"]
                row[f"valid_parse_rate_{suffix}_for_row"] = valid_parse_rate(
                    greedy_answer, readout["answer"]
                )
            rows.append(row)
            raw_rows.append(
                {
                    "example_id": example["example_id"],
                    "question_id": question.question_id,
                    "prompt": prompt,
                    "observed_action_prompt": observed_action_prompt,
                    "observed_action_raw": observed_action_raw,
                    "observed_action_source": observed_action_source,
                    "greedy_raw_output": greedy_raw,
                    "mc_raw_outputs": mc_raw_outputs,
                    "logprob_t0_answer_token_logprobs": readouts["t0"]["answer_token_logprobs"],
                    "logprob_t07_answer_token_logprobs": readouts["t07"]["answer_token_logprobs"],
                    "logprob_t1_answer_token_logprobs": readouts["t1"]["answer_token_logprobs"],
                    "parsed_row": row,
                }
            )

    config = {
        "model_name": model_name,
        "prompt_preset": prompt_preset,
        "logprob_temperatures": list(logprob_temperatures),
        "mc_temperature": mc_temperature,
        "mc_sample_repeats": mc_sample_repeats,
        "top_logprobs": top_logprobs,
    }
    return rows, raw_rows, config


def _write_behavioral_outputs(
    *,
    out_dir: Path,
    rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    config: dict[str, Any],
    dataset_validation: list[dict[str, Any]] | None,
    smoke_examples: list[dict[str, Any]] | None,
    min_valid_parse_rate: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize_probe_rows(rows)
    probability_diagnostics_rows = _build_probability_diagnostics_rows(rows)
    raw_payload = {
        "config": {**config, "min_valid_parse_rate": min_valid_parse_rate},
        "dataset_validation": dataset_validation,
        "smoke_examples": smoke_examples,
        "rows": raw_rows,
        "summary": summary_rows,
        "probability_diagnostics": probability_diagnostics_rows,
    }

    _write_csv(out_dir / "behavioral_probe_rows.csv", rows)
    _write_csv(out_dir / "behavioral_probe_summary.csv", summary_rows)
    _write_csv(out_dir / "behavioral_probe_probability_diagnostics.csv", probability_diagnostics_rows)
    (out_dir / "behavioral_probe_raw.json").write_text(json.dumps(raw_payload, indent=2))

    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    label_rows = [row for row in summary_rows if row.get("answer_space") == "label3"]
    coord_rows = [row for row in summary_rows if row.get("answer_space") == "coord_json"]
    if label_rows:
        plot_behavioral_probe_summary(label_rows, figs_dir / "behavioral_probe_summary.png")
        try:
            plot_directional_probe_heatmap(label_rows, figs_dir / "behavioral_probe_directional_heatmap.png")
        except ValueError:
            pass
    if coord_rows:
        plot_coordinate_probe_summary(coord_rows, figs_dir / "behavioral_probe_coordinate_summary.png")

    failures = [
        row
        for row in label_rows
        if float(row["valid_parse_rate"]) < min_valid_parse_rate
        or float(row["valid_parse_rate_t07"]) < min_valid_parse_rate
        or float(row["valid_parse_rate_t1"]) < min_valid_parse_rate
        or float(row["mc_valid_parse_rate"]) < min_valid_parse_rate
    ]
    if failures:
        failed_questions = ", ".join(f"{row['question_id']}={row['valid_parse_rate']:.3f}" for row in failures)
        raise ValueError(
            "Behavioral probe run failed due to poor parse rate: " + failed_questions
        )


def run_behavioral_probe_on_instances(
    *,
    model_name: str,
    instances: list[dict[str, Any]],
    questions: list[BehavioralProbeQuestion],
    output_dir: str,
    prompt_preset: str = "default_observable_state",
    min_valid_parse_rate: float = 0.90,
    mc_sample_repeats: int = 10,
    mc_temperature: float = 0.7,
    top_logprobs: int = 20,
    logprob_temperatures: tuple[float, ...] = (0.0, 0.7, 1.0),
    verbose: bool = True,
) -> None:
    rows, raw_rows, config = _run_probe_rows(
        model_name=model_name,
        examples=instances,
        questions=questions,
        mc_sample_repeats=mc_sample_repeats,
        mc_temperature=mc_temperature,
        prompt_preset=prompt_preset,
        logprob_temperatures=logprob_temperatures,
        top_logprobs=top_logprobs,
        verbose=verbose,
    )
    _write_behavioral_outputs(
        out_dir=Path(output_dir),
        rows=rows,
        raw_rows=raw_rows,
        config=config,
        dataset_validation=None,
        smoke_examples=None,
        min_valid_parse_rate=min_valid_parse_rate,
    )


def run_behavioral_probe_smoke_test(
    model_name: str = "together_ai/openai/gpt-oss-20b",
    output_dir: str = "data/behavioral_probes/smoke_test",
    min_valid_parse_rate: float = 0.90,
    mc_sample_repeats: int = 10,
    mc_temperature: float = 0.7,
    top_logprobs: int = 20,
    prompt_preset: str = "default_observable_state",
    question_family: str = "all",
    answer_space: str | None = None,
    door_open_after_action_variant: str = "observed",
    logprob_temperatures: tuple[float, ...] = (0.0, 0.7, 1.0),
    verbose: bool = True,
) -> None:
    """Run the manual DoorKey behavioral-probe smoke test."""
    if answer_space != "coord_json" and any(
        required not in logprob_temperatures for required in SUPPORTED_LOGPROB_TEMPS
    ):
        raise ValueError("Label3 smoke tests require logprob temperatures 0.0, 0.7, and 1.0.")

    questions = get_behavioral_probe_questions(
        question_family=question_family,
        answer_space=answer_space,
        door_open_after_action_variant=door_open_after_action_variant,
    )
    rows, raw_rows, config = _run_probe_rows(
        model_name=model_name,
        examples=_smoke_instances(),
        questions=questions,
        mc_sample_repeats=mc_sample_repeats,
        mc_temperature=mc_temperature,
        prompt_preset=prompt_preset,
        logprob_temperatures=logprob_temperatures,
        top_logprobs=top_logprobs,
        verbose=verbose,
    )
    _write_behavioral_outputs(
        out_dir=Path(output_dir),
        rows=rows,
        raw_rows=raw_rows,
        config=config,
        dataset_validation=validate_smoke_examples(),
        smoke_examples=[asdict(example) for example in SMOKE_TEST_STATES],
        min_valid_parse_rate=min_valid_parse_rate,
    )


def run_behavioral_probe_door_semantics_ablation(
    model_name: str = "together_ai/openai/gpt-oss-20b",
    output_dir: str = "data/behavioral_probes/door_semantics_ablation",
    min_valid_parse_rate: float = 0.90,
    mc_sample_repeats: int = 10,
    mc_temperature: float = 0.7,
    top_logprobs: int = 20,
    prompt_preset: str = "default_observable_state",
    verbose: bool = True,
) -> None:
    """Run a focused semantics ablation for door_open_after_right, then rerun the winner on full smoke."""
    out_dir = Path(output_dir)
    focused_ids = {
        "smoke_004_carrying_open_door_visible",
        "smoke_005_carrying_wall_right_open_door",
        "smoke_010_open_door_elsewhere",
    }
    focused_examples = [item for item in _smoke_instances() if item["example_id"] in focused_ids]

    focused_rows: list[dict[str, Any]] = []
    focused_summary: list[dict[str, Any]] = []
    winner_variant = "observed"
    winner_score = (-1.0, -1.0, -1.0)

    for variant_id in DOOR_OPEN_AFTER_ACTION_VARIANTS:
        questions = [
            question
            for question in get_behavioral_probe_questions(
                question_family="action_effects",
                answer_space="label3",
                door_open_after_action_variant=variant_id,
            )
            if question.question_id == "door_open_after_right"
        ]
        rows, _, _ = _run_probe_rows(
            model_name=model_name,
            examples=focused_examples,
            questions=questions,
            mc_sample_repeats=mc_sample_repeats,
            mc_temperature=mc_temperature,
            prompt_preset=prompt_preset,
            logprob_temperatures=SUPPORTED_LOGPROB_TEMPS,
            top_logprobs=top_logprobs,
            verbose=verbose,
        )
        focused_rows.extend(rows)
        summary = summarize_probe_rows(rows)[0]
        summary["variant_id"] = variant_id
        focused_summary.append(summary)
        score = (
            float(summary["greedy_mc_agreement"]),
            float(summary["mc_accuracy"]),
            float(summary["greedy_logprob_t0_agreement"]),
        )
        if score > winner_score:
            winner_variant = variant_id
            winner_score = score

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "focused_rows.csv", focused_rows)
    _write_csv(out_dir / "focused_summary.csv", focused_summary)
    (out_dir / "decision.json").write_text(
        json.dumps({"winner_variant": winner_variant, "winner_score": winner_score}, indent=2)
    )
    run_behavioral_probe_smoke_test(
        model_name=model_name,
        output_dir=str(out_dir / "winning_full_smoke"),
        min_valid_parse_rate=min_valid_parse_rate,
        mc_sample_repeats=mc_sample_repeats,
        mc_temperature=mc_temperature,
        top_logprobs=top_logprobs,
        prompt_preset=prompt_preset,
        question_family="all",
        answer_space=None,
        door_open_after_action_variant=winner_variant,
        logprob_temperatures=SUPPORTED_LOGPROB_TEMPS,
        verbose=verbose,
    )


def run_behavioral_probe_prompt_ablation(
    model_name: str = "together_ai/openai/gpt-oss-20b",
    output_dir: str = "data/behavioral_probes/prompt_ablation",
    prompt_presets: tuple[str, ...] = (
        "default_observable_state",
        "cardinal_action_explicit",
        "coordinate_output_explicit",
    ),
    question_family: str = "all_label3",
    answer_space: str | None = "label3",
    mc_sample_repeats: int = 10,
    mc_temperature: float = 0.7,
    top_logprobs: int = 20,
    verbose: bool = True,
) -> None:
    """Compare prompt presets on the smoke dataset for a chosen question subset."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows: list[dict[str, Any]] = []
    questions = get_behavioral_probe_questions(question_family=question_family, answer_space=answer_space)
    for preset in prompt_presets:
        preset_dir = out_dir / preset
        run_behavioral_probe_smoke_test(
            model_name=model_name,
            output_dir=str(preset_dir),
            mc_sample_repeats=mc_sample_repeats,
            mc_temperature=mc_temperature,
            top_logprobs=top_logprobs,
            prompt_preset=preset,
            question_family=question_family,
            answer_space=answer_space,
            verbose=verbose,
        )
        summary_path = preset_dir / "behavioral_probe_summary.csv"
        with open(summary_path, newline="") as handle:
            for row in csv.DictReader(handle):
                row["prompt_preset"] = preset
                comparison_rows.append(row)
    _write_csv(out_dir / "prompt_ablation_summary.csv", comparison_rows)


__all__ = [
    "BehavioralProbeLLM",
    "PROMPT_PRESETS",
    "_build_probability_diagnostics_rows",
    "_probabilities_from_choice_logprobs",
    "_run_probe_rows",
    "extract_raw_answer_candidates_from_logprobs",
    "map_raw_candidates_to_semantic_probs",
    "run_behavioral_probe_door_semantics_ablation",
    "run_behavioral_probe_on_instances",
    "run_behavioral_probe_prompt_ablation",
    "run_behavioral_probe_smoke_test",
]
