"""Black-box behavioral-probe smoke test for DoorKey."""

from __future__ import annotations

import csv
import json
import logging
import re
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from reveng.agents.llm_templates import ActionResponse
from reveng.experiments.behavioral_probe_metrics import (
    answer_probabilities,
    belief_action_consistency,
    modal_answer,
    shannon_entropy,
    summarize_probe_rows,
    valid_parse_rate,
)
from reveng.experiments.behavioral_probe_plots import (
    plot_behavioral_probe_summary,
)
from reveng.experiments.behavioral_probe_parse import parse_behavioral_probe_answer
from reveng.experiments.behavioral_probe_questions import (
    BEHAVIORAL_PROBE_QUESTIONS,
    BehavioralProbeQuestion,
)
from reveng.experiments.behavioral_probe_smoke_data import (
    SMOKE_TEST_STATES,
    validate_smoke_examples,
)
from reveng.llm_interface import BaseLLMInterface

logger = logging.getLogger(__name__)


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


def _behavioral_probe_preamble() -> str:
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
        "Interpret RIGHT as the cell one column to the right (east) in the rendered grid.\n"
        "Answer using only the current observable state shown below.\n"
    )


def _observed_action_prompt(grid_text: str, carrying_key: bool) -> str:
    return (
        _behavioral_probe_preamble()
        + "\n"
        + "Choose the next move that best advances toward the goal while respecting the "
        + "DoorKey mechanics in the current state.\n\n"
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
) -> str:
    prompt = (
        _behavioral_probe_preamble()
        + "\n"
        + "# Inputs\n\n"
        + "Current grid state:\n\n"
        + grid_text
        + "\n\n"
    )
    if question.show_carrying_key:
        prompt += f"Agent status:\n- Carrying key: {str(carrying_key).lower()}\n\n"
    prompt += question.prompt_text
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

    def ask_action(self, prompt: str, *, seed: int) -> tuple[str, str]:
        raw_text = self.ask_text(prompt, seed=seed)
        return _extract_action_label(raw_text), raw_text


def _run_behavioral_probe_smoke_test(
    *,
    model_name: str,
    output_dir: str,
    sampled_repeats: int = 10,
    min_valid_parse_rate: float = 0.90,
    verbose: bool = True,
    greedy_client: BehavioralProbeLLM | None = None,
    sampled_client: BehavioralProbeLLM | None = None,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_validation = validate_smoke_examples()
    if verbose:
        logger.info(
            "Starting behavioral probe smoke test: model=%s, examples=%s, questions=%s, sampled_repeats=%s, output_dir=%s",
            model_name,
            len(SMOKE_TEST_STATES),
            len(BEHAVIORAL_PROBE_QUESTIONS),
            sampled_repeats,
            out_dir,
        )
    greedy_client = greedy_client or BehavioralProbeLLM(model_name, temperature=0.0)
    sampled_client = sampled_client or BehavioralProbeLLM(model_name, temperature=0.7)

    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []

    total_examples = len(SMOKE_TEST_STATES)
    total_questions = len(BEHAVIORAL_PROBE_QUESTIONS)

    for example_idx, example in enumerate(SMOKE_TEST_STATES, start=1):
        if verbose:
            logger.info(
                "Behavioral probe example %s/%s: %s",
                example_idx,
                total_examples,
                example.example_id,
            )
        observed_action_prompt = _observed_action_prompt(
            example.grid_text,
            example.carrying_key,
        )
        if example.observed_action is not None:
            observed_action = example.observed_action
            observed_action_raw = None
            observed_action_source = "manual_override"
        else:
            try:
                observed_action, observed_action_raw = greedy_client.ask_action(
                    observed_action_prompt,
                    seed=0,
                )
                observed_action_source = "model_query"
            except Exception as exc:
                observed_action = "INVALID"
                observed_action_raw = f"ERROR: {exc}"
                observed_action_source = "model_query_error"
                logger.error(
                    "Observed action query failed for %s: %s\n%s",
                    example.example_id,
                    exc,
                    traceback.format_exc(),
                )
        if verbose:
            logger.info(
                "Observed action for %s: %s (%s)",
                example.example_id,
                observed_action,
                observed_action_source,
            )

        for question_idx, question in enumerate(BEHAVIORAL_PROBE_QUESTIONS, start=1):
            if verbose:
                logger.info(
                    "  Question %s/%s: %s",
                    question_idx,
                    total_questions,
                    question.question_id,
                )
            prompt = _belief_prompt(
                grid_text=example.grid_text,
                carrying_key=example.carrying_key,
                question=question,
            )

            try:
                greedy_raw = greedy_client.ask_text(prompt, seed=0)
            except Exception as exc:
                greedy_raw = f"ERROR: {exc}"
                logger.error(
                    "Greedy belief query failed for %s/%s: %s\n%s",
                    example.example_id,
                    question.question_id,
                    exc,
                    traceback.format_exc(),
                )
            greedy_answer = parse_behavioral_probe_answer(greedy_raw)

            sampled_raw_outputs: list[str] = []
            sampled_answers: list[str] = []
            for seed in range(1, sampled_repeats + 1):
                try:
                    sampled_raw = sampled_client.ask_text(prompt, seed=seed)
                except Exception as exc:
                    sampled_raw = f"ERROR: {exc}"
                    logger.error(
                        "Sampled belief query failed for %s/%s seed=%s: %s\n%s",
                        example.example_id,
                        question.question_id,
                        seed,
                        exc,
                        traceback.format_exc(),
                    )
                sampled_raw_outputs.append(sampled_raw)
                sampled_answers.append(parse_behavioral_probe_answer(sampled_raw))
                if verbose:
                    logger.info(
                        "    sample %s/%s parsed=%s",
                        seed,
                        sampled_repeats,
                        sampled_answers[-1],
                    )

            probs = answer_probabilities(sampled_answers)
            modal = modal_answer(sampled_answers)
            row = {
                "example_id": example.example_id,
                "question_id": question.question_id,
                "target_variable": question.target_variable,
                "grid_text": example.grid_text,
                "carrying_key": example.carrying_key,
                "observed_action": observed_action,
                "greedy_answer": greedy_answer,
                "sampled_answers_json": json.dumps(sampled_answers),
                "modal_answer": modal,
                "answer_probs_json": json.dumps(probs, sort_keys=True),
                "entropy": shannon_entropy(probs),
                "valid_parse_rate_for_row": valid_parse_rate(greedy_answer, sampled_answers),
                "ground_truth_label": getattr(example, f"{question.question_id}_label"),
                "belief_action_consistency": belief_action_consistency(
                    modal,
                    observed_action,
                    question.question_id,
                ),
            }
            rows.append(row)
            if verbose:
                logger.info(
                    "  Completed %s/%s for %s: greedy=%s modal=%s valid_parse_rate=%.3f",
                    question_idx,
                    total_questions,
                    example.example_id,
                    greedy_answer,
                    modal,
                    float(row["valid_parse_rate_for_row"]),
                )
            raw_rows.append(
                {
                    "example_id": example.example_id,
                    "question_id": question.question_id,
                    "target_variable": question.target_variable,
                    "prompt": prompt,
                    "observed_action_prompt": observed_action_prompt,
                    "observed_action_source": observed_action_source,
                    "observed_action_raw": observed_action_raw,
                    "greedy_raw_output": greedy_raw,
                    "sampled_raw_outputs": sampled_raw_outputs,
                    "parsed_row": row,
                }
            )

    summary_rows = summarize_probe_rows(rows)
    raw_payload = {
        "config": {
            "model_name": model_name,
            "sampled_repeats": sampled_repeats,
            "min_valid_parse_rate": min_valid_parse_rate,
        },
        "dataset_validation": dataset_validation,
        "smoke_examples": [asdict(example) for example in SMOKE_TEST_STATES],
        "rows": raw_rows,
        "summary": summary_rows,
    }

    _write_csv(out_dir / "behavioral_probe_rows.csv", rows)
    _write_csv(out_dir / "behavioral_probe_summary.csv", summary_rows)
    (out_dir / "behavioral_probe_raw.json").write_text(json.dumps(raw_payload, indent=2))
    figure_path = plot_behavioral_probe_summary(
        summary_rows,
        out_dir / "figs" / "behavioral_probe_summary.png",
    )
    if verbose:
        logger.info(
            "Behavioral probe outputs written: rows=%s summary=%s raw=%s fig=%s",
            out_dir / "behavioral_probe_rows.csv",
            out_dir / "behavioral_probe_summary.csv",
            out_dir / "behavioral_probe_raw.json",
            figure_path,
        )

    failures = [
        row
        for row in summary_rows
        if float(row["valid_parse_rate"]) < min_valid_parse_rate
    ]
    if failures:
        failed_questions = ", ".join(
            f"{row['question_id']}={row['valid_parse_rate']:.3f}" for row in failures
        )
        raise ValueError(
            "Behavioral probe smoke test failed due to poor parse rate: "
            + failed_questions
        )


def run_behavioral_probe_smoke_test(
    model_name: str = "together_ai/openai/gpt-oss-20b",
    output_dir: str = "data/behavioral_probes/smoke_test",
    sampled_repeats: int = 10,
    min_valid_parse_rate: float = 0.90,
    verbose: bool = True,
) -> None:
    """Run the manual DoorKey behavioral-probe smoke test."""
    _run_behavioral_probe_smoke_test(
        model_name=model_name,
        output_dir=output_dir,
        sampled_repeats=sampled_repeats,
        min_valid_parse_rate=min_valid_parse_rate,
        verbose=verbose,
    )


__all__ = [
    "BehavioralProbeLLM",
    "_behavioral_probe_preamble",
    "_belief_prompt",
    "_extract_action_label",
    "_observed_action_prompt",
    "_run_behavioral_probe_smoke_test",
    "run_behavioral_probe_smoke_test",
]
