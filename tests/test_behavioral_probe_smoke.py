import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reveng.experiments.behavioral_probe_case_studies import (
    run_behavioral_probe_case_studies,
)
from reveng.experiments.behavioral_probe_merge import (
    merge_behavioral_probe_outputs,
)
from reveng.experiments.behavioral_probe_metrics import (
    belief_action_consistency,
    sampled_answer_from_probabilities,
    shannon_entropy,
    summarize_coordinate_samples,
    summarize_probe_rows,
    valid_parse_rate,
)
from reveng.experiments.behavioral_probe_parse import (
    parse_behavioral_probe_answer,
    parse_coordinate_answer,
)
from reveng.experiments.behavioral_probe_questions import (
    ACTIONS,
    BEHAVIORAL_PROBE_QUESTIONS,
    DOOR_OPEN_AFTER_ACTION_VARIANTS,
    PROMPT_FAMILIES,
    QUESTION_IDS,
    get_behavioral_probe_questions,
)
from reveng.experiments.behavioral_probe_runner import (
    _belief_prompt,
    _build_probability_diagnostics_rows,
    _extract_answer_token_logprobs,
    _probabilities_from_choice_logprobs,
    _run_probe_rows,
    _write_csv,
    _write_behavioral_outputs,
    extract_raw_answer_candidates_from_logprobs,
    map_raw_candidates_to_semantic_probs,
    run_behavioral_probe_door_semantics_ablation,
)
from reveng.experiments.behavioral_probe_smoke_data import (
    SMOKE_TEST_STATES,
    derive_example_truth,
    derive_probe_truths_from_state,
    grid_text_to_layout,
    validate_smoke_examples,
)
from reveng.experiments.behavioral_probe_trajectory_data import (
    mine_behavioral_probe_instances,
)


class FakeBehavioralProbeClient:
    def __init__(
        self,
        response_text: str,
        *,
        logprob_primary: str | None = None,
        top_candidates: list[tuple[str, float]] | None = None,
        action_text: str = "RIGHT",
        temperature: float = 0.0,
    ):
        self.response_text = response_text
        self.logprob_primary = logprob_primary or response_text
        self.top_candidates = top_candidates or [(self.logprob_primary, -0.1), ("B", -1.0), ("C", -2.0)]
        self.action_text = action_text
        self.temperature = temperature
        self.ask_text_calls: list[tuple[str, int]] = []
        self.ask_text_with_logprobs_calls: list[tuple[str, int, int]] = []
        self.ask_action_calls: list[tuple[str, int]] = []

    def ask_text(self, prompt: str, *, seed: int) -> str:
        self.ask_text_calls.append((prompt, seed))
        return self.response_text

    def ask_text_with_logprobs(self, prompt: str, *, seed: int, top_logprobs: int = 20):
        self.ask_text_with_logprobs_calls.append((prompt, seed, top_logprobs))
        raw = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    logprobs=SimpleNamespace(
                        content=[
                            SimpleNamespace(
                                token=self.logprob_primary,
                                logprob=-0.1,
                                top_logprobs=[
                                    SimpleNamespace(token=token, logprob=logprob)
                                    for token, logprob in self.top_candidates
                                ],
                            )
                        ]
                    )
                )
            ]
        )
        return self.response_text, raw

    def ask_action(self, prompt: str, *, seed: int):
        self.ask_action_calls.append((prompt, seed))
        return self.action_text, json.dumps({"action": self.action_text})


class FakeBehavioralProbeClientWithUsage(FakeBehavioralProbeClient):
    def ask_text_with_metadata(self, prompt: str, *, seed: int):
        self.ask_text_calls.append((prompt, seed))
        return self.response_text, {
            "cost_usd": 0.25,
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "reasoning_tokens": 1,
        }

    def ask_text_with_logprobs_and_metadata(self, prompt: str, *, seed: int, top_logprobs: int = 20):
        self.ask_text_with_logprobs_calls.append((prompt, seed, top_logprobs))
        raw = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    logprobs=SimpleNamespace(
                        content=[
                            SimpleNamespace(
                                token=self.logprob_primary,
                                logprob=-0.1,
                                top_logprobs=[
                                    SimpleNamespace(token=token, logprob=logprob)
                                    for token, logprob in self.top_candidates
                                ],
                            )
                        ]
                    )
                )
            ]
        )
        return self.response_text, raw, {
            "cost_usd": 0.5,
            "prompt_tokens": 20,
            "completion_tokens": 3,
            "total_tokens": 23,
            "reasoning_tokens": 2,
        }

    def ask_action_with_metadata(self, prompt: str, *, seed: int):
        self.ask_action_calls.append((prompt, seed))
        return self.action_text, json.dumps({"action": self.action_text}), {
            "cost_usd": 1.0,
            "prompt_tokens": 30,
            "completion_tokens": 4,
            "total_tokens": 34,
            "reasoning_tokens": 3,
        }


def _single_instance() -> list[dict]:
    example = SMOKE_TEST_STATES[0]
    return [
        {
            "example_id": example.example_id,
            "grid_text": example.grid_text,
            "carrying_key": example.carrying_key,
            "observed_action": "RIGHT",
            "probe_truths": derive_probe_truths_from_state(example.grid_text, example.carrying_key),
        }
    ]


def test_question_registry_expands_actions_and_coordinates():
    label3_questions = get_behavioral_probe_questions(question_family="all_label3", answer_space="label3")
    coord_questions = get_behavioral_probe_questions(question_family="coordinates", answer_space="coord_json")
    assert all(f"hit_wall_after_{action.lower()}" in [q.question_id for q in label3_questions] for action in ACTIONS)
    assert "agent_location" in [q.question_id for q in coord_questions]
    assert "agent_location_after_left" in [q.question_id for q in coord_questions]
    assert "coordinates" in PROMPT_FAMILIES
    assert len(set(QUESTION_IDS)) == len(QUESTION_IDS)
    assert "door_open_after_right" in QUESTION_IDS
    assert all("Answer with exactly one of: yes, no, unknown." not in q.prompt_text for q in label3_questions)


def test_door_variants_keep_same_question_ids():
    baseline = get_behavioral_probe_questions(question_family="action_effects", answer_space="label3")
    baseline_ids = [q.question_id for q in baseline]
    for variant in DOOR_OPEN_AFTER_ACTION_VARIANTS:
        variant_questions = get_behavioral_probe_questions(
            question_family="action_effects",
            answer_space="label3",
            door_open_after_action_variant=variant,
        )
        assert [q.question_id for q in variant_questions] == baseline_ids
        right_question = next(q for q in variant_questions if q.question_id == "door_open_after_right")
        assert right_question.variant_id == variant


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("yes", "yes"),
        ("Yes", "yes"),
        (" yes. ", "yes"),
        ('"unknown"', "unknown"),
        ("A", "yes"),
        ("b.", "no"),
        ("C", "unknown"),
        ("yes because of the wall", "invalid"),
    ],
)
def test_parse_behavioral_probe_answer(raw_text: str, expected: str):
    assert parse_behavioral_probe_answer(raw_text) == expected


def test_parse_coordinate_answer():
    assert parse_coordinate_answer('{"row": 3, "col": 4}') == {"row": 3, "col": 4}
    assert parse_coordinate_answer('answer: {"row": 1, "col": 2}') == {"row": 1, "col": 2}
    assert parse_coordinate_answer("not json") is None


def test_validate_smoke_examples_and_derived_truth():
    rows = validate_smoke_examples()
    assert len(rows) == len(SMOKE_TEST_STATES)
    example = next(item for item in SMOKE_TEST_STATES if item.example_id == "smoke_003_pick_key_right")
    derived = derive_example_truth(example)
    assert derived["has_key_after_right_label"] == "yes"
    assert derived["wall_right_label"] == "no"
    layout = grid_text_to_layout(example.grid_text)
    assert len(layout) == 9
    assert sum(cell == "A" for row in layout for cell in row) == 1


def test_metrics_helpers():
    probs = {"yes": 0.5, "no": 0.25, "unknown": 0.25, "invalid": 0.0}
    assert sampled_answer_from_probabilities(probs) == "yes"
    assert shannon_entropy(probs) > 0.0
    assert valid_parse_rate("yes", "no") == 1.0
    assert belief_action_consistency("yes", "RIGHT", "wall_right") == "inconsistent"
    assert belief_action_consistency("no", "RIGHT", "hit_wall_after_right") == "potentially_consistent"
    assert belief_action_consistency("yes", "LEFT", "wall_right") == "not_applicable"
    coord_summary = summarize_coordinate_samples(
        [{"row": 1, "col": 2}, {"row": 1, "col": 2}, None],
        {"row": 1, "col": 2},
    )
    assert coord_summary["valid_parse_rate"] == pytest.approx(2 / 3)
    assert coord_summary["unique_coordinate_count"] == 1


def test_probability_extraction_skips_control_tokens():
    logprobs = SimpleNamespace(
        content=[
            SimpleNamespace(
                token="<|channel|>",
                logprob=0.0,
                top_logprobs=[SimpleNamespace(token="<|channel|>", logprob=0.0)],
            ),
            SimpleNamespace(
                token="B",
                logprob=-0.1,
                top_logprobs=[
                    SimpleNamespace(token="B", logprob=-0.1),
                    SimpleNamespace(token="A", logprob=-2.0),
                    SimpleNamespace(token="C", logprob=-3.0),
                ],
            ),
        ]
    )
    probs = _probabilities_from_choice_logprobs(logprobs)
    assert probs["no"] > probs["yes"]
    assert probs["no"] > probs["unknown"]


def test_raw_probability_extraction_preserves_label_space():
    logprobs = SimpleNamespace(
        content=[
            SimpleNamespace(
                token="A",
                logprob=-0.1,
                top_logprobs=[
                    SimpleNamespace(token="A", logprob=-0.1),
                    SimpleNamespace(token="yes", logprob=-0.3),
                    SimpleNamespace(token="B", logprob=-1.5),
                ],
            )
        ]
    )
    raw_probs = extract_raw_answer_candidates_from_logprobs(logprobs)
    assert raw_probs["A"] > 0.0
    assert raw_probs["yes"] > 0.0
    semantic_probs = map_raw_candidates_to_semantic_probs(raw_probs)
    assert semantic_probs["yes"] > semantic_probs["no"]


def test_probability_extraction_anchors_to_surfaced_answer():
    logprobs = SimpleNamespace(
        content=[
            SimpleNamespace(
                token="A",
                logprob=-0.1,
                top_logprobs=[
                    SimpleNamespace(token="A", logprob=-0.1),
                    SimpleNamespace(token="B", logprob=-2.0),
                ],
            ),
            SimpleNamespace(
                token="B",
                logprob=-0.2,
                top_logprobs=[
                    SimpleNamespace(token="B", logprob=-0.2),
                    SimpleNamespace(token="A", logprob=-3.0),
                ],
            ),
        ]
    )
    raw_probs = extract_raw_answer_candidates_from_logprobs(logprobs, surfaced_answer="no")
    assert raw_probs["B"] > raw_probs["A"]
    token_logprobs = _extract_answer_token_logprobs(logprobs, surfaced_answer="no")
    assert token_logprobs is not None
    assert token_logprobs[0]["token"] == "B"


def test_probability_extraction_falls_back_to_last_answer_like_token():
    logprobs = SimpleNamespace(
        content=[
            SimpleNamespace(
                token="A",
                logprob=-0.1,
                top_logprobs=[SimpleNamespace(token="A", logprob=-0.1)],
            ),
            SimpleNamespace(
                token="B",
                logprob=-0.2,
                top_logprobs=[SimpleNamespace(token="B", logprob=-0.2)],
            ),
        ]
    )
    raw_probs = extract_raw_answer_candidates_from_logprobs(logprobs, surfaced_answer="invalid")
    assert raw_probs["B"] > 0.0
    assert raw_probs["A"] == 0.0


def test_prompt_preset_selection_changes_preamble_not_question_label():
    question = next(q for q in BEHAVIORAL_PROBE_QUESTIONS if q.question_id == "wall_right")
    prompt_default = _belief_prompt(
        grid_text=SMOKE_TEST_STATES[0].grid_text,
        carrying_key=False,
        question=question,
        prompt_preset="default_observable_state",
    )
    prompt_cardinal = _belief_prompt(
        grid_text=SMOKE_TEST_STATES[0].grid_text,
        carrying_key=False,
        question=question,
        prompt_preset="cardinal_action_explicit",
    )
    assert "Answer with exactly one of: yes, no, unknown." not in prompt_default
    assert "Answer with exactly one of: yes, no, unknown." not in prompt_cardinal
    assert "Return exactly one label and nothing else:" in prompt_default
    assert "Return exactly one label and nothing else:" in prompt_cardinal
    assert prompt_default != prompt_cardinal


def test_summarize_probe_rows_for_label3_and_coord():
    rows = [
        {
            "question_id": "wall_right",
            "target_variable": "wall_right",
            "question_family": "wall_directional",
            "answer_space": "label3",
            "greedy_answer": "yes",
            "greedy_modal_answer": "yes",
            "greedy_valid_parse_rate_for_row": 1.0,
            "greedy_repeated_agreement_rate_for_row": 1.0,
            "greedy_entropy": 0.0,
            "logprob_answer_t0": "yes",
            "logprob_answer_t07": "yes",
            "logprob_answer_t1": "no",
            "mc_answer": "yes",
            "entropy_t0": 0.0,
            "entropy_t07": 0.0,
            "entropy_t1": 1.0,
            "mc_entropy": 0.0,
            "valid_parse_rate_t0_for_row": 1.0,
            "valid_parse_rate_t07_for_row": 1.0,
            "valid_parse_rate_t1_for_row": 1.0,
            "mc_valid_parse_rate_for_row": 1.0,
            "ground_truth_label": "yes",
            "belief_action_consistency": "inconsistent",
            "mc_belief_action_consistency": "inconsistent",
        },
        {
            "question_id": "agent_location",
            "target_variable": "agent_location",
            "question_family": "coordinates",
            "answer_space": "coord_json",
            "greedy_coordinate_key": "1,2",
            "greedy_exact_match": 1.0,
            "greedy_manhattan_distance": 0,
            "mc_modal_exact_match": 1.0,
            "mc_modal_manhattan_distance": 0,
            "mc_mean_manhattan_distance": 0.4,
            "mc_unique_coordinate_count": 2,
            "mc_valid_parse_rate_for_row": 1.0,
        },
    ]
    summary = summarize_probe_rows(rows)
    assert len(summary) == 2
    label_row = next(row for row in summary if row["answer_space"] == "label3")
    coord_row = next(row for row in summary if row["answer_space"] == "coord_json")
    assert label_row["logprob_t0_accuracy"] == 1.0
    assert label_row["greedy_repeated_modal_accuracy"] == 1.0
    assert coord_row["greedy_exact_match_accuracy"] == 1.0


def test_run_probe_rows_uses_t0_baseline_and_writes_probability_diagnostics():
    action_client = FakeBehavioralProbeClient('{"action": "RIGHT"}', action_text="RIGHT")
    logprob_clients = {
        "t0": FakeBehavioralProbeClient("A", logprob_primary="A", temperature=0.0),
        "t07": FakeBehavioralProbeClient("B", logprob_primary="B", temperature=0.7),
        "t1": FakeBehavioralProbeClient("A", logprob_primary="A", temperature=1.0),
    }
    mc_client = FakeBehavioralProbeClient("no", temperature=0.7)
    questions = get_behavioral_probe_questions(question_family="core", answer_space="label3")
    rows, _, config = _run_probe_rows(
        model_name="together_ai/openai/gpt-oss-20b",
        examples=_single_instance(),
        questions=questions,
        mc_sample_repeats=3,
        mc_temperature=0.7,
        prompt_preset="default_observable_state",
        logprob_temperatures=(0.0, 0.7, 1.0),
        action_client=action_client,
        logprob_clients=logprob_clients,
        mc_client=mc_client,
        verbose=False,
        greedy_repeats=3,
    )
    assert len(rows) == len(questions)
    assert rows[0]["greedy_answer"] == "yes"
    assert rows[0]["greedy_modal_answer"] == "yes"
    assert rows[0]["logprob_answer_t0"] == "yes"
    assert rows[0]["logprob_answer_t07"] == "no"
    assert json.loads(rows[0]["greedy_answers_json"]) == ["yes", "yes", "yes"]
    assert config["logprob_temperatures"] == [0.0, 0.7, 1.0]
    assert config["greedy_repeats"] == 3
    diagnostics = _build_probability_diagnostics_rows(rows)
    assert diagnostics
    prompts_seen = {call[0] for client in logprob_clients.values() for call in client.ask_text_with_logprobs_calls}
    prompts_seen.update(call[0] for call in mc_client.ask_text_calls)
    assert len(prompts_seen) == len(questions)


def test_run_probe_rows_accumulates_usage_and_writes_usage_summary(tmp_path: Path):
    action_client = FakeBehavioralProbeClientWithUsage('{"action": "RIGHT"}', action_text="RIGHT")
    logprob_clients = {
        "t0": FakeBehavioralProbeClientWithUsage("A", logprob_primary="A", temperature=0.0),
        "t07": FakeBehavioralProbeClientWithUsage("B", logprob_primary="B", temperature=0.7),
        "t1": FakeBehavioralProbeClientWithUsage("A", logprob_primary="A", temperature=1.0),
    }
    mc_client = FakeBehavioralProbeClientWithUsage("no", temperature=0.7)
    questions = get_behavioral_probe_questions(question_family="core", answer_space="label3")
    rows, raw_rows, config = _run_probe_rows(
        model_name="together_ai/openai/gpt-oss-20b",
        examples=_single_instance(),
        questions=questions,
        mc_sample_repeats=2,
        mc_temperature=0.7,
        prompt_preset="default_observable_state",
        logprob_temperatures=(0.0, 0.7, 1.0),
        action_client=action_client,
        logprob_clients=logprob_clients,
        mc_client=mc_client,
        verbose=False,
        checkpoint_dir=tmp_path / "checkpoints",
        greedy_repeats=3,
    )
    usage_summary = config["usage_summary"]
    assert usage_summary["total"]["requests"] == len(questions) * (3 + 2 + 2)
    assert "observed_action" not in usage_summary["by_phase"]
    assert usage_summary["by_phase"]["logprob_t0"]["requests"] == len(questions)
    assert usage_summary["by_phase"]["greedy_repeated_t0"]["requests"] == len(questions) * 2
    assert usage_summary["by_phase"]["mc"]["requests"] == len(questions) * 2

    _write_behavioral_outputs(
        out_dir=tmp_path,
        rows=rows,
        raw_rows=raw_rows,
        config=config,
        dataset_validation=None,
        smoke_examples=None,
        min_valid_parse_rate=0.0,
    )
    usage_path = tmp_path / "usage_summary.json"
    assert usage_path.exists()
    usage_payload = json.loads(usage_path.read_text())
    assert usage_payload["total"]["requests"] == usage_summary["total"]["requests"]
    assert (tmp_path / "checkpoints" / "checkpoint_rows.jsonl").exists()
    assert (tmp_path / "checkpoints" / "checkpoint_raw_rows.jsonl").exists()
    assert (tmp_path / "checkpoints" / "checkpoint_status.json").exists()


def test_write_csv_supports_mixed_row_schemas(tmp_path: Path):
    out_path = tmp_path / "mixed.csv"
    _write_csv(
        out_path,
        [
            {"a": 1, "b": 2},
            {"a": 3, "c": 4},
        ],
    )
    with open(out_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["a"] == "1"
    assert rows[0]["b"] == "2"
    assert rows[0]["c"] == ""
    assert rows[1]["a"] == "3"
    assert rows[1]["b"] == ""
    assert rows[1]["c"] == "4"


def test_merge_behavioral_probe_outputs_repairs_rows_and_merges(tmp_path: Path):
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    source_a.mkdir()
    source_b.mkdir()
    payload_a = {
        "rows": [
            {
                "parsed_row": {
                    "question_id": "wall_right",
                    "target_variable": "wall_right",
                    "question_family": "wall_directional",
                    "answer_space": "label3",
                    "greedy_answer": "yes",
                    "logprob_answer_t0": "yes",
                    "logprob_answer_t07": "yes",
                    "logprob_answer_t1": "yes",
                    "mc_answer": "yes",
                    "entropy_t0": 0.0,
                    "entropy_t07": 0.0,
                    "entropy_t1": 0.0,
                    "mc_entropy": 0.0,
                    "valid_parse_rate_t0_for_row": 1.0,
                    "valid_parse_rate_t07_for_row": 1.0,
                    "valid_parse_rate_t1_for_row": 1.0,
                    "mc_valid_parse_rate_for_row": 1.0,
                    "ground_truth_label": "yes",
                    "belief_action_consistency": "not_applicable",
                    "mc_belief_action_consistency": "not_applicable",
                }
            }
        ],
        "usage_summary": {"source": "a"},
    }
    payload_b = {
        "rows": [
            {
                "parsed_row": {
                    "question_id": "agent_location",
                    "target_variable": "agent_location",
                    "question_family": "coordinates",
                    "answer_space": "coord_json",
                    "greedy_coordinate_key": "1,2",
                    "greedy_exact_match": 1.0,
                    "greedy_manhattan_distance": 0.0,
                    "mc_modal_exact_match": 1.0,
                    "mc_modal_manhattan_distance": 0.0,
                    "mc_mean_manhattan_distance": 0.0,
                    "mc_unique_coordinate_count": 1.0,
                    "mc_valid_parse_rate_for_row": 1.0,
                }
            }
        ],
        "usage_summary": {"source": "b"},
    }
    (source_a / "behavioral_probe_raw.json").write_text(json.dumps(payload_a))
    (source_b / "behavioral_probe_raw.json").write_text(json.dumps(payload_b))
    out_dir = tmp_path / "merged"
    merge_behavioral_probe_outputs(
        input_dirs=(str(source_a), str(source_b)),
        output_dir=str(out_dir),
        repair_source_rows_csv=True,
    )
    assert (source_a / "behavioral_probe_rows.csv").exists()
    assert (source_b / "behavioral_probe_rows.csv").exists()
    assert (out_dir / "behavioral_probe_rows.csv").exists()
    assert (out_dir / "behavioral_probe_summary.csv").exists()
    assert (out_dir / "behavioral_probe_raw.json").exists()
    usage = json.loads((out_dir / "usage_summary.json").read_text())
    assert str(source_a) in usage["per_source"]
    assert str(source_b) in usage["per_source"]


def test_mine_behavioral_probe_instances_from_trace_viewer_json(tmp_path: Path):
    trajectory_path = tmp_path / "traj.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "grid_state": SMOKE_TEST_STATES[0].grid_text.splitlines(),
                        "carrying_key": False,
                        "agent_action": "RIGHT",
                    },
                    {
                        "grid_state": SMOKE_TEST_STATES[0].grid_text.splitlines(),
                        "carrying_key": False,
                        "agent_action": "DOWN",
                    },
                ]
            }
        )
    )
    rows = mine_behavioral_probe_instances(str(tmp_path), slice_type="all_steps")
    assert len(rows) == 2
    assert rows[0]["trajectory_id"] == "traj"
    assert "probe_truths" in rows[0]


def test_case_studies_write_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    trajectory_path = tmp_path / "traj.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "grid_state": SMOKE_TEST_STATES[1].grid_text.splitlines(),
                        "carrying_key": False,
                        "agent_action": "RIGHT",
                    }
                ]
            }
        )
    )

    called = {"count": 0}

    def fake_run_behavioral_probe_on_instances(**kwargs):
        called["count"] += 1
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "behavioral_probe_rows.csv").write_text("ok\n")
        (out_dir / "behavioral_probe_summary.csv").write_text("ok\n")

    monkeypatch.setattr(
        "reveng.experiments.behavioral_probe_case_studies.run_behavioral_probe_on_instances",
        fake_run_behavioral_probe_on_instances,
    )
    run_behavioral_probe_case_studies(
        trajectory_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        verbose=False,
    )
    assert called["count"] == 1
    assert (tmp_path / "out" / "case_study_rows.csv").exists()
    assert (tmp_path / "out" / "case_study.md").exists()


def test_door_semantics_ablation_writes_focused_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from reveng.experiments import behavioral_probe_runner as runner_mod

    def fake_run_probe_rows(**kwargs):
        question = kwargs["questions"][0]
        rows = []
        raw_rows = []
        for example in kwargs["examples"]:
            mc_answer = "yes" if question.variant_id == "observed" else "no"
            row = {
                "example_id": example["example_id"],
                "question_id": question.question_id,
                "target_variable": question.target_variable,
                "question_family": question.family,
                "answer_space": "label3",
                "variant_id": question.variant_id or "",
                "prompt_preset": kwargs["prompt_preset"],
                "grid_text": example["grid_text"],
                "carrying_key": example["carrying_key"],
                "observed_action": "RIGHT",
                "greedy_answer": "yes",
                "ground_truth_label": "yes",
                "belief_action_consistency": "not_applicable",
                "mc_belief_action_consistency": "not_applicable",
                "mc_answer": mc_answer,
                "mc_answers_json": json.dumps([mc_answer] * kwargs["mc_sample_repeats"]),
                "mc_answer_probs_json": json.dumps({"yes": 1.0 if mc_answer == "yes" else 0.0, "no": 0.0 if mc_answer == "yes" else 1.0, "unknown": 0.0, "invalid": 0.0}),
                "mc_entropy": 0.0,
                "mc_valid_parse_rate_for_row": 1.0,
                "sampled_answer": "yes",
                "answer_probs_json": json.dumps({"yes": 1.0, "no": 0.0, "unknown": 0.0, "invalid": 0.0}),
                "entropy": 0.0,
            }
            for suffix in ("t0", "t07", "t1"):
                row[f"logprob_{suffix}_raw_output"] = "A"
                row[f"raw_answer_probs_{suffix}_json"] = json.dumps({"A": 1.0, "B": 0.0, "C": 0.0, "yes": 0.0, "no": 0.0, "unknown": 0.0})
                row[f"answer_probs_{suffix}_json"] = json.dumps({"yes": 1.0, "no": 0.0, "unknown": 0.0, "invalid": 0.0})
                row[f"logprob_answer_{suffix}"] = "yes"
                row[f"entropy_{suffix}"] = 0.0
                row[f"valid_parse_rate_{suffix}_for_row"] = 1.0
            rows.append(row)
            raw_rows.append({"parsed_row": row})
        config = {"model_name": kwargs["model_name"]}
        return rows, raw_rows, config

    monkeypatch.setattr(runner_mod, "_run_probe_rows", fake_run_probe_rows)
    output_dir = tmp_path / "door_ablation"
    run_behavioral_probe_door_semantics_ablation(
        model_name="together_ai/openai/gpt-oss-20b",
        output_dir=str(output_dir),
        verbose=False,
    )
    assert (output_dir / "focused_rows.csv").exists()
    assert (output_dir / "decision.json").exists()
    decision = json.loads((output_dir / "decision.json").read_text())
    assert decision["winner_variant"] == "observed"
