import json
from pathlib import Path

import torch

from reveng.experiments import activation_oracle_batch as aob
from reveng.experiments.activation_oracle_compare import compare_activation_oracle_results


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    padding_side = "left"

    def _tokenize_text(self, text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            if text.startswith(aob.SPECIAL_TOKEN, i):
                ids.append(1)
                i += len(aob.SPECIAL_TOKEN)
            else:
                ids.append(ord(text[i]) + 100)
                i += 1
        return ids

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_kwargs):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        if tokenize:
            return self._tokenize_text(text)
        return text

    def __call__(self, texts, return_tensors="pt", add_special_tokens=False, padding=False):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self._tokenize_text(text) for text in texts]
        max_len = max(len(row) for row in encoded)
        padded = []
        attn = []
        for row in encoded:
            pad_len = max_len - len(row)
            padded.append([self.pad_token_id] * pad_len + row)
            attn.append([0] * pad_len + [1] * len(row))
        return FakeBatch(
            {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            }
        )

    def encode(self, text, add_special_tokens=False):
        return self._tokenize_text(text)

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        chars = []
        for token_id in token_ids:
            if token_id == 0:
                continue
            if token_id == 1:
                chars.append(aob.SPECIAL_TOKEN)
            else:
                chars.append(chr(token_id - 100))
        return "".join(chars)

    def batch_decode(self, sequences, skip_special_tokens=False):
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]


class FakeModel:
    class Config:
        _name_or_path = aob.DEFAULT_MODEL_NAME

    def __init__(self):
        self.config = self.Config()


def _write_jsonl(path: Path, rows: list[dict]):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_load_activation_oracle_batch_rows_parses_list_and_string_fields(tmp_path: Path):
    input_path = tmp_path / "rows.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "row_id": "native",
                "target_messages_json": [{"role": "user", "content": "alpha"}],
                "oracle_prompt": "question",
            },
            {
                "row_id": "stringified",
                "target_messages_json": json.dumps([{"role": "user", "content": "beta"}]),
                "oracle_prompt": "question",
                "oracle_input_types": "[\"segment\", \"full_seq\"]",
                "segment_start": 0,
            },
        ],
    )

    rows, failures = aob.load_activation_oracle_batch_rows(input_path)
    assert not failures
    assert [row.row_id for row in rows] == ["native", "stringified"]
    assert rows[1].oracle_input_types == ["segment", "full_seq"]


def test_validate_row_token_selection_requires_explicit_indices():
    row = aob.ActivationOracleBatchRow(
        row_id="r1",
        target_messages_json=[{"role": "user", "content": "hello"}],
        oracle_prompt="q",
        oracle_input_types=["segment"],
    )
    try:
        aob.validate_row_token_selection(row=row, token_count=10, oracle_input_types=["segment"])
    except ValueError as exc:
        assert "segment_start" in str(exc)
    else:
        raise AssertionError("Expected segment_start validation failure")


def test_format_token_selection_preview_marks_segment_and_token_ranges():
    tokenizer = FakeTokenizer()
    preview = aob.format_token_selection_preview(
        tokenizer=tokenizer,
        input_text="abcd",
        segment_start=1,
        segment_end=3,
        token_start=2,
        token_end=4,
    )
    assert "[001]  S" in preview
    assert "[002] ST" in preview
    assert "[003]  T" in preview


def test_run_activation_oracle_batch_writes_results_and_failures(tmp_path: Path, monkeypatch):
    input_path = tmp_path / "batch.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(
        input_path,
        [
            {
                "row_id": "segment_ok",
                "target_messages_json": [{"role": "user", "content": "Prompt A"}],
                "oracle_prompt": "What is encoded?",
                "behavioral_label": "yes",
                "segment_start": 0,
                "segment_end": 3,
                "oracle_input_types": ["segment", "full_seq"],
            },
            {
                "row_id": "tokens_ok",
                "target_messages_json": [{"role": "user", "content": "Prompt B"}],
                "oracle_prompt": "Token beliefs?",
                "token_start": 1,
                "token_end": 4,
                "oracle_input_types": ["tokens"],
            },
            {
                "row_id": "missing_prompt",
                "target_messages_json": [{"role": "user", "content": "Prompt C"}],
            },
            {
                "row_id": "missing_segment_start",
                "target_messages_json": [{"role": "user", "content": "Prompt D"}],
                "oracle_prompt": "Need segment",
                "oracle_input_types": ["segment"],
            },
        ],
    )

    fake_tokenizer = FakeTokenizer()

    def fake_loader(**_kwargs):
        return FakeModel(), fake_tokenizer, torch.device("cpu")

    def fake_load_lora_adapter(_model, _lora_path):
        return "fake_adapter"

    def fake_run_oracle(**kwargs):
        target_ids = fake_tokenizer._tokenize_text(kwargs["target_prompt"])
        token_responses = [None] * len(target_ids)
        if "tokens" in kwargs["oracle_input_types"]:
            start = kwargs["token_start_idx"]
            end = len(target_ids) if kwargs["token_end_idx"] is None else kwargs["token_end_idx"]
            for idx in range(start, end):
                token_responses[idx] = f"resp-{idx}"
        return aob.OracleResults(
            oracle_lora_path=kwargs["oracle_lora_path"],
            target_lora_path=kwargs["target_lora_path"],
            target_prompt=kwargs["target_prompt"],
            act_key="lora",
            oracle_prompt=kwargs["oracle_prompt"],
            ground_truth=kwargs.get("ground_truth", ""),
            num_tokens=len(target_ids),
            token_responses=token_responses,
            full_sequence_responses=["full-seq"],
            segment_responses=["segment"] if "segment" in kwargs["oracle_input_types"] else [],
            target_input_ids=target_ids,
            act_layer=18,
        )

    monkeypatch.setattr(aob, "_load_qwen_model_and_tokenizer", fake_loader)
    monkeypatch.setattr(aob, "load_lora_adapter", fake_load_lora_adapter)
    monkeypatch.setattr(aob, "run_oracle", fake_run_oracle)

    manifest = aob.run_activation_oracle_batch(
        input_path=str(input_path),
        output_dir=str(output_dir),
    )

    assert manifest["succeeded_rows"] == 2
    assert manifest["failed_rows"] == 2

    results = [json.loads(line) for line in (output_dir / "results.jsonl").read_text().splitlines() if line.strip()]
    failures = [json.loads(line) for line in (output_dir / "failures.jsonl").read_text().splitlines() if line.strip()]
    token_rows = [json.loads(line) for line in (output_dir / "token_responses.jsonl").read_text().splitlines() if line.strip()]

    assert [row["row_id"] for row in results] == ["segment_ok", "tokens_ok"]
    assert results[0]["resolved_oracle_input_types"] == ["segment", "full_seq"]
    assert results[1]["resolved_act_layer"] == 18
    assert results[0]["behavioral_label"] == "yes"
    assert {row["stage"] for row in failures} == {"row_validation", "resolved_validation"}
    assert all(row["row_id"] == "tokens_ok" for row in token_rows)
    assert len(token_rows) == 3


def test_committed_local_inference_batch_parses_cleanly():
    rows, failures = aob.load_activation_oracle_batch_rows(
        "data/activation_oracle/doorkey_qwen_local_inference_batch_v1.jsonl"
    )
    assert not failures
    assert [row.row_id for row in rows] == [
        "q1_right_cell_wall_after_agent_row",
        "q2_blocked_moves_local_neighborhood",
        "q3_closed_door_presence_door_row",
        "q4_carrying_key_status",
        "q5_forced_action_commitment_prompt_tail",
    ]


def test_committed_revealed_cot_batch_parses_cleanly():
    rows, failures = aob.load_activation_oracle_batch_rows(
        "data/activation_oracle/doorkey_qwen_revealed_cot_batch_v1.jsonl"
    )
    assert not failures
    assert [row.row_id for row in rows] == [
        "cot_q1_goal_location_early_analysis",
        "cot_q2_down_on_shortest_path_mid_analysis",
        "cot_q3_right_also_shortest_path_route_detail",
        "cot_q4_multiple_equally_good_first_moves",
        "cot_q5_forced_action_commitment_late_analysis",
    ]


def test_compare_activation_oracle_results_writes_csv_and_summary(tmp_path: Path):
    results_path = tmp_path / "results.jsonl"
    _write_jsonl(
        results_path,
        [
            {
                "row_id": "r1",
                "oracle_prompt": "Q1",
                "behavioral_label": "no",
                "ground_truth": "no",
                "segment_responses": ["No"],
                "full_sequence_responses": ["yes"],
            },
            {
                "row_id": "r2",
                "oracle_prompt": "Q2",
                "behavioral_label": '{"UP":"free","DOWN":"blocked"}',
                "ground_truth": '{"UP":"free","DOWN":"blocked"}',
                "segment_responses": ['{"DOWN":"blocked","UP":"free"}'],
                "full_sequence_responses": [""],
            },
        ],
    )

    summary = compare_activation_oracle_results(str(results_path))

    csv_path = results_path.with_name("ao_behavioral_comparison.csv")
    summary_path = results_path.with_name("ao_behavioral_comparison_summary.json")
    csv_text = csv_path.read_text()
    summary_payload = json.loads(summary_path.read_text())

    assert "segment_matches_behavioral_label" in csv_text
    assert summary["rows"] == 2
    assert summary["segment_matches_behavioral"] == 2
    assert summary_payload["full_sequence_matches_ground_truth"] == 0
