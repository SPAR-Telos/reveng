from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch

from reveng.experiments.maze_local_replay import (
    capture_replay_activations,
    prepare_replay_record,
    reconstruct_completion_ids,
    validate_static_record,
)


class FakeTokenizer:
    def __init__(self, tokens: list[str]):
        self.tokens = sorted(tokens, key=len, reverse=True)
        self.to_id = {token: index + 1 for index, token in enumerate(tokens)}
        self.to_token = {value: key for key, value in self.to_id.items()}

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **_kwargs,
    ):
        assert not add_special_tokens
        ids = []
        offsets = []
        cursor = 0
        while cursor < len(text):
            token = next(
                (value for value in self.tokens if text.startswith(value, cursor)),
                None,
            )
            if token is None:
                raise ValueError(f"No fake token at {text[cursor:]!r}")
            ids.append(self.to_id[token])
            offsets.append((cursor, cursor + len(token)))
            cursor += len(token)
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def decode(self, token_ids, **_kwargs):
        return "".join(self.to_token[int(value)] for value in token_ids)


class FakeProcessor:
    def __init__(self, tokenizer: FakeTokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, messages, **_kwargs):
        text = "CHAT:" + messages[0]["content"]
        return torch.tensor(
            [self.tokenizer(text, add_special_tokens=False)["input_ids"]]
        )


class FakeLayer(torch.nn.Module):
    def forward(self, hidden):
        return (hidden + 1,)


class FakeBody(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeLayer(), FakeLayer()])

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return hidden


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 4)
        self.model = FakeBody()

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, input_ids, past_key_values, use_cache, return_dict):
        assert use_cache and return_dict
        hidden = self.model(self.embedding(input_ids))
        return SimpleNamespace(past_key_values=())


def _fixture():
    prompt = "grid"
    pieces = ["reason ", '{"action":"UP"}']
    tokenizer = FakeTokenizer(["CHAT:", prompt, *pieces])
    row = {
        "trajectory_id": "trajectory-1",
        "step_index": 0,
        "model": "Fake",
        "reasoning_setting": "native",
        "prompt_text": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_tokens": 2,
        "output_tokens": 2,
        "generated_token_ids": [],
        "generated_tokens": pieces,
        "generated_token_bytes": [list(value.encode()) for value in pieces],
        "token_sequence_complete": True,
        "api_success": True,
    }
    config = {"name": "Fake", "reasoning_trace_format": "before_action_json"}
    return row, config, tokenizer


def test_reconstructs_ids_and_checks_provider_boundaries():
    row, _config, tokenizer = _fixture()
    ids, source, boundary = reconstruct_completion_ids(row, tokenizer)
    assert ids == [tokenizer.to_id[value] for value in row["generated_tokens"]]
    assert source == "reconstructed_exact_checkpoint_tokenizer"
    assert boundary == "provider_boundaries_match_local_offsets"


def test_reconstruction_rejects_a_different_local_segmentation():
    row, _config, _tokenizer = _fixture()
    incompatible = FakeTokenizer(["CHAT:", "grid", "reason ", "{", '"action":"UP"}'])
    with pytest.raises(ValueError, match="token count"):
        reconstruct_completion_ids(row, incompatible)


def test_prepare_and_chunked_teacher_forced_capture():
    row, config, tokenizer = _fixture()
    prepared = prepare_replay_record(
        row,
        processor=FakeProcessor(tokenizer),
        tokenizer=tokenizer,
        model_config=config,
    )
    assert prepared.prompt_ids == [tokenizer.to_id["CHAT:"], tokenizer.to_id["grid"]]
    assert prepared.reasoning_indices == [0]

    tensors, metrics = capture_replay_activations(
        FakeModel(),
        prepared,
        layers=[0, 1],
        forward_chunk_size=1,
        capture_scope="completion",
    )
    assert tensors["generated_token_ids"].tolist() == prepared.completion_ids
    assert tensors["captured_completion_positions"].tolist() == [0, 1]
    assert tensors["layer_0"].shape == (2, 4)
    assert tensors["layer_1"].shape == (2, 4)
    assert metrics["input_tokens"] == 4
    assert metrics["captured_tokens"] == 2


def test_static_validation_rejects_prompt_tampering():
    row, _config, _tokenizer = _fixture()
    row["prompt_text"] = "changed"
    with pytest.raises(ValueError, match="prompt_sha256"):
        validate_static_record(row, model_name="Fake")
