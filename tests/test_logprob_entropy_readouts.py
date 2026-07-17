from types import SimpleNamespace

import pytest

from reveng.experiments.behavioral_probe_runner import (
    _entry_candidate_diagnostics,
    _select_answer_logprob_entry,
)
from reveng.experiments.step_reasoning_drift import action_entropy_from_probabilities


def _normalize(token: str) -> str | None:
    return token.strip().upper() if token.strip().upper() in {"UP", "DOWN"} else None


def test_candidate_diagnostics_preserve_mass_and_normalize() -> None:
    entry = SimpleNamespace(
        token="UP",
        logprob=-0.6931471805599453,
        top_logprobs=[
            SimpleNamespace(token="UP", logprob=-0.6931471805599453),
            SimpleNamespace(token="DOWN", logprob=-0.6931471805599453),
        ],
    )
    result = _entry_candidate_diagnostics(
        entry,
        labels=("UP", "DOWN"),
        normalize_token=_normalize,
    )
    assert result["candidate_probability_mass"] == pytest.approx(1.0)
    assert result["probabilities"] == pytest.approx({"UP": 0.5, "DOWN": 0.5})
    assert result["missing_candidates"] == []


def test_generated_candidate_repeated_in_top_logprobs_is_not_double_counted() -> None:
    entry = SimpleNamespace(
        token="UP",
        logprob=-0.6931471805599453,
        top_logprobs=[
            SimpleNamespace(token="UP", logprob=-0.6931471805599453),
            SimpleNamespace(token="DOWN", logprob=-0.6931471805599453),
        ],
    )
    result = _entry_candidate_diagnostics(
        entry,
        labels=("UP", "DOWN"),
        normalize_token=_normalize,
    )
    assert result["probabilities"] == pytest.approx({"UP": 0.5, "DOWN": 0.5})


def test_action_entropy_is_in_bits() -> None:
    assert action_entropy_from_probabilities(
        {"UP": 0.25, "DOWN": 0.25, "LEFT": 0.25, "RIGHT": 0.25}
    ) == pytest.approx(2.0)
    assert action_entropy_from_probabilities({"UP": 1.0}) == pytest.approx(0.0)


def test_elaborated_label_selects_requested_label_token() -> None:
    label_entry = SimpleNamespace(token="B", logprob=0.0, top_logprobs=[])
    semantic_entry = SimpleNamespace(token=" no", logprob=0.0, top_logprobs=[])
    logprobs = SimpleNamespace(content=[label_entry, semantic_entry])
    assert _select_answer_logprob_entry(logprobs, "invalid") is label_entry
