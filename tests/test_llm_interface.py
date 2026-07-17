from types import SimpleNamespace

from tenacity.wait import wait_none

from reveng.llm_interface import BaseLLMInterface


def _fake_response(content: str, finish_reason: str = "stop") -> SimpleNamespace:
    message = SimpleNamespace(content=content, role="assistant")
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def test_make_completion_request_retries_empty_content(monkeypatch) -> None:
    calls: list[str] = []

    def fake_completion(**kwargs):
        calls.append("call")
        if len(calls) == 1:
            return _fake_response("")
        return _fake_response("UP")

    monkeypatch.setattr("reveng.llm_interface.completion", fake_completion)
    monkeypatch.setattr("reveng.llm_interface.completion_cost", lambda completion_response: 0.0)
    monkeypatch.setattr("reveng.llm_interface.wait_random_exponential", lambda **kwargs: wait_none())

    llm = BaseLLMInterface(model_name="test-model", temperature=0.0)
    response, cost, _ = llm._make_completion_request("test prompt")

    assert response == "UP"
    assert cost == 0.0
    assert len(calls) == 2
    retry_info = llm._get_last_retry_info()
    assert retry_info["had_retry"] is True
    assert retry_info["retry_count"] == 1


def test_extract_response_content_falls_back_to_reasoning_content() -> None:
    message = SimpleNamespace(
        content="",
        reasoning_content='Reasoning text. assistantfinal{"action":"LEFT"}',
        role="assistant",
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")]
    )

    llm = BaseLLMInterface(model_name="test-model", temperature=0.0)

    assert llm._extract_response_content(response).endswith('{"action":"LEFT"}')
