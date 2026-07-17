import logging
import os
import traceback
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Optional, Tuple

import litellm
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, Template
from litellm import completion, completion_cost
from pydantic import BaseModel
from tenacity import Retrying, RetryCallState, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)

ENV_FILE = Path(__file__).parent.parent.parent / ".env"

litellm.enable_json_schema_validation = True


def _set_provider_key_aliases() -> None:
    """Map common provider API-key aliases to keys expected by LiteLLM."""
    # LiteLLM's Together provider uses TOGETHERAI_API_KEY. Accept TOGETHER_API_KEY too.
    together_alias = os.getenv("TOGETHER_API_KEY")
    if together_alias and not os.getenv("TOGETHERAI_API_KEY"):
        os.environ["TOGETHERAI_API_KEY"] = together_alias


def _configure_hf_transfer() -> None:
    """Disable hf_transfer optimization when requested but unavailable.

    Some environments set HF_HUB_ENABLE_HF_TRANSFER=1 globally. If the optional
    `hf_transfer` package is not installed, huggingface_hub may fail while
    downloading model/tokenizer assets. We downgrade to the default transport.
    """
    enabled = os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return
    if find_spec("hf_transfer") is not None:
        return
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    logger.warning(
        "HF_HUB_ENABLE_HF_TRANSFER is enabled but hf_transfer is not installed; "
        "falling back to default Hugging Face download transport."
    )


class BaseLLMInterface:
    """Base class for LLM interactions with common functionality."""

    def __init__(
        self,
        model_name: str,
        template_path: Optional[Path | str] = None,
        temperature: float = 1.0,
    ) -> None:
        """
        Args:
            model_name: Identifier understood by ``litellm`` (e.g. ``"gpt-4"``).
            template_path: Optional override for the Jinja prompt template.
            temperature: Forwarded to the model.
        """
        self.model_name = model_name
        self.temperature = temperature

        resolved_template = Path(template_path) if template_path is not None else None
        self._template = (
            self._load_template(resolved_template) if resolved_template else None
        )
        self._last_retry_info: dict[str, Any] = self._new_retry_info()

        # Load environment variables
        if ENV_FILE.exists():
            load_dotenv(ENV_FILE)
            logger.info(f"Loaded environment variables from {ENV_FILE}")
        else:
            logger.error(
                f"No .env file found at {ENV_FILE}! Please create a .env file in the root of the project."
            )
        _set_provider_key_aliases()
        _configure_hf_transfer()

    @staticmethod
    def _load_template(template_path: Path) -> Template:
        """Load a Jinja2 template from file."""
        env = Environment(
            loader=FileSystemLoader(template_path.parent),
        )
        return env.get_template(template_path.name)

    def render_template(self, **kwargs) -> str:
        """Render the template with given context."""
        if self._template is None:
            raise ValueError("No template provided")
        return self._template.render(**kwargs)

    @staticmethod
    def _new_retry_info() -> dict[str, Any]:
        return {
            "attempts": 0,
            "had_retry": False,
            "retry_count": 0,
            "retry_sleep_seconds": 0.0,
            "failure_messages": [],
        }

    def _record_retry_before_sleep(self, retry_state: RetryCallState) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        sleep_seconds = 0.0
        if retry_state.next_action is not None:
            sleep_seconds = float(retry_state.next_action.sleep or 0.0)
        self._last_retry_info["retry_count"] += 1
        self._last_retry_info["had_retry"] = True
        self._last_retry_info["retry_sleep_seconds"] += sleep_seconds
        if exc is not None:
            self._last_retry_info["failure_messages"].append(str(exc))
        logger.warning(
            "Retrying model request for %s after attempt %s failed; sleep %.2fs; error=%s",
            self.model_name,
            retry_state.attempt_number,
            sleep_seconds,
            exc,
        )

    def _get_last_retry_info(self) -> dict[str, Any]:
        return {
            "attempts": int(self._last_retry_info.get("attempts", 0)),
            "had_retry": bool(self._last_retry_info.get("had_retry", False)),
            "retry_count": int(self._last_retry_info.get("retry_count", 0)),
            "retry_sleep_seconds": float(self._last_retry_info.get("retry_sleep_seconds", 0.0)),
            "failure_messages": list(self._last_retry_info.get("failure_messages", [])),
        }

    @staticmethod
    def _choice_get(choice: Any, key: str, default: Any = None) -> Any:
        value = getattr(choice, key, None)
        if value is None and isinstance(choice, dict):
            value = choice.get(key, default)
        return default if value is None else value

    def _extract_response_content(self, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("Model response is missing choices.")
        choice = choices[0]
        message = self._choice_get(choice, "message")
        content = self._choice_get(message, "content")
        if not content:
            content = self._choice_get(message, "reasoning_content")
        if not content:
            provider_fields = self._choice_get(message, "provider_specific_fields", {})
            if isinstance(provider_fields, dict):
                content = provider_fields.get("reasoning_content") or provider_fields.get("reasoning")
        if not content:
            logger.error(f"Empty response from model. Full choice object: {choice}")
            finish_reason = self._choice_get(choice, "finish_reason", "N/A")
            raise ValueError(
                f"Empty response from model. Finish reason: '{finish_reason}'"
            )
        return str(content)

    def _completion_with_retry(
        self,
        response_format: Optional[BaseModel] = None,
        **kwargs,
    ) -> Tuple[any, float]:
        """Make LLM completion request with retry logic.

        Returns:
            Tuple of (response_object, cost_in_usd)
        """
        self._last_retry_info = self._new_retry_info()
        retryer = Retrying(
            stop=stop_after_attempt(5),
            wait=wait_random_exponential(multiplier=1, min=5, max=120),
            reraise=True,
            before_sleep=self._record_retry_before_sleep,
        )
        response = None
        for attempt in retryer:
            with attempt:
                self._last_retry_info["attempts"] = attempt.retry_state.attempt_number
                try:
                    if "cohere" in self.model_name or "fireworks" in self.model_name:
                        max_tokens = 4096
                    else:
                        max_tokens = 10000
                    response = completion(
                        **kwargs,
                        response_format=response_format,
                        max_tokens=max_tokens,
                    )
                    self._extract_response_content(response)
                except Exception as e:
                    logger.error(f"Model request failed: {e}\n{traceback.format_exc()}")
                    raise
        assert response is not None

        # Calculate cost using litellm's completion_cost function
        try:
            cost = completion_cost(completion_response=response)
        except Exception as e:
            logger.warning(f"Failed to calculate cost: {e}")
            cost = 0.0

        return response, cost

    def _make_completion_request(
        self,
        prompt: str,
        response_format: Optional[BaseModel] = None,
        **kwargs,
    ) -> Tuple[any, float, any]:
        """Make a completion request with standard parameters.

        Returns:
            Tuple of (parsed_response, cost_in_usd)
        """
        try:
            response, cost = self._completion_with_retry(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format=response_format,
                **kwargs,
            )

            # Parse response
            content = self._extract_response_content(response)

            content, response = inject_thinking_for_qwen(
                model_name=self.model_name, content=content, response=response
            )

            if response_format:
                try:
                    parsed_response = response_format.model_validate_json(content)
                    return parsed_response, cost, response
                except Exception as exc:
                    logger.error(f"Failed to parse response: {exc}")
                    logger.debug(f"Raw response content: {content}")
                    raise ValueError(f"Invalid response format from model: {exc}")

            return content, cost, response
        except Exception as exc:
            logger.error(f"Model request failed: {exc}\n{traceback.format_exc()}")
            raise

    def _make_chat_completion_request(
        self,
        messages: list[dict],
        response_format: Optional[BaseModel] = None,
        **kwargs,
    ) -> Tuple[any, float, any]:
        """Make a completion request using provided chat messages.

        Returns:
            Tuple of (parsed_response_or_content, cost_in_usd, raw_response)
        """
        try:
            response, cost = self._completion_with_retry(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                response_format=response_format,
                **kwargs,
            )

            # Parse response
            content = self._extract_response_content(response)

            content, response = inject_thinking_for_qwen(
                model_name=self.model_name, content=content, response=response
            )

            if response_format:
                try:
                    parsed_response = response_format.model_validate_json(content)
                    return parsed_response, cost, response
                except Exception as exc:
                    logger.error(f"Failed to parse response: {exc}")
                    logger.debug(f"Raw response content: {content}")
                    raise ValueError(f"Invalid response format from model: {exc}")

            return content, cost, response
        except Exception as exc:
            logger.error(f"Model request failed: {exc}\n{traceback.format_exc()}")
            raise


def inject_thinking_for_qwen(
    model_name: str, content: str, response: dict
) -> Tuple[str, dict]:
    """Inject thinking for Qwen models. There is a bug for qwen3 30b a3b in fireworks where the response is not properly formatted."""
    if "</think>" in content and (
        "qwen3-30b-a3b".lower() in model_name.lower()
        or "Qwen3-235B-A22B-Thinking-2507".lower() in model_name.lower()
    ):
        assert "</think>" in content, "No </think> in qwen content response!"
        reasoning_content = content.split("</think>")[0]
        content = content.split("</think>")[1]
        response.choices[0].message.content = content.strip()
        response["choices"][0]["message"]["reasoning_content"] = (
            reasoning_content.strip()
        )
        return content, response
    return content, response
