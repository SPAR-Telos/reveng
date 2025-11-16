import json
import logging
import traceback
from pathlib import Path
from typing import Optional, Tuple

import litellm
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, Template
from litellm import completion, completion_cost
from litellm.exceptions import JSONSchemaValidationError
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)

ENV_FILE = Path(__file__).parent.parent.parent / ".env"

litellm.enable_json_schema_validation = True


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

        # Load environment variables
        if ENV_FILE.exists():
            load_dotenv(ENV_FILE)
            logger.info(f"Loaded environment variables from {ENV_FILE}")
        else:
            logger.error(
                f"No .env file found at {ENV_FILE}! Please create a .env file in the root of the project."
            )

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

    @retry(
        stop=stop_after_attempt(5),
        # 5-120 seconds between attempts, to help avoid rate limiting
        wait=wait_random_exponential(multiplier=1, min=5, max=120),
        reraise=True,
    )
    def _completion_with_retry(
        self,
        response_format: Optional[BaseModel] = None,
        **kwargs,
    ) -> Tuple[any, float]:
        """Make LLM completion request with retry logic.

        Returns:
            Tuple of (response_object, cost_in_usd)
        """
        try:
            # For action sequences, we don't need 10000 tokens. Use 5000 to avoid streaming requirement.
            # Fireworks requires stream=True when max_tokens > 5000, so we cap at 5000 for Fireworks
            is_fireworks = "fireworks" in self.model_name.lower()
            if is_fireworks:
                max_tokens = 5000  # Avoid streaming requirement for Fireworks
            else:
                max_tokens = 10000 if "cohere" not in self.model_name else 8192
            
            # Use non-streaming completion (streaming disabled since we cap at 5000 for Fireworks)
            response = completion(
                **kwargs,
                response_format=response_format,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"Model request failed: {e}\n{traceback.format_exc()}")
            raise

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
            try:
                response, cost = self._completion_with_retry(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    response_format=response_format,
                    **kwargs,
                )
            except JSONSchemaValidationError as e:
                # Handle case where model returns string action names instead of integers
                if response_format:
                    try:
                        # Extract JSON from error message (format: "returned an invalid response={...}")
                        import re
                        error_str = str(e)
                        # Look for JSON object in error message - handle both single action and action_sequence
                        json_match = re.search(r'response=(\{[^}]+\})', error_str)
                        if json_match:
                            parsed = json.loads(json_match.group(1))
                            
                            # Handle single action case
                            if isinstance(parsed, dict) and "action" in parsed and "action_sequence" not in parsed:
                                action = parsed["action"]
                                # Convert string action names to integers
                                action_map = {"LEFT": 0, "RIGHT": 1, "UP": 2, "DOWN": 3}
                                if isinstance(action, str) and action.upper() in action_map:
                                    parsed["action"] = action_map[action.upper()]
                                    # Create a mock response object with corrected content
                                    corrected_content = json.dumps(parsed)
                                    from types import SimpleNamespace
                                    mock_response = SimpleNamespace()
                                    mock_response.choices = [SimpleNamespace()]
                                    mock_response.choices[0].message = SimpleNamespace()
                                    mock_response.choices[0].message.content = corrected_content
                                    response = mock_response
                                    # Try to get cost from error's raw_response if available
                                    try:
                                        raw_resp = getattr(e, 'raw_response', None)
                                        if raw_resp:
                                            cost = completion_cost(completion_response=raw_resp)
                                        else:
                                            cost = 0.0
                                    except Exception:
                                        cost = 0.0
                                    logger.warning(f"Converted string action '{action}' to integer {parsed['action']}")
                                else:
                                    raise
                            # Handle action_sequence case
                            elif isinstance(parsed, dict) and "action_sequence" in parsed:
                                action_seq = parsed["action_sequence"]
                                if isinstance(action_seq, list):
                                    # Convert string action names in sequence to integers
                                    action_map = {"LEFT": 0, "RIGHT": 1, "UP": 2, "DOWN": 3}
                                    converted_seq = []
                                    for action in action_seq:
                                        if isinstance(action, int) and 0 <= action <= 3:
                                            converted_seq.append(action)
                                        elif isinstance(action, str):
                                            if action.isdigit():
                                                action_int = int(action)
                                                if 0 <= action_int <= 3:
                                                    converted_seq.append(action_int)
                                            elif action.upper() in action_map:
                                                converted_seq.append(action_map[action.upper()])
                                    if len(converted_seq) > 0:
                                        parsed["action_sequence"] = converted_seq
                                        # Create a mock response object with corrected content
                                        corrected_content = json.dumps(parsed)
                                        from types import SimpleNamespace
                                        mock_response = SimpleNamespace()
                                        mock_response.choices = [SimpleNamespace()]
                                        mock_response.choices[0].message = SimpleNamespace()
                                        mock_response.choices[0].message.content = corrected_content
                                        response = mock_response
                                        # Try to get cost from error's raw_response if available
                                        try:
                                            raw_resp = getattr(e, 'raw_response', None)
                                            if raw_resp:
                                                cost = completion_cost(completion_response=raw_resp)
                                            else:
                                                cost = 0.0
                                        except Exception:
                                            cost = 0.0
                                        logger.warning(f"Converted action sequence with string actions to integers")
                                    else:
                                        raise
                                else:
                                    raise
                            else:
                                raise
                        else:
                            raise
                    except Exception as conv_exc:
                        logger.error(f"Failed to convert action string to integer: {conv_exc}")
                        raise e
                else:
                    raise

            # Parse response
            content = response.choices[0].message.content
            if not content:
                choice = response.choices[0]
                # Log the entire choice object to see the finish_reason
                logger.error(f"Empty response from model. Full choice object: {choice}")
                # Raise a more informative error
                finish_reason = choice.get("finish_reason", "N/A")
                raise ValueError(
                    f"Empty response from model. Finish reason: '{finish_reason}'"
                )

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
    if "</think>" in content and "qwen3-30b-a3b" in model_name:
        reasoning_content = content.split("</think>")[0]
        content = content.split("</think>")[1]
        response.choices[0].message.content = content.strip()
        response["choices"][0]["message"]["reasoning_content"] = (
            reasoning_content.strip()
        )
        return content, response
    return content, response
