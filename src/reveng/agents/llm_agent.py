import logging
import traceback
from pathlib import Path
from typing import Any, Optional, Tuple

from minigrid.minigrid_env import MiniGridEnv
from tenacity import retry, stop_after_attempt, wait_random_exponential

from reveng.agents.agent_abc import Agent
from reveng.agents.llm_templates import ActionResponse
from reveng.environment_generator.wrappers.text_obs_wrapper import (
    FullObservabilityTextWrapper,
)
from reveng.llm_interface import BaseLLMInterface

logger = logging.getLogger(__name__)


class LLMAgent(Agent, BaseLLMInterface):
    """An agent that uses a Language Model to select actions in a MiniGrid environment."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        name: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_name: Identifier understood by ``litellm`` (e.g. ``"gpt-4"``).
            temperature: Temperature for the model (keep at 0.0 for consistent action selection).
            name: Optional name for the agent. Defaults to class name.
        """
        Agent.__init__(self, name)
        BaseLLMInterface.__init__(
            self,
            model_name=model_name,
            temperature=temperature,
            template_path=Path(__file__).parent / "prompt_templates" / "maze.j2",
        )
        # Cost tracking
        self.total_cost = 0.0
        self.call_count = 0

    def _serialize_logprobs(self, logprobs_obj: Any) -> Optional[list[dict]]:
        """Convert model logprobs object(s) to JSON-serializable dictionaries.

        The LiteLLM/OpenAI responses expose rich objects like ChatCompletionTokenLogprob
        and TopLogprob. Those are not JSON-serializable out of the box. This helper
        extracts the relevant fields into plain Python types.
        """
        if logprobs_obj is None:
            return None

        def serialize_top(top: Any) -> dict:
            return {
                "token": getattr(top, "token", None),
                "bytes": getattr(top, "bytes", None),
                "logprob": getattr(top, "logprob", None),
                "token_id": getattr(top, "token_id", None),
            }

        def serialize_token_logprob(item: Any) -> dict:
            top_list = getattr(item, "top_logprobs", None) or []
            return {
                "token": getattr(item, "token", None),
                "bytes": getattr(item, "bytes", None),
                "logprob": getattr(item, "logprob", None),
                "token_id": getattr(item, "token_id", None),
                "text_offset": getattr(item, "text_offset", None),
                "top_logprobs": [serialize_top(t) for t in top_list],
            }

        try:
            return [serialize_token_logprob(x) for x in list(logprobs_obj)]
        except Exception:
            # Fall back: if it's already a list[dict] or otherwise JSON-ready, return as-is
            return logprobs_obj  # type: ignore[return-value]

    def select_action(
        self, env: MiniGridEnv, return_logprobs: bool = False, **kwargs: Any
    ) -> Tuple[int, dict]:
        """
        Select an action using the LLM based on the current environment state.
        """
        try:
            return self._select_action(env, return_logprobs, **kwargs)
        except Exception as e:
            logger.error(
                f"Error getting action for position {env.agent_pos} from LLM after retrying: {e}"
            )
            return -1, {"agents_name": self.name}

    # We need to retry if the model response is not valid json
    @retry(
        stop=stop_after_attempt(3),
        # 1-10 seconds between attempts, to help avoid rate limiting
        wait=wait_random_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _select_action(
        self, env: MiniGridEnv, return_logprobs: bool = False, **kwargs: Any
    ) -> Tuple[int, dict]:
        """
        Select an action using the LLM based on the current environment state.

        Args:
            env: The environment to interact with
            **kwargs: Additional arguments (ignored by this agent)

        Returns:
            The selected action and a dictionary with related metadata.
        """
        base_env = getattr(env, "unwrapped", env)
        agent_pos = (
            tuple(base_env.agent_pos) if hasattr(base_env, "agent_pos") else "unknown"
        )

        logger.info(
            f"Getting action from LLM model {self.model_name} for position {agent_pos}"
        )
        prompt = self._generate_action_query_prompt(env)

        try:
            extra_kwargs = {}
            response_format = ActionResponse
            logprobs_raw = None
            if return_logprobs:
                extra_kwargs["logprobs"] = True
                extra_kwargs["top_logprobs"] = 5
                extra_kwargs["allowed_openai_params"] = ["logprobs", "top_logprobs"]
                response_format = None

            response, cost, raw_response = self._make_completion_request(
                prompt,
                response_format=response_format,
                **extra_kwargs,
            )

            if response_format is None:
                action_response = ActionResponse.model_validate_json(response)
            else:
                action_response = response

            # Track costs
            self.total_cost += cost
            self.call_count += 1

            if return_logprobs:
                logprobs_raw = raw_response.choices[0].logprobs.content
                logprobs_serialized = self._serialize_logprobs(logprobs_raw)

            logger.info(
                f"LLM selected action {action_response.action}, cost: ${cost:.6f}, total: ${self.total_cost:.6f}"
            )
            return action_response.action.value, {
                "agents_name": self.name,
                "llm_response": action_response.action.value,
                "call_cost": cost,
                "total_cost": self.total_cost,
                "call_count": self.call_count,
                "logprobs": logprobs_serialized,
            }

        except Exception as e:
            logger.error(
                f"Error getting action from LLM: {e}\n{traceback.format_exc()}"
            )
            raise

    def _get_text_observation(self, env: MiniGridEnv) -> str:
        """Get text observation of the environment using existing wrapper."""
        # Wrap env temporarily to get text observation
        text_env = FullObservabilityTextWrapper(env)
        return text_env.observation(None)

    def _generate_action_query_prompt(self, env: MiniGridEnv) -> str:
        """Generate a prompt for the LLM to select an action."""
        # Get text observation using existing wrapper
        obs_text = self._get_text_observation(env)

        prompt = self.render_template(
            obs_text=obs_text,
        )
        return prompt

    def reset(self) -> None:
        """Reset the agent's internal state and cost tracking."""
        super().reset()
        self.total_cost = 0.0
        self.call_count = 0

    def get_cost_summary(self) -> dict:
        """Get a summary of costs incurred by this agent.

        Returns:
            Dictionary with total_cost, call_count, and avg_cost_per_call
        """
        avg_cost = self.total_cost / self.call_count if self.call_count > 0 else 0.0
        return {
            "total_cost": self.total_cost,
            "call_count": self.call_count,
            "avg_cost_per_call": avg_cost,
        }
