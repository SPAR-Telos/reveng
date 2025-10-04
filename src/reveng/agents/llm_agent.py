import logging
from typing import Any, Optional, Tuple

from litellm import completion
from minigrid.minigrid_env import MiniGridEnv
from tenacity import retry, stop_after_attempt, wait_random_exponential

from reveng.agents.agent_abc import Agent
from reveng.agents.llm_templates import ActionResponse
from reveng.environment_generator.wrappers.text_obs_wrapper import (
    FullObservabilityTextWrapper,
)

logger = logging.getLogger(__name__)


class LLMAgent(Agent):
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
        super().__init__(name)
        self.model_name = model_name
        self.temperature = temperature

    def select_action(self, env: MiniGridEnv, **kwargs: Any) -> Tuple[int, dict]:
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
            response = self._completion_with_retry(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format=ActionResponse,
            )

            # Parse response
            action_response = self._parse_response(response)
            action = action_response.action

            # Validate action is in valid range
            if hasattr(env, "action_space") and not (0 <= action < env.action_space.n):
                logger.warning(
                    f"Invalid action {action} from LLM, using random fallback"
                )
                action = env.action_space.sample()
                return action, {
                    "agents_name": self.name,
                    "llm_response": f"invalid_action_{action_response.action}",
                    "explanation": action_response.explanation,
                }

            logger.info(f"LLM selected action {action}: {action_response.explanation}")
            return action, {
                "agents_name": self.name,
                "llm_response": action,
                "explanation": action_response.explanation,
            }

        except Exception as e:
            logger.error(f"Error getting action from LLM: {e}")
            # Fallback to random action if LLM fails
            if hasattr(env, "action_space"):
                action = env.action_space.sample()
                return action, {
                    "agents_name": self.name,
                    "llm_response": "error_fallback",
                    "error": str(e),
                }
            else:
                raise ValueError(
                    "Environment must have an action_space attribute for fallback"
                )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=5, max=120),
        reraise=True,
    )
    def _completion_with_retry(self, **kwargs):
        """Call the LLM API with retry logic for robustness."""
        return completion(**kwargs)

    def _parse_response(self, response) -> ActionResponse:
        """Parse LLM response into ActionResponse model."""
        # Check if LiteLLM supports direct Pydantic parsing
        if (
            hasattr(response.choices[0].message, "parsed")
            and response.choices[0].message.parsed
        ):
            return response.choices[0].message.parsed

        # Fallback to JSON parsing
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from model")

        try:
            return ActionResponse.model_validate_json(content)
        except Exception as exc:
            logger.error(f"Failed to parse action response: {exc}")
            logger.debug(f"Raw response content: {content}")
            raise ValueError(f"Invalid response format from model: {exc}")

    def _get_text_observation(self, env: MiniGridEnv) -> str:
        """Get text observation of the environment using existing wrapper."""
        # Wrap env temporarily to get text observation
        text_env = FullObservabilityTextWrapper(env)
        return text_env.observation(None)

    def _generate_action_query_prompt(self, env: MiniGridEnv) -> str:
        """Generate a prompt for the LLM to select an action."""
        # Get text observation using existing wrapper
        obs_text = self._get_text_observation(env)

        prompt = (
            f"You are controlling an agent in a grid environment.\n\n"
            f"{obs_text}\n\n"
            f"Available actions:\n"
            f"  0: LEFT\n"
            f"  1: RIGHT\n"
            f"  2: UP\n"
            f"  3: DOWN\n\n"
            f"Select the best action to reach the goal.\n"
            f"Respond with JSON containing:\n"
            f"  - action: the action number (0-3)\n"
            f"  - explanation: brief explanation of your choice"
        )
        return prompt
