import logging
from pathlib import Path
from typing import List, Optional

from jinja2 import Environment, FileSystemLoader, Template
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_random_exponential

from reveng.datatypes import Trajectory
from reveng.scoring.models import (
    IndividualScoringResponse,
    TrajectoryComparisonResponse,
)

logger = logging.getLogger(__name__)


class LLMJudgeScorer:
    """Score trajectories using an LLM as a judge with rubrics."""

    def __init__(
        self,
        model_name: str,
        individual_template_path: Optional[Path | str] = None,
        comparison_template_path: Optional[Path | str] = None,
        temperature: float = 0.0,
    ) -> None:
        """
        Args:
            model_name: Identifier understood by ``litellm`` (e.g. ``"gpt-4"``).
            individual_template_path: Optional override for the individual scoring Jinja template.
            comparison_template_path: Optional override for the comparison Jinja template.
            temperature: Temperature for the model (keep at 0.0 for consistent scoring).
        """
        self.model_name = model_name
        self.temperature = temperature

        # Load individual scoring template
        individual_template = (
            Path(individual_template_path)
            if individual_template_path is not None
            else Path(__file__).parent / "templates" / "individual_scoring_rubric.j2"
        )
        self._individual_template = self._load_template(individual_template)

        # Load comparison template
        comparison_template = (
            Path(comparison_template_path)
            if comparison_template_path is not None
            else Path(__file__).parent / "templates" / "trajectory_comparison_rubric.j2"
        )
        self._comparison_template = self._load_template(comparison_template)

    @staticmethod
    def _load_template(template_path: Path) -> Template:
        env = Environment(
            loader=FileSystemLoader(template_path.parent),
        )
        return env.get_template(template_path.name)

    def score_individual_trajectory(
        self, trajectory: Trajectory
    ) -> IndividualScoringResponse:
        """
        Score a single trajectory using the individual scoring rubric.

        Returns a structured Pydantic response with scoring details.
        """
        logger.info(f"Scoring individual trajectory with {len(trajectory.steps)} steps")

        prompt = self._generate_individual_prompt(trajectory)
        logger.debug(f"Individual scoring prompt:\n{prompt}")

        try:
            response = self._completion_with_retry(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format=IndividualScoringResponse,
            )
        except Exception as exc:
            logger.error(f"Model request failed for individual scoring: {exc}")
            # Return a default response with error information
            return IndividualScoringResponse(
                overall_score=0,
                category_scores=[],
                detailed_reasoning=f"Error during scoring: {str(exc)}",
                key_strengths=[],
                areas_for_improvement=[],
                final_assessment=f"Scoring failed due to error: {str(exc)}",
            )

        # Check if LiteLLM supports direct Pydantic parsing
        if (
            hasattr(response.choices[0].message, "parsed")
            and response.choices[0].message.parsed
        ):
            # Direct Pydantic model support
            return response.choices[0].message.parsed

        # Fallback to JSON parsing
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from model")

        try:
            # Parse the JSON response into our Pydantic model
            scoring_response = IndividualScoringResponse.model_validate_json(content)
            return scoring_response
        except Exception as exc:
            logger.error(f"Failed to parse scoring response: {exc}")
            logger.debug(f"Raw response content: {content}")
            raise ValueError(f"Invalid response format from model: {exc}")

    def compare_trajectories(
        self,
        trajectories: List[Trajectory],
        trajectory_names: Optional[List[str]] = None,
    ) -> TrajectoryComparisonResponse:
        """
        Compare multiple trajectories and determine which is preferred.

        Args:
            trajectories: List of trajectories to compare
            trajectory_names: Optional names for each trajectory (for identification)

        Returns a structured Pydantic response with comparison results.
        """
        if len(trajectories) < 2:
            raise ValueError("Need at least 2 trajectories to compare")

        if trajectory_names and len(trajectory_names) != len(trajectories):
            raise ValueError(
                "Number of trajectory names must match number of trajectories"
            )

        logger.info(f"Comparing {len(trajectories)} trajectories")

        # Generate names if not provided
        if trajectory_names is None:
            trajectory_names = [f"Trajectory {i + 1}" for i in range(len(trajectories))]

        prompt = self._generate_comparison_prompt(trajectories, trajectory_names)
        logger.debug(f"Comparison prompt:\n{prompt}")

        try:
            response = self._completion_with_retry(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format=TrajectoryComparisonResponse,
            )
        except Exception as exc:
            logger.error(f"Model request failed for trajectory comparison: {exc}")
            # Return a default response with error information
            return TrajectoryComparisonResponse(
                ranking=[],
                category_analyses=[],
                key_differentiators=[],
                winner_justification=f"Error during comparison: {str(exc)}",
                recommendations=[],
                overall_assessment=f"Comparison failed due to error: {str(exc)}",
            )

        # Check if LiteLLM supports direct Pydantic parsing
        if (
            hasattr(response.choices[0].message, "parsed")
            and response.choices[0].message.parsed
        ):
            # Direct Pydantic model support
            return response.choices[0].message.parsed

        # Fallback to JSON parsing
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from model")

        try:
            # Parse the JSON response into our Pydantic model
            comparison_response = TrajectoryComparisonResponse.model_validate_json(
                content
            )
            return comparison_response
        except Exception as exc:
            logger.error(f"Failed to parse comparison response: {exc}")
            logger.debug(f"Raw response content: {content}")
            raise ValueError(f"Invalid response format from model: {exc}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=5, max=120),
        reraise=True,
    )
    def _completion_with_retry(self, **kwargs):
        response = completion(**kwargs)
        return response

    def _generate_individual_prompt(self, trajectory: Trajectory) -> str:
        """Generate prompt for individual trajectory scoring."""
        return self._individual_template.render(
            trajectory=trajectory,
            steps=trajectory.steps,
            action_space=trajectory.action_space,
            final_reward=trajectory.final_reward,
        )

    def _generate_comparison_prompt(
        self, trajectories: List[Trajectory], trajectory_names: List[str]
    ) -> str:
        """Generate prompt for trajectory comparison."""
        return self._comparison_template.render(
            trajectories=trajectories,
            trajectory_names=trajectory_names,
            action_space=trajectories[
                0
            ].action_space,  # Assume all trajectories have same action space
        )
