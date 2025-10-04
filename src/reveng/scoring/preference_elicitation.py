import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from reveng.datatypes import (
    ComparisonResult,
    PreferenceResult,
    Trajectory,
    load_trajectory_from_file,
    trajectory_to_json,
)
from reveng.scoring.llm_interface import BaseLLMInterface

logger = logging.getLogger(__name__)


class PreferenceElicitor(BaseLLMInterface):
    """Elicit preferences from LLMs for trajectories."""

    def __init__(
        self,
        model_name: str,
        template_path: Optional[Path | str] = None,
        temperature: float = 0.7,  # Slightly higher for more creative responses
    ) -> None:
        """
        Args:
            model_name: Identifier understood by ``litellm`` (e.g. ``"gpt-4"``).
            template_path: Optional override for the Jinja prompt template.
            temperature: Forwarded to the model; higher for more creative responses.
        """
        super().__init__(model_name, template_path, temperature)

        # Load specific templates
        self._single_eval_template = self._load_template(
            Path(__file__).parent / "templates" / "single_trajectory_evaluation.j2"
        )
        self._comparison_template = self._load_template(
            Path(__file__).parent / "templates" / "trajectory_comparison.j2"
        )

    def evaluate_single_trajectory(
        self, trajectory: Trajectory, trajectory_id: str = "trajectory"
    ) -> PreferenceResult:
        """Get open-ended assessment of a single trajectory."""
        logger.info(
            f"Evaluating trajectory {trajectory_id} with {len(trajectory.steps)} steps"
        )

        # Extract mission from first observation
        mission = self._extract_mission(trajectory.steps[0].observation)

        # Create trajectory summary
        trajectory_summary = self._create_trajectory_summary(trajectory)

        # Generate prompt
        prompt = self._single_eval_template.render(
            mission=mission,
            trajectory_summary=trajectory_summary,
            steps=trajectory.steps,
            final_reward=trajectory.final_reward,
        )

        logger.debug(f"Prompt for trajectory {trajectory_id}:\n{prompt}")

        try:
            response = self._make_completion_request(prompt)
            content = (response.choices[0].message.content or "").strip()

            # Parse response to extract assessment and reasoning
            assessment, reasoning = self._parse_single_evaluation_response(content)

            return PreferenceResult(
                trajectory_id=trajectory_id,
                model=self.model_name,
                assessment=assessment,
                reasoning=reasoning,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as exc:
            logger.error(f"Evaluation failed for trajectory {trajectory_id}: {exc}")
            return PreferenceResult(
                trajectory_id=trajectory_id,
                model=self.model_name,
                assessment="Evaluation failed",
                reasoning=f"Error: {str(exc)}",
                timestamp=datetime.now().isoformat(),
            )

    def compare_trajectories(
        self,
        trajectory_a: Trajectory,
        trajectory_b: Trajectory,
        trajectory_a_id: str = "A",
        trajectory_b_id: str = "B",
    ) -> ComparisonResult:
        """Compare two trajectories and determine preference."""
        logger.info(f"Comparing trajectories {trajectory_a_id} and {trajectory_b_id}")

        # Extract mission from first observation
        mission = self._extract_mission(trajectory_a.steps[0].observation)

        # Create summaries
        summary_a = self._create_trajectory_summary(trajectory_a)
        summary_b = self._create_trajectory_summary(trajectory_b)

        # Generate prompt
        prompt = self._comparison_template.render(
            mission=mission,
            trajectory_a_summary=summary_a,
            trajectory_a_steps=trajectory_a.steps,
            trajectory_a_final_reward=trajectory_a.final_reward,
            trajectory_b_summary=summary_b,
            trajectory_b_steps=trajectory_b.steps,
            trajectory_b_final_reward=trajectory_b.final_reward,
        )

        logger.debug(f"Comparison prompt:\n{prompt}")

        try:
            response = self._make_completion_request(prompt)
            content = (response.choices[0].message.content or "").strip()

            # Parse response to extract preference details
            preferred, strength, reasoning = self._parse_comparison_response(content)

            return ComparisonResult(
                trajectory_a_id=trajectory_a_id,
                trajectory_b_id=trajectory_b_id,
                model=self.model_name,
                preferred_trajectory=preferred,
                preference_strength=strength,
                reasoning=reasoning,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as exc:
            logger.error(f"Comparison failed: {exc}")
            return ComparisonResult(
                trajectory_a_id=trajectory_a_id,
                trajectory_b_id=trajectory_b_id,
                model=self.model_name,
                preferred_trajectory="tie",
                preference_strength="weak",
                reasoning=f"Error: {str(exc)}",
                timestamp=datetime.now().isoformat(),
            )

    def rank_trajectories(
        self, trajectories: List[Trajectory], trajectory_ids: Optional[List[str]] = None
    ) -> List[PreferenceResult]:
        """Rank multiple trajectories by preference."""
        if trajectory_ids is None:
            trajectory_ids = [f"trajectory_{i}" for i in range(len(trajectories))]

        logger.info(f"Ranking {len(trajectories)} trajectories")

        # Evaluate each trajectory individually
        results = []
        for trajectory, trajectory_id in zip(trajectories, trajectory_ids):
            result = self.evaluate_single_trajectory(trajectory, trajectory_id)
            results.append(result)

        # Sort by assessment quality (this is a simple heuristic)
        # In practice, you might want to use pairwise comparisons for ranking
        results.sort(key=lambda x: len(x.assessment), reverse=True)

        return results

    def _extract_mission(self, observation: str) -> str:
        """Extract mission from observation text."""
        lines = observation.splitlines()
        for line in lines:
            if line.startswith("Mission:"):
                return line.removeprefix("Mission:").strip()
        return "Unknown mission"

    def _create_trajectory_summary(self, trajectory: Trajectory) -> str:
        """Create a summary of the trajectory."""
        summary_parts = [
            f"Length: {len(trajectory.steps)} steps",
            f"Final reward: {trajectory.final_reward if trajectory.final_reward is not None else 'N/A'}",
            f"Action space: {', '.join(trajectory.action_space)}",
        ]
        return "\n".join(summary_parts)

    def _parse_single_evaluation_response(self, content: str) -> tuple[str, str]:
        """Parse single trajectory evaluation response."""
        # Simple parsing - in practice, you might want more sophisticated parsing
        lines = content.split("\n")
        assessment_lines = []
        reasoning_lines = []

        in_reasoning = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(
                keyword in line.lower()
                for keyword in ["reasoning", "explanation", "because"]
            ):
                in_reasoning = True
            if in_reasoning:
                reasoning_lines.append(line)
            else:
                assessment_lines.append(line)

        assessment = " ".join(assessment_lines) if assessment_lines else content
        reasoning = (
            " ".join(reasoning_lines)
            if reasoning_lines
            else "No specific reasoning provided"
        )

        return assessment, reasoning

    def _parse_comparison_response(self, content: str) -> tuple[str, str, str]:
        """Parse trajectory comparison response."""
        content_lower = content.lower()

        # Extract preference
        if (
            "prefer trajectory a" in content_lower
            or "trajectory a is better" in content_lower
        ):
            preferred = "A"
        elif (
            "prefer trajectory b" in content_lower
            or "trajectory b is better" in content_lower
        ):
            preferred = "B"
        else:
            preferred = "tie"

        # Extract strength
        if "strong" in content_lower:
            strength = "strong"
        elif "moderate" in content_lower:
            strength = "moderate"
        else:
            strength = "weak"

        # Extract reasoning (everything after the preference)
        reasoning = content  # For now, return the full response as reasoning

        return preferred, strength, reasoning


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Elicit preferences for trajectories")
    parser.add_argument("trajectory_file", help="Path to a trajectory json file")
    parser.add_argument("--model", default="gpt-4", help="Model identifier for litellm")
    parser.add_argument(
        "--template",
        type=Path,
        help="Optional path to a custom Jinja prompt template",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the preference result as JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase log verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    trajectory = load_trajectory_from_file(args.trajectory_file)
    elicitor = PreferenceElicitor(
        args.model,
        template_path=args.template,
    )
    result = elicitor.evaluate_single_trajectory(trajectory)

    output = trajectory_to_json(result.__dict__)
    if args.output:
        Path(args.output).write_text(output)
        logger.info(f"Wrote preference result to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
