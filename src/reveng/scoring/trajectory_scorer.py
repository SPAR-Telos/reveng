import argparse
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from reveng.datatypes import (
    Step,
    Trajectory,
    load_trajectory_from_file,
    trajectory_to_json,
)
from reveng.scoring.llm_interface import BaseLLMInterface

logger = logging.getLogger(__name__)


class TrajectoryScorer(BaseLLMInterface):
    """Score trajectories by querying an LLM for action log probabilities."""

    def __init__(
        self,
        model_name: str,
        template_path: Optional[Path | str] = None,
        top_logprobs: Optional[int] = None,
        temperature: float = 0.0,  # Keep at 0.0 for scoring
    ) -> None:
        """
        Args:
            model_name: Identifier understood by ``litellm`` (e.g. ``"gpt-4"``).
            template_path: Optional override for the Jinja prompt template.
            top_logprobs: Optional override for ``top_logprobs`` when calling
                the model. Defaults to the action-space size.
            temperature: Forwarded to the model; keep at ``0.0`` for scoring.
        """
        super().__init__(model_name, template_path, temperature)
        self.top_logprobs = top_logprobs

    def score_trajectory_with_probs(self, trajectory: Trajectory) -> Dict[str, object]:
        """Return aggregated log probability metrics for a trajectory."""
        logger.info(f"Scoring trajectory with {len(trajectory.steps)} steps")

        results: Dict[str, object] = {
            "model": self.model_name,
            "trajectory_length": len(trajectory.steps),
            "step_scores": [],
            "total_logprob": 0.0,
            "final_reward": trajectory.final_reward,
        }

        step_scores: List[Dict[str, object]] = []
        total_logprob = 0.0

        for step_index, current_step in enumerate(trajectory.steps):
            logger.debug(f"Scoring step {step_index}")
            previous_steps = trajectory.steps[:step_index]
            step_score = self._score_step(
                step_index,
                current_step,
                previous_steps,
                trajectory.action_space,
            )
            step_scores.append(step_score)
            total_logprob += step_score["logprob"]

        results["step_scores"] = step_scores
        results["total_logprob"] = total_logprob
        return results

    def _score_step(
        self,
        step_index: int,
        current_step: Step,
        previous_steps: Sequence[Step],
        action_space: Sequence[str],
    ) -> Dict[str, object]:
        prompt = self._generate_prompt(current_step, previous_steps, action_space)
        logger.debug(f"Prompt for step {step_index}:\n{prompt}")

        top_logprobs = self.top_logprobs or max(len(action_space), 5)
        if len(action_space) > 5:
            logger.warning(
                f"Action space is too large, setting top_logprobs to {top_logprobs}"
            )

        try:
            response = self._completion_with_retry(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                logprobs=True,
                top_logprobs=top_logprobs,
                temperature=self.temperature,
                allowed_openai_params=["logprobs", "top_logprobs"],
            )
        except Exception as exc:
            logger.error(f"Model request failed for step {step_index}: {exc}")
            return {
                "step_index": step_index,
                "action_taken": current_step.action,
                "logprob": float("-inf"),
                "available_logprobs": {},
                "model_response": None,
                "error": str(exc),
            }

        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        action_logprobs = self._extract_action_logprobs(choice.logprobs, action_space)
        taken_action_logprob = action_logprobs.get(current_step.action, float("-inf"))

        logger.debug(
            f"Step {step_index} action {current_step.action} logprob {taken_action_logprob}"
        )

        return {
            "step_index": step_index,
            "action_taken": current_step.action,
            "logprob": taken_action_logprob,
            "available_logprobs": action_logprobs,
            "model_response": content,
        }

    def _generate_prompt(
        self,
        current_step: Step,
        previous_steps: Sequence[Step],
        action_space: Sequence[str],
    ) -> str:
        observation_lines = current_step.observation.splitlines()
        mission = ""
        grid_lines: List[str] = []

        for index, line in enumerate(observation_lines):
            if index == 0 and line.startswith("Mission:"):
                mission = line.removeprefix("Mission:").strip()
                continue
            if line.strip().startswith("Legend"):
                break
            grid_lines.append(line)

        grid_state = "\n".join(line for line in grid_lines if line)

        return self.render_template(
            mission=mission or "",
            grid_state=grid_state or current_step.observation,
            previous_steps=previous_steps,
            current_observation=current_step.observation,
            action_space=list(action_space),
        )

    @staticmethod
    def _extract_action_logprobs(
        logprobs: Optional[object], action_space: Sequence[str]
    ) -> Dict[str, float]:
        """Map actions present in the first token's top logprobs to their values."""
        if logprobs is None or getattr(logprobs, "content", None) is None:
            return {}

        content_entries = logprobs.content
        if not content_entries:
            return {}

        first_entry = content_entries[0]
        candidates: Dict[str, float] = {}

        def normalise(token: str) -> str:
            return token.strip()

        token = normalise(getattr(first_entry, "token", ""))
        if token in action_space and math.isfinite(
            getattr(first_entry, "logprob", float("nan"))
        ):
            candidates[token] = first_entry.logprob

        for top in getattr(first_entry, "top_logprobs", []) or []:
            normalised = normalise(getattr(top, "token", ""))
            logprob_value = getattr(top, "logprob", float("nan"))
            if normalised in action_space and math.isfinite(logprob_value):
                existing = candidates.get(normalised, float("-inf"))
                if logprob_value > existing:
                    candidates[normalised] = logprob_value

        ordered: Dict[str, float] = {}
        for action in action_space:
            if action in candidates:
                ordered[action] = candidates[action]
        return ordered


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a trajectory JSON file")
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
        help="Optional path to write the scoring result as JSON",
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
    scorer = TrajectoryScorer(
        args.model,
        template_path=args.template,
    )
    result = scorer.score_trajectory_with_probs(trajectory)

    output = trajectory_to_json(result)
    if args.output:
        Path(args.output).write_text(output)
        logger.info(f"Wrote scoring output to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
