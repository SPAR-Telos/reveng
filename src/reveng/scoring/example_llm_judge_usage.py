#!/usr/bin/env python3
"""
Example usage of the LLMJudgeScorer for scoring and comparing trajectories.
"""

import json
import logging
from pathlib import Path

from reveng.datatypes import Step, Trajectory
from reveng.scoring.llm_judge import LLMJudgeScorer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_trajectory() -> Trajectory:
    """Create a sample trajectory for testing."""
    steps = [
        Step(
            observation="Mission: Find the red key and open the door\nGrid:\n# # # # #\n# A # # #\n# # # # #\n# # K # #\n# # # # #\nLegend:\nA: Agent\nK: Key\n#: Wall",
            action="move_right",
            reward=0.0,
            thought="I need to find the red key first",
        ),
        Step(
            observation="Mission: Find the red key and open the door\nGrid:\n# # # # #\n# # A # #\n# # # # #\n# # K # #\n# # # # #\nLegend:\nA: Agent\nK: Key\n#: Wall",
            action="move_down",
            reward=0.0,
            thought="Moving down to get closer to the key",
        ),
        Step(
            observation="Mission: Find the red key and open the door\nGrid:\n# # # # #\n# # # # #\n# # A # #\n# # K # #\n# # # # #\nLegend:\nA: Agent\nK: Key\n#: Wall",
            action="pickup",
            reward=1.0,
            thought="Picked up the key!",
        ),
        Step(
            observation="Mission: Find the red key and open the door\nGrid:\n# # # # #\n# # # # #\n# # A # #\n# # # # #\n# # # # #\nLegend:\nA: Agent\n#: Wall",
            action="move_up",
            reward=0.0,
            thought="Now I need to find the door",
        ),
        Step(
            observation="Mission: Find the red key and open the door\nGrid:\n# # # # #\n# # A # #\n# # # # #\n# # # # #\n# # # # #\nLegend:\nA: Agent\n#: Wall",
            action="move_up",
            reward=0.0,
            thought="Looking for the door",
        ),
        Step(
            observation="Mission: Find the red key and open the door\nGrid:\n# # # # #\n# # A # #\n# # # # #\n# # # # #\n# # # # #\nLegend:\nA: Agent\n#: Wall",
            action="open_door",
            reward=10.0,
            thought="Found and opened the door! Mission complete!",
        ),
    ]

    return Trajectory(
        steps=steps,
        action_space=[
            "move_up",
            "move_down",
            "move_left",
            "move_right",
            "pickup",
            "open_door",
        ],
        final_reward=11.0,
    )


def create_alternative_trajectory() -> Trajectory:
    """Create an alternative trajectory for comparison."""
    steps = [
        Step(
            observation="Mission: Find the red key and open the door\nGrid:\n# # # # #\n# A # # #\n# # # # #\n# # K # #\n# # # # #\nLegend:\nA: Agent\nK: Key\n#: Wall",
            action="move_left",
            reward=0.0,
            thought="Let me explore left first",
        ),
        Step(
            observation="Mission: Find the red key and open the door\nGrid:\n# # # # #\nA # # # #\n# # # # #\n# # K # #\n# # # # #\nLegend:\nA: Agent\nK: Key\n#: Wall",
            action="move_right",
            reward=0.0,
            thought="That was a dead end, going back",
        ),
        Step(
            observation="Mission: Find the red key and open the door\nGrid:\n# # # # #\n# A # # #\n# # # # #\n# # K # #\n# # # # #\nLegend:\nA: Agent\nK: Key\n#: Wall",
            action="move_right",
            reward=0.0,
            thought="Now moving right",
        ),
        Step(
            observation="Mission: Find the red key and open the door\nGrid:\n# # # # #\n# # A # #\n# # # # #\n# # K # #\n# # # # #\nLegend:\nA: Agent\nK: Key\n#: Wall",
            action="move_down",
            reward=0.0,
            thought="Moving down towards the key",
        ),
        Step(
            observation="Mission: Find the red key and open the door\nGrid:\n# # # # #\n# # # # #\n# # A # #\n# # K # #\n# # # # #\nLegend:\nA: Agent\nK: Key\n#: Wall",
            action="pickup",
            reward=1.0,
            thought="Got the key!",
        ),
        Step(
            observation="Mission: Find the red key and open the door\nGrid:\n# # # # #\n# # # # #\n# # A # #\n# # # # #\n# # # # #\nLegend:\nA: Agent\n#: Wall",
            action="move_up",
            reward=0.0,
            thought="Now looking for the door",
        ),
        Step(
            observation="Mission: Find the red key and open the door\nGrid:\n# # # # #\n# # A # #\n# # # # #\n# # # # #\n# # # # #\nLegend:\nA: Agent\n#: Wall",
            action="move_up",
            reward=0.0,
            thought="Still looking for the door",
        ),
        Step(
            observation="Mission: Find the red key and open the door\nGrid:\n# # # # #\n# # A # #\n# # # # #\n# # # # #\n# # # # #\nLegend:\nA: Agent\n#: Wall",
            action="open_door",
            reward=10.0,
            thought="Found the door and opened it!",
        ),
    ]

    return Trajectory(
        steps=steps,
        action_space=[
            "move_up",
            "move_down",
            "move_left",
            "move_right",
            "pickup",
            "open_door",
        ],
        final_reward=11.0,
    )


def main():
    """Demonstrate LLM judge scoring functionality."""
    # Create sample trajectories
    trajectory1 = create_sample_trajectory()
    trajectory2 = create_alternative_trajectory()

    # Initialize the LLM judge scorer
    scorer = LLMJudgeScorer(
        model_name="gpt-4",  # You can change this to any supported model
        temperature=0.0,
    )

    print("=== Individual Trajectory Scoring ===\n")

    # Score individual trajectories
    print("Scoring Trajectory 1 (Direct approach)...")
    result1 = scorer.score_individual_trajectory(trajectory1)
    print(f"Overall Score: {result1.overall_score}/100")
    print("Category Scores:")
    for category in result1.category_scores:
        print(
            f"  - {category.category}: {category.points}/{category.max_points} - {category.explanation}"
        )
    print(f"Key Strengths: {', '.join(result1.key_strengths)}")
    print(f"Areas for Improvement: {', '.join(result1.areas_for_improvement)}")
    print(f"Final Assessment: {result1.final_assessment[:200]}...")
    print()

    print("Scoring Trajectory 2 (Exploratory approach)...")
    result2 = scorer.score_individual_trajectory(trajectory2)
    print(f"Overall Score: {result2.overall_score}/100")
    print("Category Scores:")
    for category in result2.category_scores:
        print(
            f"  - {category.category}: {category.points}/{category.max_points} - {category.explanation}"
        )
    print(f"Key Strengths: {', '.join(result2.key_strengths)}")
    print(f"Areas for Improvement: {', '.join(result2.areas_for_improvement)}")
    print(f"Final Assessment: {result2.final_assessment[:200]}...")
    print()

    print("=== Trajectory Comparison ===\n")

    # Compare trajectories
    comparison_result = scorer.compare_trajectories(
        trajectories=[trajectory1, trajectory2],
        trajectory_names=["Direct Approach", "Exploratory Approach"],
    )

    print("Ranking:")
    for comparison in comparison_result.ranking:
        print(
            f"{comparison.rank}. {comparison.trajectory_name} - {comparison.justification}"
        )

    print("\nCategory Analyses:")
    for analysis in comparison_result.category_analyses:
        print(f"  - {analysis.category}: {analysis.analysis[:100]}...")

    print(f"\nKey Differentiators: {', '.join(comparison_result.key_differentiators)}")
    print(f"Winner Justification: {comparison_result.winner_justification[:300]}...")
    print(f"Overall Assessment: {comparison_result.overall_assessment[:300]}...")

    # Save results to files
    output_dir = Path("scoring_results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "individual_scores.json", "w") as f:
        json.dump(
            {"trajectory1": result1.model_dump(), "trajectory2": result2.model_dump()},
            f,
            indent=2,
        )

    with open(output_dir / "comparison_result.json", "w") as f:
        json.dump(comparison_result.model_dump(), f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
