import json
import typing as t
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict


@dataclass
class Step:
    observation: str
    action: str
    reward: t.Optional[float]
    metadata: Dict


@dataclass
class Trajectory:
    steps: list[Step]
    action_space: t.List[str]
    final_reward: t.Optional[float]
    metadata: t.Optional[Dict]


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


def load_trajectory_from_file(file_path: str | Path) -> Trajectory:
    """Read a trajectory from disk."""
    data = json.loads(Path(file_path).read_text())

    steps = [Step(**step_dict) for step_dict in data.get("steps", [])]
    action_space = data.get("action_space") or []
    final_reward = data.get("final_reward")
    metadata = data.get("metadata")
    return Trajectory(
        steps=steps,
        action_space=action_space,
        final_reward=final_reward,
        metadata=metadata,
    )


@dataclass
class PreferenceResult:
    """Result of preference elicitation for a single trajectory."""

    trajectory_id: str
    model: str
    assessment: str
    reasoning: str
    timestamp: str


@dataclass
class ComparisonResult:
    """Result of pairwise trajectory comparison."""

    trajectory_a_id: str
    trajectory_b_id: str
    model: str
    preferred_trajectory: str  # "A", "B", or "tie"
    preference_strength: str  # "weak", "moderate", "strong"
    reasoning: str
    timestamp: str


@dataclass
class PreferenceAnalysis:
    """Aggregated analysis of multiple preference results."""

    model: str
    total_evaluations: int
    preference_patterns: Dict[str, object]
    common_themes: list[str]
    timestamp: str


def trajectory_to_json(result: Dict[str, object]) -> str:
    """Serialise a trajectory scoring result."""
    return json.dumps(result, indent=2)
