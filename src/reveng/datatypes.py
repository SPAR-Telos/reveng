import json
import typing as t
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict


@dataclass
class Step:
    observation: str
    action: t.Optional[int]
    reward: t.Optional[float]
    note: t.Optional[str]
    metadata: Dict
    agent_pos: t.Optional[t.Tuple[int, int]] = None

    def __dict__(self) -> dict:
        return {
            "observation": self.observation,
            "action": self.action,
            "reward": self.reward,
            "note": self.note,
            "metadata": self.metadata,
            "agent_pos": self.agent_pos,
        }

    def __json__(self) -> dict:
        return self.__dict__()


@dataclass
class Trajectory:
    steps: list[Step]
    final_reward: t.Optional[float]
    traj_metadata: t.Optional[Dict]

    def __dict__(self) -> dict:
        return {
            "steps": [step.__dict__() for step in self.steps],
            "final_reward": self.final_reward,
            "traj_metadata": self.traj_metadata,
        }

    def __json__(self) -> dict:
        return self.__dict__()


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

    def to_json(self) -> int:
        """Convert Action to JSON-serializable value."""
        return self.value

    @classmethod
    def from_json(cls, value: int) -> "Action":
        """Create Action from JSON value."""
        return cls(value)

    def __json__(self) -> int:
        """Support for JSON serialization."""
        return self.value

    def to_str(self) -> str:
        """Convert Action to string."""
        return {"0": "LEFT", "1": "RIGHT", "2": "UP", "3": "DOWN"}[str(self.value)]


def load_trajectory_from_file(file_path: str | Path) -> Trajectory:
    """Read a trajectory from disk."""
    data = json.loads(Path(file_path).read_text())

    steps = [Step(**step_dict) for step_dict in data.get("steps", [])]
    action_space = data.get("action_space") or []
    final_reward = data.get("final_reward")
    traj_metadata = data.get("traj_metadata")
    return Trajectory(
        steps=steps,
        action_space=action_space,
        final_reward=final_reward,
        traj_metadata=traj_metadata,
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


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles objects with __json__ methods."""

    def default(self, obj):
        if hasattr(obj, "__json__"):
            return obj.__json__()
        return super().default(obj)


def trajectory_to_json(result: Dict[str, object]) -> str:
    """Serialise a trajectory scoring result."""
    return json.dumps(result, indent=2, cls=CustomJSONEncoder)
