import json
import typing as t
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class Step:
    observation: str
    action: str
    reward: t.Optional[float]
    thought: t.Optional[str]


@dataclass
class Trajectory:
    steps: list[Step]
    action_space: t.List[str]
    final_reward: t.Optional[float]


def load_trajectory_from_file(file_path: str | Path) -> Trajectory:
    """Read a trajectory from disk."""
    data = json.loads(Path(file_path).read_text())

    steps = [Step(**step_dict) for step_dict in data.get("steps", [])]
    action_space = data.get("action_space") or []
    final_reward = data.get("final_reward")
    return Trajectory(steps=steps, action_space=action_space, final_reward=final_reward)


def trajectory_to_json(result: Dict[str, object]) -> str:
    """Serialise a trajectory scoring result."""
    return json.dumps(result, indent=2)
