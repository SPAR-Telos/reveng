"""types.py
================

Typed containers (dataclasses) for trajectory generation.

These are intentionally lightweight and framework-agnostic.
Implementations consuming them may extend or wrap for vectorized
experience buffers later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(slots=True)
class Transition:
    """A single environment interaction step.

    Attributes:
        obs: Observation before taking the action.
        action: Discrete action index taken.
        reward: Scalar reward returned by env.
        next_obs: Observation after transition.
        terminated: True if episode ended via success condition.
        truncated: True if episode ended via time/step limit.
        info: Arbitrary environment auxiliary data.
    """

    obs: Any
    action: int
    reward: float
    next_obs: Any
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


@dataclass(slots=True)
class EpisodeSummary:
    """Aggregate metrics for a single episode rollout.

    Attributes:
        total_reward: Sum of rewards across the episode.
        steps: Number of environment steps executed.
        success: Boolean indicator (terminated and not truncated) or
            alternative success definition depending on policy.
        terminated: Whether env signaled terminal condition.
        truncated: Whether env signaled truncation condition.
    """

    total_reward: float
    steps: int
    success: bool
    terminated: bool
    truncated: bool


Trajectory = List[Transition]
"""Type alias representing an ordered list of ``Transition`` objects.

Semantics:
        - The list order is chronological.
        - Implementations SHOULD ensure transitions[i].next_obs == transitions[i+1].obs
            for 0 <= i < len(trajectory)-1 (except in truncated edge cases).
        - May be empty if an early termination occurs at reset (rare) or if
            partial rollouts stop before any action.
"""

__all__ = ["Transition", "EpisodeSummary", "Trajectory"]
