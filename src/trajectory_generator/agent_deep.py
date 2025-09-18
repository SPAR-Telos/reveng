"""agent_deep.py
=================

Deep RL agent scaffolds (DQN, PPO). No concrete implementation included.
"""

from __future__ import annotations
from typing import Any, Dict, List
from .base import NavigationPolicy


class DQNAgent(NavigationPolicy):  # pragma: no cover - stub
    """Deep Q-Network agent stub.

    Planned features:
        - Replay buffer
        - Target network sync
        - Epsilon scheduling
    """

    def act(self, observation: Any) -> int:  # pragma: no cover - stub
        raise NotImplementedError

    def optimize(self, batch: Any) -> Dict[str, float]:  # pragma: no cover - stub
        raise NotImplementedError


class PPOAgent(NavigationPolicy):  # pragma: no cover - stub
    """Proximal Policy Optimization agent stub.

    Planned features:
        - Actor-critic networks
        - GAE advantage estimation
        - Clipped surrogate objective
    """

    def act(self, observation: Any) -> int:  # pragma: no cover - stub
        raise NotImplementedError

    def update_from_trajectories(
        self, trajectories: List[Dict[str, Any]]
    ) -> Dict[str, float]:  # pragma: no cover - stub
        raise NotImplementedError


__all__ = ["DQNAgent", "PPOAgent"]
