"""agent_qlearning.py
======================

Tabular Q-Learning agent scaffold.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
from .base import NavigationPolicy


class QLearningAgent(NavigationPolicy):
    """Tabular Q-Learning agent skeleton.

    Q-table structure:
        Dict[state_tuple, List[float]] with length == action_space.n
    """

    def __init__(
        self,
        action_space: Any,
        learning_rate: float = 0.1,
        discount: float = 0.99,
        epsilon: float = 0.2,
        min_epsilon: float = 0.01,
        epsilon_decay: float = 0.995,
    ) -> None:
        super().__init__(action_space)
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[Tuple[int, ...], List[float]] = {}

    def encode_state(
        self, observation: Any
    ) -> Tuple[int, ...]:  # pragma: no cover - stub
        raise NotImplementedError

    def act(self, observation: Any) -> int:  # pragma: no cover - stub
        raise NotImplementedError

    def update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        terminated: bool,
    ) -> None:  # pragma: no cover - stub
        raise NotImplementedError

    def decay_epsilon(self) -> None:  # pragma: no cover - stub
        raise NotImplementedError


__all__ = ["QLearningAgent"]
