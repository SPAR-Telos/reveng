"""base.py
=================

Core abstract interfaces and shared types for navigation policies and
trajectory generation utilities. This file intentionally contains only
lightweight abstractions to minimize import overhead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class NavigationPolicy(ABC):
    """Abstract base interface for all navigation policies.

    Responsibilities:
        - Provide an `act` method to choose an action given the current observation.
        - Optionally maintain internal state (e.g., path queue for A*).
        - Offer lifecycle hooks (`reset`, `on_episode_end`).
    """

    def __init__(self, action_space: Any):
        self.action_space = action_space

    def reset(
        self, *, env: Optional[Any] = None, observation: Optional[Any] = None
    ) -> None:
        """Reset internal policy state at the beginning of an episode."""

    @abstractmethod
    def act(self, observation: Any) -> int:
        """Select an action given the current observation."""

    def on_episode_end(
        self, *, trajectory: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Lifecycle hook invoked after an episode completes."""


__all__ = ["NavigationPolicy"]
