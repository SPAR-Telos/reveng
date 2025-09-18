"""policy_astar.py
===================

A* search-based navigation policy scaffolding.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from .base import NavigationPolicy


class AStarPolicy(NavigationPolicy):
    """Policy that pre-computes a shortest (or near-shortest) path via A*.

    Planned outline:
        - On reset: inspect env grid & goal, run A* to produce node path.
        - Convert path into environment action indices.
        - On act: pop next planned action; optionally re-plan if depleted.
    """

    def __init__(self, action_space: Any, allow_replan: bool = False):
        super().__init__(action_space)
        self.allow_replan = allow_replan
        self._planned_actions: List[int] = []
        self._path_nodes: List[Tuple[int, int]] = []

    def reset(
        self, *, env: Optional[Any] = None, observation: Optional[Any] = None
    ) -> None:  # pragma: no cover - stub
        raise NotImplementedError

    def act(self, observation: Any) -> int:  # pragma: no cover - stub
        raise NotImplementedError


__all__ = ["AStarPolicy"]
