"""policy_random.py
====================

Random / stochastic baseline navigation policies.
"""

from __future__ import annotations

from typing import Any

from .base import NavigationPolicy


class RandomMovePolicy(NavigationPolicy):
    """Policy that samples uniformly from feasible actions each step."""

    def act(self, observation: Any) -> int:  # pragma: no cover - stub
        raise NotImplementedError


__all__ = ["RandomMovePolicy"]
