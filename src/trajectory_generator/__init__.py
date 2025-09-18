"""Trajectory generator package.

Provides navigation policy abstractions and (future) implementations for
search-based, heuristic, and reinforcement learning strategies.

"""

from .agent_deep import DQNAgent, PPOAgent
from .agent_qlearning import QLearningAgent
from .base import NavigationPolicy
from .policy_astar import AStarPolicy
from .policy_random import RandomMovePolicy
from .types import EpisodeSummary, Trajectory, Transition
from .utils import (
    evaluate_policy,
    extract_state_representation,
    generate_trajectories,
    generate_trajectory,
    get_policy,
    list_available_policies,
    run_episode,
)

__all__ = [
    "NavigationPolicy",
    "RandomMovePolicy",
    "AStarPolicy",
    "QLearningAgent",
    "DQNAgent",
    "PPOAgent",
    "Transition",
    "EpisodeSummary",
    "Trajectory",
    "extract_state_representation",
    "run_episode",
    "generate_trajectory",
    "generate_trajectories",
    "evaluate_policy",
    "get_policy",
    "list_available_policies",
]
