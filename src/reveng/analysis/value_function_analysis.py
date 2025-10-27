# reveng/src/reveng/analysis/value_function_analysis.py
"""
Utilities to compute value functions for the Simple2DNavigationEnv grids.

This module provides:
- Policy evaluation: compute V for a given policy π and discount γ.
- Value iteration: compute the optimal value function V* for discount γ.
- Shortest-path-based value: compute V assuming reward only at the goal and
  transitions are deterministic, yielding V(s) = goal_reward * γ^(d(s, goal)).
- Greedy policy extraction from a value function.
- Variants that make use of analysis helpers in `run_analysis.py`.

Assumptions
- Environment: The grid environment is the custom Simple2DNavigationEnv used in this
  project (deterministic cardinal moves; walls block movement).
- Rewards: Unless otherwise specified, we assume 0 reward everywhere except
  when transitioning into the goal cell, which yields `goal_reward` (default 1.0).
- Terminal handling: The goal cell is treated as absorbing with V(goal) = 0 (future
  rewards after entering the goal are zero).
- Indexing: Values/policies are represented as 2D lists [row=y][col=x].

Notes on integration
- We mirror the passability and neighbor conventions used by
  `compute_optimal_actions` in `reveng/analysis/run_analysis.py`.
- When appropriate, we leverage `compute_optimal_actions` to constrain value
  iteration to optimal moves (for the terminal-reward setting).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple, Union

from reveng.analysis.run_analysis import (
    compute_optimal_actions as ra_compute_optimal_actions,
)
from reveng.analysis.run_analysis import extract_llm_policy as ra_extract_llm_policy

# Type aliases
Coord = Tuple[int, int]
Policy2D = List[List[int]]  # action ids with -1 for walls/unassigned
Value2D = List[List[float]]


def _base_env(env):
    """Return the unwrapped/base env (compat with wrappers)."""
    return getattr(env, "unwrapped", env)


def _grid_dims(env) -> Tuple[int, int]:
    """Return (width, height) for the env grid."""
    base = _base_env(env)
    grid = base.grid
    return grid.width, grid.height


def _goal_pos(env) -> Coord:
    """Return (x, y) coordinates of the goal."""
    base = _base_env(env)
    gx, gy = base.goal_pos
    return int(gx), int(gy)


def _is_passable(env, x: int, y: int) -> bool:
    """Cell passability check mirroring logic used in run_analysis.py."""
    width, height = _grid_dims(env)
    if x < 0 or y < 0 or x >= width or y >= height:
        return False
    base = _base_env(env)
    cell = base.grid.get(x, y)
    return (cell is None) or (getattr(cell, "can_overlap", lambda: False)())


def _action_deltas(env) -> List[Tuple[int, int, int]]:
    """
    Map action ids to (dx, dy) using the environment's action->direction mapping.
    Falls back to standard LEFT,RIGHT,UP,DOWN if mapping unavailable.
    """
    base = _base_env(env)
    action_to_dir = getattr(base, "_action_to_direction", None)
    # MiniGrid directions: 0=right, 1=down, 2=left, 3=up
    dir_to_vec = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
    deltas: List[Tuple[int, int, int]] = []
    try:
        Actions = getattr(base, "actions", None)
        for a in range(4):
            key = Actions(a) if Actions is not None else a
            # Default mapping if env mapping is missing: LEFT->2, RIGHT->0, UP->3, DOWN->1
            default_dir = {0: 2, 1: 0, 2: 3, 3: 1}[a]
            dir_idx = action_to_dir[key] if action_to_dir is not None else default_dir
            dx, dy = dir_to_vec[int(dir_idx)]
            deltas.append((dx, dy, a))
    except Exception:
        # Fallback to canonical deltas if anything goes wrong
        deltas = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]
    return deltas


def _step_dynamics(env, x: int, y: int, action: int) -> Coord:
    """
    Deterministic one-step transition for the custom grid env.

    Behavior: Attempts to move 1 cell in the action's direction based on the
    environment's action->direction mapping. If the next cell is not passable
    (wall or out-of-bounds), the agent remains in place.
    """
    width, height = _grid_dims(env)
    # Map action to dx,dy using env mapping (falls back to standard deltas)
    deltas = _action_deltas(env)
    if action < 0 or action >= len(deltas):
        # Invalid action -> no movement
        return (x, y)
    dx, dy, _ = deltas[action]

    nx, ny = x + dx, y + dy
    if 0 <= nx < width and 0 <= ny < height and _is_passable(env, nx, ny):
        return (nx, ny)
    return (x, y)


def _init_value_grid(env, fill: float = 0.0) -> Value2D:
    """Initialize a value grid with given fill value."""
    width, height = _grid_dims(env)
    return [[fill for _ in range(width)] for _ in range(height)]


def evaluate_policy(
    env,
    policy: Union[Policy2D, Callable[[int, int], int]],
    gamma: float,
    goal_reward: float = 1.0,
    tol: float = 1e-8,
    max_iters: int = 100000,
) -> Value2D:
    """
    Evaluate a deterministic stationary policy on the grid.

    Assumptions:
    - Reward is 0 everywhere except when transitioning into the goal cell,
      which yields `goal_reward`.
    - The goal is absorbing with V(goal)=0.

    Args:
      env: A Simple2DNavigationEnv-like environment used elsewhere in this repo.
      policy: Either:
        - A 2D list [y][x] of action ids (LEFT=0, RIGHT=1, UP=2, DOWN=3), with -1 for walls.
        - A callable policy(x, y) -> action id.
      gamma: Discount factor in [0, 1].
      goal_reward: Reward received upon entering the goal state.
      tol: Convergence tolerance for infinity norm of the value update.
      max_iters: Maximum number of iterations.

    Returns:
      A 2D list V[y][x] with the evaluated value function.
    """
    width, height = _grid_dims(env)
    gx, gy = _goal_pos(env)

    # Initialize values to 0, with V(goal)=0 as absorbing
    V = _init_value_grid(env, fill=0.0)

    def get_action(x: int, y: int) -> int:
        if callable(policy):
            return int(policy(x, y))
        a = policy[y][x]
        return int(a)

    for it in range(max_iters):
        delta = 0.0
        # Compute a synchronous update (Jacobi style)
        V_new = [row[:] for row in V]

        for y in range(height):
            for x in range(width):
                if not _is_passable(env, x, y):
                    continue
                if (x, y) == (gx, gy):
                    V_new[y][x] = 0.0
                    continue

                a = get_action(x, y)
                nx, ny = _step_dynamics(env, x, y, a)
                r = goal_reward if (nx, ny) == (gx, gy) else 0.0
                new_val = r + gamma * V[ny][nx]
                delta = max(delta, abs(new_val - V[y][x]))
                V_new[y][x] = new_val

        V = V_new
        if delta < tol:
            break

    return V


def value_iteration(
    env,
    gamma: float,
    goal_reward: float = 1.0,
    tol: float = 1e-8,
    max_iters: int = 100000,
) -> Value2D:
    """
    Compute the optimal value function via value iteration.

    Assumptions:
    - Reward is 0 everywhere except when transitioning into the goal cell,
      which yields `goal_reward`.
    - The goal is absorbing with V(goal)=0.

    Args:
      env: Simple2DNavigationEnv-like grid env.
      gamma: Discount factor in [0, 1].
      goal_reward: Reward upon entering the goal.
      tol: Convergence tolerance.
      max_iters: Iteration cap.

    Returns:
      A 2D list V*[y][x] (optimal value function).
    """
    width, height = _grid_dims(env)
    gx, gy = _goal_pos(env)
    deltas = _action_deltas(env)
    actions = [a for _, _, a in deltas]

    V = _init_value_grid(env, fill=0.0)

    for it in range(max_iters):
        delta = 0.0
        V_new = [row[:] for row in V]

        for y in range(height):
            for x in range(width):
                if not _is_passable(env, x, y):
                    continue
                if (x, y) == (gx, gy):
                    V_new[y][x] = 0.0
                    continue

                best = float("-inf")
                for a in actions:
                    nx, ny = _step_dynamics(env, x, y, a)
                    r = goal_reward if (nx, ny) == (gx, gy) else 0.0
                    val = r + gamma * V[ny][nx]
                    if val > best:
                        best = val

                new_val = best if best != float("-inf") else 0.0
                delta = max(delta, abs(new_val - V[y][x]))
                V_new[y][x] = new_val

        V = V_new
        if delta < tol:
            break

    return V


def greedy_policy_from_value(env, V: Value2D) -> Policy2D:
    """
    Extract a greedy (deterministic) policy from a given value function.

    For each passable non-goal cell, choose the action a that maximizes
    r(s,a) + gamma*V(s'), but since gamma is not provided here and V may already
    be scaled by some gamma, we simply pick the action that maximizes V(s')
    with a tie-break to the smallest action id (consistent, deterministic).

    Note: If you need gamma-aware greedy extraction, use `greedy_policy_from_q`
    after calling `compute_q_from_value`.

    Args:
      env: Grid env
      V: 2D value function

    Returns:
      A 2D policy grid with action ids; walls get -1, goal gets -1 (no action).
    """
    width, height = _grid_dims(env)
    gx, gy = _goal_pos(env)
    policy: Policy2D = [[-1 for _ in range(width)] for _ in range(height)]
    deltas = _action_deltas(env)
    actions = [a for _, _, a in deltas]

    for y in range(height):
        for x in range(width):
            if not _is_passable(env, x, y):
                continue
            if (x, y) == (gx, gy):
                policy[y][x] = -1
                continue

            # Pick action that leads to neighbor with max V
            best_a = 0
            best_val = float("-inf")
            for a in actions:
                nx, ny = _step_dynamics(env, x, y, a)
                val = V[ny][nx]
                if val > best_val or (val == best_val and a < best_a):
                    best_val = val
                    best_a = a
            policy[y][x] = best_a

    return policy


def compute_q_from_value(
    env,
    V: Value2D,
    gamma: float,
    goal_reward: float = 1.0,
) -> List[List[List[float]]]:
    """
    Compute state-action values Q(s,a) = r(s,a) + gamma * V(s') from a given V.

    Args:
      env: Grid env
      V: 2D state value function
      gamma: Discount factor
      goal_reward: Terminal reward upon entering goal

    Returns:
      Q grid with shape [height][width][n_actions], where n_actions aligns with the environment's action mapping.
      For walls and goal, values are computed for completeness but are typically unused.
    """
    width, height = _grid_dims(env)
    gx, gy = _goal_pos(env)
    deltas = _action_deltas(env)
    n_actions = len(deltas)
    Q = [[[0.0 for _ in range(n_actions)] for _ in range(width)] for _ in range(height)]

    for y in range(height):
        for x in range(width):
            for a in range(n_actions):
                nx, ny = _step_dynamics(env, x, y, a)
                r = goal_reward if (nx, ny) == (gx, gy) else 0.0
                Q[y][x][a] = r + gamma * V[ny][nx]
    return Q


def greedy_policy_from_q(env, Q: List[List[List[float]]]) -> Policy2D:
    """
    Choose argmax_a Q(s,a) for each passable non-goal cell.

    Args:
      env: Grid env
      Q: 3D list [y][x][a]

    Returns:
      2D policy grid with action ids; walls and goal set to -1.
    """
    width, height = _grid_dims(env)
    gx, gy = _goal_pos(env)
    policy: Policy2D = [[-1 for _ in range(width)] for _ in range(height)]

    for y in range(height):
        for x in range(width):
            if not _is_passable(env, x, y) or (x, y) == (gx, gy):
                continue
            qa = Q[y][x]
            best_a = 0
            best_val = qa[0]
            for a in range(1, len(qa)):
                if qa[a] > best_val or (qa[a] == best_val and a < best_a):
                    best_val = qa[a]
                    best_a = a
            policy[y][x] = best_a

    return policy


def shortest_path_value(
    env,
    gamma: float,
    goal_reward: float = 1.0,
) -> Value2D:
    """
    Compute value via shortest path distances from every cell to goal.

    Under the assumption:
    - Reward is only given upon entering the goal (goal_reward).
    - Deterministic moves and passability as implemented here.
    Then the optimal value at state s equals goal_reward * gamma^(d(s, goal)),
    where d(s, goal) is the shortest number of steps to the goal (infinite if unreachable).

    This is equivalent to value iteration with only terminal reward; implemented
    more directly via a Dijkstra/BFS from the goal.

    Args:
      env: Grid env
      gamma: Discount factor
      goal_reward: Reward obtained when entering the goal

    Returns:
      2D value grid V[y][x]. Cells unreachable from the goal get V=0.0 (no reward).
    """
    width, height = _grid_dims(env)
    gx, gy = _goal_pos(env)
    V = _init_value_grid(env, fill=0.0)

    # Dijkstra/BFS from goal to get distances
    from heapq import heappop, heappush

    dist: Dict[Coord, int] = {(gx, gy): 0}
    heap: List[Tuple[int, Coord]] = [(0, (gx, gy))]

    while heap:
        d, (x, y) = heappop(heap)
        if d > dist.get((x, y), 10**9):
            continue
        for dx, dy, _ in _action_deltas(env):
            nx, ny = x + dx, y + dy
            if _is_passable(env, nx, ny):
                nd = d + 1
                if nd < dist.get((nx, ny), 10**9):
                    dist[(nx, ny)] = nd
                    heappush(heap, (nd, (nx, ny)))

    # Convert distances to values
    for (x, y), d in dist.items():
        if (x, y) == (gx, gy):
            V[y][x] = 0.0
        else:
            V[y][x] = goal_reward * (gamma**d)

    # Non-passable or unreachable cells remain at 0.0
    return V


def value_iteration_using_optimal_actions(
    env,
    gamma: float,
    goal_reward: float = 1.0,
    tol: float = 1e-8,
    max_iters: int = 100000,
) -> Value2D:
    """
    Compute the optimal value function but restrict actions to those returned by
    `compute_optimal_actions` in run_analysis.py. This function will only work if
    that utility is importable.

    This produces the same result as `shortest_path_value` (for pure terminal-reward
    setting) and is a convenient way to tie into existing analysis code.

    Args:
      env: Grid env
      gamma: Discount factor
      goal_reward: Terminal reward upon entering goal
      tol: Convergence tolerance
      max_iters: Iteration cap

    Returns:
      2D value grid V[y][x]
    """

    width, height = _grid_dims(env)
    gx, gy = _goal_pos(env)

    # optimal_actions[y][x] is a set of action ids leading closer to goal
    optimal_actions = ra_compute_optimal_actions(env)
    V = _init_value_grid(env, fill=0.0)

    for _ in range(max_iters):
        delta = 0.0
        V_new = [row[:] for row in V]

        for y in range(height):
            for x in range(width):
                if not _is_passable(env, x, y):
                    continue
                if (x, y) == (gx, gy):
                    V_new[y][x] = 0.0
                    continue

                actions_set = optimal_actions[y][x]
                if not actions_set:
                    # No optimal actions (wall/unreachable) -> value stays as is (0)
                    V_new[y][x] = V[y][x]
                    continue

                best = float("-inf")
                for a in actions_set:
                    nx, ny = _step_dynamics(env, x, y, int(a))
                    r = goal_reward if (nx, ny) == (gx, gy) else 0.0
                    val = r + gamma * V[ny][nx]
                    if val > best:
                        best = val
                new_val = best if best > float("-inf") else 0.0
                delta = max(delta, abs(new_val - V[y][x]))
                V_new[y][x] = new_val

        V = V_new
        if delta < tol:
            break

    return V


def evaluate_llm_policy_metadata(
    env,
    policy_metadata: Sequence[Sequence[object]],
    gamma: float,
    goal_reward: float = 1.0,
    tol: float = 1e-8,
    max_iters: int = 100000,
) -> Value2D:
    """
    Convenience wrapper: evaluate an LLM-derived policy stored as metadata JSON.

    This uses `extract_llm_policy` from run_analysis.py when available to parse
    the metadata into a 2D action grid.

    Args:
      env: Grid env
      policy_metadata: Nested structure from the metadata files used elsewhere
      gamma: Discount factor
      goal_reward: Terminal reward upon entering the goal
      tol: Convergence tolerance
      max_iters: Iteration cap

    Returns:
      2D value grid from policy evaluation
    """
    policy_2d = ra_extract_llm_policy(policy_metadata)  # type: ignore[call-arg]

    return evaluate_policy(
        env=env,
        policy=policy_2d,
        gamma=gamma,
        goal_reward=goal_reward,
        tol=tol,
        max_iters=max_iters,
    )


__all__ = [
    "evaluate_policy",
    "value_iteration",
    "greedy_policy_from_value",
    "compute_q_from_value",
    "greedy_policy_from_q",
    "shortest_path_value",
    "value_iteration_using_optimal_actions",
    "evaluate_llm_policy_metadata",
]
