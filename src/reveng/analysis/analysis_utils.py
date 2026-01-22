"""Shared utilities for grid analysis and visualization.

This module contains reusable components for:
- Loading grids and metadata
- Computing optimal actions
- Parsing action distributions from logprobs
- Computing entropy and cross-entropy metrics
- Statistical analysis utilities (correlations, regression, controlled analysis)
"""

import heapq
import json
import math
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# Constants
# =============================================================================

ACTION_NAME_TO_ID = {"LEFT": 0, "RIGHT": 1, "UP": 2, "DOWN": 3}
ACTION_ID_TO_NAME = {v: k for k, v in ACTION_NAME_TO_ID.items()}
ACTION_NAMES_UPPER = set(ACTION_NAME_TO_ID.keys())
LOGPROB_EPS = 1e-12

# Type aliases for clarity
ActionID = int
ActionDist = dict[ActionID, float]
OptimalActionSet = set[ActionID]
GridCoord = tuple[int, int]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class GridMetadata:
    """Metadata for a single grid instance."""

    grid_size: int
    complexity: float
    instance_id: int
    policy_metadata: list[list[dict[str, Any]]]


@dataclass
class CellMetrics:
    """Computed metrics for a single grid cell."""

    grid_id: str
    grid_size: int
    complexity: float
    instance_id: int
    x: int
    y: int
    llm_action: int
    num_optimal_actions: int
    entropy_bits: float
    optimal_entropy_bits: float
    cross_entropy_bits: Optional[float]
    jsd: Optional[float]
    optimal_mass: float
    is_action_optimal: int
    action_probs: dict[str, float]
    distance_to_goal: int = 0  # Manhattan distance in optimal path

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        base = {
            "grid_id": self.grid_id,
            "grid_size": self.grid_size,
            "complexity": self.complexity,
            "instance_id": self.instance_id,
            "x": self.x,
            "y": self.y,
            "llm_action": self.llm_action,
            "num_optimal_actions": self.num_optimal_actions,
            "entropy_bits": self.entropy_bits,
            "optimal_entropy_bits": self.optimal_entropy_bits,
            "cross_entropy_bits": self.cross_entropy_bits,
            "jsd": self.jsd,
            "optimal_mass": self.optimal_mass,
            "is_action_optimal": self.is_action_optimal,
            "distance_to_goal": self.distance_to_goal,
        }
        base.update({f"p_{name}": prob for name, prob in self.action_probs.items()})
        return base


# =============================================================================
# Trajectory Data Classes (for trajectory-based analysis)
# =============================================================================


@dataclass
class TrajectoryGridParams:
    """Essential grid parameters extracted from trajectory file."""

    grid_size: int
    complexity: float
    grid_id: int
    astar_distance: int
    agent_start: tuple[int, int]
    goal: tuple[int, int]


@dataclass
class TrajectoryStep:
    """A single step in a trajectory."""

    step_id: int
    agent_position: tuple[int, int]
    agent_action: str  # "UP", "DOWN", "LEFT", "RIGHT"


@dataclass
class LightweightTrajectory:
    """Memory-efficient trajectory representation with only essential fields."""

    grid_params: TrajectoryGridParams
    steps: list[TrajectoryStep]
    reached_goal: bool
    transform_type: str = "base"  # "base", "ReflectEnv", "RotateEnv", etc.

    @property
    def trajectory_length(self) -> int:
        return len(self.steps)


# =============================================================================
# Trajectory Parsing Utilities
# =============================================================================


def extract_agent_position_from_grid_state(grid_state: list[str]) -> tuple[int, int]:
    """Extract agent position (x, y) from grid state strings.

    Grid state format: ['  0 1 2 ...', '0 # # # ...', '1 # A _ ...', ...]
    Agent is marked with 'A'.

    Note: In the grid representation, row index is Y and column index is X.
    Format is "row col1 col2 ..." where row number is the Y coordinate.
    """
    for row_idx, row in enumerate(grid_state):
        if row_idx == 0:
            # Header row with column numbers
            continue

        # Split row into cells
        parts = row.split()
        if len(parts) < 2:
            continue

        # First part is row number (Y coordinate)
        y = int(parts[0])

        # Find 'A' in remaining parts
        for col_idx, cell in enumerate(parts[1:], start=0):
            if cell == "A":
                return (col_idx, y)

    # Fallback - shouldn't happen with valid data
    return (-1, -1)


def compute_optimal_actions_from_text_grid(
    grid_layout: list[list[str]],
    goal: tuple[int, int],
) -> tuple[dict[tuple[int, int], OptimalActionSet], dict[tuple[int, int], int]]:
    """Compute optimal actions for each cell using backward Dijkstra.

    This is a simplified version that works with text grid layouts
    (list of lists of symbols) rather than MiniGrid environments.

    Args:
        grid_layout: 2D grid where '#' is wall, others are passable
        goal: Goal position (x, y)

    Returns:
        Tuple of (optimal_actions dict, distances dict)
    """
    height = len(grid_layout)
    width = len(grid_layout[0]) if height > 0 else 0

    def is_passable(x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        return grid_layout[y][x] != "#"

    # (dx, dy, action_id): LEFT, RIGHT, UP, DOWN
    neighbors = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]

    # Backward Dijkstra from goal
    distances: dict[tuple[int, int], int] = {goal: 0}
    heap_queue: list[tuple[int, tuple[int, int]]] = [(0, goal)]

    while heap_queue:
        dist, (x, y) = heapq.heappop(heap_queue)
        if dist > distances.get((x, y), float("inf")):
            continue

        for dx, dy, _ in neighbors:
            nx, ny = x + dx, y + dy
            if is_passable(nx, ny):
                new_dist = dist + 1
                if new_dist < distances.get((nx, ny), float("inf")):
                    distances[(nx, ny)] = new_dist
                    heapq.heappush(heap_queue, (new_dist, (nx, ny)))

    # Determine optimal actions for each cell
    optimal_actions: dict[tuple[int, int], OptimalActionSet] = {}

    for y in range(height):
        for x in range(width):
            if not is_passable(x, y):
                continue

            current_dist = distances.get((x, y), float("inf"))
            if current_dist == float("inf"):
                continue

            optimal_set: OptimalActionSet = set()
            for dx, dy, action in neighbors:
                nx, ny = x + dx, y + dy
                if is_passable(nx, ny):
                    neighbor_dist = distances.get((nx, ny), float("inf"))
                    if neighbor_dist == current_dist - 1:
                        optimal_set.add(action)

            optimal_actions[(x, y)] = optimal_set

    # Goal cell has no optimal actions
    optimal_actions[goal] = set()

    return optimal_actions, distances


def compute_trajectory_action_accuracy(
    trajectory: LightweightTrajectory,
    optimal_actions: dict[tuple[int, int], OptimalActionSet],
) -> float:
    """Compute action accuracy for a single trajectory.

    Acc(τ) = (1/T) * Σ_{t=0}^{T-1} 1(a^t ∈ π*(s^t))
    """
    if not trajectory.steps:
        return 0.0

    correct = 0
    for step in trajectory.steps:
        pos = step.agent_position
        action_id = ACTION_NAME_TO_ID.get(step.agent_action.upper())
        optimal_set = optimal_actions.get(pos, set())

        if action_id is not None and action_id in optimal_set:
            correct += 1

    return correct / len(trajectory.steps)


def compute_spl(
    trajectories: list[LightweightTrajectory],
    optimal_path_length: int,
) -> float:
    """Compute Success weighted by Path Length (SPL).

    SPL = (1/N) * Σ (1_S * L*) / max(L*, L)

    Where:
    - 1_S = 1 if trajectory reached goal, 0 otherwise
    - L* = optimal path length
    - L = actual trajectory length
    """
    if not trajectories or optimal_path_length <= 0:
        return 0.0

    total = 0.0
    for traj in trajectories:
        if traj.reached_goal:
            traj_length = traj.trajectory_length
            total += optimal_path_length / max(optimal_path_length, traj_length)

    return total / len(trajectories)


# =============================================================================
# File Parsing Utilities
# =============================================================================


def sanitize_label(value: str) -> str:
    """Sanitize a string for use as a file/directory label."""
    if not value:
        return "model"
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def parse_filename(filepath: Path) -> tuple[int, float, int]:
    """Extract grid_size, complexity, and instance_id from filename.

    Expected format: grid_size{N}_complexity{X.XX}_{NNNN}_metadata.json
    """
    name = filepath.stem
    parts = name.split("_")
    try:
        grid_size = int(parts[1].replace("size", ""))
        complexity = float(parts[2].replace("complexity", ""))
        instance_id = int(parts[3])
        return grid_size, complexity, instance_id
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid filename format: {filepath.name}") from e


def grid_id_from_parts(grid_size: int, complexity: float, instance_id: int) -> str:
    """Construct grid ID from components."""
    return f"grid_size{grid_size}_complexity{complexity:.2f}_{instance_id:04d}"


def parse_filename_isotransform(filepath: Path) -> tuple[int, float, int, str]:
    """Extract grid_size, complexity, instance_id, and transform_type from filename.

    Expected formats:
    - With transform: grid_size{N}_complexity{X.XX}_{NNNN}_{TransformType}_metadata.json
    - Baseline: grid_size{N}_complexity{X.XX}_{NNNN}_metadata.json

    Args:
        filepath: Path to the metadata file

    Returns:
        Tuple of (grid_size, complexity, instance_id, transform_type)
    """
    name = filepath.stem
    parts = name.split("_")
    try:
        grid_size = int(parts[1].replace("size", ""))
        complexity = float(parts[2].replace("complexity", ""))
        instance_id = int(parts[3])
        # Transform type is the 5th part (index 4), but only if it's not "metadata"
        # Baseline files end with {instance_id}_metadata.json, so parts[4] == "metadata"
        # Transform files end with {TransformType}_metadata.json, so parts[4] is the transform
        if len(parts) > 5 and parts[4] != "metadata":
            transform_type = parts[4]
        else:
            transform_type = "baseline"
        return grid_size, complexity, instance_id, transform_type
    except (IndexError, ValueError) as e:
        raise ValueError(
            f"Invalid isotransform filename format: {filepath.name}"
        ) from e


def grid_id_from_parts_isotransform(
    grid_size: int, complexity: float, instance_id: int, transform_type: str
) -> str:
    """Construct grid ID with transform type from components.

    For baseline grids, the grid ID does NOT include a transform suffix.
    For transformed grids, the transform type is appended.
    """
    base_id = f"grid_size{grid_size}_complexity{complexity:.2f}_{instance_id:04d}"
    if transform_type == "baseline":
        return base_id
    return f"{base_id}_{transform_type}"


# =============================================================================
# Optimal Actions Computation
# =============================================================================


def compute_optimal_actions(env: Any) -> list[list[OptimalActionSet]]:
    """Compute the set of optimal actions for each cell in the grid.

    Uses backward Dijkstra from the goal to compute shortest distances,
    then determines which actions reduce the distance to goal.

    Args:
        env: MiniGrid environment instance

    Returns:
        2D list where each cell contains the set of optimal action IDs
    """
    base_env = getattr(env, "unwrapped", env)
    grid = base_env.grid
    goal: GridCoord = tuple(base_env.goal_pos)
    width, height = grid.width, grid.height

    def is_passable(x: int, y: int) -> bool:
        """Check if a position is within bounds and not blocked."""
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        cell = grid.get(x, y)
        return (cell is None) or (getattr(cell, "can_overlap", lambda: False)())

    # (dx, dy, action_id): LEFT, RIGHT, UP, DOWN
    neighbors = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]

    # Backward Dijkstra from goal
    distances: dict[GridCoord, float] = {goal: 0}
    heap: list[tuple[float, GridCoord]] = [(0, goal)]

    while heap:
        dist, (x, y) = heapq.heappop(heap)
        if dist > distances.get((x, y), float("inf")):
            continue

        for dx, dy, _ in neighbors:
            nx, ny = x + dx, y + dy
            if is_passable(nx, ny):
                new_dist = dist + 1
                if new_dist < distances.get((nx, ny), float("inf")):
                    distances[(nx, ny)] = new_dist
                    heapq.heappush(heap, (new_dist, (nx, ny)))

    # Determine optimal actions for each cell
    optimal_actions: list[list[OptimalActionSet]] = [
        [set() for _ in range(width)] for _ in range(height)
    ]

    for y in range(height):
        for x in range(width):
            if not is_passable(x, y):
                continue

            current_dist = distances.get((x, y), float("inf"))
            if current_dist == float("inf"):
                continue

            for dx, dy, action in neighbors:
                nx, ny = x + dx, y + dy
                if is_passable(nx, ny):
                    neighbor_dist = distances.get((nx, ny), float("inf"))
                    if neighbor_dist == current_dist - 1:
                        optimal_actions[y][x].add(action)

    # Goal cell has no optimal actions (already at goal)
    gx, gy = goal
    optimal_actions[gy][gx] = set()

    return optimal_actions


def compute_optimal_actions_and_distances(
    env: Any,
) -> tuple[list[list[OptimalActionSet]], list[list[int]]]:
    """Compute optimal actions and distances to goal for each cell.

    This is an extended version of compute_optimal_actions that also returns
    the shortest path distance from each cell to the goal.

    Args:
        env: MiniGrid environment instance

    Returns:
        Tuple of:
        - 2D list of optimal action sets per cell
        - 2D list of distances to goal per cell (-1 for walls/unreachable)
    """
    base_env = getattr(env, "unwrapped", env)
    grid = base_env.grid
    goal: GridCoord = tuple(base_env.goal_pos)
    width, height = grid.width, grid.height

    def is_passable(x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        cell = grid.get(x, y)
        return (cell is None) or (getattr(cell, "can_overlap", lambda: False)())

    neighbors = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]

    # Backward Dijkstra from goal
    distances: dict[GridCoord, int] = {goal: 0}
    heap: list[tuple[int, GridCoord]] = [(0, goal)]

    while heap:
        dist, (x, y) = heapq.heappop(heap)
        if dist > distances.get((x, y), float("inf")):
            continue

        for dx, dy, _ in neighbors:
            nx, ny = x + dx, y + dy
            if is_passable(nx, ny):
                new_dist = dist + 1
                if new_dist < distances.get((nx, ny), float("inf")):
                    distances[(nx, ny)] = new_dist
                    heapq.heappush(heap, (new_dist, (nx, ny)))

    # Build optimal actions and distance grids
    optimal_actions: list[list[OptimalActionSet]] = [
        [set() for _ in range(width)] for _ in range(height)
    ]
    distance_grid: list[list[int]] = [[-1 for _ in range(width)] for _ in range(height)]

    for y in range(height):
        for x in range(width):
            if not is_passable(x, y):
                continue

            current_dist = distances.get((x, y), float("inf"))
            if current_dist == float("inf"):
                continue

            distance_grid[y][x] = int(current_dist)

            for dx, dy, action in neighbors:
                nx, ny = x + dx, y + dy
                if is_passable(nx, ny):
                    neighbor_dist = distances.get((nx, ny), float("inf"))
                    if neighbor_dist == current_dist - 1:
                        optimal_actions[y][x].add(action)

    # Goal cell
    gx, gy = goal
    optimal_actions[gy][gx] = set()
    distance_grid[gy][gx] = 0

    return optimal_actions, distance_grid


def compute_optimal_distribution(optimal_actions: OptimalActionSet) -> ActionDist:
    """Compute uniform distribution over optimal actions.

    Args:
        optimal_actions: Set of optimal action IDs

    Returns:
        Dictionary mapping action IDs to probabilities (uniform over optimal actions)
    """
    if not optimal_actions:
        return {aid: 0.0 for aid in ACTION_ID_TO_NAME}

    prob = 1.0 / len(optimal_actions)
    return {aid: prob if aid in optimal_actions else 0.0 for aid in ACTION_ID_TO_NAME}


# =============================================================================
# Logprob Parsing
# =============================================================================


def normalize_action_token(token: Optional[str]) -> str:
    """Normalize an action token string to uppercase."""
    if token is None:
        return ""
    return token.strip().strip('"').strip("'").upper()


def is_action_token(token: str) -> bool:
    """Check if a token is a valid action name."""
    normalized = normalize_action_token(token)
    return normalized in ACTION_NAMES_UPPER


def find_action_token_entry(logprobs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Find the token entry corresponding to the action value in logprobs."""
    if not logprobs:
        return None

    n = len(logprobs)

    # Strategy 1: Structured search for "action": "<VALUE>" pattern
    for i, entry in enumerate(logprobs):
        token = entry.get("token", "")

        if "action" not in token.lower():
            continue

        # Look for colon after the "action" key
        colon_idx = None
        for j in range(i + 1, min(i + 5, n)):
            if ":" in logprobs[j].get("token", ""):
                colon_idx = j
                break

        if colon_idx is None:
            continue

        # Search for action value after colon
        for k in range(colon_idx + 1, min(colon_idx + 5, n)):
            candidate_token = logprobs[k].get("token", "")

            if candidate_token.strip() in ("", '"', "'"):
                continue

            if is_action_token(candidate_token):
                return logprobs[k]

            if "}" in candidate_token:
                break

    # Strategy 2: Fallback - find any high-confidence standalone action token
    for entry in logprobs:
        token = entry.get("token", "")
        if is_action_token(token):
            logprob = entry.get("logprob")
            if logprob is not None and logprob > -1.0:
                return entry

    return None


def distribution_from_logprobs(
    logprobs: Optional[list[dict[str, Any]]],
) -> Optional[ActionDist]:
    """Extract action probability distribution from logprobs.

    Args:
        logprobs: List of token logprob entries from the LLM

    Returns:
        Dictionary mapping action IDs to probabilities, or None if parsing fails
    """
    if not logprobs:
        return None

    token_entry = find_action_token_entry(logprobs)
    if not token_entry:
        return None

    entries: dict[ActionID, float] = {}

    def register_action(token_value: Optional[str], logprob: Optional[float]) -> None:
        action = normalize_action_token(token_value)
        if action in ACTION_NAME_TO_ID and logprob is not None:
            action_id = ACTION_NAME_TO_ID[action]
            entries[action_id] = max(entries.get(action_id, -math.inf), logprob)

    register_action(token_entry.get("token"), token_entry.get("logprob"))

    for candidate in token_entry.get("top_logprobs") or []:
        register_action(candidate.get("token"), candidate.get("logprob"))

    if not entries:
        return None

    # Convert logprobs to probabilities using numerically stable softmax
    max_logprob = max(entries.values())
    probs = {aid: math.exp(lp - max_logprob) for aid, lp in entries.items()}

    total = sum(probs.values())
    if total <= 0:
        return None

    return {aid: probs.get(aid, 0.0) / total for aid in ACTION_ID_TO_NAME}


# =============================================================================
# Entropy and Information Theory Metrics
# =============================================================================


def shannon_entropy(dist: ActionDist) -> float:
    """Compute Shannon entropy of a probability distribution.

    H(X) = -sum_i p_i * log2(p_i)

    Args:
        dist: Dictionary mapping action IDs to probabilities

    Returns:
        Entropy in bits
    """
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def cross_entropy(
    optimal_actions: OptimalActionSet, dist: ActionDist, eps: float = LOGPROB_EPS
) -> Optional[float]:
    """Compute cross-entropy between uniform optimal and model distributions.

    H(p_opt, q_model) = -sum_a p_opt(a) * log2(q_model(a))

    Args:
        optimal_actions: Set of optimal action IDs
        dist: Model's action probability distribution
        eps: Small epsilon to avoid log(0)

    Returns:
        Cross-entropy in bits, or None if no optimal actions
    """
    if not optimal_actions:
        return None

    weight = 1.0 / len(optimal_actions)
    ce = 0.0
    for action in optimal_actions:
        model_prob = max(dist.get(action, 0.0), eps)
        ce -= weight * math.log2(model_prob)
    return ce


def compute_optimal_mass(optimal_actions: OptimalActionSet, dist: ActionDist) -> float:
    """Compute total probability mass assigned to optimal actions."""
    return sum(dist.get(action, 0.0) for action in optimal_actions)


def optimal_entropy(num_optimal_actions: int) -> float:
    """Entropy of uniform distribution over optimal actions: H = log2(k)."""
    if num_optimal_actions <= 0:
        return 0.0
    return math.log2(num_optimal_actions)


def kl_divergence(
    optimal_actions: OptimalActionSet, dist: ActionDist, eps: float = LOGPROB_EPS
) -> Optional[float]:
    """Compute KL divergence from optimal to model distribution.

    D_KL(optimal || model) = H(optimal, model) - H(optimal)

    This measures the "excess" bits needed beyond the optimal entropy.
    If the model perfectly matches optimal, KL = 0.

    Args:
        optimal_actions: Set of optimal action IDs
        dist: Model's action probability distribution
        eps: Small epsilon to avoid log(0)

    Returns:
        KL divergence in bits, or None if no optimal actions
    """
    ce = cross_entropy(optimal_actions, dist, eps)
    if ce is None:
        return None
    h_opt = optimal_entropy(len(optimal_actions))
    return ce - h_opt


def jensen_shannon_divergence(
    optimal_actions: OptimalActionSet, dist: ActionDist, eps: float = LOGPROB_EPS
) -> Optional[float]:
    """Compute Jensen-Shannon divergence between optimal and model distributions.

    JSD(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
    where M = 0.5 * (P + Q)

    JSD is symmetric and bounded in [0, 1] when using log base 2.
    JSD = 0 means identical distributions, JSD = 1 means completely different.

    Args:
        optimal_actions: Set of optimal action IDs
        dist: Model's action probability distribution
        eps: Small epsilon for numerical stability

    Returns:
        JSD in [0, 1], or None if no optimal actions
    """
    if not optimal_actions:
        return None

    # Compute uniform optimal distribution
    num_optimal = len(optimal_actions)
    p_opt = 1.0 / num_optimal

    # Compute mixture distribution M = 0.5 * (P + Q)
    # and KL divergences
    kl_p_m = 0.0  # D_KL(optimal || M)
    kl_q_m = 0.0  # D_KL(model || M)

    for action_id in ACTION_ID_TO_NAME:
        # Optimal distribution: uniform over optimal actions
        p = p_opt if action_id in optimal_actions else 0.0
        # Model distribution
        q = dist.get(action_id, 0.0)
        # Mixture
        m = 0.5 * (p + q)

        # Add to KL(P || M)
        if p > 0 and m > eps:
            kl_p_m += p * math.log2(p / m)

        # Add to KL(Q || M)
        if q > 0 and m > eps:
            kl_q_m += q * math.log2(q / m)

    jsd = 0.5 * kl_p_m + 0.5 * kl_q_m
    return jsd


# =============================================================================
# Data Loading
# =============================================================================


def load_environments(dataset_path: str) -> dict[str, Any]:
    """Load grid environments from pickle file."""
    with open(dataset_path, "rb") as f:
        return pickle.load(f)


def discover_metadata_files(metadata_dir: Path) -> list[Path]:
    """Find all metadata JSON files in a directory."""
    return sorted(metadata_dir.glob("*_metadata.json"))


def _load_single_metadata(fpath: Path) -> Optional[tuple[str, GridMetadata]]:
    """Load a single metadata file."""
    try:
        grid_size, complexity, instance_id = parse_filename(fpath)
        key = grid_id_from_parts(grid_size, complexity, instance_id)

        with open(fpath, "r") as f:
            policy_metadata = json.load(f)

        return key, GridMetadata(
            grid_size=grid_size,
            complexity=complexity,
            instance_id=instance_id,
            policy_metadata=policy_metadata,
        )
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Warning: Skipping invalid file {fpath.name}: {e}")
        return None


def load_metadata_batch(
    metadata_files: list[Path], max_workers: int = 8, show_progress: bool = False
) -> dict[str, GridMetadata]:
    """Load multiple metadata files in parallel."""
    metadata_dict: dict[str, GridMetadata] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_load_single_metadata, fpath): fpath
            for fpath in metadata_files
        }
        completed = as_completed(futures)
        if show_progress:
            completed = tqdm(
                completed, total=len(futures), desc="Loading metadata", leave=False
            )
        for future in completed:
            result = future.result()
            if result is not None:
                key, metadata = result
                metadata_dict[key] = metadata

    return metadata_dict


def load_single_grid_metadata(
    grid_id: str, metadata_dir: Path
) -> Optional[GridMetadata]:
    """Load metadata for a specific grid by ID.

    Args:
        grid_id: Grid identifier (e.g., "grid_size5_complexity0.30_0001")
        metadata_dir: Directory containing metadata files

    Returns:
        GridMetadata instance or None if not found
    """
    metadata_file = metadata_dir / f"{grid_id}_metadata.json"
    if not metadata_file.exists():
        return None

    result = _load_single_metadata(metadata_file)
    return result[1] if result else None


# =============================================================================
# Isotransform Metadata Loading
# =============================================================================


@dataclass
class IsotransformGridMetadata:
    """Container for a single isotransform grid's metadata."""

    grid_size: int
    complexity: float
    instance_id: int
    transform_type: str
    policy_metadata: list[list[dict]]


def _load_single_metadata_isotransform(
    fpath: Path,
) -> Optional[tuple[str, IsotransformGridMetadata]]:
    """Load a single isotransform metadata file."""
    try:
        grid_size, complexity, instance_id, transform_type = (
            parse_filename_isotransform(fpath)
        )
        key = grid_id_from_parts_isotransform(
            grid_size, complexity, instance_id, transform_type
        )

        with open(fpath, "r") as f:
            policy_metadata = json.load(f)

        return key, IsotransformGridMetadata(
            grid_size=grid_size,
            complexity=complexity,
            instance_id=instance_id,
            transform_type=transform_type,
            policy_metadata=policy_metadata,
        )
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Warning: Skipping invalid isotransform file {fpath.name}: {e}")
        return None


def load_metadata_batch_isotransform(
    metadata_files: list[Path], max_workers: int = 8, show_progress: bool = False
) -> dict[str, IsotransformGridMetadata]:
    """Load multiple isotransform metadata files in parallel.

    Args:
        metadata_files: List of paths to metadata files
        max_workers: Number of parallel workers
        show_progress: Whether to show progress bar

    Returns:
        Dictionary mapping grid_id (with transform) to metadata
    """
    metadata_dict: dict[str, IsotransformGridMetadata] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_load_single_metadata_isotransform, fpath): fpath
            for fpath in metadata_files
        }
        completed = as_completed(futures)
        if show_progress:
            completed = tqdm(
                completed, total=len(futures), desc="Loading metadata", leave=False
            )
        for future in completed:
            result = future.result()
            if result is not None:
                key, metadata = result
                metadata_dict[key] = metadata

    return metadata_dict


# =============================================================================
# Cell and Grid Processing
# =============================================================================


def process_cell(
    cell: Any,
    optimal_set: OptimalActionSet,
    grid_id: str,
    metadata: GridMetadata,
    x: int,
    y: int,
    distance_to_goal: int = 0,
) -> Optional[CellMetrics]:
    """Process a single cell and compute its metrics."""
    if not isinstance(cell, dict):
        return None

    if not optimal_set:
        return None

    dist = distribution_from_logprobs(cell.get("logprobs"))
    if dist is None:
        return None

    num_optimal = len(optimal_set)
    entropy_bits = shannon_entropy(dist)
    optimal_entropy_bits = optimal_entropy(num_optimal)
    cross_entropy_bits = cross_entropy(optimal_set, dist)
    jsd_value = jensen_shannon_divergence(optimal_set, dist)
    optimal_mass_val = compute_optimal_mass(optimal_set, dist)
    llm_action = cell.get("llm_response", -1)

    action_probs = {
        ACTION_ID_TO_NAME[aid].lower(): dist.get(aid, 0.0) for aid in ACTION_ID_TO_NAME
    }

    return CellMetrics(
        grid_id=grid_id,
        grid_size=metadata.grid_size,
        complexity=metadata.complexity,
        instance_id=metadata.instance_id,
        x=x,
        y=y,
        llm_action=llm_action,
        num_optimal_actions=num_optimal,
        entropy_bits=entropy_bits,
        optimal_entropy_bits=optimal_entropy_bits,
        cross_entropy_bits=cross_entropy_bits,
        jsd=jsd_value,
        optimal_mass=optimal_mass_val,
        is_action_optimal=int(llm_action in optimal_set),
        action_probs=action_probs,
        distance_to_goal=distance_to_goal,
    )


def process_grid(grid_id: str, env: Any, metadata: GridMetadata) -> list[CellMetrics]:
    """Process all cells in a grid and return their metrics."""
    results: list[CellMetrics] = []
    optimal_actions, distance_grid = compute_optimal_actions_and_distances(env)

    for y, row in enumerate(metadata.policy_metadata):
        for x, cell in enumerate(row):
            cell_result = process_cell(
                cell=cell,
                optimal_set=optimal_actions[y][x],
                grid_id=grid_id,
                metadata=metadata,
                x=x,
                y=y,
                distance_to_goal=distance_grid[y][x],
            )
            if cell_result:
                results.append(cell_result)

    return results


def compute_grid_mean_cross_entropy(
    grid_id: str, env: Any, metadata: GridMetadata
) -> Optional[float]:
    """Compute mean cross-entropy across all cells in a grid.

    Args:
        grid_id: Grid identifier
        env: MiniGrid environment instance
        metadata: Grid metadata with policy information

    Returns:
        Mean cross-entropy in bits, or None if no valid cells
    """
    cell_metrics = process_grid(grid_id, env, metadata)
    valid_ce = [
        m.cross_entropy_bits for m in cell_metrics if m.cross_entropy_bits is not None
    ]

    if not valid_ce:
        return None

    return sum(valid_ce) / len(valid_ce)


def compute_grid_mean_jsd(
    grid_id: str, env: Any, metadata: GridMetadata
) -> Optional[float]:
    """Compute mean Jensen-Shannon divergence across all cells in a grid.

    JSD is bounded [0, 1] and symmetric, making it easier to compare across
    grids with different numbers of optimal actions per cell.

    Args:
        grid_id: Grid identifier
        env: MiniGrid environment instance
        metadata: Grid metadata with policy information

    Returns:
        Mean JSD in [0, 1], or None if no valid cells
    """
    optimal_actions_grid = compute_optimal_actions(env)
    jsd_values: list[float] = []

    for y, row in enumerate(metadata.policy_metadata):
        for x, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue

            optimal_set = optimal_actions_grid[y][x]
            if not optimal_set:
                continue

            dist = distribution_from_logprobs(cell.get("logprobs"))
            if dist is None:
                continue

            jsd = jensen_shannon_divergence(optimal_set, dist)
            if jsd is not None:
                jsd_values.append(jsd)

    if not jsd_values:
        return None

    return sum(jsd_values) / len(jsd_values)


# =============================================================================
# Statistical Analysis Utilities
# =============================================================================


@dataclass
class CorrelationResult:
    """Result of a correlation computation."""

    r: float
    n: int
    p_value: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"r": self.r, "n": self.n, "p_value": self.p_value}


@dataclass
class RegressionResult:
    """Result of a regression analysis."""

    coefficients: dict[str, float]
    p_values: dict[str, float]
    r_squared: float
    adj_r_squared: float
    n: int
    residuals: Optional[np.ndarray] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excluding residuals)."""
        return {
            "coefficients": self.coefficients,
            "p_values": self.p_values,
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "n": self.n,
        }


@dataclass
class ControlledAnalysisResult:
    """Container for controlled analysis results."""

    raw_correlations: dict[str, CorrelationResult]
    within_stratum_correlations: dict[str, CorrelationResult]
    partial_correlations: dict[str, CorrelationResult]
    regression: Optional[RegressionResult] = None
    stratified_summary: Optional[pd.DataFrame] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "raw_correlations": {
                k: v.to_dict() for k, v in self.raw_correlations.items()
            },
            "within_stratum_correlations": {
                k: v.to_dict() for k, v in self.within_stratum_correlations.items()
            },
            "partial_correlations": {
                k: v.to_dict() for k, v in self.partial_correlations.items()
            },
            "regression": self.regression.to_dict() if self.regression else None,
        }


def compute_correlation(
    x: pd.Series, y: pd.Series, min_samples: int = 10
) -> Optional[CorrelationResult]:
    """Compute Pearson correlation with p-value.

    Args:
        x: First variable
        y: Second variable
        min_samples: Minimum samples required for valid correlation

    Returns:
        CorrelationResult or None if insufficient data
    """
    from scipy import stats

    # Drop NaN values
    mask = x.notna() & y.notna()
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < min_samples:
        return None

    r, p = stats.pearsonr(x_clean, y_clean)
    return CorrelationResult(r=r, n=len(x_clean), p_value=p)


def compute_correlations_for_columns(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    min_samples: int = 10,
) -> dict[str, CorrelationResult]:
    """Compute correlations between one column and multiple target columns.

    Args:
        df: DataFrame containing the data
        x_col: Name of the independent variable column
        y_cols: Names of the dependent variable columns
        min_samples: Minimum samples required

    Returns:
        Dictionary mapping column names to CorrelationResults
    """
    results = {}
    for y_col in y_cols:
        if y_col not in df.columns:
            continue
        corr = compute_correlation(df[x_col], df[y_col], min_samples)
        if corr is not None:
            results[y_col] = corr
    return results


def compute_within_stratum_correlations(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    strata_cols: list[str],
    min_stratum_size: int = 10,
    min_strata: int = 3,
) -> dict[str, CorrelationResult]:
    """Compute correlations within each stratum, then aggregate.

    This controls for confounding by computing correlations separately within
    each unique combination of strata_cols, then averaging across strata
    (weighted by stratum size).

    Args:
        df: DataFrame containing the data
        x_col: Name of the independent variable column
        y_cols: Names of the dependent variable columns
        strata_cols: Columns defining the strata (e.g., ["grid_size", "complexity"])
        min_stratum_size: Minimum samples per stratum to include
        min_strata: Minimum number of valid strata required

    Returns:
        Dictionary mapping column names to aggregated CorrelationResults
    """
    stratum_correlations: dict[str, list[tuple[float, int]]] = {y: [] for y in y_cols}

    for _, group in df.groupby(strata_cols):
        if len(group) < min_stratum_size:
            continue

        for y_col in y_cols:
            if y_col not in group.columns:
                continue
            corr = compute_correlation(group[x_col], group[y_col], min_stratum_size)
            if corr is not None and not np.isnan(corr.r):
                stratum_correlations[y_col].append((corr.r, corr.n))

    # Aggregate: weighted average by stratum size
    results = {}
    for y_col, corr_list in stratum_correlations.items():
        if len(corr_list) < min_strata:
            continue

        total_n = sum(n for _, n in corr_list)
        weighted_r = sum(r * n for r, n in corr_list) / total_n
        results[y_col] = CorrelationResult(r=weighted_r, n=total_n, p_value=None)

    return results


def residualize(y: np.ndarray, X: np.ndarray, add_intercept: bool = True) -> np.ndarray:
    """Compute residuals from OLS regression of y on X.

    Args:
        y: Dependent variable (n,)
        X: Independent variables (n, k)
        add_intercept: Whether to add an intercept column

    Returns:
        Residuals from the regression
    """
    if add_intercept:
        X = np.column_stack([np.ones(len(X)), X])

    # OLS: beta = (X'X)^-1 X'y
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred = X @ beta
        return y - y_pred
    except np.linalg.LinAlgError:
        return y  # Return original if regression fails


def compute_partial_correlations(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    control_cols: list[str],
    min_samples: int = 30,
) -> dict[str, CorrelationResult]:
    """Compute partial correlations controlling for specified variables.

    Uses residualization: regress both x and y on controls, then correlate residuals.

    Args:
        df: DataFrame containing the data
        x_col: Name of the independent variable column
        y_cols: Names of the dependent variable columns
        control_cols: Columns to control for
        min_samples: Minimum samples required

    Returns:
        Dictionary mapping column names to partial CorrelationResults
    """
    from scipy import stats

    # Build control matrix
    all_cols = [x_col] + y_cols + control_cols
    df_clean = df[all_cols].dropna()

    if len(df_clean) < min_samples:
        return {}

    controls = df_clean[control_cols].values
    x_resid = residualize(df_clean[x_col].values, controls)

    results = {}
    for y_col in y_cols:
        y_resid = residualize(df_clean[y_col].values, controls)
        r, p = stats.pearsonr(x_resid, y_resid)
        results[y_col] = CorrelationResult(r=r, n=len(df_clean), p_value=p)

    return results


def run_ols_regression(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    min_samples: int = 30,
) -> Optional[RegressionResult]:
    """Run OLS regression with multiple predictors.

    Args:
        df: DataFrame containing the data
        y_col: Dependent variable column name
        x_cols: Independent variable column names
        min_samples: Minimum samples required

    Returns:
        RegressionResult or None if insufficient data
    """
    from scipy import stats

    all_cols = [y_col] + x_cols
    df_clean = df[all_cols].dropna()

    if len(df_clean) < min_samples:
        return None

    y = df_clean[y_col].values
    X = df_clean[x_cols].values
    X_with_intercept = np.column_stack([np.ones(len(X)), X])

    n, k = X_with_intercept.shape

    try:
        # OLS estimation
        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        y_pred = X_with_intercept @ beta
        residuals = y - y_pred

        # Statistics
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)

        # Standard errors and p-values
        mse = ss_res / (n - k)
        try:
            var_beta = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            se_beta = np.sqrt(np.diag(var_beta))
            t_stats = beta / se_beta
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))
        except np.linalg.LinAlgError:
            p_values = np.full(k, np.nan)

        # Build result
        coef_names = ["intercept"] + x_cols
        coefficients = dict(zip(coef_names, beta))
        p_value_dict = dict(zip(coef_names, p_values))

        return RegressionResult(
            coefficients=coefficients,
            p_values=p_value_dict,
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            n=n,
            residuals=residuals,
        )

    except np.linalg.LinAlgError:
        return None


def compute_stratified_summary(
    df: pd.DataFrame,
    strata_cols: list[str],
    agg_config: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Compute summary statistics stratified by specified columns.

    Args:
        df: DataFrame containing the data
        strata_cols: Columns to group by
        agg_config: Dictionary mapping output column names to (input_col, agg_func) tuples
            e.g., {"mean_entropy": ("entropy_bits", "mean"), "count": ("entropy_bits", "count")}

    Returns:
        DataFrame with stratified summary statistics
    """
    agg_dict = {
        out_col: (in_col, func) for out_col, (in_col, func) in agg_config.items()
    }
    return df.groupby(strata_cols).agg(**agg_dict).reset_index()


def run_controlled_analysis(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    control_cols: list[str],
    min_samples: int = 30,
    min_stratum_size: int = 10,
) -> ControlledAnalysisResult:
    """Run complete controlled analysis: raw, within-stratum, and partial correlations.

    This is the main entry point for controlled analysis, computing:
    1. Raw (unadjusted) correlations
    2. Within-stratum correlations (stratified by control_cols)
    3. Partial correlations (residualized on control_cols)
    4. OLS regression with controls

    Args:
        df: DataFrame containing the data
        x_col: Independent variable column name
        y_cols: Dependent variable column names
        control_cols: Columns to control for
        min_samples: Minimum samples for regression/partial correlations
        min_stratum_size: Minimum samples per stratum

    Returns:
        ControlledAnalysisResult containing all computed statistics
    """
    # 1. Raw correlations
    raw_corrs = compute_correlations_for_columns(df, x_col, y_cols)

    # 2. Within-stratum correlations
    within_stratum_corrs = compute_within_stratum_correlations(
        df, x_col, y_cols, control_cols, min_stratum_size=min_stratum_size
    )

    # 3. Partial correlations
    partial_corrs = compute_partial_correlations(
        df, x_col, y_cols, control_cols, min_samples=min_samples
    )

    # 4. Regression for primary outcome (first y_col)
    regression = None
    if y_cols:
        primary_y = y_cols[0]
        regression = run_ols_regression(
            df, primary_y, [x_col] + control_cols, min_samples=min_samples
        )

    return ControlledAnalysisResult(
        raw_correlations=raw_corrs,
        within_stratum_correlations=within_stratum_corrs,
        partial_correlations=partial_corrs,
        regression=regression,
    )


def format_correlation_report(
    result: ControlledAnalysisResult,
    x_label: str = "X",
    control_labels: Optional[list[str]] = None,
) -> str:
    """Format controlled analysis results as a readable report.

    Args:
        result: ControlledAnalysisResult to format
        x_label: Label for the independent variable
        control_labels: Labels for control variables

    Returns:
        Formatted string report
    """
    lines = []
    control_str = ", ".join(control_labels) if control_labels else "controls"

    lines.append(f"CORRELATION ANALYSIS: {x_label}")
    lines.append("=" * 60)

    # Raw correlations
    lines.append("\n1. RAW CORRELATIONS (no controls):")
    for y_col, corr in result.raw_correlations.items():
        p_str = f", p={corr.p_value:.4f}" if corr.p_value is not None else ""
        lines.append(f"   {y_col}: r={corr.r:.4f} (n={corr.n}){p_str}")

    # Within-stratum
    lines.append(f"\n2. WITHIN-STRATUM CORRELATIONS (stratified by {control_str}):")
    for y_col, corr in result.within_stratum_correlations.items():
        lines.append(f"   {y_col}: r={corr.r:.4f} (n={corr.n})")

    # Partial correlations
    lines.append(f"\n3. PARTIAL CORRELATIONS (controlling for {control_str}):")
    for y_col, corr in result.partial_correlations.items():
        p_str = f", p={corr.p_value:.4f}" if corr.p_value is not None else ""
        lines.append(f"   {y_col}: r={corr.r:.4f} (n={corr.n}){p_str}")

    # Regression
    if result.regression:
        lines.append(f"\n4. OLS REGRESSION (with {control_str} as controls):")
        lines.append(
            f"   R² = {result.regression.r_squared:.4f}, "
            f"Adj R² = {result.regression.adj_r_squared:.4f}, "
            f"n = {result.regression.n}"
        )
        lines.append("   Coefficients:")
        for var, coef in result.regression.coefficients.items():
            p_val = result.regression.p_values.get(var)
            p_str = f", p={p_val:.4f}" if p_val is not None else ""
            sig = "*" if p_val is not None and p_val < 0.05 else ""
            lines.append(f"      {var}: {coef:.4f}{p_str}{sig}")

    return "\n".join(lines)


# =============================================================================
# Calibration Metrics
# =============================================================================


@dataclass
class CalibrationMetrics:
    """Calibration metrics comparing LLM uncertainty to optimal uncertainty."""

    mean_entropy: float
    mean_optimal_entropy: float
    mean_divergence: float
    calibration_error: float  # Mean |H_llm - H_opt|
    calibration_bias: float  # Mean (H_llm - H_opt), + = overconfident
    entropy_correlation: float  # Correlation between H_llm and H_opt
    n_samples: int

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "mean_entropy": self.mean_entropy,
            "mean_optimal_entropy": self.mean_optimal_entropy,
            "mean_divergence": self.mean_divergence,
            "calibration_error": self.calibration_error,
            "calibration_bias": self.calibration_bias,
            "entropy_correlation": self.entropy_correlation,
            "n_samples": self.n_samples,
        }


def compute_calibration_metrics(
    df: pd.DataFrame,
    entropy_col: str = "entropy_bits",
    optimal_entropy_col: str = "optimal_entropy_bits",
    divergence_col: str = "cross_entropy_bits",
) -> CalibrationMetrics:
    """Compute calibration metrics from a DataFrame.

    Calibration measures how well the LLM's uncertainty aligns with
    the true uncertainty (number of optimal actions).

    Args:
        df: DataFrame with entropy columns
        entropy_col: Column name for LLM entropy
        optimal_entropy_col: Column name for optimal entropy
        divergence_col: Column name for divergence metric

    Returns:
        CalibrationMetrics with computed values
    """
    df_clean = df[[entropy_col, optimal_entropy_col, divergence_col]].dropna()

    if len(df_clean) == 0:
        return CalibrationMetrics(
            mean_entropy=0.0,
            mean_optimal_entropy=0.0,
            mean_divergence=0.0,
            calibration_error=0.0,
            calibration_bias=0.0,
            entropy_correlation=0.0,
            n_samples=0,
        )

    h_llm = df_clean[entropy_col]
    h_opt = df_clean[optimal_entropy_col]
    divergence = df_clean[divergence_col]

    # Calibration error: mean absolute difference
    calibration_error = (h_llm - h_opt).abs().mean()

    # Calibration bias: positive means under-confident (too much entropy)
    calibration_bias = (h_llm - h_opt).mean()

    # Correlation between LLM and optimal entropy
    entropy_correlation = h_llm.corr(h_opt)

    return CalibrationMetrics(
        mean_entropy=h_llm.mean(),
        mean_optimal_entropy=h_opt.mean(),
        mean_divergence=divergence.mean(),
        calibration_error=calibration_error,
        calibration_bias=calibration_bias,
        entropy_correlation=entropy_correlation,
        n_samples=len(df_clean),
    )


# =============================================================================
# Uncertainty-Accuracy Analysis
# =============================================================================


@dataclass
class UncertaintyAccuracyMetrics:
    """Metrics relating uncertainty to prediction accuracy."""

    accuracy: float  # Overall accuracy (is_action_optimal)
    mean_entropy_correct: float  # Mean entropy when correct
    mean_entropy_incorrect: float  # Mean entropy when incorrect
    entropy_gap: float  # Difference (incorrect - correct)
    auroc: Optional[float]  # AUROC: can entropy predict errors?
    ece: float  # Expected Calibration Error
    n_correct: int
    n_incorrect: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "mean_entropy_correct": self.mean_entropy_correct,
            "mean_entropy_incorrect": self.mean_entropy_incorrect,
            "entropy_gap": self.entropy_gap,
            "auroc": self.auroc,
            "ece": self.ece,
            "n_correct": self.n_correct,
            "n_incorrect": self.n_incorrect,
        }


def compute_uncertainty_accuracy_metrics(
    df: pd.DataFrame,
    entropy_col: str = "entropy_bits",
    correct_col: str = "is_action_optimal",
    n_bins: int = 10,
) -> UncertaintyAccuracyMetrics:
    """Compute metrics relating uncertainty to accuracy.

    This answers: "Does the model know what it doesn't know?"

    Args:
        df: DataFrame with entropy and correctness columns
        entropy_col: Column name for entropy
        correct_col: Column name for correctness (0/1)
        n_bins: Number of bins for ECE computation

    Returns:
        UncertaintyAccuracyMetrics
    """
    df_clean = df[[entropy_col, correct_col]].dropna()

    if len(df_clean) == 0:
        return UncertaintyAccuracyMetrics(
            accuracy=0.0,
            mean_entropy_correct=0.0,
            mean_entropy_incorrect=0.0,
            entropy_gap=0.0,
            auroc=None,
            ece=0.0,
            n_correct=0,
            n_incorrect=0,
        )

    correct_mask = df_clean[correct_col] == 1
    n_correct = correct_mask.sum()
    n_incorrect = (~correct_mask).sum()
    accuracy = n_correct / len(df_clean)

    # Mean entropy by correctness
    mean_entropy_correct = (
        df_clean.loc[correct_mask, entropy_col].mean() if n_correct > 0 else 0.0
    )
    mean_entropy_incorrect = (
        df_clean.loc[~correct_mask, entropy_col].mean() if n_incorrect > 0 else 0.0
    )
    entropy_gap = mean_entropy_incorrect - mean_entropy_correct

    # AUROC: Can entropy predict errors?
    auroc = None
    if n_correct > 0 and n_incorrect > 0:
        try:
            from sklearn.metrics import roc_auc_score

            # Higher entropy should predict errors (is_correct=0)
            # So we predict "error" (1-correct) from entropy
            auroc = roc_auc_score(1 - df_clean[correct_col], df_clean[entropy_col])
        except ImportError:
            # Fall back to manual computation
            auroc = _compute_auroc_manual(
                df_clean[entropy_col].values, df_clean[correct_col].values
            )

    # Expected Calibration Error (using confidence = 1 - normalized_entropy)
    # Normalize entropy to [0, 1] range (max entropy for 4 actions is 2 bits)
    max_entropy = 2.0  # log2(4) for 4 actions
    confidence = 1 - (df_clean[entropy_col] / max_entropy).clip(0, 1)
    ece = _compute_ece(confidence.values, df_clean[correct_col].values, n_bins)

    return UncertaintyAccuracyMetrics(
        accuracy=accuracy,
        mean_entropy_correct=mean_entropy_correct,
        mean_entropy_incorrect=mean_entropy_incorrect,
        entropy_gap=entropy_gap,
        auroc=auroc,
        ece=ece,
        n_correct=int(n_correct),
        n_incorrect=int(n_incorrect),
    )


def _compute_auroc_manual(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC manually (fallback if sklearn not available).

    Args:
        scores: Prediction scores (higher = more likely positive)
        labels: Binary labels (0 or 1), where 1 is the positive class

    Returns:
        AUROC value
    """
    # We want high entropy to predict errors (label=0)
    # So we predict (1 - labels) from scores
    labels = 1 - labels

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by scores descending
    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]

    # Count pairs where positive has higher score than negative
    tp_cumsum = np.cumsum(sorted_labels)

    # AUC = sum of (negative_i * cumulative_positives) / (n_pos * n_neg)
    auc = np.sum((1 - sorted_labels) * tp_cumsum) / (n_pos * n_neg)
    return float(auc)


def _compute_ece(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error.

    ECE measures the gap between confidence and accuracy across bins.

    Args:
        confidence: Predicted confidence [0, 1]
        correct: Binary correctness (0/1)
        n_bins: Number of confidence bins

    Returns:
        ECE value (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidence >= bin_boundaries[i]) & (
            confidence < bin_boundaries[i + 1]
        )
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = confidence[in_bin].mean()
            avg_accuracy = correct[in_bin].mean()
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_selective_prediction_curve(
    df: pd.DataFrame,
    entropy_col: str = "entropy_bits",
    correct_col: str = "is_action_optimal",
    n_thresholds: int = 20,
) -> pd.DataFrame:
    """Compute selective prediction curve (accuracy vs coverage).

    If we only predict when entropy is below a threshold, how does accuracy change?

    Args:
        df: DataFrame with entropy and correctness
        entropy_col: Column name for entropy
        correct_col: Column name for correctness
        n_thresholds: Number of threshold points

    Returns:
        DataFrame with columns: threshold, coverage, accuracy, n_samples
    """
    df_clean = df[[entropy_col, correct_col]].dropna()
    n_total = len(df_clean)

    if n_total == 0:
        return pd.DataFrame(columns=["threshold", "coverage", "accuracy", "n_samples"])

    # Generate thresholds spanning the entropy range
    min_ent = df_clean[entropy_col].min()
    max_ent = df_clean[entropy_col].max()
    thresholds = np.linspace(min_ent, max_ent, n_thresholds)

    rows = []
    for thresh in thresholds:
        mask = df_clean[entropy_col] <= thresh
        n_selected = mask.sum()
        coverage = n_selected / n_total
        accuracy = df_clean.loc[mask, correct_col].mean() if n_selected > 0 else 0.0

        rows.append(
            {
                "threshold": thresh,
                "coverage": coverage,
                "accuracy": accuracy,
                "n_samples": n_selected,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Distance-to-Goal Analysis
# =============================================================================


@dataclass
class DistanceToGoalMetrics:
    """Metrics relating uncertainty to distance from goal."""

    correlation_entropy_distance: float  # Corr(entropy, distance)
    correlation_divergence_distance: float  # Corr(divergence, distance)
    correlation_accuracy_distance: float  # Corr(is_correct, distance)
    mean_entropy_by_distance: dict[int, float]  # Mean entropy at each distance
    accuracy_by_distance: dict[int, float]  # Accuracy at each distance
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "correlation_entropy_distance": self.correlation_entropy_distance,
            "correlation_divergence_distance": self.correlation_divergence_distance,
            "correlation_accuracy_distance": self.correlation_accuracy_distance,
            "n_samples": self.n_samples,
        }


def compute_distance_to_goal_metrics(
    df: pd.DataFrame,
    entropy_col: str = "entropy_bits",
    divergence_col: str = "jsd",
    correct_col: str = "is_action_optimal",
    distance_col: str = "distance_to_goal",
) -> DistanceToGoalMetrics:
    """Compute metrics relating uncertainty to distance from goal.

    Args:
        df: DataFrame with required columns
        entropy_col: Column for entropy
        divergence_col: Column for divergence metric
        correct_col: Column for correctness
        distance_col: Column for distance to goal

    Returns:
        DistanceToGoalMetrics
    """
    required_cols = [entropy_col, divergence_col, correct_col, distance_col]
    df_clean = df[required_cols].dropna()
    df_clean = df_clean[df_clean[distance_col] >= 0]  # Exclude unreachable cells

    if len(df_clean) == 0:
        return DistanceToGoalMetrics(
            correlation_entropy_distance=0.0,
            correlation_divergence_distance=0.0,
            correlation_accuracy_distance=0.0,
            mean_entropy_by_distance={},
            accuracy_by_distance={},
            n_samples=0,
        )

    # Correlations
    corr_ent = df_clean[entropy_col].corr(df_clean[distance_col])
    corr_div = df_clean[divergence_col].corr(df_clean[distance_col])
    corr_acc = df_clean[correct_col].corr(df_clean[distance_col])

    # Group by distance
    grouped = df_clean.groupby(distance_col).agg(
        {
            entropy_col: "mean",
            correct_col: "mean",
        }
    )

    mean_entropy_by_distance = grouped[entropy_col].to_dict()
    accuracy_by_distance = grouped[correct_col].to_dict()

    return DistanceToGoalMetrics(
        correlation_entropy_distance=corr_ent,
        correlation_divergence_distance=corr_div,
        correlation_accuracy_distance=corr_acc,
        mean_entropy_by_distance=mean_entropy_by_distance,
        accuracy_by_distance=accuracy_by_distance,
        n_samples=len(df_clean),
    )


def compute_distance_summary(
    df: pd.DataFrame,
    entropy_col: str = "entropy_bits",
    divergence_col: str = "jsd",
    correct_col: str = "is_action_optimal",
    distance_col: str = "distance_to_goal",
    model_col: Optional[str] = None,
) -> pd.DataFrame:
    """Compute summary statistics by distance to goal.

    This function works for both single-model and multi-model DataFrames.

    Args:
        df: DataFrame with required columns
        entropy_col: Column for entropy
        divergence_col: Column for divergence
        correct_col: Column for correctness
        distance_col: Column for distance
        model_col: Optional column for model name (for multi-model analysis)

    Returns:
        DataFrame with summary by distance (and optionally model)
    """
    df_clean = df.copy()
    df_clean = df_clean[df_clean[distance_col] >= 0]

    group_cols = [distance_col]
    if model_col and model_col in df_clean.columns:
        group_cols = [model_col, distance_col]

    summary = (
        df_clean.groupby(group_cols)
        .agg(
            mean_entropy=(entropy_col, "mean"),
            std_entropy=(entropy_col, "std"),
            mean_divergence=(divergence_col, "mean"),
            accuracy=(correct_col, "mean"),
            n_samples=(entropy_col, "count"),
        )
        .reset_index()
    )

    return summary


# =============================================================================
# Isotransform Analysis Utilities
# =============================================================================

TRANSFORM_TYPES = [
    "baseline",
    "ReflectEnv",
    "RotateEnv",
    "StartGoalSwap",
    "TransposeEnv",
]


@dataclass
class TransformRegressionResult:
    """Result of OLS regression for transform effects."""

    outcome_variable: str
    coefficients: dict[str, float]
    std_errors: dict[str, float]
    p_values: dict[str, float]
    r_squared: float
    n_samples: int
    baseline_mean: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "outcome_variable": self.outcome_variable,
            "coefficients": self.coefficients,
            "std_errors": self.std_errors,
            "p_values": self.p_values,
            "r_squared": self.r_squared,
            "n_samples": self.n_samples,
            "baseline_mean": self.baseline_mean,
        }

    def get_significant_transforms(self, alpha: float = 0.05) -> list[str]:
        """Get list of transforms with p-value < alpha."""
        return [
            t
            for t, p in self.p_values.items()
            if t.startswith("transform_type") and p < alpha
        ]


def run_transform_regression(
    df: pd.DataFrame,
    outcome_col: str,
    transform_col: str = "transform_type",
    control_cols: Optional[list[str]] = None,
) -> Optional[TransformRegressionResult]:
    """Run OLS regression: outcome ~ transform_type + controls.

    Uses baseline as reference category for transform_type.

    Args:
        df: DataFrame with outcome and transform columns
        outcome_col: Column name for outcome variable (e.g., 'error_rate', 'entropy_bits')
        transform_col: Column name for transform type
        control_cols: Control variables (default: ['grid_size', 'complexity'])

    Returns:
        TransformRegressionResult or None if regression fails
    """
    from scipy import stats

    if control_cols is None:
        control_cols = ["grid_size", "complexity"]

    # Ensure baseline is reference category
    df = df.copy()
    if "baseline" not in df[transform_col].values:
        return None

    # Create dummy variables with baseline as reference
    transform_dummies = pd.get_dummies(
        df[transform_col], prefix="transform_type", drop_first=False
    )
    # Drop baseline column to make it reference
    if "transform_type_baseline" in transform_dummies.columns:
        transform_dummies = transform_dummies.drop("transform_type_baseline", axis=1)

    # Build design matrix
    X_cols = list(transform_dummies.columns) + control_cols
    X = pd.concat([transform_dummies, df[control_cols]], axis=1)
    X = X.dropna()

    # Ensure all values are float64 (dummies can be bool, controls can be object)
    X_values = X.values.astype(np.float64)
    X_with_intercept = np.column_stack([np.ones(len(X)), X_values])

    y = df.loc[X.index, outcome_col].values.astype(np.float64)

    if len(y) < len(X_cols) + 2:
        return None

    n, k = X_with_intercept.shape

    try:
        # OLS estimation
        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        y_pred = X_with_intercept @ beta
        residuals = y - y_pred

        # Statistics
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Standard errors and p-values
        mse = ss_res / (n - k)
        try:
            var_beta = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            se_beta = np.sqrt(np.diag(var_beta))
            t_stats = beta / se_beta
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))
        except np.linalg.LinAlgError:
            se_beta = np.full(k, np.nan)
            p_values = np.full(k, np.nan)

        # Build result
        coef_names = ["intercept"] + X_cols
        coefficients = dict(zip(coef_names, beta))
        std_errors = dict(zip(coef_names, se_beta))
        p_value_dict = dict(zip(coef_names, p_values))

        # Baseline mean
        baseline_mean = df[df[transform_col] == "baseline"][outcome_col].mean()

        return TransformRegressionResult(
            outcome_variable=outcome_col,
            coefficients=coefficients,
            std_errors=std_errors,
            p_values=p_value_dict,
            r_squared=r_squared,
            n_samples=n,
            baseline_mean=baseline_mean,
        )

    except np.linalg.LinAlgError:
        return None


def compute_transform_summary(
    df: pd.DataFrame,
    transform_col: str = "transform_type",
    metrics: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Compute summary statistics by transform type.

    Args:
        df: DataFrame with transform and metric columns
        transform_col: Column name for transform type
        metrics: List of metric columns to summarize

    Returns:
        DataFrame with summary by transform type
    """
    if metrics is None:
        metrics = ["entropy_bits", "jsd", "is_action_optimal"]

    agg_dict = {}
    for m in metrics:
        if m in df.columns:
            agg_dict[f"{m}_mean"] = (m, "mean")
            agg_dict[f"{m}_std"] = (m, "std")

    agg_dict["n_samples"] = (metrics[0], "count")

    return df.groupby(transform_col).agg(**agg_dict).reset_index()


def compute_transform_diff_from_baseline(
    df: pd.DataFrame,
    metric_col: str,
    transform_col: str = "transform_type",
    group_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Compute difference from baseline for each transform.

    Args:
        df: DataFrame with transform and metric columns
        metric_col: Column to compute difference for
        transform_col: Column name for transform type
        group_cols: Columns to group by when matching baseline
                   (default: ['grid_size', 'complexity', 'instance_id'])

    Returns:
        DataFrame with diff_from_baseline column
    """
    if group_cols is None:
        group_cols = ["grid_size", "complexity", "instance_id"]

    # Get baseline rows
    baseline = df[df[transform_col] == "baseline"].rename(
        columns={metric_col: "baseline_value"}
    )[group_cols + ["baseline_value"]]

    # Merge baseline back
    df_merged = df.merge(baseline, on=group_cols, how="inner")

    # Compute difference
    df_merged["diff_from_baseline"] = (
        df_merged[metric_col] - df_merged["baseline_value"]
    )

    return df_merged


def compute_distance_summary_by_transform(
    df: pd.DataFrame,
    entropy_col: str = "entropy_bits",
    divergence_col: str = "jsd",
    correct_col: str = "is_action_optimal",
    distance_col: str = "distance_to_goal",
    transform_col: str = "transform_type",
) -> pd.DataFrame:
    """Compute summary statistics by distance and transform type.

    Args:
        df: DataFrame with required columns
        entropy_col: Column for entropy
        divergence_col: Column for divergence
        correct_col: Column for correctness
        distance_col: Column for distance
        transform_col: Column for transform type

    Returns:
        DataFrame with summary by distance and transform
    """
    df_clean = df.copy()
    df_clean = df_clean[df_clean[distance_col] >= 0]

    summary = (
        df_clean.groupby([transform_col, distance_col])
        .agg(
            mean_entropy=(entropy_col, "mean"),
            std_entropy=(entropy_col, "std"),
            mean_divergence=(divergence_col, "mean"),
            std_divergence=(divergence_col, "std"),
            accuracy=(correct_col, "mean"),
            error_rate=(correct_col, lambda x: 1 - x.mean()),
            n_samples=(entropy_col, "count"),
        )
        .reset_index()
    )

    return summary
