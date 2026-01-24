"""Analyze fully observable grid trajectories using empirical action distributions.

This module analyzes trajectories from fully observable grids, computing metrics
based on empirical action distributions (counts across 10 trajectories per grid)
rather than logprobs.

Key metrics (trajectory-based definitions from the paper):

Capability Metrics:
- Action Accuracy: (1/T) * Σ_{t=0}^{T-1} 1(a^t ∈ π*_G(s^t))
- Goal Success: 1(τ_G^T(s^0) = goal)
- SPL: (1/N) * Σ (1_S * L*) / max(L*, L)

Uncertainty/Calibration Metrics (using empirical distributions):
- Mean Entropy: (1/T) * Σ H(π^θ_G(·|s^t))
- Mean JSD: (1/T) * Σ JSD(π^θ_G(·|s^t) || π*_G(s^t))
- ECE: Expected Calibration Error
"""

import argparse
import gc
import json
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from reveng.analysis.analysis_utils import (
    ACTION_ID_TO_NAME,
    ACTION_NAME_TO_ID,
    ActionDist,
    LightweightTrajectory,
    OptimalActionSet,
    TrajectoryGridParams,
    TrajectoryStep,
    compute_optimal_actions_from_text_grid,
    compute_spl,
    compute_trajectory_action_accuracy,
    extract_agent_position_from_grid_state,
    jensen_shannon_divergence,
    sanitize_label,
    shannon_entropy,
)

# =============================================================================
# Constants
# =============================================================================

# Maximum number of trajectories expected per grid
NUM_TRAJECTORIES_PER_GRID = 10

# Paper-quality plot settings
PAPER_RC = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
}

MODEL_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
]


def setup_paper_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(PAPER_RC)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class StateActionCounts:
    """Counts of actions taken at each state across multiple trajectories."""

    # Map from (x, y) position to action counts {action_id: count}
    counts: dict[tuple[int, int], dict[int, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )

    def add(self, position: tuple[int, int], action: str) -> None:
        """Add an action observation at a position."""
        action_id = ACTION_NAME_TO_ID.get(action.upper())
        if action_id is not None:
            self.counts[position][action_id] += 1

    def get_empirical_distribution(self, position: tuple[int, int]) -> ActionDist:
        """Get empirical action distribution at a position."""
        action_counts = self.counts.get(position, {})
        total = sum(action_counts.values())
        if total == 0:
            return {aid: 0.0 for aid in ACTION_ID_TO_NAME}
        return {aid: action_counts.get(aid, 0) / total for aid in ACTION_ID_TO_NAME}

    def get_total_visits(self, position: tuple[int, int]) -> int:
        """Get total number of visits to a position."""
        return sum(self.counts.get(position, {}).values())


@dataclass
class GridTrajectoryMetrics:
    """Metrics computed from all trajectories for a single grid."""

    grid_id: str
    grid_size: int
    complexity: float
    instance_id: int
    optimal_path_length: int

    # Capability metrics (trajectory-based)
    num_trajectories: int
    num_successful: int
    goal_success_rate: float
    mean_trajectory_length: float
    mean_action_accuracy: float  # Avg across trajectories
    spl: float  # Success weighted by Path Length

    # Uncertainty metrics (using empirical distribution)
    mean_entropy: float  # Avg entropy of empirical dist at visited states
    mean_jsd: float  # Avg JSD between empirical and optimal dist
    ece: float  # Expected Calibration Error

    # Additional
    mean_step_accuracy: float  # Per-step accuracy (all steps across all trajs)
    total_steps: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        return {
            "grid_id": self.grid_id,
            "grid_size": self.grid_size,
            "complexity": self.complexity,
            "instance_id": self.instance_id,
            "optimal_path_length": self.optimal_path_length,
            "num_trajectories": self.num_trajectories,
            "num_successful": self.num_successful,
            "goal_success_rate": self.goal_success_rate,
            "mean_trajectory_length": self.mean_trajectory_length,
            "mean_action_accuracy": self.mean_action_accuracy,
            "spl": self.spl,
            "mean_entropy": self.mean_entropy,
            "mean_jsd": self.mean_jsd,
            "ece": self.ece,
            "mean_step_accuracy": self.mean_step_accuracy,
            "total_steps": self.total_steps,
        }


@dataclass
class ModelTrajectoryResults:
    """Results for a single model."""

    model_name: str
    df: pd.DataFrame  # Per-grid metrics
    state_df: pd.DataFrame  # Per-state metrics with distance, size, complexity
    summary_by_size_complexity: pd.DataFrame
    summary_by_distance: pd.DataFrame  # Per-distance metrics
    overall_summary: dict[str, Any]

    @property
    def n_grids(self) -> int:
        return len(self.df)


# =============================================================================
# File Parsing and Discovery
# =============================================================================


def parse_trajectory_filename(filename: str) -> Optional[dict[str, Any]]:
    """Parse trajectory filename to extract metadata.

    Expected format: {model}_size{N}_comp{X.X}_grid{N}_base_traj{N}.json

    Returns None if not a valid baseline trajectory file.
    """
    # Skip non-base (isotransform) files
    if "_base_traj" not in filename:
        return None

    # Pattern: {model}_size{size}_comp{comp}_grid{grid_id}_base_traj{traj_id}.json
    pattern = r"(.+)_size(\d+)_comp([\d.]+)_grid(\d+)_base_traj(\d+)\.json"
    match = re.match(pattern, filename)

    if not match:
        return None

    model, size, comp, grid_id, traj_id = match.groups()
    return {
        "model": model,
        "grid_size": int(size),
        "complexity": float(comp),
        "grid_id": int(grid_id),
        "trajectory_id": int(traj_id),
    }


def discover_trajectory_files(
    trajectory_dir: Path,
) -> dict[str, list[Path]]:
    """Discover trajectory files grouped by grid.

    Returns:
        Dictionary mapping grid_key to list of trajectory file paths
        Grid key format: "size{N}_comp{X.X}_grid{N}"
    """
    grid_trajectories: dict[str, list[Path]] = defaultdict(list)

    for filepath in sorted(trajectory_dir.glob("*_base_traj*.json")):
        parsed = parse_trajectory_filename(filepath.name)
        if parsed:
            grid_key = (
                f"size{parsed['grid_size']}_"
                f"comp{parsed['complexity']}_"
                f"grid{parsed['grid_id']}"
            )
            grid_trajectories[grid_key].append(filepath)

    return dict(grid_trajectories)


def discover_model_directories(parent_dir: Path, max_depth: int = 3) -> list[Path]:
    """Discover directories containing trajectory files.

    Searches recursively up to max_depth levels for directories with
    baseline trajectory files (*_base_traj*.json).

    Args:
        parent_dir: Root directory to search from
        max_depth: Maximum directory depth to search

    Returns:
        List of directories containing trajectory files
    """
    model_dirs = []

    def _search_recursive(current_dir: Path, depth: int) -> None:
        if depth > max_depth:
            return

        # Check if this directory has trajectory files
        traj_files = list(current_dir.glob("*_base_traj*.json"))
        if traj_files:
            model_dirs.append(current_dir)
            return  # Don't search subdirectories if we found files here

        # Otherwise, search subdirectories
        try:
            for subdir in sorted(current_dir.iterdir()):
                if subdir.is_dir() and not subdir.name.startswith("."):
                    _search_recursive(subdir, depth + 1)
        except PermissionError:
            pass

    _search_recursive(parent_dir, 0)
    return model_dirs


# =============================================================================
# Grid State Parsing
# =============================================================================


def check_reached_goal(final_position: tuple[int, int], goal: tuple[int, int]) -> bool:
    """Check if trajectory reached the goal."""
    return final_position == goal


# =============================================================================
# Trajectory Loading (Memory-Efficient)
# =============================================================================


def load_lightweight_trajectory(filepath: Path) -> Optional[LightweightTrajectory]:
    """Load a trajectory file, keeping only essential fields.

    Discards token-level data to save memory.
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        # Extract grid params
        gp = data.get("grid_params", {})
        # Note: agent_start_coordinates and goal_coordinates are in [row, col] format
        # We convert to (x, y) = (col, row) format for consistency
        start_coords = gp.get("agent_start_coordinates", [0, 0])
        goal_coords = gp.get("goal_coordinates", [0, 0])
        grid_params = TrajectoryGridParams(
            grid_size=gp.get("grid_width", 0),
            complexity=gp.get("grid_complexity", 0.0),
            grid_id=0,  # Will be parsed from filename
            astar_distance=gp.get("astar_distance", 0),
            agent_start=(
                start_coords[1],
                start_coords[0],
            ),  # Convert [row, col] to (x, y)
            goal=(goal_coords[1], goal_coords[0]),  # Convert [row, col] to (x, y)
        )

        # Parse grid_id from filename
        parsed = parse_trajectory_filename(filepath.name)
        if parsed:
            grid_params.grid_id = parsed["grid_id"]

        # Extract steps (only essential fields)
        steps = []
        raw_steps = data.get("steps", [])

        for i, step in enumerate(raw_steps):
            grid_state = step.get("grid_state", [])
            agent_pos = extract_agent_position_from_grid_state(grid_state)
            agent_action = step.get("agent_action", "")

            steps.append(
                TrajectoryStep(
                    step_id=i,
                    agent_position=agent_pos,
                    agent_action=agent_action,
                )
            )

        # Determine if reached goal
        # The final position after taking the last action
        if steps:
            final_pos = steps[-1].agent_position
            # Apply last action to get actual final position
            last_action = steps[-1].agent_action.upper()
            dx, dy = 0, 0
            if last_action == "UP":
                dy = -1
            elif last_action == "DOWN":
                dy = 1
            elif last_action == "LEFT":
                dx = -1
            elif last_action == "RIGHT":
                dx = 1
            final_pos = (final_pos[0] + dx, final_pos[1] + dy)
            reached_goal = check_reached_goal(final_pos, grid_params.goal)
        else:
            reached_goal = False

        return LightweightTrajectory(
            grid_params=grid_params,
            steps=steps,
            reached_goal=reached_goal,
        )

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Warning: Error loading {filepath.name}: {e}")
        return None


def batch_grid_keys(grid_keys: list[str], batch_size: int) -> Iterator[list[str]]:
    """Yield batches of grid keys."""
    for i in range(0, len(grid_keys), batch_size):
        yield grid_keys[i : i + batch_size]


# =============================================================================
# Grid Layout Loading
# =============================================================================


def load_grid_layout(grid_file: Path) -> Optional[list[list[str]]]:
    """Load grid layout from the grid metadata file.

    Grid file format: {model}_size{N}_comp{X.X}_grid{N}_base.json (no _traj suffix)
    """
    try:
        with open(grid_file, "r") as f:
            data = json.load(f)
        return data.get("grid_layout", None)
    except (json.JSONDecodeError, KeyError):
        return None


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_empirical_uncertainty_metrics(
    state_action_counts: StateActionCounts,
    optimal_actions: dict[tuple[int, int], OptimalActionSet],
) -> tuple[float, float]:
    """Compute mean entropy and JSD using empirical distributions.

    Returns:
        (mean_entropy, mean_jsd)
    """
    entropies = []
    jsds = []

    for pos, action_counts in state_action_counts.counts.items():
        if sum(action_counts.values()) == 0:
            continue

        empirical_dist = state_action_counts.get_empirical_distribution(pos)
        optimal_set = optimal_actions.get(pos, set())

        # Entropy of empirical distribution
        entropy = shannon_entropy(empirical_dist)
        entropies.append(entropy)

        # JSD between empirical and optimal
        if optimal_set:
            jsd = jensen_shannon_divergence(optimal_set, empirical_dist)
            if jsd is not None:
                jsds.append(jsd)

    mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    mean_jsd = sum(jsds) / len(jsds) if jsds else 0.0

    return mean_entropy, mean_jsd


def compute_ece(
    state_action_counts: StateActionCounts,
    optimal_actions: dict[tuple[int, int], OptimalActionSet],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    ECE = Σ (|B_m|/n) * |acc(B_m) - conf(B_m)|

    For each state, confidence = max probability in empirical distribution,
    accuracy = 1 if most likely action is optimal, 0 otherwise.
    """
    confidences = []
    accuracies = []

    for pos, action_counts in state_action_counts.counts.items():
        total = sum(action_counts.values())
        if total == 0:
            continue

        empirical_dist = state_action_counts.get_empirical_distribution(pos)
        optimal_set = optimal_actions.get(pos, set())

        # Confidence = max probability
        max_prob = max(empirical_dist.values()) if empirical_dist else 0.0

        # Most likely action
        most_likely_action = max(empirical_dist, key=lambda a: empirical_dist[a])

        # Accuracy = 1 if most likely is optimal
        is_correct = 1 if most_likely_action in optimal_set else 0

        confidences.append(max_prob)
        accuracies.append(is_correct)

    if not confidences:
        return 0.0

    confidences = np.array(confidences)
    accuracies = np.array(accuracies)

    # Bin by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences >= bin_boundaries[i]) & (
            confidences < bin_boundaries[i + 1]
        )
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_grid_metrics(
    trajectories: list[LightweightTrajectory],
    optimal_actions: dict[tuple[int, int], OptimalActionSet],
    distances: dict[tuple[int, int], int],
    grid_key: str,
) -> tuple[Optional[GridTrajectoryMetrics], list[dict[str, Any]]]:
    """Compute all metrics for a grid from its trajectories.

    Returns:
        Tuple of (grid metrics, list of per-state metrics with distance)
    """
    if not trajectories:
        return None, []

    # Parse grid key
    # Format: "size{N}_comp{X.X}_grid{N}"
    match = re.match(r"size(\d+)_comp([\d.]+)_grid(\d+)", grid_key)
    if not match:
        return None

    grid_size = int(match.group(1))
    complexity = float(match.group(2))
    instance_id = int(match.group(3))

    # Get optimal path length from first trajectory
    optimal_path_length = trajectories[0].grid_params.astar_distance

    # Build state-action counts across all trajectories
    state_action_counts = StateActionCounts()
    total_steps = 0
    total_correct_steps = 0

    trajectory_accuracies = []

    for traj in trajectories:
        # Compute per-trajectory accuracy
        acc = compute_trajectory_action_accuracy(traj, optimal_actions)
        trajectory_accuracies.append(acc)

        # Aggregate state-action counts
        for step in traj.steps:
            state_action_counts.add(step.agent_position, step.agent_action)
            total_steps += 1

            # Count correct steps
            action_id = ACTION_NAME_TO_ID.get(step.agent_action.upper())
            optimal_set = optimal_actions.get(step.agent_position, set())
            if action_id is not None and action_id in optimal_set:
                total_correct_steps += 1

    # Capability metrics
    num_successful = sum(1 for t in trajectories if t.reached_goal)
    goal_success_rate = num_successful / len(trajectories)
    mean_traj_length = sum(t.trajectory_length for t in trajectories) / len(
        trajectories
    )
    mean_action_accuracy = (
        sum(trajectory_accuracies) / len(trajectory_accuracies)
        if trajectory_accuracies
        else 0.0
    )
    spl = compute_spl(trajectories, optimal_path_length)

    # Uncertainty metrics
    mean_entropy, mean_jsd = compute_empirical_uncertainty_metrics(
        state_action_counts, optimal_actions
    )
    ece = compute_ece(state_action_counts, optimal_actions)

    # Per-step accuracy
    mean_step_accuracy = total_correct_steps / total_steps if total_steps > 0 else 0.0

    # Compute per-state metrics with distance for distance-to-goal analysis
    state_metrics_list: list[dict[str, Any]] = []
    for pos, action_counts in state_action_counts.counts.items():
        total = sum(action_counts.values())
        if total == 0:
            continue

        empirical_dist = state_action_counts.get_empirical_distribution(pos)
        optimal_set = optimal_actions.get(pos, set())
        distance = distances.get(pos, -1)

        if distance < 0:  # Skip unreachable states
            continue

        # Entropy
        entropy = shannon_entropy(empirical_dist)

        # JSD
        jsd = None
        if optimal_set:
            jsd = jensen_shannon_divergence(optimal_set, empirical_dist)

        # Accuracy: is most likely action optimal?
        most_likely_action = max(empirical_dist, key=lambda a: empirical_dist[a])
        is_optimal = 1 if most_likely_action in optimal_set else 0

        state_metrics_list.append(
            {
                "grid_size": grid_size,
                "complexity": complexity,
                "distance_to_goal": distance,
                "entropy": entropy,
                "jsd": jsd,
                "is_optimal": is_optimal,
                "n_observations": total,
            }
        )

    grid_metrics = GridTrajectoryMetrics(
        grid_id=grid_key,
        grid_size=grid_size,
        complexity=complexity,
        instance_id=instance_id,
        optimal_path_length=optimal_path_length,
        num_trajectories=len(trajectories),
        num_successful=num_successful,
        goal_success_rate=goal_success_rate,
        mean_trajectory_length=mean_traj_length,
        mean_action_accuracy=mean_action_accuracy,
        spl=spl,
        mean_entropy=mean_entropy,
        mean_jsd=mean_jsd,
        ece=ece,
        mean_step_accuracy=mean_step_accuracy,
        total_steps=total_steps,
    )

    return grid_metrics, state_metrics_list


# =============================================================================
# Main Processing Pipeline
# =============================================================================


def process_model_trajectories(
    trajectory_dir: Path,
    model_name: Optional[str] = None,
    batch_size: int = 20,
) -> ModelTrajectoryResults:
    """Process all trajectories for a model.

    Args:
        trajectory_dir: Directory containing trajectory files
        model_name: Optional model name override
        batch_size: Number of grids to process per batch

    Returns:
        ModelTrajectoryResults with per-grid metrics
    """
    if model_name is None:
        model_name = sanitize_label(trajectory_dir.name)

    print(f"\nProcessing model: {model_name}")
    print(f"Trajectory directory: {trajectory_dir}")

    # Discover trajectory files grouped by grid
    grid_trajectories = discover_trajectory_files(trajectory_dir)
    print(f"Found {len(grid_trajectories)} grids with trajectories")

    if not grid_trajectories:
        raise ValueError(f"No trajectory files found in {trajectory_dir}")

    # Process in batches
    grid_keys = sorted(grid_trajectories.keys())
    total_batches = (len(grid_keys) + batch_size - 1) // batch_size
    all_metrics: list[GridTrajectoryMetrics] = []
    all_state_metrics: list[dict[str, Any]] = []

    for batch_idx, batch_keys in enumerate(batch_grid_keys(grid_keys, batch_size)):
        print(
            f"\n  Batch {batch_idx + 1}/{total_batches}: "
            f"processing {len(batch_keys)} grids..."
        )

        for grid_key in tqdm(batch_keys, desc=f"Batch {batch_idx + 1}", leave=False):
            traj_files = grid_trajectories[grid_key]

            # Load trajectories for this grid
            trajectories = []
            for traj_file in traj_files:
                traj = load_lightweight_trajectory(traj_file)
                if traj is not None:
                    trajectories.append(traj)

            if not trajectories:
                continue

            # Load grid layout for optimal action computation
            # Grid file has same pattern but without _trajN suffix
            grid_file_pattern = traj_files[0].name.replace(
                f"_traj{parse_trajectory_filename(traj_files[0].name)['trajectory_id']}.json",
                ".json",
            )
            grid_file = traj_files[0].parent / grid_file_pattern

            if grid_file.exists():
                grid_layout = load_grid_layout(grid_file)
            else:
                # Try to construct from trajectory's grid_state
                # Use first step of first trajectory
                if trajectories[0].steps:
                    # Re-load to get grid_state (not stored in lightweight)
                    with open(traj_files[0], "r") as f:
                        data = json.load(f)
                    grid_state = data.get("steps", [{}])[0].get("grid_state", [])
                    # Parse grid_state into layout
                    grid_layout = []
                    for row in grid_state[1:]:  # Skip header
                        parts = row.split()[1:]  # Skip row number
                        grid_layout.append(parts)
                else:
                    continue

            if not grid_layout:
                continue

            # Compute optimal actions and distances
            goal = trajectories[0].grid_params.goal
            optimal_actions, distances = compute_optimal_actions_from_text_grid(
                grid_layout, goal
            )

            # Compute metrics
            metrics, state_metrics = compute_grid_metrics(
                trajectories, optimal_actions, distances, grid_key
            )
            if metrics:
                all_metrics.append(metrics)
                all_state_metrics.extend(state_metrics)

        # Free memory
        gc.collect()

    # Build DataFrame
    df = pd.DataFrame([m.to_dict() for m in all_metrics])

    # Build state-level DataFrame for distance analysis
    state_df = pd.DataFrame(all_state_metrics)

    # Compute summaries
    summary_df = compute_summary_by_size_complexity(df)
    distance_df = compute_summary_by_distance(state_df)
    overall = compute_overall_summary(df)

    return ModelTrajectoryResults(
        model_name=model_name,
        df=df,
        state_df=state_df,
        summary_by_size_complexity=summary_df,
        summary_by_distance=distance_df,
        overall_summary=overall,
    )


def compute_summary_by_size_complexity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics grouped by grid_size and complexity."""
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(["grid_size", "complexity"])
        .agg(
            n_grids=("grid_id", "count"),
            mean_goal_success=("goal_success_rate", "mean"),
            se_goal_success=("goal_success_rate", "sem"),
            mean_action_accuracy=("mean_action_accuracy", "mean"),
            se_action_accuracy=("mean_action_accuracy", "sem"),
            mean_spl=("spl", "mean"),
            se_spl=("spl", "sem"),
            mean_entropy=("mean_entropy", "mean"),
            se_entropy=("mean_entropy", "sem"),
            mean_jsd=("mean_jsd", "mean"),
            se_jsd=("mean_jsd", "sem"),
            mean_ece=("ece", "mean"),
            se_ece=("ece", "sem"),
        )
        .reset_index()
    )

    return summary


def compute_summary_by_distance(
    state_df: pd.DataFrame,
    max_distance: int = 50,
) -> pd.DataFrame:
    """Compute summary statistics grouped by distance to goal.

    Args:
        state_df: DataFrame with per-state metrics
        max_distance: Maximum distance to show; all distances >= max_distance
                      are grouped into a single bucket (e.g., "50+")

    Returns:
        DataFrame with summary statistics by distance, capped at max_distance
    """
    if state_df.empty:
        return pd.DataFrame()

    # Create a copy and cap the distance
    df = state_df.copy()
    df["distance_capped"] = df["distance_to_goal"].clip(upper=max_distance)

    summary = (
        df.groupby("distance_capped")
        .agg(
            n_states=("entropy", "count"),
            mean_entropy=("entropy", "mean"),
            se_entropy=("entropy", "sem"),
            mean_jsd=("jsd", "mean"),
            se_jsd=("jsd", "sem"),
            accuracy=("is_optimal", "mean"),
            total_observations=("n_observations", "sum"),
        )
        .reset_index()
    )

    # Rename back to distance_to_goal for compatibility
    summary = summary.rename(columns={"distance_capped": "distance_to_goal"})

    # Mark the max distance as "50+" in a separate column for labeling
    summary["is_capped"] = summary["distance_to_goal"] == max_distance

    return summary


def compute_overall_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute overall summary statistics."""
    if df.empty:
        return {}

    return {
        "n_grids": int(len(df)),
        "total_trajectories": int(df["num_trajectories"].sum()),
        "total_steps": int(df["total_steps"].sum()),
        "overall_goal_success": float(df["goal_success_rate"].mean()),
        "overall_action_accuracy": float(df["mean_action_accuracy"].mean()),
        "overall_spl": float(df["spl"].mean()),
        "overall_entropy": float(df["mean_entropy"].mean()),
        "overall_jsd": float(df["mean_jsd"].mean()),
        "overall_ece": float(df["ece"].mean()),
    }


# =============================================================================
# Visualizations
# =============================================================================


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    """Save figure to both PNG and PDF subfolders.

    Args:
        fig: Matplotlib figure to save
        output_dir: Base output directory
        filename: Filename without extension (e.g., "metrics_by_distance")

    Returns:
        Path to the PNG file
    """
    # Create subfolders
    png_dir = output_dir / "png"
    pdf_dir = output_dir / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Save both formats
    png_path = png_dir / f"{filename}.png"
    pdf_path = pdf_dir / f"{filename}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    return png_path


def plot_metrics_by_size_complexity(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> dict[str, Path]:
    """Generate plots of metrics by grid size and complexity."""
    setup_paper_style()
    output_paths = {}

    # Metrics to plot
    metrics = [
        ("goal_success_rate", "Goal Success Rate"),
        ("mean_action_accuracy", "Action Accuracy"),
        ("spl", "SPL"),
        ("mean_entropy", "Mean Entropy (bits)"),
        ("mean_jsd", "Mean JSD"),
        ("ece", "ECE"),
    ]

    for metric_col, metric_label in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # By grid size
        size_summary = df.groupby("grid_size")[metric_col].agg(["mean", "sem"])
        axes[0].errorbar(
            size_summary.index,
            size_summary["mean"],
            yerr=size_summary["sem"],
            marker="o",
            capsize=3,
            color=MODEL_COLORS[0],
        )
        axes[0].set_xlabel("Grid Size")
        axes[0].set_ylabel(metric_label)
        axes[0].set_title(f"{metric_label} by Grid Size")
        axes[0].grid(True, alpha=0.3)

        # By complexity
        comp_summary = df.groupby("complexity")[metric_col].agg(["mean", "sem"])
        axes[1].errorbar(
            comp_summary.index,
            comp_summary["mean"],
            yerr=comp_summary["sem"],
            marker="o",
            capsize=3,
            color=MODEL_COLORS[1],
        )
        axes[1].set_xlabel("Complexity")
        axes[1].set_ylabel(metric_label)
        axes[1].set_title(f"{metric_label} by Complexity")
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"{metric_label}", fontweight="bold")
        # Add note explaining error bars
        fig.text(
            0.99,
            0.01,
            "Mean over grids; Error bars: ±1 SE",
            ha="right",
            va="bottom",
            fontsize=8,
            style="italic",
            color="gray",
        )
        plt.tight_layout(rect=[0, 0.03, 1, 1])

        output_path = save_figure(fig, output_dir, f"{metric_col}_by_size_complexity")
        plt.close(fig)
        output_paths[metric_col] = output_path

    return output_paths


def plot_metrics_by_distance(
    distance_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    smoothing_window: int = 5,
    max_distance: int = 50,
) -> Path:
    """Plot metrics vs distance to goal with smoothing.

    Args:
        distance_df: DataFrame with distance summary
        output_dir: Output directory
        model_name: Model name for title
        smoothing_window: Rolling window size for smoothing
        max_distance: Maximum distance value (last bucket is "{max_distance}+")
    """
    if distance_df.empty:
        return output_dir / "png" / "metrics_by_distance.png"

    setup_paper_style()

    # Sort by distance and apply rolling average for smoothing
    df_sorted = distance_df.sort_values("distance_to_goal").copy()
    df_sorted["entropy_smooth"] = (
        df_sorted["mean_entropy"].rolling(window=smoothing_window, center=True).mean()
    )
    df_sorted["jsd_smooth"] = (
        df_sorted["mean_jsd"].rolling(window=smoothing_window, center=True).mean()
    )
    df_sorted["accuracy_smooth"] = (
        df_sorted["accuracy"].rolling(window=smoothing_window, center=True).mean()
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Entropy vs distance
    axes[0].scatter(
        df_sorted["distance_to_goal"],
        df_sorted["mean_entropy"],
        alpha=0.3,
        s=15,
        color=MODEL_COLORS[0],
        label="Raw",
    )
    axes[0].plot(
        df_sorted["distance_to_goal"],
        df_sorted["entropy_smooth"],
        linewidth=2,
        color=MODEL_COLORS[0],
        label=f"Smoothed (window={smoothing_window})",
    )
    axes[0].set_xlabel("Distance to Goal")
    axes[0].set_ylabel("Mean Entropy (bits)")
    axes[0].set_title("Entropy vs Distance")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left", frameon=False)

    # JSD vs distance
    axes[1].scatter(
        df_sorted["distance_to_goal"],
        df_sorted["mean_jsd"],
        alpha=0.3,
        s=15,
        color=MODEL_COLORS[1],
        label="Raw",
    )
    axes[1].plot(
        df_sorted["distance_to_goal"],
        df_sorted["jsd_smooth"],
        linewidth=2,
        color=MODEL_COLORS[1],
        label=f"Smoothed (window={smoothing_window})",
    )
    axes[1].set_xlabel("Distance to Goal")
    axes[1].set_ylabel("Mean JSD")
    axes[1].set_title("JSD vs Distance")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, loc="upper left", frameon=False)

    # Accuracy vs distance
    axes[2].scatter(
        df_sorted["distance_to_goal"],
        df_sorted["accuracy"],
        alpha=0.3,
        s=15,
        color=MODEL_COLORS[2],
        label="Raw",
    )
    axes[2].plot(
        df_sorted["distance_to_goal"],
        df_sorted["accuracy_smooth"],
        linewidth=2,
        color=MODEL_COLORS[2],
        label=f"Smoothed (window={smoothing_window})",
    )
    axes[2].set_xlabel("Distance to Goal")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Accuracy vs Distance")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=7, loc="lower left", frameon=False)

    # Set explicit x-axis limits and ticks for all panels
    # The data is capped at max_distance, so we set appropriate limits
    for ax in axes:
        ax.set_xlim(-1, max_distance + 3)
        # Create evenly spaced ticks: 0, 10, 20, 30, 40, 50
        tick_values = list(range(0, max_distance + 1, 10))
        tick_labels = [
            str(t) if t < max_distance else f"{max_distance}+" for t in tick_values
        ]
        ax.set_xticks(tick_values)
        ax.set_xticklabels(tick_labels)

    plt.suptitle("Metrics by Distance to Goal", fontweight="bold")
    fig.text(
        0.99,
        0.01,
        f"Points: per-distance aggregates (capped at {max_distance}+); Line: rolling average",
        ha="right",
        va="bottom",
        fontsize=8,
        style="italic",
        color="gray",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    output_path = save_figure(fig, output_dir, "metrics_by_distance")
    plt.close(fig)

    return output_path


def plot_capability_vs_uncertainty(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> Path:
    """Plot capability metrics vs uncertainty metrics."""
    setup_paper_style()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Accuracy vs Entropy
    axes[0].scatter(
        df["mean_entropy"],
        df["mean_action_accuracy"],
        alpha=0.5,
        s=20,
        color=MODEL_COLORS[0],
    )
    axes[0].set_xlabel("Mean Entropy (bits)")
    axes[0].set_ylabel("Action Accuracy")
    axes[0].set_title("Accuracy vs Entropy")
    axes[0].grid(True, alpha=0.3)

    # Accuracy vs JSD
    axes[1].scatter(
        df["mean_jsd"],
        df["mean_action_accuracy"],
        alpha=0.5,
        s=20,
        color=MODEL_COLORS[1],
    )
    axes[1].set_xlabel("Mean JSD")
    axes[1].set_ylabel("Action Accuracy")
    axes[1].set_title("Accuracy vs JSD")
    axes[1].grid(True, alpha=0.3)

    # SPL vs ECE
    axes[2].scatter(
        df["ece"],
        df["spl"],
        alpha=0.5,
        s=20,
        color=MODEL_COLORS[2],
    )
    axes[2].set_xlabel("ECE")
    axes[2].set_ylabel("SPL")
    axes[2].set_title("SPL vs ECE")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Capability vs Uncertainty", fontweight="bold")
    plt.tight_layout()

    output_path = save_figure(fig, output_dir, "capability_vs_uncertainty")
    plt.close(fig)

    return output_path


def plot_heatmaps(
    summary_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> Path:
    """Plot heatmaps of metrics by grid_size x complexity."""
    setup_paper_style()

    metrics = [
        ("mean_goal_success", "Goal Success Rate"),
        ("mean_action_accuracy", "Action Accuracy"),
        ("mean_spl", "SPL"),
        ("mean_jsd", "Mean JSD"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, (metric_col, metric_label) in enumerate(metrics):
        pivot = summary_df.pivot(
            index="complexity", columns="grid_size", values=metric_col
        )

        im = axes[idx].imshow(pivot.values, cmap="RdYlGn", aspect="auto")
        axes[idx].set_xticks(range(len(pivot.columns)))
        axes[idx].set_xticklabels(pivot.columns)
        axes[idx].set_yticks(range(len(pivot.index)))
        axes[idx].set_yticklabels([f"{c:.1f}" for c in pivot.index])
        axes[idx].set_xlabel("Grid Size")
        axes[idx].set_ylabel("Complexity")
        axes[idx].set_title(metric_label)

        # Add values
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    axes[idx].text(
                        j, i, f"{val:.2f}", ha="center", va="center", fontsize=8
                    )

        plt.colorbar(im, ax=axes[idx])

    plt.suptitle("Metrics Heatmaps", fontweight="bold")
    plt.tight_layout()

    output_path = save_figure(fig, output_dir, "metrics_heatmaps")
    plt.close(fig)

    return output_path


def plot_distance_complexity_heatmap(
    state_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    metric: str = "is_optimal",
    metric_label: str = "Action Accuracy",
    n_distance_bins: int = 10,
    max_distance: int = 50,
) -> Path:
    """Plot multi-panel heatmap of metric by distance, complexity, and grid size.

    Creates a figure with 5 columns (one per grid size), where:
    - X-axis: complexity bins (0 to 1)
    - Y-axis: binned distance to goal
    - Fill: mean metric value for that combination

    Args:
        state_df: DataFrame with per-state metrics including grid_size, complexity,
                  distance_to_goal, and the metric column
        output_dir: Directory to save the figure
        model_name: Name of the model for the title
        metric: Column name for the metric to plot
        metric_label: Human-readable label for the metric
        n_distance_bins: Number of bins for distance
        max_distance: Maximum distance to show; distances >= max_distance are capped

    Returns:
        Path to the saved figure
    """
    if state_df.empty or metric not in state_df.columns:
        return output_dir / "png" / f"heatmap_{metric}_by_distance_complexity.png"

    setup_paper_style()

    # Get unique grid sizes (sorted)
    grid_sizes = sorted(state_df["grid_size"].unique())
    n_sizes = len(grid_sizes)

    if n_sizes == 0:
        return output_dir / "png" / f"heatmap_{metric}_by_distance_complexity.png"

    # Cap the distance at max_distance + 0.5 so values >= max_distance
    # fall into the "50+" bin (which is (50, 51])
    df = state_df.copy()
    df["distance_capped"] = df["distance_to_goal"].clip(upper=max_distance + 0.5)

    # Create distance bins with integer-aligned edges
    # Use bin_width to divide max_distance evenly
    bin_width = max_distance // n_distance_bins
    distance_bins = list(range(0, max_distance, bin_width)) + [
        max_distance,
        max_distance + 1,
    ]

    distance_labels = []
    for i in range(len(distance_bins) - 1):
        start = distance_bins[i]
        end = distance_bins[i + 1]
        if start == max_distance:
            # Last bin: "50+"
            distance_labels.append(f"{max_distance}+")
        else:
            distance_labels.append(f"{start}-{end}")

    # Bin the distances
    df["distance_bin"] = pd.cut(
        df["distance_capped"],
        bins=distance_bins,
        labels=distance_labels,
        include_lowest=True,
    )

    # Get unique complexity values (sorted)
    complexity_values = sorted(df["complexity"].unique())

    # Create figure with subplots for each grid size
    fig, axes = plt.subplots(1, n_sizes, figsize=(2.5 * n_sizes, 6), sharey=True)

    # Handle single grid size case
    if n_sizes == 1:
        axes = [axes]

    # Track global min/max for consistent colorbar
    all_values = []

    # First pass: compute all pivot tables and find global min/max
    pivots = []
    for size in grid_sizes:
        size_df = df[df["grid_size"] == size]

        # Aggregate by complexity and distance bin
        pivot = (
            size_df.groupby(["distance_bin", "complexity"], observed=False)[metric]
            .mean()
            .unstack()
        )

        # Reindex to ensure all complexity values are present
        pivot = pivot.reindex(columns=complexity_values)

        # Reindex rows to ensure all distance bins are present
        pivot = pivot.reindex(distance_labels)

        pivots.append(pivot)

        # Collect non-NaN values for colorbar range
        valid_vals = pivot.values[~np.isnan(pivot.values)]
        if len(valid_vals) > 0:
            all_values.extend(valid_vals)

    # Determine color range
    if all_values:
        vmin, vmax = np.min(all_values), np.max(all_values)
    else:
        vmin, vmax = 0, 1

    # Second pass: plot heatmaps
    for idx, (size, pivot) in enumerate(zip(grid_sizes, pivots)):
        ax = axes[idx]

        # Create heatmap
        im = ax.imshow(
            pivot.values,
            cmap="RdYlGn",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            origin="lower",
        )

        # X-axis: complexity
        ax.set_xticks(range(len(complexity_values)))
        ax.set_xticklabels([f"{c:.1f}" for c in complexity_values], fontsize=7)
        ax.set_xlabel("Complexity", fontsize=9)

        ax.set_title(f"{size}x{size}", fontsize=10, fontweight="bold")

        # Add grid lines between cells (minor ticks)
        ax.set_xticks(np.arange(-0.5, len(complexity_values), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(distance_labels), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=3)

        # Y-axis: distance bins (only show labels for first subplot)
        if idx == 0:
            ax.set_ylabel("Distance to Goal", fontsize=9)
        else:
            ax.tick_params(labelleft=False)

    # Set y-tick labels on first axis (do this after loop to avoid sharey issues)
    axes[0].set_yticks(range(len(distance_labels)))
    axes[0].set_yticklabels(distance_labels, fontsize=7)

    # Adjust layout first to position subplots
    plt.subplots_adjust(top=0.92, wspace=0.08, left=0.06, right=0.88)

    # Add colorbar in dedicated axes on the right (doesn't steal space from subplots)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.65])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(metric_label, fontsize=9)

    plt.suptitle(
        f"{metric_label} by Grid Size, Complexity, and Distance",
        fontweight="bold",
        fontsize=11,
        y=1.02,
    )

    output_path = save_figure(
        fig, output_dir, f"heatmap_{metric}_by_distance_complexity"
    )
    plt.close(fig)

    return output_path


# =============================================================================
# Output Saving
# =============================================================================


def save_results(
    results: ModelTrajectoryResults,
    output_dir: Path,
) -> dict[str, Path]:
    """Save results to CSV files and generate visualizations."""
    model_dir = output_dir / results.model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}

    # Save per-grid metrics
    grid_path = model_dir / f"trajectory_metrics_{results.model_name}.csv"
    results.df.to_csv(grid_path, index=False)
    output_paths["grid_metrics"] = grid_path
    print(f"  Saved: {grid_path}")

    # Save summary
    summary_path = model_dir / "summary_by_size_complexity.csv"
    results.summary_by_size_complexity.to_csv(summary_path, index=False)
    output_paths["summary"] = summary_path
    print(f"  Saved: {summary_path}")

    # Save distance summary
    distance_path = model_dir / "summary_by_distance.csv"
    results.summary_by_distance.to_csv(distance_path, index=False)
    output_paths["distance_summary"] = distance_path
    print(f"  Saved: {distance_path}")

    # Save overall summary
    overall_path = model_dir / "overall_summary.json"
    with open(overall_path, "w") as f:
        json.dump(results.overall_summary, f, indent=2)
    output_paths["overall"] = overall_path
    print(f"  Saved: {overall_path}")

    # Generate visualizations
    print("  Generating visualizations...")
    plot_metrics_by_size_complexity(results.df, model_dir, results.model_name)
    plot_capability_vs_uncertainty(results.df, model_dir, results.model_name)
    if not results.summary_by_size_complexity.empty:
        plot_heatmaps(results.summary_by_size_complexity, model_dir, results.model_name)
    if not results.summary_by_distance.empty:
        plot_metrics_by_distance(
            results.summary_by_distance, model_dir, results.model_name
        )

    # Generate distance-complexity heatmaps for multiple metrics
    if not results.state_df.empty:
        plot_distance_complexity_heatmap(
            results.state_df,
            model_dir,
            results.model_name,
            metric="is_optimal",
            metric_label="Action Accuracy",
        )
        plot_distance_complexity_heatmap(
            results.state_df,
            model_dir,
            results.model_name,
            metric="entropy",
            metric_label="Entropy",
        )
        plot_distance_complexity_heatmap(
            results.state_df,
            model_dir,
            results.model_name,
            metric="jsd",
            metric_label="JSD",
        )

    return output_paths


def print_summary(results: ModelTrajectoryResults) -> None:
    """Print summary to console."""
    print("\n" + "=" * 60)
    print(f"TRAJECTORY ANALYSIS SUMMARY: {results.model_name}")
    print("=" * 60)

    overall = results.overall_summary
    print(f"\nTotal grids analyzed: {overall.get('n_grids', 0)}")
    print(f"Total trajectories: {overall.get('total_trajectories', 0)}")
    print(f"Total steps: {overall.get('total_steps', 0)}")

    print("\n--- Capability Metrics ---")
    print(f"Goal Success Rate: {overall.get('overall_goal_success', 0):.4f}")
    print(f"Action Accuracy: {overall.get('overall_action_accuracy', 0):.4f}")
    print(f"SPL: {overall.get('overall_spl', 0):.4f}")

    print("\n--- Uncertainty Metrics (Empirical) ---")
    print(f"Mean Entropy: {overall.get('overall_entropy', 0):.4f} bits")
    print(f"Mean JSD: {overall.get('overall_jsd', 0):.4f}")
    print(f"ECE: {overall.get('overall_ece', 0):.4f}")

    print("\n--- By Grid Size ---")
    size_summary = results.df.groupby("grid_size").agg(
        {
            "goal_success_rate": "mean",
            "mean_action_accuracy": "mean",
            "spl": "mean",
            "mean_jsd": "mean",
        }
    )
    print(size_summary.to_string())

    print("\n" + "=" * 60)


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze fully observable grid trajectories using empirical distributions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--trajectory-dir",
        type=str,
        required=True,
        help="Directory containing trajectory JSON files (or parent with model subdirs)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/reveng/analysis/outputs/full_obs_trajectory_analysis",
        help="Directory to save analysis outputs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Number of grids to process per batch (limits RAM usage)",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override model name (default: derived from directory name)",
    )

    parser.add_argument(
        "--multi-model",
        action="store_true",
        help="Process multiple models from subdirectories of trajectory-dir",
    )

    args = parser.parse_args()

    traj_path = Path(args.trajectory_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.multi_model:
        # Process multiple model directories
        model_dirs = discover_model_directories(traj_path)
        print(f"Found {len(model_dirs)} model directories")

        for model_dir in model_dirs:
            try:
                results = process_model_trajectories(
                    model_dir,
                    batch_size=args.batch_size,
                )
                save_results(results, output_path)
                print_summary(results)
            except Exception as e:
                print(f"Error processing {model_dir.name}: {e}")
                import traceback

                traceback.print_exc()
    else:
        # Process single directory
        results = process_model_trajectories(
            traj_path,
            model_name=args.model_name,
            batch_size=args.batch_size,
        )
        save_results(results, output_path)
        print_summary(results)


if __name__ == "__main__":
    main()
