"""Analyze iso-difficulty transformed trajectories and compare to baseline.

This module compares model performance on base grids vs iso-difficulty
transformed versions (ReflectEnv, RotateEnv, StartGoalSwap, TransposeEnv).

Key questions:
- Does the model perform differently on transformed versions of the same grid?
- Which transforms cause the largest performance degradation?

Metrics compared:
- Goal Success Rate (GS)
- Action Accuracy
- SPL (Success weighted by Path Length)

Statistical approach:
- Paired comparisons: same grid, different transforms
- Wilcoxon signed-rank test (non-parametric)
- Effect sizes for practical significance
"""

import argparse
import gc
import json
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from tqdm import tqdm

from reveng.analysis.analysis_utils import (
    LightweightTrajectory,
    OptimalActionSet,
    TrajectoryGridParams,
    TrajectoryStep,
    compute_optimal_actions_from_text_grid,
    compute_spl,
    compute_trajectory_action_accuracy,
    extract_agent_position_from_grid_state,
    sanitize_label,
)

# =============================================================================
# Constants
# =============================================================================

# Transform types (including baseline)
TRANSFORM_TYPES = ["base", "ReflectEnv", "RotateEnv", "StartGoalSwap", "TransposeEnv"]

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

# Colorblind-friendly palette for transforms
TRANSFORM_COLORS = {
    "base": "#888888",  # Grey
    "ReflectEnv": "#0072B2",  # Blue
    "RotateEnv": "#D55E00",  # Vermillion
    "StartGoalSwap": "#009E73",  # Bluish green
    "TransposeEnv": "#CC79A7",  # Reddish purple
}


def setup_paper_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(PAPER_RC)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TransformTrajectoryMetrics:
    """Metrics for trajectories of a single (grid, transform) combination."""

    grid_key: str  # e.g., "size7_comp0.0_grid0"
    grid_size: int
    complexity: float
    instance_id: int
    transform_type: str  # "base", "ReflectEnv", etc.
    optimal_path_length: int

    # Capability metrics
    num_trajectories: int
    num_successful: int
    goal_success_rate: float
    mean_action_accuracy: float
    spl: float
    mean_trajectory_length: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_key": self.grid_key,
            "grid_size": self.grid_size,
            "complexity": self.complexity,
            "instance_id": self.instance_id,
            "transform_type": self.transform_type,
            "optimal_path_length": self.optimal_path_length,
            "num_trajectories": self.num_trajectories,
            "num_successful": self.num_successful,
            "goal_success_rate": self.goal_success_rate,
            "mean_action_accuracy": self.mean_action_accuracy,
            "spl": self.spl,
            "mean_trajectory_length": self.mean_trajectory_length,
        }


@dataclass
class PairedTestResult:
    """Result of a paired statistical test."""

    metric: str
    transform: str
    baseline_mean: float
    transform_mean: float
    difference: float  # transform - baseline
    statistic: float  # Wilcoxon statistic
    p_value: float
    n_pairs: int
    effect_size: float  # Rank-biserial correlation

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "transform": self.transform,
            "baseline_mean": self.baseline_mean,
            "transform_mean": self.transform_mean,
            "difference": self.difference,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n_pairs": self.n_pairs,
            "effect_size": self.effect_size,
        }


@dataclass
class IsotransformComparisonResults:
    """Results of isotransform comparison analysis."""

    model_name: str
    df: pd.DataFrame  # Per (grid, transform) metrics
    summary_by_transform: pd.DataFrame
    paired_tests: list[PairedTestResult]
    summary_by_size_transform: pd.DataFrame

    @property
    def n_grids(self) -> int:
        return self.df["grid_key"].nunique()

    @property
    def transforms(self) -> list[str]:
        return sorted(self.df["transform_type"].unique())


# =============================================================================
# File Parsing
# =============================================================================


def parse_trajectory_filename_with_transform(
    filename: str,
) -> Optional[dict[str, Any]]:
    """Parse trajectory filename to extract metadata including transform type.

    Expected formats:
    - Base: {model}_size{N}_comp{X.X}_grid{N}_base_traj{N}.json
    - Transform: {model}_size{N}_comp{X.X}_grid{N}_{TransformType}_traj{N}.json

    Returns None if not a valid trajectory file.
    """
    # Pattern for any trajectory file
    # {model}_size{size}_comp{comp}_grid{grid_id}_{transform}_traj{traj_id}.json
    pattern = r"(.+)_size(\d+)_comp([\d.]+)_grid(\d+)_([A-Za-z]+)_traj(\d+)\.json"
    match = re.match(pattern, filename)

    if not match:
        return None

    model, size, comp, grid_id, transform, traj_id = match.groups()
    return {
        "model": model,
        "grid_size": int(size),
        "complexity": float(comp),
        "grid_id": int(grid_id),
        "transform_type": transform,
        "trajectory_id": int(traj_id),
    }


def discover_all_trajectory_files(
    trajectory_dir: Path,
) -> dict[str, dict[str, list[Path]]]:
    """Discover all trajectory files grouped by grid and transform.

    Returns:
        Nested dict: grid_key -> transform_type -> list of trajectory paths
    """
    # grid_key -> transform_type -> [paths]
    grid_transform_trajectories: dict[str, dict[str, list[Path]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for filepath in sorted(trajectory_dir.glob("*_traj*.json")):
        parsed = parse_trajectory_filename_with_transform(filepath.name)
        if parsed:
            grid_key = (
                f"size{parsed['grid_size']}_"
                f"comp{parsed['complexity']}_"
                f"grid{parsed['grid_id']}"
            )
            transform = parsed["transform_type"]
            grid_transform_trajectories[grid_key][transform].append(filepath)

    # Convert to regular dict
    return {
        grid_key: dict(transforms)
        for grid_key, transforms in grid_transform_trajectories.items()
    }


# =============================================================================
# Trajectory Loading
# =============================================================================


def check_reached_goal(final_position: tuple[int, int], goal: tuple[int, int]) -> bool:
    """Check if trajectory reached the goal."""
    return final_position == goal


def load_lightweight_trajectory_with_transform(
    filepath: Path,
) -> Optional[LightweightTrajectory]:
    """Load a trajectory file, keeping only essential fields.

    Also extracts transform_type from filename.
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        # Parse filename to get transform type
        parsed = parse_trajectory_filename_with_transform(filepath.name)
        transform_type = parsed["transform_type"] if parsed else "base"

        # Extract grid params
        gp = data.get("grid_params", {})
        start_coords = gp.get("agent_start_coordinates", [0, 0])
        goal_coords = gp.get("goal_coordinates", [0, 0])
        grid_params = TrajectoryGridParams(
            grid_size=gp.get("grid_width", 0),
            complexity=gp.get("grid_complexity", 0.0),
            grid_id=parsed["grid_id"] if parsed else 0,
            astar_distance=gp.get("astar_distance", 0),
            agent_start=(start_coords[1], start_coords[0]),
            goal=(goal_coords[1], goal_coords[0]),
        )

        # Extract steps
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
        if steps:
            final_pos = steps[-1].agent_position
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
            transform_type=transform_type,
        )

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Warning: Error loading {filepath.name}: {e}")
        return None


def load_grid_layout_from_trajectory(traj_file: Path) -> Optional[list[list[str]]]:
    """Load grid layout from a trajectory file's first step."""
    try:
        with open(traj_file, "r") as f:
            data = json.load(f)

        steps = data.get("steps", [])
        if not steps:
            return None

        grid_state = steps[0].get("grid_state", [])
        if not grid_state:
            return None

        # Parse grid_state into layout
        grid_layout = []
        for row in grid_state[1:]:  # Skip header
            parts = row.split()[1:]  # Skip row number
            grid_layout.append(parts)

        return grid_layout

    except (json.JSONDecodeError, KeyError):
        return None


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_transform_metrics(
    trajectories: list[LightweightTrajectory],
    optimal_actions: dict[tuple[int, int], OptimalActionSet],
    grid_key: str,
    transform_type: str,
) -> Optional[TransformTrajectoryMetrics]:
    """Compute metrics for a (grid, transform) combination."""
    if not trajectories:
        return None

    # Parse grid key
    match = re.match(r"size(\d+)_comp([\d.]+)_grid(\d+)", grid_key)
    if not match:
        return None

    grid_size = int(match.group(1))
    complexity = float(match.group(2))
    instance_id = int(match.group(3))

    # Get optimal path length
    optimal_path_length = trajectories[0].grid_params.astar_distance

    # Compute per-trajectory accuracy
    trajectory_accuracies = []
    for traj in trajectories:
        acc = compute_trajectory_action_accuracy(traj, optimal_actions)
        trajectory_accuracies.append(acc)

    # Aggregate metrics
    num_successful = sum(1 for t in trajectories if t.reached_goal)
    goal_success_rate = num_successful / len(trajectories)
    mean_action_accuracy = (
        sum(trajectory_accuracies) / len(trajectory_accuracies)
        if trajectory_accuracies
        else 0.0
    )
    spl = compute_spl(trajectories, optimal_path_length)
    mean_traj_length = sum(t.trajectory_length for t in trajectories) / len(
        trajectories
    )

    return TransformTrajectoryMetrics(
        grid_key=grid_key,
        grid_size=grid_size,
        complexity=complexity,
        instance_id=instance_id,
        transform_type=transform_type,
        optimal_path_length=optimal_path_length,
        num_trajectories=len(trajectories),
        num_successful=num_successful,
        goal_success_rate=goal_success_rate,
        mean_action_accuracy=mean_action_accuracy,
        spl=spl,
        mean_trajectory_length=mean_traj_length,
    )


# =============================================================================
# Statistical Tests
# =============================================================================


def compute_paired_test(
    df: pd.DataFrame,
    metric: str,
    transform: str,
    baseline: str = "base",
) -> Optional[PairedTestResult]:
    """Compute Wilcoxon signed-rank test comparing transform to baseline.

    Args:
        df: DataFrame with columns [grid_key, transform_type, <metric>]
        metric: Metric column to compare
        transform: Transform type to compare against baseline
        baseline: Baseline transform type (default: "base")

    Returns:
        PairedTestResult or None if insufficient data
    """
    # Get baseline and transform values for same grids
    baseline_df = df[df["transform_type"] == baseline][["grid_key", metric]].rename(
        columns={metric: f"{metric}_baseline"}
    )
    transform_df = df[df["transform_type"] == transform][["grid_key", metric]].rename(
        columns={metric: f"{metric}_transform"}
    )

    # Merge on grid_key to get paired data
    paired = baseline_df.merge(transform_df, on="grid_key", how="inner")

    if len(paired) < 10:  # Need sufficient pairs
        return None

    baseline_vals = paired[f"{metric}_baseline"].values
    transform_vals = paired[f"{metric}_transform"].values
    differences = transform_vals - baseline_vals

    # Wilcoxon signed-rank test
    # Filter out zero differences (ties)
    non_zero_diff = differences[differences != 0]
    if len(non_zero_diff) < 5:
        return None

    try:
        statistic, p_value = stats.wilcoxon(
            transform_vals, baseline_vals, alternative="two-sided"
        )
    except ValueError:
        return None

    # Effect size: rank-biserial correlation
    # r = 1 - (2W) / (n(n+1)/2) where W is the smaller rank sum
    n = len(non_zero_diff)
    effect_size = 1 - (2 * statistic) / (n * (n + 1) / 2)

    return PairedTestResult(
        metric=metric,
        transform=transform,
        baseline_mean=float(baseline_vals.mean()),
        transform_mean=float(transform_vals.mean()),
        difference=float(differences.mean()),
        statistic=float(statistic),
        p_value=float(p_value),
        n_pairs=len(paired),
        effect_size=float(effect_size),
    )


def run_all_paired_tests(
    df: pd.DataFrame,
    metrics: list[str] = ["goal_success_rate", "mean_action_accuracy", "spl"],
) -> list[PairedTestResult]:
    """Run paired tests for all metrics and transforms."""
    results = []

    transforms = [t for t in df["transform_type"].unique() if t != "base"]

    for metric in metrics:
        for transform in transforms:
            result = compute_paired_test(df, metric, transform)
            if result:
                results.append(result)

    return results


# =============================================================================
# Processing Pipeline
# =============================================================================


def batch_grid_keys(grid_keys: list[str], batch_size: int) -> Iterator[list[str]]:
    """Yield batches of grid keys."""
    for i in range(0, len(grid_keys), batch_size):
        yield grid_keys[i : i + batch_size]


def process_isotransform_trajectories(
    trajectory_dir: Path,
    model_name: Optional[str] = None,
    batch_size: int = 20,
) -> IsotransformComparisonResults:
    """Process all trajectories for isotransform comparison.

    Args:
        trajectory_dir: Directory containing trajectory files
        model_name: Optional model name override
        batch_size: Number of grids to process per batch

    Returns:
        IsotransformComparisonResults with metrics and statistical tests
    """
    if model_name is None:
        model_name = sanitize_label(trajectory_dir.name)

    print(f"\nProcessing isotransform trajectories for: {model_name}")
    print(f"Trajectory directory: {trajectory_dir}")

    # Discover all trajectory files
    grid_transform_files = discover_all_trajectory_files(trajectory_dir)
    print(f"Found {len(grid_transform_files)} grids with trajectories")

    if not grid_transform_files:
        raise ValueError(f"No trajectory files found in {trajectory_dir}")

    # Count transforms
    transform_counts: dict[str, int] = defaultdict(int)
    for transforms in grid_transform_files.values():
        for t in transforms:
            transform_counts[t] += 1
    print(f"Transform distribution: {dict(transform_counts)}")

    # Process in batches
    grid_keys = sorted(grid_transform_files.keys())
    total_batches = (len(grid_keys) + batch_size - 1) // batch_size
    all_metrics: list[TransformTrajectoryMetrics] = []

    for batch_idx, batch_keys in enumerate(batch_grid_keys(grid_keys, batch_size)):
        print(
            f"\n  Batch {batch_idx + 1}/{total_batches}: "
            f"processing {len(batch_keys)} grids..."
        )

        for grid_key in tqdm(batch_keys, desc=f"Batch {batch_idx + 1}", leave=False):
            transform_files = grid_transform_files[grid_key]

            # For each transform type available for this grid
            for transform_type, traj_files in transform_files.items():
                # Load grid layout from a trajectory of THIS transform type
                # (transformed grids have different wall positions!)
                grid_layout = load_grid_layout_from_trajectory(traj_files[0])

                if not grid_layout:
                    continue

                # Load trajectories
                trajectories = []
                for traj_file in traj_files:
                    traj = load_lightweight_trajectory_with_transform(traj_file)
                    if traj is not None:
                        trajectories.append(traj)

                if not trajectories:
                    continue

                # Compute optimal actions for this transform's grid layout
                goal = trajectories[0].grid_params.goal
                optimal_actions, _ = compute_optimal_actions_from_text_grid(
                    grid_layout, goal
                )

                # Compute metrics
                metrics = compute_transform_metrics(
                    trajectories, optimal_actions, grid_key, transform_type
                )
                if metrics:
                    all_metrics.append(metrics)

        gc.collect()

    # Build DataFrame
    df = pd.DataFrame([m.to_dict() for m in all_metrics])

    # Compute summaries
    summary_by_transform = compute_summary_by_transform(df)
    summary_by_size_transform = compute_summary_by_size_transform(df)

    # Run paired statistical tests
    paired_tests = run_all_paired_tests(df)

    return IsotransformComparisonResults(
        model_name=model_name,
        df=df,
        summary_by_transform=summary_by_transform,
        paired_tests=paired_tests,
        summary_by_size_transform=summary_by_size_transform,
    )


def compute_summary_by_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics grouped by transform type."""
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby("transform_type")
        .agg(
            n_grids=("grid_key", "nunique"),
            n_trajectories=("num_trajectories", "sum"),
            mean_goal_success=("goal_success_rate", "mean"),
            se_goal_success=("goal_success_rate", "sem"),
            mean_action_accuracy=("mean_action_accuracy", "mean"),
            se_action_accuracy=("mean_action_accuracy", "sem"),
            mean_spl=("spl", "mean"),
            se_spl=("spl", "sem"),
        )
        .reset_index()
    )

    return summary


def compute_summary_by_size_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary grouped by grid_size and transform_type."""
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(["grid_size", "transform_type"])
        .agg(
            n_grids=("grid_key", "nunique"),
            mean_goal_success=("goal_success_rate", "mean"),
            se_goal_success=("goal_success_rate", "sem"),
            mean_action_accuracy=("mean_action_accuracy", "mean"),
            se_action_accuracy=("mean_action_accuracy", "sem"),
            mean_spl=("spl", "mean"),
            se_spl=("spl", "sem"),
        )
        .reset_index()
    )

    return summary


# =============================================================================
# Visualizations
# =============================================================================


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    """Save figure to both PNG and PDF subfolders.

    Args:
        fig: Matplotlib figure to save
        output_dir: Base output directory
        filename: Filename without extension (e.g., "metrics_by_transform")

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


def plot_metrics_by_transform(
    df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Plot bar charts of metrics by transform type."""
    setup_paper_style()

    metrics = [
        ("goal_success_rate", "Goal Success Rate"),
        ("mean_action_accuracy", "Action Accuracy"),
        ("spl", "SPL"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for idx, (metric_col, metric_label) in enumerate(metrics):
        summary = df.groupby("transform_type")[metric_col].agg(["mean", "sem"])
        summary = summary.reindex([t for t in TRANSFORM_TYPES if t in summary.index])

        colors = [TRANSFORM_COLORS.get(t, "gray") for t in summary.index]
        x = range(len(summary))

        axes[idx].bar(x, summary["mean"], yerr=summary["sem"], capsize=3, color=colors)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(summary.index, rotation=45, ha="right")
        axes[idx].set_ylabel(metric_label)
        axes[idx].set_title(f"{metric_label} by Transform")
        axes[idx].grid(True, alpha=0.3, axis="y")

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

    output_path = save_figure(fig, output_dir, "metrics_by_transform")
    plt.close(fig)

    return output_path


def plot_delta_from_baseline(
    df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Plot change in metrics relative to baseline."""
    setup_paper_style()

    # Compute delta from baseline for each grid
    baseline_df = df[df["transform_type"] == "base"].set_index("grid_key")
    metrics = ["goal_success_rate", "mean_action_accuracy", "spl"]
    metric_labels = ["Δ Goal Success", "Δ Accuracy", "Δ SPL"]

    transforms = [
        t for t in TRANSFORM_TYPES if t != "base" and t in df["transform_type"].values
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        deltas = []
        for transform in transforms:
            transform_df = df[df["transform_type"] == transform].set_index("grid_key")
            common_grids = baseline_df.index.intersection(transform_df.index)

            if len(common_grids) > 0:
                delta = (
                    transform_df.loc[common_grids, metric].values
                    - baseline_df.loc[common_grids, metric].values
                )
                n = len(delta)
                se_delta = delta.std() / (n**0.5) if n > 1 else 0.0
                deltas.append(
                    {
                        "transform": transform,
                        "mean_delta": delta.mean(),
                        "se_delta": se_delta,
                        "n": n,
                    }
                )

        if deltas:
            delta_df = pd.DataFrame(deltas)
            colors = [TRANSFORM_COLORS.get(t, "gray") for t in delta_df["transform"]]
            x = range(len(delta_df))

            axes[idx].bar(
                x,
                delta_df["mean_delta"],
                yerr=delta_df["se_delta"],
                capsize=3,
                color=colors,
            )
            axes[idx].axhline(y=0, color="black", linestyle="-", linewidth=0.8)
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(delta_df["transform"], rotation=45, ha="right")
            axes[idx].set_ylabel(label)
            axes[idx].set_title(f"{label} (vs Baseline)")
            axes[idx].grid(True, alpha=0.3, axis="y")

    fig.text(
        0.99,
        0.01,
        "Mean over paired grids; Error bars: ±1 SE",
        ha="right",
        va="bottom",
        fontsize=8,
        style="italic",
        color="gray",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    output_path = save_figure(fig, output_dir, "delta_from_baseline")
    plt.close(fig)

    return output_path


def plot_metrics_by_size_transform(
    df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Plot metrics by grid size, with separate lines for each transform."""
    setup_paper_style()

    metrics = [
        ("goal_success_rate", "Goal Success Rate"),
        ("mean_action_accuracy", "Action Accuracy"),
        ("spl", "SPL"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    transforms = [t for t in TRANSFORM_TYPES if t in df["transform_type"].values]

    for idx, (metric_col, metric_label) in enumerate(metrics):
        for transform in transforms:
            subset = df[df["transform_type"] == transform]
            summary = subset.groupby("grid_size")[metric_col].agg(["mean", "sem"])

            color = TRANSFORM_COLORS.get(transform, "gray")
            axes[idx].errorbar(
                summary.index,
                summary["mean"],
                yerr=summary["sem"],
                marker="o",
                capsize=3,
                color=color,
                label=transform,
            )

        axes[idx].set_xlabel("Grid Size")
        axes[idx].set_ylabel(metric_label)
        axes[idx].set_title(f"{metric_label} by Grid Size")
        axes[idx].grid(True, alpha=0.3)
        if idx == 0:
            axes[idx].legend(fontsize=7, loc="best", frameon=False)

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

    output_path = save_figure(fig, output_dir, "metrics_by_size_transform")
    plt.close(fig)

    return output_path


def plot_paired_test_results(
    paired_tests: list[PairedTestResult],
    output_dir: Path,
) -> Path:
    """Plot summary of paired test results."""
    if not paired_tests:
        return output_dir / "png" / "paired_test_summary.png"

    setup_paper_style()

    test_df = pd.DataFrame([t.to_dict() for t in paired_tests])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Effect sizes
    pivot_effect = test_df.pivot(
        index="transform", columns="metric", values="effect_size"
    )
    x = range(len(pivot_effect.index))
    width = 0.25
    metric_colors = ["#0072B2", "#D55E00", "#009E73"]

    for i, metric in enumerate(pivot_effect.columns):
        offset = (i - 1) * width
        axes[0].bar(
            [xi + offset for xi in x],
            pivot_effect[metric],
            width,
            label=metric.replace("_", " ").title(),
            color=metric_colors[i],
        )

    axes[0].axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pivot_effect.index, rotation=45, ha="right")
    axes[0].set_ylabel("Effect Size (rank-biserial r)")
    axes[0].set_title("Effect Size by Transform")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Collect legend handles from first plot
    handles, labels = axes[0].get_legend_handles_labels()

    # P-values (log scale)
    pivot_pval = test_df.pivot(index="transform", columns="metric", values="p_value")

    for i, metric in enumerate(pivot_pval.columns):
        offset = (i - 1) * width
        axes[1].bar(
            [xi + offset for xi in x],
            pivot_pval[metric],
            width,
            color=metric_colors[i],
        )

    alpha_line = axes[1].axhline(y=0.05, color="red", linestyle="--", linewidth=1)
    handles.append(alpha_line)
    labels.append("α=0.05")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pivot_pval.index, rotation=45, ha="right")
    axes[1].set_ylabel("p-value")
    axes[1].set_yscale("log")
    axes[1].set_title("Statistical Significance")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.legend(
        handles,
        labels,
        loc="center right",
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(1.12, 0.5),
    )
    plt.tight_layout(rect=[0, 0, 0.88, 1])

    output_path = save_figure(fig, output_dir, "paired_test_summary")
    plt.close(fig)

    return output_path


# =============================================================================
# Output Saving
# =============================================================================


def save_results(
    results: IsotransformComparisonResults,
    output_dir: Path,
) -> dict[str, Path]:
    """Save results to files and generate visualizations."""
    model_dir = output_dir / results.model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}

    # Save per-(grid, transform) metrics
    metrics_path = model_dir / f"isotransform_metrics_{results.model_name}.csv"
    results.df.to_csv(metrics_path, index=False)
    output_paths["metrics"] = metrics_path
    print(f"  Saved: {metrics_path}")

    # Save summary by transform
    summary_path = model_dir / "summary_by_transform.csv"
    results.summary_by_transform.to_csv(summary_path, index=False)
    output_paths["summary_transform"] = summary_path
    print(f"  Saved: {summary_path}")

    # Save summary by size and transform
    size_summary_path = model_dir / "summary_by_size_transform.csv"
    results.summary_by_size_transform.to_csv(size_summary_path, index=False)
    output_paths["summary_size"] = size_summary_path
    print(f"  Saved: {size_summary_path}")

    # Save paired test results
    if results.paired_tests:
        tests_df = pd.DataFrame([t.to_dict() for t in results.paired_tests])
        tests_path = model_dir / "paired_tests.csv"
        tests_df.to_csv(tests_path, index=False)
        output_paths["paired_tests"] = tests_path
        print(f"  Saved: {tests_path}")

    # Generate visualizations
    print("  Generating visualizations...")
    plot_metrics_by_transform(results.df, model_dir)
    plot_delta_from_baseline(results.df, model_dir)
    plot_metrics_by_size_transform(results.df, model_dir)
    if results.paired_tests:
        plot_paired_test_results(results.paired_tests, model_dir)

    return output_paths


def print_summary(results: IsotransformComparisonResults) -> None:
    """Print summary to console."""
    print("\n" + "=" * 70)
    print(f"ISOTRANSFORM COMPARISON SUMMARY: {results.model_name}")
    print("=" * 70)

    print(f"\nTotal grids analyzed: {results.n_grids}")
    print(f"Transforms found: {results.transforms}")

    print("\n--- Summary by Transform ---")
    print(results.summary_by_transform.to_string(index=False))

    if results.paired_tests:
        print("\n--- Paired Test Results (vs Baseline) ---")
        sig_tests = [t for t in results.paired_tests if t.p_value < 0.05]
        if sig_tests:
            print("Significant differences (p < 0.05):")
            for t in sig_tests:
                direction = "↓" if t.difference < 0 else "↑"
                print(
                    f"  {t.transform} - {t.metric}: "
                    f"Δ={t.difference:+.4f} {direction}, "
                    f"p={t.p_value:.4f}, r={t.effect_size:.3f}"
                )
        else:
            print("No significant differences found at α=0.05")

    print("\n" + "=" * 70)


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze iso-difficulty transformed trajectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--trajectory-dir",
        type=str,
        required=True,
        help="Directory containing trajectory JSON files (base + transforms)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/reveng/analysis/outputs/isotransform_trajectory_analysis",
        help="Directory to save analysis outputs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Number of grids to process per batch",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override model name (default: derived from directory name)",
    )

    args = parser.parse_args()

    traj_path = Path(args.trajectory_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = process_isotransform_trajectories(
        traj_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )

    save_results(results, output_path)
    print_summary(results)


if __name__ == "__main__":
    main()
