"""Analyze trajectory success rates and step efficiency across grid configurations."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TrajectoryMetrics:
    """Metrics extracted from a single trajectory."""

    grid_id: str
    grid_size: int
    complexity: float
    instance_id: int
    reached_goal: bool
    steps_taken: int
    optimal_path_length: int
    step_difference: int  # steps_taken - optimal_path_length
    step_overhead_pct: float  # ((steps_taken - optimal) / optimal) * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_id": self.grid_id,
            "grid_size": self.grid_size,
            "complexity": self.complexity,
            "instance_id": self.instance_id,
            "reached_goal": self.reached_goal,
            "steps_taken": self.steps_taken,
            "optimal_path_length": self.optimal_path_length,
            "step_difference": self.step_difference,
            "step_overhead_pct": self.step_overhead_pct,
        }


# =============================================================================
# File Parsing
# =============================================================================


def parse_grid_id(grid_id: str) -> tuple[int, float, int]:
    """Extract grid_size, complexity, and instance_id from grid_id.

    Expected format: grid_size{N}_complexity{X.XX}_{NNNN}
    Example: grid_size7_complexity0.20_0009
    """
    parts = grid_id.split("_")
    try:
        # parts = ["grid", "size7", "complexity0.20", "0009"]
        grid_size = int(parts[1].replace("size", ""))
        complexity = float(parts[2].replace("complexity", ""))
        instance_id = int(parts[3])
        return grid_size, complexity, instance_id
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid grid_id format: {grid_id}") from e


def discover_trajectory_files(trajectory_dir: Path) -> list[Path]:
    """Find all trajectory JSON files in a directory."""
    return sorted(trajectory_dir.glob("*_trajectories.json"))


def load_trajectory_file(filepath: Path) -> list[dict[str, Any]]:
    """Load trajectories from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def extract_trajectory_metrics(
    trajectory: dict[str, Any],
) -> Optional[TrajectoryMetrics]:
    """Extract metrics from a single trajectory object."""
    metadata = trajectory.get("traj_metadata", {})

    grid_id = metadata.get("grid_id")
    if not grid_id:
        return None

    reached_goal = metadata.get("reached_goal", False)
    steps_taken = metadata.get("steps_taken", 0)
    optimal_path_length = metadata.get("optimal_path_length", 0)

    try:
        grid_size, complexity, instance_id = parse_grid_id(grid_id)
    except ValueError:
        return None

    step_difference = steps_taken - optimal_path_length
    # Percentage overhead: how much longer than optimal (0% = optimal, 100% = 2x optimal)
    if optimal_path_length > 0:
        step_overhead_pct = (step_difference / optimal_path_length) * 100
    else:
        step_overhead_pct = 0.0 if steps_taken == 0 else float("inf")

    return TrajectoryMetrics(
        grid_id=grid_id,
        grid_size=grid_size,
        complexity=complexity,
        instance_id=instance_id,
        reached_goal=reached_goal,
        steps_taken=steps_taken,
        optimal_path_length=optimal_path_length,
        step_difference=step_difference,
        step_overhead_pct=step_overhead_pct,
    )


# =============================================================================
# Data Processing
# =============================================================================


def process_trajectory_directory(trajectory_dir: Path) -> pd.DataFrame:
    """Process all trajectory files in a directory and return a DataFrame."""
    trajectory_files = discover_trajectory_files(trajectory_dir)

    if not trajectory_files:
        print(f"No trajectory files found in {trajectory_dir}")
        return pd.DataFrame()

    all_metrics: list[dict[str, Any]] = []

    for filepath in tqdm(trajectory_files, desc="Processing trajectory files"):
        try:
            trajectories = load_trajectory_file(filepath)
            for traj in trajectories:
                metrics = extract_trajectory_metrics(traj)
                if metrics:
                    all_metrics.append(metrics.to_dict())
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Skipping {filepath.name}: {e}")
            continue

    return pd.DataFrame(all_metrics)


# =============================================================================
# Analysis and Reporting
# =============================================================================


def compute_summary_by_grid_size_complexity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics grouped by grid_size and complexity."""
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(["grid_size", "complexity"])
        .agg(
            num_trajectories=("reached_goal", "count"),
            num_successes=("reached_goal", "sum"),
            success_rate=("reached_goal", "mean"),
            mean_steps_taken=("steps_taken", "mean"),
            mean_optimal_path=("optimal_path_length", "mean"),
            mean_step_diff=("step_difference", "mean"),
            std_step_diff=("step_difference", "std"),
            mean_overhead_pct=("step_overhead_pct", "mean"),
            std_overhead_pct=("step_overhead_pct", "std"),
        )
        .reset_index()
    )

    # Round for display
    summary["success_rate"] = summary["success_rate"].round(4)
    summary["mean_steps_taken"] = summary["mean_steps_taken"].round(2)
    summary["mean_optimal_path"] = summary["mean_optimal_path"].round(2)
    summary["mean_step_diff"] = summary["mean_step_diff"].round(2)
    summary["std_step_diff"] = summary["std_step_diff"].round(2)
    summary["mean_overhead_pct"] = summary["mean_overhead_pct"].round(1)
    summary["std_overhead_pct"] = summary["std_overhead_pct"].round(1)

    return summary


def compute_overall_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute overall summary statistics."""
    if df.empty:
        return {}

    successful_df = df[df["reached_goal"]]

    return {
        "total_trajectories": len(df),
        "total_successes": int(df["reached_goal"].sum()),
        "overall_success_rate": round(df["reached_goal"].mean(), 4),
        "mean_overhead_pct": round(df["step_overhead_pct"].mean(), 1),
        "std_overhead_pct": round(df["step_overhead_pct"].std(), 1),
        "successful_mean_overhead_pct": round(
            successful_df["step_overhead_pct"].mean(), 1
        )
        if len(successful_df) > 0
        else None,
    }


def print_summary_table(summary_df: pd.DataFrame, overall: dict[str, Any]) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 100)
    print("TRAJECTORY ANALYSIS RESULTS")
    print("=" * 100)

    # Overall summary
    print("\nOVERALL SUMMARY:")
    print(f"  Total trajectories:       {overall.get('total_trajectories', 0)}")
    print(f"  Total successes:          {overall.get('total_successes', 0)}")
    print(f"  Overall success rate:     {overall.get('overall_success_rate', 0):.2%}")
    print(f"  Mean step overhead:       {overall.get('mean_overhead_pct', 0):.1f}%")
    print(f"  Std step overhead:        {overall.get('std_overhead_pct', 0):.1f}%")
    if overall.get("successful_mean_overhead_pct") is not None:
        print(
            f"  Mean overhead (success):  {overall.get('successful_mean_overhead_pct'):.1f}%"
        )

    # Detailed table
    print("\n" + "-" * 100)
    print("RESULTS BY GRID SIZE AND COMPLEXITY:")
    print("-" * 100)

    if summary_df.empty:
        print("No data available.")
        return

    # Format for display
    display_df = summary_df.copy()
    display_df["success_rate"] = display_df["success_rate"].apply(lambda x: f"{x:.2%}")
    display_df["overhead_pct"] = display_df.apply(
        lambda r: f"{r['mean_overhead_pct']:.1f}% ± {r['std_overhead_pct']:.1f}%",
        axis=1,
    )

    # Select columns for display
    display_cols = [
        "grid_size",
        "complexity",
        "num_trajectories",
        "num_successes",
        "success_rate",
        "mean_optimal_path",
        "mean_steps_taken",
        "overhead_pct",
    ]

    print(
        display_df[display_cols].to_string(
            index=False,
            col_space={
                "grid_size": 10,
                "complexity": 12,
                "num_trajectories": 15,
                "num_successes": 14,
                "success_rate": 14,
                "mean_optimal_path": 16,
                "mean_steps_taken": 16,
                "overhead_pct": 20,
            },
        )
    )

    print("\n" + "=" * 100)


# =============================================================================
# Visualization
# =============================================================================


def plot_heatmaps(summary_df: pd.DataFrame, output_path: Path, model_tag: str) -> Path:
    """Create heatmaps for success rate and step overhead by grid size and complexity."""
    if summary_df.empty:
        return output_path

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Success rate heatmap
    pivot_success = summary_df.pivot(
        index="complexity", columns="grid_size", values="success_rate"
    )
    # Convert to percentage for display
    pivot_success_pct = pivot_success * 100

    sns.heatmap(
        pivot_success_pct,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        ax=axes[0],
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Success Rate (%)"},
    )
    axes[0].set_title(
        "Success Rate by Grid Size & Complexity", fontsize=13, fontweight="bold"
    )
    axes[0].set_xlabel("Grid Size", fontsize=11)
    axes[0].set_ylabel("Complexity", fontsize=11)

    # Step overhead heatmap
    pivot_overhead = summary_df.pivot(
        index="complexity", columns="grid_size", values="mean_overhead_pct"
    )

    sns.heatmap(
        pivot_overhead,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=axes[1],
        cbar_kws={"label": "Step Overhead (%)"},
    )
    axes[1].set_title(
        "Step Overhead by Grid Size & Complexity", fontsize=13, fontweight="bold"
    )
    axes[1].set_xlabel("Grid Size", fontsize=11)
    axes[1].set_ylabel("Complexity", fontsize=11)

    plt.suptitle(f"Model: {model_tag}", fontsize=14, y=1.02)
    plt.tight_layout()

    heatmap_path = output_path / f"trajectory_heatmaps_{model_tag}.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return heatmap_path


def plot_trends(summary_df: pd.DataFrame, output_path: Path, model_tag: str) -> Path:
    """Create line plots showing trends across complexity for each grid size."""
    if summary_df.empty:
        return output_path

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    grid_sizes = sorted(summary_df["grid_size"].unique())
    colors = plt.cm.viridis(
        [i / max(1, len(grid_sizes) - 1) for i in range(len(grid_sizes))]
    )

    # Success rate vs complexity
    for i, grid_size in enumerate(grid_sizes):
        subset = summary_df[summary_df["grid_size"] == grid_size].sort_values(
            "complexity"
        )
        axes[0].plot(
            subset["complexity"],
            subset["success_rate"] * 100,
            marker="o",
            linewidth=2,
            markersize=8,
            label=f"Grid {grid_size}×{grid_size}",
            color=colors[i],
        )

    axes[0].set_title("Success Rate vs Complexity", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Complexity", fontsize=11)
    axes[0].set_ylabel("Success Rate (%)", fontsize=11)
    axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Step overhead vs complexity
    for i, grid_size in enumerate(grid_sizes):
        subset = summary_df[summary_df["grid_size"] == grid_size].sort_values(
            "complexity"
        )
        axes[1].plot(
            subset["complexity"],
            subset["mean_overhead_pct"],
            marker="s",
            linewidth=2,
            markersize=8,
            label=f"Grid {grid_size}×{grid_size}",
            color=colors[i],
        )

    axes[1].set_title("Step Overhead vs Complexity", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Complexity", fontsize=11)
    axes[1].set_ylabel("Step Overhead (%)", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Model: {model_tag}", fontsize=14, y=1.02)
    plt.tight_layout()

    trends_path = output_path / f"trajectory_trends_{model_tag}.png"
    plt.savefig(trends_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return trends_path


def create_visualizations(
    summary_df: pd.DataFrame, output_dir: Path, model_tag: str
) -> tuple[Path, Path]:
    """Generate all visualization plots."""
    model_dir = output_dir / model_tag
    model_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = plot_heatmaps(summary_df, model_dir, model_tag)
    trends_path = plot_trends(summary_df, model_dir, model_tag)

    return heatmap_path, trends_path


def save_results(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    overall: dict[str, Any],
    output_dir: Path,
    model_tag: str,
) -> tuple[Path, Path]:
    """Save analysis results to CSV files."""
    model_dir = output_dir / model_tag
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save per-trajectory data
    trajectories_path = model_dir / f"trajectory_metrics_{model_tag}.csv"
    df.to_csv(trajectories_path, index=False)

    # Save summary
    summary_path = model_dir / f"trajectory_summary_{model_tag}.csv"
    summary_df.to_csv(summary_path, index=False)

    # Save overall stats as JSON
    overall_path = model_dir / f"trajectory_overall_{model_tag}.json"
    with open(overall_path, "w") as f:
        json.dump(overall, f, indent=2)

    return trajectories_path, summary_path


# =============================================================================
# Main Analysis Pipeline
# =============================================================================


def analyze_trajectories(
    trajectory_dir: str,
    output_dir: Optional[str] = None,
    save_csv: bool = True,
    generate_plots: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the complete trajectory analysis pipeline.

    Args:
        trajectory_dir: Directory containing trajectory JSON files
        output_dir: Directory to save analysis outputs (optional)
        save_csv: Whether to save CSV outputs
        generate_plots: Whether to generate visualization plots

    Returns:
        Tuple of (raw_df, summary_df, overall_stats)
    """
    trajectory_path = Path(trajectory_dir)
    model_tag = trajectory_path.name

    print(f"\n1. Discovering trajectory files in: {trajectory_path}")
    trajectory_files = discover_trajectory_files(trajectory_path)
    print(f"   Found {len(trajectory_files)} trajectory files")

    if not trajectory_files:
        print("   No trajectory files found. Aborting.")
        return pd.DataFrame(), pd.DataFrame(), {}

    print("\n2. Processing trajectories...")
    df = process_trajectory_directory(trajectory_path)

    if df.empty:
        print("   No valid trajectories extracted. Aborting.")
        return pd.DataFrame(), pd.DataFrame(), {}

    print(f"   Extracted {len(df)} trajectory records")

    print("\n3. Computing summary statistics...")
    summary_df = compute_summary_by_grid_size_complexity(df)
    overall = compute_overall_summary(df)

    # Save results if requested
    if save_csv and output_dir:
        output_path = Path(output_dir)
        traj_path, summ_path = save_results(
            df, summary_df, overall, output_path, model_tag
        )
        print("\n4. Saved results to:")
        print(f"   - {traj_path}")
        print(f"   - {summ_path}")

    # Print summary table
    print_summary_table(summary_df, overall)

    # Generate plots if requested
    if generate_plots and output_dir:
        output_path = Path(output_dir)
        print("\n5. Generating plots...")
        heatmap_path, trends_path = create_visualizations(
            summary_df, output_path, model_tag
        )
        print(f"   - {heatmap_path}")
        print(f"   - {trends_path}")

    return df, summary_df, overall


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze trajectory success rates and step efficiency",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--trajectory-dir",
        type=str,
        required=True,
        help="Directory containing trajectory JSON files",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/reveng/analysis",
        help="Directory to save analysis outputs",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving CSV outputs",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating visualization plots",
    )

    args = parser.parse_args()

    analyze_trajectories(
        trajectory_dir=args.trajectory_dir,
        output_dir=args.output_dir,
        save_csv=not args.no_save,
        generate_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
