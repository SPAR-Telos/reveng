"""Script to analyze LLM policies vs optimal A* policies."""

import argparse
import heapq
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def parse_filename(filepath):
    """Extract grid_size, complexity, and instance_id from filename."""
    name = filepath.stem  # e.g., 'grid_size7_complexity0.00_0000_metadata'
    parts = name.split("_")
    grid_size = int(parts[1].replace("size", ""))
    complexity = float(parts[2].replace("complexity", ""))
    instance_id = int(parts[3])
    return grid_size, complexity, instance_id


def extract_llm_policy(policy_metadata):
    """Extract the action choices from the policy metadata."""
    height = len(policy_metadata)
    width = len(policy_metadata[0]) if height > 0 else 0

    policy = [[-1 for _ in range(width)] for _ in range(height)]

    for j in range(height):
        for i in range(width):
            cell = policy_metadata[j][i]
            if isinstance(cell, dict) and "llm_response" in cell:
                policy[j][i] = cell["llm_response"]
            else:
                policy[j][i] = -1

    return policy


def compute_optimal_actions(env):
    """Compute ALL optimal actions for each position in the environment.

    Returns a 2D list where each cell contains a set of optimal actions,
    or an empty set for walls. This accounts for multiple equally-optimal paths.
    """

    base_env = getattr(env, "unwrapped", env)
    grid = base_env.grid
    goal = tuple(base_env.goal_pos)
    width, height = grid.width, grid.height

    # Helper to check if a cell is passable
    def is_passable(x, y):
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        cell = grid.get(x, y)
        return (cell is None) or (getattr(cell, "can_overlap", lambda: False)())

    # Run A* from goal backwards to get cost-to-goal for all cells
    # Neighbor deltas: (dx, dy, action_id)
    neighbors = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]  # LEFT, RIGHT, UP, DOWN

    # Dijkstra/A* from goal to find shortest distance to all cells
    distances = {}
    distances[goal] = 0
    heap = [(0, goal)]

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

    # Now for each cell, find ALL actions that lead to optimal neighbors
    optimal_actions = [[set() for _ in range(width)] for _ in range(height)]

    for y in range(height):
        for x in range(width):
            if not is_passable(x, y):
                continue

            current_dist = distances.get((x, y), float("inf"))
            if current_dist == float("inf"):
                continue

            # Check each possible action
            for dx, dy, action in neighbors:
                nx, ny = x + dx, y + dy
                if is_passable(nx, ny):
                    neighbor_dist = distances.get((nx, ny), float("inf"))
                    # This action is optimal if it reduces distance by 1
                    if neighbor_dist == current_dist - 1:
                        optimal_actions[y][x].add(action)

    # Special case: goal cell (no actions needed)
    gx, gy = goal
    optimal_actions[gy][gx] = set()  # or could mark as "goal"

    return optimal_actions


def compare_policies(llm_policy, optimal_actions):
    """Compare LLM policy with ALL optimal actions.

    Args:
        llm_policy: 2D list of LLM actions (or -1 for walls)
        optimal_actions: 2D list of sets of optimal actions (or empty set for walls/goal)

    Returns:
        total_cells: Number of traversable (non-wall) cells in the grid (excluding goal)
        errors: Number of cells where LLM chose a non-optimal action
        error_rate: Normalized error rate (errors / total_cells)
        multi_optimal_cells: Number of cells with multiple optimal actions
    """
    height = len(llm_policy)
    width = len(llm_policy[0]) if height > 0 else 0

    total_cells = 0
    errors = 0
    multi_optimal_cells = 0

    for j in range(height):
        for i in range(width):
            llm_action = llm_policy[j][i]
            optimal_set = optimal_actions[j][i]

            # Skip walls and goal cells
            if not optimal_set or llm_action == -1:
                continue

            total_cells += 1

            # Track cells with multiple optimal actions
            if len(optimal_set) > 1:
                multi_optimal_cells += 1

            # Check if LLM's action is in the set of optimal actions
            if llm_action not in optimal_set:
                errors += 1

    # Normalized error rate: errors divided by number of traversable cells
    error_rate = errors / total_cells if total_cells > 0 else 0.0
    return total_cells, errors, error_rate, multi_optimal_cells


def main(dataset_path, metadata_dir, output_dir):
    print("=" * 80)
    print("LLM POLICY vs OPTIMAL A* POLICY ANALYSIS")
    print("=" * 80)

    # Load the baseline grids dataset
    print("\n1. Loading datasets...")
    print(f"   Dataset: {dataset_path}")
    with open(dataset_path, "rb") as f:
        grids_dataset = pickle.load(f)
    print(f"   Loaded {len(grids_dataset)} environments from dataset")

    # Load all metadata JSON files
    metadata_dir = Path(metadata_dir)
    print(f"   Metadata directory: {metadata_dir}")
    metadata_files = list(metadata_dir.glob("*_metadata.json"))
    print(f"   Found {len(metadata_files)} metadata files")

    # Load metadata with parsed info
    print("\n2. Loading metadata files...")
    metadata_dict = {}
    for fpath in tqdm(metadata_files, desc="Loading metadata"):
        grid_size, complexity, instance_id = parse_filename(fpath)
        key = f"grid_size{grid_size}_complexity{complexity:.2f}_{instance_id:04d}"

        with open(fpath, "r") as f:
            policy_metadata = json.load(f)

        metadata_dict[key] = {
            "policy_metadata": policy_metadata,
            "grid_size": grid_size,
            "complexity": complexity,
            "instance_id": instance_id,
        }

    # Process all grids
    print("\n3. Processing grids and comparing policies...")
    results = []

    common_keys = set(metadata_dict.keys()) & set(grids_dataset.keys())
    print(
        f"   Processing {len(common_keys)} grids with both metadata and environment data"
    )

    for key in tqdm(sorted(common_keys), desc="Analyzing grids"):
        # Extract metadata info
        meta_info = metadata_dict[key]
        grid_size = meta_info["grid_size"]
        complexity = meta_info["complexity"]
        instance_id = meta_info["instance_id"]

        # Get LLM policy
        llm_policy = extract_llm_policy(meta_info["policy_metadata"])

        # Get environment and compute ALL optimal actions
        env = grids_dataset[key]
        optimal_actions = compute_optimal_actions(env)

        # Compare policies (now accounts for multiple optimal actions)
        total_cells, errors, error_rate, multi_optimal_cells = compare_policies(
            llm_policy, optimal_actions
        )

        # Store results
        results.append(
            {
                "grid_id": key,
                "grid_size": grid_size,
                "complexity": complexity,
                "instance_id": instance_id,
                "total_cells": total_cells,
                "errors": errors,
                "error_rate": error_rate,
                "multi_optimal_cells": multi_optimal_cells,
            }
        )

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    print("\n4. Saving results...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "policy_comparison_results.csv"
    df.to_csv(output_path, index=False)
    print(f"   Results saved to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("KEY INSIGHTS - LLM Policy vs Optimal A* Policy")
    print("=" * 80)

    # Overall statistics
    print("\n1. OVERALL PERFORMANCE:")
    print(f"   - Total grids analyzed: {len(df)}")
    print(
        f"   - Mean normalized error rate: {df['error_rate'].mean():.2%} ± {df['error_rate'].std():.2%}"
    )
    print(f"   - Median normalized error rate: {df['error_rate'].median():.2%}")
    print(f"   - Min error rate: {df['error_rate'].min():.2%}")
    print(f"   - Max error rate: {df['error_rate'].max():.2%}")
    print(
        f"   - Mean traversable cells per grid: {df['total_cells'].mean():.1f} ± {df['total_cells'].std():.1f}"
    )
    print(f"   - Mean absolute errors per grid: {df['errors'].mean():.2f}")
    print(
        f"   - Mean cells with multiple optimal actions: {df['multi_optimal_cells'].mean():.1f} ({df['multi_optimal_cells'].mean() / df['total_cells'].mean() * 100:.1f}% of traversable cells)"
    )

    # By grid size
    print("\n2. PERFORMANCE BY GRID SIZE:")
    for size in sorted(df["grid_size"].unique()):
        subset = df[df["grid_size"] == size]
        print(f"   Grid Size {size}x{size}:")
        print(
            f"     - Mean normalized error rate: {subset['error_rate'].mean():.2%} ± {subset['error_rate'].std():.2%}"
        )
        print(f"     - Mean absolute errors: {subset['errors'].mean():.2f}")
        print(
            f"     - Mean traversable cells: {subset['total_cells'].mean():.1f} (out of {size * size} total)"
        )
        print(f"     - Number of grids: {len(subset)}")

    # By complexity
    print("\n3. PERFORMANCE BY COMPLEXITY:")
    print("   (Note: Higher complexity = more walls = fewer traversable cells)")
    for comp in sorted(df["complexity"].unique()):
        subset = df[df["complexity"] == comp]
        print(f"   Complexity {comp:.2f}:")
        print(
            f"     - Mean normalized error rate: {subset['error_rate'].mean():.2%} ± {subset['error_rate'].std():.2%}"
        )
        print(f"     - Mean absolute errors: {subset['errors'].mean():.2f}")
        print(f"     - Mean traversable cells: {subset['total_cells'].mean():.1f}")
        print(
            f"     - Mean cells with multiple optimal actions: {subset['multi_optimal_cells'].mean():.1f} ({subset['multi_optimal_cells'].mean() / subset['total_cells'].mean() * 100:.1f}%)"
        )
        print(f"     - Number of grids: {len(subset)}")

    # Correlation analysis
    print("\n4. CORRELATION ANALYSIS:")
    print(
        f"   - Correlation between grid_size and normalized error_rate: {df['grid_size'].corr(df['error_rate']):.4f}"
    )
    print(
        f"   - Correlation between complexity and normalized error_rate: {df['complexity'].corr(df['error_rate']):.4f}"
    )
    print(
        f"   - Correlation between traversable_cells and absolute errors: {df['total_cells'].corr(df['errors']):.4f}"
    )
    print(
        f"   - Correlation between traversable_cells and normalized error_rate: {df['total_cells'].corr(df['error_rate']):.4f}"
    )
    print(
        f"   - Correlation between multi_optimal_cells and normalized error_rate: {df['multi_optimal_cells'].corr(df['error_rate']):.4f}"
    )
    print(
        f"   - Correlation between complexity and multi_optimal_cells: {df['complexity'].corr(df['multi_optimal_cells']):.4f}"
    )

    # Best and worst cases
    print("\n5. BEST AND WORST CASES:")
    best_idx = df["error_rate"].idxmin()
    worst_idx = df["error_rate"].idxmax()
    print("   Best performance:")
    print(f"     - Grid: {df.loc[best_idx, 'grid_id']}")
    print(f"     - Normalized error rate: {df.loc[best_idx, 'error_rate']:.2%}")
    print(
        f"     - Errors: {df.loc[best_idx, 'errors']:.0f} / {df.loc[best_idx, 'total_cells']:.0f} traversable cells"
    )
    print(
        f"     - Grid size: {df.loc[best_idx, 'grid_size']}, Complexity: {df.loc[best_idx, 'complexity']:.2f}"
    )
    print("   Worst performance:")
    print(f"     - Grid: {df.loc[worst_idx, 'grid_id']}")
    print(f"     - Normalized error rate: {df.loc[worst_idx, 'error_rate']:.2%}")
    print(
        f"     - Errors: {df.loc[worst_idx, 'errors']:.0f} / {df.loc[worst_idx, 'total_cells']:.0f} traversable cells"
    )
    print(
        f"     - Grid size: {df.loc[worst_idx, 'grid_size']}, Complexity: {df.loc[worst_idx, 'complexity']:.2f}"
    )

    print("\n" + "=" * 80)

    # Create visualizations
    print("\n5. Creating visualizations...")

    # Pivot tables
    pivot_errors = (
        df.groupby(["grid_size", "complexity"])["error_rate"].mean().reset_index()
    )
    pivot_table = pivot_errors.pivot(
        index="complexity", columns="grid_size", values="error_rate"
    )

    pivot_errors_abs = (
        df.groupby(["grid_size", "complexity"])["errors"].mean().reset_index()
    )
    pivot_table_abs = pivot_errors_abs.pivot(
        index="complexity", columns="grid_size", values="errors"
    )

    # Create heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        ax=axes[0],
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Error Rate"},
    )
    axes[0].set_title(
        "Mean Error Rate by Grid Size and Complexity", fontsize=14, fontweight="bold"
    )
    axes[0].set_xlabel("Grid Size", fontsize=12)
    axes[0].set_ylabel("Complexity", fontsize=12)

    sns.heatmap(
        pivot_table_abs,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        ax=axes[1],
        cbar_kws={"label": "Mean # Errors"},
    )
    axes[1].set_title(
        "Mean Number of Errors by Grid Size and Complexity",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].set_xlabel("Grid Size", fontsize=12)
    axes[1].set_ylabel("Complexity", fontsize=12)

    plt.tight_layout()
    heatmap_path = output_dir / "heatmaps.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    print(f"   Saved heatmaps to: {heatmap_path}")

    # Line plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for grid_size in sorted(df["grid_size"].unique()):
        subset = df[df["grid_size"] == grid_size]
        mean_by_complexity = subset.groupby("complexity")["error_rate"].mean()
        axes[0].plot(
            mean_by_complexity.index,
            mean_by_complexity.values,
            marker="o",
            linewidth=2,
            markersize=8,
            label=f"Grid Size {grid_size}",
        )

    axes[0].set_xlabel("Complexity", fontsize=12)
    axes[0].set_ylabel("Mean Error Rate", fontsize=12)
    axes[0].set_title(
        "Error Rate vs Complexity by Grid Size", fontsize=14, fontweight="bold"
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    for complexity in sorted(df["complexity"].unique()):
        subset = df[df["complexity"] == complexity]
        mean_by_size = subset.groupby("grid_size")["error_rate"].mean()
        axes[1].plot(
            mean_by_size.index,
            mean_by_size.values,
            marker="o",
            linewidth=2,
            markersize=8,
            label=f"Complexity {complexity:.2f}",
        )

    axes[1].set_xlabel("Grid Size", fontsize=12)
    axes[1].set_ylabel("Mean Error Rate", fontsize=12)
    axes[1].set_title(
        "Error Rate vs Grid Size by Complexity", fontsize=14, fontweight="bold"
    )
    axes[1].legend(fontsize=10, ncol=2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    lineplot_path = output_dir / "line_plots.png"
    plt.savefig(lineplot_path, dpi=150, bbox_inches="tight")
    print(f"   Saved line plots to: {lineplot_path}")

    # Box plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(data=df, x="grid_size", y="error_rate", ax=axes[0], palette="Set2")
    axes[0].set_xlabel("Grid Size", fontsize=12)
    axes[0].set_ylabel("Error Rate", fontsize=12)
    axes[0].set_title(
        "Error Rate Distribution by Grid Size", fontsize=14, fontweight="bold"
    )
    axes[0].grid(True, alpha=0.3, axis="y")

    sns.boxplot(data=df, x="complexity", y="error_rate", ax=axes[1], palette="Set3")
    axes[1].set_xlabel("Complexity", fontsize=12)
    axes[1].set_ylabel("Error Rate", fontsize=12)
    axes[1].set_title(
        "Error Rate Distribution by Complexity", fontsize=14, fontweight="bold"
    )
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    boxplot_path = output_dir / "box_plots.png"
    plt.savefig(boxplot_path, dpi=150, bbox_inches="tight")
    print(f"   Saved box plots to: {boxplot_path}")

    # Scatter plot: traversable cells vs errors
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot colored by grid size
    for grid_size in sorted(df["grid_size"].unique()):
        subset = df[df["grid_size"] == grid_size]
        axes[0].scatter(
            subset["total_cells"],
            subset["errors"],
            alpha=0.6,
            s=100,
            label=f"Grid Size {grid_size}",
        )

    axes[0].set_xlabel("Number of Traversable Cells", fontsize=12)
    axes[0].set_ylabel("Absolute Errors", fontsize=12)
    axes[0].set_title(
        "Absolute Errors vs Traversable Cells", fontsize=14, fontweight="bold"
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Scatter plot colored by complexity
    for complexity in sorted(df["complexity"].unique()):
        subset = df[df["complexity"] == complexity]
        axes[1].scatter(
            subset["total_cells"],
            subset["error_rate"],
            alpha=0.6,
            s=100,
            label=f"Complexity {complexity:.2f}",
        )

    axes[1].set_xlabel("Number of Traversable Cells", fontsize=12)
    axes[1].set_ylabel("Normalized Error Rate", fontsize=12)
    axes[1].set_title(
        "Normalized Error Rate vs Traversable Cells", fontsize=14, fontweight="bold"
    )
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    scatterplot_path = output_dir / "scatter_plots.png"
    plt.savefig(scatterplot_path, dpi=150, bbox_inches="tight")
    print(f"   Saved scatter plots to: {scatterplot_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nIMPORTANT NOTES:")
    print(
        "1. Error rates are NORMALIZED by the number of traversable cells in each grid."
    )
    print(
        "   This accounts for different grid sizes and complexities (wall densities)."
    )
    print("2. The analysis accounts for MULTIPLE OPTIMAL ACTIONS per cell.")
    print("   LLM is only counted as wrong if it chooses a non-optimal action.")
    print("   This fixes the issue where simple grids have many equally-valid paths!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze LLM policies vs optimal A* policies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="src/reveng/experiments/datasets/baseline_grids.pkl",
        help="Path to the baseline grids pickle file",
    )

    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="/Users/niall/Downloads/together_ai_openai_gpt-oss-20b",
        help="Directory containing LLM policy metadata JSON files",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/reveng/analysis",
        help="Directory to save analysis results and visualizations",
    )

    args = parser.parse_args()

    main(
        dataset_path=args.dataset,
        metadata_dir=args.metadata_dir,
        output_dir=args.output_dir,
    )
