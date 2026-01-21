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
import os, sys, glob
sys.path.append(os.path.join('C:/Users\hchen\Dropbox/reveng', "src"))
# sys.path.append(os.path.join('D:/Phoebe\Dropbox/reveng', "src"))

import reveng
basepath = 'C:/Users\hchen\Dropbox/reveng'

def parse_filename(filepath, include_transform=False):
    """Extract grid_size, complexity, and instance_id from filename. adding transform type for the iso difficulties"""
    name = filepath.stem  # e.g., 'grid_size7_complexity0.00_0000_metadata'
    parts = name.split("_")
    grid_size = int(parts[1].replace("size", ""))
    complexity = float(parts[2].replace("complexity", ""))
    instance_id = int(parts[3])
    if include_transform:
        transform_type = str(parts[4])
        return grid_size, complexity, instance_id, transform_type
    else:
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

"""
saving result csvs
"""
dataset_path_iso = basepath+'/experiments/datasets/isodifficulty_grids.pkl'
metadata_dir = 'D:\Phoebe\maps/together_ai_openai_gpt-oss-120b-iso-transforms/together_ai_openai_gpt-oss-120b'
output_dir = basepath+'/src/reveng/experiments/results/together_ai_openai_gpt-oss-120b-iso-transforms'
is_isodifficulty = 'iso-transforms' in metadata_dir

print("=" * 80)
print("LLM POLICY vs OPTIMAL A* POLICY ANALYSIS")
print("=" * 80)

# Load the baseline grids dataset
if is_isodifficulty:
    print("\n1. Loading datasets...")
    print(f"   Dataset: {dataset_path_iso}")
    with open(dataset_path_iso, "rb") as f:
        grids_dataset = pickle.load(f)
else:
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

batch_size = 50
print(f"   Processing in batches of {batch_size}")
# Prepare output csv
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# ADD (replaces the whole metadata loading + processing section)
print("\n2. Loading metadata files and processing in batches...")
batch_n = 1
for start in range(0, len(metadata_files), batch_size):
    output_path = output_dir / ("policy_comparison_results_%s.csv" % batch_n)
    batch_n += 1
    header_written = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    if os.path.exists(output_path):
        continue
    end = min(start + batch_size, len(metadata_files))
    batch = metadata_files[start:end]
    print(f"\n   Batch {start // batch_size + 1}: files {start}–{end - 1}")

    # Load metadata for THIS batch only
    metadata_dict = {}
    for fpath in tqdm(batch, desc="Loading metadata (batch)"):
        if is_isodifficulty:
            grid_size, complexity, instance_id, transform_type = parse_filename(fpath, is_isodifficulty)
            key = f"grid_size{grid_size}_complexity{complexity:.2f}_{instance_id:04d}_{transform_type}"
        else:
            grid_size, complexity, instance_id = parse_filename(fpath)
            key = f"grid_size{grid_size}complexity{complexity:.2f}{instance_id:04d}"
        if key in metadata_dict:
            continue
        try:
            with open(fpath, "r") as f:
                policy_metadata = json.load(f)
            if is_isodifficulty:
                metadata_dict[key] = {
                    "policy_metadata": policy_metadata,
                    "grid_size": grid_size,
                    "complexity": complexity,
                    "instance_id": instance_id,
                    'transform_type': transform_type
                }
            else:
                metadata_dict[key] = {
                    "policy_metadata": policy_metadata,
                    "grid_size": grid_size,
                    "complexity": complexity,
                    "instance_id": instance_id,
                }
        except Exception as e:
            print(f"   Error in {fpath.stem}: {e}")

    # Process only the grids in this batch
    print("3. Processing grids and comparing policies (batch)...")
    results = []

    common_keys = set(metadata_dict.keys()) & set(grids_dataset.keys())
    print(f"   Processing {len(common_keys)} grids with both metadata and environment data")

    for key in tqdm(sorted(common_keys), desc="Analyzing grids (batch)"):
        meta_info = metadata_dict[key]
        grid_size = meta_info["grid_size"]
        complexity = meta_info["complexity"]
        instance_id = meta_info["instance_id"]
        if is_isodifficulty:
            transform_type = meta_info["transform_type"]
        # LLM policy
        llm_policy = extract_llm_policy(meta_info["policy_metadata"])

        # Environment and optimal actions
        env = grids_dataset[key]
        optimal_actions = compute_optimal_actions(env)

        # Compare policies
        total_cells, errors, error_rate, multi_optimal_cells = compare_policies(
            llm_policy, optimal_actions
        )

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

    # Convert this batch to DataFrame and append to CSV
    df_batch = pd.DataFrame(results)

    # Save/append results for this batch
    print("\n4. Saving batch results...")
    mode = "a" if header_written else "w"
    df_batch.to_csv(output_path, index=False, mode=mode, header=not header_written)
    header_written = True
    print(f"   Batch results appended to: {output_path}")

"""
analyse iso transforms and plot
"""
df_dir = basepath+'/src/reveng/experiments/results/together_ai_openai_gpt-oss-20b-iso-transforms'
csv_files = glob.glob(os.path.join(df_dir, "*.csv"))
# Read and concatenate all CSVs
df_transform = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
df_transform['transform_type'] = df_transform.grid_id.str.split('_').str[-1]
df_transform.loc[df_transform['transform_type'].str.contains('00'),'transform_type'] = 'baseline'
df_all = df_transform

output_dir = df_dir
# save the df for mixed effects modeling
# df_all.to_csv(df_dir+'/isodiff.csv')

# Aggregate mean error_rate by grid_size, complexity, and transform_type
agg = (
    df_all
    .groupby(["grid_size", "complexity", "transform_type"], as_index=False)
    .agg(mean_error_rate=("error_rate", "mean"))
)
"""
a. barplot
"""
import seaborn as sns

grid_vals = sorted(agg["grid_size"].unique())
complexity_vals = sorted(agg["complexity"].unique())

nrows, ncols = len(grid_vals), len(complexity_vals)
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False, sharey=True)

for i, g in enumerate(grid_vals):
    for j, c in enumerate(complexity_vals):
        ax = axes[i, j]
        sub = agg[(agg["grid_size"]==g) & (agg["complexity"]==c)]
        sns.barplot(data=sub, x="transform_type", y="mean_error_rate", ax=ax)
        ax.set_title(f"grid={g}, complexity={c}")
        ax.set_xlabel("")
        if j == 0:
            ax.set_ylabel("mean error_rate")
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
fig.savefig(output_dir+"/barplot.png", dpi=300, bbox_inches="tight")

"""
b. lineplot
"""
# Get baseline rows
baseline = (
    df_all[df_all["transform_type"] == "baseline"]
    .rename(columns={"error_rate": "baseline_error_rate"})
    [["grid_size", "complexity", "instance_id", "baseline_error_rate"]]
)

# Join baseline back onto all rows (including baseline itself)
df_merged = df_all.merge(
    baseline,
    on=["grid_size", "complexity", "instance_id"],
    how="inner",
    validate="many_to_one"
)

# Compute per-unit difference from baseline
df_merged["diff_from_baseline"] = (
    df_merged["error_rate"] - df_merged["baseline_error_rate"]
)

agg_diff = (
    df_merged
    # we usually don't care about the baseline row itself (diff = 0)
    [df_merged["transform_type"] != "baseline"]
    .groupby(["grid_size", "complexity", "transform_type"])
    .agg(
        mean_diff=("diff_from_baseline", "mean"),
        std_diff=("diff_from_baseline", "std"),   # <<< HERE: std of differences
    )
    .reset_index()
)

mean_wide = agg_diff.pivot_table(
    index=["grid_size", "complexity"],
    columns="transform_type",
    values="mean_diff"
)

std_wide = agg_diff.pivot_table(
    index=["grid_size", "complexity"],
    columns="transform_type",
    values="std_diff"
)

# Put into a single dataframe like your original `diff`
diff = (
    mean_wide
    .reset_index()
    .merge(
        std_wide.reset_index(),
        on=["grid_size", "complexity"],
        suffixes=("", "_std")
    )
)

# Remove complexity == 0
diff = diff[diff["complexity"] != 0]

# Determine transform types and grid sizes
transforms = [t for t in mean_wide.columns]  # all non-baseline transforms
grid_vals = sorted(diff["grid_size"].unique())

fig, axes = plt.subplots(
    1, len(grid_vals),
    figsize=(5 * len(grid_vals), 4),
    sharey=True
)

if len(grid_vals) == 1:
    axes = [axes]

for ax, g in zip(axes, grid_vals):
    sub = diff[diff["grid_size"] == g].sort_values("complexity")

    for t in transforms:
        ax.errorbar(
            sub["complexity"],
            sub[t],                      # mean difference
            yerr=sub[f"{t}_std"],        # std of difference
            marker="o",
            capsize=4,
            label=t,
        )

    # Baseline = 0 difference by construction
    ax.axhline(0.0, color="gray", linestyle="--")
    ax.set_title(f"grid={g}")
    ax.set_xlabel("complexity")
    ax.grid(True, linestyle=":", alpha=0.6)

    if ax == axes[0]:
        ax.set_ylabel("Δ error_rate (transform − baseline)")

# Shared legend
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    ncol=len(transforms),
    frameon=False
)

plt.tight_layout()
plt.show()

fig.savefig(output_dir+"/lineplot.png", dpi=300, bbox_inches="tight")

"""
c.line plots with datapoints
"""

baseline = (
    df_all[df_all["transform_type"] == "baseline"]
    .rename(columns={"error_rate": "baseline_error_rate"})
    [["grid_size", "complexity", "instance_id", "baseline_error_rate"]]
)

df_merged = df_all.merge(
    baseline,
    on=["grid_size", "complexity", "instance_id"],
    how="inner",
    validate="many_to_one",
)

# Per-instance difference from baseline
df_merged["diff_from_baseline"] = (
    df_merged["error_rate"] - df_merged["baseline_error_rate"]
)

# We won't plot the baseline rows themselves (their diff is always 0)
df_merged_nb = df_merged[df_merged["transform_type"] != "baseline"]

# ---------------------------------------------------------
# 2. Aggregate means of the differences for each condition
# ---------------------------------------------------------

agg_mean = (
    df_merged_nb
    .groupby(["grid_size", "complexity", "transform_type"])
    .agg(mean_diff=("diff_from_baseline", "mean"))
    .reset_index()
)

# ---------------------------------------------------------
# 3. Determine transforms (rows) and grid sizes (columns)
# ---------------------------------------------------------

transforms = sorted(df_merged_nb["transform_type"].unique())   # rows
grid_vals = sorted(df_merged_nb["grid_size"].unique())        # columns

n_rows = len(transforms)  # expected 4
n_cols = len(grid_vals)   # expected 5

# ---------------------------------------------------------
# 4. Create n x m subplot grid
# ---------------------------------------------------------
color_map = {
    "ReflectEnv": "blue",
    "RotateEnv": "orange",
    "StartGoalSwap": "green",
    "TransposeEnv": "red",
}
fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(4 * n_cols, 3 * n_rows),
    sharex=True,
    sharey=True,
)

# In case n_rows or n_cols is 1, make axes indexable as [row][col]
if n_rows == 1:
    axes = [axes]
if n_cols == 1:
    axes = [[ax] for ax in axes]

for i, t in enumerate(transforms):          # row index
    t_color = color_map.get(t, "black")     # default black if missing

    for j, g in enumerate(grid_vals):       # col index
        ax = axes[i][j]

        # Mean differences
        mean_sub = (
            agg_mean[
                (agg_mean["transform_type"] == t)
                & (agg_mean["grid_size"] == g)
            ]
            .sort_values("complexity")
        )

        # Per-instance diffs
        pts_sub = df_merged_nb[
            (df_merged_nb["transform_type"] == t)
            & (df_merged_nb["grid_size"] == g)
        ]

        # Instance-level scatter
        ax.scatter(
            pts_sub["complexity"],
            pts_sub["diff_from_baseline"],
            alpha=0.3,
            s=20,
            color=t_color,
        )

        # Mean line
        ax.plot(
            mean_sub["complexity"],
            mean_sub["mean_diff"],
            marker="o",
            linestyle="-",
            color=t_color,
        )

        # Labels & cosmetics
        if i == 0:
            ax.set_title(f"grid={g}")
        if j == 0:
            ax.set_ylabel(f"{t}\nΔ error_rate", rotation=0, labelpad=40)

        if i == n_rows - 1:
            ax.set_xlabel("complexity")

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
        ax.grid(True, linestyle=":", alpha=0.4)

fig.suptitle(
    "Per-instance diffs from baseline by transform (rows) and grid size (columns)",
    fontsize=14,
    y=1.02,
)
plt.tight_layout()
plt.show()



"""
d. heatmap
"""
# Compute difference vs baseline
delta = (
    agg.pivot_table(index=["grid_size", "complexity"],
                    columns="transform_type",
                    values="mean_error_rate")
    .pipe(lambda t: t.subtract(t["baseline"], axis=0))
)

# Remove complexity == 0
delta = delta.reset_index()
delta = delta[delta["complexity"] != 0]
delta = delta.set_index(["grid_size", "complexity"])

# Get transform types (exclude baseline)
transforms = [t for t in delta.columns if t != "baseline"]

# Determine shared vmin/vmax for consistent color scale
vmin = delta[transforms].min().min()
vmax = delta[transforms].max().max()

# Layout: 2x2 subplots (adjust if you have more/fewer transforms)
n = len(transforms)
nrows, ncols = 2, 2
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))

# Flatten axes for easy indexing
axes = axes.flatten()

# Plot each transform heatmap
for i, t in enumerate(transforms):
    ax = axes[i]
    pivot = delta[t].unstack("complexity")
    sns.heatmap(pivot, ax=ax, cmap="coolwarm", center=0,
                vmin=vmin, vmax=vmax, cbar=False, annot=True, fmt=".2f")
    ax.set_title(f"Δ error_rate vs baseline: {t}")
    ax.set_xlabel("complexity")
    ax.set_ylabel("grid_size")

# Remove any unused subplots if transforms < 4
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label="Δ mean error_rate")

plt.suptitle("Difference from baseline across transforms", fontsize=15, y=0.98)
plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.show()
fig.savefig(output_dir+"/heatmap.png", dpi=300, bbox_inches="tight")


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
