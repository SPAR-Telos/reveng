import pickle
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv


def generate_baseline(
    num_grids: int,
    grid_size: int,
    complexity: float,
    output_file: str = "baseline_grids.pkl",
) -> Dict[str, Simple2DNavigationEnv]:
    """
    Generate a dataset of Simple2DNavigationEnv grids in fully observable setting.

    Args:
        num_grids: Number of different grid environments to generate
        grid_size: Size of each grid (grid_size x grid_size)
        complexity: Maze complexity from 0.0 (empty room) to 1.0 (perfect maze)
        output_file: Path to save the pickle file

    Returns:
        A dictionary mapping grid_id -> Simple2DNavigationEnv
    """
    dataset = {}

    print(
        f"Generating {num_grids} grids with size {grid_size}x{grid_size} and complexity {complexity}..."
    )

    for i in tqdm(range(num_grids), desc="Generating grids"):
        # Create environment with specified parameters
        env = Simple2DNavigationEnv(
            size=grid_size,
            complexity=complexity,
            render_mode=None,  # No rendering needed for dataset generation
        )

        # Reset to generate the grid
        env.reset()

        # Create unique ID for this grid
        grid_id = f"grid_{i:04d}"

        # Store the environment directly
        dataset[grid_id] = env

    # Save to pickle file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to: {output_path}")
    print(f"Total grids: {len(dataset)}")

    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate baseline grid dataset")
    parser.add_argument(
        "--num-grids",
        type=int,
        default=10,
        help="Number of grids to generate (default: 10)",
    )
    parser.add_argument(
        "--grid-size", type=int, default=9, help="Size of each grid (default: 9)"
    )
    parser.add_argument(
        "--complexity",
        type=float,
        default=0.5,
        help="Maze complexity from 0.0 to 1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="src/reveng/experiments/datasets/baseline_grids.pkl",
        help="Output pickle file path (default: src/reveng/experiments/datasets/baseline_grids.pkl)",
    )

    args = parser.parse_args()

    # Generate the dataset
    generate_baseline(
        num_grids=args.num_grids,
        grid_size=args.grid_size,
        complexity=args.complexity,
        output_file=args.output,
    )
