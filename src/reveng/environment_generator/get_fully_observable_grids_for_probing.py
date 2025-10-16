import pandas as pd
import numpy as np
import argparse
from custom_minigrid import Simple2DNavigationEnv
import reveng.environment_generator.wrappers.text_obs_wrapper as text_wrappers
import reveng.trajectory_generator.trajectory_generator as traj_gen
import reveng.agents as agents
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--trajectory-steps",
        type=int,
        default=0,
        help="0: only save the initial observation, >0: save K trajectory steps for each environment",
    )
    parser.add_argument("--results-dir", type=str, default="grids_for_probing")
    parser.add_argument(
        "--file-name",
        type=str,
        default="grids_for_probing.csv",
        help="Name of the output CSV file",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = []
    complexities = np.linspace(0.0, 1.0, args.num_envs)

    print(
        f"Generating {args.num_envs} fully observable environments of size {args.size} with complexities: {complexities}"
    )

    for env_idx in range(args.num_envs):
        env = Simple2DNavigationEnv(size=args.size, complexity=complexities[env_idx])

        # Use fully observable wrapper instead of fog of war
        # Create a custom symbols config without fog and unknown_obj
        custom_symbols = {
            "agent": "A",
            "wall": "#", 
            "goal": "G",
            "empty": " ",
        }
        fully_observable_env = text_wrappers.FullObservabilityTextWrapper(env)
        # Override the symbols to remove unnecessary ones
        fully_observable_env.symbols = custom_symbols
        fully_observable_env._generate_legend()
        width = fully_observable_env.unwrapped.width
        height = fully_observable_env.unwrapped.height
        print(f"Width: {width}, Height: {height}")
        
        fo_observation_str, info = fully_observable_env.reset(seed=args.seed)
        print(f"Fully observable observation: {fo_observation_str}")

        # For fully observable, we only have one observation type
        fo_cell_types = []
        height = fully_observable_env.unwrapped.height
        width = fully_observable_env.unwrapped.width
        for j in range(height):
            for i in range(width):
                # Check for agent position first
                if i == fully_observable_env.unwrapped.agent_pos[0] and j == fully_observable_env.unwrapped.agent_pos[1]:
                    fo_cell_types.append((i, j, "agent"))
                else:
                    # Render the cell content
                    cell = fully_observable_env.unwrapped.grid.get(i, j)
                    if cell is None:
                        fo_cell_types.append((i, j, "empty"))
                    elif cell.type == "wall":
                        fo_cell_types.append((i, j, "wall"))
                    elif cell.type == "goal":
                        fo_cell_types.append((i, j, "goal"))
                    else:
                        # Skip unknown objects, only include the 4 main types
                        continue

        print("Generating a trajectory from AlphaStar")
        agent = agents.AlphaStarAgent()

        trajectory = traj_gen.generate_one_trajectory(
            env=fully_observable_env,
            observation=fo_observation_str,
            info=info,
            agent=agent,
            max_steps_per_trajectory=args.size**2,
        )
        optimal_trajectory_length = len(trajectory.steps)
        print(f"Optimal trajectory length: {optimal_trajectory_length}")

        if args.trajectory_steps == 0:
            print("Only saving the initial observation")
            # Create one row per cell (like the original script)
            for x, y, cell_type in fo_cell_types:
                results.append(
                    {
                        "env_idx": env_idx,
                        "observation": fo_observation_str,  # Same observation for all cells in this env
                        "x": x,
                        "y": y,
                        "cell_type": cell_type,
                        "symbol": fully_observable_env.symbols.get(cell_type, "?"),
                        "classes_map": repr(fully_observable_env.symbols),
                        "optimal_trajectory_length": optimal_trajectory_length,
                        "trajectory_step": 0,
                    }
                )
        else:
            # For trajectory steps, we'd need to modify the wrapper to log observations
            # For now, just save the initial observation
            print("Trajectory steps not implemented for fully observable grids yet")
            results.append(
                {
                    "env_idx": env_idx,
                    "observation": fo_observation_str,
                    "cell_types": fo_cell_types,
                    "classes_map": repr(fully_observable_env.symbols),
                    "optimal_trajectory_length": optimal_trajectory_length,
                    "trajectory_step": 0,
                }
            )

    df = pd.DataFrame(results)

    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, args.file_name)
    df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
