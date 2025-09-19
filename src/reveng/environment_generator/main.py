import argparse

from custom_minigrid import manual_control, run_random_episodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a simulation in the Simple2DNavigationEnv."
    )
    parser.add_argument(
        "--mode",
        choices=["random", "manual"],
        default="random",
        help="The mode to run the simulation in: 'random' for random actions, 'manual' for user control.",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run in 'random' mode. (Default: 5)",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=10,
        help="The height and width of the grid for the environment. (Default: 10)",
    )
    args = parser.parse_args()

    if args.mode == "random":
        print(
            f"Running in random mode for {args.episodes} episodes with a grid size of {args.size}x{args.size}."
        )
        run_random_episodes(episodes=args.episodes, size=args.size)
    elif args.mode == "manual":
        print(f"Running in manual mode with a grid size of {args.size}x{args.size}.")
        manual_control(size=args.size)
