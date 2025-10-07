import os
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import pygame
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import RGBImgObsWrapper

import reveng.agents as agents
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.environment_generator.wrappers.rgb_obs_wrappers import (
    OmnidirectionalFogOfWarRGBImgObsWrapper,
)
from reveng.environment_generator.wrappers.text_obs_wrapper import (
    FogOfWarTextWrapper,
    FullObservabilityTextWrapper,
)
from reveng.trajectory_generator.trajectory_generator import generate_one_trajectory


class ObsWrapperRegistry:
    wrappers = {
        "image": {
            "full": RGBImgObsWrapper,
            "partial": OmnidirectionalFogOfWarRGBImgObsWrapper,
        },
        "text": {
            "full": FullObservabilityTextWrapper,
            "partial": FogOfWarTextWrapper,
        },
    }

    @staticmethod
    def get_wrapper(modality: str, observability: str):
        return ObsWrapperRegistry.wrappers.get(modality, {}).get(observability)


def get_all_dead_ends(env: MiniGridEnv) -> list[tuple[int, int]]:
    dead_ends = []
    for x in range(1, env.width - 1):
        for y in range(1, env.height - 1):
            if env.grid.get(x, y) is None:
                # Count empty neighbors
                neighbors = 0
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (
                        env.grid.get(nx, ny) is None
                        or env.grid.get(nx, ny).type == "goal"
                    ):
                        neighbors += 1

                # Dead end has exactly 1 neighbor
                if neighbors == 1:
                    dead_ends.append((x, y))
    return dead_ends


def is_internal_point(nx: int, ny: int, env) -> bool:
    """
    Checks if a point (nx, ny) is an internal point of the environment,
    meaning it's not on the boundary (i.e., not at x=0, y=0, x=width-1, or y=height-1).

    Args:
        nx (int): The x-coordinate to check.
        ny (int): The y-coordinate to check.
        env: An env

    Returns:
        bool: True if the point is internal, False otherwise.
    """
    return nx > 0 and ny > 0 and nx < env.width - 1 and ny < env.height - 1


def clone_env(env: MiniGridEnv) -> MiniGridEnv:
    """Deep copy the environment while avoiding copying renderer state."""

    window = getattr(env, "window", None)
    clock = getattr(env, "clock", None)

    if hasattr(env, "window"):
        env.window = None
    if hasattr(env, "clock"):
        env.clock = None

    cloned = deepcopy(env)

    if hasattr(env, "window"):
        env.window = window
    if hasattr(env, "clock"):
        env.clock = clock

    return cloned


def compute_optimal_path_length(env: MiniGridEnv) -> float:
    """
    Compute the shortest path length using generate_one_trajectory with AlphaStarAgent.

    Args:
        env: The environment to compute the optimal path for

    Returns:
        The length of the optimal path, or float('inf') if no path exists
    """
    # Create a fresh copy of the environment to avoid side effects
    test_env = clone_env(env)

    # Create an AlphaStarAgent
    agent = agents.AlphaStarAgent()

    # Generate a trajectory using the agent
    trajectory = generate_one_trajectory(
        env=test_env, observation=None, info=None, agent=agent
    )

    # The optimal path length is the number of steps in the trajectory
    if trajectory and trajectory.steps:
        return len(trajectory.steps)

    return float("inf")  # No path found


def run_random_episodes(
    episodes=5,
    size=10,
    complexity=0.0,
    obs_modality: str = "image",
    observability: str = "full",
    save_images=False,
    config_path=None,
):
    """
    Runs episodes with a random agent
    """
    base_env = Simple2DNavigationEnv(
        render_mode="human", size=size, complexity=complexity
    )
    obs_wrapper = ObsWrapperRegistry.get_wrapper(obs_modality, observability)
    if obs_modality == "text" and config_path:
        env = obs_wrapper(base_env, config_path=config_path)
    else:
        env = obs_wrapper(base_env)

    for i in range(episodes):
        # Reset the environment
        env.reset()
        total_reward = 0
        print(f"--- Starting Episode {i + 1}/{episodes} ---")

        terminated, truncated = False, False

        # Run the episode until done
        while not (terminated or truncated):
            env.render()

            # Choose a random action
            action = env.action_space.sample()
            print(f"Action sampled: {action} ({base_env.actions(action).name})")

            # Take the action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Save observation images if requested
            if save_images and obs_modality == "image":
                # Create images directory if it doesn't exist
                if not os.path.exists("images"):
                    os.makedirs("images")

                plt.figure(figsize=(8, 8))
                plt.imshow(obs["image"])
                plt.title(f"Episode {i + 1}, Step {base_env.step_count}")
                plt.savefig(f"images/episode_{i + 1}_step_{base_env.step_count}.png")
                plt.close()

            # A small delay to make the simulation watchable
            time.sleep(0.1)

        # --- Episode End ---
        env.render()

        # Print Episode Summary
        print(f"--- Episode {i + 1} Finished ---")
        if terminated:
            print("Goal was reached!")
        elif truncated:
            print("Time limit (max_steps) was reached.")
        print(f"Total reward for the episode: {total_reward}\n")

        time.sleep(1.5)  # Pause before the next episode

    env.close()


def manual_control(
    size=10,
    complexity=0.0,
    obs_modality: str = "image",
    observability: str = "full",
    save_images=True,
    config_path=None,
):
    base_env = Simple2DNavigationEnv(
        render_mode="human", size=size, complexity=complexity
    )
    obs_wrapper = ObsWrapperRegistry.get_wrapper(obs_modality, observability)
    if obs_modality == "text" and config_path:
        env = obs_wrapper(base_env, config_path=config_path)
    else:
        env = obs_wrapper(base_env)
    env.reset()

    # Map pygame keys to environment actions for cleaner handling
    key_to_action = {
        pygame.K_LEFT: base_env.actions.LEFT,
        pygame.K_RIGHT: base_env.actions.RIGHT,
        pygame.K_UP: base_env.actions.UP,
        pygame.K_DOWN: base_env.actions.DOWN,
    }

    running = True
    while running:
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    action = key_to_action[event.key]
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(obs)

                    # Save observation images if requested
                    if save_images and obs_modality == "image":
                        # Create images directory if it doesn't exist
                        if not os.path.exists("images"):
                            os.makedirs("images")

                        plt.figure(figsize=(8, 8))
                        plt.imshow(obs["image"])
                        plt.title(f"Step {base_env.step_count}")
                        plt.savefig(f"images/{base_env.step_count}.png")
                        plt.close()

                    print(
                        f"Step: {base_env.step_count}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}"
                    )

                    if terminated or truncated:
                        print("Episode finished. Resetting.")
                        env.reset()

    env.close()
