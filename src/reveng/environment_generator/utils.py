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


def remove_doors_and_keys(env: MiniGridEnv) -> MiniGridEnv:
    """
    Create a copy of the environment with all doors and keys removed.

    Args:
        env: The environment to copy and modify

    Returns:
        A new environment instance with doors and keys removed
    """
    # Clone the environment
    cloned_env = clone_env(env)

    # Remove any doors and keys from the grid
    for x in range(cloned_env.width):
        for y in range(cloned_env.height):
            cell = cloned_env.grid.get(x, y)
            if cell is not None and cell.type in ["door", "key"]:
                cloned_env.grid.set(x, y, None)

    # Clear the carrying inventory if it contains a key
    if hasattr(cloned_env, "carrying") and cloned_env.carrying is not None:
        if cloned_env.carrying.type == "key":
            cloned_env.carrying = None

    return cloned_env


def remove_door(env: MiniGridEnv) -> MiniGridEnv:
    """
    Create a copy of the environment with the door removed.

    Args:
        env: The environment to copy and modify

    Returns:
        A new environment instance with the door removed
    """
    # Clone the environment
    cloned_env = clone_env(env)

    # Remove any doors from the grid
    for x in range(cloned_env.width):
        for y in range(cloned_env.height):
            cell = cloned_env.grid.get(x, y)
            if cell is not None and cell.type == "door":
                cloned_env.grid.set(x, y, None)

    return cloned_env


def remove_coin(env: MiniGridEnv) -> MiniGridEnv:
    """
    Create a copy of the environment with the coin removed.

    Args:
        env: The environment to copy and modify

    Returns:
        A new environment instance with the coin removed
    """
    # Clone the environment
    cloned_env = clone_env(env)

    # Remove any coins (balls) from the grid
    for x in range(cloned_env.width):
        for y in range(cloned_env.height):
            cell = cloned_env.grid.get(x, y)
            if cell is not None and cell.type == "ball":
                cloned_env.grid.set(x, y, None)

    # Reset coin_collected flag if it exists
    if hasattr(cloned_env, "coin_collected"):
        cloned_env.coin_collected = None

    return cloned_env


def remove_key(env: MiniGridEnv) -> MiniGridEnv:
    """
    Create a copy of the environment with the key removed.

    Args:
        env: The environment to copy and modify

    Returns:
        A new environment instance with the key removed
    """
    # Clone the environment
    cloned_env = clone_env(env)

    # Remove any keys from the grid
    for x in range(cloned_env.width):
        for y in range(cloned_env.height):
            cell = cloned_env.grid.get(x, y)
            if cell is not None and cell.type == "key":
                cloned_env.grid.set(x, y, None)

    # Clear the carrying inventory if it contains a key
    if hasattr(cloned_env, "carrying") and cloned_env.carrying is not None:
        if cloned_env.carrying.type == "key":
            cloned_env.carrying = None

    return cloned_env


def replace_key_with_goal(env: MiniGridEnv) -> MiniGridEnv:
    """
    Create a copy of the environment with the key replaced by a goal.

    This function:
    1. Clones the environment
    2. Finds and removes the current goal from the grid
    3. Finds the key position
    4. Removes the key and places a goal at that position
    5. Updates the environment's goal_pos attribute

    Args:
        env: The environment to copy and modify

    Returns:
        A new environment instance with the key replaced by a goal

    Raises:
        ValueError: If no key is found in the environment
    """
    from minigrid.core.world_object import Goal

    # Clone the environment
    cloned_env = clone_env(env)

    key_pos = None
    goal_pos = None

    # Scan the grid to find key and goal positions
    for x in range(cloned_env.width):
        for y in range(cloned_env.height):
            cell = cloned_env.grid.get(x, y)
            if cell is not None:
                if cell.type == "key":
                    key_pos = (x, y)
                elif cell.type == "goal":
                    goal_pos = (x, y)

    # Check if key was found
    if key_pos is None:
        raise ValueError("No key found in the environment")

    # Remove the current goal if it exists
    if goal_pos is not None:
        cloned_env.grid.set(goal_pos[0], goal_pos[1], None)

    # Replace the key with a new goal
    cloned_env.grid.set(key_pos[0], key_pos[1], None)  # Remove key
    cloned_env.put_obj(Goal(), key_pos[0], key_pos[1])  # Place goal
    cloned_env.goal_pos = key_pos

    return cloned_env


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


def compute_optimal_actions_from_position(
    env: MiniGridEnv, position: tuple[int, int]
) -> list[int]:
    """
    Compute all optimal actions from a given position to the goal using BFS.

    An action is optimal if it moves the agent one step closer to the goal
    along any shortest path.

    Args:
        env: The environment to analyze
        position: The (x, y) position to find optimal actions from

    Returns:
        A list of action indices that are optimal from the given position.
        Returns an empty list if the position is unreachable or is the goal.
    """
    from collections import deque

    # If there's no goal, return empty
    if not hasattr(env, "goal_pos") or env.goal_pos is None:
        return []

    goal = tuple(env.goal_pos)

    # If we're already at the goal, no actions needed
    if position == goal:
        return []

    # First, compute distances from goal to all positions using BFS backwards
    distances = {}
    distances[goal] = 0
    queue = deque([goal])

    while queue:
        x, y = queue.popleft()
        current_dist = distances[(x, y)]

        # Check all four directions
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy

            # Skip if already visited
            if (nx, ny) in distances:
                continue

            # Check if the new position is within bounds
            if nx < 0 or ny < 0 or nx >= env.width or ny >= env.height:
                continue

            # Get the cell at the new position
            cell = env.grid.get(nx, ny)

            # Check if we can move to this cell (None or can_overlap)
            if cell is None or (hasattr(cell, "can_overlap") and cell.can_overlap()):
                distances[(nx, ny)] = current_dist + 1
                queue.append((nx, ny))

    # If the given position is not reachable from the goal, return empty
    if position not in distances:
        return []

    current_distance = distances[position]
    optimal_actions = []

    # Check each possible action from the current position
    # Action mapping: LEFT=0, RIGHT=1, UP=2, DOWN=3 (from Simple2DNavigationEnv)
    action_to_delta = {
        0: (-1, 0),  # LEFT
        1: (1, 0),  # RIGHT
        2: (0, -1),  # UP
        3: (0, 1),  # DOWN
    }

    x, y = position
    for action, (dx, dy) in action_to_delta.items():
        nx, ny = x + dx, y + dy

        # Check if this neighbor exists and has distance exactly 1 less
        if (nx, ny) in distances and distances[(nx, ny)] == current_distance - 1:
            optimal_actions.append(action)

    return optimal_actions


def is_solvable(env: MiniGridEnv) -> bool:
    """
    Check if the agent can reach the goal using BFS (Breadth-First Search).

    Args:
        env: The MiniGrid environment to check

    Returns:
        True if the agent can reach the goal, False otherwise
    """
    # If there's no goal, consider it unsolvable
    if not hasattr(env, "goal_pos") or env.goal_pos is None:
        return False

    # Get start and goal positions
    start = tuple(env.agent_pos)
    goal = tuple(env.goal_pos)

    # If agent is already at the goal
    if start == goal:
        return True

    # BFS to find if there's a path from start to goal
    from collections import deque

    queue = deque([start])
    visited = {start}

    while queue:
        x, y = queue.popleft()

        # Check all four directions
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy

            # Skip if already visited
            if (nx, ny) in visited:
                continue

            # Check if the new position is within bounds
            if nx < 0 or ny < 0 or nx >= env.width or ny >= env.height:
                continue

            # Get the cell at the new position
            cell = env.grid.get(nx, ny)

            # Check if we can move to this cell (None or goal)
            if cell is None or (hasattr(cell, "can_overlap") and cell.can_overlap()):
                visited.add((nx, ny))
                queue.append((nx, ny))

                # Check if we reached the goal
                if (nx, ny) == goal:
                    return True

    return False


def goal_exists(env: MiniGridEnv) -> bool:
    """
    Check if a goal exists in the environment.

    Args:
        env: The MiniGrid environment to check

    Returns:
        True if a goal exists, False otherwise
    """
    return hasattr(env, "goal_pos") and env.goal_pos is not None


def get_env_diagnostics(env: MiniGridEnv) -> dict:
    """
    Get comprehensive diagnostics about the environment.

    Args:
        env: The MiniGrid environment to analyze

    Returns:
        Dictionary containing:
            - goal_exists (bool): Whether a goal exists in the environment
            - is_solvable (bool): Whether the agent can reach the goal
            - optimal_path_length (float): Length of optimal path, or inf if unsolvable
            - num_dead_ends (int): Number of dead ends in the environment
            - dead_ends (list[tuple[int, int]]): List of dead end positions
    """
    has_goal = goal_exists(env)
    solvable = is_solvable(env)
    optimal_length = compute_optimal_path_length(env) if solvable else float("inf")
    dead_ends = get_all_dead_ends(env)
    num_dead_ends = len(dead_ends)

    return {
        "goal_exists": has_goal,
        "is_solvable": solvable,
        "optimal_path_length": optimal_length,
        "num_dead_ends": num_dead_ends,
        "dead_ends": dead_ends,
    }


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
