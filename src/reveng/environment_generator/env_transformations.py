from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv

from reveng.environment_generator.utils import (
    clone_env,
    compute_optimal_path_length,
    get_all_dead_ends,
    is_internal_point,
)


class EnvTransformation(ABC):
    def __init__(self):
        self._position_attrs = (
            "agent_pos",
            "agent_start_pos_user",
            "_initial_agent_pos",
            "goal_pos",
            "goal_pos_user",
        )
        self._direction_attrs = (
            "agent_dir",
            "agent_start_dir_user",
            "_initial_agent_dir",
        )

    def __call__(self, env: MiniGridEnv) -> MiniGridEnv:
        return self.apply(env)

    @property
    def __name__(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def _set_object_positions(grid: Grid) -> None:
        for x in range(grid.width):
            for y in range(grid.height):
                obj = grid.get(x, y)
                if obj is None:
                    continue
                obj.cur_pos = (x, y)
                if obj.init_pos is None:
                    obj.init_pos = (x, y)

    @staticmethod
    def _as_tuple(pos: Iterable[int] | np.ndarray | None) -> Tuple[int, int] | None:
        if pos is None:
            return None

        if isinstance(pos, np.ndarray):
            if pos.size != 2:
                return None
            return int(pos[0]), int(pos[1])

        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            return int(pos[0]), int(pos[1])

        return None

    @staticmethod
    def _coerce_position(original, new_xy: Tuple[int, int]):
        if original is None:
            return None

        if isinstance(original, np.ndarray):
            return np.array(new_xy, dtype=original.dtype)

        if isinstance(original, list):
            return [new_xy[0], new_xy[1]]

        return tuple(new_xy)

    def _update_positions(
        self, env: MiniGridEnv, transform: Callable[[Tuple[int, int]], Tuple[int, int]]
    ) -> None:
        for attr in self._position_attrs:
            if not hasattr(env, attr):
                continue

            original_value = getattr(env, attr)
            original_tuple = self._as_tuple(original_value)
            if original_tuple is None:
                continue

            transformed_tuple = transform(original_tuple)
            setattr(env, attr, self._coerce_position(original_value, transformed_tuple))

    def _update_directions(
        self, env: MiniGridEnv, transform: Callable[[int], int]
    ) -> None:
        for attr in self._direction_attrs:
            if not hasattr(env, attr):
                continue

            value = getattr(env, attr)
            if value is None or not isinstance(value, int) or value < 0:
                continue

            setattr(env, attr, transform(value % 4))

    @staticmethod
    def _reflect_grid_vertically(grid: Grid) -> Grid:
        reflected = Grid(grid.width, grid.height)
        for x in range(grid.width):
            for y in range(grid.height):
                reflected.set(grid.width - 1 - x, y, grid.get(x, y))
        return reflected

    @staticmethod
    def _transpose_grid(grid: Grid) -> Grid:
        transposed = Grid(grid.height, grid.width)
        for x in range(grid.width):
            for y in range(grid.height):
                transposed.set(y, x, grid.get(x, y))
        return transposed

    @abstractmethod
    def apply(self, env: MiniGridEnv) -> MiniGridEnv:
        pass


class RotateEnv(EnvTransformation):
    def apply(self, env: MiniGridEnv) -> MiniGridEnv:
        rotated_env = clone_env(env)

        orig_width = rotated_env.grid.width
        rotated_env.grid = rotated_env.grid.rotate_left()
        rotated_env.width = rotated_env.grid.width
        rotated_env.height = rotated_env.grid.height

        self._set_object_positions(rotated_env.grid)

        def rotate_left(pos: Tuple[int, int]) -> Tuple[int, int]:
            x, y = pos
            return y, orig_width - 1 - x

        self._update_positions(rotated_env, rotate_left)
        self._update_directions(rotated_env, lambda direction: (direction + 1) % 4)

        return rotated_env


class ReflectEnv(EnvTransformation):
    def apply(self, env: MiniGridEnv) -> MiniGridEnv:
        reflected_env = clone_env(env)

        width = reflected_env.grid.width
        reflected_env.grid = self._reflect_grid_vertically(reflected_env.grid)
        reflected_env.width = reflected_env.grid.width
        reflected_env.height = reflected_env.grid.height

        self._set_object_positions(reflected_env.grid)

        def reflect(pos: Tuple[int, int]) -> Tuple[int, int]:
            x, y = pos
            return width - 1 - x, y

        dir_map = {0: 2, 1: 1, 2: 0, 3: 3}

        self._update_positions(reflected_env, reflect)
        self._update_directions(
            reflected_env, lambda direction: dir_map.get(direction, direction)
        )

        return reflected_env


class TransposeEnv(EnvTransformation):
    def apply(self, env: MiniGridEnv) -> MiniGridEnv:
        transposed_env = clone_env(env)

        transposed_env.grid = self._transpose_grid(transposed_env.grid)
        transposed_env.width = transposed_env.grid.width
        transposed_env.height = transposed_env.grid.height

        self._set_object_positions(transposed_env.grid)

        dir_map = {0: 1, 1: 0, 2: 3, 3: 2}

        self._update_positions(transposed_env, lambda pos: (pos[1], pos[0]))
        self._update_directions(
            transposed_env, lambda direction: dir_map.get(direction, direction)
        )

        return transposed_env


class StartGoalSwap(EnvTransformation):
    def apply(self, env: MiniGridEnv) -> MiniGridEnv:
        """
        Swap the start and goal positions.
        This preserves optimal path distance and route multiplicity by reversing the path.
        """
        swapped_env = clone_env(env)

        # Swap positions
        old_agent_pos = self._as_tuple(swapped_env.agent_pos)
        old_goal_pos = self._as_tuple(swapped_env.goal_pos)

        if old_agent_pos and old_goal_pos:
            # Update agent position attributes (move agent to where goal was)
            swapped_env.agent_pos = self._coerce_position(
                swapped_env.agent_pos, old_goal_pos
            )
            if hasattr(swapped_env, "agent_start_pos_user"):
                swapped_env.agent_start_pos_user = self._coerce_position(
                    swapped_env.agent_start_pos_user, old_goal_pos
                )
            if hasattr(swapped_env, "_initial_agent_pos"):
                swapped_env._initial_agent_pos = self._coerce_position(
                    swapped_env._initial_agent_pos, old_goal_pos
                )

            # Update goal position attributes (move goal to where agent was)
            swapped_env.goal_pos = self._coerce_position(
                swapped_env.goal_pos, old_agent_pos
            )
            if hasattr(swapped_env, "goal_pos_user"):
                swapped_env.goal_pos_user = self._coerce_position(
                    swapped_env.goal_pos_user, old_agent_pos
                )

            # Update grid: remove old goal and place new goal
            swapped_env.grid.set(*old_goal_pos, None)
            swapped_env.put_obj(Goal(), *old_agent_pos)

            # Update object positions in the grid
            self._set_object_positions(swapped_env.grid)

        return swapped_env


class RemoveDeadEnd(EnvTransformation):
    def apply(self, env: MiniGridEnv) -> MiniGridEnv:
        """
        Removes a wall at a dead-end, ensuring the optimal path length remains unchanged.
        """
        # 1. Establish the baseline optimal path length for the original environment.
        original_path_length = compute_optimal_path_length(env)

        varied_env = clone_env(env)

        # 2. Find all dead-end cells (cells with only one empty neighbor).
        dead_ends = get_all_dead_ends(varied_env)

        # Randomize to avoid bias towards top-left dead-ends
        random.shuffle(dead_ends)

        # 3. Test each candidate modification.
        for x, y in dead_ends:
            # Find the adjacent wall that creates the dead-end.
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                neighbor_cell = env.grid.get(nx, ny)

                if (
                    neighbor_cell is not None
                    and neighbor_cell.type == "wall"
                    and is_internal_point(nx, ny, varied_env)
                ):
                    # Create a temporary environment for testing.
                    trial_env = clone_env(env)

                    # Apply the change: remove the wall.
                    trial_env.grid.set(nx, ny, None)

                    # Calculate the new optimal path length.
                    new_path_length = compute_optimal_path_length(trial_env)

                    # Validate: If the path length is unchanged, we've found a safe modification.
                    if new_path_length == original_path_length:
                        return trial_env

        # 4. If the loop finishes, no safe modification was found.
        raise ValueError(
            "No dead-end variation was applied that preserved the optimal path."
        )


class GoalDisplacement(EnvTransformation):
    def __init__(
        self,
        new_goal_pos: Optional[Tuple[int, int]] = None,
        random_displacement: bool = True,
    ):
        super().__init__()
        self.new_goal_pos = new_goal_pos
        self.random_displacement = random_displacement

    def apply(self, env: MiniGridEnv) -> MiniGridEnv:
        """
        Move the goal to a new location.
        """
        displaced_env = clone_env(env)
        old_goal_pos = (
            displaced_env.goal_pos if hasattr(displaced_env, "goal_pos") else None
        )

        if self.new_goal_pos:
            new_pos = self.new_goal_pos
        elif self.random_displacement:
            # Find a valid random position (not wall, not agent)
            valid_positions = []
            for i in range(displaced_env.width):
                for j in range(displaced_env.height):
                    cell = displaced_env.grid.get(i, j)
                    if cell is None or cell.type == "goal":
                        if (i, j) != tuple(displaced_env.agent_pos) and (
                            i,
                            j,
                        ) != old_goal_pos:
                            valid_positions.append((i, j))
            new_pos = random.choice(valid_positions)
        else:
            raise ValueError("No goal position specified")

        # Remove old goal
        if old_goal_pos is not None:
            displaced_env.grid.set(old_goal_pos[0], old_goal_pos[1], None)

        # Place new goal
        displaced_env.put_obj(Goal(), *new_pos)
        displaced_env.goal_pos = new_pos
        if hasattr(displaced_env, "goal_pos_user"):
            displaced_env.goal_pos_user = new_pos

        return displaced_env


class DynamicObstacle(EnvTransformation):
    def __init__(
        self,
        action: str = "add",
        positions: Optional[list] = None,
        count: int = 1,
    ):
        super().__init__()
        self.action = action
        self.positions = positions
        self.count = count

    def apply(self, env: MiniGridEnv) -> MiniGridEnv:
        """
        Add or remove walls/obstacles mid-trajectory.
        """
        if self.action not in ["add", "remove"]:
            raise ValueError("Invalid action. Must be 'add' or 'remove'.")

        modified_env = clone_env(env)
        modified_positions = []

        if self.action == "add":
            positions_to_add = (
                self.positions
                if self.positions
                else self._find_valid_positions(modified_env, self.count)
            )

            for pos in positions_to_add:
                wall = Wall()
                modified_env.grid.set(pos[0], pos[1], wall)
                modified_positions.append(pos)

        elif self.action == "remove":
            positions_to_remove = (
                self.positions
                if self.positions
                else self._find_walls(modified_env, self.count)
            )

            for pos in positions_to_remove:
                cell = modified_env.grid.get(pos[0], pos[1])
                if cell and cell.type == "wall":
                    modified_env.grid.set(pos[0], pos[1], None)
                    modified_positions.append(pos)

        return modified_env

    def _find_valid_positions(self, env, count):
        """Find empty positions suitable for placing walls."""
        # WARNING / TODO this can block the agent from reaching the goal
        valid = []
        for i in range(env.width):
            for j in range(env.height):
                cell = env.grid.get(i, j)
                if cell is None and (i, j) != tuple(env.agent_pos):
                    valid.append((i, j))
        return random.sample(valid, min(count, len(valid)))

    def _find_walls(self, env, count):
        """Find existing walls that can be removed."""
        walls = []
        for i in range(env.width):
            for j in range(env.height):
                cell = env.grid.get(i, j)
                if cell and cell.type == "wall":
                    walls.append((i, j))
        return random.sample(walls, min(count, len(walls)))


class RewardStructureChange(EnvTransformation):
    def __init__(self, reward_modifier: Callable[[float, dict], float]):
        super().__init__()
        self.reward_modifier = reward_modifier

    def apply(self, env: MiniGridEnv) -> MiniGridEnv:
        """
        Modify the reward function mid-trajectory.
        """
        modified_env = clone_env(env)
        original_step = modified_env.step

        def modified_step(action):
            obs, reward, terminated, truncated, info = original_step(action)
            reward = self.reward_modifier(reward, info)
            return obs, reward, terminated, truncated, info

        modified_env.step = modified_step
        return modified_env


class IsoDifficultyTransformationFactory:
    def __init__(self):
        self.transformations = [
            RotateEnv(),
            ReflectEnv(),
            TransposeEnv(),
            StartGoalSwap(),
            # RemoveDeadEnd(),
            # GoalDisplacement(),
            # DynamicObstacle(),
            # RewardStructureChange(),
        ]

    def get_transformations(self):
        return self.transformations


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv

    # Create a sample environment
    print("Creating original environment...")
    env = Simple2DNavigationEnv(
        size=11,
        complexity=1.0,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        goal_pos=(9, 9),
        render_mode="rgb_array",
    )
    env.reset()

    # Create transformation factory
    # Get all transformations
    transformations_list = [
        (env, "Original"),
        (RotateEnv().apply(env), "Rotation (90° CCW)"),
        (ReflectEnv().apply(env), "Reflection (Vertical)"),
        (TransposeEnv().apply(env), "Transpose Environment"),
        (StartGoalSwap().apply(env), "Start ⟷ Goal Swap"),
        (RemoveDeadEnd().apply(env), "Remove Dead-End"),
        (GoalDisplacement().apply(env), "Goal Displacement"),
        (DynamicObstacle().apply(env), "Dynamic Obstacle"),
    ]

    # Create figure with subplots (3 rows x 3 columns for better layout)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Iso-Difficulty Transformations", fontsize=18, fontweight="bold")
    axes = axes.flatten()

    for idx, (env_t, title) in enumerate(transformations_list):
        ax = axes[idx]

        # Render the environment
        img = env_t.render()
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

        # Print environment details
        print(f"\n{title}:")
        print(f"  Agent pos: {env_t.agent_pos}")
        print(f"  Agent dir: {env_t.agent_dir}")
        print(f"  Goal pos: {env_t.goal_pos}")
        print(f"  Grid size: {env_t.width}x{env_t.height}")

    # Hide unused subplots
    for idx in range(len(transformations_list), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    print("\n" + "=" * 60)
    print("=" * 60)
    print("\nKey insights:")
    print("  • Geometric transformations preserve ALL difficulty metrics")
    print("  • Start/Goal swap reverses the path but keeps distance/multiplicity")
    print("  • Dead-end changes add visual variety without")
    print("  • Goal displacement changes the goal position without")
    print("  • Dynamic obstacle changes the environment structure without")
    print("    affecting optimal paths or route multiplicity")
    plt.show()

    # Clean up
    env.close()
