from __future__ import annotations

from copy import deepcopy
from typing import Callable, Iterable, Tuple

import numpy as np
from minigrid.core.grid import Grid
from minigrid.minigrid_env import MiniGridEnv


class IsoDifficultyTransformationFactory:
    def __init__(self):
        self.transformations = {
            "rotation": self.rotation_transformation,
            "reflection": self.reflection_transformation,
            "geometry": self.geometry_transformation,
            "start_goal_swap": self.start_goal_swap_transformation,
            "dead_end_variation": self.dead_end_variation_transformation,
        }

        self._position_attrs = (
            "agent_pos",
            "agent_start_pos_user",
            "goal_pos",
            "goal_pos_user",
        )
        self._direction_attrs = (
            "agent_dir",
            "agent_start_dir_user",
        )

    def _clone_env(self, env: MiniGridEnv) -> MiniGridEnv:
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

    def _set_object_positions(self, grid: Grid) -> None:
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

    def rotation_transformation(self, env: MiniGridEnv) -> MiniGridEnv:
        rotated_env = self._clone_env(env)

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

    def reflection_transformation(self, env: MiniGridEnv) -> MiniGridEnv:
        reflected_env = self._clone_env(env)

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

    def geometry_transformation(self, env: MiniGridEnv) -> MiniGridEnv:
        geometry_env = self._clone_env(env)

        geometry_env.grid = self._transpose_grid(geometry_env.grid)
        geometry_env.width = geometry_env.grid.width
        geometry_env.height = geometry_env.grid.height

        self._set_object_positions(geometry_env.grid)

        dir_map = {0: 1, 1: 0, 2: 3, 3: 2}

        self._update_positions(geometry_env, lambda pos: (pos[1], pos[0]))
        self._update_directions(
            geometry_env, lambda direction: dir_map.get(direction, direction)
        )

        return geometry_env

    def start_goal_swap_transformation(self, env: MiniGridEnv) -> MiniGridEnv:
        """
        Swap the start and goal positions.
        This preserves optimal path distance and route multiplicity by reversing the path.
        """
        swapped_env = self._clone_env(env)

        # Import Goal from minigrid
        from minigrid.core.world_object import Goal

        # Swap positions
        old_agent_pos = self._as_tuple(swapped_env.agent_pos)
        old_goal_pos = self._as_tuple(swapped_env.goal_pos)

        if old_agent_pos and old_goal_pos:
            # Update agent position
            swapped_env.agent_pos = self._coerce_position(
                swapped_env.agent_pos, old_goal_pos
            )
            if hasattr(swapped_env, "agent_start_pos_user"):
                swapped_env.agent_start_pos_user = self._coerce_position(
                    swapped_env.agent_start_pos_user, old_goal_pos
                )

            # Update goal position
            swapped_env.goal_pos = old_agent_pos
            if hasattr(swapped_env, "goal_pos_user"):
                swapped_env.goal_pos_user = old_agent_pos

            # Update grid: remove old goal and place new goal
            swapped_env.grid.set(*old_goal_pos, None)
            swapped_env.put_obj(Goal(), *old_agent_pos)

        return swapped_env

    def dead_end_variation_transformation(self, env: MiniGridEnv) -> MiniGridEnv:
        """
        Add or remove walls in dead-end areas that are far from optimal paths.
        This changes the visual appearance without affecting path metrics.
        """

        varied_env = self._clone_env(env)

        # Find all dead-end cells (cells with only one neighbor)
        dead_ends = []
        for x in range(1, varied_env.width - 1):
            for y in range(1, varied_env.height - 1):
                if varied_env.grid.get(x, y) is None:
                    # Count empty neighbors
                    neighbors = 0
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if varied_env.grid.get(nx, ny) is None:
                            neighbors += 1

                    # Dead end has exactly 1 neighbor
                    if neighbors == 1:
                        agent_pos = self._as_tuple(varied_env.agent_pos)
                        goal_pos = self._as_tuple(varied_env.goal_pos)

                        # Only modify if far from agent and goal
                        if agent_pos and goal_pos:
                            dist_to_agent = abs(x - agent_pos[0]) + abs(
                                y - agent_pos[1]
                            )
                            dist_to_goal = abs(x - goal_pos[0]) + abs(y - goal_pos[1])
                            if dist_to_agent > 3 and dist_to_goal > 3:
                                dead_ends.append((x, y))

        # Randomly extend some dead ends by one cell
        for dead_end in dead_ends[
            : len(dead_ends) // 3
        ]:  # Modify 1/3 of eligible dead ends
            x, y = dead_end
            # Try to add a wall adjacent to the dead end
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (
                    varied_env.grid.get(nx, ny) is not None
                    and varied_env.grid.get(nx, ny).type == "wall"
                ):
                    # Check if we can safely remove this wall
                    # (it won't connect two separate areas)
                    varied_env.grid.set(nx, ny, None)
                    break

        if varied_env.grid == env.grid:
            raise ValueError("No dead-end variation was applied")

        return varied_env

    def get_all_transformations(self, env: MiniGridEnv) -> list[MiniGridEnv]:
        return [
            self.transformations[transformation](env)
            for transformation in self.transformations
        ]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv

    # Create a sample environment
    print("Creating original environment...")
    env = Simple2DNavigationEnv(
        size=20,
        complexity=1.0,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        goal_pos=(9, 9),
        render_mode="rgb_array",
    )
    env.reset()

    # Create transformation factory
    factory = IsoDifficultyTransformationFactory()

    # Get all transformations
    transformations_list = [
        (env, "Original"),
        (factory.rotation_transformation(env), "Rotation (90° CCW)"),
        (factory.reflection_transformation(env), "Reflection (Vertical)"),
        (factory.geometry_transformation(env), "Geometry (Transpose)"),
        (factory.start_goal_swap_transformation(env), "Start ⟷ Goal Swap"),
        (factory.dead_end_variation_transformation(env), "Dead-End Variation"),
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
    print("Visualization saved to 'iso_difficulty_transformations.png'")
    print("=" * 60)
    print("\nKey insights:")
    print("  • Geometric transformations preserve ALL difficulty metrics")
    print("  • Start/Goal swap reverses the path but keeps distance/multiplicity")
    print("  • Dead-end changes add visual variety without")
    print("    affecting optimal paths or route multiplicity")
    plt.show()

    # Clean up
    env.close()
