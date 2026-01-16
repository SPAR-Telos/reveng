import random
from enum import IntEnum

from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv


class Simple2DNavigationEnv(MiniGridEnv):
    # https://minigrid.farama.org/content/create_env_tutorial/
    class Actions(IntEnum):
        LEFT = 0
        RIGHT = 1
        UP = 2
        DOWN = 3

    class ExtendedActions(IntEnum):
        # https://docs.python.org/3/howto/enum.html#restricted-enum-subclassing
        # Creating a subclass of an Enum with numbers is not possible
        LEFT = 0
        RIGHT = 1
        UP = 2
        DOWN = 3
        QUIT = 4

    def __init__(
        self,
        size=11,
        complexity=0.0,  # 0.0: empty room, 1.0: perfect maze
        agent_start_dir: int | None = None,
        agent_start_pos: tuple[int, int] | None = None,
        goal_pos: tuple[int, int] | None = None,
        max_steps: int | None = None,
        allow_quit_action: bool = False,
        **kwargs,
    ):
        self.complexity = max(0.0, min(1.0, complexity))  # Clamp between 0 and 1
        self.agent_start_pos_user = agent_start_pos
        self.agent_start_dir_user = agent_start_dir
        self.goal_pos_user = goal_pos
        self.allow_quit_action = allow_quit_action

        if size % 2 == 0:
            size += 1
            print(f"Grid size must be odd for maze generation. Adjusting to {size}.")

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

        if self.allow_quit_action:
            self.actions = Simple2DNavigationEnv.ExtendedActions
        else:
            self.actions = Simple2DNavigationEnv.Actions

        self.action_space = spaces.Discrete(len(self.actions))

        self._action_to_direction = {
            self.actions.LEFT: 2,
            self.actions.RIGHT: 0,
            self.actions.UP: 3,
            self.actions.DOWN: 1,
        }

    @staticmethod
    def _gen_mission():
        return "Reach the Goal"

    def _generate_perfect_maze(self, width, height):
        """Generates a perfect maze using randomized Depth-First Search."""
        # Start with a grid full of walls
        for i in range(width):
            for j in range(height):
                self.grid.set(i, j, Wall())

        start_x, start_y = (
            random.randint(0, (width - 3) // 2) * 2 + 1,
            random.randint(0, (height - 3) // 2) * 2 + 1,
        )

        stack = [(start_x, start_y)]
        self.grid.set(start_x, start_y, None)

        while stack:
            cx, cy = stack[-1]
            neighbors = []

            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if (
                    0 < nx < width - 1
                    and 0 < ny < height - 1
                    and self.grid.get(nx, ny)
                    and self.grid.get(nx, ny).type == "wall"
                ):
                    neighbors.append((nx, ny))

            if neighbors:
                nx, ny = random.choice(neighbors)
                self.grid.set(nx, ny, None)
                self.grid.set((cx + nx) // 2, (cy + ny) // 2, None)
                stack.append((nx, ny))
            else:
                stack.pop()

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # 1. Generate a perfect, complex maze
        self._generate_perfect_maze(width, height)

        # 2. Identify all internal walls that can be removed
        internal_walls = []
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                if self.grid.get(i, j) is not None:
                    internal_walls.append((i, j))

        random.shuffle(internal_walls)

        # 3. Calculate how many walls to remove
        num_walls_to_remove = int((1.0 - self.complexity) * len(internal_walls))

        # 4. Remove the walls
        for i in range(num_walls_to_remove):
            self.grid.set(internal_walls[i][0], internal_walls[i][1], None)

        # 5. Place Agent and Goal in valid empty spaces
        empty_cells = []
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                if self.grid.get(i, j) is None:
                    empty_cells.append((i, j))

        if len(empty_cells) < 2:
            raise Exception("Maze generation failed, not enough empty cells.")

        # Place Goal
        self.goal_pos = (
            self.goal_pos_user
            if self.goal_pos_user in empty_cells
            else random.choice(empty_cells)
        )
        self.put_obj(Goal(), *self.goal_pos)
        empty_cells.remove(self.goal_pos)

        # Place Agent
        self.agent_pos = (
            self.agent_start_pos_user
            if self.agent_start_pos_user in empty_cells
            else random.choice(empty_cells)
        )
        self.agent_dir = (
            self.agent_start_dir_user
            if self.agent_start_dir_user is not None
            else random.randint(0, 3)
        )

        # Store initial positions for safe_reset
        self._initial_agent_pos = self.agent_pos
        self._initial_agent_dir = self.agent_dir

        self.mission = "Reach the Goal"

    def step(self, action):
        self.step_count += 1
        reward = 0
        terminated = False
        truncated = False

        # Handle QUIT action
        if (
            hasattr(self, "allow_quit_action")
            and self.allow_quit_action
            and action == self.ExtendedActions.QUIT
        ):
            terminated = True
            obs = self.gen_obs()
            return obs, reward, terminated, truncated, {}

        # Set agent's direction based on the action
        if action in self._action_to_direction:
            self.agent_dir = self._action_to_direction[action]
        else:
            raise ValueError(f"Unknown action: {action}")

        # Get the position in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        # Move if the cell is empty or can be overlapped
        if fwd_cell is None or fwd_cell.can_overlap():
            self.agent_pos = tuple(fwd_pos)

        if self.agent_pos == self.goal_pos:
            terminated = True
            reward = self._reward()

        # Check if the episode is truncated
        if self.step_count >= self.max_steps:
            truncated = True

        obs = self.gen_obs()
        return obs, reward, terminated, truncated, {}

    def set_env_from_list(self, grid_list):
        """
        Set the environment from a list of lists with strings.

        Args:
            grid_list: List of lists where each string represents a cell.
                Uses symbols from DEFAULT_CELLS_CONFIG in text_obs_wrapper.py:
                'A' - Agent position
                '#' - Wall
                'G' - Goal
                '_' - Empty space (open space)

        Example:
            grid = [
                ['#', '#', '#', '#', '#'],
                ['#', 'A', '_', '_', '#'],
                ['#', '_', '#', '_', '#'],
                ['#', '_', '_', 'G', '#'],
                ['#', '#', '#', '#', '#']
            ]
        """
        height = len(grid_list)
        width = len(grid_list[0]) if height > 0 else 0

        # Recreate the grid with the specified dimensions
        self.grid = Grid(width, height)

        agent_pos = None
        goal_pos = None

        # Parse the grid list and place objects
        # Using symbols from DEFAULT_CELLS_CONFIG:
        # 'A' = agent, '#' = wall, 'G' = goal, '_' = empty
        for j, row in enumerate(grid_list):
            for i, cell in enumerate(row):
                cell_str = cell.strip()

                if cell_str == "#":
                    self.grid.set(i, j, Wall())
                elif cell_str == "A":
                    agent_pos = (i, j)
                    self.grid.set(i, j, None)  # Agent position is empty
                elif cell_str == "G":
                    goal_pos = (i, j)
                    self.put_obj(Goal(), i, j)
                elif cell_str == "_":
                    self.grid.set(i, j, None)  # Empty space
                else:
                    # Default to empty for unknown characters
                    self.grid.set(i, j, None)

        # Set agent position and direction
        if agent_pos is None:
            raise ValueError("Agent position 'A' not found in grid_list")
        self.agent_pos = agent_pos
        self.agent_dir = (
            self.agent_start_dir_user if self.agent_start_dir_user is not None else 0
        )
        self.goal_pos = goal_pos

        # Store initial positions for safe_reset
        self._initial_agent_pos = self.agent_pos
        self._initial_agent_dir = self.agent_dir

        self.mission = "Reach the Goal"

    def reset(self, **kwargs):
        print(
            "*******************************************WARNING: YOU HAVE CALLED RESET METHOD. THIS WILL REGENERATE THE GRID!!!*******************************************"
        )
        return super().reset(**kwargs)

    def safe_reset(self):
        """Reset the agent to its starting position without regenerating the grid.

        This method resets the agent's position and direction to the initial values
        from when the grid was generated or set, but keeps the grid layout unchanged.
        Use this when you want to run multiple trajectories on the same grid.

        Returns:
            tuple[ObsType, dict]: A tuple of (observation, info_dict).

        Raises:
            RuntimeError: If called before the environment has been initialized
                (i.e., before reset() or set_env_from_list() has been called).
        """
        if not hasattr(self, "_initial_agent_pos") or self._initial_agent_pos is None:
            raise RuntimeError(
                "safe_reset() called before environment was initialized. "
                "Call reset() or set_env_from_list() first."
            )

        # Reset agent to initial position and direction
        self.agent_pos = self._initial_agent_pos
        self.agent_dir = self._initial_agent_dir

        # Reset episode state
        self.step_count = 0
        self.carrying = None

        # Generate and return observation
        obs = self.gen_obs()
        return obs, {}


if __name__ == "__main__":
    import time

    print("--- Low complexity maze (complexity=0.2) ---")
    low_complexity_env = Simple2DNavigationEnv(
        size=21, complexity=0.2, render_mode="human"
    )
    low_complexity_env.reset()
    low_complexity_env.render()
    time.sleep(3)
    low_complexity_env.close()

    print("--- Highest complexity maze (complexity=1.0) ---")
    high_complexity_env = Simple2DNavigationEnv(
        size=21, complexity=1.0, render_mode="human"
    )
    high_complexity_env.reset()
    high_complexity_env.render()
    time.sleep(3)
    high_complexity_env.close()

    print("--- Custom environment from list ---")
    custom_env = Simple2DNavigationEnv(size=7, render_mode="human")
    custom_grid = [
        ["#", "#", "#", "#", "#", "#", "#"],
        ["#", "A", "_", "_", "_", "_", "#"],
        ["#", "_", "#", "#", "#", "_", "#"],
        ["#", "_", "_", "_", "#", "_", "#"],
        ["#", "#", "#", "_", "#", "_", "#"],
        ["#", "_", "_", "_", "_", "G", "#"],
        ["#", "#", "#", "#", "#", "#", "#"],
    ]
    custom_env.reset()
    custom_env.set_env_from_list(custom_grid)
    custom_env.render()
    time.sleep(3)
    custom_env.close()
