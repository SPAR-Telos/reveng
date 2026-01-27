import random
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall, Door, Key
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv


class RoomsMinigridEnv(Simple2DNavigationEnv):
    """
    A multi-room environment where each room is 3x3 and connected via unique paths.

    Supports 2x2 and 3x3 room layouts:

    2x2 layout (4 rooms):
    [Room 0] [Room 1]
    [Room 2] [Room 3]

    3x3 layout (9 rooms):
    [Room 0] [Room 1] [Room 2]
    [Room 3] [Room 4] [Room 5]
    [Room 6] [Room 7] [Room 8]

    Each room is connected to adjacent rooms through single openings in walls,
    forming a unique path structure (typically a spanning tree).
    """

    def __init__(
        self,
        agent_start_dir: int | None = None,
        agent_start_pos: tuple[int, int] | None = None,
        goal_pos: tuple[int, int] | None = None,
        max_steps: int | None = None,
        allow_quit_action: bool = False,
        add_door_key: bool = True,
        rooms_per_side: int = 3,
        **kwargs,
    ):
        # Size calculation: rooms_per_side rooms * 3 cells per room + (rooms_per_side + 1) walls
        # For 2x2: 2 * 3 + 3 = 9
        # For 3x3: 3 * 3 + 4 = 13
        if rooms_per_side not in [2, 3]:
            raise ValueError("rooms_per_side must be 2 or 3")

        self.rooms_per_side = rooms_per_side
        size = rooms_per_side * 3 + (rooms_per_side + 1)
        self.add_door_key = add_door_key

        super().__init__(
            size=size,
            complexity=0.0,
            agent_start_dir=agent_start_dir,
            agent_start_pos=agent_start_pos,
            goal_pos=goal_pos,
            max_steps=max_steps,
            allow_quit_action=allow_quit_action,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        """Generate a grid with 9 rooms (3x3 layout) connected via unique paths."""
        # Reset carrying state (use MiniGrid's standard inventory)
        self.carrying = None

        self.grid = Grid(width, height)

        # First, fill everything with walls
        for i in range(width):
            for j in range(height):
                self.grid.set(i, j, Wall())

        # Define the rooms (each is 3x3 interior space)
        # Room positions in the grid (top-left corner of interior)
        room_size = 3
        wall_thickness = 1
        room_coords = []

        for row in range(self.rooms_per_side):
            for col in range(self.rooms_per_side):
                x = wall_thickness + col * (room_size + wall_thickness)
                y = wall_thickness + row * (room_size + wall_thickness)
                room_coords.append((x, y))

        # Clear the interior of each room
        for x, y in room_coords:
            for i in range(room_size):
                for j in range(room_size):
                    self.grid.set(x + i, y + j, None)

        # Create a spanning tree to connect rooms with unique paths
        # Rooms are numbered 0-8 in row-major order
        connections = self._generate_room_connections()

        # Create openings based on connections and track doorway positions
        doorway_positions = []
        for room1, room2 in connections:
            doorway_pos = self._create_opening(
                room1, room2, room_coords, room_size, wall_thickness
            )
            doorway_positions.append(doorway_pos)

        # Collect all empty cells for agent and goal placement, excluding doorways
        empty_cells = []
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                if self.grid.get(i, j) is None and (i, j) not in doorway_positions:
                    empty_cells.append((i, j))

        if len(empty_cells) < 2:
            raise Exception("Not enough empty cells for agent and goal placement.")

        # Place agent and goal based on whether we need to add door/key
        if self.add_door_key:
            # Try multiple times to find valid agent/goal positions where a door can block the path
            max_attempts = 200
            door_placed = False

            for attempt in range(max_attempts):
                # Place Goal
                if (
                    attempt == 0
                    and self.goal_pos_user
                    and self.goal_pos_user in empty_cells
                ):
                    self.goal_pos = self.goal_pos_user
                else:
                    self.goal_pos = random.choice(empty_cells)

                # Place Agent
                available_for_agent = [
                    cell for cell in empty_cells if cell != self.goal_pos
                ]
                if (
                    attempt == 0
                    and self.agent_start_pos_user
                    and self.agent_start_pos_user in available_for_agent
                ):
                    self.agent_pos = self.agent_start_pos_user
                else:
                    self.agent_pos = random.choice(available_for_agent)

                # Try to find a doorway that blocks the path from agent to goal
                door_pos = self._find_blocking_doorway(
                    self.agent_pos, self.goal_pos, doorway_positions, width, height
                )

                if door_pos:
                    # Find all cells reachable from agent position without going through the door
                    reachable_cells = self._find_reachable_cells(
                        self.agent_pos, door_pos, width, height
                    )

                    # Check if we can place a key in a reachable location
                    key_cells = [
                        cell
                        for cell in reachable_cells
                        if cell != self.agent_pos and cell not in doorway_positions
                    ]

                    if key_cells and self.goal_pos not in reachable_cells:
                        # Success! We found valid positions
                        door = Door("yellow", is_locked=True)
                        self.put_obj(door, door_pos[0], door_pos[1])

                        key_pos = random.choice(key_cells)
                        key = Key("yellow")
                        self.put_obj(key, key_pos[0], key_pos[1])

                        door_placed = True
                        break

            # If we failed to place door/key after max attempts, regenerate the entire grid
            if not door_placed:
                return self._gen_grid(width, height)
        else:
            # Simple placement without door/key
            # Place Goal
            self.goal_pos = (
                self.goal_pos_user
                if self.goal_pos_user and self.goal_pos_user in empty_cells
                else random.choice(empty_cells)
            )

            # Place Agent
            available_for_agent = [
                cell for cell in empty_cells if cell != self.goal_pos
            ]
            self.agent_pos = (
                self.agent_start_pos_user
                if self.agent_start_pos_user
                and self.agent_start_pos_user in available_for_agent
                else random.choice(available_for_agent)
            )

        # Place goal object
        self.put_obj(Goal(), *self.goal_pos)

        self.agent_dir = (
            self.agent_start_dir_user
            if self.agent_start_dir_user is not None
            else random.randint(0, 3)
        )

        self.mission = "Reach the Goal"

    def step(self, action):
        """Override step to handle automatic key pickup and door opening."""
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

        # Move if the cell is empty, can be overlapped, is an open door, or contains a key
        can_move = False
        if fwd_cell is None or fwd_cell.can_overlap():
            can_move = True
        elif fwd_cell.type == "door" and fwd_cell.is_open:
            can_move = True
        elif fwd_cell.type == "key":
            can_move = True  # Allow moving onto key to pick it up

        if can_move:
            self.agent_pos = tuple(fwd_pos)

        # Check if agent is on a key and pick it up automatically
        current_cell = self.grid.get(*self.agent_pos)
        if current_cell is not None and current_cell.type == "key":
            self.carrying = current_cell
            # Remove the key from the grid
            self.grid.set(*self.agent_pos, None)

        # Check all adjacent cells for locked doors and open them if we have the key
        if self.carrying is not None and self.carrying.type == "key":
            x, y = self.agent_pos
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                adj_x, adj_y = x + dx, y + dy
                adj_cell = self.grid.get(adj_x, adj_y)
                if (
                    adj_cell is not None
                    and adj_cell.type == "door"
                    and adj_cell.is_locked
                ):
                    # Remove the door from the grid
                    self.grid.set(adj_x, adj_y, None)
                    # Previous behavior (kept for reference):
                    # adj_cell.is_locked = False
                    # adj_cell.is_open = True

        # Check if agent reached the goal
        if self.agent_pos == self.goal_pos:
            terminated = True
            reward = self._reward()

        # Check if the episode is truncated
        if self.step_count >= self.max_steps:
            truncated = True

        obs = self.gen_obs()
        return obs, reward, terminated, truncated, {}

    def _find_blocking_doorway(
        self, agent_pos, goal_pos, doorway_positions, width, height
    ):
        """
        Find a doorway that, when blocked, prevents the agent from reaching the goal.
        Tests each doorway to see if blocking it makes the goal unreachable.
        """
        random.shuffle(doorway_positions)

        for doorway in doorway_positions:
            # Check if blocking this doorway makes the goal unreachable
            reachable = self._find_reachable_cells(agent_pos, doorway, width, height)
            if goal_pos not in reachable:
                # This doorway blocks the path to the goal
                return doorway

        # No doorway blocks the path (shouldn't happen with spanning tree)
        return None

    def _find_reachable_cells(self, start_pos, blocked_pos, width, height):
        """
        Find all cells reachable from start_pos without going through blocked_pos.
        Uses BFS to explore the grid.
        """
        from collections import deque

        reachable = set()
        queue = deque([start_pos])
        reachable.add(start_pos)

        while queue:
            x, y = queue.popleft()

            # Check all 4 adjacent cells
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy

                # Skip if out of bounds
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue

                # Skip if already visited
                if (nx, ny) in reachable:
                    continue

                # Skip if this is the blocked position (door)
                if (nx, ny) == blocked_pos:
                    continue

                # Skip if there's a wall
                cell = self.grid.get(nx, ny)
                if cell is not None and cell.type == "wall":
                    continue

                # Add to reachable set and queue
                reachable.add((nx, ny))
                queue.append((nx, ny))

        return list(reachable)

    def _generate_room_connections(self):
        """
        Generate a spanning tree of room connections using randomized Kruskal's algorithm.
        Returns a list of tuples (room1, room2) representing connections.
        """
        # All possible connections between adjacent rooms
        edges = []

        # Horizontal connections (room to room on the right)
        for row in range(self.rooms_per_side):
            for col in range(self.rooms_per_side - 1):
                room1 = row * self.rooms_per_side + col
                room2 = row * self.rooms_per_side + col + 1
                edges.append((room1, room2))

        # Vertical connections (room to room below)
        for row in range(self.rooms_per_side - 1):
            for col in range(self.rooms_per_side):
                room1 = row * self.rooms_per_side + col
                room2 = (row + 1) * self.rooms_per_side + col
                edges.append((room1, room2))

        # Shuffle edges for randomization
        random.shuffle(edges)

        # Kruskal's algorithm with Union-Find
        num_rooms = self.rooms_per_side * self.rooms_per_side
        parent = list(range(num_rooms))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        connections = []
        for room1, room2 in edges:
            if union(room1, room2):
                connections.append((room1, room2))
                if len(connections) == num_rooms - 1:  # Spanning tree has n-1 edges
                    break

        return connections

    def _create_opening(self, room1, room2, room_coords, room_size, wall_thickness):
        """
        Create an opening in the wall between two adjacent rooms.
        Returns the (x, y) position of the doorway.
        """
        r1_x, r1_y = room_coords[room1]
        r2_x, r2_y = room_coords[room2]

        # Determine if rooms are horizontally or vertically adjacent
        if r1_y == r2_y:  # Horizontal adjacency
            # Remove one cell from the vertical wall between them
            wall_x = max(r1_x, r2_x) - wall_thickness
            wall_y = r1_y + random.randint(0, room_size - 1)
            self.grid.set(wall_x, wall_y, None)
            return (wall_x, wall_y)
        else:  # Vertical adjacency
            # Remove one cell from the horizontal wall between them
            wall_x = r1_x + random.randint(0, room_size - 1)
            wall_y = max(r1_y, r2_y) - wall_thickness
            self.grid.set(wall_x, wall_y, None)
            return (wall_x, wall_y)

    def set_env_from_list(self, grid_list):
        """
        Set the environment from a list of lists with strings.

        Extends the base implementation to support doors and keys.

        Args:
            grid_list: List of lists where each string represents a cell.
                Supported symbols:
                'A' - Agent position
                '#' - Wall
                'G' - Goal
                '_' - Empty space (open space)
                'K' - Key (yellow)
                'D' - Door (yellow, locked)
                'O' - Door (yellow, open)

        Example:
            grid = [
                ['#', '#', '#', '#', '#'],
                ['#', 'A', '_', 'K', '#'],
                ['#', '_', '#', 'D', '#'],
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

        # Reset carrying state
        self.carrying = None

        # Parse the grid list and place objects
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
                elif cell_str == "K":
                    # Place a yellow key
                    key = Key("yellow")
                    self.put_obj(key, i, j)
                elif cell_str == "D":
                    # Place a yellow locked door
                    door = Door("yellow", is_locked=True)
                    self.put_obj(door, i, j)
                elif cell_str == "O":
                    # Place a yellow open door
                    door = Door("yellow", is_locked=False)
                    door.is_open = True
                    self.put_obj(door, i, j)
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

        self.mission = "Reach the Goal"


if __name__ == "__main__":
    import pygame
    from reveng.environment_generator.wrappers.text_obs_wrapper import (
        FullObservabilityTextWrapper,
    )
    from reveng.environment_generator.utils import remove_door

    # Manual control for 4-Room Environment (2x2)
    print("--- Manual Control: 4-Room Environment (2x2) without door ---")
    print("Use arrow keys to move the agent")
    print("The environment has a key but the door has been removed")

    base_env = RoomsMinigridEnv(render_mode="human", add_door_key=True)
    env = FullObservabilityTextWrapper(base_env)
    obs, info = env.reset()

    # Remove the door after reset
    env.unwrapped.grid = remove_door(env.unwrapped).grid

    # Map pygame keys to environment actions
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

                    print(f"\nStep: {base_env.step_count}")
                    print(f"Action: {base_env.actions(action).name}")
                    print(
                        f"Carrying: {base_env.carrying.type if base_env.carrying else 'Nothing'}"
                    )
                    print(
                        f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}"
                    )
                    print("\nCurrent Grid State:")
                    print(obs)

                    if terminated:
                        print("\n🎉 Goal reached!")
                        print("Press any key to reset or close the window to exit")
                    elif truncated:
                        print("\n⏰ Time limit reached!")
                        print("Press any key to reset or close the window to exit")

                    if terminated or truncated:
                        # Wait for next key press or window close
                        waiting = True
                        while waiting:
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    running = False
                                    waiting = False
                                elif event.type == pygame.KEYDOWN:
                                    env.reset()
                                    waiting = False
                                    print("\n--- Environment Reset ---\n")

    env.close()
