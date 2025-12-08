import random, sys, os
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall
sys.path.append(os.path.join('C:/Users\hchen\Dropbox/reveng', "src"))
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv

#%%
class CorridorMazeEnv(Simple2DNavigationEnv):
    """
    A 11x11 maze with two corridors leading to the goal:
    1. Direct corridor: shortest path to goal
    2. Detour corridor: longer path with coins equal to the path length difference

    The maze walls block all other paths, leaving only these two 1-cell wide corridors.
    """

    def __init__(
        self,
        agent_start_dir: Optional[int] = None,
        agent_start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
        max_steps: Optional[int] = None,
        allow_quit_action: bool = False,
        **kwargs,
    ):
        # Fixed size for this maze type
        size = 11

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
        """Generate an 11x11 grid with two corridors and coins on the detour path."""
        self.carrying = None
        self.grid = Grid(width, height)

        # Fill everything with walls initially
        for i in range(width):
            for j in range(height):
                self.grid.set(i, j, Wall())

        # Generate the two corridor paths
        direct_path, detour_path = self._generate_corridor_paths(width, height)

        # Clear the corridor cells (remove walls)
        all_corridor_cells = set(direct_path + detour_path)
        for x, y in all_corridor_cells:
            self.grid.set(x, y, None)

        # Calculate path length difference
        path_diff = len(detour_path) - len(direct_path)

        # Place coins on detour path (excluding start and goal positions)
        detour_cells_for_coins = [
            cell for cell in detour_path
            if cell not in [direct_path[0], direct_path[-1]]  # Exclude start and goal
        ]

        # Place coins equal to path difference
        if path_diff > 0 and len(detour_cells_for_coins) >= path_diff:
            coin_positions = random.sample(detour_cells_for_coins, min(path_diff, len(detour_cells_for_coins)))

            # For visualization, we'll use a custom object type or mark these cells
            # Since MiniGrid doesn't have coins, we'll create a simple marker
            for pos in coin_positions:
                # We'll store coin positions and handle them in rendering
                pass

            self.coin_positions = coin_positions
        else:
            self.coin_positions = []

        # Set agent and goal positions
        self.agent_pos = direct_path[0]  # Start of direct path
        self.goal_pos = direct_path[-1]  # End of direct path (same for both paths)

        # Place goal object
        self.put_obj(Goal(), *self.goal_pos)

        # Set random agent direction
        self.agent_dir = (
            self.agent_start_dir_user
            if self.agent_start_dir_user is not None
            else random.randint(0, 3)
        )

        self.mission = "Reach the Goal via Direct or Detour Path"

        # Store paths for analysis
        self.direct_path = direct_path
        self.detour_path = detour_path
        self.path_difference = path_diff

    import random

    def _generate_corridor_paths(self, width, height,
                                start_pos=(1,7), goal_pos=(9,2),
                                detour_extra_steps=5,
                                max_attempts=5000):
        """
        Generate two corridor paths: a shortest 'direct' path and a longer 'detour' path.

        Constraints:
        - Direct and detour only share start_pos and goal_pos.
        - No detour cell (except neighbors in sequence) is 4-neighbor-adjacent to another detour cell.
        => The detour corridor never "touches itself".
        - Detour has at least `detour_extra_steps` more cells than direct.
        - Works for arbitrary start_pos and goal_pos inside the inner grid.

        Returns:
            direct_path, detour_path  (each: list[(x, y)])
        """
        def in_bounds(x, y):
            # Keep corridors away from hard outer border, like your original code
            return 1 <= x < width - 1 and 1 <= y < height - 1

        # 4-neighborhood moves
        MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        # ----- 1) direct path: simple Manhattan shortest path -----
        sx, sy = start_pos
        gx, gy = goal_pos

        if not (in_bounds(sx, sy) and in_bounds(gx, gy)):
            raise ValueError("start_pos and goal_pos must be inside the inner grid")

        direct_path = [(sx, sy)]
        x, y = sx, sy

        # Move horizontally towards goal
        if gx != x:
            step_x = 1 if gx > x else -1
            while x != gx:
                x += step_x
                direct_path.append((x, y))

        # Move vertically towards goal
        if gy != y:
            step_y = 1 if gy > y else -1
            while y != gy:
                y += step_y
                direct_path.append((x, y))

        direct_set = set(direct_path)

        # ----- 2) forbid overlapping or touching the direct path -----
        forbidden = set()

        for (dx, dy) in direct_path:
            pos = (dx, dy)
            if pos not in (start_pos, goal_pos):
                forbidden.add(pos)
                for mx, my in MOVES:
                    nx, ny = dx + mx, dy + my
                    npos = (nx, ny)
                    # allow adjacency to start/goal so they can be shared
                    if in_bounds(nx, ny) and npos not in (start_pos, goal_pos):
                        forbidden.add(npos)

        forbidden.discard(start_pos)
        forbidden.discard(goal_pos)

        # ----- 3) detour path via DFS with length + no-self-touch constraint -----
        min_len = len(direct_path) + detour_extra_steps
        max_len = (width - 2) * (height - 2) - len(forbidden)
        if min_len > max_len:
            raise ValueError(
                f"Requested detour_extra_steps={detour_extra_steps} is too large "
                f"for this grid and these start/goal positions."
            )

        goal_x, goal_y = gx, gy
        best_path = None
        attempts = 0

        def dfs(pos, path, visited):
            nonlocal best_path, attempts

            if best_path is not None:
                return  # already found a valid path

            attempts += 1
            if attempts > max_attempts:
                return  # give up

            # prune if path too long
            if len(path) > max_len:
                return

            if pos == goal_pos:
                if len(path) >= min_len:
                    best_path = list(path)
                return

            x, y = pos

            neighbors = []
            for mx, my in MOVES:
                nx, ny = x + mx, y + my
                npos = (nx, ny)

                if not in_bounds(nx, ny):
                    continue
                if npos in visited:
                    continue
                if npos in forbidden:
                    continue

                # NEW: ensure the detour doesn't "touch itself":
                # npos must not be 4-neighbor adjacent to ANY visited cell
                # except the current position 'pos'.
                touches_other = False
                for qx, qy in MOVES:
                    tx, ty = nx + qx, ny + qy
                    tpos = (tx, ty)
                    if tpos in visited and tpos != pos:
                        touches_other = True
                        break
                if touches_other:
                    continue

                neighbors.append(npos)

            # bias towards moving roughly towards the goal, but keep randomness
            random.shuffle(neighbors)
            neighbors.sort(key=lambda c: abs(c[0] - goal_x) + abs(c[1] - goal_y))

            for npos in neighbors:
                visited.add(npos)
                path.append(npos)
                dfs(npos, path, visited)
                if best_path is not None:
                    return
                path.pop()
                visited.remove(npos)

        visited = {start_pos}
        dfs(start_pos, [start_pos], visited)

        if best_path is None:
            raise RuntimeError(
                "Failed to find a valid detour path with the given constraints. "
                "Try reducing detour_extra_steps or using a larger grid."
            )

        detour_path = best_path

        # sanity checks
        assert detour_path[0] == start_pos and detour_path[-1] == goal_pos
        assert set(direct_path).isdisjoint(set(detour_path) - {start_pos, goal_pos})

        # Check no self-touch in detour (other than consecutive steps)
        detour_set = set(detour_path)
        for i, (x, y) in enumerate(detour_path):
            for mx, my in MOVES:
                nx, ny = x + mx, y + my
                npos = (nx, ny)
                if npos in detour_set:
                    # it’s allowed only if it's the previous or next cell in the path
                    if not (
                        i > 0 and detour_path[i - 1] == npos or
                        i < len(detour_path) - 1 and detour_path[i + 1] == npos
                    ):
                        raise RuntimeError("Detour path self-touches unexpectedly.")

        return direct_path, detour_path




def generate_and_plot_corridor_maze(seed=None):
    """
    Generate a corridor maze and plot it using matplotlib.

    Args:
        seed: Random seed for reproducible generation

    Returns:
        dict: Information about the generated maze
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Create the environment
    env = CorridorMazeEnv()
    env.reset()

    # Create the plot
    fig, ax = plt.subplots(figsize=(11, 11))

    # Create a grid to visualize the maze
    grid_size = 11
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)

    # Plot each cell type with specific colors
    for x in range(grid_size):
        for y in range(grid_size):
            cell = env.grid.get(x, y)
            if cell is None:  # Empty corridor cell
                # White cells for corridors
                rect = plt.Rectangle((x-0.5, y-0.5), 1, 1, facecolor='white', edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
            else:  # Wall
                # Gray cells for walls
                rect = plt.Rectangle((x-0.5, y-0.5), 1, 1, facecolor='gray', edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)

    # Plot coins as white cells with yellow circles
    for x, y in env.coin_positions:
        # White background
        rect = plt.Rectangle((x-0.5, y-0.5), 1, 1, facecolor='white', edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        # Yellow circle for coin
        circle = plt.Circle((x, y), 0.3, facecolor='gold', edgecolor='orange', linewidth=2)
        ax.add_patch(circle)

    # Plot agent start position
    agent_x, agent_y = env.agent_pos
    rect = plt.Rectangle((agent_x-0.5, agent_y-0.5), 1, 1, facecolor='green', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(rect)

    # Plot goal position
    goal_x, goal_y = env.goal_pos
    rect = plt.Rectangle((goal_x-0.5, goal_y-0.5), 1, 1, facecolor='red', edgecolor='darkred', linewidth=2)
    ax.add_patch(rect)

    # Set axis properties
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')

    # Invert y-axis to match grid coordinates (0,0 at top-left)
    ax.invert_yaxis()

    # Add grid lines for clarity
    for i in range(grid_size + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
        ax.axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)

    # Customize the plot
    ax.set_title(f'Corridor Maze (11x11)\nDirect Path: {len(env.direct_path)} cells, Detour Path: {len(env.detour_path)} cells\nPath Difference: {env.path_difference}, Coins: {len(env.coin_positions)}',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')

    # Create custom legend with correct colors
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='gray', edgecolor='black', label='Wall'),
        plt.Rectangle((0,0),1,1, facecolor='white', edgecolor='black', label='Corridor'),
        plt.Circle((0,0), 0.1, facecolor='gold', edgecolor='orange', label='Coin'),
        plt.Rectangle((0,0),1,1, facecolor='green', edgecolor='darkgreen', label='Agent Start'),
        plt.Rectangle((0,0),1,1, facecolor='red', edgecolor='darkred', label='Goal'),
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()

    # Return maze information
    return {
        'direct_path_length': len(env.direct_path),
        'detour_path_length': len(env.detour_path),
        'path_difference': env.path_difference,
        'coin_count': len(env.coin_positions),
        'coin_positions': env.coin_positions,
        'agent_start': env.agent_pos,
        'goal_position': env.goal_pos,
        'environment': env
    }


if __name__ == "__main__":

 
    print("Generating Corridor Maze...")

    # Generate and plot the maze
    maze_info = generate_and_plot_corridor_maze(seed=42)

    print(f"\nMaze Information:")
    print(f"Direct path length: {maze_info['direct_path_length']} cells")
    print(f"Detour path length: {maze_info['detour_path_length']} cells")
    print(f"Path difference: {maze_info['path_difference']} cells")
    print(f"Number of coins: {maze_info['coin_count']}")
    print(f"Coin positions: {maze_info['coin_positions']}")
    print(f"Agent starts at: {maze_info['agent_start']}")
    print(f"Goal is at: {maze_info['goal_position']}")

    # Generate a few different mazes
    print("\n" + "="*50)
    print("Generating 3 different maze variants...")

    for i in range(3):
        print(f"\nMaze #{i+1}:")
        info = generate_and_plot_corridor_maze(seed=i*10)
        print(f"  Direct: {info['direct_path_length']}, Detour: {info['detour_path_length']}, Difference: {info['path_difference']}, Coins: {info['coin_count']}")


#%% 
env = CorridorMazeEnv()
env.reset()
# %%
env.agent_start_dir_user            