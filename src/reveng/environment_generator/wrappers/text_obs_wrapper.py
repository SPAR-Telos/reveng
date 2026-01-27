import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Text
from minigrid.minigrid_env import MiniGridEnv

from reveng.environment_generator.wrappers.fog_of_war import FogOfWarWrapper

# Default configuration for cells. This can be overridden at initialization.
DEFAULT_CELLS_CONFIG = {
    "agent": {"symbol": "A", "description": "Current agent position"},
    "wall": {"symbol": "#", "description": "Wall"},
    "goal": {"symbol": "G", "description": "Goal"},
    "empty": {"symbol": "_", "description": "Open space (can be visited)"},
    "door": {"symbol": "D", "description": "Door (locked or unlocked)"},
    "key": {"symbol": "K", "description": "Key (can unlock doors)"},
    "ball": {"symbol": "C", "description": "Coin (collectible item)"},
    "unknown_obj": {"symbol": "?", "description": "Unknown object"},
    "fog": {
        "symbol": "*",
        "description": "Hidden Space (reveals a Wall, Goal or Open Space when observed)",
    },
}


class TextWrapper(gym.ObservationWrapper):
    """
    A mixin that provides methods to configure and render a MiniGrid environment
    as a text observation that aligns with the template formats:
      - Full observability: grid with coordinates and space-separated symbols
      - Partial observability: same format, but hidden cells rendered as '*'
    """

    def _setup_from_config(
        self,
        config_path: str | os.PathLike | None = None,
    ):
        """
        Initializes symbols from a configuration dictionary or JSON file.
        """
        # Use default config if none is provided
        final_config = DEFAULT_CELLS_CONFIG.copy()

        if config_path:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path, "r") as f:
                file_config = json.load(f)
                final_config.update(file_config)
        self.grid_cells = final_config

    @property
    def legend(self) -> str:
        """Dynamically creates the legend string from the current grid_cells."""
        legend_items = [
            f"{cell['symbol']} : {cell['description']}"
            for cell in self.grid_cells.values()
        ]
        return "\n".join(legend_items)

    def _get_cell_type_at_position(
        self,
        env: MiniGridEnv,
        i: int,
        j: int,
        seen_mask: np.ndarray | None = None,
    ) -> str:
        # If a mask exists and the cell is not visible, apply fog
        if seen_mask is not None and not seen_mask[j, i]:
            return "fog"

        # Check for agent position first
        if i == env.agent_pos[0] and j == env.agent_pos[1]:
            return "agent"

        # Apply fog if not visible
        if seen_mask is not None and not seen_mask[j, i]:
            return "fog"

        # Render underlying cell
        cell = env.grid.get(i, j)
        if cell is None:
            return "empty"
        return cell.type

    def _get_cell_at_position(
        self,
        env: MiniGridEnv,
        i: int,
        j: int,
        seen_mask: np.ndarray | None = None,
    ) -> str:
        cell_type = self._get_cell_type_at_position(env, i, j, seen_mask)
        return (
            self.grid_cells[cell_type]["symbol"]
            if cell_type in self.grid_cells
            else "?"
        )

    def _get_grid(
        self,
        seen_mask: np.ndarray | None = None,
        add_indices: bool = True,
    ) -> str:
        env: MiniGridEnv = self.unwrapped  # type: ignore
        if add_indices:
            tot_height = env.height + 1
            tot_width = env.width + 1
        else:
            tot_height = env.height
            tot_width = env.width
        grid = np.full((tot_height, tot_width), "unknown_obj", dtype=object)

        # Pad appropriately for column indices
        padding = len(str(env.width)) + 1

        for j in range(tot_height):
            for i in range(tot_width):
                if add_indices:
                    if j == 0:
                        if i == 0:
                            grid[j, i] = " " * padding  # Top-left corner
                        else:
                            grid[j, i] = f"{i - 1:<{padding}}"  # Column indices
                    elif i == 0:
                        grid[j, i] = f"{j - 1:<{padding}}"  # Row indices
                    else:
                        if i == tot_width - 1:
                            grid[j, i] = (
                                self._get_cell_at_position(env, i - 1, j - 1, seen_mask)
                                + " "
                            )
                        else:
                            grid[j, i] = (
                                f"{self._get_cell_at_position(env, i - 1, j - 1, seen_mask):<{padding}}"
                            )
                else:
                    grid[j, i] = self._get_cell_at_position(env, i, j, seen_mask)
        return "\n".join("".join(row) for row in grid)

    def _render(
        self,
        seen_mask: np.ndarray | None = None,
        add_indices: bool = True,
    ) -> str:
        """Renders the grid as a text observation string."""
        return self._get_grid(seen_mask=seen_mask, add_indices=add_indices)


class FullObservabilityTextWrapper(TextWrapper):
    """
    Observation wrapper that returns a fully observable grid formatted to align
    with grid_full_observability.j2:
      - Includes a header row with column indices
      - Each row begins with the row index
      - Cells are space-separated symbols
      - No mission or legend text is included
    """

    def __init__(self, env, config_path=None):
        super().__init__(env)
        self.observation_space = Text(max_length=4096, charset="utf-8")
        self._setup_from_config(config_path)

    def observation(self, observation):
        """Generates a fully observable text grid by calling the mixin method."""
        return self._render()


class FogOfWarTextWrapper(FogOfWarWrapper, TextWrapper):
    """
    Observation wrapper that returns a partially observable grid formatted to align
    with grid_partial_observability.j2:
      - Same coordinate and spacing format as the full observability case
      - Hidden/unseen cells are rendered as '*' (fog)
      - No mission or legend text is included
    """

    def __init__(self, env, view_radius=None, config_path=None):
        super().__init__(env, view_radius=view_radius)
        self.observation_space = Text(max_length=4096, charset="utf-8")
        self._setup_from_config(config_path)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self.observation(obs), info

    def observation(self, observation):
        """Updates the mask and then renders the text grid with fog."""
        super().observation(observation)  # Updates self.seen_mask
        return self._render(seen_mask=self.seen_mask)


class LoggingFogOfWarTextWrapper(FogOfWarTextWrapper):
    """
    Extends FogOfWarTextWrapper by logging both partial and full observability
    grid strings as well as the underlying cell types at each step.
    """

    def __init__(self, env, view_radius=None, config_path=None):
        super().__init__(env, view_radius=view_radius, config_path=config_path)
        self.partially_observable_observation_log = []
        self.fully_observable_observation_log = []
        self.partially_observable_cell_type_log = []
        self.fully_observable_cell_type_log = []

    def save_cell_types(self):
        env: MiniGridEnv = self.unwrapped  # type: ignore
        po_cell_types = []  # list of tuples (x, y, cell_type)
        fo_cell_types = []  # list of tuples (x, y, cell_type)
        height = env.height
        width = env.width
        for j in range(height):
            for i in range(width):
                po_cell_type = self._get_cell_type_at_position(
                    env, i, j, self.seen_mask
                )
                po_cell_types.append((i, j, po_cell_type))
                fo_cell_type = self._get_cell_type_at_position(env, i, j, None)
                fo_cell_types.append((i, j, fo_cell_type))
        self.partially_observable_cell_type_log.append(repr(po_cell_types))
        self.fully_observable_cell_type_log.append(repr(fo_cell_types))

    def observation(self, observation):
        obs = super().observation(observation)
        env: MiniGridEnv = self.unwrapped  # type: ignore
        self.partially_observable_observation_log.append(obs)
        fully_observable_mask = np.ones((env.height, env.width), dtype=bool)
        fo_obs = self._render(seen_mask=fully_observable_mask)
        self.fully_observable_observation_log.append(fo_obs)

        self.save_cell_types()
        return obs
