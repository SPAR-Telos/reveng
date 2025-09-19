import random
import time
from enum import IntEnum

import pygame
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv


class Simple2DNavigationEnv(MiniGridEnv):
    # https://minigrid.farama.org/content/create_env_tutorial/
    class Actions(IntEnum):
        LEFT = 0
        RIGHT = 1
        UP = 2
        DOWN = 3

    def __init__(
        self,
        size=10,
        agent_start_dir: int | None = None,
        agent_start_pos: tuple[int, int] | None = None,
        goal_pos: tuple[int, int] | None = None,
        max_steps: int | None = None,
        **kwargs,
    ):
        if agent_start_pos is None:
            self.agent_start_pos = (
                random.randint(1, size - 2),
                random.randint(1, size - 2),
            )
        else:
            self.agent_start_pos = agent_start_pos

        if agent_start_dir is None:
            self.agent_start_dir = random.randint(0, 3)
        else:
            self.agent_start_dir = agent_start_dir

        if goal_pos is None:
            self.goal_pos = (random.randint(1, size - 2), random.randint(1, size - 2))
            while self.goal_pos == self.agent_start_pos:
                self.goal_pos = (
                    random.randint(1, size - 2),
                    random.randint(1, size - 2),
                )
        else:
            self.goal_pos = goal_pos

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

        self.actions = Simple2DNavigationEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))

        # Map actions to agent direction
        self._action_to_direction = {
            self.Actions.LEFT: 2,
            self.Actions.RIGHT: 0,
            self.Actions.UP: 3,
            self.Actions.DOWN: 1,
        }

    @staticmethod
    def _gen_mission():
        return "Reach the Goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(
            0, 0, width, height
        )  # To avoid terminating the game when stepping outside the grid
        self.put_obj(Goal(), *self.goal_pos)
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        self.mission = "Reach the Goal"

    def step(self, action):
        self.step_count += 1
        reward = 0
        terminated = False
        truncated = False

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

        # Check if the agent reached the goal
        if fwd_cell is not None and fwd_cell.type == "goal":
            terminated = True
            reward = self._reward()

        # Check if the episode is truncated
        if self.step_count >= self.max_steps:
            truncated = True

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}


def manual_control(size):
    env = Simple2DNavigationEnv(render_mode="human", size=size)
    env.reset()

    # Map pygame keys to environment actions for cleaner handling
    key_to_action = {
        pygame.K_LEFT: env.actions.LEFT,
        pygame.K_RIGHT: env.actions.RIGHT,
        pygame.K_UP: env.actions.UP,
        pygame.K_DOWN: env.actions.DOWN,
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
                    print(
                        f"Step: {env.step_count}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}"
                    )

                    if terminated or truncated:
                        print("Episode finished. Resetting.")
                        env.reset()

    env.close()


def run_random_episodes(episodes=5, size=10):
    """
    Runs episodes with a random agent
    """
    env = Simple2DNavigationEnv(render_mode="human", size=size)

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
            print(f"Action sampled: {action} ({env.actions(action).name})")

            # Take the action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

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
