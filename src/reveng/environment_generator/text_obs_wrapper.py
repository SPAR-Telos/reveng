import time

import gymnasium
from custom_minigrid import Simple2DNavigationEnv
from gymnasium.spaces import Text


class Simple2DNavigationEnvTextWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Define the new observation space.
        self.observation_space = Text(max_length=4096, charset="utf-8")

        # Map agent's direction to a character
        self.dir_map = {0: ">", 1: "v", 2: "<", 3: "^"}

    def observation(self, obs):
        """
        Generates the text observation from the environment's grid state.
        """
        env = self.unwrapped

        # 1. Start with the mission description
        mission = f"Mission: {env.mission}\n"

        # 2. Create a character grid
        grid_repr = []
        for j in range(env.height):
            row_str = ""
            for i in range(env.width):
                # Check for agent position first
                if i == env.agent_pos[0] and j == env.agent_pos[1]:
                    row_str += self.dir_map[env.agent_dir]
                    continue

                cell = env.grid.get(i, j)
                if cell is None:
                    row_str += " "  # Empty space
                elif cell.type == "wall":
                    row_str += "#"
                elif cell.type == "goal":
                    row_str += "G"
                else:
                    row_str += "?"
            grid_repr.append(row_str)

        grid_str = "\n".join(grid_repr)

        # 3. Add a simplified legend
        legend = (
            "\n--- Legend ---\n> v < ^ : Agent\n# : Wall\nG : Goal\n---------------\n"
        )

        # Combine the strings to create the full observation
        full_observation = mission + grid_str + legend
        return full_observation


def run_random_episodes_with_text_observations(episodes=5, size=10):
    """
    Runs episodes with a random agent
    """
    base_env = Simple2DNavigationEnv(render_mode="human", size=size)
    env = Simple2DNavigationEnvTextWrapper(base_env)

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


if __name__ == "__main__":
    run_random_episodes_with_text_observations()
