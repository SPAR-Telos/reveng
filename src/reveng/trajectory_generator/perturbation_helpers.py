"""
Helper functions for applying environment perturbations during trajectory generation.

These perturbations allow testing agent goal-directedness by modifying the environment
mid-trajectory and observing behavioral responses.
"""

import random
from typing import Callable, Optional, Tuple


class EnvironmentPerturbation:
    """Base class for environment perturbations."""

    def should_apply(self, step_count: int, env) -> bool:
        """Determine if perturbation should be applied at this step."""
        raise NotImplementedError

    def apply(self, env) -> dict:
        """Apply the perturbation and return metadata about what changed."""
        raise NotImplementedError


class GoalDisplacement(EnvironmentPerturbation):
    """
    Move the goal to a new location mid-trajectory.

    Tests whether agent updates its behavior toward the new goal or continues
    toward the original location (pattern-matching vs. genuine goal pursuit).
    """

    def __init__(
        self,
        trigger_step: int,
        new_goal_pos: Optional[Tuple[int, int]] = None,
        random_displacement: bool = True,
    ):
        """
        Args:
            trigger_step: Step number at which to move the goal
            new_goal_pos: Specific (x, y) position for new goal, or None for random
            random_displacement: If True and new_goal_pos is None, randomly place goal
        """
        self.trigger_step = trigger_step
        self.new_goal_pos = new_goal_pos
        self.random_displacement = random_displacement
        self.applied = False

    def should_apply(self, step_count: int, env) -> bool:
        return step_count == self.trigger_step and not self.applied

    def apply(self, env) -> dict:
        """Move the goal to a new position."""
        unwrapped_env = env.unwrapped
        old_goal_pos = (
            unwrapped_env.goal_pos.copy()
            if hasattr(unwrapped_env, "goal_pos")
            else None
        )

        if self.new_goal_pos:
            new_pos = self.new_goal_pos
        elif self.random_displacement:
            # Find a valid random position (not wall, not agent)
            valid_positions = []
            for i in range(unwrapped_env.width):
                for j in range(unwrapped_env.height):
                    cell = unwrapped_env.grid.get(i, j)
                    if cell is None or cell.type == "goal":
                        if (i, j) != tuple(unwrapped_env.agent_pos):
                            valid_positions.append((i, j))
            new_pos = random.choice(valid_positions)
        else:
            return {
                "perturbation": "goal_displacement",
                "applied": False,
                "reason": "no position specified",
            }

        # Remove old goal
        if old_goal_pos is not None:
            unwrapped_env.grid.set(old_goal_pos[0], old_goal_pos[1], None)

        # Place new goal
        unwrapped_env.put_obj(unwrapped_env.goal, new_pos[0], new_pos[1])
        unwrapped_env.goal_pos = new_pos

        self.applied = True
        return {
            "perturbation": "goal_displacement",
            "old_goal_pos": old_goal_pos,
            "new_goal_pos": new_pos,
            "step": self.trigger_step,
        }


class DynamicObstacle(EnvironmentPerturbation):
    """
    Add or remove walls/obstacles mid-trajectory.

    Tests agent's ability to replan when environment structure changes.
    """

    def __init__(
        self,
        trigger_step: int,
        action: str = "add",  # "add" or "remove"
        positions: Optional[list] = None,
        count: int = 1,
    ):
        """
        Args:
            trigger_step: Step at which to modify obstacles
            action: "add" to create walls, "remove" to delete walls
            positions: List of (x, y) positions, or None for random
            count: Number of walls to add/remove if positions not specified
        """
        self.trigger_step = trigger_step
        self.action = action
        self.positions = positions
        self.count = count
        self.applied = False

    def should_apply(self, step_count: int, env) -> bool:
        return step_count == self.trigger_step and not self.applied

    def apply(self, env) -> dict:
        """Add or remove obstacles."""
        unwrapped_env = env.unwrapped
        modified_positions = []

        if self.action == "add":
            positions_to_add = (
                self.positions
                if self.positions
                else self._find_valid_positions(unwrapped_env, self.count)
            )

            for pos in positions_to_add:
                from minigrid.core.world_object import Wall

                wall = Wall()
                unwrapped_env.grid.set(pos[0], pos[1], wall)
                modified_positions.append(pos)

        elif self.action == "remove":
            positions_to_remove = (
                self.positions
                if self.positions
                else self._find_walls(unwrapped_env, self.count)
            )

            for pos in positions_to_remove:
                cell = unwrapped_env.grid.get(pos[0], pos[1])
                if cell and cell.type == "wall":
                    unwrapped_env.grid.set(pos[0], pos[1], None)
                    modified_positions.append(pos)

        self.applied = True
        return {
            "perturbation": "dynamic_obstacle",
            "action": self.action,
            "positions": modified_positions,
            "step": self.trigger_step,
        }

    def _find_valid_positions(self, env, count):
        """Find empty positions suitable for placing walls."""
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


class RewardStructureChange(EnvironmentPerturbation):
    """
    Modify the reward function mid-trajectory.

    Tests whether agent adapts to new incentive structure.
    """

    def __init__(
        self, trigger_step: int, reward_modifier: Callable[[float, dict], float]
    ):
        """
        Args:
            trigger_step: Step at which to change reward structure
            reward_modifier: Function that takes (original_reward, step_info) -> modified_reward
        """
        self.trigger_step = trigger_step
        self.reward_modifier = reward_modifier
        self.applied = False

    def should_apply(self, step_count: int, env) -> bool:
        return step_count == self.trigger_step and not self.applied

    def apply(self, env) -> dict:
        """Wrap environment step function to modify rewards."""
        unwrapped_env = env.unwrapped
        original_step = unwrapped_env.step

        def modified_step(action):
            obs, reward, terminated, truncated, info = original_step(action)
            modified_reward = self.reward_modifier(reward, info)
            return obs, modified_reward, terminated, truncated, info

        unwrapped_env.step = modified_step
        self.applied = True

        return {
            "perturbation": "reward_structure_change",
            "step": self.trigger_step,
            "modifier": self.reward_modifier.__name__
            if hasattr(self.reward_modifier, "__name__")
            else "custom",
        }


def generate_trajectory_with_perturbations(
    env,
    observation,
    info,
    agent,
    perturbations: list,
    max_steps_per_trajectory: Optional[int] = None,
):
    """
    Generate a single trajectory while applying perturbations.

    This is a modified version of generate_one_trajectory that checks for and
    applies perturbations at each step.

    Args:
        env: Gymnasium environment
        observation: Initial observation
        info: Initial info dict
        agent: Agent instance
        perturbations: List of EnvironmentPerturbation objects
        max_steps_per_trajectory: Optional step limit

    Returns:
        Trajectory object with additional perturbation metadata
    """
    from reveng.datatypes import Step, Trajectory

    steps = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False
    perturbation_log = []

    while not (terminated or truncated):
        if (
            max_steps_per_trajectory is not None
            and step_count >= max_steps_per_trajectory
        ):
            break

        # Check and apply perturbations
        for perturbation in perturbations:
            if perturbation.should_apply(step_count, env):
                metadata = perturbation.apply(env)
                perturbation_log.append(metadata)

        # Normal trajectory generation
        action, metadata = agent.select_action(
            env=env, observation=observation, info=info
        )
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        total_reward += float(reward)

        # Add perturbation info to step metadata if any were applied this step
        if perturbation_log and perturbation_log[-1].get("step") == step_count:
            if metadata is None:
                metadata = {}
            metadata["perturbation"] = perturbation_log[-1]

        steps.append(
            Step(
                observation=str(observation),
                action=str(action),
                reward=float(reward),
                metadata=metadata,
            )
        )

        observation = next_obs
        info = next_info
        step_count += 1

    traj = Trajectory(steps=steps, action_space=[], final_reward=total_reward)
    # Store perturbation log in trajectory metadata
    if not hasattr(traj, "metadata"):
        traj.metadata = {}
    traj.metadata["perturbations"] = perturbation_log

    return traj
