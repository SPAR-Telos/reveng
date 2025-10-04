"""
Trajectory generation utilities for collecting agent behavior data.

This module defines the contract for generating trajectories by running agent
policies in environments and storing the results in standardized Trajectory
format.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from reveng.datatypes import Step, Trajectory
from reveng.agents.agent_abc import Agent


def _save_trajectory_to_json(
    trajectory: Trajectory,
    save_dir: Path,
    file_name: str,
    indent: int,
):
    """Saves a single trajectory object to a JSON file."""
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / file_name
    file_path.write_text(json.dumps(asdict(trajectory), indent=indent))


def generate_trajectories(
    env,
    agent: Agent,
    num_trajectories: int,
    max_steps_per_trajectory: Optional[int] = None,
    reset_between_trajectories: bool = True,
    save_dir: Optional[str] = None,
    save_prefix: str = "trajectory",
    save_indent: int = 2,
) -> List[Trajectory]:
    """
    Define the interface to generate Trajectory objects by running an agent in an environment.

    This function declares the contract for collecting trajectories using an
    agent callable that selects actions given the current environment context
    and observation. Implementations should follow the project's standardized
    Trajectory and Step data models as defined in `reveng.datatypes`.

    Contract
    - Inputs:
      - env: gymnasium-compatible environment (e.g., Simple2DNavigationEnv wrapped
        with an observation wrapper from environment_generator). Must support
        reset() and step(action). The environment typically also exposes an
        action_space for sampling.
      - agent: Callable that maps (observation, info, env) -> action, or when
        include_thoughts=True, returns (action, thought). The agent may ignore
        any of the inputs it doesn't need.
      - num_trajectories: Positive integer number of trajectories to generate.
      - max_steps_per_trajectory: Optional positive integer; if provided, cap the
        number of steps per trajectory at this limit.
      - include_thoughts: When True, expect agent to return (action, thought).
      - reset_between_trajectories: When True, call env.reset() before each new
        trajectory; when False, continue from current env state across trajectories.
        - Outputs:
      - List[Trajectory]: Each Trajectory contains:
        - steps: List[Step] where each Step has:
          - observation: str (string form of env observation; wrappers should provide
            meaningful str() representations for text/image modalities)
          - action: str (string form of the action taken)
          - reward: float (reward from env.step)
          - thought: Optional[str] (present only if include_thoughts=True)
        - action_space: List[str] (optional names of actions; may be left empty if
          action semantics are captured in Step.action)
        - final_reward: float (sum of rewards over the trajectory)
        - Error modes:
      - ValueError for invalid num_trajectories or max_steps_per_trajectory values
      - TypeError if the agent return type does not match include_thoughts setting
      - AttributeError if env does not conform to the expected gym API

    Notes
    - Observation wrappers from environment_generator (e.g., text or image wrappers)
      are expected to provide useful string representations when cast via str().
    - For MiniGrid-based environments, actions are typically integers; storing
      str(action) in Step.action maintains compatibility with scoring utilities.
    - This is a definition-only stub; an implementation should mirror the control
      flow of a typical gym loop: reset, loop over steps, call the agent to get an
      action, step the environment, append Step entries, and stop on terminated or
      truncated or max_steps_per_trajectory.

        Optional JSON saving
        - When save_dir is provided, each generated trajectory will be written as a
            JSON file compliant with `reveng.datatypes.load_trajectory_from_file` schema:
                {
                    "steps": [
                        {"observation": str, "action": str, "reward": float | null, "thought": str | null},
                        ...
                    ],
                    "action_space": [str, ...],
                    "final_reward": float | null
                }
            Files are named as f"{save_prefix}_{i}.json" in save_dir, where i starts at 1.

        Example
        from reveng.trajectory_generator.policies import random_policy

        # Agent signature: agent(observation, info, env) -> action
        agent = random_policy

        # trajectories = generate_trajectories(
        #     env=env,
        #     agent=agent,
        #     num_trajectories=5,
        #     max_steps_per_trajectory=50,
        #     include_thoughts=False,
        # )

    """
    # Validate inputs
    if num_trajectories <= 0:
        raise ValueError(f"num_trajectories must be positive, got {num_trajectories}")

    if max_steps_per_trajectory is not None and max_steps_per_trajectory <= 0:
        raise ValueError(
            f"max_steps_per_trajectory must be positive, got {max_steps_per_trajectory}"
        )

    trajectories: List[Trajectory] = []
    observation, info = env.reset()

    for traj_idx in range(num_trajectories):
        # Reset environment if requested
        if reset_between_trajectories:
            observation, info = env.reset()

        traj_obj = generate_one_trajectory(
            env=env,
            observation=observation,
            info=info,
            agent=agent,
            max_steps_per_trajectory=max_steps_per_trajectory,
        )

        trajectories.append(traj_obj)

        # If a save directory is provided, create it and save the trajectory data.
        if save_dir:
            file_name = f"{save_prefix}_{traj_idx + 1}.json"
            _save_trajectory_to_json(traj_obj, Path(save_dir), file_name, save_indent)

    return trajectories


def generate_one_trajectory(
    env,
    observation,
    info,
    agent: Agent,
    max_steps_per_trajectory: Optional[int] = None,
):
    steps: List[Step] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    # Roll out one trajectory
    while not (terminated or truncated):
        if (
            max_steps_per_trajectory is not None
            and step_count >= max_steps_per_trajectory
        ):
            break

        action, metadata = agent.select_action(
            env=env, observation=observation, info=info
        )

        # Step the environment
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        total_reward += float(reward)

        # Record step (use string forms to align with scoring pipeline expectations)
        steps.append(
            Step(
                observation=str(observation),
                action=str(action),
                reward=float(reward),
                metadata=metadata,
            )
        )

        # Prepare next iteration
        observation = next_obs
        info = next_info
        step_count += 1

    traj_obj = Trajectory(steps=steps, action_space=[], final_reward=total_reward)
    return traj_obj


def visualize_trajectory(
    trajectory: Trajectory,
    env,
    sleep: float = 0.1,
    save_gif_path: Optional[str] = None,
    fps: int = 10,
) -> None:
    """
    Replay a trajectory on the provided environment and render each step.

    This utility resets the environment, then sequentially applies the actions
    stored in the given Trajectory. After each step it calls env.render() and
    optionally sleeps for a short duration to make the visualization watchable.

    Important:
    - This does not reproduce the exact original states unless the environment
      and RNG are seeded identically and the environment dynamics are deterministic.
    - Actions in Trajectory.Step are stored as strings; this function attempts
      to convert them back to integers via int(step.action). If parsing fails,
      that step will be skipped.
    - For GIF saving, the environment must be created with a render_mode that
      returns RGB frames (e.g., "rgb_array" or "rgb_array_list"). When using
      render_mode="human", env.render() typically won't return image frames,
      so GIF capture will be skipped unless a frame can be obtained from the
      environment via other means.

    Args:
      trajectory: The Trajectory to replay.
      env: A gymnasium-compatible environment configured with the same wrappers
         used during generation (e.g., image/text, full/partial observability).
      sleep: Seconds to sleep between rendered steps (default: 0.1s).
      save_gif_path: Optional file path to save a GIF of the replay. Requires
         an env render mode that yields RGB frames and either imageio or Pillow.
      fps: Frames per second for GIF saving (default: 10).
    """
    frames = [] if save_gif_path else None

    def _maybe_capture_frame():
        if frames is None:
            return
        frame = None
        try:
            frame = env.render()
        except Exception:
            frame = None
        if frame is None:
            # Try unwrapped render (some envs expose frame via base env)
            try:
                frame = env.unwrapped.render()
            except Exception:
                frame = None
        if frame is not None:
            try:
                import numpy as _np  # local import only when needed

                arr = _np.asarray(frame)
                if arr.ndim >= 2:
                    # Ensure uint8 3-channel if possible
                    if arr.dtype != _np.uint8:
                        arr = _np.clip(arr, 0, 255).astype(_np.uint8)
                    if arr.ndim == 2:  # grayscale -> RGB
                        arr = _np.stack([arr] * 3, axis=-1)
                    frames.append(arr)
            except Exception:
                # If numpy is unavailable or conversion fails, skip capture
                pass

    obs, info = env.reset()
    _maybe_capture_frame()
    terminated = False
    truncated = False
    for step in trajectory.steps:
        if terminated or truncated:
            break
        try:
            action = int(step.action)
        except (TypeError, ValueError):
            # Skip steps with non-integer-parsable actions
            continue
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        _maybe_capture_frame()
        if sleep and sleep > 0:
            time.sleep(sleep)

    # Save GIF if requested and frames were captured
    if frames:
        try:
            try:
                import imageio.v2 as _iio  # type: ignore

                _iio.mimsave(save_gif_path, frames, duration=1 / max(fps, 1))
            except Exception:
                # Fallback to Pillow if available
                from PIL import Image as _Image  # type: ignore

                images = [_Image.fromarray(f) for f in frames]
                images[0].save(
                    save_gif_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=int(1000 / max(fps, 1)),
                    loop=0,
                )
        except Exception as e:  # If saving fails, emit a brief notice
            print(
                f"[visualize_trajectory] Failed to save GIF to '{save_gif_path}': {e}. "
                "Ensure env returns RGB frames (render_mode='rgb_array') and install imageio or pillow."
            )
