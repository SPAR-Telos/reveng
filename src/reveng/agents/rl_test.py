#!/usr/bin/env python3
"""
Simple smoke test for RLAgent training and inference on the project's custom MiniGrid
environment using the project's RGB fog-of-war wrapper.

This script:
- Trains a small SB3 model (PPO by default) for a few steps.
- Saves and reloads the model.
- Runs a brief inference rollout to ensure the end-to-end pipeline works.

Usage:
  uv run python -m reveng.agents.rl_test --algorithm ppo --timesteps 2000
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from typing import Optional, Tuple

from reveng.agents.rl_agent import RLAgent, RLConfig, make_simple2d_env_fn
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.environment_generator.wrappers.rgb_obs_wrappers import (
    OmnidirectionalFogOfWarRGBImgObsWrapper,
)


def train_agent(
    algorithm: str = "ppo",
    total_timesteps: int = 2_000,
    n_envs: int = 2,
    size: int = 11,
    complexity: float = 0.3,
    wrapper_mode: str = "rgb_fog",
    seed: Optional[int] = 42,
) -> RLAgent:
    """
    Train an RLAgent briefly to verify the training pipeline.
    """
    cfg = RLConfig(
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        tensorboard_log=None,
        seed=seed,
        wrapper_mode=wrapper_mode,  # use project wrapper by default
        use_img_obs=True,
        transpose_images=True,
    )
    agent = RLAgent(cfg)

    env_fn = make_simple2d_env_fn(size=size, complexity=complexity)
    agent.train(env_fn=env_fn, verbose=1)

    return agent


def run_inference_rollout(
    agent: RLAgent,
    steps: int = 50,
    size: int = 11,
    complexity: float = 0.3,
    use_wrapper: bool = True,
) -> Tuple[float, int]:
    """
    Run a short inference rollout to verify select_action + env.step works.

    Returns:
        total_reward, num_steps_executed
    """
    # Create a single env for inference
    env = Simple2DNavigationEnv(size=size, complexity=complexity)

    # For consistency with training, wrap the env with the project RGB fog-of-war wrapper
    if use_wrapper:
        env = OmnidirectionalFogOfWarRGBImgObsWrapper(env)

    # Reset both agent and env
    agent.reset()
    obs, info = env.reset()
    del obs, info  # agent will generate its own observation internally

    total_reward = 0.0
    num_steps = 0
    terminated, truncated = False, False

    while not (terminated or truncated) and num_steps < steps:
        action, meta = agent.select_action(env, deterministic=True)
        assert 0 <= action < env.action_space.n, (
            f"Selected action {action} not in action space!"
        )

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        num_steps += 1

    # Close env if it supports it
    try:
        env.close()
    except Exception:
        pass

    return total_reward, num_steps


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Smoke test for RLAgent.")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "a2c", "dqn", "recurrent_ppo"],
        help="Algorithm to train",
    )
    parser.add_argument(
        "--timesteps", type=int, default=2000, help="Training timesteps for smoke test"
    )
    parser.add_argument(
        "--n-envs", type=int, default=2, help="Number of parallel envs during training"
    )
    parser.add_argument(
        "--size", type=int, default=11, help="Grid size for the custom minigrid env"
    )
    parser.add_argument(
        "--complexity",
        type=float,
        default=0.3,
        help="Maze complexity (0.0 empty room, 1.0 perfect maze)",
    )
    parser.add_argument(
        "--wrapper-mode",
        type=str,
        default="rgb_fog",
        choices=["rgb_fog", "img", "none"],
        help="Observation wrapper used during training",
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=50,
        help="Number of steps for the inference rollout",
    )
    args = parser.parse_args(argv)

    # Train briefly
    print(
        f"[SMOKE TEST] Training RLAgent with {args.algorithm} for {args.timesteps} steps..."
    )
    try:
        agent = train_agent(
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            size=args.size,
            complexity=args.complexity,
            wrapper_mode=args.wrapper_mode,
        )
    except ImportError as e:
        if args.algorithm == "recurrent_ppo":
            print(
                "Recurrent PPO requires sb3-contrib. Install with: pip install sb3-contrib"
            )
        raise e

    # Save and reload to verify persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = f"{tmpdir}/{args.algorithm}_smoke_model.zip"
        print(f"[SMOKE TEST] Saving model to {model_path}")
        agent.save(model_path)

        print(f"[SMOKE TEST] Loading model from {model_path}")
        agent = RLAgent.load(model_path, cfg=agent.cfg)

    # Run a short inference rollout
    print("[SMOKE TEST] Running inference rollout...")
    total_reward, num_steps = run_inference_rollout(
        agent=agent,
        steps=args.inference_steps,
        size=args.size,
        complexity=args.complexity,
        use_wrapper=(args.wrapper_mode == "rgb_fog"),
    )

    print(
        f"[SMOKE TEST] Inference finished: steps={num_steps}, total_reward={total_reward:.2f}"
    )
    print("[SMOKE TEST] SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
