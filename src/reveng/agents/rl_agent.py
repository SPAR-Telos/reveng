from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np
from gymnasium import Env
from gymnasium.core import ObservationWrapper
from minigrid.minigrid_env import MiniGridEnv
from sb3_contrib import RecurrentPPO

# Stable-Baselines3 imports
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage

from reveng.agents.agent_abc import Agent
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.environment_generator.wrappers.rgb_obs_wrappers import (
    OmnidirectionalFogOfWarRGBImgObsWrapper,
)


class ImageOnlyObsWrapper(ObservationWrapper):
    """
    Observation wrapper that extracts only the 'image' from a dict observation,
    turning a Dict space into a Box space compatible with SB3 CnnPolicy.
    If the observation is already a Box (image), it passes it through unchanged.
    """

    def __init__(self, env):
        super().__init__(env)
        try:
            # If the env observation space is a Dict, select the 'image' subspace
            self.observation_space = self.env.observation_space["image"]  # type: ignore[index]
        except Exception:
            # If not a Dict, keep the original observation space
            self.observation_space = self.env.observation_space

    def observation(self, observation):
        if isinstance(observation, dict) and "image" in observation:
            return observation["image"]
        return observation


# Minigrid wrappers to get image-only observations (no dict, no mission strings)
try:
    # minigrid >= 3
    from minigrid.wrappers import ImgObsWrapper
except Exception:
    # Fallback for older minigrid (not expected per project deps)
    ImgObsWrapper = None  # type: ignore


SupportedAlgo = Literal["ppo", "recurrent_ppo", "dqn", "a2c"]


@dataclass
class RLConfig:
    """
    Configuration for the RL agent.

    You can pass additional algorithm-specific kwargs via `algo_kwargs`.
    """

    algorithm: str = "ppo"  # "ppo" | "recurrent_ppo" | "dqn" | "a2c"
    policy: Optional[str] = None  # if None, auto-select based on obs type ("CnnPolicy")
    device: Union[str, int] = "auto"
    seed: Optional[int] = None
    # Observation processing
    use_img_obs: bool = True  # use ImgObsWrapper (image-only obs from MiniGrid)
    transpose_images: bool = True  # apply VecTransposeImage (HWC->CHW) for CNNs
    # Project wrapper selection: "rgb_fog" (project wrapper), "img" (Minigrid ImgObsWrapper), or "none"
    wrapper_mode: Literal["rgb_fog", "img", "none"] = "rgb_fog"
    # RGB Fog-of-War wrapper params
    fog_of_war_view_radius: Optional[int] = None
    fog_of_war_tile_size: int = 8
    fog_of_war_color: Union[int, Tuple[int, int, int]] = 100
    # Learning params
    total_timesteps: int = 100_000
    n_envs: int = 1
    tensorboard_log: Optional[str] = None
    # Extra SB3 algorithm kwargs
    algo_kwargs: Dict[str, Any] = field(default_factory=dict)


def _wrap_env_for_training(env: Env, cfg: RLConfig) -> Env:
    """
    Apply necessary wrappers to a single environment to make it compatible with SB3.
    Priority:
      - If wrapper_mode == "rgb_fog": use project OmnidirectionalFogOfWarRGBImgObsWrapper
      - Else if wrapper_mode == "img" or use_img_obs: use Minigrid ImgObsWrapper
      - Else: no additional observation wrapper
    """
    base_env = env
    if cfg.wrapper_mode == "rgb_fog":
        base_env = OmnidirectionalFogOfWarRGBImgObsWrapper(
            base_env,
            tile_size=cfg.fog_of_war_tile_size,
            fog_color=cfg.fog_of_war_color,
            view_radius=cfg.fog_of_war_view_radius,
        )
        # Convert Dict observation with {"image": ...} to a plain Box image
        base_env = ImageOnlyObsWrapper(base_env)
    elif cfg.wrapper_mode == "img" or cfg.use_img_obs:
        if ImgObsWrapper is None:
            raise ImportError(
                "minigrid.wrappers.ImgObsWrapper not available. "
                "Please ensure 'minigrid' is installed and compatible."
            )
        base_env = ImgObsWrapper(base_env)
    return base_env


def _make_vec_env_from_factory(env_fn: Callable[[], Env], cfg: RLConfig) -> DummyVecEnv:
    """
    Create a vectorized environment using the user-provided factory `env_fn`,
    applying the same wrappers used during training/inference.
    """

    def _wrapped_env_fn() -> Env:
        env = env_fn()
        env = _wrap_env_for_training(env, cfg)
        return env

    vec_env = make_vec_env(_wrapped_env_fn, n_envs=cfg.n_envs, seed=cfg.seed)
    # Record episode stats
    vec_env = VecMonitor(vec_env)

    # For CNN policies, SB3 expects CHW tensors; Minigrid returns HWC images.
    if (
        cfg.use_img_obs or cfg.wrapper_mode in ("rgb_fog", "img")
    ) and cfg.transpose_images:
        vec_env = VecTransposeImage(vec_env)  # HWC -> CHW
    return vec_env


def make_simple2d_env_fn(
    size: int = 11,
    complexity: float = 0.0,
    agent_start_dir: Optional[int] = None,
    agent_start_pos: Optional[Tuple[int, int]] = None,
    goal_pos: Optional[Tuple[int, int]] = None,
    max_steps: Optional[int] = None,
    allow_quit_action: bool = False,
    **kwargs: Any,
) -> Callable[[], Env]:
    """
    Convenience factory returning an env_fn that builds the project's
    Simple2DNavigationEnv with the provided parameters.
    """

    def _factory() -> Env:
        return Simple2DNavigationEnv(
            size=size,
            complexity=complexity,
            agent_start_dir=agent_start_dir,
            agent_start_pos=agent_start_pos,
            goal_pos=goal_pos,
            max_steps=max_steps,
            allow_quit_action=allow_quit_action,
            **kwargs,
        )

    return _factory


class RLAgent(Agent):
    """
    A Stable-Baselines3 based RL agent that supports PPO, RecurrentPPO, DQN, and A2C
    for Minigrid-like 2D grid environments.

    Notes:
    - This agent assumes 0/1 reward structure is handled by the environment itself.
    - It uses image-only observations (ImgObsWrapper) by default and CnnPolicy.
    - For RecurrentPPO, sb3-contrib must be installed.
    """

    def __init__(self, cfg: Optional[RLConfig] = None, name: Optional[str] = None):
        super().__init__(name)
        self.cfg = cfg or RLConfig()
        self.model: Optional[Any] = None  # SB3 model
        self._rnn_states: Any = None  # for RecurrentPPO, maintained during inference
        self._episode_start: bool = True  # for RecurrentPPO
        self._obs_is_image: bool = self.cfg.use_img_obs

        # Determine default policy if not provided
        if self.cfg.policy is None:
            # With image obs, default to CNN policies, otherwise MLP
            if self._obs_is_image:
                # All supported algos accept "CnnPolicy"
                self.cfg.policy = "CnnPolicy"
            else:
                self.cfg.policy = "MlpPolicy"

        # Algorithm verification
        algo = self.cfg.algorithm.lower()
        if algo not in ("ppo", "recurrent_ppo", "dqn", "a2c"):
            raise ValueError(
                f"Unsupported algorithm '{self.cfg.algorithm}'. "
                "Use one of: 'ppo', 'recurrent_ppo', 'dqn', 'a2c'."
            )
        if algo == "recurrent_ppo" and RecurrentPPO is None:
            raise ImportError(
                "Recurrent PPO requires 'sb3-contrib'. "
                "Install with: pip install sb3-contrib"
            )

    # --------------- Training -----------------

    def train(
        self,
        env_fn: Callable[[], Env],
        total_timesteps: Optional[int] = None,
        reset_model: bool = False,
        verbose: int = 1,
    ) -> None:
        """
        Train the agent on environments created by `env_fn`.

        Args:
            env_fn: A callable that returns a fresh environment instance.
                    Example: lambda: Simple2DNavigationEnv(size=11, complexity=0.3)
            total_timesteps: Number of training steps. Overrides cfg.total_timesteps if provided.
            reset_model: If True, discard any existing model and create a new one.
            verbose: SB3 verbosity level (0: silent, 1: info, 2: debug)
        """
        # Build vec env
        vec_env = _make_vec_env_from_factory(env_fn, self.cfg)

        # Create or reuse model
        if reset_model or self.model is None:
            self.model = self._create_model(vec_env=vec_env, verbose=verbose)
        else:
            # Ensure observation/action spaces match (simple check)
            if (
                vec_env.observation_space.shape != self.model.observation_space.shape  # type: ignore[attr-defined]
                or vec_env.action_space.n != self.model.action_space.n  # type: ignore[attr-defined]
            ):
                raise ValueError(
                    "Existing model spaces don't match the new training environment. "
                    "Set reset_model=True to recreate the model."
                )

        # Train
        total_steps = int(total_timesteps or self.cfg.total_timesteps)
        # Enable progress bar only if tqdm and rich are available
        _use_progress_bar = False
        if verbose > 0:
            try:
                import rich  # noqa: F401
                import tqdm  # noqa: F401

                _use_progress_bar = True
            except Exception:
                _use_progress_bar = False
        self.model.learn(
            total_timesteps=total_steps,
            progress_bar=_use_progress_bar,
            tb_log_name=f"{self.cfg.algorithm}",
        )

        # Close vec env explicitly
        vec_env.close()

    # --------------- Inference -----------------

    def _process_single_obs_for_inference(self, env: MiniGridEnv) -> np.ndarray:
        """
        Build an observation matching the preprocessing used for training.

        Preference mirrors RLConfig.wrapper_mode used during training:
        - "rgb_fog": use project RGB fog-of-war wrapper's observation(); fallback to get_frame()
        - "img": use dict['image'] from env.gen_obs(); fallback to get_frame()
        - "none": prefer get_frame(); fallback to dict['image'] or raw np array

        If transpose_images=True, converts HWC -> CHW to match CNN policy expectations.
        """
        img: Optional[np.ndarray] = None
        mode = getattr(self.cfg, "wrapper_mode", "rgb_fog")

        # Try to get a base observation dict from the env if available
        base_obs = None
        if hasattr(env, "gen_obs") and callable(getattr(env, "gen_obs")):
            try:
                base_obs = env.gen_obs()
            except Exception:
                base_obs = None

        if mode == "rgb_fog":
            # Prefer project RGB fog wrapper processed image
            try:
                from reveng.environment_generator.wrappers.rgb_obs_wrappers import (
                    OmnidirectionalFogOfWarRGBImgObsWrapper,
                )

                if isinstance(env, OmnidirectionalFogOfWarRGBImgObsWrapper):
                    processed = env.observation(
                        base_obs if isinstance(base_obs, dict) else {}
                    )
                    if isinstance(processed, dict) and "image" in processed:
                        img = processed["image"]
            except Exception:
                img = None
            # Fallback to a rendered frame
            if (
                img is None
                and hasattr(env, "get_frame")
                and callable(getattr(env, "get_frame"))
            ):
                try:
                    tile_size = getattr(self.cfg, "fog_of_war_tile_size", 8)
                    img = env.get_frame(highlight=False, tile_size=tile_size)
                except Exception:
                    img = None
            # Last resort: if base_obs had an image
            if img is None and isinstance(base_obs, dict) and "image" in base_obs:
                img = base_obs["image"]

        elif mode == "img":
            # Prefer dict['image'] from env.gen_obs()
            if isinstance(base_obs, dict) and "image" in base_obs:
                img = base_obs["image"]
            # Fallback to a rendered frame
            if (
                img is None
                and hasattr(env, "get_frame")
                and callable(getattr(env, "get_frame"))
            ):
                try:
                    tile_size = getattr(self.cfg, "fog_of_war_tile_size", 8)
                    img = env.get_frame(highlight=False, tile_size=tile_size)
                except Exception:
                    img = None

        else:  # "none"
            # Prefer a rendered frame from the env
            if hasattr(env, "get_frame") and callable(getattr(env, "get_frame")):
                try:
                    tile_size = getattr(self.cfg, "fog_of_war_tile_size", 8)
                    img = env.get_frame(highlight=False, tile_size=tile_size)
                except Exception:
                    img = None
            # Fallback to dict['image'] or raw numpy base_obs
            if img is None and isinstance(base_obs, dict) and "image" in base_obs:
                img = base_obs["image"]
            if img is None and isinstance(base_obs, np.ndarray):
                img = base_obs

        if img is None:
            raise RuntimeError(
                "Could not construct an inference observation. Expected a dict with 'image', "
                "a raw numpy image, or an env providing get_frame()."
            )

        if not isinstance(img, np.ndarray):
            img = np.array(img)
        obs = img.astype(np.float32)

        # If we trained with VecTransposeImage, apply the same transpose for inference
        if self.cfg.transpose_images and obs.ndim == 3:
            # HWC -> CHW
            obs = np.transpose(obs, (2, 0, 1))

        return obs

    def select_action(
        self, env: MiniGridEnv, deterministic: bool = True, **kwargs: Any
    ) -> Tuple[int, dict]:
        """
        Select an action from the current environment state using the trained model.

        Args:
            env: The environment to act in (must be synchronized with training obs processing).
            deterministic: Whether to use deterministic actions (greedy).
            **kwargs: Unused, for interface compatibility.

        Returns:
            A tuple (action, metadata)
        """
        if self.model is None:
            raise RuntimeError(
                "Model is not trained or loaded. Call `train` or `load` first."
            )

        # Prepare observation consistent with training-time preprocessing
        obs = self._process_single_obs_for_inference(env)

        algo = self.cfg.algorithm.lower()
        info: Dict[str, Any] = {"deterministic": deterministic, "algo": algo}

        # SB3 predict expects obs shape to match model's observation_space (single env)
        if algo == "recurrent_ppo":
            # Handle RNN states and episode starts
            # sb3-contrib RecurrentPPO expects episode_start and state
            if self._episode_start:
                # Reset hidden state at the beginning of an episode
                self._rnn_states = None

            # episode_start array of shape (n_envs,)
            episode_starts = np.array([self._episode_start], dtype=bool)
            action, self._rnn_states = self.model.predict(  # type: ignore[attr-defined]
                obs,
                state=self._rnn_states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
            self._episode_start = False
        else:
            action, _ = self.model.predict(obs, deterministic=deterministic)

        action_int = int(action)
        return action_int, info

    def reset(self) -> None:
        """
        Reset internal inference state (e.g., RNN hidden states) at episode start.
        """
        self._episode_start = True
        self._rnn_states = None

    def update(self, **kwargs: Any) -> None:
        """
        Not used for SB3 agents (learning happens in `train`).
        Provided to satisfy the Agent interface.
        """
        # No online update step here; SB3 learns during `learn()`.
        return

    # --------------- Persistence -----------------

    def save(self, path: str) -> None:
        """
        Save the underlying SB3 model to disk.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train or load a model first.")
        self.model.save(path)

    @classmethod
    def load(
        cls, path: str, cfg: Optional[RLConfig] = None, name: Optional[str] = None
    ) -> "RLAgent":
        """
        Load an SB3 model from disk and wrap it in an RLAgent.

        Args:
            path: Path to the model zip saved by SB3.
            cfg: RLConfig that describes how observations/actions are processed. If None,
                 a default RLConfig is used. Make sure it matches the preprocessing used during training!
            name: Optional agent name.

        Returns:
            RLAgent with loaded model.
        """
        cfg = cfg or RLConfig()
        agent = cls(cfg=cfg, name=name)

        algo = cfg.algorithm.lower()
        if algo == "ppo":
            agent.model = PPO.load(path, device=cfg.device)
        elif algo == "a2c":
            agent.model = A2C.load(path, device=cfg.device)
        elif algo == "dqn":
            agent.model = DQN.load(path, device=cfg.device)
        elif algo == "recurrent_ppo":
            if RecurrentPPO is None:
                raise ImportError(
                    "Recurrent PPO requires 'sb3-contrib'. "
                    "Install with: pip install sb3-contrib"
                )
            agent.model = RecurrentPPO.load(path, device=cfg.device)
        else:
            raise ValueError(f"Unsupported algorithm '{cfg.algorithm}'.")

        return agent

    # --------------- Internal helpers -----------------

    def _create_model(self, vec_env: DummyVecEnv, verbose: int = 1) -> Any:
        """
        Instantiate the appropriate SB3 model based on config.
        """
        algo = self.cfg.algorithm.lower()
        policy = str(self.cfg.policy)
        common_kwargs = dict(
            policy=policy,
            env=vec_env,
            device=self.cfg.device,
            verbose=verbose,
            tensorboard_log=self.cfg.tensorboard_log,
            **self.cfg.algo_kwargs,
        )

        if algo == "ppo":
            return PPO(**common_kwargs)
        elif algo == "a2c":
            return A2C(**common_kwargs)
        elif algo == "dqn":
            # DQN does not accept 'tensorboard_log' before v2.3? In v2.7, it does.
            return DQN(**common_kwargs)
        elif algo == "recurrent_ppo":
            if RecurrentPPO is None:
                raise ImportError(
                    "Recurrent PPO requires 'sb3-contrib'. "
                    "Install with: pip install sb3-contrib"
                )
            return RecurrentPPO(**common_kwargs)
        else:
            raise ValueError(f"Unsupported algorithm '{self.cfg.algorithm}'.")


__all__ = ["RLConfig", "RLAgent", "make_simple2d_env_fn"]
