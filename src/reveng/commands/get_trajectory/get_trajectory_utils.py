"""Utility functions for trajectory generation, token processing, and LLM completion handling."""

import json
import logging
import math
import os
import re
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, login
from litellm import completion, completion_cost
from tenacity import retry, stop_after_attempt, wait_random_exponential

from reveng.agents.alpha_start_agent import AlphaStarAgent
from reveng.agents.llm_agent import LLMAgent
from reveng.datatypes import Step, Trajectory
from reveng.environment_generator.env_transformations import (
    EnvTransformation,
    IsoDifficultyTransformationFactory,
)
from reveng.environment_generator.utils import remove_door
from reveng.environment_generator.wrappers.text_obs_wrapper import (
    FullObservabilityTextWrapper,
)
from reveng.trajectory_generator.trajectory_generator import generate_one_trajectory

logger = logging.getLogger(__file__)

# Default transform names (matches IsoDifficultyTransformationFactory)
DEFAULT_TRANSFORM_NAMES = ["RotateEnv", "ReflectEnv", "TransposeEnv", "StartGoalSwap"]


def get_iso_difficulty_transforms() -> list[tuple[str, EnvTransformation]]:
    """Get the list of iso-difficulty transformations with their names.

    Returns:
        list[tuple[str, EnvTransformation]]: List of (name, transform) tuples
            containing RotateEnv, ReflectEnv, TransposeEnv, and StartGoalSwap.
    """
    factory = IsoDifficultyTransformationFactory()
    transforms = factory.get_transformations()
    return [(t.__name__, t) for t in transforms]


def get_transformed_environments(
    base_env: FullObservabilityTextWrapper,
    include_base: bool = True,
    transform_names: list[str] | None = None,
) -> list[tuple[str, FullObservabilityTextWrapper]]:
    """Create transformed versions of an environment.

    Takes a base environment wrapped in FullObservabilityTextWrapper and returns
    a list of (transform_name, wrapped_env) tuples for each requested transform.

    Args:
        base_env: The base environment wrapped in FullObservabilityTextWrapper.
        include_base: If True, includes ("base", base_env) as the first item.
        transform_names: List of transform names to apply. If None, uses all
            available iso-difficulty transforms (RotateEnv, ReflectEnv,
            TransposeEnv, StartGoalSwap).

    Returns:
        list[tuple[str, FullObservabilityTextWrapper]]: List of (name, env) tuples
            where name is either "base" or the transform class name.
    """
    result = []

    if include_base:
        result.append(("base", base_env))

    # Get available transforms
    available_transforms = {name: t for name, t in get_iso_difficulty_transforms()}

    # Use all transforms if none specified
    if transform_names is None:
        transform_names = list(available_transforms.keys())

    # Apply each requested transform
    for name in transform_names:
        if name not in available_transforms:
            logger.warning(
                f"Unknown transform '{name}', skipping. "
                f"Available: {list(available_transforms.keys())}"
            )
            continue

        transform = available_transforms[name]
        # Apply transform to the unwrapped env, then wrap the result
        transformed_unwrapped = transform.apply(base_env.unwrapped)
        transformed_wrapped = FullObservabilityTextWrapper(transformed_unwrapped)
        result.append((name, transformed_wrapped))

    return result


@retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, min=5, max=120),
    reraise=True,
)
def completion_with_retry(**generation_kwargs) -> tuple[Any, float]:
    """Call LLM completion API with automatic retry on failure.

    Wraps the litellm completion API with exponential backoff retry logic
    to handle transient failures. Also calculates the cost of the completion.

    Args:
        **generation_kwargs: Keyword arguments passed to litellm.completion().

    Returns:
        tuple[Any, float]: A tuple of (response object, completion cost in USD).

    Raises:
        Exception: Re-raises the last exception after 5 retry attempts.
    """
    try:
        response = completion(**generation_kwargs)
    except Exception as e:
        logger.error(f"Model request failed: {e}\n{traceback.format_exc()}")
        raise
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        logger.error(f"Failed to calculate cost: {e}")
        cost = 0.0
    return response, cost


def parse_action(model_output: str) -> tuple[int, str]:
    """Parse the model's text output to extract the action.

    Expects the model output to be a JSON string containing an "action" key
    with one of the directional values: LEFT, RIGHT, UP, DOWN.

    Args:
        model_output: The raw text output from the model.

    Returns:
        tuple[int, str]: A tuple of (action_id, action_name) where:
            - action_id: Integer action code (0=LEFT, 1=RIGHT, 2=UP, 3=DOWN, -1=invalid)
            - action_name: String action name or "N/A" if parsing failed.
    """
    try:
        action_dic = json.loads(model_output)
        action = action_dic["action"]
        if action == "LEFT":
            return 0, action
        elif action == "RIGHT":
            return 1, action
        elif action == "UP":
            return 2, action
        elif action == "DOWN":
            return 3, action
        else:
            return -1, "N/A"
    except Exception:
        return -1, "N/A"


def get_astar_distance(env, observation):
    """Calculate the optimal path length using A* algorithm.

    Runs the A* pathfinding algorithm on a cloned environment to determine
    the optimal number of steps required to reach the goal from the current state.

    Args:
        env: The environment instance.
        observation: Current observation from the environment.

    Returns:
        int: The number of steps in the optimal A* path to the goal.
    """
    astar = AlphaStarAgent()
    clone_env = deepcopy(env)
    trajectory = generate_one_trajectory(
        env=clone_env,
        observation=observation,
        info={},
        agent=astar,
        max_steps_per_trajectory=env.unwrapped.width**2,
    )
    return len(trajectory.steps)


def get_dynamic_max_steps(astar_distance: int) -> int:
    """Calculate dynamic max steps based on A* optimal path length.

    Computes a reasonable upper bound for trajectory length as 1.5 times
    the optimal A* path length. This allows some slack for suboptimal
    actions while preventing excessively long trajectories.

    Args:
        astar_distance: The optimal A* path length to the goal.

    Returns:
        int: The dynamic max steps (ceil of 1.5 * astar_distance).
    """
    return math.ceil(1.5 * astar_distance)


def to_dic_list(txt, tokenizer, groups=["prompt"], start_idx=0):
    """Convert text to a list of token dictionaries with metadata.

    Tokenizes the input text and creates a structured representation where each token
    has an ID, token string, token ID, and group labels for categorization.

    Args:
        txt: The text to tokenize.
        tokenizer: HuggingFace tokenizer instance.
        groups: List of group labels to assign to all tokens (default: ["prompt"]).
        start_idx: Starting index for token IDs (default: 0).

    Returns:
        list[dict]: List of token dictionaries, each containing:
            - id: Sequential token index
            - token: Token string representation
            - token_id: Vocabulary ID of the token
            - token_groups: List of group labels for this token
    """
    tokens = tokenizer.tokenize(txt)
    ids = tokenizer.encode(txt)

    out = []
    for i, (t, id) in enumerate(zip(tokens, ids), start=start_idx):
        out.append(
            {"id": i, "token": t, "token_id": id, "token_groups": deepcopy(groups)}
        )
    return out


def annotate_output_tokens(model_name: str, output_tokens):
    """Annotate output tokens with model-specific group labels.

    Adds semantic group labels to output tokens based on model-specific formatting
    and special tokens. Currently supports the GPT-OSS-20B model format with
    structured output channels (analysis, final, etc.).

    Args:
        model_name: Name of the model in format "provider/model_id".
        output_tokens: List of token dictionaries from to_dic_list().

    Returns:
        list[dict]: The same token list with updated token_groups containing
            labels like 'template', 'analysis', 'final', 'action' based on
            the model's output structure.

    Raises:
        NotImplementedError: If the model is not supported for annotation.
    """
    if "openai/gpt-oss-20b" in model_name:
        template_special_tokens = {
            "<|channel|>",
            "<|message|>",
            "<|end|>",
            "<|start|>",
            "<|return|>",
        }

        in_template_mode = True
        current_section = None

        for token in output_tokens:
            token_str = token["token"]

            if token_str in template_special_tokens:
                token["token_groups"].append("template")

                if token_str == "<|end|>":
                    in_template_mode = True
                elif token_str == "<|message|>":
                    in_template_mode = False
                elif token_str == "<|return|>":
                    current_section = None

            elif in_template_mode:
                # Template mode: tokens like 'analysis', 'final', 'assistant'
                token["token_groups"].append("template")

                # Track which section we're about to enter
                if token_str in ["analysis", "final"]:
                    current_section = token_str

            else:
                # Content mode
                if current_section:
                    token["token_groups"].append(current_section)

                # Check for action words (case-insensitive, ignoring Ġ and Ċ)
                clean_token = token_str.replace("Ġ", "").replace("Ċ", "")
                if re.match(r"^(up|down|left|right)$", clean_token, re.IGNORECASE):
                    token["token_groups"].append("action")

        return output_tokens
    else:
        raise NotImplementedError(
            f"The selected model {model_name} is not supported for output annotation."
        )


def generate_trajectory(
    env: FullObservabilityTextWrapper,
    agent: LLMAgent,
    max_steps_per_trajectory: int,
    generation_kwargs: dict = {},
    metadata: dict = {},
    verbose: bool = False,
    enable_dynamic_max_steps: bool = False,
    use_safe_reset: bool = False,
    remove_door_from_env: bool = False,
    skip_reset: bool = False,
):
    """Generate a complete agent trajectory in the environment.

    Runs an LLM agent through the environment, collecting observations, actions,
    rewards, and model outputs at each step. Also computes A* optimal distance
    and records agent/goal positions.

    Args:
        env: The wrapped environment instance with text observations.
        agent: The LLM agent to generate actions.
        max_steps_per_trajectory: Maximum number of steps before truncating.
        generation_kwargs: Dictionary of parameters for the LLM completion call
            (e.g., temperature, top_p, max_tokens, top_logprobs).
        metadata: Additional metadata to include (currently unused).
        verbose: If True, log detailed information during generation.
        enable_dynamic_max_steps: If True, override max_steps_per_trajectory with
            a dynamic value based on 1.5x the A* optimal path length.
        use_safe_reset: If True, use safe_reset() which resets agent position
            without regenerating the grid. Useful for generating multiple
            trajectories on the same grid layout.
        remove_door_from_env: If True, remove the door from the environment after reset (keeps the key).
        skip_reset: If True, skip environment reset and use current state. Useful when
            environment is already pre-configured (e.g., key already removed). Default: False.

    Returns:
        Trajectory: A Trajectory object containing:
            - steps: List of Step objects with observations, actions, rewards, and metadata
            - final_reward: Total accumulated reward
            - traj_metadata: Dictionary with start/goal positions and A* distance
    """
    generation_kwargs["allowed_openai_params"] = list(generation_kwargs.keys())

    steps: list[Step] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    # Use safe_reset if specified, otherwise use regular reset
    if use_safe_reset:
        env.unwrapped.safe_reset()
        # Re-generate the text observation after safe reset using the wrapper's render method
        observation = env._render()
    else:
        observation, _ = env.reset()

    if skip_reset:
        # Don't reset the environment, just get the current observation
        raw_obs = env.unwrapped.gen_obs()
        observation = env.observation(raw_obs)
    else:
        observation, _ = env.reset()

        # Remove the door after reset if requested
        if remove_door_from_env:
            env.unwrapped.grid = remove_door(env.unwrapped).grid
            # Regenerate observation after modifying the grid
            raw_obs = env.unwrapped.gen_obs()
            observation = env.observation(raw_obs)

    traj_metadata = {}
    start_pos = tuple(int(x) for x in env.unwrapped.agent_pos)
    goal_pos = tuple(int(x) for x in env.unwrapped.goal_pos)
    traj_metadata["agent_start_coordinates"] = start_pos[1], start_pos[0]
    traj_metadata["goal_coordinates"] = goal_pos[1], goal_pos[0]
    traj_metadata["astar_distance"] = get_astar_distance(env, observation)

    if enable_dynamic_max_steps:
        max_steps_per_trajectory = get_dynamic_max_steps(
            traj_metadata["astar_distance"]
        )

    while not (terminated or truncated):
        if (
            max_steps_per_trajectory is not None
            and step_count >= max_steps_per_trajectory
        ):
            break

        unwrapped_env = getattr(env, "unwrapped", env)
        prompt = agent._generate_action_query_prompt(unwrapped_env)
        if verbose:
            logger.info(f"Step {step_count}")
            logger.info(agent._get_text_observation(env))
        full_output, cost = completion_with_retry(
            model=agent.model_name,
            messages=[{"role": "user", "content": prompt}],
            logprobs=True,
            **generation_kwargs,
        )
        final_output = full_output.choices[0].message.content
        action, action_name = parse_action(final_output)
        logprobs_serialized = agent._finalize_cost_and_logprobs(
            cost, full_output, generation_kwargs.get("top_logprobs") is not None
        )

        if verbose:
            print("Output text:", final_output)
            print("Predicted action:", action_name)

        metadata = agent._build_base_metadata(action, cost, logprobs_serialized)
        metadata["action"] = action_name

        # Capture carrying_key status before taking the step
        base_env = getattr(env, "unwrapped", env)
        carrying_key = False
        if hasattr(base_env, "carrying") and base_env.carrying is not None:
            carrying_key = base_env.carrying.type == "key"
        metadata["carrying_key"] = carrying_key

        next_obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        steps.append(
            Step(
                observation=str(observation),
                action=action,
                reward=float(reward),
                metadata=metadata,
                note=None,
            )
        )

        observation = next_obs
        step_count += 1
    traj_obj = Trajectory(
        steps=steps,
        final_reward=total_reward,
        traj_metadata=traj_metadata,
    )
    return traj_obj


def upload_to_huggingface(
    file_path: str | Path,
    repo_id: str,
    path_in_repo: str | None = None,
    hf_token: str | None = None,
    repo_type: str = "dataset",
    commit_message: str | None = None,
) -> str:
    """Upload a file to a Hugging Face repository.

    Uploads a single file to the specified Hugging Face Hub repository. Handles
    authentication via token parameter or environment variable.

    Args:
        file_path: Local path to the file to upload.
        repo_id: The Hugging Face repository ID (e.g., "username/repo-name").
        path_in_repo: Path where the file will be stored in the repo.
            If None, uses the original filename.
        hf_token: Hugging Face API token. If None, attempts to use the
            HF_TOKEN environment variable or cached credentials.
        repo_type: Type of repository ("dataset", "model", or "space").
        commit_message: Commit message for the upload. If None, generates
            a default message.

    Returns:
        str: URL of the uploaded file on Hugging Face Hub.

    Raises:
        ValueError: If authentication fails or file doesn't exist.
        Exception: If upload fails after authentication.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise ValueError(f"File not found: {file_path}")

    # Handle authentication
    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    # If no token provided, huggingface_hub will use cached credentials

    api = HfApi()

    # Use filename if path_in_repo not specified
    if path_in_repo is None:
        path_in_repo = file_path.name

    # Generate default commit message
    if commit_message is None:
        commit_message = f"Upload {file_path.name}"

    try:
        url = api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )
        logger.info(
            f"Successfully uploaded {file_path.name} to {repo_id}/{path_in_repo}"
        )
        return url
    except Exception as e:
        logger.error(f"Failed to upload {file_path} to Hugging Face: {e}")
        raise


def upload_directory_to_huggingface(
    directory: str | Path,
    repo_id: str,
    glob_pattern: str = "*.json",
    hf_token: str | None = None,
    repo_type: str = "dataset",
    num_workers: int = 8,
) -> str:
    """Upload all matching files from a directory to a Hugging Face repository.

    Uses upload_large_folder for reliable uploads of large directories with many
    files. This method is resumable, uses multiple workers for parallel uploads,
    and automatically handles batching to avoid timeouts.

    Useful for uploading trajectory and grid JSON files generated by
    get_trajectories_multiple_per_grid.

    Note: Files are uploaded to the root of the repository. If you need files
    in a subdirectory, organize them locally in a parent folder first.

    Args:
        directory: Path to the directory containing files to upload.
        repo_id: The Hugging Face repository ID (e.g., "username/repo-name").
        glob_pattern: Glob pattern to match files (default: "*.json").
        hf_token: Hugging Face API token. If None, attempts to use the
            HF_TOKEN environment variable or cached credentials.
        repo_type: Type of repository ("dataset", "model", or "space").
        num_workers: Number of parallel workers for uploading (default: 8).
            Increase for faster uploads on fast connections.

    Returns:
        str: URL of the repository on Hugging Face Hub.

    Raises:
        ValueError: If directory doesn't exist or no matching files found.
        Exception: If upload fails after authentication.
    """
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    # Find all matching files to validate and log count
    file_paths = sorted(directory.glob(glob_pattern))

    if not file_paths:
        raise ValueError(f"No files matching '{glob_pattern}' found in {directory}")

    logger.info(
        f"Found {len(file_paths)} files matching '{glob_pattern}' in {directory}"
    )

    # Handle authentication
    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    api = HfApi()

    # Ensure repo exists
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)

    # Convert glob pattern to allow_patterns format
    # e.g., "*.json" -> ["*.json"], "*_traj*.json" -> ["*_traj*.json"]
    allow_patterns = [glob_pattern]

    try:
        # Use upload_large_folder for reliable uploads of many files
        # This is resumable and handles batching automatically
        api.upload_large_folder(
            folder_path=str(directory),
            repo_id=repo_id,
            repo_type=repo_type,
            allow_patterns=allow_patterns,
            num_workers=num_workers,
        )
        repo_url = f"https://huggingface.co/{repo_type}s/{repo_id}"
        logger.info(f"Successfully uploaded {len(file_paths)} files to {repo_id}")
        return repo_url
    except Exception as e:
        logger.error(f"Failed to upload directory to Hugging Face: {e}")
        raise


def upload_files_to_huggingface(
    file_paths: list[str | Path],
    repo_id: str,
    path_prefix: str = "",
    hf_token: str | None = None,
    repo_type: str = "dataset",
    commit_message: str | None = None,
) -> list[str]:
    """Upload multiple files to a Hugging Face repository in a single commit.

    Uploads a batch of files to the specified Hugging Face Hub repository
    efficiently in a single commit operation.

    Args:
        file_paths: List of local file paths to upload.
        repo_id: The Hugging Face repository ID (e.g., "username/repo-name").
        path_prefix: Prefix path in the repository for all files
            (e.g., "trajectories/" to put files in a subfolder).
        hf_token: Hugging Face API token. If None, attempts to use the
            HF_TOKEN environment variable or cached credentials.
        repo_type: Type of repository ("dataset", "model", or "space").
        commit_message: Commit message for the upload. If None, generates
            a default message.

    Returns:
        list[str]: List of URLs for the uploaded files.

    Raises:
        ValueError: If authentication fails or any file doesn't exist.
        Exception: If upload fails after authentication.
    """
    # Validate all files exist
    validated_paths = []
    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        validated_paths.append(path)

    if not validated_paths:
        logger.warning("No files to upload")
        return []

    # Handle authentication
    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    api = HfApi()

    # Generate default commit message
    if commit_message is None:
        commit_message = f"Upload {len(validated_paths)} trajectory files"

    # Prepare upload operations
    operations = []
    for path in validated_paths:
        path_in_repo = f"{path_prefix}{path.name}" if path_prefix else path.name
        operations.append(
            CommitOperationAdd(
                path_in_repo=path_in_repo,
                path_or_fileobj=str(path),
            )
        )

    try:
        commit_info = api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=operations,
            commit_message=commit_message,
        )
        logger.info(
            f"Successfully uploaded {len(validated_paths)} files to {repo_id} "
            f"(commit: {commit_info.commit_url})"
        )
        # Return URLs for all uploaded files
        base_url = f"https://huggingface.co/{repo_type}s/{repo_id}/blob/main"
        urls = [
            f"{base_url}/{path_prefix}{path.name}"
            if path_prefix
            else f"{base_url}/{path.name}"
            for path in validated_paths
        ]
        return urls
    except Exception as e:
        logger.error(f"Failed to upload files to Hugging Face: {e}")
        raise
