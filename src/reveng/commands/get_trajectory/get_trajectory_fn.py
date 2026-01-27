"""Generate and save agent trajectories in navigation environments with detailed token-level analysis."""

import copy
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from reveng.agents.llm_agent import LLMAgent
from reveng.commands.get_trajectory.compact_json_encoder import CompactJSONEncoder
from reveng.commands.get_trajectory.get_trajectory_utils import (
    DEFAULT_TRANSFORM_NAMES,
    annotate_output_tokens,
    generate_trajectory,
    get_transformed_environments,
    to_dic_list,
    upload_directory_to_huggingface,
    upload_files_to_huggingface,
    upload_to_huggingface,
)
from reveng.commands.get_trajectory.rate_limiter import (
    RateLimiter,
)
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.environment_generator.key_minigrid import Key2PathMinigridEnv
from reveng.environment_generator.rooms_minigrid import RoomsMinigridEnv
from reveng.environment_generator.utils import remove_key
from reveng.environment_generator.wrappers.text_obs_wrapper import (
    FullObservabilityTextWrapper,
)

logger = logging.getLogger(__file__)

logging.getLogger("LiteLLM").setLevel(logging.WARNING)


def get_trajectory(
    grid_size: int = 5,
    grid_complexity: float = 0.0,
    max_steps_per_trajectory: int = 50,
    max_tokens: int = 10000,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_logprobs: int = 5,
    seed: int = 42,
    reasoning_effort: Literal["low", "medium", "high"] = "low",
    model_name: str = "together_ai/openai/gpt-oss-20b",
    observation_placeholders: list[str] = ["grid_state"],
    output_path: str = "get_trajectory_example_output.json",
    verbose: bool = False,
    enable_dynamic_max_steps: bool = False,
    hf_repo_id: str | None = None,
    hf_path_in_repo: str | None = None,
    hf_token: str | None = None,
    env: FullObservabilityTextWrapper | None = None,
    use_safe_reset: bool = False,
    transform_type: str = "base",
):
    """Generate an agent trajectory in a 2D navigation environment and save detailed results to JSON.

    Creates a Simple2D navigation environment, runs an LLM agent to generate a trajectory,
    and saves comprehensive information including grid parameters, model parameters, prompt
    template with token-level analysis, and trajectory steps with token probabilities.

    Args:
        grid_size: Size of the square grid environment.
        grid_complexity: Complexity level of obstacles in the grid (higher = more obstacles).
        max_steps_per_trajectory: Maximum number of steps to generate in the trajectory.
        max_tokens: Maximum tokens for model generation per step.
        temperature: Sampling temperature for the model (higher = more random).
        top_p: Nucleus sampling parameter (cumulative probability threshold).
        top_logprobs: Number of top log probabilities to return for each token.
        seed: Random seed for reproducibility.
        reasoning_effort: Reasoning effort level for the model ("low", "medium", or "high").
        model_name: Name of the model in format "provider/model_id".
        observation_placeholders: List of placeholder names in the prompt template.
        output_path: Path to save the output JSON file.
        verbose: If True, print detailed logging during trajectory generation.
        enable_dynamic_max_steps: If True, override max_steps_per_trajectory with
            a dynamic value based on 1.5x the A* optimal path length.
        hf_repo_id: Hugging Face repository ID to upload to (e.g., "username/repo-name").
            If None, no upload is performed.
        hf_path_in_repo: Path within the HF repo where the file will be stored.
            If None, uses the output filename.
        hf_token: Hugging Face API token. If None, uses HF_TOKEN env var or cached credentials.
        env: Optional pre-created environment. If provided, grid_size and grid_complexity
            are ignored. Useful for generating multiple trajectories on the same grid.
        use_safe_reset: If True, use safe_reset() which resets agent position without
            regenerating the grid. Only applicable when env is provided.
        transform_type: Type of environment transform applied ("base", "RotateEnv",
            "ReflectEnv", "TransposeEnv", "StartGoalSwap"). Stored in grid_params.

    Returns:
        str | None: URL of uploaded file if hf_repo_id is provided, otherwise None.
        Results are always saved to the specified output_path.

    The output JSON structure follows the format expected by the trace viewer: https://github.com/SPAR-Telos/interp/tree/trace-viewer
        - grid_params: Grid configuration (size, complexity, start/goal positions, A* distance, legend)
        - model_params: Model configuration (name, provider, sampling parameters, seed)
        - prompt: Prompt template with token-level annotations (prefix, suffix, placeholder tokens)
        - steps: List of trajectory steps, each containing:
            - step_id: Step number
            - grid_state: Grid visualization as list of strings
            - grid_state_tokens: Tokenized grid state with annotations
            - prompt_suffix_tokens: Tokenized prompt suffix
            - agent_action: Action taken by the agent
            - output_text: Model's generated output text
            - output_tokens: Tokenized output with probabilities and annotations
    """
    # Use provided env or create a new one
    if env is None:
        base_env = FullObservabilityTextWrapper(
            Simple2DNavigationEnv(size=grid_size, complexity=grid_complexity)
        )
        use_safe_reset = False  # Cannot use safe_reset on a fresh env
    else:
        base_env = env
        # Get grid_size from the provided env
        grid_size = base_env.unwrapped.width
        grid_complexity = base_env.unwrapped.complexity

    agent = LLMAgent(model_name)
    model_id = "/".join(model_name.split("/")[1:])
    provider = model_name.split("/")[0]
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)

    traj = generate_trajectory(
        env=base_env,
        agent=agent,
        max_steps_per_trajectory=max_steps_per_trajectory,
        generation_kwargs={
            "top_logprobs": top_logprobs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning_effort": reasoning_effort,
            "seed": seed,
        },
        verbose=verbose,
        enable_dynamic_max_steps=enable_dynamic_max_steps,
        use_safe_reset=use_safe_reset,
    )

    grid_params = {}

    grid_params["grid_width"] = grid_size
    grid_params["grid_height"] = grid_size
    grid_params["grid_complexity"] = grid_complexity
    grid_params["fully_observable"] = True
    grid_params["transform_type"] = transform_type
    grid_params["astar_distance"] = traj.traj_metadata["astar_distance"]
    grid_params["agent_start_coordinates"] = traj.traj_metadata[
        "agent_start_coordinates"
    ]
    grid_params["goal_coordinates"] = traj.traj_metadata["goal_coordinates"]
    grid_params["legend"] = base_env.grid_cells

    grid_symbols = [cell["symbol"] for cell in base_env.grid_cells.values()]

    model_params = {}

    model_params["model_id"] = model_id
    model_params["provider"] = provider
    model_params["interface"] = "litellm"
    model_params["n_interactions_in_context"] = 0
    model_params["max_tokens"] = max_tokens
    model_params["max_steps_per_trajectory"] = max_steps_per_trajectory
    model_params["temperature"] = temperature
    model_params["reasoning_effort"] = reasoning_effort
    model_params["top_p"] = top_p
    model_params["top_logprobs"] = top_logprobs
    model_params["seed"] = seed

    prompt = {}

    render_kwargs = {"grid_state": "{{grid_state}}"}
    template = agent._template.render(**render_kwargs)
    formatted_template: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": template}],
        tokenize=False,
        add_generation_prompt=True,
    )
    template_tokens = to_dic_list(formatted_template, tokenizer)

    # TODO: Handle multiple observation placeholders if needed (for now, we default to [0] only)
    observation_placeholder = "{{" + observation_placeholders[0] + "}}"
    prefix, suffix = formatted_template.split(observation_placeholder)
    raw_prefix, raw_suffix = template.split(observation_placeholder)
    prompt["prompt_template"] = formatted_template
    prompt["prompt_template_n_tokens"] = len(template_tokens)
    prompt["prompt_prefix_tokens"] = to_dic_list(prefix, tokenizer)
    raw_prefix_tokens = to_dic_list(raw_prefix, tokenizer)
    start_raw_prefix_idx = len(prompt["prompt_prefix_tokens"]) - len(raw_prefix_tokens)
    for i in range(start_raw_prefix_idx):
        prompt["prompt_prefix_tokens"][i]["token_groups"] += ["template"]
    prompt["prompt_prefix_n_tokens"] = len(prompt["prompt_prefix_tokens"])
    prompt["prompt_placeholder_tokens"] = to_dic_list(
        observation_placeholder, tokenizer, groups=["prompt", "placeholder"]
    )
    prompt["prompt_placeholder_n_tokens"] = len(prompt["prompt_placeholder_tokens"])
    prompt["prompt_suffix_tokens"] = to_dic_list(suffix, tokenizer)
    raw_suffix_tokens = to_dic_list(raw_suffix, tokenizer)
    start_raw_suffix_idx = (
        len(prompt["prompt_suffix_tokens"]) - len(raw_suffix_tokens) + 1
    )
    for i in range(
        len(prompt["prompt_suffix_tokens"]) - start_raw_suffix_idx,
        len(prompt["prompt_suffix_tokens"]) - 1,
    ):
        prompt["prompt_suffix_tokens"][i]["token_groups"] += ["template"]
    prompt["prompt_suffix_n_tokens"] = len(prompt["prompt_suffix_tokens"])

    steps = []

    for step_id, traj_step in enumerate(traj.steps):
        step_dic = {}
        step_dic["step_id"] = step_id
        step_dic["grid_state"] = traj_step.observation.split("\n")
        step_dic["grid_state_tokens"] = to_dic_list(
            traj_step.observation, tokenizer, groups=["prompt", "grid_state"]
        )
        step_dic["grid_state_n_tokens"] = len(step_dic["grid_state_tokens"])

        for i, t in enumerate(step_dic["grid_state_tokens"]):
            if any(sym in t["token"] for sym in grid_symbols):
                step_dic["grid_state_tokens"][i]["token_groups"] += ["grid_tile"]

        step_dic["prompt_suffix_tokens"] = prompt["prompt_suffix_tokens"]
        step_dic["prompt_suffix_n_tokens"] = len(step_dic["prompt_suffix_tokens"])
        step_dic["agent_action"] = traj_step.metadata["action"]

        out_tokens = [t["token"] for t in traj_step.metadata["logprobs"]]
        step_dic["output_text"] = tokenizer.convert_tokens_to_string(out_tokens)
        step_dic["output_tokens"] = to_dic_list(
            step_dic["output_text"], tokenizer, groups=["output"]
        )
        step_dic["output_n_tokens"] = len(step_dic["output_tokens"])
        step_dic["output_tokens"] = annotate_output_tokens(
            model_name, step_dic["output_tokens"]
        )

        # Note: output_tokens (from local tokenizer) and logprobs (from API) may differ
        # in length due to tokenization differences, so we need bounds checking
        api_logprobs = traj_step.metadata["logprobs"]
        for i, t in enumerate(step_dic["output_tokens"]):
            if i >= len(api_logprobs):
                # Local tokenizer produced more tokens than API returned
                break
            if "top_logprobs" not in api_logprobs[i] or "template" in t["token_groups"]:
                continue
            curr_probs = {}
            for logprob_dic in api_logprobs[i]["top_logprobs"]:
                curr_probs[logprob_dic["token"]] = np.round(
                    np.exp(logprob_dic["logprob"]), 4
                )
            step_dic["output_tokens"][i]["probabilities"] = curr_probs

        steps.append(step_dic)

    out = {
        "grid_params": grid_params,
        "model_params": model_params,
        "prompt": prompt,
        "steps": steps,
    }
    with open(output_path, "w") as f:
        json.dump(out, f, cls=CompactJSONEncoder, ensure_ascii=False, indent=4)

    # Upload to Hugging Face if repo_id is provided
    if hf_repo_id is not None:
        return upload_to_huggingface(
            file_path=output_path,
            repo_id=hf_repo_id,
            path_in_repo=hf_path_in_repo,
            hf_token=hf_token,
        )
    return None


def get_trajectories(
    grid_sizes: list[int] = [5],
    grid_complexities: list[float] = [0.0],
    max_steps_per_trajectory: int = 50,
    max_tokens: int = 10000,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_logprobs: int = 5,
    seed: int = 42,
    reasoning_effort: Literal["low", "medium", "high"] = "low",
    model_names: list[str] = ["together_ai/openai/gpt-oss-20b"],
    observation_placeholders: list[str] = ["grid_state"],
    output_dir: str = ".",
    verbose: bool = False,
    enable_dynamic_max_steps: bool = False,
    num_examples: int = 1,
    max_workers: int | None = None,
    enable_rate_limit: bool = False,
    rate_limit: int = 1000,
    rate_limit_period: float = 300.0,
    hf_repo_id: str | None = None,
    hf_path_prefix: str = "",
    hf_token: str | None = None,
):
    """Generate multiple agent trajectories across parameter combinations in parallel.

    Creates trajectories for all combinations of grid_sizes, grid_complexities, and model_names,
    running the specified number of examples per combination in parallel. Each trajectory is saved
    to a separate JSON file with a name based on the parameters.

    Args:
        grid_sizes: List of grid sizes to use for trajectory generation.
        grid_complexities: List of grid complexity levels to use.
        max_steps_per_trajectory: Maximum number of steps to generate in each trajectory.
        max_tokens: Maximum tokens for model generation per step.
        temperature: Sampling temperature for the model (higher = more random).
        top_p: Nucleus sampling parameter (cumulative probability threshold).
        top_logprobs: Number of top log probabilities to return for each token.
        seed: Base random seed for reproducibility. Each example uses seed + example_id.
        reasoning_effort: Reasoning effort level for the model ("low", "medium", or "high").
        model_names: List of model names in format "provider/model_id".
        observation_placeholders: List of placeholder names in the prompt template.
        output_dir: Directory to save the output JSON files.
        verbose: If True, print detailed logging during trajectory generation.
        enable_dynamic_max_steps: If True, override max_steps_per_trajectory with
            a dynamic value based on 1.5x the A* optimal path length.
        num_examples: Number of different examples to generate per parameter combination.
        max_workers: Maximum number of parallel workers. If None, uses min(32, total_tasks).
        enable_rate_limit: If True, enforce rate limiting on API requests.
        rate_limit: Maximum number of requests allowed per rate_limit_period.
        rate_limit_period: Time period in seconds for rate limiting (default: 300 = 5 minutes).
        hf_repo_id: Hugging Face repository ID to upload to (e.g., "username/repo-name").
            If None, no upload is performed.
        hf_path_prefix: Path prefix within the HF repo (e.g., "trajectories/" to put files
            in a subfolder).
        hf_token: Hugging Face API token. If None, uses HF_TOKEN env var or cached credentials.

    Returns:
        list[str] | None: List of URLs if uploaded to HF, otherwise None.
        Results are always saved to individual JSON files in output_dir with format:
        {model_sanitized}_size{grid_size}_comp{grid_complexity}_{example_id}.json
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate all parameter combinations
    all_combinations = list(
        product(grid_sizes, grid_complexities, model_names, range(num_examples))
    )

    total_tasks = len(all_combinations)
    logger.info(
        f"Generating {total_tasks} trajectories across {len(grid_sizes)} grid sizes, "
        f"{len(grid_complexities)} complexities, {len(model_names)} models, "
        f"with {num_examples} examples each."
    )

    # Create rate limiter if enabled
    rate_limiter: RateLimiter | None = None
    if enable_rate_limit:
        rate_limiter = RateLimiter(rate_limit=rate_limit, period=rate_limit_period)
        logger.info(
            f"Rate limiting enabled: {rate_limit} requests per {rate_limit_period} seconds "
            f"({rate_limit / rate_limit_period:.2f} requests/second)"
        )

    def _generate_single_task(params: tuple) -> dict:
        """Generate a single trajectory for given parameters."""
        # Acquire rate limit token if enabled
        if rate_limiter is not None:
            rate_limiter.acquire()

        grid_size, grid_complexity, model_name, example_id = params
        task_seed = seed + example_id

        # Sanitize model name for filename
        model_sanitized = model_name.replace("/", "_").replace(".", "_")

        output_filename = (
            f"{model_sanitized}_size{grid_size}_comp{grid_complexity}_{example_id}.json"
        )
        output_path = str(Path(output_dir) / output_filename)

        if verbose:
            logger.info(
                f"Starting task: model={model_name}, size={grid_size}, "
                f"complexity={grid_complexity}, example={example_id}, seed={task_seed}"
            )

        try:
            get_trajectory(
                grid_size=grid_size,
                grid_complexity=grid_complexity,
                max_steps_per_trajectory=max_steps_per_trajectory,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_logprobs=top_logprobs,
                seed=task_seed,
                reasoning_effort=reasoning_effort,
                model_name=model_name,
                observation_placeholders=observation_placeholders,
                output_path=output_path,
                verbose=verbose,
                enable_dynamic_max_steps=enable_dynamic_max_steps,
            )
            return {"status": "success", "output_path": output_path, "params": params}
        except Exception as e:
            logger.error(f"Failed to generate trajectory for {params}: {e}")
            return {"status": "error", "error": str(e), "params": params}

    # Set default max_workers if not specified
    if max_workers is None:
        max_workers = min(32, total_tasks)

    # Adjust max_workers based on rate limit to avoid excessive idle workers
    if enable_rate_limit and rate_limiter is not None:
        # Calculate the sustainable number of workers based on request rate
        # If each task takes at least 1 second, don't exceed the rate limit per second
        sustainable_workers = min(max_workers, int(rate_limiter.tokens_per_second * 2))
        if sustainable_workers < max_workers:
            logger.info(
                f"Adjusting max_workers from {max_workers} to {sustainable_workers} "
                f"based on rate limit ({rate_limiter.tokens_per_second:.2f} req/sec)"
            )
            max_workers = sustainable_workers

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_single_task, combo): combo
            for combo in all_combinations
        }

        for future in tqdm(
            as_completed(futures),
            total=total_tasks,
            desc="Generating trajectories",
            unit="trajectory",
        ):
            combo = futures[future]
            try:
                result = future.result()
                results.append(result)
                if result["status"] != "success":
                    tqdm.write(
                        f"Failed for {combo}: {result.get('error', 'Unknown error')}"
                    )
            except Exception as e:
                tqdm.write(f"Exception for {combo}: {e}")
                results.append({"status": "error", "error": str(e), "params": combo})

    success_count = sum(1 for r in results if r["status"] == "success")
    logger.info(f"Completed {success_count}/{total_tasks} trajectories successfully.")

    # Upload to Hugging Face if repo_id is provided
    if hf_repo_id is not None and success_count > 0:
        successful_paths = [
            r["output_path"] for r in results if r["status"] == "success"
        ]
        return upload_files_to_huggingface(
            file_paths=successful_paths,
            repo_id=hf_repo_id,
            path_prefix=hf_path_prefix,
            hf_token=hf_token,
        )
    return None


def get_trajectory_no_key_env(
    env,
    max_steps_per_trajectory: int = 30,
    max_tokens: int = 10000,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_logprobs: int = 5,
    seed: int = 42,
    reasoning_effort: Literal["low", "medium", "high"] = "low",
    model_name: str = "together_ai/openai/gpt-oss-20b",
    template_name: str = "grid_full_observability_instrumental_goals.j2",
    observation_placeholders: list[str] = ["grid_state"],
    output_path: str = "get_trajectory_no_key_example_output.json",
    verbose: bool = False,
):
    """Generate an agent trajectory in an environment without the key and save detailed results to JSON.

    This function takes a pre-configured environment (with the key already removed using remove_key),
    wraps it with FullObservabilityTextWrapper, runs an LLM agent to generate a trajectory,
    and saves comprehensive information including grid parameters, model parameters, prompt
    template with token-level analysis, and trajectory steps with token probabilities.

    Args:
        env: The environment with the key already removed (unwrapped base environment).
        max_steps_per_trajectory: Maximum number of steps to generate in the trajectory.
        max_tokens: Maximum tokens for model generation per step.
        temperature: Sampling temperature for the model (higher = more random).
        top_p: Nucleus sampling parameter (cumulative probability threshold).
        top_logprobs: Number of top log probabilities to return for each token.
        seed: Random seed for reproducibility.
        reasoning_effort: Reasoning effort level for the model ("low", "medium", or "high").
        model_name: Name of the model in format "provider/model_id".
        template_name: Name of the Jinja2 template file to use for prompts.
        observation_placeholders: List of placeholder names in the prompt template.
        output_path: Path to save the output JSON file.
        verbose: If True, print detailed logging during trajectory generation.

    Returns:
        None. Results are saved to the specified output_path.

    The output JSON structure follows the format expected by the trace viewer: https://github.com/SPAR-Telos/interp/tree/trace-viewer
        - grid_params: Grid configuration (includes "key_removed": true flag)
        - model_params: Model configuration (name, provider, sampling parameters, seed)
        - prompt: Prompt template with token-level annotations (prefix, suffix, placeholder tokens)
        - steps: List of trajectory steps, each containing detailed token information
    """
    # Wrap the environment with FullObservabilityTextWrapper
    base_env = FullObservabilityTextWrapper(env)

    # Get template path
    template_path = Path(__file__).parent.parent.parent / "templates" / template_name

    # Create agent with custom template
    agent = LLMAgent(model_name=model_name, template_path=template_path)
    model_id = "/".join(model_name.split("/")[1:])
    provider = model_name.split("/")[0]
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)

    # Generate trajectory with skip_reset=True to avoid resetting the pre-configured environment
    traj = generate_trajectory(
        env=base_env,
        agent=agent,
        max_steps_per_trajectory=max_steps_per_trajectory,
        generation_kwargs={
            "top_logprobs": top_logprobs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning_effort": reasoning_effort,
            "seed": seed,
        },
        verbose=verbose,
        remove_door_from_env=False,
        skip_reset=True,
    )

    # Prepare grid parameters
    grid_params = {}
    grid_size = env.width
    grid_params["grid_width"] = grid_size
    grid_params["grid_height"] = grid_size
    grid_params["key_removed"] = True  # Flag to indicate key was removed
    grid_params["fully_observable"] = True
    grid_params["astar_distance"] = traj.traj_metadata["astar_distance"]
    grid_params["agent_start_coordinates"] = traj.traj_metadata[
        "agent_start_coordinates"
    ]
    grid_params["goal_coordinates"] = traj.traj_metadata["goal_coordinates"]
    grid_params["legend"] = base_env.grid_cells

    grid_symbols = [cell["symbol"] for cell in base_env.grid_cells.values()]

    # Prepare model parameters
    model_params = {}
    model_params["model_id"] = model_id
    model_params["provider"] = provider
    model_params["interface"] = "litellm"
    model_params["template_name"] = template_name
    model_params["n_interactions_in_context"] = 0
    model_params["max_tokens"] = max_tokens
    model_params["max_steps_per_trajectory"] = max_steps_per_trajectory
    model_params["temperature"] = temperature
    model_params["reasoning_effort"] = reasoning_effort
    model_params["top_p"] = top_p
    model_params["top_logprobs"] = top_logprobs
    model_params["seed"] = seed

    # Prepare prompt template information
    prompt = {}

    # For key-door template, we need both grid_state and carrying_key placeholders
    render_kwargs = {"grid_state": "{{grid_state}}", "carrying_key": "{{carrying_key}}"}
    template = agent._template.render(**render_kwargs)
    formatted_template: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": template}],
        tokenize=False,
        add_generation_prompt=True,
    )
    template_tokens = to_dic_list(formatted_template, tokenizer)

    # Handle the primary observation placeholder (grid_state)
    observation_placeholder = "{{" + observation_placeholders[0] + "}}"
    prefix, suffix = formatted_template.split(observation_placeholder)
    raw_prefix, raw_suffix = template.split(observation_placeholder)
    prompt["prompt_template"] = formatted_template
    prompt["prompt_template_n_tokens"] = len(template_tokens)
    prompt["prompt_prefix_tokens"] = to_dic_list(prefix, tokenizer)
    raw_prefix_tokens = to_dic_list(raw_prefix, tokenizer)
    start_raw_prefix_idx = len(prompt["prompt_prefix_tokens"]) - len(raw_prefix_tokens)
    for i in range(start_raw_prefix_idx):
        prompt["prompt_prefix_tokens"][i]["token_groups"] += ["template"]
    prompt["prompt_prefix_n_tokens"] = len(prompt["prompt_prefix_tokens"])
    prompt["prompt_placeholder_tokens"] = to_dic_list(
        observation_placeholder, tokenizer, groups=["prompt", "placeholder"]
    )
    prompt["prompt_placeholder_n_tokens"] = len(prompt["prompt_placeholder_tokens"])
    prompt["prompt_suffix_tokens"] = to_dic_list(suffix, tokenizer)
    raw_suffix_tokens = to_dic_list(raw_suffix, tokenizer)
    start_raw_suffix_idx = (
        len(prompt["prompt_suffix_tokens"]) - len(raw_suffix_tokens) + 1
    )
    for i in range(
        len(prompt["prompt_suffix_tokens"]) - start_raw_suffix_idx,
        len(prompt["prompt_suffix_tokens"]) - 1,
    ):
        prompt["prompt_suffix_tokens"][i]["token_groups"] += ["template"]
    prompt["prompt_suffix_n_tokens"] = len(prompt["prompt_suffix_tokens"])

    # Process trajectory steps
    steps = []

    for step_id, traj_step in enumerate(traj.steps):
        step_dic = {}
        step_dic["step_id"] = step_id
        step_dic["grid_state"] = traj_step.observation.split("\n")
        step_dic["grid_state_tokens"] = to_dic_list(
            traj_step.observation, tokenizer, groups=["prompt", "grid_state"]
        )
        step_dic["grid_state_n_tokens"] = len(step_dic["grid_state_tokens"])

        for i, t in enumerate(step_dic["grid_state_tokens"]):
            if any(sym in t["token"] for sym in grid_symbols):
                step_dic["grid_state_tokens"][i]["token_groups"] += ["grid_tile"]

        # Render the suffix with the actual carrying_key value for this step
        step_dic["carrying_key"] = traj_step.metadata.get("carrying_key", False)
        carrying_key_placeholder = "{{carrying_key}}"
        if carrying_key_placeholder in suffix:
            # Replace the placeholder with the actual value
            rendered_suffix = suffix.replace(
                carrying_key_placeholder, str(step_dic["carrying_key"])
            )
            step_dic["prompt_suffix_tokens"] = to_dic_list(rendered_suffix, tokenizer)
            # Mark template tokens appropriately
            raw_suffix_tokens = to_dic_list(
                raw_suffix.replace(
                    carrying_key_placeholder, str(step_dic["carrying_key"])
                ),
                tokenizer,
            )
            start_raw_suffix_idx = (
                len(step_dic["prompt_suffix_tokens"]) - len(raw_suffix_tokens) + 1
            )
            for i in range(
                len(step_dic["prompt_suffix_tokens"]) - start_raw_suffix_idx,
                len(step_dic["prompt_suffix_tokens"]) - 1,
            ):
                step_dic["prompt_suffix_tokens"][i]["token_groups"] += ["template"]
        else:
            step_dic["prompt_suffix_tokens"] = prompt["prompt_suffix_tokens"]
        step_dic["prompt_suffix_n_tokens"] = len(step_dic["prompt_suffix_tokens"])
        step_dic["agent_action"] = traj_step.metadata["action"]

        out_tokens = [t["token"] for t in traj_step.metadata["logprobs"]]
        step_dic["output_text"] = tokenizer.convert_tokens_to_string(out_tokens)
        step_dic["output_tokens"] = to_dic_list(
            step_dic["output_text"], tokenizer, groups=["output"]
        )
        step_dic["output_n_tokens"] = len(step_dic["output_tokens"])
        step_dic["output_tokens"] = annotate_output_tokens(
            model_name, step_dic["output_tokens"]
        )

        for i, t in enumerate(step_dic["output_tokens"]):
            if (
                "top_logprobs" not in traj_step.metadata["logprobs"][i]
                or "template" in t["token_groups"]
            ):
                continue
            curr_probs = {}
            for logprob_dic in traj_step.metadata["logprobs"][i]["top_logprobs"]:
                curr_probs[logprob_dic["token"]] = np.round(
                    np.exp(logprob_dic["logprob"]), 4
                )
            step_dic["output_tokens"][i]["probabilities"] = curr_probs

        steps.append(step_dic)

    # Prepare final output
    out = {
        "grid_params": grid_params,
        "model_params": model_params,
        "prompt": prompt,
        "steps": steps,
    }

    # Save to JSON file
    with open(output_path, "w") as f:
        json.dump(out, f, cls=CompactJSONEncoder, ensure_ascii=False, indent=4)


def get_trajectory_key_door_env(
    rooms_per_side: int = 2,
    add_door_key: bool = True,
    remove_door_from_env: bool = False,
    use_key_2path_env: bool = False,
    max_steps_per_trajectory: int = 30,
    max_tokens: int = 10000,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_logprobs: int = 5,
    seed: int = 42,
    reasoning_effort: Literal["low", "medium", "high"] = "low",
    model_name: str = "together_ai/openai/gpt-oss-20b",
    template_name: str = "grid_full_observability_instrumental_goals.j2",
    observation_placeholders: list[str] = ["grid_state"],
    output_path: str = "get_trajectory_key_door_example_output.json",
    verbose: bool = False,
):
    """Generate an agent trajectory in a rooms environment with key-door mechanics and save detailed results to JSON.

    Creates a RoomsMinigrid environment (or Key2PathMinigridEnv if use_key_2path_env=True) with optional key-door obstacles,
    runs an LLM agent to generate a trajectory, and saves comprehensive information including grid parameters, model parameters,
    prompt template with token-level analysis, and trajectory steps with token probabilities.

    Args:
        rooms_per_side: Number of rooms per side (2 for 2x2=4 rooms, 3 for 3x3=9 rooms). Ignored if use_key_2path_env=True.
        add_door_key: Whether to include a locked door and key in the environment. Ignored if use_key_2path_env=True.
        remove_door_from_env: If True, remove the door from the environment after reset (keeps the key).
        use_key_2path_env: If True, use Key2PathMinigridEnv instead of RoomsMinigridEnv.
        max_steps_per_trajectory: Maximum number of steps to generate in the trajectory.
        max_tokens: Maximum tokens for model generation per step.
        temperature: Sampling temperature for the model (higher = more random).
        top_p: Nucleus sampling parameter (cumulative probability threshold).
        top_logprobs: Number of top log probabilities to return for each token.
        seed: Random seed for reproducibility.
        reasoning_effort: Reasoning effort level for the model ("low", "medium", or "high").
        model_name: Name of the model in format "provider/model_id".
        template_name: Name of the Jinja2 template file to use for prompts.
        observation_placeholders: List of placeholder names in the prompt template.
        output_path: Path to save the output JSON file.
        verbose: If True, print detailed logging during trajectory generation.

    Returns:
        None. Results are saved to the specified output_path.

    The output JSON structure follows the format expected by the trace viewer: https://github.com/SPAR-Telos/interp/tree/trace-viewer
        - grid_params: Grid configuration (size, rooms_per_side, add_door_key, start/goal positions, A* distance, legend)
        - model_params: Model configuration (name, provider, sampling parameters, seed)
        - prompt: Prompt template with token-level annotations (prefix, suffix, placeholder tokens)
        - steps: List of trajectory steps, each containing:
            - step_id: Step number
            - grid_state: Grid visualization as list of strings
            - grid_state_tokens: Tokenized grid state with annotations
            - prompt_suffix_tokens: Tokenized prompt suffix
            - agent_action: Action taken by the agent
            - output_text: Model's generated output text
            - output_tokens: Tokenized output with probabilities and annotations
    """
    # Create environment based on use_key_2path_env flag
    if use_key_2path_env:
        base_env_unwrapped = Key2PathMinigridEnv(
            size=11,  # Default size for Key2PathMinigridEnv
            max_steps=max_steps_per_trajectory,
        )
        base_env_unwrapped.reset()
        base_env_unwrapped_no_key = remove_key(base_env_unwrapped)

        # Generate output path for the no-key environment
        output_path_no_key = output_path.replace(".json", "_no_key.json")

        get_trajectory_no_key_env(
            env=base_env_unwrapped_no_key,
            max_steps_per_trajectory=max_steps_per_trajectory,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_logprobs=top_logprobs,
            seed=seed,
            reasoning_effort=reasoning_effort,
            model_name=model_name,
            template_name=template_name,
            observation_placeholders=observation_placeholders,
            output_path=output_path_no_key,
            verbose=verbose,
        )

    else:
        base_env_unwrapped = RoomsMinigridEnv(
            add_door_key=add_door_key,
            max_steps=max_steps_per_trajectory,
            rooms_per_side=rooms_per_side,
        )
    base_env = FullObservabilityTextWrapper(base_env_unwrapped)

    # Get template path
    template_path = Path(__file__).parent.parent.parent / "templates" / template_name

    # Create agent with custom template
    agent = LLMAgent(model_name=model_name, template_path=template_path)
    model_id = "/".join(model_name.split("/")[1:])
    provider = model_name.split("/")[0]
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)

    traj = generate_trajectory(
        env=base_env,
        agent=agent,
        max_steps_per_trajectory=max_steps_per_trajectory,
        generation_kwargs={
            "top_logprobs": top_logprobs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning_effort": reasoning_effort,
            "seed": seed,
        },
        verbose=verbose,
        remove_door_from_env=remove_door_from_env,
        skip_reset=True,
    )

    grid_params = {}
    grid_size = base_env_unwrapped.width
    grid_params["grid_width"] = grid_size
    grid_params["grid_height"] = grid_size
    if use_key_2path_env:
        grid_params["use_key_2path_env"] = True
        # Key2PathMinigridEnv doesn't have rooms_per_side or add_door_key parameters
    else:
        grid_params["rooms_per_side"] = rooms_per_side
        grid_params["add_door_key"] = add_door_key
    grid_params["fully_observable"] = True
    grid_params["astar_distance"] = traj.traj_metadata["astar_distance"]
    grid_params["agent_start_coordinates"] = traj.traj_metadata[
        "agent_start_coordinates"
    ]
    grid_params["goal_coordinates"] = traj.traj_metadata["goal_coordinates"]
    grid_params["legend"] = base_env.grid_cells

    grid_symbols = [cell["symbol"] for cell in base_env.grid_cells.values()]

    model_params = {}
    model_params["model_id"] = model_id
    model_params["provider"] = provider
    model_params["interface"] = "litellm"
    model_params["template_name"] = template_name
    model_params["n_interactions_in_context"] = 0
    model_params["max_tokens"] = max_tokens
    model_params["max_steps_per_trajectory"] = max_steps_per_trajectory
    model_params["temperature"] = temperature
    model_params["reasoning_effort"] = reasoning_effort
    model_params["top_p"] = top_p
    model_params["top_logprobs"] = top_logprobs
    model_params["seed"] = seed

    prompt = {}

    # For key-door template, we need both grid_state and carrying_key placeholders
    render_kwargs = {"grid_state": "{{grid_state}}", "carrying_key": "{{carrying_key}}"}
    template = agent._template.render(**render_kwargs)
    formatted_template: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": template}],
        tokenize=False,
        add_generation_prompt=True,
    )
    template_tokens = to_dic_list(formatted_template, tokenizer)

    # Handle the primary observation placeholder (grid_state)
    observation_placeholder = "{{" + observation_placeholders[0] + "}}"
    prefix, suffix = formatted_template.split(observation_placeholder)
    raw_prefix, raw_suffix = template.split(observation_placeholder)
    prompt["prompt_template"] = formatted_template
    prompt["prompt_template_n_tokens"] = len(template_tokens)
    prompt["prompt_prefix_tokens"] = to_dic_list(prefix, tokenizer)
    raw_prefix_tokens = to_dic_list(raw_prefix, tokenizer)
    start_raw_prefix_idx = len(prompt["prompt_prefix_tokens"]) - len(raw_prefix_tokens)
    for i in range(start_raw_prefix_idx):
        prompt["prompt_prefix_tokens"][i]["token_groups"] += ["template"]
    prompt["prompt_prefix_n_tokens"] = len(prompt["prompt_prefix_tokens"])
    prompt["prompt_placeholder_tokens"] = to_dic_list(
        observation_placeholder, tokenizer, groups=["prompt", "placeholder"]
    )
    prompt["prompt_placeholder_n_tokens"] = len(prompt["prompt_placeholder_tokens"])
    prompt["prompt_suffix_tokens"] = to_dic_list(suffix, tokenizer)
    raw_suffix_tokens = to_dic_list(raw_suffix, tokenizer)
    start_raw_suffix_idx = (
        len(prompt["prompt_suffix_tokens"]) - len(raw_suffix_tokens) + 1
    )
    for i in range(
        len(prompt["prompt_suffix_tokens"]) - start_raw_suffix_idx,
        len(prompt["prompt_suffix_tokens"]) - 1,
    ):
        prompt["prompt_suffix_tokens"][i]["token_groups"] += ["template"]
    prompt["prompt_suffix_n_tokens"] = len(prompt["prompt_suffix_tokens"])

    steps = []

    for step_id, traj_step in enumerate(traj.steps):
        step_dic = {}
        step_dic["step_id"] = step_id
        step_dic["grid_state"] = traj_step.observation.split("\n")
        step_dic["grid_state_tokens"] = to_dic_list(
            traj_step.observation, tokenizer, groups=["prompt", "grid_state"]
        )
        step_dic["grid_state_n_tokens"] = len(step_dic["grid_state_tokens"])

        for i, t in enumerate(step_dic["grid_state_tokens"]):
            if any(sym in t["token"] for sym in grid_symbols):
                step_dic["grid_state_tokens"][i]["token_groups"] += ["grid_tile"]

        # Render the suffix with the actual carrying_key value for this step
        step_dic["carrying_key"] = traj_step.metadata.get("carrying_key", False)
        carrying_key_placeholder = "{{carrying_key}}"
        if carrying_key_placeholder in suffix:
            # Replace the placeholder with the actual value
            rendered_suffix = suffix.replace(
                carrying_key_placeholder, str(step_dic["carrying_key"])
            )
            step_dic["prompt_suffix_tokens"] = to_dic_list(rendered_suffix, tokenizer)
            # Mark template tokens appropriately
            raw_suffix_tokens = to_dic_list(
                raw_suffix.replace(
                    carrying_key_placeholder, str(step_dic["carrying_key"])
                ),
                tokenizer,
            )
            start_raw_suffix_idx = (
                len(step_dic["prompt_suffix_tokens"]) - len(raw_suffix_tokens) + 1
            )
            for i in range(
                len(step_dic["prompt_suffix_tokens"]) - start_raw_suffix_idx,
                len(step_dic["prompt_suffix_tokens"]) - 1,
            ):
                step_dic["prompt_suffix_tokens"][i]["token_groups"] += ["template"]
        else:
            step_dic["prompt_suffix_tokens"] = prompt["prompt_suffix_tokens"]
        step_dic["prompt_suffix_n_tokens"] = len(step_dic["prompt_suffix_tokens"])
        step_dic["agent_action"] = traj_step.metadata["action"]

        out_tokens = [t["token"] for t in traj_step.metadata["logprobs"]]
        step_dic["output_text"] = tokenizer.convert_tokens_to_string(out_tokens)
        step_dic["output_tokens"] = to_dic_list(
            step_dic["output_text"], tokenizer, groups=["output"]
        )
        step_dic["output_n_tokens"] = len(step_dic["output_tokens"])
        step_dic["output_tokens"] = annotate_output_tokens(
            model_name, step_dic["output_tokens"]
        )

        for i, t in enumerate(step_dic["output_tokens"]):
            if (
                "top_logprobs" not in traj_step.metadata["logprobs"][i]
                or "template" in t["token_groups"]
            ):
                continue
            curr_probs = {}
            for logprob_dic in traj_step.metadata["logprobs"][i]["top_logprobs"]:
                curr_probs[logprob_dic["token"]] = np.round(
                    np.exp(logprob_dic["logprob"]), 4
                )
            step_dic["output_tokens"][i]["probabilities"] = curr_probs

        steps.append(step_dic)

    out = {
        "grid_params": grid_params,
        "model_params": model_params,
        "prompt": prompt,
        "steps": steps,
    }
    with open(output_path, "w") as f:
        json.dump(out, f, cls=CompactJSONEncoder, ensure_ascii=False, indent=4)


def get_trajectories_key_door_env(
    rooms_per_side_options: list[int] = [2],
    add_door_key_options: list[bool] = [True],
    remove_door_from_env_options: list[bool] = [False],
    use_key_2path_env_options: list[bool] = [False],
    max_steps_per_trajectory: int = 30,
    max_tokens: int = 10000,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_logprobs: int = 5,
    seed: int = 42,
    reasoning_effort: Literal["low", "medium", "high"] = "low",
    model_names: list[str] = ["together_ai/openai/gpt-oss-20b"],
    template_name: str = "grid_full_observability_instrumental_goals.j2",
    observation_placeholders: list[str] = ["grid_state"],
    output_dir: str = ".",
    verbose: bool = False,
    num_examples: int = 1,
    max_workers: int | None = None,
    enable_rate_limit: bool = False,
    rate_limit: int = 1000,
    rate_limit_period: float = 300.0,
):
    """Generate multiple agent trajectories in rooms environments with key-door mechanics across parameter combinations in parallel.

    Creates trajectories for all combinations of rooms_per_side, add_door_key, use_key_2path_env, and model_names,
    running the specified number of examples per combination in parallel. Each trajectory is saved
    to a separate JSON file with a name based on the parameters.

    Args:
        rooms_per_side_options: List of rooms_per_side values to use (2 for 2x2=4 rooms, 3 for 3x3=9 rooms). Ignored if use_key_2path_env=True.
        add_door_key_options: List of booleans indicating whether to include locked door and key. Ignored if use_key_2path_env=True.
        remove_door_from_env_options: List of booleans indicating whether to remove the door after reset (keeps the key).
        use_key_2path_env_options: List of booleans indicating whether to use Key2PathMinigridEnv instead of RoomsMinigridEnv.
        max_steps_per_trajectory: Maximum number of steps to generate in each trajectory.
        max_tokens: Maximum tokens for model generation per step.
        temperature: Sampling temperature for the model (higher = more random).
        top_p: Nucleus sampling parameter (cumulative probability threshold).
        top_logprobs: Number of top log probabilities to return for each token.
        seed: Base random seed for reproducibility. Each example uses seed + example_id.
        reasoning_effort: Reasoning effort level for the model ("low", "medium", or "high").
        model_names: List of model names in format "provider/model_id".
        template_name: Name of the Jinja2 template file to use for prompts.
        observation_placeholders: List of placeholder names in the prompt template.
        output_dir: Directory to save the output JSON files.
        verbose: If True, print detailed logging during trajectory generation.
        num_examples: Number of different examples to generate per parameter combination.
        max_workers: Maximum number of parallel workers. If None, uses min(32, total_tasks).
        enable_rate_limit: If True, enforce rate limiting on API requests.
        rate_limit: Maximum number of requests allowed per rate_limit_period.
        rate_limit_period: Time period in seconds for rate limiting (default: 300 = 5 minutes).

    Returns:
        None. Results are saved to individual JSON files in output_dir with format:
        {model_sanitized}_rooms{rooms_per_side}_door{add_door_key}_{example_id}.json
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate all parameter combinations
    all_combinations = list(
        product(
            rooms_per_side_options,
            add_door_key_options,
            remove_door_from_env_options,
            use_key_2path_env_options,
            model_names,
            range(num_examples),
        )
    )

    total_tasks = len(all_combinations)
    logger.info(
        f"Generating {total_tasks} trajectories across {len(rooms_per_side_options)} rooms_per_side options, "
        f"{len(add_door_key_options)} door/key configs, {len(use_key_2path_env_options)} env type options, "
        f"{len(model_names)} models, with {num_examples} examples each."
    )

    # Create rate limiter if enabled
    rate_limiter: RateLimiter | None = None
    if enable_rate_limit:
        rate_limiter = RateLimiter(rate_limit=rate_limit, period=rate_limit_period)
        logger.info(
            f"Rate limiting enabled: {rate_limit} requests per {rate_limit_period} seconds "
            f"({rate_limit / rate_limit_period:.2f} requests/second)"
        )

    def _generate_single_task(params: tuple) -> dict:
        """Generate a single trajectory for given parameters."""
        # Acquire rate limit token if enabled
        if rate_limiter is not None:
            rate_limiter.acquire()

        (
            rooms_per_side,
            add_door_key,
            remove_door_from_env,
            use_key_2path_env,
            model_name,
            example_id,
        ) = params
        task_seed = seed + example_id

        # Sanitize model name for filename
        model_sanitized = model_name.replace("/", "_").replace(".", "_")

        if use_key_2path_env:
            env_str = "key2path"
        else:
            door_key_str = "doorkey" if add_door_key else "nodoor"
            env_str = f"rooms{rooms_per_side}_{door_key_str}"

        remove_door_str = "rmdoor" if remove_door_from_env else "keepdoor"
        output_filename = (
            f"{model_sanitized}_{env_str}_{remove_door_str}_{example_id}.json"
        )
        output_path = str(Path(output_dir) / output_filename)

        if verbose:
            logger.info(
                f"Starting task: model={model_name}, rooms_per_side={rooms_per_side}, "
                f"add_door_key={add_door_key}, remove_door_from_env={remove_door_from_env}, "
                f"use_key_2path_env={use_key_2path_env}, example={example_id}, seed={task_seed}"
            )

        try:
            get_trajectory_key_door_env(
                rooms_per_side=rooms_per_side,
                add_door_key=add_door_key,
                remove_door_from_env=remove_door_from_env,
                use_key_2path_env=use_key_2path_env,
                max_steps_per_trajectory=max_steps_per_trajectory,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_logprobs=top_logprobs,
                seed=task_seed,
                reasoning_effort=reasoning_effort,
                model_name=model_name,
                template_name=template_name,
                observation_placeholders=observation_placeholders,
                output_path=output_path,
                verbose=verbose,
            )
            return {"status": "success", "output_path": output_path, "params": params}
        except Exception as e:
            logger.error(f"Failed to generate trajectory for {params}: {e}")
            return {"status": "error", "error": str(e), "params": params}

    # Set default max_workers if not specified
    if max_workers is None:
        max_workers = min(32, total_tasks)

    # Adjust max_workers based on rate limit to avoid excessive idle workers
    if enable_rate_limit and rate_limiter is not None:
        # Calculate the sustainable number of workers based on request rate
        # If each task takes at least 1 second, don't exceed the rate limit per second
        sustainable_workers = min(max_workers, int(rate_limiter.tokens_per_second * 2))
        if sustainable_workers < max_workers:
            logger.info(
                f"Adjusting max_workers from {max_workers} to {sustainable_workers} "
                f"based on rate limit ({rate_limiter.tokens_per_second:.2f} req/sec)"
            )
            max_workers = sustainable_workers

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_single_task, combo): combo
            for combo in all_combinations
        }

        for future in as_completed(futures):
            combo = futures[future]
            try:
                result = future.result()
                results.append(result)
                if result["status"] == "success":
                    logger.info(f"Completed: {result['output_path']}")
                else:
                    logger.error(
                        f"Failed for {combo}: {result.get('error', 'Unknown error')}"
                    )
            except Exception as e:
                tqdm.write(f"Exception for {combo}: {e}")
                results.append({"status": "error", "error": str(e), "params": combo})

    success_count = sum(1 for r in results if r["status"] == "success")
    logger.info(f"Completed {success_count}/{total_tasks} trajectories successfully.")


def get_trajectories_multiple_per_grid(
    grid_sizes: list[int] = [5],
    grid_complexities: list[float] = [0.0],
    max_steps_per_trajectory: int = 50,
    max_tokens: int = 10000,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_logprobs: int = 5,
    seed: int = 42,
    reasoning_effort: Literal["low", "medium", "high"] = "low",
    model_names: list[str] = ["together_ai/openai/gpt-oss-20b"],
    observation_placeholders: list[str] = ["grid_state"],
    output_dir: str = ".",
    verbose: bool = False,
    enable_dynamic_max_steps: bool = False,
    num_trajectories_per_grid: int = 5,
    num_grids_per_config: int = 1,
    max_workers: int | None = None,
    max_workers_per_grid: int | None = None,
    enable_rate_limit: bool = False,
    rate_limit: int = 1000,
    rate_limit_period: float = 300.0,
    hf_repo_id: str | None = None,
    hf_path_prefix: str = "",
    hf_token: str | None = None,
    include_transforms: bool = False,
    transform_names: list[str] | None = None,
):
    """Generate multiple trajectories on the same grid layout for each configuration.

    Creates trajectories where multiple runs are performed on the same grid without
    regenerating it. This is useful for studying variability in agent behavior on
    identical environments.

    For each (grid_size, grid_complexity, model_name) combination:
    - Creates `num_grids_per_config` unique grid layouts
    - For each grid, generates `num_trajectories_per_grid` trajectories in parallel

    Note: Trajectories within the same grid use deepcopy of the environment to enable
    parallel execution while ensuring identical grid layouts. Different grids are also
    processed in parallel.

    Args:
        grid_sizes: List of grid sizes to use for trajectory generation.
        grid_complexities: List of grid complexity levels to use.
        max_steps_per_trajectory: Maximum number of steps to generate in each trajectory.
        max_tokens: Maximum tokens for model generation per step.
        temperature: Sampling temperature for the model (higher = more random).
        top_p: Nucleus sampling parameter (cumulative probability threshold).
        top_logprobs: Number of top log probabilities to return for each token.
        seed: Base random seed for reproducibility. Each grid uses seed + grid_id.
        reasoning_effort: Reasoning effort level for the model ("low", "medium", or "high").
        model_names: List of model names in format "provider/model_id".
        observation_placeholders: List of placeholder names in the prompt template.
        output_dir: Directory to save the output JSON files.
        verbose: If True, print detailed logging during trajectory generation.
        enable_dynamic_max_steps: If True, override max_steps_per_trajectory with
            a dynamic value based on 1.5x the A* optimal path length.
        num_trajectories_per_grid: Number of trajectories to generate per grid layout.
        num_grids_per_config: Number of different grid layouts per (size, complexity, model) combo.
        max_workers: Maximum number of parallel workers for processing different grids.
            If None, uses min(32, total_grid_tasks).
        max_workers_per_grid: Maximum number of parallel workers for trajectories within each grid.
            If None, uses num_trajectories_per_grid (full parallelism within each grid).
        enable_rate_limit: If True, enforce rate limiting on API requests.
        rate_limit: Maximum number of requests allowed per rate_limit_period.
        rate_limit_period: Time period in seconds for rate limiting (default: 300 = 5 minutes).
        hf_repo_id: Hugging Face repository ID to upload to (e.g., "username/repo-name").
            If None, no upload is performed.
        hf_path_prefix: Path prefix within the HF repo (e.g., "trajectories/" to put files
            in a subfolder).
        hf_token: Hugging Face API token. If None, uses HF_TOKEN env var or cached credentials.
        include_transforms: If True, also generate trajectories for transformed versions
            of each grid (RotateEnv, ReflectEnv, TransposeEnv, StartGoalSwap). Each
            transform gets num_trajectories_per_grid trajectories.
        transform_names: List of transform names to include when include_transforms=True.
            If None, uses all available transforms. Options: "RotateEnv", "ReflectEnv",
            "TransposeEnv", "StartGoalSwap".

    Returns:
        list[str] | None: List of URLs if uploaded to HF, otherwise None.
        Results are saved to individual JSON files in output_dir with format:
        {model_sanitized}_size{grid_size}_comp{grid_complexity}_grid{grid_id}_{transform}_traj{traj_id}.json
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate all grid configurations (each will have multiple trajectories)
    grid_configs = list(
        product(
            grid_sizes,
            grid_complexities,
            model_names,
            range(num_grids_per_config),
        )
    )

    total_grid_tasks = len(grid_configs)

    # Determine number of transforms (including base)
    if include_transforms:
        used_transform_names = (
            transform_names if transform_names else DEFAULT_TRANSFORM_NAMES
        )
        num_transforms = 1 + len(used_transform_names)  # base + transforms
    else:
        used_transform_names = []
        num_transforms = 1  # just base

    total_trajectories = total_grid_tasks * num_trajectories_per_grid * num_transforms
    transform_info = (
        f" with transforms ({', '.join(['base'] + used_transform_names)})"
        if include_transforms
        else ""
    )
    logger.info(
        f"Generating {total_trajectories} trajectories across {len(grid_sizes)} grid sizes, "
        f"{len(grid_complexities)} complexities, {len(model_names)} models, "
        f"with {num_grids_per_config} grids each and {num_trajectories_per_grid} trajectories per grid"
        f"{transform_info}."
    )

    # Create rate limiter if enabled
    rate_limiter: RateLimiter | None = None
    if enable_rate_limit:
        rate_limiter = RateLimiter(rate_limit=rate_limit, period=rate_limit_period)
        logger.info(
            f"Rate limiting enabled: {rate_limit} requests per {rate_limit_period} seconds "
            f"({rate_limit / rate_limit_period:.2f} requests/second)"
        )

    def _generate_trajectories_for_grid(config: tuple) -> dict:
        """Generate multiple trajectories on a single grid in parallel using deepcopy.

        When include_transforms is True, also generates trajectories for transformed
        versions of the grid (RotateEnv, ReflectEnv, TransposeEnv, StartGoalSwap).
        """
        grid_size, grid_complexity, model_name, grid_id = config
        grid_seed = seed + grid_id

        # Sanitize model name for filename
        model_sanitized = model_name.replace("/", "_").replace(".", "_")

        # Create the master environment once for this grid
        np.random.seed(grid_seed)
        master_env = FullObservabilityTextWrapper(
            Simple2DNavigationEnv(size=grid_size, complexity=grid_complexity)
        )
        master_env.reset()

        # Get all environments to generate trajectories for (base + transforms if enabled)
        if include_transforms:
            env_variants = get_transformed_environments(
                master_env,
                include_base=True,
                transform_names=used_transform_names if used_transform_names else None,
            )
        else:
            env_variants = [("base", master_env)]

        # Save grid layout for base environment
        grid_paths = []
        base_grid_path = _save_grid_layout(
            env=master_env,
            grid_size=grid_size,
            grid_complexity=grid_complexity,
            grid_id=grid_id,
            grid_seed=grid_seed,
            model_sanitized=model_sanitized,
            transform_type="base",
        )
        grid_paths.append(base_grid_path)

        # Save grid layouts for transformed environments
        if include_transforms:
            for transform_name, transformed_env in env_variants:
                if transform_name == "base":
                    continue  # Already saved
                transform_grid_path = _save_grid_layout(
                    env=transformed_env,
                    grid_size=grid_size,
                    grid_complexity=grid_complexity,
                    grid_id=grid_id,
                    grid_seed=grid_seed,
                    model_sanitized=model_sanitized,
                    transform_type=transform_name,
                )
                grid_paths.append(transform_grid_path)

        def _generate_single_trajectory(
            traj_id: int, transform_name: str, env_to_use: FullObservabilityTextWrapper
        ) -> dict:
            """Generate a single trajectory using a deepcopy of the given environment."""
            # Acquire rate limit token if enabled
            if rate_limiter is not None:
                rate_limiter.acquire()

            # Offset seeds for different trajectories and transforms
            transform_offset = hash(transform_name) % 10000
            traj_seed = grid_seed + traj_id * 1000 + transform_offset

            # Deep copy the environment to avoid race conditions
            env_copy = copy.deepcopy(env_to_use)

            output_filename = (
                f"{model_sanitized}_size{grid_size}_comp{grid_complexity}"
                f"_grid{grid_id}_{transform_name}_traj{traj_id}.json"
            )
            output_path = str(Path(output_dir) / output_filename)

            if verbose:
                logger.info(
                    f"Starting trajectory: model={model_name}, size={grid_size}, "
                    f"complexity={grid_complexity}, grid={grid_id}, "
                    f"transform={transform_name}, traj={traj_id}"
                )

            try:
                get_trajectory(
                    grid_size=grid_size,
                    grid_complexity=grid_complexity,
                    max_steps_per_trajectory=max_steps_per_trajectory,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_logprobs=top_logprobs,
                    seed=traj_seed,
                    reasoning_effort=reasoning_effort,
                    model_name=model_name,
                    observation_placeholders=observation_placeholders,
                    output_path=output_path,
                    verbose=verbose,
                    enable_dynamic_max_steps=enable_dynamic_max_steps,
                    env=env_copy,
                    use_safe_reset=True,
                    transform_type=transform_name,
                )
                return {
                    "status": "success",
                    "output_path": output_path,
                    "config": config,
                    "traj_id": traj_id,
                    "transform_type": transform_name,
                }
            except Exception as e:
                logger.error(
                    f"Failed to generate trajectory for grid {grid_id}, "
                    f"transform {transform_name}, traj {traj_id}: {e}"
                )
                return {
                    "status": "error",
                    "error": str(e),
                    "config": config,
                    "traj_id": traj_id,
                    "transform_type": transform_name,
                }

        # Parallelize trajectories within this grid (across all transforms)
        total_tasks_per_grid = num_trajectories_per_grid * len(env_variants)
        inner_workers = (
            max_workers_per_grid
            if max_workers_per_grid is not None
            else min(total_tasks_per_grid, num_trajectories_per_grid * 2)
        )
        trajectory_results = []

        with ThreadPoolExecutor(max_workers=inner_workers) as inner_executor:
            futures = {}
            for transform_name, env_variant in env_variants:
                for traj_id in range(num_trajectories_per_grid):
                    future = inner_executor.submit(
                        _generate_single_trajectory,
                        traj_id,
                        transform_name,
                        env_variant,
                    )
                    futures[future] = (traj_id, transform_name)

            for future in as_completed(futures):
                traj_id, transform_name = futures[future]
                try:
                    result = future.result()
                    trajectory_results.append(result)
                except Exception as e:
                    logger.error(
                        f"Exception for grid {grid_id}, transform {transform_name}, "
                        f"traj {traj_id}: {e}"
                    )
                    trajectory_results.append(
                        {
                            "status": "error",
                            "error": str(e),
                            "config": config,
                            "traj_id": traj_id,
                            "transform_type": transform_name,
                        }
                    )

        return {
            "trajectory_results": trajectory_results,
            "grid_paths": grid_paths,
        }

    def _save_grid_layout(
        env: FullObservabilityTextWrapper,
        grid_size: int,
        grid_complexity: float,
        grid_id: int,
        grid_seed: int,
        model_sanitized: str,
        transform_type: str = "base",
    ) -> str:
        """Save the grid layout to a JSON file."""
        unwrapped = env.unwrapped

        # Build grid representation as list of lists
        grid_list = []
        for j in range(unwrapped.height):
            row = []
            for i in range(unwrapped.width):
                cell = unwrapped.grid.get(i, j)
                if (i, j) == tuple(unwrapped.agent_pos):
                    row.append("A")
                elif cell is None:
                    row.append("_")
                elif cell.type == "wall":
                    row.append("#")
                elif cell.type == "goal":
                    row.append("G")
                else:
                    row.append("?")
            grid_list.append(row)

        # Get agent position - handle both array and tuple forms
        agent_pos = unwrapped.agent_pos
        if hasattr(agent_pos, "tolist"):
            agent_start_pos = agent_pos.tolist()
        else:
            agent_start_pos = list(agent_pos)

        # Get initial agent position if available
        if (
            hasattr(unwrapped, "_initial_agent_pos")
            and unwrapped._initial_agent_pos is not None
        ):
            initial_pos = unwrapped._initial_agent_pos
            if hasattr(initial_pos, "tolist"):
                agent_start_pos = initial_pos.tolist()
            else:
                agent_start_pos = list(initial_pos)

        # Get initial agent direction if available
        agent_start_dir = getattr(unwrapped, "_initial_agent_dir", unwrapped.agent_dir)

        grid_data = {
            "grid_id": grid_id,
            "grid_seed": grid_seed,
            "grid_size": grid_size,
            "grid_complexity": grid_complexity,
            "grid_width": unwrapped.width,
            "grid_height": unwrapped.height,
            "transform_type": transform_type,
            "agent_start_pos": agent_start_pos,
            "agent_start_dir": agent_start_dir,
            "goal_pos": list(unwrapped.goal_pos),
            "grid_layout": grid_list,
            "grid_text": env._render(),
            "legend": env.grid_cells,
        }

        grid_filename = (
            f"{model_sanitized}_size{grid_size}_comp{grid_complexity}"
            f"_grid{grid_id}_{transform_type}.json"
        )
        grid_path = str(Path(output_dir) / grid_filename)

        with open(grid_path, "w") as f:
            json.dump(grid_data, f, indent=2)

        if verbose:
            logger.info(f"Saved grid layout to {grid_path}")

        return grid_path

    # Set default max_workers if not specified
    if max_workers is None:
        max_workers = min(32, total_grid_tasks)

    # Adjust max_workers based on rate limit to avoid excessive idle workers
    if enable_rate_limit and rate_limiter is not None:
        sustainable_workers = min(max_workers, int(rate_limiter.tokens_per_second * 2))
        if sustainable_workers < max_workers:
            logger.info(
                f"Adjusting max_workers from {max_workers} to {sustainable_workers} "
                f"based on rate limit ({rate_limiter.tokens_per_second:.2f} req/sec)"
            )
            max_workers = sustainable_workers

    all_trajectory_results = []
    all_grid_paths = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_trajectories_for_grid, config): config
            for config in grid_configs
        }

        for future in tqdm(
            as_completed(futures),
            total=total_grid_tasks,
            desc="Processing grids",
            unit="grid",
        ):
            config = futures[future]
            try:
                result = future.result()
                grid_results = result["trajectory_results"]
                grid_paths = result["grid_paths"]

                all_trajectory_results.extend(grid_results)
                all_grid_paths.extend(grid_paths)

                # Report any failures for this grid
                failures = [r for r in grid_results if r["status"] != "success"]
                for failure in failures:
                    transform_info = failure.get("transform_type", "base")
                    tqdm.write(
                        f"Failed for grid {config}, transform {transform_info}, "
                        f"traj {failure['traj_id']}: {failure.get('error', 'Unknown error')}"
                    )
            except Exception as e:
                tqdm.write(f"Exception for grid {config}: {e}")
                # Mark all trajectories for this grid as failed
                transforms_to_fail = ["base"]
                if include_transforms:
                    transforms_to_fail.extend(
                        used_transform_names
                        if used_transform_names
                        else DEFAULT_TRANSFORM_NAMES
                    )
                for transform_name in transforms_to_fail:
                    for traj_id in range(num_trajectories_per_grid):
                        all_trajectory_results.append(
                            {
                                "status": "error",
                                "error": str(e),
                                "config": config,
                                "traj_id": traj_id,
                                "transform_type": transform_name,
                            }
                        )

    success_count = sum(1 for r in all_trajectory_results if r["status"] == "success")
    logger.info(
        f"Completed {success_count}/{total_trajectories} trajectories successfully."
    )
    logger.info(f"Saved {len(all_grid_paths)} grid layout files.")

    # Upload to Hugging Face if repo_id is provided
    if hf_repo_id is not None and (success_count > 0 or all_grid_paths):
        all_paths_to_upload = []

        # Add successful trajectory files
        successful_traj_paths = [
            r["output_path"] for r in all_trajectory_results if r["status"] == "success"
        ]
        all_paths_to_upload.extend(successful_traj_paths)

        # Add grid layout files
        all_paths_to_upload.extend(all_grid_paths)

        if all_paths_to_upload:
            return upload_files_to_huggingface(
                file_paths=all_paths_to_upload,
                repo_id=hf_repo_id,
                path_prefix=hf_path_prefix,
                hf_token=hf_token,
            )
    return None


def upload_trajectories_dir(
    directory: str,
    hf_repo_id: str,
    glob_pattern: str = "*.json",
    hf_token: str | None = None,
    num_workers: int = 8,
) -> str:
    """Upload a directory of trajectory and grid JSON files to Hugging Face.

    Uses upload_large_folder for reliable uploads of large directories with many
    files. This method is resumable (can be re-run if interrupted), uses multiple
    workers for parallel uploads, and automatically handles batching to avoid
    timeouts.

    This is useful for uploading trajectory and grid files generated by
    get_trajectories_multiple_per_grid after they have been generated.

    Note: Files are uploaded to the root of the repository. If you need files
    in a subdirectory, organize them locally in a parent folder first.

    Args:
        directory: Path to the directory containing JSON files to upload.
        hf_repo_id: Hugging Face repository ID (e.g., "username/repo-name").
        glob_pattern: Glob pattern to match files (default: "*.json").
            Use "*.json" to upload all JSON files, or a more specific pattern
            like "*_traj*.json" to upload only trajectory files.
        hf_token: Hugging Face API token. If None, uses HF_TOKEN env var
            or cached credentials.
        num_workers: Number of parallel workers for uploading (default: 8).
            Increase for faster uploads on fast connections.

    Returns:
        str: URL of the repository on Hugging Face Hub.

    Raises:
        ValueError: If directory doesn't exist or no matching files found.

    Example:
        # Upload all JSON files from output directory
        upload_trajectories_dir(
            directory="./trajectory_output",
            hf_repo_id="myuser/trajectories",
        )

        # Upload only trajectory files (excluding grid layout files)
        upload_trajectories_dir(
            directory="./trajectory_output",
            hf_repo_id="myuser/trajectories",
            glob_pattern="*_traj*.json",
        )

        # Use more workers for faster upload
        upload_trajectories_dir(
            directory="./trajectory_output",
            hf_repo_id="myuser/trajectories",
            num_workers=16,
        )
    """
    return upload_directory_to_huggingface(
        directory=directory,
        repo_id=hf_repo_id,
        glob_pattern=glob_pattern,
        hf_token=hf_token,
        repo_type="dataset",
        num_workers=num_workers,
    )
