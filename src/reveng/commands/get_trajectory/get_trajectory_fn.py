"""Generate and save agent trajectories in navigation environments with detailed token-level analysis."""

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
    annotate_output_tokens,
    generate_trajectory,
    to_dic_list,
    upload_files_to_huggingface,
    upload_to_huggingface,
)
from reveng.commands.get_trajectory.rate_limiter import RateLimiter
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
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
    base_env = FullObservabilityTextWrapper(
        Simple2DNavigationEnv(size=grid_size, complexity=grid_complexity)
    )
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
    )

    grid_params = {}

    grid_params["grid_width"] = grid_size
    grid_params["grid_height"] = grid_size
    grid_params["grid_complexity"] = grid_complexity
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
