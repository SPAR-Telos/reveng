"""Generate multiple trajectories on a fixed key-door grid.

Creates a RoomsMinigridEnv once to fix the grid layout, then generates N trajectories
on that exact same grid using deepcopy. Each trajectory is saved as a separate JSON file
with the same structure as get_trajectory_key_door_env.

Usage:
    python scripts/get_trajectories_fixed_key_door_grid.py \\
        --model-name together_ai/openai/gpt-oss-20b \\
        --num-trajectories 20 \\
        --output-dir ./trajectories/fixed_grid/ \\
        --seed 42
"""

import argparse
import copy
import json
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import reveng
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from reveng.agents.llm_agent import LLMAgent
from reveng.commands.get_trajectory.compact_json_encoder import CompactJSONEncoder
from reveng.commands.get_trajectory.get_trajectory_utils import (
    annotate_output_tokens,
    generate_trajectory,
    to_dic_list,
)
from reveng.environment_generator.rooms_minigrid import RoomsMinigridEnv
from reveng.environment_generator.wrappers.text_obs_wrapper import FullObservabilityTextWrapper

logger = logging.getLogger(__name__)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multiple trajectories on a single fixed key-door grid."
    )
    parser.add_argument("--model-name", type=str, default="together_ai/openai/gpt-oss-20b")
    parser.add_argument("--rooms-per-side", type=int, default=2, choices=[2, 3])
    parser.add_argument("--no-door-key", dest="add_door_key", action="store_false", default=True)
    parser.add_argument("--num-trajectories", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reasoning-effort", type=str, default="low", choices=["low", "medium", "high"])
    parser.add_argument(
        "--template-name",
        type=str,
        default="grid_full_observability_instrumental_goals.j2",
    )
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_prompt_block(
    agent: LLMAgent,
    tokenizer: PreTrainedTokenizer,
    observation_placeholders: list[str],
) -> tuple[dict, str]:
    """Build the shared prompt metadata block and return (prompt_dict, suffix_str).

    The suffix_str still contains {{carrying_key}} as a literal — it is rendered
    per-step inside build_trajectory_json.
    """
    render_kwargs = {"grid_state": "{{grid_state}}", "carrying_key": "{{carrying_key}}"}
    template = agent._template.render(**render_kwargs)
    formatted_template: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": template}],
        tokenize=False,
        add_generation_prompt=True,
    )
    template_tokens = to_dic_list(formatted_template, tokenizer)

    observation_placeholder = "{{" + observation_placeholders[0] + "}}"
    prefix, suffix = formatted_template.split(observation_placeholder)
    raw_prefix, raw_suffix = template.split(observation_placeholder)

    prompt = {}
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
    start_raw_suffix_idx = len(prompt["prompt_suffix_tokens"]) - len(raw_suffix_tokens) + 1
    for i in range(
        len(prompt["prompt_suffix_tokens"]) - start_raw_suffix_idx,
        len(prompt["prompt_suffix_tokens"]) - 1,
    ):
        prompt["prompt_suffix_tokens"][i]["token_groups"] += ["template"]
    prompt["prompt_suffix_n_tokens"] = len(prompt["prompt_suffix_tokens"])

    return prompt, suffix, raw_suffix


def save_grid_layout(
    master_env: FullObservabilityTextWrapper,
    output_dir: str,
    rooms_per_side: int,
    add_door_key: bool,
    grid_seed: int,
) -> None:
    """Save the fixed grid layout to grid_layout.json."""
    unwrapped = master_env.unwrapped
    width, height = unwrapped.width, unwrapped.height

    grid_layout = []
    for row in range(height):
        grid_row = []
        for col in range(width):
            cell = unwrapped.grid.get(col, row)
            if (col, row) == tuple(unwrapped.agent_pos):
                grid_row.append("A")
            elif cell is None:
                grid_row.append("_")
            elif cell.type == "wall":
                grid_row.append("#")
            elif cell.type == "goal":
                grid_row.append("G")
            elif cell.type == "key":
                grid_row.append("K")
            elif cell.type == "door":
                grid_row.append("D" if cell.is_locked else "O")
            else:
                grid_row.append("?")
        grid_layout.append(grid_row)

    grid_data = {
        "grid_seed": grid_seed,
        "rooms_per_side": rooms_per_side,
        "add_door_key": add_door_key,
        "grid_width": width,
        "grid_height": height,
        "agent_start_pos": list(unwrapped.agent_pos),
        "agent_start_dir": int(unwrapped.agent_dir),
        "goal_pos": list(unwrapped.goal_pos),
        "grid_layout": grid_layout,
        "grid_text": master_env._render(),
        "legend": master_env.grid_cells,
    }

    out_path = Path(output_dir) / "grid_layout.json"
    with open(out_path, "w") as f:
        json.dump(grid_data, f, indent=2)
    logger.info(f"Saved grid layout to {out_path}")


def get_valid_start_positions(master_env: FullObservabilityTextWrapper) -> list[tuple[int, int]]:
    """Return all empty cells that are valid agent start positions.

    Excludes walls, goal, key, and door cells — only open floor tiles.
    """
    unwrapped = master_env.unwrapped
    valid = []
    for col in range(unwrapped.width):
        for row in range(unwrapped.height):
            cell = unwrapped.grid.get(col, row)
            if cell is None:  # empty floor tile (goal/key/door are non-None objects)
                valid.append((col, row))
    return valid


def build_trajectory_json(
    traj_id: int,
    master_env: FullObservabilityTextWrapper,
    model_name: str,
    template_path: Path,
    tokenizer: PreTrainedTokenizer,
    prompt_block: dict,
    suffix: str,
    raw_suffix: str,
    grid_params_base: dict,
    model_params_base: dict,
    output_path: str,
    traj_seed: int,
    start_pos: tuple[int, int],
    args: argparse.Namespace,
) -> dict:
    """Generate a single trajectory on a deepcopy of master_env and save to JSON."""
    # Fresh copy of the full grid state (key and door intact)
    env_copy = copy.deepcopy(master_env)

    # Set the per-trajectory start position (varies across trajectories for state coverage)
    env_copy.unwrapped.agent_pos = start_pos
    env_copy.unwrapped._initial_agent_pos = start_pos  # keep safe_reset consistent

    # Fresh agent per thread to avoid shared mutable cost state
    agent = LLMAgent(model_name=model_name, template_path=template_path)

    try:
        traj = generate_trajectory(
            env=env_copy,
            agent=agent,
            max_steps_per_trajectory=args.max_steps,
            generation_kwargs={
                "top_logprobs": args.top_logprobs,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "reasoning_effort": args.reasoning_effort,
                "seed": traj_seed,
            },
            verbose=args.verbose,
            use_safe_reset=True,  # prevents env.reset() in block 1 (no grid regen)
            skip_reset=True,       # prevents env.reset() in block 2
        )
    except Exception as e:
        logger.error(f"traj {traj_id} generation failed: {e}\n{traceback.format_exc()}")
        return {"status": "error", "traj_id": traj_id, "error": str(e)}

    grid_symbols = [cell["symbol"] for cell in env_copy.grid_cells.values()]

    # Build grid_params: shared base + per-trajectory fields
    grid_params = dict(grid_params_base)
    grid_params["traj_id"] = traj_id
    grid_params["astar_distance"] = traj.traj_metadata["astar_distance"]
    grid_params["agent_start_coordinates"] = traj.traj_metadata["agent_start_coordinates"]
    grid_params["goal_coordinates"] = traj.traj_metadata["goal_coordinates"]
    grid_params["legend"] = env_copy.grid_cells

    # Build model_params: shared base + per-trajectory seed
    model_params = dict(model_params_base)
    model_params["seed"] = traj_seed

    # Build steps
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

        # Render suffix with actual carrying_key value for this step
        step_dic["carrying_key"] = traj_step.metadata.get("carrying_key", False)
        carrying_key_placeholder = "{{carrying_key}}"
        if carrying_key_placeholder in suffix:
            rendered_suffix = suffix.replace(carrying_key_placeholder, str(step_dic["carrying_key"]))
            step_dic["prompt_suffix_tokens"] = to_dic_list(rendered_suffix, tokenizer)
            raw_suffix_rendered = to_dic_list(
                raw_suffix.replace(carrying_key_placeholder, str(step_dic["carrying_key"])),
                tokenizer,
            )
            start_raw_suffix_idx = (
                len(step_dic["prompt_suffix_tokens"]) - len(raw_suffix_rendered) + 1
            )
            for i in range(
                len(step_dic["prompt_suffix_tokens"]) - start_raw_suffix_idx,
                len(step_dic["prompt_suffix_tokens"]) - 1,
            ):
                step_dic["prompt_suffix_tokens"][i]["token_groups"] += ["template"]
        else:
            step_dic["prompt_suffix_tokens"] = prompt_block["prompt_suffix_tokens"]
        step_dic["prompt_suffix_n_tokens"] = len(step_dic["prompt_suffix_tokens"])

        step_dic["agent_action"] = traj_step.metadata["action"]

        api_logprobs = traj_step.metadata["logprobs"]
        out_tokens = [t["token"] for t in api_logprobs]
        step_dic["output_text"] = tokenizer.convert_tokens_to_string(out_tokens)
        step_dic["output_tokens"] = to_dic_list(
            step_dic["output_text"], tokenizer, groups=["output"]
        )
        step_dic["output_n_tokens"] = len(step_dic["output_tokens"])
        step_dic["output_tokens"] = annotate_output_tokens(model_name, step_dic["output_tokens"])

        for i, t in enumerate(step_dic["output_tokens"]):
            if i >= len(api_logprobs):
                break
            if "top_logprobs" not in api_logprobs[i] or "template" in t["token_groups"]:
                continue
            curr_probs = {}
            for logprob_dic in api_logprobs[i]["top_logprobs"]:
                curr_probs[logprob_dic["token"]] = np.round(np.exp(logprob_dic["logprob"]), 4)
            step_dic["output_tokens"][i]["probabilities"] = curr_probs

        steps.append(step_dic)

    out = {
        "grid_params": grid_params,
        "model_params": model_params,
        "prompt": prompt_block,
        "steps": steps,
    }
    with open(output_path, "w") as f:
        json.dump(out, f, cls=CompactJSONEncoder, ensure_ascii=False, indent=4)

    logger.info(f"Saved traj {traj_id} to {output_path}")
    return {"status": "success", "traj_id": traj_id, "output_path": output_path}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()

    # 1. Create the master env and reset ONCE to fix the grid
    np.random.seed(args.seed)
    master_env_unwrapped = RoomsMinigridEnv(
        add_door_key=args.add_door_key,
        max_steps=args.max_steps,
        rooms_per_side=args.rooms_per_side,
    )
    master_env = FullObservabilityTextWrapper(master_env_unwrapped)
    master_env.reset()
    # RoomsMinigridEnv overrides _gen_grid() without setting these, so safe_reset() would fail.
    # Set them manually here so deepcopies can use use_safe_reset=True.
    master_env_unwrapped._initial_agent_pos = master_env_unwrapped.agent_pos
    master_env_unwrapped._initial_agent_dir = master_env_unwrapped.agent_dir
    logger.info(
        f"Fixed grid created: rooms_per_side={args.rooms_per_side}, "
        f"add_door_key={args.add_door_key}, seed={args.seed}"
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 2. Save grid layout (must come before get_valid_start_positions call below)
    save_grid_layout(master_env, args.output_dir, args.rooms_per_side, args.add_door_key, args.seed)

    # 3. Build shared agent/tokenizer/prompt (once)
    model_id = "/".join(args.model_name.split("/")[1:])
    provider = args.model_name.split("/")[0]
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)
    template_path = Path(reveng.__file__).parent / "templates" / args.template_name
    agent_for_prompt = LLMAgent(model_name=args.model_name, template_path=template_path)
    prompt_block, suffix, raw_suffix = build_prompt_block(agent_for_prompt, tokenizer, ["grid_state"])

    # 4. Build shared param dicts (fields that don't vary across trajectories)
    grid_params_base = {
        "grid_width": master_env_unwrapped.width,
        "grid_height": master_env_unwrapped.height,
        "rooms_per_side": args.rooms_per_side,
        "add_door_key": args.add_door_key,
        "fully_observable": True,
    }
    model_params_base = {
        "model_id": model_id,
        "provider": provider,
        "interface": "litellm",
        "template_name": args.template_name,
        "n_interactions_in_context": 0,
        "max_tokens": args.max_tokens,
        "max_steps_per_trajectory": args.max_steps,
        "temperature": args.temperature,
        "reasoning_effort": args.reasoning_effort,
        "top_p": args.top_p,
        "top_logprobs": args.top_logprobs,
    }

    # 5. Sample a random start position per trajectory (fixed MDP: walls/key/door/goal unchanged)
    valid_starts = get_valid_start_positions(master_env)
    logger.info(f"Found {len(valid_starts)} valid start positions in the grid.")
    rng = np.random.RandomState(args.seed + 1)
    start_positions = [
        valid_starts[i] for i in rng.choice(len(valid_starts), size=args.num_trajectories, replace=True)
    ]

    # 6. Dispatch trajectories in parallel
    model_sanitized = args.model_name.replace("/", "_").replace(".", "_")
    door_str = "doorkey" if args.add_door_key else "nodoor"
    n_workers = args.max_workers or min(32, args.num_trajectories)

    logger.info(
        f"Generating {args.num_trajectories} trajectories with {n_workers} workers "
        f"(model={args.model_name})"
    )

    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for traj_id in range(args.num_trajectories):
            traj_seed = args.seed + traj_id * 1000
            out_name = f"{model_sanitized}_rooms{args.rooms_per_side}_{door_str}_grid0_traj{traj_id}.json"
            out_path = str(Path(args.output_dir) / out_name)
            future = executor.submit(
                build_trajectory_json,
                traj_id,
                master_env,
                args.model_name,
                template_path,
                tokenizer,
                prompt_block,
                suffix,
                raw_suffix,
                grid_params_base,
                model_params_base,
                out_path,
                traj_seed,
                start_positions[traj_id],
                args,
            )
            futures[future] = traj_id

        for future in tqdm(as_completed(futures), total=args.num_trajectories, desc="Trajectories"):
            traj_id = futures[future]
            try:
                result = future.result()
            except Exception as e:
                logger.error(f"traj {traj_id} raised unexpected exception: {e}")
                result = {"status": "error", "traj_id": traj_id, "error": str(e)}
            results.append(result)

    successes = sum(1 for r in results if r.get("status") == "success")
    print(f"\nDone: {successes}/{args.num_trajectories} trajectories succeeded.")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
