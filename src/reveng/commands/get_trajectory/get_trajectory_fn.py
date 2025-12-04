import os
import json
from transformers import AutoTokenizer
from reveng.agents.llm_agent import LLMAgent
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.environment_generator.wrappers.text_obs_wrapper import FullObservabilityTextWrapper

from reveng.commands.get_trajectory.get_trajectory_utils import generate_trajectory, to_dic_list
from reveng.commands.get_trajectory.compact_json_encoder import CompactJSONEncoder

def get_trajectory(
    grid_size = 5,
    grid_complexity = 0,
    max_steps_per_trajectory = 50,
    max_tokens = 10000,
    temperature = 0.7,
    top_p = 0.95,
    top_logprobs = 5,
    reasoning_effort = "low",
    model_name = "together_ai/openai/gpt-oss-20b",
    api_key = "58dc901e1334837bf0b0196032da3a7f222b5961813183570a502d1b130f7553",
    output_path = "output.json"
):
    base_env = FullObservabilityTextWrapper(
        Simple2DNavigationEnv(size=grid_size, complexity=grid_complexity)
    )
    agent = LLMAgent(model_name)
    model_id = "/".join(model_name.split("/")[1:])
    provider = model_name.split("/")[0]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    os.environ['TOGETHER_AI_API_KEY'] = api_key

    traj = generate_trajectory(
        env=base_env,
        agent=agent,
        max_steps_per_trajectory=max_steps_per_trajectory,
        generation_kwargs={
            "top_logprobs": top_logprobs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning_effort": reasoning_effort
        },
        verbose=True
    )

    grid_params = {}

    grid_params["grid_width"] = grid_size
    grid_params["grid_height"] = grid_size
    grid_params["grid_complexity"] = grid_complexity
    grid_params["fully_observable"] = True
    grid_params["astar_distance"] = traj.traj_metadata["astar_distance"]
    grid_params["agent_start_coordinates"] = traj.traj_metadata["agent_start_coordinates"]
    grid_params["goal_coordinates"] = traj.traj_metadata["goal_coordinates"]
    grid_params["legend"] = base_env.grid_cells

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

    prompt = {}

    template = agent._template.render(grid_state="{{grid_state}}")
    formatted_template: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": template}],
        tokenize=False,
        add_generation_prompt=True,
    )
    template_tokens = to_dic_list(formatted_template, tokenizer)
    prefix, suffix = formatted_template.split("{{grid_state}}")
    prompt["prompt_template"] = formatted_template
    prompt["prompt_template_n_tokens"] = len(template_tokens)
    prompt["prompt_prefix_tokens"] = to_dic_list(prefix, tokenizer)
    prompt["prompt_prefix_n_tokens"] = len(prompt["prompt_prefix_tokens"])
    prompt["prompt_placeholder_tokens"] = to_dic_list("{{grid_state}}", tokenizer)
    prompt["prompt_placeholder_n_tokens"] = len(prompt["prompt_placeholder_tokens"])
    prompt["prompt_suffix_tokens"] = to_dic_list(suffix, tokenizer)
    prompt["prompt_suffix_n_tokens"] = len(prompt["prompt_suffix_tokens"])

    out = {
        "grid_params": grid_params,
        "model_params": model_params,
        "prompt": prompt
    }
    with open(output_path, "w") as f:
        json.dump(out, f, cls=CompactJSONEncoder, ensure_ascii=False, indent=4)
