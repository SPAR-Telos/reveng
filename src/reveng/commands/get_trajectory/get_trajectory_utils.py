import json
import traceback
from typing import Any
from copy import deepcopy
from tenacity import retry, stop_after_attempt, wait_random_exponential
from litellm import completion, completion_cost

from reveng.agents.alpha_start_agent import AlphaStarAgent
from reveng.agents.llm_agent import LLMAgent
from reveng.datatypes import Step, Trajectory
from reveng.trajectory_generator.trajectory_generator import generate_one_trajectory
from reveng.environment_generator.wrappers.text_obs_wrapper import FullObservabilityTextWrapper

@retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, min=5, max=120),
    reraise=True,
)
def completion_with_retry(**generation_kwargs) -> tuple[Any, float]:
    try:
        response = completion(**generation_kwargs)
    except Exception as e:
        print(f"Model request failed: {e}\n{traceback.format_exc()}")
        raise
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        print(f"Failed to calculate cost: {e}")
        cost = 0.0
    return response, cost


def parse_action(model_output: str) -> tuple[int, str]:
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
    except:
        return -1, "N/A"


def get_astar_distance(env, observation):
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


def to_dic_list(txt, tokenizer, groups = ["prompt"], start_idx = 0):
    tokens = tokenizer.tokenize(txt)
    ids = tokenizer.encode(txt)

    out = []
    for i, (t, id) in enumerate(zip(tokens, ids), start=start_idx):
        out.append({
            "id": i,
            "token": t,
            "token_id": id,
            "token_groups": groups
        })
    return out


def generate_trajectory(
    env: FullObservabilityTextWrapper,
    agent: LLMAgent,
    max_steps_per_trajectory: int,
    generation_kwargs: dict = {},
    metadata: dict = {},
    verbose: bool = False
):
    generation_kwargs["allowed_openai_params"] = list(generation_kwargs.keys())

    steps: list[Step] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    observation, _ = env.reset()

    traj_metadata = {}
    start_pos = tuple(int(x) for x in env.unwrapped.agent_pos)
    goal_pos = tuple(int(x) for x in env.unwrapped.goal_pos)
    traj_metadata["agent_start_coordinates"] = start_pos
    traj_metadata["goal_coordinates"] = goal_pos
    traj_metadata["astar_distance"] = get_astar_distance(env, observation)

    while not (terminated or truncated):
        if (
            max_steps_per_trajectory is not None
            and step_count >= max_steps_per_trajectory
        ):
            break
        
        unwrapped_env = getattr(env, "unwrapped", env)
        prompt = agent._generate_action_query_prompt(unwrapped_env)
        if verbose:
            print(f"Step {step_count}")
            print(agent._get_text_observation(env))
        full_output, cost = completion_with_retry(
            model=agent.model_name,
            messages=[{"role": "user", "content": prompt}],
            logprobs=True,
            **generation_kwargs
        )
        final_output = full_output.choices[0].message.content
        action, action_name = parse_action(final_output)
        logprobs_serialized = agent._finalize_cost_and_logprobs(cost, full_output, generation_kwargs.get("top_logprobs") is not None)

        if verbose:
            print("Output text:", final_output)
            print("Predicted action:", action)

        metadata = agent._build_base_metadata(action, cost, logprobs_serialized)
        metadata["action"] = action_name
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