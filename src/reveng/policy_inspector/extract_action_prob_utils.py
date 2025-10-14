import json
import pprint
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.special import softmax

from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv

# Define a constant for the action mapping for clarity and easy modification.
ACTION_MAP = {"0": "LEFT", "1": "RIGHT", "2": "UP", "3": "DOWN"}


def get_action_logprob_robust(
    logprobs: List[Dict[str, Any]], action_value: int
) -> Optional[Dict[str, Any]]:
    """
    Finds the logprob dictionary for a specific action token by searching
    for it within the context of the final JSON structure.
    """
    try:
        action_key_index = -1
        for i in range(len(logprobs) - 1, -1, -1):
            if logprobs[i]["token"] == "action":
                if i + 1 < len(logprobs) and '":' in logprobs[i + 1]["token"]:
                    action_key_index = i
                    break
        if action_key_index == -1:
            return None

        for i in range(action_key_index + 1, len(logprobs)):
            token = logprobs[i]["token"].strip()
            if token == str(action_value):
                return logprobs[i]
    except (ValueError, IndexError) as e:
        print(f"An error occurred during logprob search: {e}")
        return None
    return None


def calculate_normalized_distribution(
    action_logprob_info: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Calculates the normalized probability distribution over the defined actions
    from the 'top_logprobs' of a given token.
    """
    if not action_logprob_info or "top_logprobs" not in action_logprob_info:
        return {}

    top_logprobs = action_logprob_info["top_logprobs"]

    # Initialize log probabilities for all possible actions to negative infinity.
    # This ensures that any action token NOT in the top_logprobs list will
    # have a logprob of -inf, which becomes a probability of 0 after np.exp().
    log_probs = {token: -np.inf for token in ACTION_MAP.keys()}

    # Populate with actual logprobs from the model's top choices.
    for item in top_logprobs:
        token = item["token"]
        if token in ACTION_MAP:
            log_probs[token] = item["logprob"]

    normalized_distribution = softmax(np.array(list(log_probs.values())))
    mapped_distribution = {
        ACTION_MAP[token]: value
        for token, value in zip(log_probs.keys(), normalized_distribution)
    }
    return mapped_distribution


def get_action_probs(data: List[List[Any]]) -> List[List[Dict[str, float]]]:
    """
    Processes the entire 2D list of policy metadata.
    """
    all_distributions = []
    for i, row in enumerate(data):
        row_distributions = []
        for j, step_data in enumerate(row):
            # If the data for this step is -1, append an empty dict and skip.
            if step_data == -1:
                row_distributions.append({})
                continue

            logprobs_list = step_data.get("logprobs")
            chosen_action = step_data.get("llm_response")

            if logprobs_list is None or chosen_action is None:
                row_distributions.append({})
                continue

            action_logprob_info = get_action_logprob_robust(
                logprobs_list, chosen_action
            )
            distribution = calculate_normalized_distribution(action_logprob_info)
            row_distributions.append(distribution)
        all_distributions.append(row_distributions)
    return all_distributions


if __name__ == "__main__":
    file_path = "llm_policy_metadata.json"
    try:
        with open(file_path, "r") as file:
            policy_data = json.load(file)

        # Process the data to get the 2D list of distributions
        action_probabilities = get_action_probs(policy_data)

        print("✅ Successfully processed data. Resulting 2D list of distributions:")
        pprint.pprint(action_probabilities)

        env = Simple2DNavigationEnv(
            size=4,  # Small grid for easier visualization
            complexity=0.3,  # Some walls but not too complex
            render_mode=None,  # No rendering needed for policy elicitation
        )
        env.reset()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
