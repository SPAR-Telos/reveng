import json
from pathlib import Path

from tqdm import tqdm

from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.environment_generator.utils import compute_optimal_actions_from_position


# Global variable for JSON file path
JSON_FILE_PATH = "pre_reasoning/size7/together_ai_openai_gpt-oss-20b_size7_comp0.0_0.json"

# Mapping for special tokens to prompt_suffix_tokens indices
PROMPT_SUFFIX_TOKEN_MAPPING = {
    "<|end|>": 0,
    "<|start|>": 1,
    "assistant": 2
}
# The decoded grids are the same for all tokens
DEFAULT_PROMPT_SUFFIX_TOKEN = "<|end|>"


def load_steps_from_json(file_path: str = JSON_FILE_PATH) -> list:
    """
    Load a JSON file and return the list of steps.

    Args:
        file_path: Path to the JSON file (defaults to JSON_FILE_PATH)

    Returns:
        List of steps from the JSON file

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        KeyError: If 'steps' key is not found in the JSON
        json.JSONDecodeError: If the file is not valid JSON
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(path, 'r') as f:
        data = json.load(f)

    if 'steps' not in data:
        raise KeyError(f"'steps' key not found in JSON file: {file_path}")

    return data['steps']


def extract_grid_state_and_probes(steps: list, token_name: str = DEFAULT_PROMPT_SUFFIX_TOKEN) -> list:
    """
    Extract grid_state, probes, and agent_action from each step.

    Args:
        steps: List of step dictionaries
        token_name: Name of the prompt_suffix_token to use (default: "<|end|>")
                   Options: "<|end|>", "<|start|>", "assistant"

    Returns:
        List of dictionaries containing grid_state, probes, and agent_action for each step
    """
    token_index = PROMPT_SUFFIX_TOKEN_MAPPING.get(token_name, 0)
    print(f"Using token: {token_name} (index {token_index})")

    extracted_data = []

    for step in steps:
        prompt_suffix_tokens = step.get("prompt_suffix_tokens", [])
        probes = prompt_suffix_tokens[token_index].get("probes")

        step_data = {
            "grid_state": step.get("grid_state"),
            "probes": probes,
            "agent_action": step.get("agent_action")
        }
        extracted_data.append(step_data)

    return extracted_data

def probe2env(probes: dict, layer_key: str = "model.layers.15.output", prob_threshold: float = 0.1, k: int = 5) -> list[Simple2DNavigationEnv]:
    """
    Convert probe data to Simple2DNavigationEnv environments using cells classified as agent and goal.
    Returns all combinations of top-k classified agent and goal positions.

    Args:
        probes: Dictionary of probe data with keys like
                "cognitive_map_probe_l15_s0_suffix_-3--1_mlp_1024_full_upsample_normalize_r{row}_c{col}"
        layer_key: The layer key to extract predictions from (default: "model.layers.15.output")
        prob_threshold: Minimum probability threshold to consider a cell as agent or goal (default: 0.1)
        k: Number of top agent and goal positions to consider (default: 5)

    Returns:
        list[Simple2DNavigationEnv]: List of environments for all combinations of top-k agent and goal positions

    Raises:
        ValueError: If no probes are found or grid dimensions cannot be determined
    """
    # Parse probe keys to extract grid positions and predictions
    grid_data = {}
    max_row = -1
    max_col = -1

    for probe_key, probe_value in probes.items():
        # Extract row and column from probe key
        # Format: "cognitive_map_probe_l15_s0_suffix_-3--1_mlp_1024_full_upsample_normalize_r{row}_c{col}"
        if "_r" in probe_key and "_c" in probe_key:
            parts = probe_key.split("_")
            row_idx = None
            col_idx = None

            for i, part in enumerate(parts):
                if part.startswith("r") and part[1:].lstrip("-").isdigit():
                    row_idx = int(part[1:])
                elif part.startswith("c") and part[1:].lstrip("-").isdigit():
                    col_idx = int(part[1:])

            if row_idx is not None and col_idx is not None:
                max_row = max(max_row, row_idx)
                max_col = max(max_col, col_idx)

                # Get predictions from the specified layer
                predictions = probe_value.get(layer_key, {})
                grid_data[(row_idx, col_idx)] = predictions

    if not grid_data:
        raise ValueError("No valid probe data found")

    # Create grid dimensions (add 1 because indices are 0-based)
    height = max_row + 1
    width = max_col + 1

    # Find cells that are classified as agent or goal (highest probability class)
    agent_positions = []
    goal_positions = []

    for (row, col), predictions in grid_data.items():
        # Find the class with highest probability
        if predictions:
            max_class = max(predictions.items(), key=lambda x: x[1])
            class_name, class_prob = max_class

            if class_name == 'agent':
                agent_positions.append((row, col, class_prob))
            elif class_name == 'goal':
                goal_positions.append((row, col, class_prob))

    # Sort by probability (descending) and take top k
    agent_positions.sort(key=lambda x: x[2], reverse=True)
    goal_positions.sort(key=lambda x: x[2], reverse=True)

    potential_agents = agent_positions[:k]
    potential_goals = goal_positions[:k]

    print(f"Classified agent positions ({len(agent_positions)} total, using top {k}):")
    for idx, (row, col, prob) in enumerate(potential_agents, 1):
        print(f"  {idx}. Position ({row}, {col}): {prob:.4f}")

    print(f"\nClassified goal positions ({len(goal_positions)} total, using top {k}):")
    for idx, (row, col, prob) in enumerate(potential_goals, 1):
        print(f"  {idx}. Position ({row}, {col}): {prob:.4f}")

    # Generate all combinations of top-k agent and goal positions
    environments = []

    for agent_row, agent_col, agent_prob in potential_agents:
        for goal_row, goal_col, goal_prob in potential_goals:
            # Skip if agent and goal are at the same position
            if agent_row == goal_row and agent_col == goal_col:
                continue

            # Build grid using the classified positions
            grid_list = []

            for row in range(height):
                grid_row = []
                for col in range(width):
                    predictions = grid_data.get((row, col), {})

                    # Handle agent position
                    if row == agent_row and col == agent_col:
                        cell_symbol = 'A'  # Agent position
                        grid_row.append(cell_symbol)

                    # Handle goal position
                    elif row == goal_row and col == goal_col:
                        cell_symbol = 'G'  # Goal position
                        grid_row.append(cell_symbol)

                    else:
                        # For other cells, use highest probability class
                        cell_mapping = {
                            'wall': '#',
                            'empty': '_',
                            'padding': '#',
                            'agent': '_',  # If agent is highest but not THE agent, treat as empty
                            'goal': '_'     # If goal is highest but not THE goal, treat as empty
                        }

                        if predictions:
                            max_class = max(predictions.items(), key=lambda x: x[1])
                            class_name, class_prob = max_class
                            cell_symbol = cell_mapping.get(class_name, '_')
                        else:
                            cell_symbol = '_'

                        grid_row.append(cell_symbol)

                grid_list.append(grid_row)

            # Create environment
            env = Simple2DNavigationEnv(size=max(width, height))
            env.reset()
            env.set_env_from_list(grid_list)

            # Store metadata about probabilities
            env.agent_prob = agent_prob
            env.goal_prob = goal_prob
            env.combination_prob = agent_prob * goal_prob

            environments.append(env)

    print(f"\nGenerated {len(environments)} environment combinations")
    return environments

def gridstate2env(grid_state: list[str]) -> Simple2DNavigationEnv:
    """
    Convert a grid_state representation to a Simple2DNavigationEnv.

    Args:
        grid_state: List of strings representing the grid, where the first row
                   contains column indices and the first column contains row indices.
                   Format example:
                   [
                       "  0 1 2 3 4 5 6 ",
                       "0 # # # # # # # ",
                       "1 # _ _ _ _ _ # ",
                       "2 # A _ _ _ _ # ",
                       ...
                   ]

    Returns:
        Simple2DNavigationEnv: The environment created from the grid_state

    Raises:
        ValueError: If the grid_state format is invalid or empty
    """
    if not grid_state or len(grid_state) < 2:
        raise ValueError("grid_state must have at least 2 rows (header + data)")

    # Skip the first row (column indices) and parse the data rows
    grid_list = []

    for i, row_str in enumerate(grid_state[1:]):  # Skip header row
        # Split the row into parts and remove the row index (first element)
        parts = row_str.strip().split()

        if len(parts) < 2:
            continue  # Skip empty or malformed rows

        # First part is the row index, rest are cell values
        row_cells = parts[1:]
        grid_list.append(row_cells)

    if not grid_list:
        raise ValueError("No valid grid data found in grid_state")

    # Determine grid size
    grid_height = len(grid_list)
    grid_width = max(len(row) for row in grid_list) if grid_list else 0
    size = max(grid_height, grid_width)

    # Create and configure the environment
    env = Simple2DNavigationEnv(size=size)
    env.reset()
    env.set_env_from_list(grid_list)

    return env


def plot_and_save_environments(envs: list[Simple2DNavigationEnv], output_dir: str = "decoded_grid_environments"):
    """
    Plot and save environment visualizations.

    Args:
        envs: List of Simple2DNavigationEnv environments to visualize
        output_dir: Directory to save the plots (default: "decoded_grid_environments")
    """
    import matplotlib.pyplot as plt

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\nSaving {len(envs)} environment visualizations...")

    for idx, env in enumerate(envs):
        print(f"\nEnvironment {idx + 1}/{len(envs)}:")
        print(f"  Agent probability: {env.agent_prob:.4f}")
        print(f"  Goal probability: {env.goal_prob:.4f}")
        print(f"  Combination probability: {env.combination_prob:.4f}")

        # Get the frame
        frame = env.get_frame()

        # Save the frame
        if frame is not None:
            plt.figure(figsize=(8, 8))
            plt.imshow(frame)
            plt.title(f"Environment {idx + 1}: Agent prob={env.agent_prob:.3f}, Goal prob={env.goal_prob:.3f}")
            plt.axis('off')
            plt.tight_layout()

            # Save to file
            filename = f"env_{idx + 1:02d}_agent{env.agent_prob:.3f}_goal{env.goal_prob:.3f}.png"
            save_path = output_path / filename
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved to: {save_path}")

    print(f"\nAll {len(envs)} environments saved to: {output_path}")


def main():
    """Main function to load and process steps from JSON file."""
    # Create a mapping from action names to action enum values
    action_name_to_enum = {
        "LEFT": Simple2DNavigationEnv.Actions.LEFT,
        "RIGHT": Simple2DNavigationEnv.Actions.RIGHT,
        "UP": Simple2DNavigationEnv.Actions.UP,
        "DOWN": Simple2DNavigationEnv.Actions.DOWN,
    }

    try:
        steps = load_steps_from_json()
        print(f"Loaded {len(steps)} steps from {JSON_FILE_PATH}")

        # Extract grid_state and probes
        extracted_steps = extract_grid_state_and_probes(steps)
        print(f"\nExtracted grid_state and probes from {len(extracted_steps)} steps")


        for step in extracted_steps:
            # Create environment from grid_state (ground truth)
            env = gridstate2env(step["grid_state"])

            # Get decoded environments from probes
            decoded_envs = probe2env(step["probes"], k=5)

            # Get the action taken by the agent (convert string to enum)
            taken_action_str = step["agent_action"]
            taken_action = action_name_to_enum.get(taken_action_str)

            if taken_action is None:
                print(f"Warning: Unknown action '{taken_action_str}', skipping step")
                continue

            # Compute optimal actions for the ground truth environment
            agent_pos = tuple(env.agent_pos)
            optimal_actions_gt = compute_optimal_actions_from_position(env, agent_pos)

            # Compute optimal actions for each decoded environment
            for decoded_env in decoded_envs:
                decoded_agent_pos = tuple(decoded_env.agent_pos)
                optimal_actions_decoded = compute_optimal_actions_from_position(
                    decoded_env, decoded_agent_pos
                )

            # Plot and save environments
            # plot_and_save_environments(decoded_envs)

    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

