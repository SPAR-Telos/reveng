from typing import Any, Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from minigrid.minigrid_env import MiniGridEnv
from tqdm import tqdm


def elicit_policy(
    env: MiniGridEnv, agent: Any, top_logprobs: int = 5
) -> Tuple[List[List[int]], List[List[dict]]]:
    """
    Elicit the policy of an agent by querying its action preference at each valid position.

    Args:
        env: The MiniGrid environment in which the agent will operate.
        agent: The agent whose policy is to be elicited. Must have a `select_action` method.
        top_logprobs: Number of top logprobs to return for LLMAgent (default: 5).

    Returns:
        A 2D list representing the policy, where policy[j][i] is the preferred action
        at position (i, j). Returns -1 for positions that are walls or unreachable.
    """
    width, height = env.grid.width, env.grid.height
    policy = [[-1 for _ in range(width)] for _ in range(height)]
    policy_metadata = [[-1 for _ in range(width)] for _ in range(height)]

    # Save original agent position to restore later
    original_pos = (
        env.agent_pos.copy() if hasattr(env.agent_pos, "copy") else tuple(env.agent_pos)
    )

    total_cells = width * height
    with tqdm(total=total_cells, desc="Eliciting policy") as pbar:
        for i in range(width):
            for j in range(height):
                cell = env.grid.get(i, j)
                # Only query policy for empty cells or cells that can be overlapped (like goal)
                if cell is None or (
                    hasattr(cell, "can_overlap") and cell.can_overlap()
                ):
                    # Temporarily set agent position to this cell
                    env.agent_pos = (i, j)
                    # Query agent for preferred action at this position
                    if agent.__class__.__name__ == "LLMAgent":
                        action, metadata = agent.select_action(
                            env, return_logprobs=True, top_logprobs=top_logprobs
                        )
                    else:
                        action, metadata = agent.select_action(env)
                    policy_metadata[j][i] = metadata
                    policy[j][i] = action
                pbar.update(1)

    # Restore original agent position
    env.agent_pos = original_pos

    return policy, policy_metadata


def _create_figure_ax_pyplot(width: int, height: int) -> tuple[any, any]:
    """Create a pyplot-based figure/axes (not thread-safe)."""
    fig, ax = plt.subplots(figsize=(width * 1.5, height * 1.5))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    return fig, ax


def _create_figure_ax_agg(width: int, height: int) -> tuple[Figure, any]:
    """Create an Agg-based figure/axes (thread-safe)."""
    fig = Figure(figsize=(width * 1.5, height * 1.5))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    return fig, ax


def _draw_policy(ax, policy: List[List[int]], env: MiniGridEnv, title: str) -> None:
    height = len(policy)
    width = len(policy[0]) if height > 0 else 0

    goal_pos = tuple(env.goal_pos) if hasattr(env, "goal_pos") else None

    WALL_COLOR = "#808080"
    GOAL_COLOR = "#90EE90"
    VALID_COLOR = "#FFFFFF"
    INVALID_COLOR = "#FF0000"
    ARROW_COLOR = "#000000"

    for j in range(height):
        for i in range(width):
            action = policy[j][i]

            if (i, j) == goal_pos:
                color = GOAL_COLOR
            elif (
                action == -1
                and env.grid.get(i, j) is not None
                and env.grid.get(i, j).type == "wall"
            ):
                color = WALL_COLOR
            elif action != -1:
                color = VALID_COLOR
            else:
                color = INVALID_COLOR

            rect = patches.Rectangle(
                (i, j), 1, 1, linewidth=1, edgecolor="black", facecolor=color
            )
            ax.add_patch(rect)

            if action != -1 and (i, j) != goal_pos:
                cx, cy = i + 0.5, j + 0.5
                arrow_length = 0.4
                if action == 0:  # LEFT
                    start_x, start_y = cx + arrow_length / 2, cy
                    end_x, end_y = cx - arrow_length / 2, cy
                elif action == 1:  # RIGHT
                    start_x, start_y = cx - arrow_length / 2, cy
                    end_x, end_y = cx + arrow_length / 2, cy
                elif action == 2:  # UP
                    start_x, start_y = cx, cy + arrow_length / 2
                    end_x, end_y = cx, cy - arrow_length / 2
                elif action == 3:  # DOWN
                    start_x, start_y = cx, cy - arrow_length / 2
                    end_x, end_y = cx, cy + arrow_length / 2
                else:
                    start_x = start_y = end_x = end_y = None  # type: ignore

                if start_x is not None:
                    arrow = FancyArrowPatch(
                        (start_x, start_y),
                        (end_x, end_y),
                        arrowstyle="-|>",
                        mutation_scale=30,
                        linewidth=3,
                        color=ARROW_COLOR,
                    )
                    ax.add_patch(arrow)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(range(width + 1))
    ax.set_yticks(range(height + 1))
    ax.grid(True, alpha=0.3)

    legend_elements = [
        patches.Patch(facecolor=GOAL_COLOR, edgecolor="black", label="Goal"),
        patches.Patch(facecolor=WALL_COLOR, edgecolor="black", label="Wall"),
        patches.Patch(facecolor=VALID_COLOR, edgecolor="black", label="Valid Position"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.05, 1))


def _draw_policy_probs(
    ax, policy_probs: List[List[Dict[str, float]]], env: MiniGridEnv, title: str
) -> None:
    height = len(policy_probs)
    width = len(policy_probs[0]) if height > 0 else 0

    goal_pos = tuple(env.goal_pos) if hasattr(env, "goal_pos") else None
    WALL_COLOR = "#808080"
    GOAL_COLOR = "#90EE90"
    VALID_COLOR = "#FFFFFF"
    TEXT_COLOR = "#000000"

    for j in range(height):
        for i in range(width):
            prob_dict = policy_probs[j][i]
            if (i, j) == goal_pos:
                color = GOAL_COLOR
            elif not prob_dict:
                color = WALL_COLOR
            else:
                color = VALID_COLOR

            rect = patches.Rectangle(
                (i, j), 1, 1, linewidth=1, edgecolor="black", facecolor=color
            )
            ax.add_patch(rect)

            if prob_dict and (i, j) != goal_pos:
                positions = {
                    "UP": (i + 0.5, j + 0.25),
                    "DOWN": (i + 0.5, j + 0.75),
                    "LEFT": (i + 0.25, j + 0.5),
                    "RIGHT": (i + 0.75, j + 0.5),
                }
                for action_name, pos in positions.items():
                    prob = prob_dict.get(action_name, 0.0)
                    text = f"{prob:.2f}"
                    ax.text(
                        pos[0],
                        pos[1],
                        text,
                        color=TEXT_COLOR,
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(range(width + 1))
    ax.set_yticks(range(height + 1))
    ax.grid(True, which="both", color="k", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    legend_elements = [
        patches.Patch(facecolor=GOAL_COLOR, edgecolor="black", label="Goal"),
        patches.Patch(facecolor=WALL_COLOR, edgecolor="black", label="Wall"),
        patches.Patch(facecolor=VALID_COLOR, edgecolor="black", label="Valid State"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1))


def visualize_policy(
    policy: List[List[int]],
    env: MiniGridEnv,
    filename: str = "policy_visualization.png",
    title: str = "Policy",
) -> None:
    """
    Visualize a policy as a PNG image with arrows showing action choices.

    Args:
        policy: 2D list of actions where -1 indicates walls/unreachable positions
        env: The environment (used to identify goal position)
        filename: Output PNG filename
        title: Title for the visualization
    """
    height = len(policy)
    width = len(policy[0]) if height > 0 else 0
    fig, ax = _create_figure_ax_pyplot(width, height)
    _draw_policy(ax, policy, env, title)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Policy visualization saved to: {filename}")


def visualize_policy_threadsafe(
    policy: List[List[int]],
    env: MiniGridEnv,
    filename: str = "policy_visualization.png",
    title: str = "Policy",
) -> None:
    """Thread-safe version: uses Agg canvas and avoids pyplot."""
    height = len(policy)
    width = len(policy[0]) if height > 0 else 0
    fig, ax = _create_figure_ax_agg(width, height)
    _draw_policy(ax, policy, env, title)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Policy visualization saved to: {filename}")


def visualize_policy_probabilities(
    policy_probs: List[List[Dict[str, float]]],
    env: MiniGridEnv,
    filename: str = "policy_prob_visualization.png",
    title: str = "Policy Probability Distribution",
) -> None:
    """
    Visualizes a policy's probability distribution as a PNG image, with
    numerical probabilities displayed inside each grid cell.

    Args:
        policy_probs: 2D list of dictionaries, where each dict maps action
                      names ('UP', 'DOWN', 'LEFT', 'RIGHT') to probabilities.
                      An empty dictionary indicates a wall or unreachable state.
        env: The environment instance, used to identify the goal position.
        filename: The name of the output PNG file.
        title: The title for the visualization plot.
    """
    height = len(policy_probs)
    width = len(policy_probs[0]) if height > 0 else 0
    fig, ax = _create_figure_ax_pyplot(width, height)
    _draw_policy_probs(ax, policy_probs, env, title)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Policy visualization saved to: {filename}")


def visualize_policy_probabilities_threadsafe(
    policy_probs: List[List[Dict[str, float]]],
    env: MiniGridEnv,
    filename: str = "policy_prob_visualization.png",
    title: str = "Policy Probability Distribution",
) -> None:
    """Thread-safe version: uses Agg canvas and avoids pyplot."""
    height = len(policy_probs)
    width = len(policy_probs[0]) if height > 0 else 0
    fig, ax = _create_figure_ax_agg(width, height)
    _draw_policy_probs(ax, policy_probs, env, title)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Policy visualization saved to: {filename}")
