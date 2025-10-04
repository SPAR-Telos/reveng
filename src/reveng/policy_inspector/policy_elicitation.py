from typing import Any, List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from minigrid.minigrid_env import MiniGridEnv


def elicit_policy(env: MiniGridEnv, agent: Any) -> List[List[int]]:
    """
    Elicit the policy of an agent by querying its action preference at each valid position.

    Args:
        env: The MiniGrid environment in which the agent will operate.
        agent: The agent whose policy is to be elicited. Must have a `select_action` method.

    Returns:
        A 2D list representing the policy, where policy[j][i] is the preferred action
        at position (i, j). Returns -1 for positions that are walls or unreachable.
    """
    width, height = env.grid.width, env.grid.height
    policy = [[-1 for _ in range(width)] for _ in range(height)]

    # Save original agent position to restore later
    original_pos = (
        env.agent_pos.copy() if hasattr(env.agent_pos, "copy") else tuple(env.agent_pos)
    )

    for i in range(width):
        for j in range(height):
            cell = env.grid.get(i, j)
            # Only query policy for empty cells or cells that can be overlapped (like goal)
            if cell is None or (hasattr(cell, "can_overlap") and cell.can_overlap()):
                # Temporarily set agent position to this cell
                env.agent_pos = (i, j)
                # Query agent for preferred action at this position
                action, _ = agent.select_action(env)
                policy[j][i] = action

    # Restore original agent position
    env.agent_pos = original_pos

    return policy


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

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(width * 1.5, height * 1.5))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # Invert y-axis so (0,0) is top-left

    # Get goal position
    goal_pos = tuple(env.goal_pos) if hasattr(env, "goal_pos") else None

    # Define colors
    WALL_COLOR = "#808080"  # Grey for walls
    GOAL_COLOR = "#90EE90"  # Light green for goal
    VALID_COLOR = "#FFFFFF"  # White for valid positions
    ARROW_COLOR = "#000000"  # Black for arrows

    # Draw grid cells and arrows
    for j in range(height):
        for i in range(width):
            action = policy[j][i]

            # Determine cell color
            if (i, j) == goal_pos:
                color = GOAL_COLOR
            elif action == -1:
                color = WALL_COLOR
            else:
                color = VALID_COLOR

            # Draw cell background
            rect = patches.Rectangle(
                (i, j), 1, 1, linewidth=1, edgecolor="black", facecolor=color
            )
            ax.add_patch(rect)

            # Draw arrow for valid positions (except walls and goal)
            if action != -1 and (i, j) != goal_pos:
                cx, cy = i + 0.5, j + 0.5  # Center of cell
                arrow_length = 0.4  # Length from center to tip

                # Arrow direction based on action - draw from center outward in both directions
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
                    continue  # Unknown action

                arrow = FancyArrowPatch(
                    (start_x, start_y),
                    (end_x, end_y),
                    arrowstyle="-|>",
                    mutation_scale=30,
                    linewidth=3,
                    color=ARROW_COLOR,
                )
                ax.add_patch(arrow)

    # Add title
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    # Add grid lines
    ax.set_xticks(range(width + 1))
    ax.set_yticks(range(height + 1))
    ax.grid(True, alpha=0.3)

    # Add legend
    legend_elements = [
        patches.Patch(facecolor=GOAL_COLOR, edgecolor="black", label="Goal"),
        patches.Patch(facecolor=WALL_COLOR, edgecolor="black", label="Wall"),
        patches.Patch(facecolor=VALID_COLOR, edgecolor="black", label="Valid Position"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.05, 1))

    # Save figure
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Policy visualization saved to: {filename}")
