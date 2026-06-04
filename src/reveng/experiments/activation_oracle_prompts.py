from __future__ import annotations

"""Central registry for activation-oracle prompt families used in DoorKey runs."""

ACTION_LABELS = ("UP", "DOWN", "LEFT", "RIGHT", "unknown")
WALL_LABELS = ("yes", "no", "unknown")

LOCAL_GRID_STATE_FAMILY = "local_grid_state"
MODEL_ACTION_FAMILY = "model_action"
OPTIMAL_ACTION_FAMILY = "optimal_action"

PROMPT_REGISTRY = {
    LOCAL_GRID_STATE_FAMILY: {
        "purpose": "Read out local grid-state information from activations.",
        "template": (
            "Given these activations, is the cell immediately {direction} of the agent a wall? "
            "Answer with exactly one of: yes, no, unknown."
        ),
        "labels": WALL_LABELS,
    },
    MODEL_ACTION_FAMILY: {
        "purpose": "Read out the action the model itself would most likely choose from activations.",
        "template": (
            "If the model had to answer now based only on these activations, which next action would it most likely output? "
            "Answer with exactly one of: UP, DOWN, LEFT, RIGHT, unknown."
        ),
        "labels": ACTION_LABELS,
    },
    OPTIMAL_ACTION_FAMILY: {
        "purpose": "Read out the action that is shortest-path optimal from activations.",
        "template": (
            "Given these activations and the current DoorKey state, which next action is shortest-path optimal? "
            "Answer with exactly one of: UP, DOWN, LEFT, RIGHT, unknown."
        ),
        "labels": ACTION_LABELS,
    },
}


def wall_state_oracle_prompt(question_id: str) -> str:
    direction = question_id.removeprefix("wall_").upper()
    return str(PROMPT_REGISTRY[LOCAL_GRID_STATE_FAMILY]["template"]).format(direction=direction)


def model_action_oracle_prompt() -> str:
    return str(PROMPT_REGISTRY[MODEL_ACTION_FAMILY]["template"])


def optimal_action_oracle_prompt() -> str:
    return str(PROMPT_REGISTRY[OPTIMAL_ACTION_FAMILY]["template"])


__all__ = [
    "ACTION_LABELS",
    "LOCAL_GRID_STATE_FAMILY",
    "MODEL_ACTION_FAMILY",
    "OPTIMAL_ACTION_FAMILY",
    "PROMPT_REGISTRY",
    "WALL_LABELS",
    "model_action_oracle_prompt",
    "optimal_action_oracle_prompt",
    "wall_state_oracle_prompt",
]
