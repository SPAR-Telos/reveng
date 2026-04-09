"""Question registry for behavioral probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

ACTIONS = ("RIGHT", "LEFT", "UP", "DOWN")
DIRECTIONS = ("right", "left", "up", "down")
DIRECTION_TO_ACTION = {
    "right": "RIGHT",
    "left": "LEFT",
    "up": "UP",
    "down": "DOWN",
}
DIRECTION_TO_OBJECT_NAME = {
    "goal": "goal",
    "key": "key",
    "door": "door",
}


@dataclass(frozen=True)
class BehavioralProbeQuestion:
    """Single-turn black-box behavioral probe question."""

    question_id: str
    target_variable: str
    mode: str
    prompt_text: str
    show_carrying_key: bool
    consistency_applies: bool
    answer_space: str = "label3"
    family: str = "base"
    variant_id: str | None = None
    action: str | None = None
    subject: str | None = None


DOOR_OPEN_AFTER_ACTION_VARIANTS: dict[str, str] = {
    "observed": (
        "Given the current grid only, if you move {action} now, will the grid still show an "
        "open door (O) after that move? Answer with exactly one of: yes, no, unknown."
    ),
    "global": (
        "Given the current grid only, if you move {action} now, will there be any open door "
        "anywhere in the grid after that move? Answer with exactly one of: yes, no, unknown."
    ),
    "relevant": (
        "Given the current grid only, if you move {action} now, will the door relevant to your "
        "move or immediate interaction be open after that move? Answer with exactly one of: yes, "
        "no, unknown."
    ),
}

PROMPT_FAMILIES = {
    "core": {"has_key", "door_open"},
    "wall_directional": {f"wall_{direction}" for direction in DIRECTIONS},
    "action_effects": {
        *(f"hit_wall_after_{action.lower()}" for action in ACTIONS),
        *(f"has_key_after_{action.lower()}" for action in ACTIONS),
        *(f"door_open_after_{action.lower()}" for action in ACTIONS),
    },
    "object_directional": {
        *(f"is_goal_{direction}" for direction in DIRECTIONS),
        *(f"is_key_{direction}" for direction in DIRECTIONS),
        *(f"is_door_{direction}" for direction in DIRECTIONS),
    },
    "coordinates": {
        "agent_location",
        "goal_location",
        "key_location",
        "door_location",
        *(f"agent_location_after_{action.lower()}" for action in ACTIONS),
    },
}
PROMPT_FAMILIES["all_label3"] = (
    PROMPT_FAMILIES["core"]
    | PROMPT_FAMILIES["wall_directional"]
    | PROMPT_FAMILIES["action_effects"]
    | PROMPT_FAMILIES["object_directional"]
)
PROMPT_FAMILIES["all"] = PROMPT_FAMILIES["all_label3"] | PROMPT_FAMILIES["coordinates"]


def _core_questions() -> list[BehavioralProbeQuestion]:
    return [
        BehavioralProbeQuestion(
            question_id="has_key",
            target_variable="has_key",
            mode="current_state",
            prompt_text=(
                "Given the current grid only, do you currently have the key? "
                "Answer with exactly one of: yes, no, unknown."
            ),
            show_carrying_key=True,
            consistency_applies=False,
            family="core",
        ),
        BehavioralProbeQuestion(
            question_id="door_open",
            target_variable="door_open",
            mode="current_state",
            prompt_text=(
                "Given the current grid only, is any door currently open? "
                "Answer with exactly one of: yes, no, unknown."
            ),
            show_carrying_key=False,
            consistency_applies=False,
            family="core",
        ),
    ]


def _directional_wall_questions() -> list[BehavioralProbeQuestion]:
    questions: list[BehavioralProbeQuestion] = []
    for direction in DIRECTIONS:
        questions.append(
            BehavioralProbeQuestion(
                question_id=f"wall_{direction}",
                target_variable=f"wall_{direction}",
                mode="current_state",
                prompt_text=(
                    f"Given the current grid only, is there a wall immediately to your {direction.upper()}? "
                    "Answer with exactly one of: yes, no, unknown."
                ),
                show_carrying_key=False,
                consistency_applies=True,
                family="wall_directional",
                action=DIRECTION_TO_ACTION[direction],
            )
        )
    return questions


def _action_effect_questions(
    *,
    door_variant: str = "observed",
) -> list[BehavioralProbeQuestion]:
    if door_variant not in DOOR_OPEN_AFTER_ACTION_VARIANTS:
        raise ValueError(f"Unknown door-open-after-action variant: {door_variant}")
    questions: list[BehavioralProbeQuestion] = []
    for action in ACTIONS:
        action_lower = action.lower()
        questions.extend(
            [
                BehavioralProbeQuestion(
                    question_id=f"hit_wall_after_{action_lower}",
                    target_variable=f"hit_wall_after_{action_lower}",
                    mode="action_conditioned",
                    prompt_text=(
                        f"Given the current grid only, if you move {action} now, will you hit a wall? "
                        "Answer with exactly one of: yes, no, unknown."
                    ),
                    show_carrying_key=False,
                    consistency_applies=True,
                    family="action_effects",
                    action=action,
                ),
                BehavioralProbeQuestion(
                    question_id=f"has_key_after_{action_lower}",
                    target_variable=f"has_key_after_{action_lower}",
                    mode="action_conditioned",
                    prompt_text=(
                        f"Given the current grid only, if you move {action} now, will you have the key "
                        "after that move? Answer with exactly one of: yes, no, unknown."
                    ),
                    show_carrying_key=True,
                    consistency_applies=False,
                    family="action_effects",
                    action=action,
                ),
                BehavioralProbeQuestion(
                    question_id=f"door_open_after_{action_lower}",
                    target_variable=f"door_open_after_{action_lower}",
                    mode="action_conditioned",
                    prompt_text=DOOR_OPEN_AFTER_ACTION_VARIANTS[door_variant].format(action=action),
                    show_carrying_key=True,
                    consistency_applies=False,
                    family="action_effects",
                    variant_id=door_variant,
                    action=action,
                ),
            ]
        )
    return questions


def _object_direction_questions() -> list[BehavioralProbeQuestion]:
    questions: list[BehavioralProbeQuestion] = []
    for subject, object_name in DIRECTION_TO_OBJECT_NAME.items():
        for direction in DIRECTIONS:
            questions.append(
                BehavioralProbeQuestion(
                    question_id=f"is_{subject}_{direction}",
                    target_variable=f"is_{subject}_{direction}",
                    mode="current_state",
                    prompt_text=(
                        f"Given the current grid only, is the {object_name} immediately to your {direction.upper()}? "
                        "Answer with exactly one of: yes, no, unknown."
                    ),
                    show_carrying_key=(subject == "key"),
                    consistency_applies=False,
                    family="object_directional",
                    action=DIRECTION_TO_ACTION[direction],
                    subject=subject,
                )
            )
    return questions


def _coordinate_questions() -> list[BehavioralProbeQuestion]:
    questions = [
        BehavioralProbeQuestion(
            question_id="agent_location",
            target_variable="agent_location",
            mode="current_state",
            prompt_text=(
                "Given the current grid only, what is the agent's current location? "
                'Return exactly a JSON object of the form {"row": <int>, "col": <int>}. '
                "If the target is not visible, use -1 for both row and col."
            ),
            show_carrying_key=False,
            consistency_applies=False,
            answer_space="coord_json",
            family="coordinates",
            subject="agent",
        ),
        BehavioralProbeQuestion(
            question_id="goal_location",
            target_variable="goal_location",
            mode="current_state",
            prompt_text=(
                "Given the current grid only, what is the goal's location? "
                'Return exactly a JSON object of the form {"row": <int>, "col": <int>}. '
                "If the target is not visible, use -1 for both row and col."
            ),
            show_carrying_key=False,
            consistency_applies=False,
            answer_space="coord_json",
            family="coordinates",
            subject="goal",
        ),
        BehavioralProbeQuestion(
            question_id="key_location",
            target_variable="key_location",
            mode="current_state",
            prompt_text=(
                "Given the current grid only, what is the key's location? "
                'Return exactly a JSON object of the form {"row": <int>, "col": <int>}. '
                "If the key is not visible, use -1 for both row and col."
            ),
            show_carrying_key=True,
            consistency_applies=False,
            answer_space="coord_json",
            family="coordinates",
            subject="key",
        ),
        BehavioralProbeQuestion(
            question_id="door_location",
            target_variable="door_location",
            mode="current_state",
            prompt_text=(
                "Given the current grid only, what is a door's location? "
                'Return exactly a JSON object of the form {"row": <int>, "col": <int>}. '
                "If no door is visible, use -1 for both row and col."
            ),
            show_carrying_key=False,
            consistency_applies=False,
            answer_space="coord_json",
            family="coordinates",
            subject="door",
        ),
    ]
    for action in ACTIONS:
        action_lower = action.lower()
        questions.append(
            BehavioralProbeQuestion(
                question_id=f"agent_location_after_{action_lower}",
                target_variable=f"agent_location_after_{action_lower}",
                mode="action_conditioned",
                prompt_text=(
                    f"Given the current grid only, if you move {action} now, what will the agent's location be "
                    "after that move? Return exactly a JSON object of the form {\"row\": <int>, \"col\": <int>}."
                ),
                show_carrying_key=True,
                consistency_applies=False,
                answer_space="coord_json",
                family="coordinates",
                action=action,
                subject="agent",
            )
        )
    return questions


def get_behavioral_probe_questions(
    *,
    question_family: str = "all_label3",
    answer_space: str | None = None,
    door_open_after_action_variant: str = "observed",
) -> list[BehavioralProbeQuestion]:
    questions = (
        _core_questions()
        + _directional_wall_questions()
        + _action_effect_questions(door_variant=door_open_after_action_variant)
        + _object_direction_questions()
        + _coordinate_questions()
    )
    allowed_ids = PROMPT_FAMILIES.get(question_family)
    if allowed_ids is None:
        raise ValueError(f"Unknown question family: {question_family}")
    filtered = [question for question in questions if question.question_id in allowed_ids]
    if answer_space is not None:
        filtered = [question for question in filtered if question.answer_space == answer_space]
    return filtered


BEHAVIORAL_PROBE_QUESTIONS = get_behavioral_probe_questions(question_family="all_label3")
BEHAVIORAL_PROBE_QUESTION_BY_ID = {
    question.question_id: question for question in BEHAVIORAL_PROBE_QUESTIONS
}
QUESTION_IDS = [question.question_id for question in BEHAVIORAL_PROBE_QUESTIONS]


__all__ = [
    "ACTIONS",
    "BEHAVIORAL_PROBE_QUESTIONS",
    "BEHAVIORAL_PROBE_QUESTION_BY_ID",
    "DIRECTIONS",
    "DOOR_OPEN_AFTER_ACTION_VARIANTS",
    "PROMPT_FAMILIES",
    "QUESTION_IDS",
    "BehavioralProbeQuestion",
    "get_behavioral_probe_questions",
]
