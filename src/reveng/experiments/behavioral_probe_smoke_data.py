"""Manual smoke-test states and state-derived labels for behavioral probes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from minigrid.core.world_object import Key

from reveng.experiments.behavioral_probe_parse import COORD_MISSING
from reveng.experiments.behavioral_probe_questions import ACTIONS, DIRECTIONS
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.environment_generator.rooms_minigrid import RoomsMinigridEnv


@dataclass(frozen=True)
class BehavioralProbeSmokeExample:
    example_id: str
    grid_text: str
    carrying_key: bool
    has_key_label: str
    door_open_label: str
    wall_right_label: str
    hit_wall_after_right_label: str
    has_key_after_right_label: str
    door_open_after_right_label: str
    observed_action: str | None = None
    notes: str | None = None


LABEL_FIELDS = [
    "has_key_label",
    "door_open_label",
    "wall_right_label",
    "hit_wall_after_right_label",
    "has_key_after_right_label",
    "door_open_after_right_label",
]
ACTION_TO_DELTA = {
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "UP": (0, -1),
    "DOWN": (0, 1),
}
ACTION_TO_ENV_ACTION = {
    "LEFT": Simple2DNavigationEnv.Actions.LEFT,
    "RIGHT": Simple2DNavigationEnv.Actions.RIGHT,
    "UP": Simple2DNavigationEnv.Actions.UP,
    "DOWN": Simple2DNavigationEnv.Actions.DOWN,
}
DIRECTION_TO_DELTA = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, -1),
    "down": (0, 1),
}


def _render_grid(rows: tuple[str, ...]) -> str:
    width = len(rows[0])
    header = "  " + " ".join(str(i) for i in range(width))
    lines = [header]
    for y, row in enumerate(rows):
        lines.append(f"{y} " + " ".join(row))
    return "\n".join(lines)


def _make_example(
    *,
    example_id: str,
    rows: tuple[str, ...],
    carrying_key: bool,
    has_key_label: str,
    door_open_label: str,
    wall_right_label: str,
    hit_wall_after_right_label: str,
    has_key_after_right_label: str,
    door_open_after_right_label: str,
    observed_action: str | None = None,
    notes: str | None = None,
) -> BehavioralProbeSmokeExample:
    return BehavioralProbeSmokeExample(
        example_id=example_id,
        grid_text=_render_grid(rows),
        carrying_key=carrying_key,
        observed_action=observed_action,
        has_key_label=has_key_label,
        door_open_label=door_open_label,
        wall_right_label=wall_right_label,
        hit_wall_after_right_label=hit_wall_after_right_label,
        has_key_after_right_label=has_key_after_right_label,
        door_open_after_right_label=door_open_after_right_label,
        notes=notes,
    )


SMOKE_TEST_STATES: list[BehavioralProbeSmokeExample] = [
    _make_example(
        example_id="smoke_001_basic_open_right",
        rows=(
            "#########",
            "#A__K___#",
            "#_______#",
            "#___D___#",
            "#___G___#",
            "#_______#",
            "#_______#",
            "#_______#",
            "#########",
        ),
        carrying_key=False,
        has_key_label="no",
        door_open_label="no",
        wall_right_label="no",
        hit_wall_after_right_label="no",
        has_key_after_right_label="no",
        door_open_after_right_label="no",
        notes="Simple open-right state with key elsewhere and a closed door elsewhere.",
    ),
    _make_example(
        example_id="smoke_002_wall_right",
        rows=(
            "#########",
            "#A#K____#",
            "#_______#",
            "#___D___#",
            "#___G___#",
            "#_______#",
            "#_______#",
            "#_______#",
            "#########",
        ),
        carrying_key=False,
        has_key_label="no",
        door_open_label="no",
        wall_right_label="yes",
        hit_wall_after_right_label="yes",
        has_key_after_right_label="no",
        door_open_after_right_label="no",
        notes="Immediate east cell is a wall.",
    ),
    _make_example(
        example_id="smoke_003_pick_key_right",
        rows=(
            "#########",
            "#AK_____#",
            "#_______#",
            "#___D___#",
            "#___G___#",
            "#_______#",
            "#_______#",
            "#_______#",
            "#########",
        ),
        carrying_key=False,
        has_key_label="no",
        door_open_label="no",
        wall_right_label="no",
        hit_wall_after_right_label="no",
        has_key_after_right_label="yes",
        door_open_after_right_label="no",
        notes="Moving RIGHT picks up the key immediately.",
    ),
    _make_example(
        example_id="smoke_004_carrying_open_door_visible",
        rows=(
            "#########",
            "#A_O____#",
            "#_______#",
            "#___G___#",
            "#_______#",
            "#_______#",
            "#_______#",
            "#_______#",
            "#########",
        ),
        carrying_key=True,
        has_key_label="yes",
        door_open_label="yes",
        wall_right_label="no",
        hit_wall_after_right_label="no",
        has_key_after_right_label="yes",
        door_open_after_right_label="yes",
        notes="Open door is already visible in the manual smoke state.",
    ),
    _make_example(
        example_id="smoke_005_carrying_wall_right_open_door",
        rows=(
            "#########",
            "#A#O____#",
            "#_______#",
            "#___G___#",
            "#_______#",
            "#_______#",
            "#_______#",
            "#_______#",
            "#########",
        ),
        carrying_key=True,
        has_key_label="yes",
        door_open_label="yes",
        wall_right_label="yes",
        hit_wall_after_right_label="yes",
        has_key_after_right_label="yes",
        door_open_after_right_label="yes",
        notes="Wall to the east, but an open door is visible elsewhere.",
    ),
    _make_example(
        example_id="smoke_006_goal_right",
        rows=(
            "#########",
            "#AG_D___#",
            "#_______#",
            "#___K___#",
            "#_______#",
            "#_______#",
            "#_______#",
            "#_______#",
            "#########",
        ),
        carrying_key=False,
        has_key_label="no",
        door_open_label="no",
        wall_right_label="no",
        hit_wall_after_right_label="no",
        has_key_after_right_label="no",
        door_open_after_right_label="no",
        notes="The goal is immediately to the east, which should often elicit RIGHT.",
    ),
    _make_example(
        example_id="smoke_007_open_door_right",
        rows=(
            "#########",
            "#AO_G___#",
            "#_______#",
            "#_______#",
            "#_______#",
            "#___K___#",
            "#_______#",
            "#_______#",
            "#########",
        ),
        carrying_key=False,
        has_key_label="no",
        door_open_label="yes",
        wall_right_label="no",
        hit_wall_after_right_label="no",
        has_key_after_right_label="no",
        door_open_after_right_label="yes",
        notes="Open door immediately to the east remains open after moving RIGHT.",
    ),
    _make_example(
        example_id="smoke_008_carrying_closed_door_elsewhere",
        rows=(
            "#########",
            "#A______#",
            "#_______#",
            "#___D___#",
            "#___G___#",
            "#_______#",
            "#_______#",
            "#_______#",
            "#########",
        ),
        carrying_key=True,
        has_key_label="yes",
        door_open_label="no",
        wall_right_label="no",
        hit_wall_after_right_label="no",
        has_key_after_right_label="yes",
        door_open_after_right_label="no",
        notes="Carrying the key does not imply the closed door is already open.",
    ),
    _make_example(
        example_id="smoke_009_wall_right_no_door",
        rows=(
            "#########",
            "#A#_____#",
            "#_______#",
            "#_______#",
            "#___G___#",
            "#___K___#",
            "#_______#",
            "#_______#",
            "#########",
        ),
        carrying_key=False,
        has_key_label="no",
        door_open_label="no",
        wall_right_label="yes",
        hit_wall_after_right_label="yes",
        has_key_after_right_label="no",
        door_open_after_right_label="no",
        notes="No door in the state; RIGHT is blocked only by a wall.",
    ),
    _make_example(
        example_id="smoke_010_open_door_elsewhere",
        rows=(
            "#########",
            "#A______#",
            "#_______#",
            "#___O___#",
            "#___G___#",
            "#_____K_#",
            "#_______#",
            "#_______#",
            "#########",
        ),
        carrying_key=False,
        has_key_label="no",
        door_open_label="yes",
        wall_right_label="no",
        hit_wall_after_right_label="no",
        has_key_after_right_label="no",
        door_open_after_right_label="yes",
        notes="Open door is visible elsewhere; moving RIGHT leaves that status unchanged.",
    ),
]


def grid_text_to_layout(grid_text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in grid_text.strip().splitlines():
        parts = raw_line.strip().split()
        if not parts:
            continue
        if all(token.isdigit() for token in parts):
            continue
        if parts[0].isdigit():
            parts = parts[1:]
        rows.append(parts)
    return rows


def _coord(x: int | None, y: int | None) -> dict[str, int]:
    if x is None or y is None:
        return dict(COORD_MISSING)
    return {"row": int(y), "col": int(x)}


def _find_positions(layout: list[list[str]]) -> dict[str, list[tuple[int, int]]]:
    positions: dict[str, list[tuple[int, int]]] = {}
    for y, row in enumerate(layout):
        for x, cell in enumerate(row):
            positions.setdefault(cell, []).append((x, y))
    return positions


def _build_env_from_state(grid_text: str, carrying_key: bool) -> RoomsMinigridEnv:
    layout = grid_text_to_layout(grid_text)
    env = RoomsMinigridEnv(rooms_per_side=2, add_door_key=False, max_steps=10)
    env.set_env_from_list(layout)
    env.step_count = 0
    if carrying_key:
        env.carrying = Key("yellow")
    return env


def _clone_and_step(grid_text: str, carrying_key: bool, action_name: str) -> RoomsMinigridEnv:
    env = _build_env_from_state(grid_text, carrying_key)
    env.step(ACTION_TO_ENV_ACTION[action_name])
    return env


def _any_open_door(env: RoomsMinigridEnv) -> bool:
    for x in range(env.width):
        for y in range(env.height):
            cell = env.grid.get(x, y)
            if cell is not None and cell.type == "door" and getattr(cell, "is_open", False):
                return True
    return False


def _first_visible_coord(layout: list[list[str]], symbols: set[str]) -> dict[str, int]:
    positions = _find_positions(layout)
    for symbol in sorted(symbols):
        if positions.get(symbol):
            x, y = positions[symbol][0]
            return _coord(x, y)
    return dict(COORD_MISSING)


def derive_probe_truths_from_state(grid_text: str, carrying_key: bool) -> dict[str, Any]:
    layout = grid_text_to_layout(grid_text)
    positions = _find_positions(layout)
    if "A" not in positions:
        raise ValueError("Grid state must contain an agent 'A'.")
    agent_x, agent_y = positions["A"][0]
    truths: dict[str, Any] = {
        "has_key": "yes" if carrying_key else "no",
        "door_open": "yes" if bool(positions.get("O")) else "no",
        "agent_location": _coord(agent_x, agent_y),
        "goal_location": _first_visible_coord(layout, {"G"}),
        "key_location": _first_visible_coord(layout, {"K"}),
        "door_location": _first_visible_coord(layout, {"D", "O"}),
    }

    for direction, (dx, dy) in DIRECTION_TO_DELTA.items():
        nx, ny = agent_x + dx, agent_y + dy
        cell = None
        if 0 <= ny < len(layout) and 0 <= nx < len(layout[0]):
            cell = layout[ny][nx]
        truths[f"wall_{direction}"] = "yes" if cell == "#" else "no"
        truths[f"is_goal_{direction}"] = "yes" if cell == "G" else "no"
        truths[f"is_key_{direction}"] = "yes" if cell == "K" else "no"
        truths[f"is_door_{direction}"] = "yes" if cell in {"D", "O"} else "no"

    before_env = _build_env_from_state(grid_text, carrying_key)
    before_pos = tuple(before_env.agent_pos)
    for action in ACTIONS:
        action_lower = action.lower()
        env_after = _clone_and_step(grid_text, carrying_key, action)
        after_pos = tuple(env_after.agent_pos)
        truths[f"hit_wall_after_{action_lower}"] = (
            "yes"
            if after_pos == before_pos and truths[f"wall_{action_lower}"] == "yes"
            else "no"
        )
        carrying_after = bool(
            env_after.carrying is not None and getattr(env_after.carrying, "type", None) == "key"
        )
        truths[f"has_key_after_{action_lower}"] = "yes" if carrying_after else "no"
        truths[f"door_open_after_{action_lower}"] = "yes" if _any_open_door(env_after) else "no"
        truths[f"agent_location_after_{action_lower}"] = _coord(after_pos[0], after_pos[1])

    return truths


def derive_probe_truths(example: BehavioralProbeSmokeExample) -> dict[str, Any]:
    return derive_probe_truths_from_state(example.grid_text, example.carrying_key)


def derive_example_truth(example: BehavioralProbeSmokeExample) -> dict[str, str]:
    truths = derive_probe_truths(example)
    return {
        "has_key_label": truths["has_key"],
        "door_open_label": truths["door_open"],
        "wall_right_label": truths["wall_right"],
        "hit_wall_after_right_label": truths["hit_wall_after_right"],
        "has_key_after_right_label": truths["has_key_after_right"],
        "door_open_after_right_label": truths["door_open_after_right"],
    }


def validate_smoke_examples(
    examples: list[BehavioralProbeSmokeExample] | None = None,
) -> list[dict[str, Any]]:
    examples = SMOKE_TEST_STATES if examples is None else examples
    if not 8 <= len(examples) <= 12:
        raise ValueError("Smoke-test dataset must contain between 8 and 12 examples.")

    seen_ids: set[str] = set()
    summary_rows: list[dict[str, Any]] = []
    for example in examples:
        if example.example_id in seen_ids:
            raise ValueError(f"Duplicate example_id: {example.example_id}")
        seen_ids.add(example.example_id)

        layout = grid_text_to_layout(example.grid_text)
        if len(layout) != 9 or any(len(row) != 9 for row in layout):
            raise ValueError(f"{example.example_id}: expected a 9x9 layout.")

        counts: dict[str, int] = {}
        for row in layout:
            for cell in row:
                counts[cell] = counts.get(cell, 0) + 1

        if counts.get("A", 0) != 1:
            raise ValueError(f"{example.example_id}: expected exactly one agent 'A'.")
        if counts.get("G", 0) != 1:
            raise ValueError(f"{example.example_id}: expected exactly one goal 'G'.")
        if example.carrying_key and counts.get("K", 0) > 0:
            raise ValueError(
                f"{example.example_id}: carrying_key=True but the grid still contains K."
            )

        derived = derive_example_truth(example)
        for label_field in LABEL_FIELDS:
            manual_value = getattr(example, label_field)
            if manual_value not in {"yes", "no", "unknown"}:
                raise ValueError(
                    f"{example.example_id}: invalid label value for {label_field}: {manual_value}"
                )
            if derived[label_field] != manual_value:
                raise ValueError(
                    f"{example.example_id}: {label_field} mismatch, manual={manual_value}, derived={derived[label_field]}"
                )

        summary_rows.append(
            {
                "example_id": example.example_id,
                "carrying_key": example.carrying_key,
                **derived,
            }
        )

    return summary_rows


def smoke_examples_as_dicts() -> list[dict[str, Any]]:
    return [asdict(example) for example in SMOKE_TEST_STATES]


__all__ = [
    "ACTION_TO_DELTA",
    "DIRECTION_TO_DELTA",
    "BehavioralProbeSmokeExample",
    "SMOKE_TEST_STATES",
    "derive_example_truth",
    "derive_probe_truths",
    "derive_probe_truths_from_state",
    "grid_text_to_layout",
    "smoke_examples_as_dicts",
    "validate_smoke_examples",
]
