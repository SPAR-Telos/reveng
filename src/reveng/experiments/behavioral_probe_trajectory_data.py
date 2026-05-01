"""Build trajectory-derived behavioral-probe datasets from trace-viewer JSON files."""

from __future__ import annotations

import csv
import json
from collections import deque
from statistics import mean
from pathlib import Path
from typing import Any

from minigrid.core.world_object import Key

from reveng.analysis.analysis_utils import ACTION_ID_TO_NAME
from reveng.experiments.behavioral_probe_questions import ACTIONS
from reveng.experiments.behavioral_probe_smoke_data import (
    derive_probe_truths_from_state,
    grid_text_to_layout,
)
from reveng.experiments.behavioral_probe_plots import (
    plot_failure_mode_case_selection,
    plot_failure_mode_summary,
)
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.environment_generator.rooms_minigrid import RoomsMinigridEnv

ACTION_TO_ENV_ACTION = {
    "LEFT": Simple2DNavigationEnv.Actions.LEFT,
    "RIGHT": Simple2DNavigationEnv.Actions.RIGHT,
    "UP": Simple2DNavigationEnv.Actions.UP,
    "DOWN": Simple2DNavigationEnv.Actions.DOWN,
}
TRAJECTORY_CLASSES = ("optimal_success", "suboptimal_success", "failed")
StateKey = tuple[str, bool]

CORE_FAILURE_MODES = (
    "wall_hit",
    "backtrack",
    "oscillation_2cycle",
    "short_loop",
    "freeze_repeat",
    "avoidable_detour",
)
REPEATED_FAILURE_MODES = ("oscillation_2cycle", "short_loop", "freeze_repeat")
PRIMARY_FAILURE_MODE_ORDER = (
    "wall_hit",
    "freeze_repeat",
    "oscillation_2cycle",
    "short_loop",
    "backtrack",
    "avoidable_detour",
    "terminal_failure_tail",
)
FAILURE_COUNT_FIELDS = {
    "wall_hit": "wall_hit_step_count",
    "backtrack": "backtrack_step_count",
    "oscillation_2cycle": "oscillation_step_count",
    "short_loop": "short_loop_step_count",
    "freeze_repeat": "freeze_repeat_step_count",
    "avoidable_detour": "avoidable_detour_step_count",
}
DEFAULT_LOOP_WINDOW = 6
DEFAULT_FREEZE_REPEAT_THRESHOLD = 2
DEFAULT_TERMINAL_FAILURE_TAIL = 3
DEFAULT_MAX_SELECTED_FAILURE_STEPS_PER_TRAJECTORY = 3
DEFAULT_MAX_SELECTED_STEPS_PER_TRAJECTORY = 4
DEFAULT_BALANCED_FAILURE_MODE_ROWS_PER_MODE = 2


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _grid_lines_to_text(grid_state: list[str] | str) -> str:
    if isinstance(grid_state, str):
        return grid_state
    return "\n".join(grid_state)


def _render_grid(rows: list[list[str]]) -> str:
    width = len(rows[0]) if rows else 0
    header = "  " + " ".join(str(i) for i in range(width))
    lines = [header]
    for y, row in enumerate(rows):
        lines.append(f"{y} " + " ".join(row))
    return "\n".join(lines)


def _canonicalize_state(grid_text: str, carrying_key: bool) -> StateKey:
    return _render_grid(grid_text_to_layout(grid_text)), bool(carrying_key)


def _build_env_from_state(grid_text: str, carrying_key: bool) -> RoomsMinigridEnv:
    layout = grid_text_to_layout(grid_text)
    env = RoomsMinigridEnv(rooms_per_side=2, add_door_key=False, max_steps=200)
    env.set_env_from_list(layout)
    env.step_count = 0
    if carrying_key:
        env.carrying = Key("yellow")
    return env


def _env_to_state(env: RoomsMinigridEnv) -> StateKey:
    rows: list[list[str]] = []
    agent_pos = tuple(int(v) for v in env.agent_pos)
    for y in range(env.height):
        row: list[str] = []
        for x in range(env.width):
            if (x, y) == agent_pos:
                row.append("A")
                continue
            cell = env.grid.get(x, y)
            if cell is None:
                row.append("_")
            elif cell.type == "wall":
                row.append("#")
            elif cell.type == "goal":
                row.append("G")
            elif cell.type == "key":
                row.append("K")
            elif cell.type == "door":
                row.append("O" if getattr(cell, "is_open", False) else "D")
            else:
                row.append("_")
        rows.append(row)
    carrying_key = bool(env.carrying is not None and getattr(env.carrying, "type", None) == "key")
    return _render_grid(rows), carrying_key


def _coord_to_tuple(coord: dict[str, int]) -> tuple[int, int]:
    return int(coord["col"]), int(coord["row"])


def _extract_model_name(payload: dict[str, Any]) -> str:
    model_params = payload.get("model_params", {})
    provider = model_params.get("provider")
    model_id = model_params.get("model_id")
    if provider and model_id:
        return f"{provider}/{model_id}"
    return model_id or ""


class DoorKeyStateSolver:
    """Compute DoorKey-aware transitions and shortest paths from rendered states.

    Complexity notes:
    - Each transition requires one environment reconstruction plus one step.
    - Shortest-path search is a BFS over reachable DoorKey states with four actions.
    - For the current 9x9, one-key, one-door setup the practical state space is small:
      agent position x carrying-key bit x door-present/removed bit, so on the order of
      a few hundred reachable states rather than thousands.
    """

    def __init__(self) -> None:
        self.transition_cache: dict[tuple[StateKey, str], dict[str, Any]] = {}
        self.distance_cache: dict[StateKey, int | None] = {}
        self.optimal_action_cache: dict[StateKey, list[str]] = {}

    def step(self, state: StateKey, action_name: str) -> dict[str, Any]:
        cache_key = (state, action_name)
        if cache_key in self.transition_cache:
            return self.transition_cache[cache_key]

        grid_text, carrying_key = state
        env = _build_env_from_state(grid_text, carrying_key)
        before_pos = tuple(int(v) for v in env.agent_pos)
        env.step(ACTION_TO_ENV_ACTION[action_name])
        after_pos = tuple(int(v) for v in env.agent_pos)
        next_state = _env_to_state(env)
        result = {
            "next_state": next_state,
            "hit_wall": after_pos == before_pos,
            "reached_goal": after_pos == tuple(int(v) for v in env.goal_pos),
            "before_pos": before_pos,
            "after_pos": after_pos,
        }
        self.transition_cache[cache_key] = result
        return result

    def shortest_distance(self, state: StateKey) -> int | None:
        if state in self.distance_cache:
            return self.distance_cache[state]

        env = _build_env_from_state(*state)
        if getattr(env, "goal_pos", None) is None:
            self.distance_cache[state] = 0
            return 0
        if tuple(int(v) for v in env.agent_pos) == tuple(int(v) for v in env.goal_pos):
            self.distance_cache[state] = 0
            return 0

        queue: deque[tuple[StateKey, int]] = deque([(state, 0)])
        visited: set[StateKey] = {state}

        while queue:
            current_state, distance = queue.popleft()
            for action_name in ACTIONS:
                transition = self.step(current_state, action_name)
                if transition["reached_goal"]:
                    self.distance_cache[state] = distance + 1
                    return distance + 1
                next_state = transition["next_state"]
                if next_state == current_state or next_state in visited:
                    continue
                visited.add(next_state)
                queue.append((next_state, distance + 1))

        self.distance_cache[state] = None
        return None

    def optimal_actions(self, state: StateKey) -> list[str]:
        if state in self.optimal_action_cache:
            return self.optimal_action_cache[state]

        scored_actions: dict[str, int] = {}
        for action_name in ACTIONS:
            transition = self.step(state, action_name)
            if transition["reached_goal"]:
                scored_actions[action_name] = 1
                continue
            next_state = transition["next_state"]
            if next_state == state:
                continue
            remaining = self.shortest_distance(next_state)
            if remaining is None:
                continue
            scored_actions[action_name] = 1 + remaining

        if not scored_actions:
            self.optimal_action_cache[state] = []
            return []

        best_score = min(scored_actions.values())
        optimal_actions = sorted(
            action_name for action_name, score in scored_actions.items() if score == best_score
        )
        self.optimal_action_cache[state] = optimal_actions
        return optimal_actions


def _legacy_optimal_actions_from_state(grid_text: str) -> list[str]:
    truths = derive_probe_truths_from_state(grid_text, carrying_key=False)
    goal = truths["goal_location"]
    if goal["row"] < 0 or goal["col"] < 0:
        return []

    from reveng.analysis.analysis_utils import compute_optimal_actions_from_text_grid

    layout = grid_text_to_layout(grid_text)
    optimal_actions, _ = compute_optimal_actions_from_text_grid(layout, (goal["col"], goal["row"]))
    agent = truths["agent_location"]
    action_ids = optimal_actions.get((agent["col"], agent["row"]), set())
    return sorted(ACTION_ID_TO_NAME[action_id] for action_id in action_ids)


def _trajectory_class(reached_goal: bool, actual_length: int, optimal_length: int | None) -> str:
    if not reached_goal or optimal_length is None:
        return "failed"
    if actual_length == optimal_length:
        return "optimal_success"
    return "suboptimal_success"


def _stored_astar_suspicious(
    stored_astar_distance: Any,
    actual_length: int,
    optimal_length: int | None,
    model_max_steps: Any,
) -> bool:
    if not isinstance(stored_astar_distance, int):
        return True
    if optimal_length is None:
        return True
    if stored_astar_distance != optimal_length:
        return True
    if actual_length < stored_astar_distance:
        return True
    if isinstance(model_max_steps, int) and stored_astar_distance == model_max_steps:
        return True
    return False


def _primary_failure_mode_from_modes(modes: list[str]) -> str:
    for mode in PRIMARY_FAILURE_MODE_ORDER:
        if mode in modes:
            return mode
    return "none"


def _serialize_position(position: tuple[int, int] | None) -> str:
    if position is None:
        return ""
    return f"{position[0]},{position[1]}"


def _dedupe_adjacent_candidates(instance_rows: list[dict[str, Any]], candidate_indices: list[int]) -> list[int]:
    deduped: list[int] = []
    for idx in sorted(set(candidate_indices)):
        if not deduped:
            deduped.append(idx)
            continue
        prev_idx = deduped[-1]
        same_mode = (
            instance_rows[idx]["primary_step_failure_mode"]
            == instance_rows[prev_idx]["primary_step_failure_mode"]
        )
        same_state = instance_rows[idx]["canonical_state"] == instance_rows[prev_idx]["canonical_state"]
        adjacent = idx == prev_idx + 1
        if same_mode and same_state and adjacent:
            continue
        deduped.append(idx)
    return deduped


def _can_add_pre_failure_context(instance_rows: list[dict[str, Any]], context_idx: int, failure_idx: int) -> bool:
    if context_idx < 0:
        return False
    if instance_rows[context_idx]["canonical_state"] == instance_rows[failure_idx]["canonical_state"]:
        return False
    return True


def _annotate_failure_modes(
    instance_rows: list[dict[str, Any]],
    *,
    trajectory_class: str,
    failed_by_truncation: bool,
    loop_window: int = DEFAULT_LOOP_WINDOW,
    freeze_repeat_threshold: int = DEFAULT_FREEZE_REPEAT_THRESHOLD,
    terminal_failure_tail: int = DEFAULT_TERMINAL_FAILURE_TAIL,
    max_selected_failure_steps_per_trajectory: int = DEFAULT_MAX_SELECTED_FAILURE_STEPS_PER_TRAJECTORY,
    max_selected_steps_per_trajectory: int = DEFAULT_MAX_SELECTED_STEPS_PER_TRAJECTORY,
) -> dict[str, Any]:
    current_positions = [row["current_position"] for row in instance_rows]
    next_positions = [row["next_position"] for row in instance_rows]
    canonical_states = [row["canonical_state"] for row in instance_rows]
    current_distances = [row["current_optimal_distance"] for row in instance_rows]

    failure_onset_index: int | None = None
    repeated_mode_indices: dict[str, list[int]] = {mode: [] for mode in REPEATED_FAILURE_MODES}
    selection_candidates: list[int] = []
    selection_reasons: dict[int, list[str]] = {}
    mode_counts = {mode: 0 for mode in CORE_FAILURE_MODES}

    for idx, row in enumerate(instance_rows):
        modes: list[str] = []
        if row["wall_hit"]:
            modes.append("wall_hit")

        if idx > 0 and row["observed_action"] and next_positions[idx] == current_positions[idx - 1]:
            modes.append("backtrack")

        if (
            idx >= 2
            and next_positions[idx] is not None
            and current_positions[idx] == current_positions[idx - 2]
            and next_positions[idx] == current_positions[idx - 1]
        ):
            modes.append("oscillation_2cycle")

        recent_positions = current_positions[max(0, idx - loop_window) : idx]
        if (
            next_positions[idx] is not None
            and next_positions[idx] in recent_positions
            and "oscillation_2cycle" not in modes
            and "backtrack" not in modes
        ):
            modes.append("short_loop")

        freeze_streak = 1
        back_idx = idx - 1
        while back_idx >= 0 and canonical_states[back_idx] == canonical_states[idx]:
            freeze_streak += 1
            back_idx -= 1
        if (
            freeze_streak >= freeze_repeat_threshold
            and idx > 0
            and current_distances[idx] is not None
            and current_distances[idx - 1] is not None
            and current_distances[idx] >= current_distances[idx - 1]
        ):
            modes.append("freeze_repeat")

        if row["observed_action"] and not row["is_optimal_action"] and not row["wall_hit"]:
            modes.append("avoidable_detour")

        if not row["trajectory_reached_goal"] and failed_by_truncation and idx >= max(0, len(instance_rows) - terminal_failure_tail):
            modes.append("terminal_failure_tail")

        row["failure_modes"] = sorted(set(modes), key=lambda mode: PRIMARY_FAILURE_MODE_ORDER.index(mode))
        row["primary_step_failure_mode"] = _primary_failure_mode_from_modes(row["failure_modes"])
        row["is_failure_onset"] = False
        row["is_terminal_failure_tail"] = "terminal_failure_tail" in row["failure_modes"]
        row["is_pre_failure_context"] = False
        row["pre_failure_for_step_index"] = None
        row["selection_stage"] = "failure"
        row["selected_for_probe"] = False
        row["selection_reason"] = ""

        core_modes = [mode for mode in row["failure_modes"] if mode in CORE_FAILURE_MODES]
        if core_modes and failure_onset_index is None:
            failure_onset_index = idx
            row["is_failure_onset"] = True
        elif failure_onset_index is not None and idx == failure_onset_index:
            row["is_failure_onset"] = True

        for mode in core_modes:
            mode_counts[mode] += 1
            if mode in {"wall_hit", "backtrack", "avoidable_detour"}:
                selection_candidates.append(idx)
                selection_reasons.setdefault(idx, []).append(mode)
            if mode in REPEATED_FAILURE_MODES:
                repeated_mode_indices[mode].append(idx)

    for mode, indices in repeated_mode_indices.items():
        if not indices:
            continue
        onset = indices[0]
        selection_candidates.append(onset)
        selection_reasons.setdefault(onset, []).append(f"{mode}_onset")
        if len(indices) > 1:
            representative = indices[1]
            selection_candidates.append(representative)
            selection_reasons.setdefault(representative, []).append(f"{mode}_representative")

    if trajectory_class == "failed":
        if failure_onset_index is not None:
            selection_candidates.append(failure_onset_index)
            selection_reasons.setdefault(failure_onset_index, []).append("failure_onset")
        for idx in range(max(0, len(instance_rows) - terminal_failure_tail), len(instance_rows)):
            selection_candidates.append(idx)
            selection_reasons.setdefault(idx, []).append("terminal_failure_tail")
    elif trajectory_class == "suboptimal_success":
        for idx, row in enumerate(instance_rows):
            if any(mode in CORE_FAILURE_MODES for mode in row["failure_modes"]):
                selection_candidates.append(idx)
                selection_reasons.setdefault(idx, []).append("suboptimal_success_failure_step")

    selection_candidates = _dedupe_adjacent_candidates(instance_rows, selection_candidates)
    selected_failure_indices = selection_candidates[:max_selected_failure_steps_per_trajectory]
    selected_indices = list(selected_failure_indices)
    for idx in selected_failure_indices:
        context_idx = idx - 1
        if len(selected_indices) >= max_selected_steps_per_trajectory:
            break
        if context_idx in selected_indices:
            continue
        if not _can_add_pre_failure_context(instance_rows, context_idx, idx):
            continue
        selected_indices.append(context_idx)
        selection_reasons.setdefault(context_idx, []).append("pre_failure_context")
        instance_rows[context_idx]["is_pre_failure_context"] = True
        instance_rows[context_idx]["pre_failure_for_step_index"] = idx
        instance_rows[context_idx]["selection_stage"] = "context"

    selected_indices = sorted(_dedupe_adjacent_candidates(instance_rows, selected_indices))
    selected_indices = selected_indices[:max_selected_steps_per_trajectory]
    selected_index_set = set(selected_indices)
    for idx in selected_indices:
        instance_rows[idx]["selected_for_probe"] = True
        instance_rows[idx]["selection_reason"] = ",".join(sorted(set(selection_reasons.get(idx, []))))
    for idx, row in enumerate(instance_rows):
        if idx not in selected_index_set:
            row["is_pre_failure_context"] = False
            row["pre_failure_for_step_index"] = None
            row["selection_stage"] = "failure"

    primary_failure_mode = "none"
    for mode in PRIMARY_FAILURE_MODE_ORDER:
        if mode in CORE_FAILURE_MODES and mode_counts[mode] > 0:
            primary_failure_mode = mode
            break
    if primary_failure_mode == "none" and trajectory_class == "failed":
        primary_failure_mode = "terminal_failure_tail"

    return {
        "failure_onset_index": failure_onset_index,
        "primary_failure_mode": primary_failure_mode,
        "mode_counts": mode_counts,
        "selected_indices": selected_indices,
    }


def _analyze_trajectory_file(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    steps = payload.get("steps", [])
    if not steps:
        return None, []

    solver = DoorKeyStateSolver()
    initial_state: StateKey = _canonicalize_state(
        _grid_lines_to_text(steps[0].get("grid_state", [])),
        bool(steps[0].get("carrying_key", False)),
    )
    actual_length = len(steps)
    stored_astar_distance = payload.get("grid_params", {}).get("astar_distance")
    model_max_steps = payload.get("model_params", {}).get("max_steps_per_trajectory")
    optimal_length = solver.shortest_distance(initial_state)

    current_state = initial_state
    reached_goal = False
    replay_mismatch_count = 0
    instance_rows: list[dict[str, Any]] = []

    for step_index, step in enumerate(steps):
        grid_text = _grid_lines_to_text(step.get("grid_state", []))
        carrying_key = bool(step.get("carrying_key", False))
        step_state = _canonicalize_state(grid_text, carrying_key)
        if step_state != current_state:
            replay_mismatch_count += 1
            current_state = step_state

        observed_action = step.get("agent_action")
        truths = derive_probe_truths_from_state(grid_text, carrying_key)
        optimal_actions = solver.optimal_actions(step_state)
        legacy_optimal_actions = _legacy_optimal_actions_from_state(grid_text)
        is_optimal_action = observed_action in optimal_actions if observed_action else False
        current_position = _coord_to_tuple(truths["agent_location"])
        current_distance = solver.shortest_distance(step_state)

        wall_hit = False
        reached_goal_after_action = False
        next_state = step_state
        next_position = current_position
        next_distance = current_distance
        if observed_action in ACTION_TO_ENV_ACTION:
            transition = solver.step(step_state, observed_action)
            wall_hit = bool(transition["hit_wall"])
            reached_goal_after_action = bool(transition["reached_goal"])
            next_state = transition["next_state"]
            current_state = next_state
            next_truths = derive_probe_truths_from_state(next_state[0], next_state[1])
            next_position = _coord_to_tuple(next_truths["agent_location"])
            next_distance = solver.shortest_distance(next_state)
        if reached_goal_after_action:
            reached_goal = True

        instance_rows.append(
            {
                "example_id": f"{path.stem}_step_{step_index:03d}",
                "trajectory_id": path.stem,
                "source_file": str(path),
                "model_name": _extract_model_name(payload),
                "step_index": step_index,
                "grid_text": grid_text,
                "carrying_key": carrying_key,
                "observed_action": observed_action,
                "probe_truths": truths,
                "optimal_actions": optimal_actions,
                "legacy_optimal_actions": legacy_optimal_actions,
                "is_optimal_action": is_optimal_action,
                "wall_hit": wall_hit,
                "current_position": current_position,
                "next_position": next_position,
                "canonical_state": step_state[0],
                "current_optimal_distance": current_distance,
                "next_optimal_distance": next_distance,
                "trajectory_actual_length": actual_length,
                "trajectory_recomputed_optimal_length": optimal_length,
                "trajectory_length_delta": actual_length - optimal_length if optimal_length is not None else None,
                "trajectory_reached_goal": reached_goal,
                "stored_astar_distance": stored_astar_distance,
                "stored_astar_suspicious": _stored_astar_suspicious(
                    stored_astar_distance,
                    actual_length,
                    optimal_length,
                    model_max_steps,
                ),
            }
        )

    failed_by_truncation = not reached_goal and (
        not isinstance(model_max_steps, int) or actual_length >= model_max_steps
    )
    trajectory_class = _trajectory_class(reached_goal, actual_length, optimal_length)
    for row in instance_rows:
        row["trajectory_reached_goal"] = reached_goal
        row["trajectory_class"] = trajectory_class
    annotations = _annotate_failure_modes(
        instance_rows,
        trajectory_class=trajectory_class,
        failed_by_truncation=failed_by_truncation,
    )

    trajectory_row = {
        "trajectory_id": path.stem,
        "source_file": str(path),
        "model_name": _extract_model_name(payload),
        "actual_length": actual_length,
        "stored_astar_distance": stored_astar_distance,
        "recomputed_optimal_length": optimal_length,
        "length_delta": actual_length - optimal_length if optimal_length is not None else None,
        "reached_goal": reached_goal,
        "failed_by_truncation": failed_by_truncation,
        "replay_mismatch_count": replay_mismatch_count,
        "model_max_steps_per_trajectory": model_max_steps,
        "stored_astar_suspicious": _stored_astar_suspicious(
            stored_astar_distance,
            actual_length,
            optimal_length,
            model_max_steps,
        ),
        "trajectory_class": trajectory_class,
        "primary_failure_mode": annotations["primary_failure_mode"],
        "failure_onset_step_index": annotations["failure_onset_index"],
        "selected_step_count": len(annotations["selected_indices"]),
    }
    for mode, count_field in FAILURE_COUNT_FIELDS.items():
        trajectory_row[count_field] = annotations["mode_counts"][mode]
        trajectory_row[f"contains_{mode}"] = annotations["mode_counts"][mode] > 0
    trajectory_row["wall_hit_count"] = trajectory_row["wall_hit_step_count"]
    trajectory_row["non_optimal_step_count"] = sum(
        1 for row in instance_rows if row["observed_action"] and not row["is_optimal_action"]
    )
    return trajectory_row, instance_rows


def _row_matches_failure_mode(row: dict[str, Any], failure_mode: str) -> bool:
    if failure_mode == "terminal_failure_tail":
        return bool(row["is_terminal_failure_tail"])
    return failure_mode in row["failure_modes"]


def _balanced_failure_mode_candidates(
    rows: list[dict[str, Any]],
    *,
    rows_per_mode: int = DEFAULT_BALANCED_FAILURE_MODE_ROWS_PER_MODE,
) -> list[dict[str, Any]]:
    """Select a compact probe subset with every available failure mode present."""
    selected_rows = [row for row in rows if row["selected_for_probe"]]
    by_step = {
        (row["trajectory_id"], int(row["step_index"])): row
        for row in selected_rows
    }
    chosen: dict[tuple[str, int], dict[str, Any]] = {}

    for mode in CORE_FAILURE_MODES:
        mode_rows = [
            row
            for row in selected_rows
            if row["selection_stage"] == "failure" and mode in row["failure_modes"]
        ]
        for row in mode_rows[:rows_per_mode]:
            key = (row["trajectory_id"], int(row["step_index"]))
            chosen[key] = row
            context_key = (row["trajectory_id"], int(row["step_index"]) - 1)
            context_row = by_step.get(context_key)
            if (
                context_row is not None
                and context_row.get("is_pre_failure_context")
                and int(context_row.get("pre_failure_for_step_index")) == int(row["step_index"])
            ):
                chosen[context_key] = context_row

    return [
        chosen[key]
        for key in sorted(chosen, key=lambda item: (chosen[item]["trajectory_id"], item[1]))
    ]


def _filter_instances(
    rows: list[dict[str, Any]],
    slice_type: str,
) -> list[dict[str, Any]]:
    if slice_type == "all_steps":
        return rows
    if slice_type == "wall_hit":
        return [row for row in rows if row["wall_hit"]]
    if slice_type == "non_optimal_action":
        return [row for row in rows if row["observed_action"] and not row["is_optimal_action"]]
    if slice_type == "suboptimal_trajectory":
        return [row for row in rows if row["trajectory_length_delta"] and row["trajectory_length_delta"] > 0]
    if slice_type == "failed_trajectory":
        return [row for row in rows if not row["trajectory_reached_goal"]]
    if slice_type == "selection_candidates":
        return [row for row in rows if row["selected_for_probe"]]
    if slice_type == "tagged_failure_states":
        return [row for row in rows if row["selected_for_probe"] and row["selection_stage"] == "failure"]
    if slice_type == "balanced_failure_modes":
        return _balanced_failure_mode_candidates(rows)
    if slice_type.startswith("failure_mode:"):
        failure_mode = slice_type.split(":", 1)[1]
        return [row for row in rows if _row_matches_failure_mode(row, failure_mode)]
    raise ValueError(f"Unknown slice_type: {slice_type}")


def build_trajectory_manifest(trajectory_dir: str) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    for path in sorted(Path(trajectory_dir).rglob("*.json")):
        trajectory_row, _ = _analyze_trajectory_file(path)
        if trajectory_row is not None:
            manifest_rows.append(trajectory_row)
    return manifest_rows


def build_trajectory_failure_mode_rows(trajectory_dir: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(trajectory_dir).rglob("*.json")):
        _, instance_rows = _analyze_trajectory_file(path)
        rows.extend(instance_rows)
    return rows


def mine_behavioral_probe_instances(
    trajectory_dir: str,
    *,
    slice_type: str = "all_steps",
    max_instances: int | None = None,
) -> list[dict[str, Any]]:
    rows = _filter_instances(build_trajectory_failure_mode_rows(trajectory_dir), slice_type)
    if max_instances is not None:
        rows = rows[:max_instances]
    return rows


def _summarize_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    optimal_rows = [row for row in rows if row["trajectory_class"] == "optimal_success"]
    suboptimal_rows = [row for row in rows if row["trajectory_class"] == "suboptimal_success"]
    failed_rows = [row for row in rows if row["trajectory_class"] == "failed"]
    scored_rows = [row for row in rows if row["recomputed_optimal_length"] is not None]
    summary: dict[str, Any] = {
        "trajectory_count": len(rows),
        "optimal_success_count": len(optimal_rows),
        "suboptimal_success_count": len(suboptimal_rows),
        "failed_count": len(failed_rows),
        "stored_astar_suspicious_count": sum(1 for row in rows if row["stored_astar_suspicious"]),
        "mean_actual_length": round(mean(row["actual_length"] for row in rows), 3) if rows else None,
        "mean_recomputed_optimal_length": (
            round(mean(row["recomputed_optimal_length"] for row in scored_rows), 3) if scored_rows else None
        ),
        "mean_length_delta_success": (
            round(mean(row["length_delta"] for row in optimal_rows + suboptimal_rows), 3)
            if optimal_rows or suboptimal_rows
            else None
        ),
    }
    for mode, count_field in FAILURE_COUNT_FIELDS.items():
        summary[f"suboptimal_contains_{mode}_count"] = sum(1 for row in suboptimal_rows if row[f"contains_{mode}"])
        summary[f"failed_contains_{mode}_count"] = sum(1 for row in failed_rows if row[f"contains_{mode}"])
        summary[f"total_{count_field}"] = sum(int(row[count_field]) for row in rows)
    return summary


def _serialize_failure_mode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serializable_rows: list[dict[str, Any]] = []
    for row in rows:
        serialized = {
            **row,
            "probe_truths_json": json.dumps(row["probe_truths"], sort_keys=True),
            "optimal_actions_json": json.dumps(row["optimal_actions"]),
            "legacy_optimal_actions_json": json.dumps(row["legacy_optimal_actions"]),
            "failure_modes_json": json.dumps(row["failure_modes"]),
            "current_position": _serialize_position(row["current_position"]),
            "next_position": _serialize_position(row["next_position"]),
        }
        serialized.pop("probe_truths")
        serialized.pop("optimal_actions")
        serialized.pop("legacy_optimal_actions")
        serialized.pop("failure_modes")
        serializable_rows.append(serialized)
    return serializable_rows


def run_behavioral_probe_trajectory_eval(
    trajectory_dir: str,
    output_dir: str = "data/behavioral_probes/trajectory_instances",
    slice_type: str = "all_steps",
    max_instances: int | None = None,
) -> None:
    """Mine trajectory-derived probe instances, failure modes, and selection candidates."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = build_trajectory_manifest(trajectory_dir)
    manifest_summary = _summarize_manifest(manifest_rows)
    failure_mode_rows = build_trajectory_failure_mode_rows(trajectory_dir)
    instances = _filter_instances(failure_mode_rows, slice_type)
    if max_instances is not None:
        instances = instances[:max_instances]
    selection_candidates = [row for row in failure_mode_rows if row["selected_for_probe"]]

    _write_csv(out_dir / "trajectory_manifest.csv", manifest_rows)
    (out_dir / "trajectory_manifest.json").write_text(json.dumps(manifest_rows, indent=2))
    _write_csv(out_dir / "trajectory_manifest_summary.csv", [manifest_summary])

    serializable_failure_mode_rows = _serialize_failure_mode_rows(failure_mode_rows)
    serializable_instances = _serialize_failure_mode_rows(instances)
    serializable_selection_candidates = _serialize_failure_mode_rows(selection_candidates)
    _write_csv(out_dir / "trajectory_failure_modes.csv", serializable_failure_mode_rows)
    _write_csv(out_dir / "trajectory_probe_instances.csv", serializable_instances)
    _write_csv(out_dir / "trajectory_selection_candidates.csv", serializable_selection_candidates)
    (out_dir / "trajectory_failure_modes.json").write_text(json.dumps(failure_mode_rows, indent=2))
    (out_dir / "trajectory_probe_instances.json").write_text(json.dumps(instances, indent=2))
    (out_dir / "trajectory_selection_candidates.json").write_text(json.dumps(selection_candidates, indent=2))

    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    if manifest_rows:
        plot_failure_mode_summary(manifest_rows, figs_dir / "failure_mode_summary.png")
    if selection_candidates:
        plot_failure_mode_case_selection(selection_candidates[:12], figs_dir / "selection_candidates.png")


__all__ = [
    "CORE_FAILURE_MODES",
    "DoorKeyStateSolver",
    "build_trajectory_failure_mode_rows",
    "build_trajectory_manifest",
    "mine_behavioral_probe_instances",
    "run_behavioral_probe_trajectory_eval",
]
