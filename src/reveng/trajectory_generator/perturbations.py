"""
Runtime perturbations for environment/trajectory pairs.

This module exposes a uniform API to attach controlled disturbances to a
MiniGrid-like navigation environment used in this project. Each function returns
(perturbed_env, plan, maybe_traj):

- perturbed_env: a wrapper or instrumented view of the original env that applies
  the requested changes at runtime without mutating the original base object
  unless explicitly required by the backend.
- plan: a PerturbationPlan describing the schedule (trigger steps) and metadata
  needed to reproduce the perturbation deterministically.
- maybe_traj: an optional trajectory re-rendered under the perturbation if a
  fully specified trajectory was provided for offline replay; otherwise None.

Contract and guarantees:
- Determinism: When a seed is provided, the perturbation behaves deterministically.
- Non-destructive by default: The returned env can be used independently; the input
  env is left intact unless the underlying framework requires in-place mutation.
- Composability: Multiple perturbations can be layered by feeding the output env
  of one call into the next, but conflicts are resolved on a “last-applied wins”
  basis unless otherwise documented by a specific function.
- Observability: When a perturbation only changes observations (not physics), it
  is noted explicitly in the docstring.

Typical usage
-------------
1) Prospective perturbation for data collection
    env2, plan, _ = apply_dynamic_obstacles(env, traj, add_walls_at={15: [(5,5)]})
    # Use env2 for rollout; persist `plan` alongside the trajectory for exact replay.

2) Retrospective perturbation for counterfactual replay
    env2, plan, perturbed_traj = apply_vision_occlusion(env, traj,
                                                        occlude_from_step=10,
                                                        duration_steps=20)
    # `perturbed_traj` is the original trajectory re-rendered under occlusion
    # (e.g., for scoring or visualization), when supported by the backend.

Edge cases
----------
- Out-of-bounds coordinates or negative step indices raise ValueError.
- When a requested change cannot be honored by the backend (e.g., missing
  state snapshots for temporal loops), a RuntimeError is raised with guidance.
- If multiple changes are scheduled for the same step, they are applied in a
  deterministic order documented per function (typically obstacles -> topology ->
  goal -> reward -> observation -> action-space -> multi-agent -> misc.).
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass

from reveng.datatypes import Step, Trajectory


@dataclass
class PerturbationPlan:
    name: str
    description: str
    trigger_steps: t.Sequence[int]
    metadata: dict[str, t.Any]


EnvType = t.Any


def apply_additional_cycle(
    env: EnvType,
    trajectory: Trajectory,
    *,
    cycle_start_step: int,
    repeat_count: int,
    cycle_length: int = 2,
    preserve_agent_memory: bool = True,
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Insert an unnecessary deviating loop that starts and ends at the same cell.

        Behavior (implemented):
        - Identifies the position immediately after `cycle_start_step` and inserts
        `repeat_count` loops at that point. Each loop has exactly `cycle_length`
        steps and returns to the same cell (net-zero displacement).
        - Loop preference: try to create a 4-step square loop when space allows; if
        not, create an out-and-back detour (go out some steps, then retrace). If
        fully boxed in, fall back to in-place back-and-forth pairs. The final
        sequence is composed to match `cycle_length` exactly (even integer >= 2).
        - The environment is not mutated; it is used read-only to inspect the grid.
        A new Trajectory is returned with the inserted steps.

        Parameters
         - cycle_start_step: Insert cycles immediately after this step index.
        - repeat_count: Number of cycles to insert; each cycle adds `cycle_length` steps.
         - cycle_length: Number of steps per cycle; must be an even integer >= 2.
             Semantics: cycle steps form a loop (square if possible, otherwise an
             out-and-back detour, else in-place pairs) and sum to exactly this length.
         - preserve_agent_memory: If False, `thought` is cleared (None) on inserted steps.

    Returns
     - (env, plan, new_trajectory)
       env: unchanged reference to the provided environment
       plan: PerturbationPlan describing insertion location and actions
       new_trajectory: trajectory with the inserted cycles, or the original if
                       repeat_count == 0

    Raises
     - ValueError: if indices are invalid/out of range, repeat_count < 0, or
       cycle_length is not an even integer >= 2.
    """
    # Validate inputs
    total = len(trajectory.steps)
    if total == 0:
        raise ValueError("trajectory has no steps")
    if cycle_start_step < 0:
        raise ValueError("cycle step indices must be non-negative")
    if cycle_start_step >= total:
        raise ValueError("cycle_start_step out of range")
    if repeat_count < 0:
        raise ValueError("repeat_count must be >= 0")
    if cycle_length < 2 or (cycle_length % 2) != 0:
        raise ValueError("cycle_length must be an even integer >= 2")

    insertion_index = min(cycle_start_step + 1, total)

    # Early no-op
    if repeat_count == 0:
        plan = PerturbationPlan(
            name="additional_cycle",
            description=(
                "No-op additional cycle (repeat_count == 0); trajectory unchanged."
            ),
            trigger_steps=[insertion_index],
            metadata={
                "cycle_start_step": cycle_start_step,
                "repeat_count": repeat_count,
                "cycle_length": cycle_length,
                "insertion_index_original": insertion_index,
                "chosen_cycle": None,
            },
        )
        return env, plan, trajectory

    # Ensure grid is available (reset initializes grid from fixed start/goal)
    base_env = getattr(env, "unwrapped", env)
    try:
        if getattr(base_env, "grid", None) is None:
            base_env.reset()
    except Exception:
        base_env.reset()

    grid = base_env.grid
    if grid is None:
        raise RuntimeError("Environment grid is unavailable; cannot plan cycle.")

    # Simulate position up to insertion_index using static grid rules
    def _parse_action(action_str: str) -> int | None:
        try:
            return int(action_str)
        except (TypeError, ValueError):
            return None

    # Direction deltas based on Simple2DNavigationEnv.Actions
    # 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN
    DELTA = {
        0: (-1, 0),  # LEFT
        1: (1, 0),  # RIGHT
        2: (0, -1),  # UP
        3: (0, 1),  # DOWN
    }
    INV = {0: 1, 1: 0, 2: 3, 3: 2}

    x, y = tuple(base_env.agent_start_pos)
    width, height = grid.width, grid.height

    def _can_move(nx: int, ny: int) -> bool:
        # Out-of-bounds should be walled by construction, but guard anyway
        if nx < 0 or ny < 0 or nx >= width or ny >= height:
            return False
        cell = grid.get(nx, ny)
        return cell is None or getattr(cell, "can_overlap", lambda: False)()

    # Roll through actions up to insertion point
    for k in range(insertion_index):
        a = _parse_action(trajectory.steps[k].action)
        if a is None or a not in DELTA:
            continue
        dx, dy = DELTA[a]
        nx, ny = x + dx, y + dy
        if _can_move(nx, ny):
            x, y = nx, ny
        # else stay in place

    # Choose a valid 2-step pair (move then inverse move)
    candidates = [1, 0, 3, 2]  # prefer RIGHT, LEFT, DOWN, UP
    chosen_first: int | None = None
    for a0 in candidates:
        dx, dy = DELTA[a0]
        nx, ny = x + dx, y + dy
        if _can_move(nx, ny):
            chosen_first = a0
            break

    if chosen_first is None:
        # Surrounded by walls: pick any pair whose both directions are blocked,
        # so both steps are no-ops and net position is unchanged.
        blocked = {a: not _can_move(x + DELTA[a][0], y + DELTA[a][1]) for a in DELTA}
        pair = None
        for a0 in candidates:
            if blocked.get(a0, True) and blocked.get(INV[a0], True):
                pair = (a0, INV[a0])
                break
        if pair is None:
            pair = (1, 0)  # fallback
    else:
        pair = (chosen_first, INV[chosen_first])

    # Build a deviating loop path that returns to (x, y)
    def _find_square_cycle() -> list[int] | None:
        # Try 4-step cycles (CW/CCW) starting from each primary direction
        square_candidates = [
            [1, 3, 0, 2],  # R, D, L, U
            [1, 2, 0, 3],  # R, U, L, D
            [0, 3, 1, 2],  # L, D, R, U
            [0, 2, 1, 3],  # L, U, R, D
            [2, 1, 3, 0],  # U, R, D, L
            [2, 0, 3, 1],  # U, L, D, R
            [3, 1, 2, 0],  # D, R, U, L
            [3, 0, 2, 1],  # D, L, U, R
        ]
        for seq in square_candidates:
            cx, cy = x, y
            ok = True
            for a in seq:
                dx, dy = DELTA[a]
                nx, ny = cx + dx, cy + dy
                if not _can_move(nx, ny):
                    ok = False
                    break
                cx, cy = nx, ny
            if ok and (cx, cy) == (x, y):
                return seq
        return None

    def _extend_out_and_back(max_half_len: int) -> list[int]:
        # Straight detour then back to start
        for a0 in [1, 0, 3, 2]:  # prefer R, L, D, U
            cx, cy = x, y
            path: list[int] = []
            for _ in range(max_half_len):
                dx, dy = DELTA[a0]
                nx, ny = cx + dx, cy + dy
                if not _can_move(nx, ny):
                    break
                path.append(a0)
                cx, cy = nx, ny
            if len(path) >= 1:
                return path
        return []

    loop_type = "square"
    base_cycle: list[int] = []
    square = _find_square_cycle()
    if square:
        # Repeat full squares
        full = cycle_length // 4
        base_cycle.extend(square * full)
        # If 2 extra steps are needed, add a short spur from origin
        rem = cycle_length % 4
        if rem == 2:
            # Prefer a perpendicular spur to deviate from the loop
            first = square[0]
            perp_candidates = [2, 3] if first in (1, 0) else [1, 0]
            chosen = None
            for a in perp_candidates + [first, INV[first]]:
                if _can_move(x + DELTA[a][0], y + DELTA[a][1]):
                    chosen = a
                    break
            if chosen is None:
                chosen = first
            base_cycle.extend([chosen, INV[chosen]])
    else:
        loop_type = "out-and-back"
        half = cycle_length // 2
        out = _extend_out_and_back(half)
        if not out:
            # Completely boxed in: fallback to in-place pair repeated
            a = 1 if not _can_move(x + DELTA[1][0], y + DELTA[1][1]) else 0
            for _ in range(cycle_length // 2):
                base_cycle.extend([a, INV[a]])
        else:
            base_cycle = out + [INV[a] for a in reversed(out)]
            # If still short (shouldn't be), pad with a small spur at origin
            rem = cycle_length - len(base_cycle)
            if rem > 0:
                # Choose a perpendicular to the outward direction
                a0 = out[0]
                perp = 2 if a0 in (1, 0) else 1
                if not _can_move(x + DELTA[perp][0], y + DELTA[perp][1]):
                    # Use the opposite perpendicular if blocked
                    perp = 3 if perp == 2 else 0
                for _ in range(rem // 2):
                    base_cycle.extend([perp, INV[perp]])

    # Build inserted steps by repeating the constructed cycle
    inserted_steps: list[Step] = []
    obs_stub = (
        trajectory.steps[insertion_index - 1].observation if insertion_index > 0 else ""
    )
    total_inserted = 0
    for _ in range(repeat_count):
        for a in base_cycle:
            inserted_steps.append(
                Step(
                    observation=obs_stub,
                    action=str(a),
                    reward=None,
                    metadata=(
                        trajectory.steps[insertion_index - 1].metadata
                        if preserve_agent_memory and insertion_index > 0
                        else None
                    ),
                )
            )
            total_inserted += 1

    # Splice into trajectory
    prefix = trajectory.steps[:insertion_index]
    suffix = trajectory.steps[insertion_index:]
    new_steps = list(prefix) + inserted_steps + list(suffix)

    new_traj = Trajectory(
        steps=new_steps,
        final_reward=trajectory.final_reward,
        traj_metadata=trajectory.traj_metadata,
    )

    plan = PerturbationPlan(
        name="additional_cycle",
        description=(
            "Inserted a net-zero displacement cycle at the specified point with configurable length."
        ),
        trigger_steps=[insertion_index],
        metadata={
            "cycle_start_step": cycle_start_step,
            "repeat_count": repeat_count,
            "cycle_length": cycle_length,
            "insertion_index_original": insertion_index,
            "chosen_cycle": pair,
            "inserted_total_steps": total_inserted,
            "loop_type": loop_type,
            "base_sequence": base_cycle,
        },
    )

    return env, plan, new_traj


def apply_dynamic_obstacles(
    env: EnvType,
    trajectory: Trajectory,
    *,
    add_walls_at: t.Mapping[int, t.Sequence[tuple[int, int]]] | None = None,
    remove_walls_at: t.Mapping[int, t.Sequence[tuple[int, int]]] | None = None,
    seed: int | None = None,
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Dynamically add or remove obstacles at scheduled steps.

    Behavior:
    - At each scheduled step in `add_walls_at`, the env introduces solid cells at
      the provided grid coordinates; in `remove_walls_at`, existing walls at those
      coordinates are cleared back to floor.
    - Physics is authoritative: newly added walls immediately block movement from
      the next physics tick; removals instantly open the cell.
    - If `seed` is provided and the backend requires randomized placement (e.g.,
      when a coordinate collides with a dynamic entity), the resolution is
      deterministic.

    Parameters
    - env: Base environment to wrap or instrument.
    - trajectory: The reference trajectory. If fully specified and the backend
      supports replay with env hooks, a perturbed replay may be produced.
    - add_walls_at: Mapping of step_index -> iterable of (x, y) to add as walls.
    - remove_walls_at: Mapping of step_index -> iterable of (x, y) to remove.
    - seed: Optional seed for deterministic tie-breaking.

    Returns
    - perturbed_env, plan, maybe_trajectory

    Raises
    - ValueError: On negative step indices or out-of-bounds coordinates.
    - RuntimeError: If the underlying env cannot apply wall mutations at runtime.
    """
    raise NotImplementedError()


def apply_goal_displacement(
    env: EnvType,
    trajectory: Trajectory,
    *,
    new_goal_pos_at: t.Mapping[int, tuple[int, int]],
    preserve_memory: bool = True,
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Relocate the goal to new coordinates at specified steps.

    Behavior:
    - On each trigger step, the env updates the goal position to the provided (x, y).
    - If `preserve_memory` is True, any agent-side map/memory retained by wrappers
      is left untouched; only the physical goal position changes.
    - If the agent currently occupies the goal at the moment of displacement, goal
      completion is re-evaluated based on the new position on the next tick.

    Parameters
    - new_goal_pos_at: Mapping of step_index -> (x, y) new goal location.
    - preserve_memory: Keep agent/wrapper memory as-is while moving the goal.

    Returns: perturbed_env, plan, maybe_trajectory

    Raises
    - ValueError: On invalid steps or coordinates; overlapping with walls if not
      supported by the backend.
    """
    raise NotImplementedError()


def apply_reward_structure_changes(
    env: EnvType,
    trajectory: Trajectory,
    *,
    reward_multipliers_at: t.Mapping[int, float] | None = None,
    action_penalties_at: t.Mapping[int, dict[str, float]] | None = None,
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Modify reward shaping at runtime.

    Behavior:
    - At trigger steps, update a running reward model:
      - `reward_multipliers_at`: multiply subsequent base rewards by the factor.
      - `action_penalties_at`: add per-action additive penalties (e.g., {"LEFT": -0.1}).
    - Changes persist until overwritten by a later trigger.

    Parameters
    - reward_multipliers_at: step -> scalar multiplier.
    - action_penalties_at: step -> {action_name: delta_reward}.

    Returns: perturbed_env, plan, maybe_trajectory

    Raises: ValueError on negative steps or invalid action names.
    """
    raise NotImplementedError()


def apply_vision_occlusion(
    env: EnvType,
    trajectory: Trajectory,
    *,
    occlude_from_step: int,
    occlusion_radius: int | None = None,
    duration_steps: int | None = None,
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Reduce the observable radius or mask parts of the observation stream.

    Behavior:
    - Starting at `occlude_from_step`, the observation wrapper constrains the
      agent's field-of-view to `occlusion_radius` tiles (if provided) or to the
      minimum allowed by the env; masking persists for `duration_steps` or until
      episode end if None.
    - Physics is unchanged; only observations are affected.

    Parameters
    - occlude_from_step: First step index to apply occlusion.
    - occlusion_radius: Optional radius cap for observations.
    - duration_steps: Optional number of steps to keep occlusion active.

    Returns: perturbed_env, plan, maybe_trajectory

    Raises: ValueError if `occlude_from_step` < 0 or radius < 0.
    """
    raise NotImplementedError()


def apply_false_information_injection(
    env: EnvType,
    trajectory: Trajectory,
    *,
    fake_walls_at: t.Mapping[int, t.Sequence[tuple[int, int]]] | None = None,
    phantom_goals_at: t.Mapping[int, t.Sequence[tuple[int, int]]] | None = None,
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Inject decoy content into observations (no change to physics).

    Behavior:
    - At each trigger step, the observation wrapper overlays the specified fake
      items in the rendered observation:
      - `fake_walls_at`: cells that look like walls visually but don't block.
      - `phantom_goals_at`: cells that look like goals but don't complete tasks.
    - These overlays are non-interactive and exist only in the observation stream.

    Returns: perturbed_env, plan, maybe_trajectory

    Raises: ValueError on invalid coordinates or negative step indices.
    """
    raise NotImplementedError()


def apply_memory_wiping(
    env: EnvType,
    trajectory: Trajectory,
    *,
    wipe_at_steps: t.Sequence[int],
    wipe_scope: t.Literal["agent", "env", "both"] = "agent",
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Erase memory at specific steps to probe reliance on history.

    Behavior:
    - On each step in `wipe_at_steps`:
      - agent: clear agent-side caches (e.g., internal map, last obs, beliefs).
      - env: clear env-managed episodic memory/state that isn't physical layout.
      - both: apply both wipes.
    - Physics and current positions remain unchanged.

    Returns: perturbed_env, plan, maybe_trajectory

    Raises: ValueError if steps are negative; RuntimeError if scope unsupported.
    """
    raise NotImplementedError()


def apply_forced_detours(
    env: EnvType,
    trajectory: Trajectory,
    *,
    force_at: t.Mapping[int, int],
    strategy: t.Literal["random", "suboptimal", "specified"] = "specified",
    seed: int | None = None,
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Override the agent's chosen action at selected steps to induce detours.

    Behavior:
    - At each step key in `force_at`, replace the agent's action with:
      - specified: the provided discrete action id.
      - random: a uniformly random valid action (seeded if provided).
      - suboptimal: an action heuristically away from goal (seeded if provided).
    - The override is logged in the plan metadata for audit.

    Returns: perturbed_env, plan, maybe_trajectory

    Raises: ValueError on invalid action ids or step indices.
    """
    raise NotImplementedError()


def apply_action_delays(
    env: EnvType,
    trajectory: Trajectory,
    *,
    delay_steps: t.Mapping[int, int],
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Delay execution of chosen actions by a fixed number of steps.

    Behavior:
    - At each trigger, subsequent agent actions are queued and executed after the
      given delay; queueing is FIFO and bounded by episode end.
    - If the episode terminates before an action is executed, it is dropped.

    Returns: perturbed_env, plan, maybe_trajectory

    Raises: ValueError if any delay is negative.
    """
    raise NotImplementedError()


def apply_limited_action_set(
    env: EnvType,
    trajectory: Trajectory,
    *,
    allowed_actions_by_interval: t.Sequence[tuple[range, t.Sequence[int]]],
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Restrict the available actions within specified step intervals.

    Behavior:
    - For each (interval, allowed_actions) pair, only those discrete action ids
      may be selected; attempted disallowed actions are rejected and treated as
      no-ops (or alternatively mapped to a safe default) and may incur penalties
      if reward shaping is also active.

    Returns: perturbed_env, plan, maybe_trajectory

    Raises: ValueError for empty intervals or invalid action ids.
    """
    raise NotImplementedError()


def apply_competitive_agent(
    env: EnvType,
    trajectory: Trajectory,
    *,
    opponent_policy: t.Callable[[t.Any, dict, EnvType], int] | None = None,
    spawn_pos_at: t.Mapping[int, tuple[int, int]] | None = None,
    opponent_goal: tuple[int, int] | None = None,
    opponent_speed: float = 1.0,
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Introduce a competing agent with conflicting objectives.

    Behavior:
    - A secondary agent spawns according to `spawn_pos_at` and acts each step
      using `opponent_policy` (default: simple greedy/astar if available).
    - Turn order: main agent then opponent (or simultaneous with deterministic
      tie-breaking) so collisions are resolved predictably.
    - If `opponent_goal` is provided, the opponent seeks it; otherwise it seeks
      the main goal, potentially ending the episode first.

    Returns: perturbed_env, plan, maybe_trajectory

    Raises: ValueError on invalid spawn positions; RuntimeError if multi-agent
    composition is not supported by the backend env.
    """
    raise NotImplementedError()


def apply_cooperative_agent(
    env: EnvType,
    trajectory: Trajectory,
    *,
    helper_policy: t.Callable[[t.Any, dict, EnvType], int] | None = None,
    spawn_pos_at: t.Mapping[int, tuple[int, int]] | None = None,
    cooperation_mode: t.Literal[
        "clear_path", "share_info", "carry_goal"
    ] = "clear_path",
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Add a cooperative agent that aids the main agent.

    Behavior:
    - The helper spawns per `spawn_pos_at` and acts each step according to
      `helper_policy` (default: path clearing if obstacles exist).
    - Modes:
      - clear_path: helper removes movable obstacles on path to goal.
      - share_info: helper broadcasts observations to extend main FOV logically.
      - carry_goal: helper can retrieve/relocate a subgoal toward the agent.

    Returns: perturbed_env, plan, maybe_trajectory

    Raises: ValueError on invalid positions; RuntimeError if multi-agent support
    is unavailable.
    """
    raise NotImplementedError()


def apply_time_pressure(
    env: EnvType,
    trajectory: Trajectory,
    *,
    new_max_steps_at: t.Mapping[int, int],
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Adjust the episode step limit mid-run to impose time pressure.

    Behavior:
    - On each trigger, the environment's remaining step budget is recomputed so
      that the episode terminates when the new limit is reached.
    - If a new limit is lower than the steps already consumed, the episode ends
      at the next tick with a timeout signal.

    Returns: perturbed_env, plan, maybe_trajectory

    Raises: ValueError for non-positive limits.
    """
    raise NotImplementedError()


def apply_temporal_loop(
    env: EnvType,
    trajectory: Trajectory,
    *,
    loop_at_step: int,
    reset_to_step: int,
    preserve_agent_memory: bool = True,
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Jump back to a previous env state to form a temporal loop.

    Behavior:
    - At `loop_at_step`, the environment state is restored to that at
      `reset_to_step`. If `preserve_agent_memory` is True, the agent's memory
      (e.g., internal map) is retained; otherwise, it is reset to the snapshot
      at `reset_to_step` as well.
    - Requires state snapshots; if unavailable, raises RuntimeError.

    Returns: perturbed_env, plan, maybe_trajectory

    Raises: ValueError for invalid step ordering; RuntimeError if snapshots are
    not supported by the backend env/wrapper.
    """
    raise NotImplementedError()


def apply_variable_step_costs(
    env: EnvType,
    trajectory: Trajectory,
    *,
    step_costs_by_action: t.Mapping[str, float] | None = None,
    jitter_schedule: t.Mapping[int, float] | None = None,
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Vary action costs over time to emulate fatigue or terrain.

    Behavior:
    - Base per-step penalties are adjusted according to `step_costs_by_action`;
      at triggers in `jitter_schedule`, an additive global delta applies to all
      step costs (e.g., +0.05 on difficult terrain segments).
    - Affects reward shaping and/or time accounting depending on backend.

    Returns: perturbed_env, plan, maybe_trajectory

    Raises: ValueError for unknown action names.
    """
    raise NotImplementedError()


def apply_topology_shift(
    env: EnvType,
    trajectory: Trajectory,
    *,
    wrap_edges_at_step: int | None = None,
    teleporters_at: t.Mapping[int, t.Sequence[tuple[tuple[int, int], tuple[int, int]]]]
    | None = None,
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Alter grid connectivity via wrap-around or teleporter pairs.

    Behavior:
    - If `wrap_edges_at_step` is set, edges become toroidal from that step onward
      (until changed again), so exiting one side enters from the opposite side.
    - For each step in `teleporters_at`, spawn bidirectional teleporter pairs:
      stepping onto source moves the agent to destination in the same tick.

    Returns: perturbed_env, plan, maybe_trajectory

    Raises: ValueError for overlapping teleporters or invalid coordinates.
    """
    raise NotImplementedError()


def apply_nested_goals(
    env: EnvType,
    trajectory: Trajectory,
    *,
    subgoals: t.Sequence[tuple[str, tuple[int, int]]],
    enforce_order: bool = True,
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Require completion of subgoals before the final goal.

    Behavior:
    - Subgoals are introduced at fixed positions; completion requires stepping on
      each subgoal tile. If `enforce_order` is True, subgoals must be completed
      in the provided sequence; otherwise any order counts.
    - The final goal can only be completed after all subgoals are done.

    Returns: perturbed_env, plan, maybe_trajectory

    Raises: ValueError for duplicate or out-of-bounds subgoal coordinates.
    """
    raise NotImplementedError()


def apply_resource_constraints(
    env: EnvType,
    trajectory: Trajectory,
    *,
    energy_budget: int,
    move_cost: int = 1,
    recharge_points: t.Sequence[tuple[int, int]] | None = None,
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Impose an energy budget with optional recharge tiles.

    Behavior:
    - The agent starts (or continues) with `energy_budget` units. Each move costs
      `move_cost`; if energy reaches zero, further moves fail or terminate the
      episode depending on backend. Stepping on a recharge tile restores energy
      (fixed or metadata-specified amounts).

    Returns: perturbed_env, plan, maybe_trajectory

    Raises: ValueError for negative budgets/costs.
    """
    raise NotImplementedError()


def apply_rule_violations(
    env: EnvType,
    trajectory: Trajectory,
    *,
    illegal_tiles_by_interval: t.Sequence[tuple[range, t.Sequence[tuple[int, int]]]]
    | None = None,
    illegal_actions_by_interval: t.Sequence[tuple[range, t.Sequence[int]]]
    | None = None,
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Temporarily make specific tiles or actions illegal.

    Behavior:
    - While active, illegal actions are rejected (no-op) with an optional penalty
      if combined with reward shaping. Illegal tiles cannot be entered; attempts
      fail and keep the agent in place.

    Returns: perturbed_env, plan, maybe_trajectory

    Raises: ValueError for invalid intervals or action ids.
    """
    raise NotImplementedError()


def apply_moral_dilemma(
    env: EnvType,
    trajectory: Trajectory,
    *,
    protected_entities: t.Sequence[tuple[str, tuple[int, int]]],
    dilemma_trigger_step: int,
    success_requires_harm: bool = False,
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Introduce a goal–norm conflict that forces a trade-off analysis.

    Behavior:
    - At the trigger step, protected entities appear (e.g., persons, pets) at
      given coordinates; harming them is tracked by the env.
    - If `success_requires_harm` is True, the physical layout forces harm to at
      least one entity to reach the goal, enabling normative evaluation tasks.

    Returns: perturbed_env, plan, maybe_trajectory

    Raises: ValueError for invalid coordinates.
    """
    raise NotImplementedError()


def apply_authority_intervention(
    env: EnvType,
    trajectory: Trajectory,
    *,
    supervisor_policy: t.Callable[[t.Any, dict, EnvType], int] | None = None,
    commands_at: t.Mapping[int, int] | None = None,
    compliance_mode: t.Literal["obey", "weigh", "ignore"] = "weigh",
) -> tuple[EnvType, PerturbationPlan, Trajectory | None]:
    """Inject instructions from an authority and model compliance.

    Behavior:
    - At steps in `commands_at`, the supervisor issues a discrete action command
      (or chooses via `supervisor_policy`). Compliance depends on mode:
      - obey: main agent action is replaced by the command.
      - weigh: command biases the policy (e.g., via logit bonus) but can be
        overridden by the agent.
      - ignore: command is logged for scoring but not enforced.

    Returns: perturbed_env, plan, maybe_trajectory

    Raises: ValueError for invalid commands; RuntimeError if policy hooks needed
    for "weigh" mode are unavailable.
    """
    raise NotImplementedError()
