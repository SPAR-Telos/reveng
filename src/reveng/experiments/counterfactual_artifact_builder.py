"""Build counterfactual activation-patching artifacts entirely in-repo.

This command generates A/B trajectories for grid pairs and creates a patched trace
artifact that is compatible with `counterfactual_activation_patching` evaluation.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

LAYER_KEY_DEFAULT = "model.layers.15.output"
COUNTERFACTUAL_CATEGORIES = (
    "goal_move",
    "start_goal_swap",
    "rotate_90",
    "reflect_vertical",
    "transpose",
)

_ACTION_NAME_TO_ID = {"LEFT": 0, "RIGHT": 1, "UP": 2, "DOWN": 3}
_ACTION_ID_TO_NAME = {v: k for k, v in _ACTION_NAME_TO_ID.items()}


@dataclass
class ArtifactPairSpec:
    pair_id: str
    grid_a_path: Path
    grid_b_path: Path
    goal_a: tuple[int, int]
    goal_b: tuple[int, int]
    category: str = "goal_move"


def normalize_counterfactual_category(category: str) -> str:
    normalized = str(category).strip().lower().replace("-", "_")
    aliases = {
        "goal": "goal_move",
        "goal_move_only": "goal_move",
        "goalmove": "goal_move",
        "swap": "start_goal_swap",
        "start_goal": "start_goal_swap",
        "rotate": "rotate_90",
        "rotate90": "rotate_90",
        "reflect": "reflect_vertical",
        "reflection": "reflect_vertical",
        "transpose_grid": "transpose",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in COUNTERFACTUAL_CATEGORIES:
        raise ValueError(
            f"Unknown counterfactual category '{category}'. "
            f"Allowed: {', '.join(COUNTERFACTUAL_CATEGORIES)}"
        )
    return normalized


def _write_pair_manifest_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    template = [
        {
            "pair_id": "pair_goal_move_000",
            "category": "goal_move",
            "grid_a_path": "data/cf/pair_goal_move_000/grid_a.txt",
            "grid_b_path": "data/cf/pair_goal_move_000/grid_b.txt",
            "goal_a": [5, 5],
            "goal_b": [5, 1],
            "goal_orig": [5, 5],
            "goal_new": [5, 1],
        }
    ]
    path.write_text(json.dumps(template, indent=2))


def _parse_coord(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, str):
        stripped = value.strip()
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                return int(parsed[0]), int(parsed[1])
        except Exception:
            pass
        cleaned = stripped.strip("[]()")
        parts = [p.strip() for p in cleaned.split(",")]
        if len(parts) == 2 and all(parts):
            return int(parts[0]), int(parts[1])
    raise ValueError(f"Invalid coordinate format: {value}")


def _action_to_id(action: Any) -> Optional[int]:
    if isinstance(action, int) and action in {0, 1, 2, 3}:
        return action
    if isinstance(action, str):
        cleaned = action.strip().upper()
        if cleaned in _ACTION_NAME_TO_ID:
            return _ACTION_NAME_TO_ID[cleaned]
        if cleaned.isdigit() and int(cleaned) in {0, 1, 2, 3}:
            return int(cleaned)
    return None


def _format_action_like(source_action: Any, action_id: int) -> Any:
    if isinstance(source_action, int):
        return action_id
    return _ACTION_ID_TO_NAME[action_id]


def map_action_a_to_b(action_id: int, category: str) -> int:
    category = normalize_counterfactual_category(category)
    maps = {
        "goal_move": {0: 0, 1: 1, 2: 2, 3: 3},
        "start_goal_swap": {0: 0, 1: 1, 2: 2, 3: 3},
        "rotate_90": {0: 3, 1: 2, 2: 0, 3: 1},
        "reflect_vertical": {0: 1, 1: 0, 2: 2, 3: 3},
        "transpose": {0: 2, 1: 3, 2: 0, 3: 1},
    }
    return maps[category][action_id]


def map_action_b_to_a(action_id: int, category: str) -> int:
    category = normalize_counterfactual_category(category)
    forward = {v: k for k, v in {k: map_action_a_to_b(k, category) for k in [0, 1, 2, 3]}.items()}
    return forward[action_id]


def map_position_a_to_b(
    pos: tuple[int, int],
    category: str,
    width: int,
    height: int,
) -> tuple[int, int]:
    category = normalize_counterfactual_category(category)
    x, y = pos
    if category in {"goal_move", "start_goal_swap"}:
        return x, y
    if category == "rotate_90":
        return y, width - 1 - x
    if category == "reflect_vertical":
        return width - 1 - x, y
    if category == "transpose":
        return y, x
    raise ValueError(f"Unsupported category for position mapping: {category}")


def _read_pair_manifest(path: Path) -> list[ArtifactPairSpec]:
    if not path.exists():
        _write_pair_manifest_template(path)
        raise FileNotFoundError(
            f"Pair manifest not found: {path}. "
            f"Created template file at this path; fill it and rerun."
        )

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "pairs" in data:
            rows = data["pairs"]
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError("JSON pair manifest must be a list or {'pairs': [...]}.")
    else:
        raise ValueError("Pair manifest must be JSON in v1")

    out: list[ArtifactPairSpec] = []
    for idx, row in enumerate(rows):
        pair_id = str(row.get("pair_id", f"pair_{idx:03d}"))
        goal_a_raw = row.get("goal_a", row.get("goal_orig"))
        goal_b_raw = row.get("goal_b", row.get("goal_new"))
        if goal_a_raw is None or goal_b_raw is None:
            raise ValueError(
                f"pair={pair_id}: manifest row requires goal_a/goal_b "
                "(legacy goal_orig/goal_new are accepted)."
            )

        category = normalize_counterfactual_category(row.get("category", "goal_move"))
        out.append(
            ArtifactPairSpec(
                pair_id=pair_id,
                category=category,
                grid_a_path=Path(str(row["grid_a_path"])),
                grid_b_path=Path(str(row["grid_b_path"])),
                goal_a=_parse_coord(goal_a_raw),
                goal_b=_parse_coord(goal_b_raw),
            )
        )
    return out


def _parse_grid_text_file(path: Path) -> list[list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Grid file not found: {path}")

    lines = path.read_text().splitlines()
    rows: list[list[str]] = []
    saw_header = False
    allowed_cells = {"#", "_", "A", "G"}

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if all(p.lstrip("-").isdigit() for p in parts):
            saw_header = True
            continue
        if not parts[0].lstrip("-").isdigit() or len(parts) <= 1:
            raise ValueError(f"Invalid row in grid file: {path}")
        cells = parts[1:]
        if any(c not in allowed_cells for c in cells):
            raise ValueError(f"Invalid cell symbol in grid file: {path}")
        rows.append(cells)

    if not saw_header or not rows:
        raise ValueError(f"Unable to parse grid file: {path}")

    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError(f"Inconsistent row widths in grid file: {path}")

    return rows


def _extract_agent_goal(layout: list[list[str]]) -> tuple[tuple[int, int], tuple[int, int]]:
    agent = None
    goal = None
    for y, row in enumerate(layout):
        for x, cell in enumerate(row):
            if cell == "A":
                agent = (x, y)
            elif cell == "G":
                goal = (x, y)
    if agent is None or goal is None:
        raise ValueError("Grid must contain one A and one G")
    return agent, goal


def _normalize_layout_for_topology(layout: list[list[str]]) -> list[list[str]]:
    out = []
    for row in layout:
        out.append(["_" if c in {"A", "G"} else c for c in row])
    return out


def _layout_transform(layout: list[list[str]], category: str) -> list[list[str]]:
    category = normalize_counterfactual_category(category)
    if category == "rotate_90":
        # 90° CCW
        h = len(layout)
        w = len(layout[0]) if h else 0
        return [[layout[x][w - 1 - y] for x in range(h)] for y in range(w)]
    if category == "reflect_vertical":
        return [list(reversed(row)) for row in layout]
    if category == "transpose":
        h = len(layout)
        w = len(layout[0]) if h else 0
        return [[layout[y][x] for y in range(h)] for x in range(w)]
    raise ValueError(f"Unsupported transform category: {category}")


def _build_start_goal_swap_layout(layout_a: list[list[str]]) -> list[list[str]]:
    layout_b = copy.deepcopy(layout_a)
    agent_a, goal_a = _extract_agent_goal(layout_a)
    layout_b[agent_a[1]][agent_a[0]] = "G"
    layout_b[goal_a[1]][goal_a[0]] = "A"
    return layout_b


def validate_counterfactual_pair(
    spec: ArtifactPairSpec,
    layout_a: list[list[str]],
    layout_b: list[list[str]],
) -> None:
    category = normalize_counterfactual_category(spec.category)

    if len(layout_a) != len(layout_b) or len(layout_a[0]) != len(layout_b[0]):
        if category not in {"transpose"}:
            raise ValueError(f"pair={spec.pair_id}: grid dimensions changed unexpectedly")

    agent_a, goal_a = _extract_agent_goal(layout_a)
    agent_b, goal_b = _extract_agent_goal(layout_b)

    if goal_a != spec.goal_a:
        raise ValueError(
            f"pair={spec.pair_id}: grid_a goal {goal_a} != manifest goal_a {spec.goal_a}"
        )
    if goal_b != spec.goal_b:
        raise ValueError(
            f"pair={spec.pair_id}: grid_b goal {goal_b} != manifest goal_b {spec.goal_b}"
        )

    if category == "goal_move":
        if spec.goal_a == spec.goal_b:
            raise ValueError(f"pair={spec.pair_id}: goal_move requires goal change")
        if agent_a != agent_b:
            raise ValueError(f"pair={spec.pair_id}: agent changed between A and B")
        if _normalize_layout_for_topology(layout_a) != _normalize_layout_for_topology(layout_b):
            raise ValueError(f"pair={spec.pair_id}: non-goal topology changed between A and B")
        return

    if category == "start_goal_swap":
        if _normalize_layout_for_topology(layout_a) != _normalize_layout_for_topology(layout_b):
            raise ValueError(f"pair={spec.pair_id}: non-goal topology changed between A and B")
        if agent_b != goal_a or goal_b != agent_a:
            raise ValueError(
                f"pair={spec.pair_id}: start_goal_swap requires A.agent==B.goal and A.goal==B.agent"
            )
        return

    if category in {"rotate_90", "reflect_vertical", "transpose"}:
        expected = _layout_transform(layout_a, category)
        if expected != layout_b:
            raise ValueError(
                f"pair={spec.pair_id}: grid_b does not match deterministic {category} transform of grid_a"
            )
        return

    raise ValueError(f"pair={spec.pair_id}: unsupported category {category}")


def _build_synthetic_probes(
    goal_a: tuple[int, int],
    goal_b: tuple[int, int],
    layer_key: str,
    linear_target: Literal["a", "b"],
    goal_prob: float,
) -> dict[str, Any]:
    bx, by = goal_b
    ax, ay = goal_a

    linear_xy = goal_a if linear_target == "a" else goal_b
    lx, ly = linear_xy

    probes = {
        f"cognitive_map_probe_l15_s0_suffix_-3--1_mlp_1024_full_upsample_normalize_r{by}_c{bx}": {
            layer_key: {"goal": goal_prob, "empty": 1.0 - goal_prob}
        },
        f"cognitive_map_probe_l15_s0_suffix_-3--1_mlp_1024_full_upsample_normalize_r{ay}_c{ax}": {
            layer_key: {"goal": 1.0 - goal_prob, "empty": goal_prob}
        },
        f"cognitive_map_probe_l15_s0_suffix_-3--1_linear_full_upsample_normalize_r{ly}_c{lx}": {
            layer_key: {"goal": goal_prob, "empty": 1.0 - goal_prob}
        },
    }
    return probes


def _inject_probes_on_last_three(tokens: list[dict[str, Any]], probes: dict[str, Any]) -> list[dict[str, Any]]:
    out = copy.deepcopy(tokens)
    if len(out) < 3:
        out.extend(
            [
                {"token": "<|end|>", "token_groups": ["template"]},
                {"token": "<|start|>", "token_groups": ["template"]},
                {"token": "assistant", "token_groups": ["template"]},
            ][: 3 - len(out)]
        )
    for tok in out[-3:]:
        if not isinstance(tok, dict):
            continue
        tok["probes"] = copy.deepcopy(probes)
    return out


def build_patched_trace(
    trace_a: dict[str, Any],
    trace_b: dict[str, Any],
    goal_a: tuple[int, int],
    goal_b: tuple[int, int],
    category: str,
    layer_key: str,
    patch_action_source: Literal["a", "b"],
    linear_target: Literal["a", "b", "orig", "new"],
    synthetic_goal_prob: float,
) -> dict[str, Any]:
    category = normalize_counterfactual_category(category)
    linear_target = "a" if linear_target == "orig" else ("b" if linear_target == "new" else linear_target)
    if linear_target not in {"a", "b"}:
        raise ValueError("linear_target must be one of: a, b, orig, new")

    patched = copy.deepcopy(trace_a)
    steps_a = trace_a.get("steps", [])
    steps_b = trace_b.get("steps", [])
    n = min(len(steps_a), len(steps_b))

    probes = _build_synthetic_probes(
        goal_a=goal_a,
        goal_b=goal_b,
        layer_key=layer_key,
        linear_target=linear_target,
        goal_prob=synthetic_goal_prob,
    )

    patched_steps: list[dict[str, Any]] = []
    for i in range(n):
        step_a = copy.deepcopy(steps_a[i])
        step_b = steps_b[i]

        if patch_action_source == "b":
            action_b = step_b.get("agent_action", step_a.get("agent_action"))
            action_b_id = _action_to_id(action_b)
            if action_b_id is not None:
                action_a_id = map_action_b_to_a(action_b_id, category)
                step_a["agent_action"] = _format_action_like(action_b, action_a_id)
            else:
                step_a["agent_action"] = action_b

        toks_a = step_a.get("prompt_suffix_tokens") or []
        toks_b = step_b.get("prompt_suffix_tokens") or []

        step_a["pre_reasoning_prompt_suffix_tokens"] = _inject_probes_on_last_three(
            toks_b if isinstance(toks_b, list) else toks_a,
            probes,
        )
        step_a["prompt_suffix_tokens"] = _inject_probes_on_last_three(
            toks_b if isinstance(toks_b, list) else toks_a,
            probes,
        )

        patched_steps.append(step_a)

    patched["steps"] = patched_steps
    patched["patch_metadata"] = {
        "pre_reasoning_last_n": 3,
        "post_reasoning_last_n": 3,
        "hook_tensor": layer_key,
        "artifact_builder": "in_repo_counterfactual_artifact_builder_v2",
        "patch_action_source": patch_action_source,
        "linear_target": linear_target,
        "synthetic_goal_prob": synthetic_goal_prob,
        "category": category,
        "action_frame": "A",
    }
    return patched


def _generate_trajectory_for_grid(
    grid_layout: list[list[str]],
    model_name: str,
    output_path: Path,
    max_steps_per_trajectory: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_logprobs: int,
    seed: int,
    reasoning_effort: Literal["low", "medium", "high"],
) -> None:
    # Lazy imports keep this module usable without full runtime deps during unit tests.
    from reveng.commands.get_trajectory.get_trajectory_fn import get_trajectory
    from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
    from reveng.environment_generator.wrappers.text_obs_wrapper import (
        FullObservabilityTextWrapper,
    )

    size = max(len(grid_layout), len(grid_layout[0]) if grid_layout else 0)
    env = FullObservabilityTextWrapper(Simple2DNavigationEnv(size=size, complexity=0.0))
    env.unwrapped.set_env_from_list(grid_layout)

    get_trajectory(
        model_name=model_name,
        output_path=str(output_path),
        env=env,
        use_safe_reset=True,
        max_steps_per_trajectory=max_steps_per_trajectory,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_logprobs=top_logprobs,
        seed=seed,
        reasoning_effort=reasoning_effort,
    )


def build_counterfactual_patch_artifacts(
    pair_manifest_path: str,
    output_dir: str = "data/cf/artifacts",
    model_name: str = "together_ai/openai/gpt-oss-20b",
    max_steps_per_trajectory: int = 50,
    max_tokens: int = 10000,
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_logprobs: int = 5,
    seed: int = 42,
    reasoning_effort: Literal["low", "medium", "high"] = "low",
    patch_action_source: Literal["a", "b"] = "b",
    linear_target: Literal["a", "b", "orig", "new"] = "a",
    synthetic_goal_prob: float = 0.99,
    layer_key: str = LAYER_KEY_DEFAULT,
    overwrite: bool = False,
    skip_trajectory_generation: bool = False,
) -> None:
    """Build A/B/patched artifacts and evaluation manifest in-repo.

    Expected pair manifest rows:
      - pair_id, category, grid_a_path, grid_b_path, goal_a, goal_b
      - legacy keys goal_orig/goal_new are accepted.

    Produces:
      - {output_dir}/{pair_id}/A.json
      - {output_dir}/{pair_id}/B.json
      - {output_dir}/{pair_id}/patched.json
      - {output_dir}/manifest_for_counterfactual_activation_patching.json
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pair_specs = _read_pair_manifest(Path(pair_manifest_path))

    eval_manifest_rows: list[dict[str, Any]] = []

    for spec in pair_specs:
        layout_a = _parse_grid_text_file(spec.grid_a_path)
        layout_b = _parse_grid_text_file(spec.grid_b_path)
        validate_counterfactual_pair(spec, layout_a, layout_b)

        pair_dir = out_root / spec.pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)

        a_trace_path = pair_dir / "A.json"
        b_trace_path = pair_dir / "B.json"
        patched_trace_path = pair_dir / "patched.json"

        if not skip_trajectory_generation:
            if overwrite or not a_trace_path.exists():
                _generate_trajectory_for_grid(
                    grid_layout=layout_a,
                    model_name=model_name,
                    output_path=a_trace_path,
                    max_steps_per_trajectory=max_steps_per_trajectory,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_logprobs=top_logprobs,
                    seed=seed,
                    reasoning_effort=reasoning_effort,
                )
            if overwrite or not b_trace_path.exists():
                _generate_trajectory_for_grid(
                    grid_layout=layout_b,
                    model_name=model_name,
                    output_path=b_trace_path,
                    max_steps_per_trajectory=max_steps_per_trajectory,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_logprobs=top_logprobs,
                    seed=seed,
                    reasoning_effort=reasoning_effort,
                )

        if not a_trace_path.exists() or not b_trace_path.exists():
            raise FileNotFoundError(
                f"Missing A/B traces for pair={spec.pair_id}. "
                "Either disable --skip-trajectory-generation or provide traces in output_dir."
            )

        trace_a = json.loads(a_trace_path.read_text())
        trace_b = json.loads(b_trace_path.read_text())

        patched_trace = build_patched_trace(
            trace_a=trace_a,
            trace_b=trace_b,
            goal_a=spec.goal_a,
            goal_b=spec.goal_b,
            category=spec.category,
            layer_key=layer_key,
            patch_action_source=patch_action_source,
            linear_target=linear_target,
            synthetic_goal_prob=synthetic_goal_prob,
        )
        patched_trace_path.write_text(json.dumps(patched_trace, indent=2))

        eval_manifest_rows.append(
            {
                "pair_id": spec.pair_id,
                "category": spec.category,
                "grid_a_path": str(spec.grid_a_path),
                "grid_b_path": str(spec.grid_b_path),
                "goal_a": list(spec.goal_a),
                "goal_b": list(spec.goal_b),
                "goal_orig": list(spec.goal_a),
                "goal_new": list(spec.goal_b),
                "a_trace_path": str(a_trace_path),
                "b_trace_path": str(b_trace_path),
                "patched_trace_path": str(patched_trace_path),
            }
        )

    eval_manifest_path = out_root / "manifest_for_counterfactual_activation_patching.json"
    eval_manifest_path.write_text(json.dumps(eval_manifest_rows, indent=2))
    print(f"Wrote artifact pairs to {out_root}")
    print(f"Wrote evaluation manifest to {eval_manifest_path}")


__all__ = [
    "COUNTERFACTUAL_CATEGORIES",
    "LAYER_KEY_DEFAULT",
    "ArtifactPairSpec",
    "build_counterfactual_patch_artifacts",
    "build_patched_trace",
    "map_action_a_to_b",
    "map_action_b_to_a",
    "map_position_a_to_b",
    "normalize_counterfactual_category",
    "validate_counterfactual_pair",
]
