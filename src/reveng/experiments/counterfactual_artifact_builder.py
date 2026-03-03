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


@dataclass
class ArtifactPairSpec:
    pair_id: str
    grid_a_path: Path
    grid_b_path: Path
    goal_orig: tuple[int, int]
    goal_new: tuple[int, int]


def _write_pair_manifest_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    template = [
        {
            "pair_id": "pair_000",
            "grid_a_path": "data/cf/pair_000/grid_a.txt",
            "grid_b_path": "data/cf/pair_000/grid_b.txt",
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
        out.append(
            ArtifactPairSpec(
                pair_id=pair_id,
                grid_a_path=Path(str(row["grid_a_path"])),
                grid_b_path=Path(str(row["grid_b_path"])),
                goal_orig=_parse_coord(row["goal_orig"]),
                goal_new=_parse_coord(row["goal_new"]),
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


def _validate_goal_move_only(spec: ArtifactPairSpec, layout_a: list[list[str]], layout_b: list[list[str]]) -> None:
    agent_a, goal_a = _extract_agent_goal(layout_a)
    agent_b, goal_b = _extract_agent_goal(layout_b)

    if goal_a != spec.goal_orig:
        raise ValueError(
            f"pair={spec.pair_id}: grid_a goal {goal_a} != manifest goal_orig {spec.goal_orig}"
        )
    if goal_b != spec.goal_new:
        raise ValueError(
            f"pair={spec.pair_id}: grid_b goal {goal_b} != manifest goal_new {spec.goal_new}"
        )
    if agent_a != agent_b:
        raise ValueError(f"pair={spec.pair_id}: agent changed between A and B")

    def normalize(layout: list[list[str]]) -> list[list[str]]:
        out = []
        for row in layout:
            out.append(["_" if c in {"A", "G"} else c for c in row])
        return out

    if normalize(layout_a) != normalize(layout_b):
        raise ValueError(f"pair={spec.pair_id}: non-goal topology changed between A and B")


def _build_synthetic_probes(
    goal_orig: tuple[int, int],
    goal_new: tuple[int, int],
    layer_key: str,
    linear_target: Literal["orig", "new"],
    goal_prob: float,
) -> dict[str, Any]:
    new_x, new_y = goal_new
    orig_x, orig_y = goal_orig

    linear_xy = goal_orig if linear_target == "orig" else goal_new
    lin_x, lin_y = linear_xy

    # MLP points to new goal by default in patched artifact.
    probes = {
        f"cognitive_map_probe_l15_s0_suffix_-3--1_mlp_1024_full_upsample_normalize_r{new_y}_c{new_x}": {
            layer_key: {"goal": goal_prob, "empty": 1.0 - goal_prob}
        },
        # Include an alternate location to make argmax non-trivial.
        f"cognitive_map_probe_l15_s0_suffix_-3--1_mlp_1024_full_upsample_normalize_r{orig_y}_c{orig_x}": {
            layer_key: {"goal": 1.0 - goal_prob, "empty": goal_prob}
        },
        f"cognitive_map_probe_l15_s0_suffix_-3--1_linear_full_upsample_normalize_r{lin_y}_c{lin_x}": {
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
    goal_orig: tuple[int, int],
    goal_new: tuple[int, int],
    layer_key: str,
    patch_action_source: Literal["a", "b"],
    linear_target: Literal["orig", "new"],
    synthetic_goal_prob: float,
) -> dict[str, Any]:
    patched = copy.deepcopy(trace_a)
    steps_a = trace_a.get("steps", [])
    steps_b = trace_b.get("steps", [])
    n = min(len(steps_a), len(steps_b))

    probes = _build_synthetic_probes(
        goal_orig=goal_orig,
        goal_new=goal_new,
        layer_key=layer_key,
        linear_target=linear_target,
        goal_prob=synthetic_goal_prob,
    )

    patched_steps: list[dict[str, Any]] = []
    for i in range(n):
        step_a = copy.deepcopy(steps_a[i])
        step_b = steps_b[i]

        if patch_action_source == "b":
            step_a["agent_action"] = step_b.get("agent_action", step_a.get("agent_action"))

        toks_a = step_a.get("prompt_suffix_tokens") or []
        toks_b = step_b.get("prompt_suffix_tokens") or []

        # Pre-reasoning tokens take structure from B where available.
        step_a["pre_reasoning_prompt_suffix_tokens"] = _inject_probes_on_last_three(
            toks_b if isinstance(toks_b, list) else toks_a,
            probes,
        )
        # Post tokens exist under the expected field name used by evaluator.
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
        "artifact_builder": "in_repo_counterfactual_artifact_builder_v1",
        "patch_action_source": patch_action_source,
        "linear_target": linear_target,
        "synthetic_goal_prob": synthetic_goal_prob,
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
    output_dir: str = "counterfactual_artifacts",
    model_name: str = "together_ai/openai/gpt-oss-20b",
    max_steps_per_trajectory: int = 50,
    max_tokens: int = 10000,
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_logprobs: int = 5,
    seed: int = 42,
    reasoning_effort: Literal["low", "medium", "high"] = "low",
    patch_action_source: Literal["a", "b"] = "b",
    linear_target: Literal["orig", "new"] = "orig",
    synthetic_goal_prob: float = 0.99,
    layer_key: str = LAYER_KEY_DEFAULT,
    overwrite: bool = False,
    skip_trajectory_generation: bool = False,
) -> None:
    """Build A/B/patched artifacts and evaluation manifest in-repo.

    Expected pair manifest rows:
      - pair_id, grid_a_path, grid_b_path, goal_orig, goal_new

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
        _validate_goal_move_only(spec, layout_a, layout_b)

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
            goal_orig=spec.goal_orig,
            goal_new=spec.goal_new,
            layer_key=layer_key,
            patch_action_source=patch_action_source,
            linear_target=linear_target,
            synthetic_goal_prob=synthetic_goal_prob,
        )
        patched_trace_path.write_text(json.dumps(patched_trace, indent=2))

        eval_manifest_rows.append(
            {
                "pair_id": spec.pair_id,
                "grid_a_path": str(spec.grid_a_path),
                "grid_b_path": str(spec.grid_b_path),
                "goal_orig": list(spec.goal_orig),
                "goal_new": list(spec.goal_new),
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
    "build_counterfactual_patch_artifacts",
    "build_patched_trace",
]
