"""Utilities to auto-generate counterfactual manifest files."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.experiments.counterfactual_artifact_builder import (
    ArtifactPairSpec,
    _extract_agent_goal,
    _parse_coord,
    _parse_grid_text_file,
    _validate_goal_move_only,
)


def _as_relative_or_absolute(path: Path) -> str:
    cwd = Path.cwd().resolve()
    try:
        return str(path.resolve().relative_to(cwd))
    except Exception:
        return str(path.resolve())


def _layout_to_text(layout: list[list[str]]) -> str:
    height = len(layout)
    width = len(layout[0]) if height else 0
    lines = ["  " + " ".join(str(i) for i in range(width))]
    for y, row in enumerate(layout):
        lines.append(f"{y} " + " ".join(row))
    return "\n".join(lines) + "\n"


def _layout_from_env(env: Simple2DNavigationEnv) -> tuple[list[list[str]], tuple[int, int], tuple[int, int]]:
    layout: list[list[str]] = []
    for y in range(env.height):
        row: list[str] = []
        for x in range(env.width):
            cell = env.grid.get(x, y)
            if cell is not None and cell.type == "wall":
                row.append("#")
            else:
                row.append("_")
        layout.append(row)

    agent = tuple(env.agent_pos)
    goal = tuple(env.goal_pos)
    layout[agent[1]][agent[0]] = "A"
    layout[goal[1]][goal[0]] = "G"
    return layout, agent, goal


def _is_reachable(layout: list[list[str]], start: tuple[int, int], goal: tuple[int, int]) -> bool:
    from collections import deque

    width = len(layout[0]) if layout else 0
    height = len(layout)
    queue = deque([start])
    seen = {start}

    def passable(x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        return layout[y][x] in {"_", "A", "G"}

    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in seen or not passable(nx, ny):
                continue
            seen.add((nx, ny))
            queue.append((nx, ny))
    return False


def generate_counterfactual_grid_pairs(
    output_root: str = "data/cf",
    num_pairs: int = 10,
    grid_size: int = 7,
    grid_complexity: float = 0.4,
    pair_dir_prefix: str = "pair_",
    grid_a_filename: str = "grid_a.txt",
    grid_b_filename: str = "grid_b.txt",
    seed: int = 42,
    min_goal_move_manhattan: int = 2,
    max_attempts_per_pair: int = 100,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate reproducible A/B grid-pair files automatically.

    A/B constraints are enforced:
    - identical non-goal topology
    - identical agent position
    - only goal position changes
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    created_pairs: list[str] = []
    for pair_idx in range(num_pairs):
        pair_id = f"{pair_dir_prefix}{pair_idx:03d}"
        pair_dir = root / pair_id
        grid_a_path = pair_dir / grid_a_filename
        grid_b_path = pair_dir / grid_b_filename

        if (grid_a_path.exists() or grid_b_path.exists()) and not overwrite:
            raise FileExistsError(
                f"Grid files already exist for {pair_id}: {grid_a_path}, {grid_b_path}. "
                "Re-run with --overwrite to replace."
            )

        success = False
        last_error = None
        for _ in range(max_attempts_per_pair):
            env_seed = rng.randint(0, 2_147_483_647)
            random.seed(env_seed)

            env = Simple2DNavigationEnv(size=grid_size, complexity=grid_complexity)
            env.reset(seed=env_seed)
            layout_a, agent, goal_orig = _layout_from_env(env)

            # Candidate goals must be open cells, not original goal, and reachable from agent.
            candidates: list[tuple[int, int]] = []
            for y, row in enumerate(layout_a):
                for x, cell in enumerate(row):
                    if cell != "_":
                        continue
                    if abs(x - goal_orig[0]) + abs(y - goal_orig[1]) < min_goal_move_manhattan:
                        continue
                    if _is_reachable(layout_a, agent, (x, y)):
                        candidates.append((x, y))

            if not candidates:
                last_error = "no candidate moved-goal positions"
                continue

            goal_new = rng.choice(candidates)

            layout_b = copy.deepcopy(layout_a)
            layout_b[agent[1]][agent[0]] = "_"
            layout_b[goal_orig[1]][goal_orig[0]] = "_"
            layout_b[goal_new[1]][goal_new[0]] = "G"
            layout_b[agent[1]][agent[0]] = "A"

            spec = ArtifactPairSpec(
                pair_id=pair_id,
                grid_a_path=grid_a_path,
                grid_b_path=grid_b_path,
                goal_orig=goal_orig,
                goal_new=goal_new,
            )
            try:
                _validate_goal_move_only(spec, layout_a, layout_b)
            except Exception as exc:
                last_error = str(exc)
                continue

            pair_dir.mkdir(parents=True, exist_ok=True)
            grid_a_path.write_text(_layout_to_text(layout_a))
            grid_b_path.write_text(_layout_to_text(layout_b))
            created_pairs.append(pair_id)
            success = True
            break

        if not success:
            raise RuntimeError(
                f"Failed to generate valid pair {pair_id} after {max_attempts_per_pair} attempts: {last_error}"
            )

    summary = {
        "status": "ok",
        "output_root": str(root),
        "num_pairs": len(created_pairs),
        "pair_ids": created_pairs,
        "grid_size": grid_size,
        "grid_complexity": grid_complexity,
        "seed": seed,
    }
    print(json.dumps(summary, indent=2))
    return summary


def read_pair_manifest_strict(path: Path) -> list[ArtifactPairSpec]:
    """Read pair manifest without creating templates or mutating disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Pair manifest not found: {path}. "
            "Generate it with `reveng-cli generate_counterfactual_pair_manifest --grids-root data/cf --output-path data/cf/pair_manifest.json`."
        )

    if path.suffix.lower() != ".json":
        raise ValueError("Pair manifest must be JSON.")

    data = json.loads(path.read_text())
    if isinstance(data, dict) and "pairs" in data:
        rows = data["pairs"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError("JSON pair manifest must be a list or {'pairs': [...]} format.")

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


def generate_counterfactual_pair_manifest(
    grids_root: str = "data/cf",
    output_path: str = "data/cf/pair_manifest.json",
    pair_dir_prefix: str = "pair_",
    grid_a_filename: str = "grid_a.txt",
    grid_b_filename: str = "grid_b.txt",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate a pair manifest by scanning grid pair directories.

    Recursively scans under `grids_root` for directories named `{pair_dir_prefix}*`
    that contain `grid_a.txt` and `grid_b.txt`,
    infers goals from grid files, validates goal-move-only topology, and writes a
    JSON manifest compatible with `build_counterfactual_patch_artifacts`.
    """
    root = Path(grids_root)
    out = Path(output_path)

    if not root.exists():
        raise FileNotFoundError(f"grids_root not found: {root}")

    if out.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {out}. Re-run with --overwrite to replace it."
        )

    pair_dirs = sorted(
        p
        for p in root.rglob(f"{pair_dir_prefix}*")
        if p.is_dir() and p.name.startswith(pair_dir_prefix)
    )
    if not pair_dirs:
        raise ValueError(
            f"No pair directories found under {root} with prefix '{pair_dir_prefix}'. "
            "Expected directories like `data/cf/pair_000/` containing `grid_a.txt` and `grid_b.txt`."
        )

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for pair_dir in pair_dirs:
        grid_a_path = pair_dir / grid_a_filename
        grid_b_path = pair_dir / grid_b_filename

        if not grid_a_path.exists() or not grid_b_path.exists():
            skipped.append(pair_dir.name)
            continue

        layout_a = _parse_grid_text_file(grid_a_path)
        layout_b = _parse_grid_text_file(grid_b_path)
        _, goal_orig = _extract_agent_goal(layout_a)
        _, goal_new = _extract_agent_goal(layout_b)

        if goal_orig == goal_new:
            raise ValueError(
                f"pair={pair_dir.name}: goal did not move between {grid_a_filename} and {grid_b_filename}"
            )

        spec = ArtifactPairSpec(
            pair_id=pair_dir.name,
            grid_a_path=grid_a_path,
            grid_b_path=grid_b_path,
            goal_orig=goal_orig,
            goal_new=goal_new,
        )
        _validate_goal_move_only(spec, layout_a, layout_b)

        rows.append(
            {
                "pair_id": pair_dir.name,
                "grid_a_path": _as_relative_or_absolute(grid_a_path),
                "grid_b_path": _as_relative_or_absolute(grid_b_path),
                "goal_orig": list(goal_orig),
                "goal_new": list(goal_new),
            }
        )

    if not rows:
        raise ValueError(
            f"No valid grid pairs found under {root}. Missing {grid_a_filename}/{grid_b_filename} in: {skipped}. "
            f"Create grid files first, then rerun generate_counterfactual_pair_manifest."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    summary = {
        "status": "ok",
        "output_path": str(out),
        "n_pairs": len(rows),
        "skipped_missing_grid_files": skipped,
    }
    print(json.dumps(summary, indent=2))
    return summary


def generate_counterfactual_eval_manifest(
    pair_manifest_path: str = "data/cf/pair_manifest.json",
    artifacts_dir: str = "data/cf/artifacts",
    output_path: str = "data/cf/artifacts/manifest_for_counterfactual_activation_patching.json",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate evaluator manifest from pair manifest plus artifact traces."""
    pair_specs = read_pair_manifest_strict(Path(pair_manifest_path))
    artifacts_root = Path(artifacts_dir)
    out = Path(output_path)

    if out.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {out}. Re-run with --overwrite to replace it."
        )

    rows: list[dict[str, Any]] = []
    for spec in pair_specs:
        pair_dir = artifacts_root / spec.pair_id
        a_trace_path = pair_dir / "A.json"
        b_trace_path = pair_dir / "B.json"
        patched_trace_path = pair_dir / "patched.json"

        missing = [str(p) for p in [a_trace_path, b_trace_path, patched_trace_path] if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing artifact traces for pair={spec.pair_id}: {missing}. "
                "Run build_counterfactual_patch_artifacts first."
            )

        rows.append(
            {
                "pair_id": spec.pair_id,
                "grid_a_path": _as_relative_or_absolute(spec.grid_a_path),
                "grid_b_path": _as_relative_or_absolute(spec.grid_b_path),
                "goal_orig": list(spec.goal_orig),
                "goal_new": list(spec.goal_new),
                "a_trace_path": _as_relative_or_absolute(a_trace_path),
                "b_trace_path": _as_relative_or_absolute(b_trace_path),
                "patched_trace_path": _as_relative_or_absolute(patched_trace_path),
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    summary = {
        "status": "ok",
        "output_path": str(out),
        "n_pairs": len(rows),
    }
    print(json.dumps(summary, indent=2))
    return summary


__all__ = [
    "generate_counterfactual_grid_pairs",
    "generate_counterfactual_pair_manifest",
    "generate_counterfactual_eval_manifest",
    "read_pair_manifest_strict",
]
