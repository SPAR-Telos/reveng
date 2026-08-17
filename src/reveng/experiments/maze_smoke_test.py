"""Portable API smoke test for the regular no-key/no-door maze task.

The module deliberately has no tokenizer or model-weight dependency.  It builds a
fixed grid manifest, runs the same grids through Together chat completions, and
reduces resumable JSONL call records to trajectory- and condition-level results.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv

ACTIONS = ("LEFT", "RIGHT", "UP", "DOWN")
ACTION_TO_INT = {name: index for index, name in enumerate(ACTIONS)}
ACTION_DELTAS = {
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "UP": (0, -1),
    "DOWN": (0, 1),
}

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "seed": 42,
    "provider": "Together AI",
    "pricing_verified_utc": "2026-08-13",
    "pricing_source_url": "https://docs.together.ai/docs/serverless/models",
    "context_mode": "current full grid only (no trajectory history)",
    "grid_sizes": [7, 11, 15],
    "difficulty_levels": [0.0, 0.6, 1.0],
    "grids_per_cell": 2,
    "grid_candidate_pool_size": 32,
    "grid_selection_quantiles": [0.5, 0.9],
    "temperature": 0.7,
    "greedy_temperature": 0.0,
    "top_p": 0.95,
    "max_output_tokens": 16000,
    "step_limit_multiplier": 2.0,
    "step_limit_slack": 8,
    "step_limit_hard_cap": 100,
    "max_api_attempts": 3,
    "models": [
        {
            "name": "GPT-OSS-20B",
            "api_model_id": "openai/gpt-oss-20b",
            "local_checkpoint": "openai/gpt-oss-20b",
            "revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
            "local_model_class": "AutoModelForCausalLM",
            "local_dtype": "bfloat16",
            "reasoning_settings": ["low", "medium"],
            "reasoning_control": True,
            "reasoning_trace_format": "gpt_oss_channels",
            "deployment": "serverless",
            "input_usd_per_million_tokens": 0.05,
            "output_usd_per_million_tokens": 0.20,
        },
        {
            "name": "Gemma-4-31B-IT",
            "api_model_id": "google/gemma-4-31B-it",
            "local_checkpoint": "google/gemma-4-31B-it",
            "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
            "local_model_class": "AutoModelForMultimodalLM",
            "minimum_transformers_version": "5.5.0",
            "local_dtype": "bfloat16",
            "reasoning_settings": ["native"],
            "reasoning_control": False,
            "reasoning_trace_format": "before_action_json",
            "chat_template_kwargs": {"enable_thinking": True},
            "deployment": "serverless",
            "input_usd_per_million_tokens": 0.20,
            "output_usd_per_million_tokens": 0.50,
        },
        {
            "name": "Qwen3-32B",
            "api_model_id": "REPLACE_WITH_TOGETHER_QWEN3_32B_ENDPOINT_NAME",
            "local_checkpoint": "Qwen/Qwen3-32B",
            "revision": "9216db5781bf21249d130ec9da846c4624c16137",
            "local_model_class": "AutoModelForCausalLM",
            "local_dtype": "bfloat16",
            "reasoning_settings": ["native"],
            "reasoning_control": False,
            "reasoning_trace_format": "qwen_think_tags",
            "chat_template_kwargs": {"enable_thinking": True},
            "deployment": "dedicated endpoint required on Together",
            "dedicated_hourly_price_usd": None,
            "input_usd_per_million_tokens": None,
            "output_usd_per_million_tokens": None,
        },
    ],
    "decision_thresholds": {
        "minimum_goal_success_rate": 0.50,
        "maximum_invalid_or_failed_rate": 0.10,
        "medium_material_success_gain": 0.10,
        "token_budget_headroom": 1.20,
    },
    "pricing_note": "Per-token prices are planning values; verify Together pricing before running. Dedicated endpoints require an hourly price entered by the user.",
}


PROMPT_TEMPLATE = """# Instructions

You are controlling an agent in a grid-based environment with full observability. The agent can move in four directions: up, down, left, and right. The environment contains walls, open spaces, and a goal location. The following symbols are used in the grid representation:

Legend:
---------------
#: Wall
_: Open Space (can be visited)
G: Goal
A: Current agent position
----------------

Your objective is to navigate from the current position (A) to the goal (G) while avoiding walls (#). Importantly, the agent should aim to reach the goal using the least amount of steps possible. You must decide your next move based on the provided information.

You will receive the current state of the grid as a NxM matrix of symbols separated by whitespaces, and with coordinates for each row an column. For example, given the 4x5 grid:

  0 1 2 3 4
0 # # # # #
1 # _ _ G #
2 # A _ _ #
3 # # # # #

The goal is in position (1,3), while the agent's position A is (2,1).

The agent's possible actions at each step are:

Actions:
- UP: Move Up
- DOWN: Move Down
- LEFT: Move Left
- RIGHT: Move Right

Your final answer should be a valid JSON object provided in the following form:

{{
  "action": "<UP|DOWN|LEFT|RIGHT>",
}}

DO NOT INCLUDE ANY `json` or `jsonb` in your response NOR TICK MARKS LIKE THIS: ```json ```. Start with {{ and end with }} exactly.

# Inputs

Current grid state:

{grid_state}
"""


CALL_COLUMNS = [
    "trajectory_id",
    "model",
    "api_model_id",
    "reasoning_setting",
    "temperature",
    "top_p",
    "seed",
    "max_output_tokens",
    "context_mode",
    "prompt_sha256",
    "grid_id",
    "grid_size",
    "difficulty",
    "observed_wall_fraction",
    "grid_replicate",
    "step_index",
    "agent_x",
    "agent_y",
    "optimal_actions",
    "selected_action",
    "is_optimal_action",
    "valid_action",
    "api_success",
    "finish_reason",
    "truncated_call",
    "latency_seconds",
    "prompt_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
    "estimated_cost_usd",
    "attempts",
    "error_type",
    "error_message",
    "goal_reached_after_action",
    "response_text",
    "reasoning_content",
    "prompt_text",
    "generated_token_ids",
    "generated_tokens",
    "generated_token_logprobs",
    "generated_token_bytes",
    "token_sequence_complete",
    "token_sequence_source",
    "response_id",
    "provider_returned_model_id",
    "system_fingerprint",
]

TRAJECTORY_COLUMNS = [
    "trajectory_id",
    "model",
    "api_model_id",
    "reasoning_setting",
    "temperature",
    "sampling_condition",
    "grid_id",
    "grid_size",
    "difficulty",
    "observed_wall_fraction",
    "grid_replicate",
    "trajectory_complete",
    "goal_reached",
    "steps",
    "optimal_path_length",
    "optimal_action_fraction",
    "valid_action_fraction",
    "invalid_output_rate",
    "api_failure_rate",
    "truncation_or_failure",
    "step_limit_reached",
    "total_output_tokens",
    "max_call_output_tokens",
    "total_runtime_seconds",
    "p50_call_runtime_seconds",
    "p95_call_runtime_seconds",
    "total_prompt_tokens",
    "total_tokens",
    "estimated_cost_usd",
]


def stable_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def read_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_CONFIG))
    config = json.loads(path.read_text())
    validate_config(config)
    return config


def apply_api_model_overrides(
    config: dict[str, Any], overrides: Iterable[str]
) -> dict[str, Any]:
    """Return a copy with explicit ``MODEL=API_ID`` replacements applied.

    Dedicated Together deployments receive an account-specific inference name,
    so it cannot be committed to the shared configuration. Applying the name at
    prepare time ensures the resolved identifier is retained in config.lock.json
    and in every scheduled trajectory.
    """
    updated = json.loads(json.dumps(config))
    by_name = {str(model["name"]): model for model in updated["models"]}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Expected MODEL=API_ID, got {item!r}.")
        name, api_model_id = (part.strip() for part in item.split("=", 1))
        if name not in by_name:
            raise ValueError(
                f"Unknown model {name!r}; expected one of {sorted(by_name)}."
            )
        if not api_model_id:
            raise ValueError(f"Empty API model ID for {name!r}.")
        by_name[name]["api_model_id"] = api_model_id
    validate_config(updated)
    return updated


def unresolved_api_model_ids(config: dict[str, Any]) -> list[str]:
    return [
        str(model["name"])
        for model in config["models"]
        if str(model["api_model_id"]).startswith("REPLACE_WITH_")
    ]


def select_schedule_models(
    config: dict[str, Any], schedule: list[dict[str, Any]], names: Iterable[str]
) -> list[dict[str, Any]]:
    selected = set(names)
    if not selected:
        return schedule
    known = {str(model["name"]) for model in config["models"]}
    unknown = selected - known
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")
    return [row for row in schedule if str(row["model"]) in selected]


def together_model_availability(api_key: str | None = None) -> list[dict[str, Any]]:
    """Check configured inference names separately from any paid generation."""
    from together import Together

    key = api_key or os.getenv("TOGETHER_API_KEY") or os.getenv("TOGETHERAI_API_KEY")
    if not key:
        raise RuntimeError("Set TOGETHER_API_KEY before checking model availability.")
    client = Together(api_key=key)
    response = client.models.list()
    entries = _get(response, "data", response)
    if not isinstance(entries, Iterable) or isinstance(entries, (str, bytes, dict)):
        raise RuntimeError(
            "Together models.list() returned an unexpected response shape."
        )
    return [
        {
            "api_model_id": str(_get(entry, "id", "")),
            "display_name": str(_get(entry, "display_name", "") or ""),
            "type": str(_get(entry, "type", "") or ""),
        }
        for entry in entries
        if _get(entry, "id", "")
    ]


def check_configured_api_models(
    config: dict[str, Any], available: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    available_ids = {str(row["api_model_id"]) for row in available}
    return [
        {
            "model": str(model["name"]),
            "api_model_id": str(model["api_model_id"]),
            "deployment": str(model.get("deployment", "")),
            "available": str(model["api_model_id"]) in available_ids,
        }
        for model in config["models"]
    ]


def validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported or missing config schema_version; expected 1.")
    sizes = [int(value) for value in config.get("grid_sizes", [])]
    if len(sizes) not in {2, 3} or any(size < 5 or size % 2 == 0 for size in sizes):
        raise ValueError("grid_sizes must contain two or three odd sizes >= 5.")
    difficulties = [float(value) for value in config.get("difficulty_levels", [])]
    if difficulties != [0.0, 0.6, 1.0]:
        raise ValueError(
            "difficulty_levels must be exactly [0.0, 0.6, 1.0] for this smoke test."
        )
    if int(config.get("grids_per_cell", 0)) != 2:
        raise ValueError("grids_per_cell must be 2 for the frozen smoke-test design.")
    if int(config.get("grid_candidate_pool_size", 0)) < 10:
        raise ValueError("grid_candidate_pool_size must be at least 10.")
    if [float(value) for value in config.get("grid_selection_quantiles", [])] != [
        0.5,
        0.9,
    ]:
        raise ValueError("grid_selection_quantiles must be exactly [0.5, 0.9].")
    if int(config.get("max_output_tokens", 0)) != 16000:
        raise ValueError("max_output_tokens must be 16000 for the smoke test.")
    if not config.get("models"):
        raise ValueError("At least one model must be configured.")
    for model in config["models"]:
        if not model.get("api_model_id") or not model.get("local_checkpoint"):
            raise ValueError("Every model needs api_model_id and local_checkpoint.")
        if not re.fullmatch(r"[0-9a-f]{40}", str(model.get("revision", ""))):
            raise ValueError(
                f"{model['name']}: local checkpoint revision must be a 40-character commit SHA."
            )
        settings = model.get("reasoning_settings", [])
        if model.get("reasoning_control") and settings != ["low", "medium"]:
            raise ValueError(
                f"{model['name']}: controllable reasoning must test low and medium."
            )
        if not model.get("reasoning_control") and settings != ["native"]:
            raise ValueError(
                f"{model['name']}: uncontrollable reasoning must use ['native']."
            )


def _env_layout(env: Simple2DNavigationEnv) -> list[list[str]]:
    layout: list[list[str]] = []
    agent = tuple(int(value) for value in env.agent_pos)
    goal = tuple(int(value) for value in env.goal_pos)
    for y in range(env.height):
        row: list[str] = []
        for x in range(env.width):
            if (x, y) == agent:
                row.append("A")
            elif (x, y) == goal:
                row.append("G")
            else:
                cell = env.grid.get(x, y)
                row.append("#" if cell is not None and cell.type == "wall" else "_")
        layout.append(row)
    return layout


def render_grid(layout: list[list[str]]) -> str:
    width = len(layout[0])
    pad = max(1, len(str(max(len(layout), width) - 1)))
    header = " " * (pad + 1) + " ".join(f"{x:>{pad}}" for x in range(width))
    lines = [header]
    for y, row in enumerate(layout):
        lines.append(f"{y:>{pad}} " + " ".join(f"{cell:>{pad}}" for cell in row))
    return "\n".join(lines)


def locate(layout: list[list[str]], symbol: str) -> tuple[int, int]:
    hits = [
        (x, y)
        for y, row in enumerate(layout)
        for x, cell in enumerate(row)
        if cell == symbol
    ]
    if len(hits) != 1:
        raise ValueError(f"Expected one {symbol!r} in layout, found {len(hits)}.")
    return hits[0]


def is_passable(layout: list[list[str]], x: int, y: int) -> bool:
    return 0 <= y < len(layout) and 0 <= x < len(layout[0]) and layout[y][x] != "#"


def distance_map(layout: list[list[str]]) -> dict[tuple[int, int], int]:
    goal = locate(layout, "G")
    distances = {goal: 0}
    queue: deque[tuple[int, int]] = deque([goal])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ACTION_DELTAS.values():
            nxt = (x + dx, y + dy)
            if nxt not in distances and is_passable(layout, *nxt):
                distances[nxt] = distances[(x, y)] + 1
                queue.append(nxt)
    return distances


def optimal_actions(layout: list[list[str]], position: tuple[int, int]) -> list[str]:
    distances = distance_map(layout)
    current = distances.get(position)
    if current is None or current == 0:
        return []
    out = []
    for action, (dx, dy) in ACTION_DELTAS.items():
        if distances.get((position[0] + dx, position[1] + dy)) == current - 1:
            out.append(action)
    return out


def apply_action(
    layout: list[list[str]], position: tuple[int, int], action: str
) -> tuple[tuple[int, int], bool]:
    dx, dy = ACTION_DELTAS[action]
    candidate = (position[0] + dx, position[1] + dy)
    new_position = candidate if is_passable(layout, *candidate) else position
    return new_position, new_position == locate(layout, "G")


def layout_at_position(
    layout: list[list[str]], position: tuple[int, int]
) -> list[list[str]]:
    out = [["_" if cell == "A" else cell for cell in row] for row in layout]
    x, y = position
    if out[y][x] == "G":
        return out
    out[y][x] = "A"
    return out


def _wall_fraction(layout: list[list[str]]) -> float:
    interior = [cell for row in layout[1:-1] for cell in row[1:-1]]
    return sum(cell == "#" for cell in interior) / len(interior)


def generate_grid_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    validate_config(config)
    rows: list[dict[str, Any]] = []
    master = random.Random(int(config["seed"]))
    saved_state = random.getstate()
    try:
        for size in config["grid_sizes"]:
            for difficulty in config["difficulty_levels"]:
                candidates: list[dict[str, Any]] = []
                for candidate_index in range(int(config["grid_candidate_pool_size"])):
                    env_seed = master.randrange(2**31)
                    random.seed(env_seed)
                    env = Simple2DNavigationEnv(
                        size=int(size), complexity=float(difficulty)
                    )
                    # This environment prints an alarming legacy reset warning even
                    # though the seeded reset is intentional here. Keep plan output
                    # machine-readable while preserving the environment behavior.
                    with contextlib.redirect_stdout(io.StringIO()):
                        env.reset(seed=env_seed)
                    layout = _env_layout(env)
                    start = locate(layout, "A")
                    distances = distance_map(layout)
                    if start not in distances:
                        raise RuntimeError("Generated maze is unexpectedly unsolvable.")
                    candidates.append(
                        {
                            "grid_size": int(size),
                            "difficulty": float(difficulty),
                            "candidate_index": candidate_index,
                            "generation_seed": env_seed,
                            "layout": layout,
                            "grid_text": render_grid(layout),
                            "start_x": start[0],
                            "start_y": start[1],
                            "goal_x": locate(layout, "G")[0],
                            "goal_y": locate(layout, "G")[1],
                            "optimal_path_length": distances[start],
                            "observed_wall_fraction": _wall_fraction(layout),
                        }
                    )
                ordered = sorted(
                    candidates,
                    key=lambda row: (
                        int(row["optimal_path_length"]),
                        int(row["generation_seed"]),
                    ),
                )
                selected_indices: set[int] = set()
                for replicate, quantile in enumerate(
                    config["grid_selection_quantiles"], start=1
                ):
                    rank = round((len(ordered) - 1) * float(quantile))
                    while rank in selected_indices and rank + 1 < len(ordered):
                        rank += 1
                    selected_indices.add(rank)
                    selected = dict(ordered[rank])
                    selected.update(
                        {
                            "grid_id": f"size{int(size)}_difficulty{float(difficulty):.1f}_grid{replicate}",
                            "grid_replicate": replicate,
                            "selection_quantile": float(quantile),
                            "candidate_pool_size": len(ordered),
                            "candidate_path_length_rank": rank + 1,
                        }
                    )
                    rows.append(selected)
    finally:
        random.setstate(saved_state)
    return rows


def build_schedule(
    config: dict[str, Any], grids: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    schedule = []
    for model in config["models"]:
        for reasoning in model["reasoning_settings"]:
            for grid in grids:
                conditions = [(float(config["temperature"]), "sampled")]
                if int(grid["grid_replicate"]) == 1:
                    conditions.append(
                        (float(config["greedy_temperature"]), "greedy_sanity_check")
                    )
                for temperature, condition in conditions:
                    trajectory_id = "__".join(
                        [
                            re.sub(r"[^a-z0-9]+", "-", model["name"].lower()).strip(
                                "-"
                            ),
                            str(reasoning),
                            grid["grid_id"],
                            "t0" if temperature == 0 else "t07",
                        ]
                    )
                    limit = min(
                        int(config["step_limit_hard_cap"]),
                        max(
                            int(grid["optimal_path_length"])
                            + int(config["step_limit_slack"]),
                            math.ceil(
                                float(config["step_limit_multiplier"])
                                * int(grid["optimal_path_length"])
                            ),
                        ),
                    )
                    schedule.append(
                        {
                            "trajectory_id": trajectory_id,
                            "model": model["name"],
                            "api_model_id": model["api_model_id"],
                            "reasoning_setting": reasoning,
                            "reasoning_control": bool(model["reasoning_control"]),
                            "temperature": temperature,
                            "sampling_condition": condition,
                            "step_limit": limit,
                            **grid,
                        }
                    )
    return schedule


def parse_action(text: str) -> str | None:
    candidates = re.findall(
        r'"action"\s*:\s*"?(LEFT|RIGHT|UP|DOWN)"?', text, flags=re.IGNORECASE
    )
    if candidates:
        return candidates[-1].upper()
    stripped = re.sub(r"```(?:json)?|```", "", text, flags=re.IGNORECASE).strip()
    try:
        payload = json.loads(stripped[stripped.rfind("{") : stripped.rfind("}") + 1])
        action = str(payload.get("action", "")).upper()
        return action if action in ACTIONS else None
    except Exception:
        return None


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _usage_int(usage: Any, key: str) -> int:
    value = _get(usage, key, 0)
    return int(value or 0)


def _reconstruct_gpt_oss_token_ids(
    tokens: list[str], token_bytes: list[list[int]]
) -> list[int]:
    """Map provider token boundaries through OpenAI o200k_harmony."""
    if len(tokens) != len(token_bytes):
        return []
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_harmony")
    try:
        token_ids = [
            encoding.encode_single_token(bytes(byte_values) if byte_values else token)
            for token, byte_values in zip(tokens, token_bytes, strict=True)
        ]
    except KeyError:
        return []
    if encoding.decode_bytes(token_ids) != b"".join(
        bytes(byte_values) for byte_values in token_bytes
    ):
        return []
    return token_ids


def extract_response(response: Any) -> dict[str, Any]:
    choices = _get(response, "choices", []) or []
    if not choices:
        raise ValueError("API response has no choices.")
    choice = choices[0]
    message = _get(choice, "message", {})
    content = str(_get(message, "content", "") or "")
    reasoning = str(
        _get(message, "reasoning", "")
        or _get(_get(message, "provider_specific_fields", {}), "reasoning", "")
        or _get(message, "reasoning_content", "")
        or _get(_get(message, "provider_specific_fields", {}), "reasoning_content", "")
        or ""
    )
    usage = _get(response, "usage", {})
    details = _get(usage, "completion_tokens_details", {})
    reasoning_tokens = _usage_int(usage, "reasoning_tokens") or _usage_int(
        details, "reasoning_tokens"
    )
    completion_tokens = _usage_int(usage, "completion_tokens")
    logprobs = _get(choice, "logprobs", {}) or {}
    token_ids = list(_get(logprobs, "token_ids", []) or [])
    tokens = list(_get(logprobs, "tokens", []) or [])
    token_logprobs = list(_get(logprobs, "token_logprobs", []) or [])
    token_bytes = list(_get(logprobs, "bytes", []) or [])
    content_logprobs = list(_get(logprobs, "content", []) or [])
    if content_logprobs:
        if not tokens:
            tokens = [str(_get(item, "token", "") or "") for item in content_logprobs]
        if not token_logprobs:
            token_logprobs = [_get(item, "logprob", None) for item in content_logprobs]
        if not token_bytes:
            token_bytes = [
                list(_get(item, "bytes", []) or []) for item in content_logprobs
            ]
        if not token_ids and all(
            _get(item, "token_id", None) is not None for item in content_logprobs
        ):
            token_ids = [int(_get(item, "token_id")) for item in content_logprobs]
    provider_model_id = str(_get(response, "model", "") or "")
    provider_ids_complete = (
        completion_tokens > 0 and len(token_ids) == completion_tokens
    )
    pieces_complete = completion_tokens > 0 and len(tokens) == completion_tokens
    token_sequence_source = (
        "provider_token_ids"
        if provider_ids_complete
        else ("provider_token_pieces" if pieces_complete else "incomplete")
    )
    if (
        not provider_ids_complete
        and pieces_complete
        and "gpt-oss" in provider_model_id.lower()
    ):
        reconstructed_ids = _reconstruct_gpt_oss_token_ids(tokens, token_bytes)
        if len(reconstructed_ids) == completion_tokens:
            token_ids = reconstructed_ids
            token_sequence_source = "reconstructed_o200k_harmony"
    return {
        "content": content,
        "reasoning_content": reasoning,
        "finish_reason": str(_get(choice, "finish_reason", "") or ""),
        "prompt_tokens": _usage_int(usage, "prompt_tokens"),
        "output_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": _usage_int(usage, "total_tokens"),
        "generated_token_ids": [int(value) for value in token_ids],
        "generated_tokens": tokens,
        "generated_token_logprobs": token_logprobs,
        "generated_token_bytes": token_bytes,
        "token_sequence_complete": token_sequence_source != "incomplete",
        "token_sequence_source": token_sequence_source,
        "response_id": str(_get(response, "id", "") or ""),
        "provider_returned_model_id": str(_get(response, "model", "") or ""),
        "system_fingerprint": str(_get(response, "system_fingerprint", "") or ""),
    }


def estimated_cost(
    model: dict[str, Any], prompt_tokens: int, output_tokens: int, latency: float
) -> float | None:
    input_price = model.get("input_usd_per_million_tokens")
    output_price = model.get("output_usd_per_million_tokens")
    if input_price is not None and output_price is not None:
        return (
            prompt_tokens * float(input_price) / 1e6
            + output_tokens * float(output_price) / 1e6
        )
    hourly = model.get("dedicated_hourly_price_usd")
    if hourly is not None:
        return latency * float(hourly) / 3600
    return None


def make_together_query(api_key: str | None = None) -> Callable[..., Any]:
    from together import Together

    key = api_key or os.getenv("TOGETHER_API_KEY") or os.getenv("TOGETHERAI_API_KEY")
    if not key:
        raise RuntimeError("Set TOGETHER_API_KEY before the query stage.")
    client = Together(api_key=key)

    def query(
        *,
        model_id: str,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        reasoning_setting: str,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
            "logprobs": 1,
        }
        if reasoning_setting != "native":
            kwargs["reasoning_effort"] = reasoning_setting
        return client.chat.completions.create(**kwargs)

    return query


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSONL at {path}:{line_number}"
                    ) from exc
    return rows


class TokenSequenceUnavailable(RuntimeError):
    """Provider did not return a replay-grade token sequence for every generated token."""


def run_api_schedule(
    config: dict[str, Any],
    schedule: list[dict[str, Any]],
    raw_path: Path,
    query_fn: Callable[..., Any] | None = None,
    progress_fn: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    query_fn = query_fn or make_together_query()
    model_map = {model["name"]: model for model in config["models"]}
    existing = read_jsonl(raw_path)
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in existing:
        by_trajectory[row["trajectory_id"]].append(row)

    for trajectory_index, spec in enumerate(schedule, start=1):
        prior = sorted(
            by_trajectory.get(spec["trajectory_id"], []),
            key=lambda row: int(row["step_index"]),
        )
        if prior and prior[-1].get("trajectory_terminal"):
            if progress_fn:
                progress_fn(
                    f"[api {trajectory_index}/{len(schedule)}] already complete: {spec['trajectory_id']}"
                )
            continue
        if progress_fn:
            progress_fn(
                f"[api {trajectory_index}/{len(schedule)}] running {spec['trajectory_id']} "
                f"(resume at step {len(prior)})"
            )
        position = (int(spec["start_x"]), int(spec["start_y"]))
        goal_reached = False
        for row in prior:
            action = row.get("selected_action")
            if action in ACTIONS and row.get("api_success") and row.get("valid_action"):
                position, goal_reached = apply_action(spec["layout"], position, action)
        next_step = len(prior)
        if goal_reached:
            continue

        for step_index in range(next_step, int(spec["step_limit"])):
            current_layout = layout_at_position(spec["layout"], position)
            prompt = PROMPT_TEMPLATE.format(grid_state=render_grid(current_layout))
            seed = int(config["seed"]) + int(
                stable_hash([spec["trajectory_id"], step_index])[:8], 16
            )
            started = time.perf_counter()
            response_data: dict[str, Any] = {}
            error: Exception | None = None
            attempts = 0
            for attempts in range(1, int(config["max_api_attempts"]) + 1):
                try:
                    response = query_fn(
                        model_id=spec["api_model_id"],
                        prompt=prompt,
                        temperature=float(spec["temperature"]),
                        top_p=float(config["top_p"]),
                        max_tokens=int(config["max_output_tokens"]),
                        seed=seed,
                        reasoning_setting=str(spec["reasoning_setting"]),
                    )
                    response_data = extract_response(response)
                    if (
                        config.get("require_complete_token_sequence")
                        and not response_data["token_sequence_complete"]
                    ):
                        raise TokenSequenceUnavailable(
                            "Together did not return token IDs or token pieces for every completion token; "
                            "aborting before the full collection continues."
                        )
                    error = None
                    break
                except TokenSequenceUnavailable as exc:
                    error = exc
                    if attempts < int(config["max_api_attempts"]):
                        time.sleep(min(2 ** (attempts - 1), 30))
                    else:
                        raise
                except (
                    Exception
                ) as exc:  # API/network/provider errors are part of the smoke test
                    error = exc
                    if attempts < int(config["max_api_attempts"]):
                        time.sleep(min(2 ** (attempts - 1), 4))
            latency = time.perf_counter() - started
            action = (
                parse_action(response_data.get("content", ""))
                if error is None
                else None
            )
            valid = action in ACTIONS
            optimal = optimal_actions(spec["layout"], position)
            reached = False
            if valid:
                position, reached = apply_action(spec["layout"], position, str(action))
            finish = response_data.get("finish_reason", "")
            truncated_call = str(finish).lower() in {"length", "max_tokens"}
            terminal = (
                reached
                or not valid
                or error is not None
                or truncated_call
                or step_index + 1 >= int(spec["step_limit"])
            )
            model_cfg = model_map[spec["model"]]
            cost = estimated_cost(
                model_cfg,
                int(response_data.get("prompt_tokens", 0)),
                int(response_data.get("output_tokens", 0)),
                latency,
            )
            row = {
                "schema_version": 1,
                "trajectory_id": spec["trajectory_id"],
                "model": spec["model"],
                "api_model_id": spec["api_model_id"],
                "reasoning_setting": spec["reasoning_setting"],
                "temperature": spec["temperature"],
                "top_p": float(config["top_p"]),
                "seed": seed,
                "max_output_tokens": int(config["max_output_tokens"]),
                "context_mode": str(config["context_mode"]),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_text": prompt,
                "sampling_condition": spec["sampling_condition"],
                "grid_id": spec["grid_id"],
                "grid_size": spec["grid_size"],
                "difficulty": spec["difficulty"],
                "observed_wall_fraction": spec["observed_wall_fraction"],
                "grid_replicate": spec["grid_replicate"],
                "step_limit": spec["step_limit"],
                "optimal_path_length": spec["optimal_path_length"],
                "step_index": step_index,
                "agent_x": current_layout and locate(current_layout, "A")[0],
                "agent_y": current_layout and locate(current_layout, "A")[1],
                "optimal_actions": optimal,
                "selected_action": action or "",
                "is_optimal_action": bool(valid and action in optimal),
                "valid_action": bool(valid),
                "api_success": error is None,
                "finish_reason": finish,
                "truncated_call": truncated_call,
                "latency_seconds": latency,
                "prompt_tokens": int(response_data.get("prompt_tokens", 0)),
                "output_tokens": int(response_data.get("output_tokens", 0)),
                "reasoning_tokens": int(response_data.get("reasoning_tokens", 0)),
                "total_tokens": int(response_data.get("total_tokens", 0)),
                "estimated_cost_usd": cost,
                "attempts": attempts,
                "error_type": type(error).__name__ if error else "",
                "error_message": str(error) if error else "",
                "goal_reached_after_action": reached,
                "trajectory_terminal": terminal,
                "response_text": response_data.get("content", ""),
                "reasoning_content": response_data.get("reasoning_content", ""),
                "generated_token_ids": response_data.get("generated_token_ids", []),
                "generated_tokens": response_data.get("generated_tokens", []),
                "generated_token_logprobs": response_data.get(
                    "generated_token_logprobs", []
                ),
                "generated_token_bytes": response_data.get("generated_token_bytes", []),
                "token_sequence_complete": bool(
                    response_data.get("token_sequence_complete", False)
                ),
                "token_sequence_source": response_data.get(
                    "token_sequence_source", "incomplete"
                ),
                "response_id": response_data.get("response_id", ""),
                "provider_returned_model_id": response_data.get(
                    "provider_returned_model_id", ""
                ),
                "system_fingerprint": response_data.get("system_fingerprint", ""),
            }
            _append_jsonl(raw_path, row)
            by_trajectory[spec["trajectory_id"]].append(row)
            if terminal:
                if progress_fn:
                    outcome = (
                        "goal reached"
                        if reached
                        else (
                            "API failure"
                            if error is not None
                            else (
                                "invalid output"
                                if not valid
                                else (
                                    "truncated"
                                    if truncated_call
                                    else "step limit reached"
                                )
                            )
                        )
                    )
                    progress_fn(
                        f"[api {trajectory_index}/{len(schedule)}] {outcome}: "
                        f"{spec['trajectory_id']} after {step_index + 1} calls"
                    )
                break
    return read_jsonl(raw_path)


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def trajectory_rows(
    call_rows: list[dict[str, Any]], schedule: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    specs = {row["trajectory_id"]: row for row in schedule}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in call_rows:
        grouped[row["trajectory_id"]].append(row)
    out = []
    for trajectory_id, calls in sorted(grouped.items()):
        calls.sort(key=lambda row: int(row["step_index"]))
        spec = specs[trajectory_id]
        valid = [row for row in calls if row.get("valid_action")]
        successful = [row for row in calls if row.get("api_success")]
        latencies = [float(row["latency_seconds"]) for row in calls]
        reached = bool(calls and calls[-1].get("goal_reached_after_action"))
        step_limit_reached = bool(
            calls
            and int(calls[-1]["step_index"]) + 1 >= int(spec["step_limit"])
            and not reached
        )
        costs = [row.get("estimated_cost_usd") for row in calls]
        costs_known = [float(value) for value in costs if value not in (None, "")]
        out.append(
            {
                "trajectory_id": trajectory_id,
                "model": spec["model"],
                "api_model_id": spec["api_model_id"],
                "reasoning_setting": spec["reasoning_setting"],
                "temperature": spec["temperature"],
                "sampling_condition": spec["sampling_condition"],
                "grid_id": spec["grid_id"],
                "grid_size": spec["grid_size"],
                "difficulty": spec["difficulty"],
                "observed_wall_fraction": spec["observed_wall_fraction"],
                "grid_replicate": spec["grid_replicate"],
                "trajectory_complete": bool(
                    calls and calls[-1].get("trajectory_terminal")
                ),
                "goal_reached": reached,
                "steps": len(valid),
                "optimal_path_length": spec["optimal_path_length"],
                "optimal_action_fraction": (
                    sum(bool(row["is_optimal_action"]) for row in valid) / len(valid)
                    if valid
                    else float("nan")
                ),
                "valid_action_fraction": (
                    len(valid) / len(calls) if calls else float("nan")
                ),
                "invalid_output_rate": sum(
                    bool(row["api_success"]) and not bool(row["valid_action"])
                    for row in calls
                )
                / len(calls),
                "api_failure_rate": 1 - len(successful) / len(calls),
                "truncation_or_failure": bool(
                    any(
                        row.get("truncated_call") or not row.get("api_success")
                        for row in calls
                    )
                    or step_limit_reached
                ),
                "step_limit_reached": step_limit_reached,
                "total_output_tokens": sum(
                    int(row.get("output_tokens", 0)) for row in calls
                ),
                "max_call_output_tokens": max(
                    int(row.get("output_tokens", 0)) for row in calls
                ),
                "total_runtime_seconds": sum(latencies),
                "p50_call_runtime_seconds": percentile(latencies, 0.5),
                "p95_call_runtime_seconds": percentile(latencies, 0.95),
                "total_prompt_tokens": sum(
                    int(row.get("prompt_tokens", 0)) for row in calls
                ),
                "total_tokens": sum(int(row.get("total_tokens", 0)) for row in calls),
                "estimated_cost_usd": (
                    sum(costs_known) if len(costs_known) == len(calls) else None
                ),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cooked = {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(cooked)


def condition_summary(
    rows: list[dict[str, Any]], call_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    calls_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    trajectories_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    fields = (
        "model",
        "api_model_id",
        "grid_size",
        "difficulty",
        "reasoning_setting",
        "temperature",
    )
    complete_ids = {
        str(row["trajectory_id"]) for row in rows if row.get("trajectory_complete")
    }
    for row in call_rows:
        if str(row["trajectory_id"]) not in complete_ids:
            continue
        key = tuple(row[field] for field in fields)
        calls_by_key[key].append(row)
    for row in rows:
        if not row.get("trajectory_complete"):
            continue
        key = tuple(row[field] for field in fields)
        trajectories_by_key[key].append(row)
    output = []
    for key, trajectories in sorted(
        trajectories_by_key.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        calls = calls_by_key[key]
        output_tokens = [
            int(row.get("output_tokens", 0)) for row in calls if row.get("api_success")
        ]
        trajectory_runtime = [
            float(row["total_runtime_seconds"]) for row in trajectories
        ]
        trajectory_output_tokens = [
            int(row["total_output_tokens"]) for row in trajectories
        ]
        output.append(
            {
                **dict(zip(fields, key)),
                "trajectories": len(trajectories),
                "calls": len(calls),
                "goal_success_rate": sum(
                    bool(row["goal_reached"]) for row in trajectories
                )
                / len(trajectories),
                "optimal_action_rate": sum(
                    bool(row["is_optimal_action"])
                    for row in calls
                    if row.get("valid_action")
                )
                / max(1, sum(bool(row.get("valid_action")) for row in calls)),
                "p50_output_tokens_per_call": percentile(output_tokens, 0.50),
                "p95_output_tokens_per_call": percentile(output_tokens, 0.95),
                "max_output_tokens_per_call": max(output_tokens, default=float("nan")),
                "p50_total_output_tokens_per_trajectory": percentile(
                    trajectory_output_tokens, 0.50
                ),
                "p95_total_output_tokens_per_trajectory": percentile(
                    trajectory_output_tokens, 0.95
                ),
                "max_total_output_tokens_per_trajectory": max(
                    trajectory_output_tokens, default=float("nan")
                ),
                "p50_trajectory_runtime_seconds": percentile(trajectory_runtime, 0.50),
                "p95_trajectory_runtime_seconds": percentile(trajectory_runtime, 0.95),
                "invalid_output_rate": sum(
                    bool(row.get("api_success")) and not bool(row.get("valid_action"))
                    for row in calls
                )
                / len(calls),
                "truncation_or_failure_rate": sum(
                    bool(row["truncation_or_failure"]) for row in trajectories
                )
                / len(trajectories),
                "estimated_cost_usd": (
                    sum(float(row["estimated_cost_usd"]) for row in trajectories)
                    if all(
                        row.get("estimated_cost_usd") not in (None, "")
                        for row in trajectories
                    )
                    else None
                ),
            }
        )
    return output


def _fmt(value: Any, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "not available"
    return f"{float(value):.{decimals}f}"


def build_report(
    config: dict[str, Any],
    schedule: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> str:
    completed = {
        row["trajectory_id"] for row in trajectories if row.get("trajectory_complete")
    }
    pending = len(schedule) - len(completed)
    lines = [
        "# Maze API smoke test",
        "",
        f"Status: {'complete' if pending == 0 else 'prepared / incomplete'} ({len(completed)}/{len(schedule)} trajectories complete).",
        "",
        "This smoke test checks maze-solving behavior, output-token requirements, and runtime before activation collection. "
        "Every model sees the same regular no-key/no-door mazes. At each step it sees only the current full grid; trajectory history is not included. "
        "The generator setting called `difficulty` controls how many walls are retained; it is not the realized wall density, which is recorded separately.",
        "",
        "## Frozen setup",
        "",
        f"- Grid sizes: {', '.join(map(str, config['grid_sizes']))}",
        "- Difficulty settings: 0.0, 0.6, 1.0; two grids per size/difficulty cell",
        "- Grid selection: median and 90th-percentile optimal path length from 32 deterministic candidates per cell",
        "- Main sampling: temperature 0.7 on both grids; one temperature-0 trajectory on the first grid in each cell",
        f"- Maximum output per action call: {int(config['max_output_tokens']):,} tokens",
        f"- Context mode: {config['context_mode']}",
        "- Reasoning effort: low and medium for GPT-OSS; provider-native/default for models without that control",
        "",
        "## Results by condition",
        "",
    ]
    if not summaries:
        lines += [
            "No complete API trajectories have been run yet. Run the query stage on the API machine, then run analyze.",
            "",
        ]
    else:
        lines += [
            "| Model | Size | Difficulty | Reasoning | T | Goal reached | Optimal actions | Output tokens/call p50 / p95 / max | Total output tokens/trajectory p50 / p95 / max | Trajectory runtime p50 / p95 (s) | Invalid outputs | Truncated or failed |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summaries:
            lines.append(
                f"| {row['model']} | {row['grid_size']} | {float(row['difficulty']):.1f} | {row['reasoning_setting']} | {float(row['temperature']):.1f} | "
                f"{100 * row['goal_success_rate']:.0f}% | {100 * row['optimal_action_rate']:.0f}% | "
                f"{_fmt(row['p50_output_tokens_per_call'], 0)} / {_fmt(row['p95_output_tokens_per_call'], 0)} / {_fmt(row['max_output_tokens_per_call'], 0)} | "
                f"{_fmt(row['p50_total_output_tokens_per_trajectory'], 0)} / {_fmt(row['p95_total_output_tokens_per_trajectory'], 0)} / {_fmt(row['max_total_output_tokens_per_trajectory'], 0)} | "
                f"{_fmt(row['p50_trajectory_runtime_seconds'])} / {_fmt(row['p95_trajectory_runtime_seconds'])} | "
                f"{100 * row['invalid_output_rate']:.0f}% | {100 * row['truncation_or_failure_rate']:.0f}% |"
            )
        lines.append("")

    lines += ["## Decisions", ""]
    model_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    call_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        if not row.get("trajectory_complete"):
            continue
        if float(row["temperature"]) == float(config["temperature"]):
            model_groups[row["model"]].append(row)
    for row in calls:
        if (
            row["trajectory_id"] in completed
            and float(row["temperature"]) == float(config["temperature"])
            and row.get("api_success")
        ):
            call_groups[row["model"]].append(row)

    suitable, settings, budgets, runtimes = [], [], [], []
    feasible_by_model: list[str] = []
    for model in config["models"]:
        name = model["name"]
        rows = model_groups.get(name, [])
        if not rows:
            settings.append(f"{name}: pending")
            budgets.append(f"{name}: pending")
            runtimes.append(f"{name}: pending")
            continue
        by_effort: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_effort[row["reasoning_setting"]].append(row)
        chosen = "native"
        if model["reasoning_control"]:
            low = by_effort.get("low", [])
            medium = by_effort.get("medium", [])
            low_rate = (
                sum(bool(row["goal_reached"]) for row in low) / len(low) if low else 0
            )
            med_rate = (
                sum(bool(row["goal_reached"]) for row in medium) / len(medium)
                if medium
                else 0
            )
            low_bad_rate = (
                sum(
                    bool(row["truncation_or_failure"])
                    or float(row["invalid_output_rate"]) > 0
                    for row in low
                )
                / len(low)
                if low
                else 1.0
            )
            med_bad_rate = (
                sum(
                    bool(row["truncation_or_failure"])
                    or float(row["invalid_output_rate"]) > 0
                    for row in medium
                )
                / len(medium)
                if medium
                else 1.0
            )
            low_optimal_values = [
                float(row["optimal_action_fraction"])
                for row in low
                if math.isfinite(float(row["optimal_action_fraction"]))
            ]
            med_optimal_values = [
                float(row["optimal_action_fraction"])
                for row in medium
                if math.isfinite(float(row["optimal_action_fraction"]))
            ]
            low_optimal = (
                sum(low_optimal_values) / len(low_optimal_values)
                if low_optimal_values
                else 0.0
            )
            med_optimal = (
                sum(med_optimal_values) / len(med_optimal_values)
                if med_optimal_values
                else 0.0
            )
            low_is_usable = low_rate >= float(
                config["decision_thresholds"]["minimum_goal_success_rate"]
            ) and low_bad_rate <= float(
                config["decision_thresholds"]["maximum_invalid_or_failed_rate"]
            )
            medium_is_usable = med_rate >= float(
                config["decision_thresholds"]["minimum_goal_success_rate"]
            ) and med_bad_rate <= float(
                config["decision_thresholds"]["maximum_invalid_or_failed_rate"]
            )
            material = float(
                config["decision_thresholds"]["medium_material_success_gain"]
            )
            medium_materially_improves = (
                med_rate - low_rate >= material
                or med_optimal - low_optimal >= material
                or low_bad_rate - med_bad_rate >= material
            )
            chosen = (
                "medium"
                if not low_is_usable and medium_is_usable and medium_materially_improves
                else "low"
            )
        settings.append(f"{name}: {chosen}")
        chosen_rows = by_effort.get(chosen, rows)
        chosen_calls = [
            row for row in call_groups[name] if row["reasoning_setting"] == chosen
        ]
        tokens = [int(row["output_tokens"]) for row in chosen_calls]
        if tokens:
            recommendation = min(
                int(config["max_output_tokens"]),
                int(
                    math.ceil(
                        percentile(tokens, 0.95)
                        * float(config["decision_thresholds"]["token_budget_headroom"])
                        / 256
                    )
                    * 256
                ),
            )
            budgets.append(
                f"{name}: {recommendation:,} tokens (observed p95 {percentile(tokens, 0.95):.0f})"
            )
        else:
            budgets.append(f"{name}: pending")
        runtimes.append(
            f"{name}: {_fmt(percentile([row['total_runtime_seconds'] for row in chosen_rows], 0.95))} s p95/trajectory"
        )
        success = sum(bool(row["goal_reached"]) for row in chosen_rows) / len(
            chosen_rows
        )
        failure = sum(
            bool(row["truncation_or_failure"]) or float(row["invalid_output_rate"]) > 0
            for row in chosen_rows
        ) / len(chosen_rows)
        if success >= float(
            config["decision_thresholds"]["minimum_goal_success_rate"]
        ) and failure <= float(
            config["decision_thresholds"]["maximum_invalid_or_failed_rate"]
        ):
            suitable.append(name)
        feasible_sizes = []
        for size in config["grid_sizes"]:
            subset = [row for row in chosen_rows if int(row["grid_size"]) == int(size)]
            if (
                subset
                and sum(bool(row["goal_reached"]) for row in subset) / len(subset)
                >= 0.5
            ):
                feasible_sizes.append(str(size))
        feasible_by_model.append(
            f"{name}: {', '.join(feasible_sizes) if feasible_sizes else 'none observed'}"
        )

    lines += [
        f"Models suitable for full run: {', '.join(suitable) if suitable else ('pending' if pending else 'none under the prespecified gate')}",
        f"Recommended reasoning setting/model: {'; '.join(settings)}",
        f"Recommended max tokens/model: {'; '.join(budgets)}",
        f"Feasible grid sizes: {'; '.join(feasible_by_model) if feasible_by_model else 'pending'}",
        f"Observed p95 tokens/model: {'; '.join(budgets)}",
        f"Observed p95 runtime/model: {'; '.join(runtimes)}",
        "Behavioral/API failures: "
        + (
            f"see api_calls.csv; {pending} trajectories remain"
            if pending
            else "see api_calls.csv for invalid, truncated, retried, and failed calls"
        ),
        "Estimated API cost so far: "
        + (
            f"${sum(float(row['estimated_cost_usd']) for row in trajectories if row.get('trajectory_complete') and row.get('estimated_cost_usd') not in (None, '')):.4f} "
            "across complete trajectories with known pricing"
            if completed
            else "not available"
        ),
        "",
        "The low-versus-medium rule deliberately prefers low. Medium is selected only when low fails the usability gate, medium passes it, and medium improves goal success, optimal-action rate, or invalid/failure rate by at least 10 percentage points on the same grid set. "
        "The recommended token cap is 120% of the observed per-call p95, rounded up to 256 tokens and never above 16k.",
        "",
        "## Limits",
        "",
        "These are small-sample engineering estimates, not paper-level capability comparisons. API trajectories contain no activations and cannot replace the local activation-bearing run. "
        "The Qwen3-32B candidate requires an account-specific Together dedicated endpoint name. Its monetary estimate remains unavailable until an hourly endpoint price is entered in the config.",
    ]
    return "\n".join(lines) + "\n"


def write_analysis(
    output_dir: Path, config: dict[str, Any], schedule: list[dict[str, Any]]
) -> dict[str, Any]:
    calls = read_jsonl(output_dir / "raw_api_calls.jsonl")
    trajectories = trajectory_rows(calls, schedule)
    summaries = condition_summary(trajectories, calls)
    _write_csv(output_dir / "api_calls.csv", calls, CALL_COLUMNS)
    _write_csv(output_dir / "api_results.csv", trajectories, TRAJECTORY_COLUMNS)
    if summaries:
        _write_csv(output_dir / "condition_summary.csv", summaries, list(summaries[0]))
    else:
        _write_csv(
            output_dir / "condition_summary.csv",
            [],
            [
                "model",
                "api_model_id",
                "grid_size",
                "difficulty",
                "reasoning_setting",
                "temperature",
                "trajectories",
                "calls",
                "goal_success_rate",
                "optimal_action_rate",
                "p50_output_tokens_per_call",
                "p95_output_tokens_per_call",
                "max_output_tokens_per_call",
                "p50_total_output_tokens_per_trajectory",
                "p95_total_output_tokens_per_trajectory",
                "max_total_output_tokens_per_trajectory",
                "p50_trajectory_runtime_seconds",
                "p95_trajectory_runtime_seconds",
                "invalid_output_rate",
                "truncation_or_failure_rate",
                "estimated_cost_usd",
            ],
        )
    (output_dir / "api_summary.md").write_text(
        build_report(config, schedule, calls, trajectories, summaries)
    )
    return {
        "calls": len(calls),
        "trajectories": len(trajectories),
        "scheduled": len(schedule),
    }


def prepare(
    output_dir: Path,
    config: dict[str, Any],
    *,
    replace_empty_plan: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / "config.lock.json"
    if lock_path.exists():
        locked = json.loads(lock_path.read_text())
        if stable_hash(locked) != stable_hash(config):
            has_api_calls = bool(read_jsonl(output_dir / "raw_api_calls.jsonl"))
            has_local_records = bool(
                read_jsonl(output_dir / "local_preflight_records.jsonl")
            )
            if not replace_empty_plan or has_api_calls or has_local_records:
                raise ValueError(
                    "Configuration differs from config.lock.json. Use a new output directory, "
                    "or pass --replace-empty-plan only when no API/preflight records exist."
                )
            lock_path.write_text(json.dumps(config, indent=2) + "\n")
    else:
        lock_path.write_text(json.dumps(config, indent=2) + "\n")
    grids = generate_grid_rows(config)
    schedule = build_schedule(config, grids)
    (output_dir / "grids.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in grids)
    )
    (output_dir / "api_schedule.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in schedule)
    )
    if replace_empty_plan or not (output_dir / "local_preflight.md").exists():
        checkpoints = "; ".join(
            f"{model['name']}: {model['local_checkpoint']} @ {model['revision']}"
            for model in config["models"]
        )
        (output_dir / "local_preflight.md").write_text(
            "# Local activation preflight\n\n"
            "Status: NOT RUN\n\n"
            f"Exact local checkpoints: {checkpoints}\n\n"
            "Activation capture at layers 8/15/23: NOT RUN\n\n"
            "Reasoning-trace activation capture: NOT RUN\n\n"
            "Peak VRAM: not measured\n\n"
            "Approx. activation storage/trajectory: not measured\n\n"
            "Local inference/hook issues: preflight has not been run\n\n"
            "Ready for full activation run: NO\n"
        )
    write_analysis(output_dir, config, schedule)
    return grids, schedule


def planned_call_bound(schedule: list[dict[str, Any]]) -> int:
    return sum(int(row["step_limit"]) for row in schedule)


def planned_call_breakdown(
    config: dict[str, Any], schedule: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return transparent upper bounds; these are not expected spend estimates."""
    model_map = {model["name"]: model for model in config["models"]}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in schedule:
        grouped[str(row["model"])].append(row)
    output = []
    for name, rows in grouped.items():
        action_calls = sum(int(row["step_limit"]) for row in rows)
        max_output_tokens = action_calls * int(config["max_output_tokens"])
        model = model_map[name]
        output_price = model.get("output_usd_per_million_tokens")
        output.append(
            {
                "model": name,
                "trajectories": len(rows),
                "maximum_action_calls": action_calls,
                "maximum_output_tokens": max_output_tokens,
                "output_only_cost_upper_bound_usd": (
                    max_output_tokens * float(output_price) / 1e6
                    if output_price is not None
                    else None
                ),
                "note": "All trajectories reaching their step cap and every call using 16k output tokens; actual use should be much lower.",
            }
        )
    return output
