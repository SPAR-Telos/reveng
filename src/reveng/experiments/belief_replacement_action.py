"""Belief-replacement action pilot for the matched-46 DoorKey cohort."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from reveng.experiments.behavioral_probe_runner import _behavioral_probe_preamble
from reveng.experiments.gpt_oss_activation_pilot import DEFAULT_MODEL_SNAPSHOT
from reveng.experiments.reasoningflow_transfer import gpu_status


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
BELIEF_IDS = (
    "wall_left",
    "wall_right",
    "wall_up",
    "wall_down",
    "has_key",
    "door_open",
)
FIVE_BELIEF_IDS = BELIEF_IDS[:-1]
BELIEF_LABELS = {
    "wall_left": "The cell immediately LEFT of the agent is a wall (#)",
    "wall_right": "The cell immediately RIGHT of the agent is a wall (#)",
    "wall_up": "The cell immediately ABOVE the agent is a wall (#)",
    "wall_down": "The cell immediately BELOW the agent is a wall (#)",
    "has_key": "The agent is carrying the key",
    "door_open": "At least one door is currently open",
}
PROBABILITY_LABELS = ("yes", "no", "unknown")
BELIEF_SET_IDS = ("six_state_beliefs", "five_state_beliefs")

BLUE = "#1769aa"
DARK_BLUE = "#0b3c5d"
GRID = "#e5eef5"
TEXT = "#172033"
SAME_BACKGROUND = "#dbeafe"
SAME_TEXT = "#0b3c5d"
DIFFERENT_BACKGROUND = "#ffedd5"
DIFFERENT_TEXT = "#9a3412"
GRID_CELL_COLORS = {
    "#": "#334155",
    "_": "#f8fafc",
    "A": "#2563eb",
    "G": "#16a34a",
    "K": "#eab308",
    "D": "#92400e",
    "O": "#0d9488",
}
GRID_CELL_LABELS = {
    "#": "Wall",
    "_": "Open cell",
    "A": "Agent",
    "G": "Goal",
    "K": "Key",
    "D": "Closed door",
    "O": "Open door",
}


@dataclass(frozen=True)
class PilotConfig:
    candidate_rows_path: Path
    prefix_action_rows_path: Path
    belief_rows_path: Path
    output_dir: Path
    model_path: Path = DEFAULT_MODEL_SNAPSHOT
    temperature: float = 0.7
    gpu_index: int = 0
    minimum_free_gib: float = 18.0
    seed: int = 42


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def parse_probabilities(value: Any, labels: Sequence[str]) -> dict[str, float]:
    """Parse and normalize a probability mapping over the requested labels."""
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("probabilities must be a mapping or JSON object")
    try:
        result = {label: float(value[label]) for label in labels}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"probabilities must contain numeric labels {tuple(labels)}"
        ) from exc
    if any(not math.isfinite(item) or item < 0 for item in result.values()):
        raise ValueError("probabilities must be finite and non-negative")
    total = sum(result.values())
    if total <= 0:
        raise ValueError("probabilities must have positive total mass")
    return {label: result[label] / total for label in labels}


def _preparation_config(config: PilotConfig) -> dict[str, Any]:
    return {
        "analysis": "belief_replacement_action_pilot_v1",
        "candidate_rows_path": str(config.candidate_rows_path.resolve()),
        "candidate_rows_sha256": _sha256(config.candidate_rows_path),
        "prefix_action_rows_path": str(config.prefix_action_rows_path.resolve()),
        "prefix_action_rows_sha256": _sha256(config.prefix_action_rows_path),
        "belief_rows_path": str(config.belief_rows_path.resolve()),
        "belief_rows_sha256": _sha256(config.belief_rows_path),
        "belief_ids": list(BELIEF_IDS),
        "selection": "two_per_action_one_all_correct_one_any_incorrect_distinct_trajectories_lexicographic",
        "seed": config.seed,
    }


def _query_config(config: PilotConfig, cohort_path: Path) -> dict[str, Any]:
    return {
        "analysis": "belief_replacement_action_pilot_v1",
        "cohort_sha256": _sha256(cohort_path),
        "model_path": str(config.model_path.resolve()),
        "temperature": config.temperature,
        "reasoning_effort": "low",
        "scoring": "direct_candidate_token_probabilities",
        "gpu_index": config.gpu_index,
        "minimum_free_gib": config.minimum_free_gib,
        "belief_set_ids": list(BELIEF_SET_IDS),
        "seed": config.seed,
    }


def _check_or_write_config(path: Path, expected: Mapping[str, Any]) -> None:
    if path.exists():
        observed = json.loads(path.read_text())
        if observed != dict(expected):
            raise ValueError(f"existing output uses a different configuration: {path}")
        return
    _write_json(path, dict(expected))


def _final_action_rows(actions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "example_id",
        "trajectory_id",
        "position_index",
        "action_label",
        "action_probabilities_json",
    }
    if missing := required - set(actions.columns):
        raise ValueError(f"prefix action table is missing columns: {sorted(missing)}")
    actions = actions.copy()
    actions["position_index"] = actions["position_index"].astype(int)
    maximum = actions.groupby("example_id")["position_index"].transform("max")
    final = actions.loc[actions["position_index"].eq(maximum)].copy()
    if final["example_id"].duplicated().any():
        raise ValueError("prefix action table has duplicate final positions")
    return final


def _initial_action_rows(actions: pd.DataFrame) -> pd.DataFrame:
    initial = actions.loc[actions["position_index"].astype(int).eq(0)].copy()
    if initial["example_id"].duplicated().any():
        raise ValueError("prefix action table has duplicate position-zero rows")
    return initial


def _final_belief_rows(
    beliefs: pd.DataFrame, final_positions: Mapping[str, int]
) -> pd.DataFrame:
    required = {
        "example_id",
        "trajectory_id",
        "position_index",
        "question_id",
        "belief_is_error",
        "probabilities_json",
    }
    if missing := required - set(beliefs.columns):
        raise ValueError(f"belief table is missing columns: {sorted(missing)}")
    beliefs = beliefs.loc[beliefs["question_id"].isin(BELIEF_IDS)].copy()
    beliefs["position_index"] = beliefs["position_index"].astype(int)
    expected = beliefs["example_id"].map(final_positions)
    beliefs = beliefs.loc[beliefs["position_index"].eq(expected)].copy()
    keys = ["example_id", "question_id"]
    if beliefs.duplicated(keys).any():
        raise ValueError("belief table has duplicate final-position belief rows")
    return beliefs


def select_smoke_cohort(
    candidates: pd.DataFrame,
    actions: pd.DataFrame,
    beliefs: pd.DataFrame,
) -> pd.DataFrame:
    """Select eight distinct traces balanced by final action and belief correctness."""
    required_candidates = {"example_id", "trajectory_id", "grid_text", "carrying_key"}
    if missing := required_candidates - set(candidates.columns):
        raise ValueError(f"candidate table is missing columns: {sorted(missing)}")
    final_actions = _final_action_rows(actions)
    initial_actions = _initial_action_rows(actions)
    final_positions = final_actions.set_index("example_id")["position_index"].to_dict()
    final_beliefs = _final_belief_rows(beliefs, final_positions)

    counts = final_beliefs.groupby("example_id")["question_id"].nunique()
    incomplete = counts[counts != len(BELIEF_IDS)]
    expected_examples = set(final_actions["example_id"])
    if set(counts.index) != expected_examples or not incomplete.empty:
        raise ValueError("every example must have all six final-position state beliefs")

    error_counts = (
        final_beliefs.assign(
            belief_is_error=final_beliefs["belief_is_error"].map(_parse_bool)
        )
        .groupby("example_id")["belief_is_error"]
        .sum()
        .astype(int)
    )
    metadata = candidates.merge(
        final_actions[
            [
                "example_id",
                "position_index",
                "action_label",
                "action_probabilities_json",
            ]
        ].rename(
            columns={
                "position_index": "final_position_index",
                "action_label": "full_cot_action",
                "action_probabilities_json": "full_cot_action_probabilities_json",
            }
        ),
        on="example_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        initial_actions[
            ["example_id", "action_label", "action_probabilities_json"]
        ].rename(
            columns={
                "action_label": "grid_only_action",
                "action_probabilities_json": "grid_only_action_probabilities_json",
            }
        ),
        on="example_id",
        how="inner",
        validate="one_to_one",
    )
    metadata["n_incorrect_state_beliefs"] = metadata["example_id"].map(error_counts)
    metadata["belief_error_group"] = np.where(
        metadata["n_incorrect_state_beliefs"].eq(0),
        "all_correct",
        "at_least_one_incorrect",
    )

    selected: list[pd.Series] = []
    used_trajectories: set[str] = set()
    for action in ACTIONS:
        for error_group in ("all_correct", "at_least_one_incorrect"):
            eligible = metadata.loc[
                metadata["full_cot_action"].eq(action)
                & metadata["belief_error_group"].eq(error_group)
                & ~metadata["trajectory_id"].astype(str).isin(used_trajectories)
            ].sort_values("example_id")
            if eligible.empty:
                raise ValueError(
                    f"cannot select distinct trajectory for {action}/{error_group}"
                )
            row = eligible.iloc[0].copy()
            selected.append(row)
            used_trajectories.add(str(row["trajectory_id"]))

    cohort = pd.DataFrame(selected).reset_index(drop=True)
    belief_lookup = {
        str(example_id): group.set_index("question_id")
        for example_id, group in final_beliefs.groupby("example_id", sort=False)
    }
    belief_payloads: list[str] = []
    for row in cohort.itertuples(index=False):
        group = belief_lookup[str(row.example_id)]
        payload: dict[str, dict[str, float | str]] = {}
        for question_id in BELIEF_IDS:
            probabilities = parse_probabilities(
                group.loc[question_id, "probabilities_json"], PROBABILITY_LABELS
            )
            payload[question_id] = {
                **probabilities,
                "top_label": max(probabilities, key=probabilities.get),
            }
        belief_payloads.append(json.dumps(payload, sort_keys=True))
    cohort["canonical_beliefs_json"] = belief_payloads
    cohort.insert(0, "smoke_order", np.arange(1, len(cohort) + 1))
    return cohort


def prepare_experiment(config: PilotConfig) -> Path:
    """Materialize the deterministic smoke cohort and lock its source configuration."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _check_or_write_config(
        config.output_dir / "preparation_manifest.json", _preparation_config(config)
    )
    cohort = select_smoke_cohort(
        pd.read_csv(config.candidate_rows_path),
        pd.read_csv(config.prefix_action_rows_path),
        pd.read_csv(config.belief_rows_path),
    )
    path = config.output_dir / "smoke_cohort.csv"
    cohort.to_csv(path, index=False)
    return path


def belief_ids_for_set(belief_set_id: str) -> tuple[str, ...]:
    if belief_set_id == "six_state_beliefs":
        return BELIEF_IDS
    if belief_set_id == "five_state_beliefs":
        return FIVE_BELIEF_IDS
    raise ValueError(f"unknown belief set: {belief_set_id}")


def build_replacement_context(
    *,
    grid_text: str,
    carrying_key: Any,
    beliefs: Mapping[str, Mapping[str, Any]],
    belief_set_id: str,
) -> str:
    """Render the intervention context without reasoning, truth, or an action label."""
    lines = [
        "The original reasoning trace is not available. It is replaced by the following",
        "belief readouts elicited at the end of that trace. The readouts may be uncertain",
        "or mistaken. Each line gives probabilities over yes, no, and unknown.",
    ]
    for question_id in belief_ids_for_set(belief_set_id):
        if question_id not in beliefs:
            raise ValueError(f"replacement block is missing belief {question_id}")
        probabilities = parse_probabilities(beliefs[question_id], PROBABILITY_LABELS)
        lines.append(
            f"- {BELIEF_LABELS[question_id]}: "
            f"yes={probabilities['yes']:.6f}, "
            f"no={probabilities['no']:.6f}, "
            f"unknown={probabilities['unknown']:.6f}"
        )
    return (
        _behavioral_probe_preamble("cardinal_action_explicit")
        + "\n# Inputs\n\nCurrent grid state:\n\n"
        + str(grid_text)
        + "\n\nAgent status:\n- Carrying key: "
        + str(_parse_bool(carrying_key)).lower()
        + "\n\nEnd-of-reasoning belief readouts:\n"
        + "\n".join(lines)
        + "\n\n"
    )


def action_question() -> tuple[str, str, tuple[str, ...]]:
    """Return the same immediate-action question used by the prefix experiment."""
    return (
        "action",
        "# Question\nChoose the next move that best advances toward the goal while respecting "
        "DoorKey mechanics. Return only the requested JSON action.",
        ACTIONS,
    )


def _blocked_report(config: PilotConfig, status: Mapping[str, Any]) -> Path:
    report = config.output_dir / "run_report.md"
    details = str(status.get("message", "Insufficient free GPU memory."))
    processes = status.get("processes", [])
    process_text = ""
    if processes:
        process_text = "\n\nGPU processes reported by `nvidia-smi`:\n\n" + "\n".join(
            f"- `{process}`" for process in processes
        )
    report.write_text(
        "# Belief-replacement action pilot\n\n"
        f"**Status: `{status['status']}`.**\n\n"
        "The deterministic eight-trace cohort was prepared, but the replacement-action "
        "queries were not run. No numerical action-belief result has been produced.\n\n"
        f"GPU check: {details}{process_text}\n"
    )
    return report


def query_experiment(
    config: PilotConfig,
    *,
    reader_factory: Callable[..., Any] | None = None,
    gpu_status_fn: Callable[[int, float], Mapping[str, Any]] = gpu_status,
) -> dict[str, Any]:
    """Query replacement actions, resuming completed example/belief-set pairs."""
    cohort_path = config.output_dir / "smoke_cohort.csv"
    if not cohort_path.exists():
        raise FileNotFoundError("run the prepare stage before query")
    query_config = _query_config(config, cohort_path)
    _check_or_write_config(config.output_dir / "query_config.json", query_config)
    checkpoint = config.output_dir / "replacement_action_readouts.jsonl"
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            completed[(str(row["example_id"]), str(row["belief_set_id"]))] = row

    cohort = pd.read_csv(cohort_path)
    planned = [
        (str(row.example_id), belief_set_id)
        for row in cohort.itertuples(index=False)
        for belief_set_id in BELIEF_SET_IDS
    ]
    pending = [key for key in planned if key not in completed]
    manifest = {
        **query_config,
        "planned_readouts": len(planned),
        "completed_readouts": len(completed),
        "pending_readouts": len(pending),
    }
    if not pending:
        manifest["status"] = "completed"
        _write_json(config.output_dir / "run_manifest.json", manifest)
        return manifest

    status = dict(gpu_status_fn(config.gpu_index, config.minimum_free_gib))
    manifest.update(status)
    _write_json(config.output_dir / "run_manifest.json", manifest)
    if status.get("status") != "ready":
        _blocked_report(config, status)
        return manifest

    if reader_factory is None:
        if config.gpu_index != 0:
            raise ValueError(
                "LocalImmediateReadout currently addresses visible GPU 0; launch with "
                "CUDA_VISIBLE_DEVICES set to the desired physical device."
            )
        from reveng.experiments.local_immediate_behavioral import LocalImmediateReadout

        reader_factory = LocalImmediateReadout
    reader = reader_factory(config.model_path, temperature=config.temperature)
    cohort_lookup = cohort.set_index("example_id")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("a") as handle:
        for example_id, belief_set_id in pending:
            row = cohort_lookup.loc[example_id]
            beliefs = json.loads(str(row["canonical_beliefs_json"]))
            context = build_replacement_context(
                grid_text=str(row["grid_text"]),
                carrying_key=row["carrying_key"],
                beliefs=beliefs,
                belief_set_id=belief_set_id,
            )
            result = reader.score_questions(
                context=context,
                questions=[action_question()],
            )["action"]
            output = {
                "example_id": example_id,
                "trajectory_id": str(row["trajectory_id"]),
                "belief_set_id": belief_set_id,
                "replacement_action": str(result["answer"]),
                "replacement_action_probabilities": result["probabilities"],
                "replacement_action_entropy_bits": float(result["entropy_bits"]),
                "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                "belief_ids": list(belief_ids_for_set(belief_set_id)),
            }
            handle.write(json.dumps(output, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            completed[(example_id, belief_set_id)] = output
            manifest["completed_readouts"] = len(completed)
            manifest["pending_readouts"] = len(planned) - len(completed)
            _write_json(config.output_dir / "run_manifest.json", manifest)
    manifest["status"] = "completed"
    _write_json(config.output_dir / "run_manifest.json", manifest)
    return manifest


def total_variation(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    p = parse_probabilities(first, ACTIONS)
    q = parse_probabilities(second, ACTIONS)
    return 0.5 * sum(abs(p[action] - q[action]) for action in ACTIONS)


def jensen_shannon_divergence(
    first: Mapping[str, float], second: Mapping[str, float]
) -> float:
    """Return Jensen-Shannon divergence in bits."""
    p = parse_probabilities(first, ACTIONS)
    q = parse_probabilities(second, ACTIONS)
    midpoint = {action: 0.5 * (p[action] + q[action]) for action in ACTIONS}

    def kl(values: Mapping[str, float], reference: Mapping[str, float]) -> float:
        return sum(
            values[action] * math.log2(values[action] / reference[action])
            for action in ACTIONS
            if values[action] > 0
        )

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def build_analysis_rows(
    cohort: pd.DataFrame, readouts: Sequence[Mapping[str, Any]]
) -> pd.DataFrame:
    lookup = {
        (str(row["example_id"]), str(row["belief_set_id"])): row for row in readouts
    }
    output: list[dict[str, Any]] = []
    for source in cohort.to_dict("records"):
        full_probabilities = parse_probabilities(
            source["full_cot_action_probabilities_json"], ACTIONS
        )
        grid_probabilities = parse_probabilities(
            source["grid_only_action_probabilities_json"], ACTIONS
        )
        grid_disagrees = source["grid_only_action"] != source["full_cot_action"]
        for belief_set_id in BELIEF_SET_IDS:
            key = (str(source["example_id"]), belief_set_id)
            if key not in lookup:
                raise ValueError(f"missing replacement action readout for {key}")
            readout = lookup[key]
            replacement_probabilities = parse_probabilities(
                readout["replacement_action_probabilities"], ACTIONS
            )
            replacement_disagrees = (
                readout["replacement_action"] != source["full_cot_action"]
            )
            output.append(
                {
                    "smoke_order": int(source["smoke_order"]),
                    "example_id": source["example_id"],
                    "trajectory_id": source["trajectory_id"],
                    "belief_error_group": source["belief_error_group"],
                    "n_incorrect_state_beliefs": int(
                        source["n_incorrect_state_beliefs"]
                    ),
                    "belief_set_id": belief_set_id,
                    "full_cot_action": source["full_cot_action"],
                    "grid_only_action": source["grid_only_action"],
                    "belief_replacement_action": readout["replacement_action"],
                    "grid_only_disagrees_with_full_cot": bool(grid_disagrees),
                    "belief_replacement_disagrees_with_full_cot": bool(
                        replacement_disagrees
                    ),
                    "belief_rescues_full_cot_action": bool(
                        grid_disagrees and not replacement_disagrees
                    ),
                    "full_cot_action_probabilities_json": json.dumps(
                        full_probabilities, sort_keys=True
                    ),
                    "grid_only_action_probabilities_json": json.dumps(
                        grid_probabilities, sort_keys=True
                    ),
                    "belief_replacement_action_probabilities_json": json.dumps(
                        replacement_probabilities, sort_keys=True
                    ),
                    "grid_only_total_variation_from_full_cot": total_variation(
                        grid_probabilities, full_probabilities
                    ),
                    "belief_replacement_total_variation_from_full_cot": total_variation(
                        replacement_probabilities, full_probabilities
                    ),
                    "grid_only_js_divergence_from_full_cot_bits": jensen_shannon_divergence(
                        grid_probabilities, full_probabilities
                    ),
                    "belief_replacement_js_divergence_from_full_cot_bits": jensen_shannon_divergence(
                        replacement_probabilities, full_probabilities
                    ),
                }
            )
    return pd.DataFrame(output).sort_values(["belief_set_id", "smoke_order"])


def summarize_analysis(rows: pd.DataFrame) -> pd.DataFrame:
    primary = rows.loc[rows["belief_set_id"].eq("six_state_beliefs")]
    records = [
        {
            "condition": "grid_only",
            "n_traces": len(primary),
            "n_disagree_with_full_cot": int(
                primary["grid_only_disagrees_with_full_cot"].sum()
            ),
            "disagreement_rate": float(
                primary["grid_only_disagrees_with_full_cot"].mean()
            ),
            "mean_total_variation_from_full_cot": float(
                primary["grid_only_total_variation_from_full_cot"].mean()
            ),
            "mean_js_divergence_from_full_cot_bits": float(
                primary["grid_only_js_divergence_from_full_cot_bits"].mean()
            ),
            "belief_rescue_denominator": np.nan,
            "belief_rescue_n": np.nan,
            "belief_rescue_rate": np.nan,
        }
    ]
    for belief_set_id in BELIEF_SET_IDS:
        subset = rows.loc[rows["belief_set_id"].eq(belief_set_id)]
        rescue_eligible = subset.loc[subset["grid_only_disagrees_with_full_cot"]]
        rescue_n = int(rescue_eligible["belief_rescues_full_cot_action"].sum())
        records.append(
            {
                "condition": belief_set_id,
                "n_traces": len(subset),
                "n_disagree_with_full_cot": int(
                    subset["belief_replacement_disagrees_with_full_cot"].sum()
                ),
                "disagreement_rate": float(
                    subset["belief_replacement_disagrees_with_full_cot"].mean()
                ),
                "mean_total_variation_from_full_cot": float(
                    subset["belief_replacement_total_variation_from_full_cot"].mean()
                ),
                "mean_js_divergence_from_full_cot_bits": float(
                    subset["belief_replacement_js_divergence_from_full_cot_bits"].mean()
                ),
                "belief_rescue_denominator": len(rescue_eligible),
                "belief_rescue_n": rescue_n,
                "belief_rescue_rate": (
                    rescue_n / len(rescue_eligible) if len(rescue_eligible) else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def _plot_primary(summary: pd.DataFrame, output: Path) -> None:
    selected = summary.set_index("condition").loc[["grid_only", "six_state_beliefs"]]
    rates = selected["disagreement_rate"].to_numpy(float)
    counts = selected["n_disagree_with_full_cot"].to_numpy(int)
    totals = selected["n_traces"].to_numpy(int)
    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 9,
            "figure.dpi": 160,
        }
    ):
        figure, axis = plt.subplots(figsize=(6.6, 4.0))
        bars = axis.bar(
            ["Task input only\n(no CoT)", "Six-belief replacement\n(no CoT)"],
            rates,
            color=["#9bd0f5", BLUE],
            edgecolor=DARK_BLUE,
            linewidth=0.8,
            width=0.62,
        )
        for bar, count, total, rate in zip(bars, counts, totals, rates):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(rate + 0.035, 0.96),
                f"{count}/{total} ({rate:.0%})",
                ha="center",
                va="bottom",
                color=TEXT,
            )
        axis.set_ylim(0, 1.08)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_ylabel(
            "Traces selecting a different action\nfrom the full-CoT condition (%)"
        )
        axis.set_title("Does the belief replacement reproduce the full-CoT action?")
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
        plt.close(figure)


def parse_grid_text(value: str) -> list[list[str]]:
    """Parse a coordinate-labelled DoorKey grid into its cell symbols."""
    lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("grid must contain a coordinate header and at least one row")
    expected_columns = lines[0].split()
    cells: list[list[str]] = []
    for row_index, line in enumerate(lines[1:]):
        fields = line.split()
        if len(fields) != len(expected_columns) + 1:
            raise ValueError(f"grid row {row_index} has an unexpected number of cells")
        row = fields[1:]
        unknown = set(row) - set(GRID_CELL_COLORS)
        if unknown:
            raise ValueError(
                f"grid contains unsupported cell symbols: {sorted(unknown)}"
            )
        cells.append(row)
    return cells


def _draw_grid(axis: Any, grid_text: str) -> None:
    cells = parse_grid_text(grid_text)
    n_rows, n_columns = len(cells), len(cells[0])
    for row_index, row in enumerate(cells):
        if len(row) != n_columns:
            raise ValueError("grid rows must have equal lengths")
        for column_index, symbol in enumerate(row):
            axis.add_patch(
                Rectangle(
                    (column_index, row_index),
                    1,
                    1,
                    facecolor=GRID_CELL_COLORS[symbol],
                    edgecolor="#cbd5e1",
                    linewidth=0.35,
                )
            )
            if symbol != "_":
                axis.text(
                    column_index + 0.5,
                    row_index + 0.52,
                    symbol,
                    ha="center",
                    va="center",
                    fontsize=6.3,
                    fontweight="bold",
                    color="white" if symbol in {"#", "A", "G", "D", "O"} else TEXT,
                )
    axis.set_xlim(0, n_columns)
    axis.set_ylim(n_rows, 0)
    axis.set_aspect("equal")
    axis.axis("off")


def _draw_action_comparison(
    axis: Any, *, selected_action: str, full_cot_action: str
) -> None:
    agrees = selected_action == full_cot_action
    axis.set_facecolor(SAME_BACKGROUND if agrees else DIFFERENT_BACKGROUND)
    comparison = "=" if agrees else "≠"
    status = "SAME ACTION" if agrees else "DIFFERENT ACTION"
    color = SAME_TEXT if agrees else DIFFERENT_TEXT
    axis.text(
        0.5,
        0.58,
        f"{selected_action} {comparison} {full_cot_action}",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=color,
    )
    axis.text(
        0.5,
        0.31,
        status,
        ha="center",
        va="center",
        fontsize=8.5,
        color=color,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(color)
        spine.set_linewidth(0.8)


def _plot_trace_level(rows: pd.DataFrame, cohort: pd.DataFrame, output: Path) -> None:
    primary = (
        rows.loc[rows["belief_set_id"].eq("six_state_beliefs")]
        .merge(
            cohort[["example_id", "grid_text", "carrying_key"]],
            on="example_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values("smoke_order")
    )
    if len(primary) != 8 or primary["grid_text"].isna().any():
        raise ValueError("trace-level figure requires all eight smoke-test grids")
    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "figure.dpi": 160,
        }
    ):
        figure, axes = plt.subplots(
            len(primary),
            3,
            figsize=(10.8, 11.8),
            gridspec_kw={
                "width_ratios": [1.45, 2.15, 2.15],
                "hspace": 0.34,
                "wspace": 0.22,
            },
        )
        for row_index, row in enumerate(primary.itertuples(index=False)):
            grid_axis, task_axis, belief_axis = axes[row_index]
            _draw_grid(grid_axis, row.grid_text)
            quality = (
                "all six readouts correct"
                if row.belief_error_group == "all_correct"
                else f"{row.n_incorrect_state_beliefs} incorrect readout"
                + ("s" if row.n_incorrect_state_beliefs != 1 else "")
            )
            carrying = "yes" if _parse_bool(row.carrying_key) else "no"
            grid_axis.text(
                -0.12,
                0.5,
                f"Trace {row.smoke_order}\n{quality}\nCarrying key: {carrying}",
                transform=grid_axis.transAxes,
                ha="right",
                va="center",
                fontsize=8.2,
                color=TEXT,
            )
            _draw_action_comparison(
                task_axis,
                selected_action=row.grid_only_action,
                full_cot_action=row.full_cot_action,
            )
            _draw_action_comparison(
                belief_axis,
                selected_action=row.belief_replacement_action,
                full_cot_action=row.full_cot_action,
            )
        axes[0, 0].set_title("Grid state", fontweight="bold", pad=8)
        axes[0, 1].set_title("Task input only (no CoT)", fontweight="bold", pad=8)
        axes[0, 2].set_title(
            "Six-belief replacement (no CoT)", fontweight="bold", pad=8
        )
        figure.suptitle(
            "Does each condition reproduce the action selected after full CoT?",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )
        figure.text(
            0.60,
            0.969,
            "Each cell compares the condition's selected action (left) with the full-CoT action (right).",
            ha="center",
            va="top",
            fontsize=9,
            color=TEXT,
        )
        present_symbols = {
            symbol
            for grid in primary["grid_text"]
            for row in parse_grid_text(grid)
            for symbol in row
        }
        legend = [
            Patch(
                facecolor=GRID_CELL_COLORS[symbol],
                edgecolor="#cbd5e1",
                label=f"{symbol}  {GRID_CELL_LABELS[symbol]}",
            )
            for symbol in ("#", "_", "A", "G", "K", "D", "O")
            if symbol in present_symbols
        ]
        figure.legend(
            handles=legend,
            loc="lower center",
            ncol=len(legend),
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.56, 0.006),
        )
        figure.subplots_adjust(top=0.94, bottom=0.055, left=0.22, right=0.98)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
        plt.close(figure)


def _format_probability(value: float) -> str:
    return f"{value:.3f}"


def _write_report(rows: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    indexed = summary.set_index("condition")
    grid = indexed.loc["grid_only"]
    six = indexed.loc["six_state_beliefs"]
    five = indexed.loc["five_state_beliefs"]
    primary = rows.loc[rows["belief_set_id"].eq("six_state_beliefs")].sort_values(
        "smoke_order"
    )
    table_lines = [
        "| Trace | Final belief quality | Task input only (no CoT) | Full CoT | Six-belief replacement | Replacement agrees? |",
        "|---:|---|---|---|---|---|",
    ]
    for row in primary.itertuples(index=False):
        agrees = "No" if row.belief_replacement_disagrees_with_full_cot else "Yes"
        table_lines.append(
            f"| {row.smoke_order} | {str(row.belief_error_group).replace('_', ' ')} "
            f"({row.n_incorrect_state_beliefs} errors) | {row.grid_only_action} | "
            f"{row.full_cot_action} | {row.belief_replacement_action} | {agrees} |"
        )
    rescue_denominator = int(six["belief_rescue_denominator"])
    rescue_n = int(six["belief_rescue_n"])
    rescue_text = (
        f"{rescue_n}/{rescue_denominator} ({six['belief_rescue_rate']:.0%})"
        if rescue_denominator
        else "not estimable because grid-only never differed from full CoT"
    )
    output.write_text(
        "# Belief-replacement action pilot\n\n"
        "## Question\n\n"
        "Can six action-independent beliefs read out at the end of a reasoning trace "
        "replace that trace while preserving the model's selected action? The original "
        "grid and agent status remain visible in every condition.\n\n"
        "## Smoke-test result\n\n"
        f"With the task input only (no CoT), {int(grid['n_disagree_with_full_cot'])}/"
        f"{int(grid['n_traces'])} traces ({grid['disagreement_rate']:.0%}) selected a "
        "different action from the full-CoT condition. With the six-belief replacement, "
        f"{int(six['n_disagree_with_full_cot'])}/{int(six['n_traces'])} "
        f"({six['disagreement_rate']:.0%}) differed. The belief rescue rate was "
        f"{rescue_text}.\n\n"
        f"The mean total-variation distance from the full-CoT action distribution was "
        f"{_format_probability(grid['mean_total_variation_from_full_cot'])} for task input only "
        f"and {_format_probability(six['mean_total_variation_from_full_cot'])} for the "
        "six-belief replacement. Lower is closer. The replacement therefore recovered "
        "the full-CoT action in one additional trace, but its action probabilities were "
        "slightly farther from the full-CoT probabilities on average. Taken together, "
        "this smoke test does not yet provide clear evidence that the six beliefs recover "
        "the action-relevant content of the omitted reasoning.\n\n"
        "![Overall disagreement with the full-CoT action](figures/action_disagreement.png)\n\n"
        "## Trace-level actions\n\n"
        "The figure shows the exact grid and both action comparisons for every smoke-test trace. "
        "In each comparison cell, the action on the left is selected under that condition and "
        "the action on the right is the full-CoT reference.\n\n"
        "![Trace-level action comparisons with grid states](figures/trace_level_action_disagreement.png)\n\n"
        + "\n".join(table_lines)
        + "\n\n## Five-belief sensitivity check\n\n"
        f"After removing the door-status belief, {int(five['n_disagree_with_full_cot'])}/"
        f"{int(five['n_traces'])} traces ({five['disagreement_rate']:.0%}) differed from "
        "the full-CoT action.\n\n"
        "## Interpretation limits\n\n"
        "This is an eight-trace smoke test, not a conclusive estimate. Agreement shows "
        "only that this measured belief block is sufficient to reproduce an action in "
        "the replacement prompt. It does not show that the beliefs caused or mediated "
        "the original action. If belief replacement performs no better than task input only, "
        "the current panel provides no evidence that it carries the action-relevant "
        "content of the omitted reasoning. The panel also omits route, goal-direction, "
        "and action-value beliefs.\n"
    )


def analyze_experiment(config: PilotConfig) -> dict[str, Path]:
    cohort_path = config.output_dir / "smoke_cohort.csv"
    checkpoint = config.output_dir / "replacement_action_readouts.jsonl"
    if not cohort_path.exists() or not checkpoint.exists():
        raise FileNotFoundError(
            "prepare and query stages must complete before analysis"
        )
    readouts = [
        json.loads(line) for line in checkpoint.read_text().splitlines() if line
    ]
    cohort = pd.read_csv(cohort_path)
    rows = build_analysis_rows(cohort, readouts)
    summary = summarize_analysis(rows)
    rows_path = config.output_dir / "action_belief_gap_rows.csv"
    summary_path = config.output_dir / "summary.csv"
    figure_path = config.output_dir / "figures" / "action_disagreement.png"
    trace_figure_path = (
        config.output_dir / "figures" / "trace_level_action_disagreement.png"
    )
    report_path = config.output_dir / "run_report.md"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    _plot_primary(summary, figure_path)
    _plot_trace_level(rows, cohort, trace_figure_path)
    _write_report(rows, summary, report_path)
    return {
        "cohort": cohort_path,
        "readouts": checkpoint,
        "rows": rows_path,
        "summary": summary_path,
        "figure": figure_path,
        "trace_figure": trace_figure_path,
        "report": report_path,
    }


def run_all(config: PilotConfig) -> dict[str, Any]:
    prepare_experiment(config)
    manifest = query_experiment(config)
    if manifest.get("status") != "completed":
        return {"manifest": config.output_dir / "run_manifest.json", "status": manifest}
    return {"paths": analyze_experiment(config), "status": manifest}


__all__ = [
    "ACTIONS",
    "BELIEF_IDS",
    "BELIEF_SET_IDS",
    "PilotConfig",
    "action_question",
    "analyze_experiment",
    "belief_ids_for_set",
    "build_analysis_rows",
    "build_replacement_context",
    "jensen_shannon_divergence",
    "parse_probabilities",
    "parse_grid_text",
    "prepare_experiment",
    "query_experiment",
    "run_all",
    "select_smoke_cohort",
    "summarize_analysis",
    "total_variation",
]
