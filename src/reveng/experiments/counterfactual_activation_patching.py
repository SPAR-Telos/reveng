"""Counterfactual activation-patching experiment runner.

This module evaluates whether patched activations (produced externally) cause
counterfactual navigation behavior toward a target environment (grid B).
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from reveng.analysis.analysis_utils import (
    ACTION_NAME_TO_ID,
    compute_optimal_actions_from_text_grid,
)
from reveng.experiments.counterfactual_artifact_builder import (
    LAYER_KEY_DEFAULT,
    ArtifactPairSpec,
    _extract_agent_goal,
    _parse_coord,
    _parse_grid_text_file,
    map_action_a_to_b,
    map_position_a_to_b,
    normalize_counterfactual_category,
    validate_counterfactual_pair,
)

EXPECTED_K = 10
ACTION_TRUE_THRESHOLD = 0.70
DISRUPTIVE_THRESHOLD = 0.35


@dataclass
class GridPairSpec:
    pair_id: str
    category: str
    grid_a_path: Path
    grid_b_path: Path
    goal_a: tuple[int, int]
    goal_b: tuple[int, int]


@dataclass
class RunArtifacts:
    a_trace_path: Path
    b_trace_path: Path
    patched_trace_path: Path


@dataclass
class PairMetrics:
    pair_id: str
    category: str
    evaluated_steps: int
    a_target: float
    a_base: float
    action_label: Optional[bool]
    disruptive: Optional[bool]
    belief_mlp_match_target: Optional[bool]
    belief_linear_match_target: Optional[bool]
    belief_available_mlp: bool
    belief_available_linear: bool
    outcome_cell_mlp: Optional[tuple[bool, bool]]
    outcome_cell_linear: Optional[tuple[bool, bool]]
    valid_pair: bool
    invalid_reason: Optional[str]
    disruptive_reason: Optional[str] = None


@dataclass
class AggregateMetrics:
    total_pairs_manifest: int
    total_pairs_processed: int
    total_pairs_valid: int
    total_pairs_invalid: int
    total_pairs_disruptive: int
    disruptive_rate: float
    total_pairs_action_evaluable: int
    total_pairs_action_true: int
    action_true_rate: Optional[float]
    table_rows_mlp: int
    table_rows_linear: int
    total_pairs_belief_available: int  # Deprecated alias for table_rows_mlp
    mlp_tt_count: int
    mlp_tf_count: int
    mlp_ft_count: int
    mlp_ff_count: int
    mlp_tt_over_tt_tf: Optional[float]
    mlp_tt_over_tt_ft: Optional[float]
    linear_tt_count: int
    linear_tf_count: int
    linear_ft_count: int
    linear_ff_count: int
    linear_tt_over_tt_tf: Optional[float]
    linear_tt_over_tt_ft: Optional[float]
    probe_both_available: int
    probe_agree_count: int
    probe_disagree_count: int
    probe_disagreement_rate: Optional[float]
    per_category: dict[str, dict[str, Any]]
    stopped_early: bool
    early_stop_reason: Optional[str]


@dataclass
class PairRecord:
    spec: GridPairSpec
    artifacts: RunArtifacts


def _read_manifest(manifest_path: Path) -> list[PairRecord]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    entries: list[dict[str, Any]]
    if manifest_path.suffix.lower() == ".json":
        data = json.loads(manifest_path.read_text())
        if isinstance(data, dict) and "pairs" in data:
            entries = data["pairs"]
        elif isinstance(data, list):
            entries = data
        else:
            raise ValueError("JSON manifest must be a list or {'pairs': [...]}.")
    elif manifest_path.suffix.lower() == ".csv":
        with open(manifest_path, newline="") as f:
            entries = list(csv.DictReader(f))
    else:
        raise ValueError("Manifest must be .json or .csv")

    records: list[PairRecord] = []
    for idx, row in enumerate(entries):
        pair_id = str(row.get("pair_id", f"pair_{idx:03d}"))
        goal_a_raw = row.get("goal_a", row.get("goal_orig"))
        goal_b_raw = row.get("goal_b", row.get("goal_new"))
        if goal_a_raw is None or goal_b_raw is None:
            raise ValueError(
                f"pair={pair_id}: missing goal_a/goal_b (legacy goal_orig/goal_new accepted)."
            )

        spec = GridPairSpec(
            pair_id=pair_id,
            category=normalize_counterfactual_category(row.get("category", "goal_move")),
            grid_a_path=Path(str(row["grid_a_path"])),
            grid_b_path=Path(str(row["grid_b_path"])),
            goal_a=_parse_coord(goal_a_raw),
            goal_b=_parse_coord(goal_b_raw),
        )
        artifacts = RunArtifacts(
            a_trace_path=Path(str(row["a_trace_path"])),
            b_trace_path=Path(str(row["b_trace_path"])),
            patched_trace_path=Path(str(row["patched_trace_path"])),
        )
        records.append(PairRecord(spec=spec, artifacts=artifacts))

    return records


def _resolve_trace_path(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        json_files = sorted(path.glob("*.json"))
        if len(json_files) != 1:
            raise ValueError(
                f"Expected exactly one JSON trace in {path}, found {len(json_files)}"
            )
        return json_files[0]
    raise FileNotFoundError(f"Trace path not found: {path}")


def _load_trace(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _action_to_id(action: Any) -> Optional[int]:
    if isinstance(action, int) and action in {0, 1, 2, 3}:
        return action
    if isinstance(action, str):
        cleaned = action.strip().upper()
        if cleaned in ACTION_NAME_TO_ID:
            return ACTION_NAME_TO_ID[cleaned]
        if cleaned.isdigit() and int(cleaned) in {0, 1, 2, 3}:
            return int(cleaned)
    return None


def _parse_grid_state_to_layout(grid_state: list[str]) -> list[list[str]]:
    allowed_cells = {"#", "_", "A", "G"}
    saw_header = False
    rows: list[list[str]] = []
    for line in grid_state:
        parts = line.strip().split()
        if not parts:
            continue
        if all(p.lstrip("-").isdigit() for p in parts):
            saw_header = True
            continue
        if not parts[0].lstrip("-").isdigit() or len(parts) <= 1:
            raise ValueError("Invalid grid row format (missing row index)")
        parts = parts[1:]
        if any(cell not in allowed_cells for cell in parts):
            raise ValueError("Invalid grid cell token detected")
        rows.append(parts)

    if not saw_header or not rows:
        raise ValueError("Unable to parse grid_state into layout")

    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("Inconsistent row widths in grid_state")

    return rows


def _validate_pair_spec_against_files(spec: GridPairSpec) -> Optional[str]:
    try:
        layout_a = _parse_grid_text_file(spec.grid_a_path)
        layout_b = _parse_grid_text_file(spec.grid_b_path)
    except Exception as exc:
        return f"failed to parse grid files: {exc}"

    try:
        validate_counterfactual_pair(
            ArtifactPairSpec(
                pair_id=spec.pair_id,
                category=spec.category,
                grid_a_path=spec.grid_a_path,
                grid_b_path=spec.grid_b_path,
                goal_a=spec.goal_a,
                goal_b=spec.goal_b,
            ),
            layout_a,
            layout_b,
        )
    except Exception as exc:
        return str(exc)

    return None


def _has_required_step_fields(trace: dict[str, Any]) -> bool:
    steps = trace.get("steps", [])
    if not steps:
        return False
    for step in steps:
        if "grid_state" in step and "agent_action" in step:
            return True
    return False


def _validate_patch_metadata(trace: dict[str, Any], layer_key: str) -> Optional[str]:
    metadata = trace.get("patch_metadata")
    if not isinstance(metadata, dict):
        return "missing patch_metadata"

    pre_n = metadata.get("pre_reasoning_last_n")
    post_n = metadata.get("post_reasoning_last_n")
    hook_tensor = metadata.get("hook_tensor")

    if pre_n != 3 or post_n != 3:
        return "patch_metadata must set pre_reasoning_last_n=3 and post_reasoning_last_n=3"
    if hook_tensor != layer_key:
        return f"patch_metadata hook_tensor must be {layer_key}"

    return None


def _collect_probe_dicts_from_step(step: dict[str, Any]) -> list[dict[str, Any]]:
    probe_dicts: list[dict[str, Any]] = []

    pre_fields = [
        "pre_reasoning_prompt_suffix_tokens",
        "prompt_suffix_tokens_pre_reasoning",
        "pre_reasoning_prompt_tokens",
    ]
    post_fields = ["post_reasoning_prompt_suffix_tokens", "prompt_suffix_tokens"]

    for field in pre_fields:
        tokens = step.get(field)
        if isinstance(tokens, list):
            for token in tokens[-3:]:
                probes = token.get("probes") if isinstance(token, dict) else None
                if isinstance(probes, dict):
                    probe_dicts.append(probes)
            break

    for field in post_fields:
        tokens = step.get(field)
        if isinstance(tokens, list):
            for token in tokens[-3:]:
                probes = token.get("probes") if isinstance(token, dict) else None
                if isinstance(probes, dict):
                    probe_dicts.append(probes)
            break

    return probe_dicts


def _decode_goal_from_probes(
    probe_dicts: list[dict[str, Any]],
    layer_key: str,
    probe_type: str,
) -> Optional[tuple[int, int]]:
    """Decode goal coordinate from probe payloads.

    Returns (x, y), using argmax of aggregated goal probability across selected tokens.
    """
    if probe_type not in {"mlp", "linear"}:
        raise ValueError("probe_type must be 'mlp' or 'linear'")

    coord_goal_scores: dict[tuple[int, int], list[float]] = {}
    matched_any_probe = False

    for probes in probe_dicts:
        for probe_key, probe_value in probes.items():
            key_l = probe_key.lower()
            is_mlp = "_mlp_" in key_l
            is_linear = ("_linear_" in key_l) or ("_lin_" in key_l)
            if probe_type == "mlp" and not is_mlp:
                continue
            if probe_type == "linear" and not is_linear:
                continue

            match = re.search(r"_r(-?\d+)_c(-?\d+)", probe_key)
            if not match:
                continue

            y = int(match.group(1))
            x = int(match.group(2))

            layer_preds = {}
            if isinstance(probe_value, dict):
                layer_preds = probe_value.get(layer_key, {})
            if not isinstance(layer_preds, dict):
                continue

            goal_prob = layer_preds.get("goal")
            if goal_prob is None:
                continue

            matched_any_probe = True
            coord_goal_scores.setdefault((x, y), []).append(float(goal_prob))

    if not matched_any_probe or not coord_goal_scores:
        return None

    best_coord = None
    best_score = -math.inf
    for coord, scores in coord_goal_scores.items():
        score = sum(scores) / len(scores)
        if score > best_score:
            best_score = score
            best_coord = coord

    return best_coord


def _compute_action_metrics(
    patched_trace: dict[str, Any],
    spec: GridPairSpec,
    layout_a: list[list[str]],
    layout_b: list[list[str]],
) -> tuple[int, float, float]:
    steps = patched_trace.get("steps", [])
    aligned_target = 0
    aligned_base = 0
    evaluated_steps = 0

    try:
        optimal_base, _ = compute_optimal_actions_from_text_grid(layout_a, spec.goal_a)
        optimal_target, _ = compute_optimal_actions_from_text_grid(layout_b, spec.goal_b)
    except Exception:
        return 0, 0.0, 0.0

    height_a = len(layout_a)
    width_a = len(layout_a[0]) if height_a else 0

    for step in steps:
        grid_state = step.get("grid_state")
        action_raw = step.get("agent_action")
        if not isinstance(grid_state, list):
            continue
        action_a = _action_to_id(action_raw)
        if action_a is None:
            continue

        try:
            layout_step = _parse_grid_state_to_layout(grid_state)
            agent_pos_a, _ = _extract_agent_goal(layout_step)
        except Exception:
            continue

        try:
            agent_pos_b = map_position_a_to_b(
                agent_pos_a,
                spec.category,
                width=width_a,
                height=height_a,
            )
            action_b = map_action_a_to_b(action_a, spec.category)
        except Exception:
            continue

        if agent_pos_a not in optimal_base or agent_pos_b not in optimal_target:
            continue

        evaluated_steps += 1
        if action_b in optimal_target[agent_pos_b]:
            aligned_target += 1
        if action_a in optimal_base[agent_pos_a]:
            aligned_base += 1

    if evaluated_steps == 0:
        return 0, 0.0, 0.0

    a_target = aligned_target / evaluated_steps
    a_base = aligned_base / evaluated_steps
    return evaluated_steps, a_target, a_base


def evaluate_pair(
    record: PairRecord,
    layer_key: str = LAYER_KEY_DEFAULT,
    action_true_threshold: float = ACTION_TRUE_THRESHOLD,
    disruptive_threshold: float = DISRUPTIVE_THRESHOLD,
) -> PairMetrics:
    spec = record.spec
    artifacts = record.artifacts

    if not spec.grid_a_path.exists() or not spec.grid_b_path.exists():
        return PairMetrics(
            pair_id=spec.pair_id,
            category=spec.category,
            evaluated_steps=0,
            a_target=0.0,
            a_base=0.0,
            action_label=None,
            disruptive=None,
            belief_mlp_match_target=None,
            belief_linear_match_target=None,
            belief_available_mlp=False,
            belief_available_linear=False,
            outcome_cell_mlp=None,
            outcome_cell_linear=None,
            valid_pair=False,
            invalid_reason="missing grid_a_path or grid_b_path",
        )

    relationship_error = _validate_pair_spec_against_files(spec)
    if relationship_error is not None:
        return PairMetrics(
            pair_id=spec.pair_id,
            category=spec.category,
            evaluated_steps=0,
            a_target=0.0,
            a_base=0.0,
            action_label=None,
            disruptive=None,
            belief_mlp_match_target=None,
            belief_linear_match_target=None,
            belief_available_mlp=False,
            belief_available_linear=False,
            outcome_cell_mlp=None,
            outcome_cell_linear=None,
            valid_pair=False,
            invalid_reason=relationship_error,
        )

    try:
        a_trace = _load_trace(_resolve_trace_path(artifacts.a_trace_path))
        b_trace = _load_trace(_resolve_trace_path(artifacts.b_trace_path))
        patched_trace = _load_trace(_resolve_trace_path(artifacts.patched_trace_path))
    except Exception as exc:
        return PairMetrics(
            pair_id=spec.pair_id,
            category=spec.category,
            evaluated_steps=0,
            a_target=0.0,
            a_base=0.0,
            action_label=None,
            disruptive=None,
            belief_mlp_match_target=None,
            belief_linear_match_target=None,
            belief_available_mlp=False,
            belief_available_linear=False,
            outcome_cell_mlp=None,
            outcome_cell_linear=None,
            valid_pair=False,
            invalid_reason=f"trace loading failed: {exc}",
        )

    if (
        not _has_required_step_fields(a_trace)
        or not _has_required_step_fields(b_trace)
        or not _has_required_step_fields(patched_trace)
    ):
        return PairMetrics(
            pair_id=spec.pair_id,
            category=spec.category,
            evaluated_steps=0,
            a_target=0.0,
            a_base=0.0,
            action_label=None,
            disruptive=None,
            belief_mlp_match_target=None,
            belief_linear_match_target=None,
            belief_available_mlp=False,
            belief_available_linear=False,
            outcome_cell_mlp=None,
            outcome_cell_linear=None,
            valid_pair=False,
            invalid_reason="trace missing required step fields",
        )

    patch_metadata_error = _validate_patch_metadata(patched_trace, layer_key)
    if patch_metadata_error is not None:
        return PairMetrics(
            pair_id=spec.pair_id,
            category=spec.category,
            evaluated_steps=0,
            a_target=0.0,
            a_base=0.0,
            action_label=None,
            disruptive=None,
            belief_mlp_match_target=None,
            belief_linear_match_target=None,
            belief_available_mlp=False,
            belief_available_linear=False,
            outcome_cell_mlp=None,
            outcome_cell_linear=None,
            valid_pair=False,
            invalid_reason=patch_metadata_error,
        )

    layout_a = _parse_grid_text_file(spec.grid_a_path)
    layout_b = _parse_grid_text_file(spec.grid_b_path)

    evaluated_steps, a_target, a_base = _compute_action_metrics(
        patched_trace=patched_trace,
        spec=spec,
        layout_a=layout_a,
        layout_b=layout_b,
    )

    action_label: Optional[bool] = None
    disruptive: Optional[bool] = None
    disruptive_reason: Optional[str] = None
    if evaluated_steps > 0:
        action_label = (a_target > a_base) and (a_target >= action_true_threshold)
        disruptive = (a_target < disruptive_threshold) and (a_base < disruptive_threshold)
        if disruptive:
            disruptive_reason = (
                f"A_target={a_target:.4f} and A_base={a_base:.4f} are both below "
                f"disruptive_threshold={disruptive_threshold:.2f}"
            )

    all_probe_dicts: list[dict[str, Any]] = []
    for step in patched_trace.get("steps", []):
        all_probe_dicts.extend(_collect_probe_dicts_from_step(step))

    mlp_goal = _decode_goal_from_probes(all_probe_dicts, layer_key=layer_key, probe_type="mlp")
    linear_goal = _decode_goal_from_probes(all_probe_dicts, layer_key=layer_key, probe_type="linear")

    belief_mlp_match_target = None if mlp_goal is None else (mlp_goal == spec.goal_b)
    belief_linear_match_target = None if linear_goal is None else (linear_goal == spec.goal_b)
    belief_available_mlp = belief_mlp_match_target is not None
    belief_available_linear = belief_linear_match_target is not None

    outcome_cell_mlp = None
    if belief_mlp_match_target is not None and action_label is not None:
        outcome_cell_mlp = (belief_mlp_match_target, action_label)

    outcome_cell_linear = None
    if belief_linear_match_target is not None and action_label is not None:
        outcome_cell_linear = (belief_linear_match_target, action_label)

    return PairMetrics(
        pair_id=spec.pair_id,
        category=spec.category,
        evaluated_steps=evaluated_steps,
        a_target=a_target,
        a_base=a_base,
        action_label=action_label,
        disruptive=disruptive,
        belief_mlp_match_target=belief_mlp_match_target,
        belief_linear_match_target=belief_linear_match_target,
        belief_available_mlp=belief_available_mlp,
        belief_available_linear=belief_available_linear,
        outcome_cell_mlp=outcome_cell_mlp,
        outcome_cell_linear=outcome_cell_linear,
        valid_pair=True,
        invalid_reason=None,
        disruptive_reason=disruptive_reason,
    )


def _safe_ratio(numer: int, denom: int) -> Optional[float]:
    if denom == 0:
        return None
    return numer / denom


def _count_outcomes(
    rows: list[PairMetrics],
    field_name: str,
) -> tuple[int, int, int, int]:
    tt = tf = ft = ff = 0
    for row in rows:
        cell = getattr(row, field_name)
        if cell == (True, True):
            tt += 1
        elif cell == (True, False):
            tf += 1
        elif cell == (False, True):
            ft += 1
        elif cell == (False, False):
            ff += 1
    return tt, tf, ft, ff


def _compute_per_category_summary(pair_metrics: list[PairMetrics]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[PairMetrics]] = defaultdict(list)
    for metric in pair_metrics:
        grouped[metric.category].append(metric)

    summary: dict[str, dict[str, Any]] = {}
    for category, rows in grouped.items():
        valid_rows = [r for r in rows if r.valid_pair]
        disruptive_rows = [r for r in valid_rows if r.disruptive is True]
        action_rows = [r for r in valid_rows if r.action_label is not None and r.disruptive is False]
        action_true = sum(1 for r in action_rows if r.action_label is True)

        mlp_rows = [r for r in action_rows if r.belief_available_mlp]
        linear_rows = [r for r in action_rows if r.belief_available_linear]

        summary[category] = {
            "total_pairs": len(rows),
            "valid_pairs": len(valid_rows),
            "invalid_pairs": len(rows) - len(valid_rows),
            "disruptive_pairs": len(disruptive_rows),
            "disruptive_rate": _safe_ratio(len(disruptive_rows), len(valid_rows)),
            "action_true_count": action_true,
            "action_true_rate": _safe_ratio(action_true, len(action_rows)),
            "table_rows_mlp": len(mlp_rows),
            "table_rows_linear": len(linear_rows),
        }

    return summary


def aggregate_results(
    pair_metrics: list[PairMetrics],
    stopped_early: bool,
    early_stop_reason: Optional[str],
) -> AggregateMetrics:
    valid_pairs = [p for p in pair_metrics if p.valid_pair]
    invalid_pairs = [p for p in pair_metrics if not p.valid_pair]

    disruptive_count = sum(1 for p in valid_pairs if p.disruptive is True)
    action_rows = [
        p for p in valid_pairs if p.action_label is not None and p.disruptive is False
    ]
    action_true_count = sum(1 for p in action_rows if p.action_label is True)

    mlp_rows = [
        p
        for p in valid_pairs
        if p.belief_available_mlp and p.action_label is not None and p.disruptive is False
    ]
    linear_rows = [
        p
        for p in valid_pairs
        if p.belief_available_linear and p.action_label is not None and p.disruptive is False
    ]

    mlp_tt, mlp_tf, mlp_ft, mlp_ff = _count_outcomes(mlp_rows, "outcome_cell_mlp")
    linear_tt, linear_tf, linear_ft, linear_ff = _count_outcomes(
        linear_rows, "outcome_cell_linear"
    )

    probe_comparison_rows = [
        p
        for p in valid_pairs
        if p.belief_available_mlp and p.belief_available_linear and p.disruptive is False
    ]
    probe_agree = sum(
        1
        for p in probe_comparison_rows
        if p.belief_mlp_match_target == p.belief_linear_match_target
    )
    probe_disagree = len(probe_comparison_rows) - probe_agree

    return AggregateMetrics(
        total_pairs_manifest=len(pair_metrics),
        total_pairs_processed=len(pair_metrics),
        total_pairs_valid=len(valid_pairs),
        total_pairs_invalid=len(invalid_pairs),
        total_pairs_disruptive=disruptive_count,
        disruptive_rate=_safe_ratio(disruptive_count, len(valid_pairs)) or 0.0,
        total_pairs_action_evaluable=len(action_rows),
        total_pairs_action_true=action_true_count,
        action_true_rate=_safe_ratio(action_true_count, len(action_rows)),
        table_rows_mlp=len(mlp_rows),
        table_rows_linear=len(linear_rows),
        total_pairs_belief_available=len(mlp_rows),
        mlp_tt_count=mlp_tt,
        mlp_tf_count=mlp_tf,
        mlp_ft_count=mlp_ft,
        mlp_ff_count=mlp_ff,
        mlp_tt_over_tt_tf=_safe_ratio(mlp_tt, mlp_tt + mlp_tf),
        mlp_tt_over_tt_ft=_safe_ratio(mlp_tt, mlp_tt + mlp_ft),
        linear_tt_count=linear_tt,
        linear_tf_count=linear_tf,
        linear_ft_count=linear_ft,
        linear_ff_count=linear_ff,
        linear_tt_over_tt_tf=_safe_ratio(linear_tt, linear_tt + linear_tf),
        linear_tt_over_tt_ft=_safe_ratio(linear_tt, linear_tt + linear_ft),
        probe_both_available=len(probe_comparison_rows),
        probe_agree_count=probe_agree,
        probe_disagree_count=probe_disagree,
        probe_disagreement_rate=_safe_ratio(probe_disagree, len(probe_comparison_rows)),
        per_category=_compute_per_category_summary(pair_metrics),
        stopped_early=stopped_early,
        early_stop_reason=early_stop_reason,
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pair_id"])
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _invalid_reason_counts(pair_metrics: list[PairMetrics]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for metric in pair_metrics:
        if metric.valid_pair:
            continue
        reason = metric.invalid_reason or "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _write_markdown_report(
    path: Path,
    aggregate: AggregateMetrics,
    pair_metrics: list[PairMetrics],
    layer_key: str,
    action_true_threshold: float,
    disruptive_threshold: float,
) -> None:
    invalid_reason_counts = _invalid_reason_counts(pair_metrics)
    disruptive_rows = [p for p in pair_metrics if p.disruptive is True]

    lines = [
        "# Counterfactual Surrogate Patching Report",
        "",
        "## Configuration",
        f"- layer_key: `{layer_key}`",
        "- hook tensor: residual output",
        "- patch token set: last 3 pre-reasoning + last 3 post-reasoning",
        "- method status: this is the saved-trace substitution baseline, not the new live hidden-state patching path",
        "- implementation note: we do not rerun the model after patching layer 15; instead, we take the saved trace from A and replace the selected last-3 PRE and last-3 POST layer-15 entries with those from B",
        "- implementation note: the saved trace from A is the scaffold for the intervened trace, so later layers are not recomputed online after intervention",
        f"- Action=True rule: A_target > A_base and A_target >= {action_true_threshold:.2f}",
        f"- Disruptive rule: A_target < {disruptive_threshold:.2f} and A_base < {disruptive_threshold:.2f}",
        "",
        "## Metric Glossary",
        "- Use this reading rule everywhere: each metric is `(behavior being scored, reference used for scoring)`.",
        "- `A_target` (alias `A_new`): `(patched/intervened trace, true grid B optimal policy)`; this tests whether the intervened trace is optimal on the actual target grid B, not on the decoded map.",
        "- `A_base` (alias `A_orig`): `(patched/intervened trace, true grid A optimal policy)`; this tests whether the same intervened trace is optimal on the actual base grid A, not on the decoded map.",
        "- Base vs target: there is only one intervened trace; `base` and `target` mean scoring that same intervened trace against true grid A vs true grid B, respectively.",
        "- Belief readout: probe-decoded cognitive map / decoded goal from probes; decoded-map information is used for belief readout only, not for `A_target` or `A_base`.",
        "- Paper comparison: `Acc. GT` in Table 2 is `(original unpatched model behavior, ground-truth grid)`; it is not directly equivalent to `A_base`, which is `(patched/intervened behavior, true grid A)`.",
        "- Why these are non-integers: each value is a ratio `aligned_steps / evaluated_steps`, not a raw count.",
        "- `evaluated_steps` excludes steps where action parsing or position mapping is invalid for that pair.",
        "",
        "## Manifest and Processing Summary",
        f"- total_pairs_manifest: {aggregate.total_pairs_manifest}",
        f"- total_pairs_processed: {aggregate.total_pairs_processed}",
        f"- total_pairs_valid: {aggregate.total_pairs_valid}",
        f"- total_pairs_invalid: {aggregate.total_pairs_invalid}",
        f"- stopped_early: {aggregate.stopped_early}",
        f"- early_stop_reason: {aggregate.early_stop_reason or 'NA'}",
        f"- action_true_count: {aggregate.total_pairs_action_true}",
        f"- action_true_rate: {aggregate.action_true_rate if aggregate.action_true_rate is not None else 'NA'}",
        "",
        "## Invalid Summary",
    ]

    if invalid_reason_counts:
        for reason, count in sorted(invalid_reason_counts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Disruptive Summary",
            f"- disruptive_count: {aggregate.total_pairs_disruptive}",
            f"- disruptive_rate: {aggregate.disruptive_rate:.4f}",
            "- disruptive_pairs:",
        ]
    )

    if disruptive_rows:
        for row in disruptive_rows:
            lines.append(f"  - {row.pair_id}: {row.disruptive_reason or 'threshold rule met'}")
    else:
        lines.append("  - None")

    lines.extend(["", "## Action Summary by Category"])
    lines.append("| category | total | valid | invalid | disruptive | action_true_rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for category, row in sorted(aggregate.per_category.items()):
        atr = row["action_true_rate"]
        lines.append(
            f"| {category} | {row['total_pairs']} | {row['valid_pairs']} | {row['invalid_pairs']} | {row['disruptive_pairs']} | {atr if atr is not None else 'NA'} |"
        )

    lines.extend(
        [
            "",
            "## MLP Belief-Action 2x2",
            f"- denominator (table_rows_mlp): {aggregate.table_rows_mlp}",
            "| Belief (MLP) \\ Action | True | False |",
            "|---|---:|---:|",
            f"| True | {aggregate.mlp_tt_count} | {aggregate.mlp_tf_count} |",
            f"| False | {aggregate.mlp_ft_count} | {aggregate.mlp_ff_count} |",
            f"- TT/(TT+TF): {aggregate.mlp_tt_over_tt_tf if aggregate.mlp_tt_over_tt_tf is not None else 'NA'}",
            f"- TT/(TT+FT): {aggregate.mlp_tt_over_tt_ft if aggregate.mlp_tt_over_tt_ft is not None else 'NA'}",
            "",
            "## Linear Belief-Action 2x2",
            f"- denominator (table_rows_linear): {aggregate.table_rows_linear}",
            "| Belief (Linear) \\ Action | True | False |",
            "|---|---:|---:|",
            f"| True | {aggregate.linear_tt_count} | {aggregate.linear_tf_count} |",
            f"| False | {aggregate.linear_ft_count} | {aggregate.linear_ff_count} |",
            f"- TT/(TT+TF): {aggregate.linear_tt_over_tt_tf if aggregate.linear_tt_over_tt_tf is not None else 'NA'}",
            f"- TT/(TT+FT): {aggregate.linear_tt_over_tt_ft if aggregate.linear_tt_over_tt_ft is not None else 'NA'}",
            "",
            "## Probe Agreement/Disagreement",
            f"- probe_both_available: {aggregate.probe_both_available}",
            f"- probe_agree_count: {aggregate.probe_agree_count}",
            f"- probe_disagree_count: {aggregate.probe_disagree_count}",
            f"- probe_disagreement_rate: {aggregate.probe_disagreement_rate if aggregate.probe_disagreement_rate is not None else 'NA'}",
            "",
            "## Pair Results",
            "",
            "| pair_id | category | valid | A_target | A_base | action | disruptive | disruptive_reason | belief_mlp | belief_linear | avail_mlp | avail_linear | invalid_reason |",
            "|---|---|---|---:|---:|---|---|---|---|---|---|---|---|",
        ]
    )

    for p in pair_metrics:
        lines.append(
            f"| {p.pair_id} | {p.category} | {p.valid_pair} | {p.a_target:.4f} | {p.a_base:.4f} | {p.action_label} | {p.disruptive} | {p.disruptive_reason or ''} | {p.belief_mlp_match_target} | {p.belief_linear_match_target} | {p.belief_available_mlp} | {p.belief_available_linear} | {p.invalid_reason or ''} |"
        )

    path.write_text("\n".join(lines) + "\n")


def _pair_metrics_to_dict(metric: PairMetrics) -> dict[str, Any]:
    row = asdict(metric)
    if row["outcome_cell_mlp"] is not None:
        row["outcome_cell_mlp"] = list(row["outcome_cell_mlp"])
    if row["outcome_cell_linear"] is not None:
        row["outcome_cell_linear"] = list(row["outcome_cell_linear"])

    # Backward-compatible aliases
    row["a_new"] = row["a_target"]
    row["a_orig"] = row["a_base"]
    row["belief_mlp_new_match"] = row["belief_mlp_match_target"]
    row["belief_linear_new_match"] = row["belief_linear_match_target"]
    row["belief_available"] = row["belief_available_mlp"]
    row["belief_label"] = row["belief_mlp_match_target"]
    row["outcome_cell"] = row["outcome_cell_mlp"]

    return row


def counterfactual_activation_patching(
    manifest_path: str,
    output_dir: str = "data/cf/eval_results",
    layer_key: str = LAYER_KEY_DEFAULT,
    expected_k: int = EXPECTED_K,
    action_true_threshold: float = ACTION_TRUE_THRESHOLD,
    disruptive_threshold: float = DISRUPTIVE_THRESHOLD,
    enable_early_stop: bool = True,
) -> None:
    """Run counterfactual activation-patching evaluation from artifacts.

    Manifest columns/keys required (JSON or CSV):
    - pair_id
    - category (optional, defaults to goal_move)
    - grid_a_path
    - grid_b_path
    - goal_a/goal_b (legacy goal_orig/goal_new accepted)
    - a_trace_path
    - b_trace_path
    - patched_trace_path
    """
    manifest = Path(manifest_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _read_manifest(manifest)
    if len(records) != expected_k:
        recommendation = (
            "Run with the manifest row count, e.g. "
            f"`reveng-cli counterfactual_activation_patching --manifest-path {manifest} "
            f"--output-dir {out_dir} --expected-k {len(records)}`"
        )
        raise ValueError(
            f"Expected exactly {expected_k} pairs in manifest, found {len(records)}. "
            f"{recommendation}"
        )

    pair_metrics: list[PairMetrics] = []
    evaluated_for_early_stop: list[PairMetrics] = []
    stopped_early = False
    early_stop_reason: Optional[str] = None

    for record in records:
        metric = evaluate_pair(
            record,
            layer_key=layer_key,
            action_true_threshold=action_true_threshold,
            disruptive_threshold=disruptive_threshold,
        )
        pair_metrics.append(metric)

        if (
            metric.valid_pair
            and metric.action_label is not None
            and metric.disruptive is not None
        ):
            evaluated_for_early_stop.append(metric)

        if enable_early_stop and len(evaluated_for_early_stop) >= 3 and not stopped_early:
            first_three = evaluated_for_early_stop[:3]
            should_stop = all(
                (m.action_label is False)
                and (m.a_base > m.a_target)
                and (m.disruptive is False)
                for m in first_three
            )
            if should_stop:
                stopped_early = True
                early_stop_reason = (
                    "first 3 evaluated pairs are Action=False with A_base > A_target and non-disruptive"
                )
                break

    aggregate = aggregate_results(
        pair_metrics=pair_metrics,
        stopped_early=stopped_early,
        early_stop_reason=early_stop_reason,
    )

    per_pair_rows = [_pair_metrics_to_dict(m) for m in pair_metrics]
    _write_jsonl(out_dir / "per_pair_results.jsonl", per_pair_rows)
    _write_csv(out_dir / "per_pair_results.csv", per_pair_rows)

    aggregate_dict = asdict(aggregate)
    aggregate_dict["a_target_label"] = "A_target"
    aggregate_dict["a_base_label"] = "A_base"
    (out_dir / "aggregate_summary.json").write_text(json.dumps(aggregate_dict, indent=2))

    _write_markdown_report(
        path=out_dir / "report.md",
        aggregate=aggregate,
        pair_metrics=pair_metrics,
        layer_key=layer_key,
        action_true_threshold=action_true_threshold,
        disruptive_threshold=disruptive_threshold,
    )

    print(f"Wrote results to {out_dir}")


__all__ = [
    "AggregateMetrics",
    "GridPairSpec",
    "PairMetrics",
    "RunArtifacts",
    "aggregate_results",
    "counterfactual_activation_patching",
    "evaluate_pair",
    "_read_manifest",
    "_validate_pair_spec_against_files",
]
