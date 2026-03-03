"""Minimal counterfactual activation-patching experiment runner.

This module evaluates whether patched activations (produced externally) cause
counterfactual navigation behavior toward a moved goal in grid-world traces.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from reveng.analysis.analysis_utils import ACTION_NAME_TO_ID, compute_optimal_actions_from_text_grid

LAYER_KEY_DEFAULT = "model.layers.15.output"
EXPECTED_K = 10
ACTION_TRUE_THRESHOLD = 0.70
DISRUPTIVE_THRESHOLD = 0.35


@dataclass
class GridPairSpec:
    pair_id: str
    grid_a_path: Path
    grid_b_path: Path
    goal_orig: tuple[int, int]
    goal_new: tuple[int, int]


@dataclass
class RunArtifacts:
    a_trace_path: Path
    b_trace_path: Path
    patched_trace_path: Path


@dataclass
class PairMetrics:
    pair_id: str
    evaluated_steps: int
    a_new: float
    a_orig: float
    belief_mlp_new_match: Optional[bool]
    belief_linear_new_match: Optional[bool]
    action_label: Optional[bool]
    belief_label: Optional[bool]
    disruptive: Optional[bool]
    outcome_cell: Optional[tuple[bool, bool]]
    valid_pair: bool
    belief_available: bool
    invalid_reason: Optional[str]


@dataclass
class AggregateMetrics:
    total_pairs_manifest: int
    total_pairs_processed: int
    total_pairs_valid: int
    total_pairs_invalid: int
    total_pairs_belief_available: int
    total_pairs_disruptive: int
    disruptive_rate: float
    tt_count: int
    tf_count: int
    ft_count: int
    ff_count: int
    tt_over_tt_tf: Optional[float]
    tt_over_tt_ft: Optional[float]
    stopped_early: bool
    early_stop_reason: Optional[str]


@dataclass
class PairRecord:
    spec: GridPairSpec
    artifacts: RunArtifacts


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
        if len(parts) == 2 and all(part for part in parts):
            return int(parts[0]), int(parts[1])
    raise ValueError(f"Invalid coordinate format: {value}")


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
        spec = GridPairSpec(
            pair_id=pair_id,
            grid_a_path=Path(str(row["grid_a_path"])),
            grid_b_path=Path(str(row["grid_b_path"])),
            goal_orig=_parse_coord(row["goal_orig"]),
            goal_new=_parse_coord(row["goal_new"]),
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
        # Skip header like: "0 1 2 3 4"
        if all(p.lstrip("-").isdigit() for p in parts):
            saw_header = True
            continue
        # Data rows must start with a row index in this artifact format.
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


def _extract_agent_goal(layout: list[list[str]]) -> tuple[tuple[int, int], tuple[int, int]]:
    agent: Optional[tuple[int, int]] = None
    goal: Optional[tuple[int, int]] = None
    for y, row in enumerate(layout):
        for x, cell in enumerate(row):
            if cell == "A":
                agent = (x, y)
            elif cell == "G":
                goal = (x, y)
    if agent is None or goal is None:
        raise ValueError("Grid must contain exactly one agent 'A' and one goal 'G'")
    return agent, goal


def _normalize_layout_for_topology(layout: list[list[str]]) -> list[list[str]]:
    normalized = []
    for row in layout:
        out_row = []
        for cell in row:
            if cell in {"A", "G"}:
                out_row.append("_")
            else:
                out_row.append(cell)
        normalized.append(out_row)
    return normalized


def _validate_goal_move_only(spec: GridPairSpec) -> Optional[str]:
    if spec.goal_orig == spec.goal_new:
        return "goal_orig and goal_new must differ"

    try:
        grid_a_layout = _parse_grid_state_to_layout(spec.grid_a_path.read_text().splitlines())
        grid_b_layout = _parse_grid_state_to_layout(spec.grid_b_path.read_text().splitlines())
    except Exception as exc:
        return f"failed to parse grid files: {exc}"

    try:
        agent_a, goal_a = _extract_agent_goal(grid_a_layout)
        agent_b, goal_b = _extract_agent_goal(grid_b_layout)
    except Exception as exc:
        return f"invalid grid files: {exc}"

    if goal_a != spec.goal_orig:
        return f"grid_a goal {goal_a} != manifest goal_orig {spec.goal_orig}"
    if goal_b != spec.goal_new:
        return f"grid_b goal {goal_b} != manifest goal_new {spec.goal_new}"

    if agent_a != agent_b:
        return "agent position changed between grid A and grid B"

    if _normalize_layout_for_topology(grid_a_layout) != _normalize_layout_for_topology(grid_b_layout):
        return "non-goal topology changed between grid A and grid B"

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
    goal_orig: tuple[int, int],
    goal_new: tuple[int, int],
) -> tuple[int, float, float]:
    steps = patched_trace.get("steps", [])
    aligned_new = 0
    aligned_orig = 0
    evaluated_steps = 0

    for step in steps:
        grid_state = step.get("grid_state")
        action_raw = step.get("agent_action")
        if not isinstance(grid_state, list):
            continue
        action_id = _action_to_id(action_raw)
        if action_id is None:
            continue

        try:
            layout = _parse_grid_state_to_layout(grid_state)
            optimal_new, _ = compute_optimal_actions_from_text_grid(layout, goal_new)
            optimal_orig, _ = compute_optimal_actions_from_text_grid(layout, goal_orig)
            agent_pos, _ = _extract_agent_goal(layout)
        except Exception:
            continue

        if agent_pos not in optimal_new or agent_pos not in optimal_orig:
            continue

        evaluated_steps += 1
        if action_id in optimal_new[agent_pos]:
            aligned_new += 1
        if action_id in optimal_orig[agent_pos]:
            aligned_orig += 1

    if evaluated_steps == 0:
        return 0, 0.0, 0.0

    a_new = aligned_new / evaluated_steps
    a_orig = aligned_orig / evaluated_steps
    return evaluated_steps, a_new, a_orig


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
            evaluated_steps=0,
            a_new=0.0,
            a_orig=0.0,
            belief_mlp_new_match=None,
            belief_linear_new_match=None,
            action_label=None,
            belief_label=None,
            disruptive=None,
            outcome_cell=None,
            valid_pair=False,
            belief_available=False,
            invalid_reason="missing grid_a_path or grid_b_path",
        )

    goal_move_only_error = _validate_goal_move_only(spec)
    if goal_move_only_error is not None:
        return PairMetrics(
            pair_id=spec.pair_id,
            evaluated_steps=0,
            a_new=0.0,
            a_orig=0.0,
            belief_mlp_new_match=None,
            belief_linear_new_match=None,
            action_label=None,
            belief_label=None,
            disruptive=None,
            outcome_cell=None,
            valid_pair=False,
            belief_available=False,
            invalid_reason=goal_move_only_error,
        )

    try:
        a_trace = _load_trace(_resolve_trace_path(artifacts.a_trace_path))
        b_trace = _load_trace(_resolve_trace_path(artifacts.b_trace_path))
        patched_trace = _load_trace(_resolve_trace_path(artifacts.patched_trace_path))
    except Exception as exc:
        return PairMetrics(
            pair_id=spec.pair_id,
            evaluated_steps=0,
            a_new=0.0,
            a_orig=0.0,
            belief_mlp_new_match=None,
            belief_linear_new_match=None,
            action_label=None,
            belief_label=None,
            disruptive=None,
            outcome_cell=None,
            valid_pair=False,
            belief_available=False,
            invalid_reason=f"trace loading failed: {exc}",
        )

    if not _has_required_step_fields(a_trace) or not _has_required_step_fields(b_trace) or not _has_required_step_fields(patched_trace):
        return PairMetrics(
            pair_id=spec.pair_id,
            evaluated_steps=0,
            a_new=0.0,
            a_orig=0.0,
            belief_mlp_new_match=None,
            belief_linear_new_match=None,
            action_label=None,
            belief_label=None,
            disruptive=None,
            outcome_cell=None,
            valid_pair=False,
            belief_available=False,
            invalid_reason="trace missing required step fields",
        )

    patch_metadata_error = _validate_patch_metadata(patched_trace, layer_key)
    if patch_metadata_error is not None:
        return PairMetrics(
            pair_id=spec.pair_id,
            evaluated_steps=0,
            a_new=0.0,
            a_orig=0.0,
            belief_mlp_new_match=None,
            belief_linear_new_match=None,
            action_label=None,
            belief_label=None,
            disruptive=None,
            outcome_cell=None,
            valid_pair=False,
            belief_available=False,
            invalid_reason=patch_metadata_error,
        )

    evaluated_steps, a_new, a_orig = _compute_action_metrics(
        patched_trace=patched_trace,
        goal_orig=spec.goal_orig,
        goal_new=spec.goal_new,
    )

    action_label = (a_new > a_orig) and (a_new >= action_true_threshold)
    disruptive = (a_new < disruptive_threshold) and (a_orig < disruptive_threshold)

    all_probe_dicts: list[dict[str, Any]] = []
    for step in patched_trace.get("steps", []):
        all_probe_dicts.extend(_collect_probe_dicts_from_step(step))

    mlp_goal = _decode_goal_from_probes(all_probe_dicts, layer_key=layer_key, probe_type="mlp")
    linear_goal = _decode_goal_from_probes(all_probe_dicts, layer_key=layer_key, probe_type="linear")

    belief_mlp_new_match = None if mlp_goal is None else (mlp_goal == spec.goal_new)
    belief_linear_new_match = None if linear_goal is None else (linear_goal == spec.goal_new)

    belief_available = belief_mlp_new_match is not None
    belief_label = belief_mlp_new_match if belief_available else None

    outcome_cell = None
    if belief_label is not None:
        outcome_cell = (belief_label, action_label)

    return PairMetrics(
        pair_id=spec.pair_id,
        evaluated_steps=evaluated_steps,
        a_new=a_new,
        a_orig=a_orig,
        belief_mlp_new_match=belief_mlp_new_match,
        belief_linear_new_match=belief_linear_new_match,
        action_label=action_label,
        belief_label=belief_label,
        disruptive=disruptive,
        outcome_cell=outcome_cell,
        valid_pair=True,
        belief_available=belief_available,
        invalid_reason=None,
    )


def _safe_ratio(numer: int, denom: int) -> Optional[float]:
    if denom == 0:
        return None
    return numer / denom


def aggregate_results(
    pair_metrics: list[PairMetrics],
    stopped_early: bool,
    early_stop_reason: Optional[str],
) -> AggregateMetrics:
    valid_pairs = [p for p in pair_metrics if p.valid_pair]
    invalid_pairs = [p for p in pair_metrics if not p.valid_pair]

    # Disruptive runs are reported separately and excluded from the main 2x2 belief/action table.
    table_pairs = [
        p
        for p in valid_pairs
        if p.belief_available and p.action_label is not None and p.disruptive is False
    ]

    tt = sum(1 for p in table_pairs if p.outcome_cell == (True, True))
    tf = sum(1 for p in table_pairs if p.outcome_cell == (True, False))
    ft = sum(1 for p in table_pairs if p.outcome_cell == (False, True))
    ff = sum(1 for p in table_pairs if p.outcome_cell == (False, False))

    disruptive_count = sum(1 for p in valid_pairs if p.disruptive is True)

    return AggregateMetrics(
        total_pairs_manifest=len(pair_metrics),
        total_pairs_processed=len(pair_metrics),
        total_pairs_valid=len(valid_pairs),
        total_pairs_invalid=len(invalid_pairs),
        total_pairs_belief_available=len(table_pairs),
        total_pairs_disruptive=disruptive_count,
        disruptive_rate=_safe_ratio(disruptive_count, len(valid_pairs)) or 0.0,
        tt_count=tt,
        tf_count=tf,
        ft_count=ft,
        ff_count=ff,
        tt_over_tt_tf=_safe_ratio(tt, tt + tf),
        tt_over_tt_ft=_safe_ratio(tt, tt + ft),
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


def _write_markdown_report(
    path: Path,
    aggregate: AggregateMetrics,
    pair_metrics: list[PairMetrics],
    layer_key: str,
    action_true_threshold: float,
    disruptive_threshold: float,
) -> None:
    lines = [
        "# Counterfactual Activation Patching Report",
        "",
        "## Configuration",
        f"- layer_key: `{layer_key}`",
        "- hook tensor: residual output",
        "- patch token set: last 3 pre-reasoning + last 3 post-reasoning",
        f"- Action=True rule: A_new > A_orig and A_new >= {action_true_threshold:.2f}",
        f"- Disruptive rule: A_new < {disruptive_threshold:.2f} and A_orig < {disruptive_threshold:.2f}",
        "- Belief label: MLP primary, linear secondary",
        "",
        "## 2x2 Outcome Table",
        "",
        "| Belief \\ Action | True | False |",
        "|---|---:|---:|",
        f"| True | {aggregate.tt_count} | {aggregate.tf_count} |",
        f"| False | {aggregate.ft_count} | {aggregate.ff_count} |",
        "",
        "## Aggregate Metrics",
        f"- total_pairs_manifest: {aggregate.total_pairs_manifest}",
        f"- total_pairs_valid: {aggregate.total_pairs_valid}",
        f"- total_pairs_invalid: {aggregate.total_pairs_invalid}",
        f"- total_pairs_belief_available: {aggregate.total_pairs_belief_available}",
        f"- disruptive_count: {aggregate.total_pairs_disruptive}",
        f"- disruptive_rate: {aggregate.disruptive_rate:.4f}",
        f"- TT/(TT+TF): {aggregate.tt_over_tt_tf if aggregate.tt_over_tt_tf is not None else 'NA'}",
        f"- TT/(TT+FT): {aggregate.tt_over_tt_ft if aggregate.tt_over_tt_ft is not None else 'NA'}",
        f"- stopped_early: {aggregate.stopped_early}",
        f"- early_stop_reason: {aggregate.early_stop_reason or 'NA'}",
        "",
        "## Pair Results",
        "",
        "| pair_id | valid | belief_available | A_new | A_orig | action | belief | disruptive | invalid_reason |",
        "|---|---|---|---:|---:|---|---|---|---|",
    ]

    for p in pair_metrics:
        lines.append(
            f"| {p.pair_id} | {p.valid_pair} | {p.belief_available} | {p.a_new:.4f} | {p.a_orig:.4f} | {p.action_label} | {p.belief_label} | {p.disruptive} | {p.invalid_reason or ''} |"
        )

    lines.append("")
    lines.append("## Disruptive Cases")
    disruptive_ids = [p.pair_id for p in pair_metrics if p.disruptive is True]
    if disruptive_ids:
        for pair_id in disruptive_ids:
            lines.append(f"- {pair_id}")
    else:
        lines.append("- None")

    path.write_text("\n".join(lines) + "\n")


def _pair_metrics_to_dict(metric: PairMetrics) -> dict[str, Any]:
    row = asdict(metric)
    if row["outcome_cell"] is not None:
        row["outcome_cell"] = list(row["outcome_cell"])
    return row


def counterfactual_activation_patching(
    manifest_path: str,
    output_dir: str = "counterfactual_activation_patching_results",
    layer_key: str = LAYER_KEY_DEFAULT,
    expected_k: int = EXPECTED_K,
    action_true_threshold: float = ACTION_TRUE_THRESHOLD,
    disruptive_threshold: float = DISRUPTIVE_THRESHOLD,
) -> None:
    """Run minimal counterfactual activation-patching evaluation from artifacts.

    Manifest columns/keys required (JSON or CSV):
    - pair_id
    - grid_a_path
    - grid_b_path
    - goal_orig
    - goal_new
    - a_trace_path
    - b_trace_path
    - patched_trace_path
    """
    manifest = Path(manifest_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _read_manifest(manifest)
    if len(records) != expected_k:
        raise ValueError(
            f"Expected exactly {expected_k} pairs in manifest, found {len(records)}"
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

        if metric.valid_pair and metric.action_label is not None and metric.disruptive is not None:
            evaluated_for_early_stop.append(metric)

        if len(evaluated_for_early_stop) >= 3 and not stopped_early:
            first_three = evaluated_for_early_stop[:3]
            should_stop = all(
                (m.action_label is False)
                and (m.a_orig > m.a_new)
                and (m.disruptive is False)
                for m in first_three
            )
            if should_stop:
                stopped_early = True
                early_stop_reason = (
                    "first 3 evaluated pairs are Action=False with A_orig > A_new and non-disruptive"
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
    "counterfactual_activation_patching",
    "evaluate_pair",
    "aggregate_results",
]
