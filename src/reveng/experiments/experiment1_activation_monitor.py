"""Experiment 1 activation monitor with chunk/activation compatibility checks."""

from __future__ import annotations

import csv
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from reveng.experiments.gradual_cot_blackbox_alignment import (
    ANALYSIS_START,
    FINAL_START,
    _extract_analysis_and_final,
)
from reveng.experiments.step_reasoning_drift import (
    compute_reasoning_geometry_metrics,
    run_step_reasoning_drift_experiment,
)

DEFAULT_OUTPUT_ROOT = "outputs/experiment1_activation_monitor"
DEFAULT_CHUNKED_TRAJECTORIES = "data/behavioral_probes/doorkey_chunking_validation/chunked_trajectories.csv"
DEFAULT_SENTENCES = "data/behavioral_probes/doorkey_chunking_validation/sentences.csv"
DEFAULT_DRIFT_RUN_DIR = "data/behavioral_probes/step_reasoning_drift_balanced_v1_corrected_boundary_windows"
DEFAULT_CANDIDATE_ROWS_PATH = (
    "data/behavioral_probes/trajectory_instances_recomputed_optimal/trajectory_selection_candidates.csv"
)
DEFAULT_TRAJECTORY_DIR = "data/hf/trajectories_key_door_100/trajectories_key_door"
DEFAULT_LAYERS = (8, 15, 23)
DEFAULT_WEISHENG_ACTIVATION_ARCHIVE = "data/together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_0.zip"
DEFAULT_GPTOSS20B_TOKENIZER = (
    "/root/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/"
    "6cee5e81ee83917806bbde320786a8fb61efebee"
)
EVENT_GEOMETRY_METRICS = (
    "aligned_change",
    "update_norm",
    "adjacent_step_cosine",
    "optimality_anchor_cosine",
)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.read_text().strip():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _as_int(value: Any) -> int | None:
    try:
        if value in {None, ""}:
            return None
        return int(float(str(value)))
    except Exception:
        return None


def _as_float(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        val = float(str(value))
        if math.isnan(val):
            return None
        return val
    except Exception:
        return None


def _trace_id(trajectory_id: str, step_index: Any) -> str:
    step = _as_int(step_index)
    return f"{trajectory_id}_step_{step or 0:03d}"


def _validate_output_root(output_root: Path) -> None:
    parts = output_root.parts
    if "data" in parts and "behavioral_probes" in parts:
        raise ValueError(
            "Experiment 1 outputs must not be written under data/behavioral_probes. "
            f"Received: {output_root}"
        )
    if "outputs" not in parts:
        raise ValueError(
            "Experiment 1 outputs must be written under an outputs/ directory. "
            f"Received: {output_root}"
        )


def _load_chunk_table(path: Path) -> tuple[dict[tuple[str, int], dict[str, str]], dict[str, int], list[str]]:
    rows = _read_csv(path)
    lookup: dict[tuple[str, int], dict[str, str]] = {}
    counts: Counter[str] = Counter()
    errors: list[str] = []
    by_trace: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        trace = row.get("trace_id", "")
        analysis_id = _as_int(row.get("analysis_id"))
        start = _as_int(row.get("char_start"))
        end = _as_int(row.get("char_end"))
        if not trace or analysis_id is None or start is None or end is None:
            errors.append(f"invalid chunk row: {row}")
            continue
        lookup[(trace, analysis_id)] = row
        counts[trace] += 1
        by_trace[trace].append(row)
    for trace, trace_rows in by_trace.items():
        ordered = sorted(trace_rows, key=lambda r: int(r["analysis_id"]))
        previous_end = -1
        for row in ordered:
            start = int(row["char_start"])
            end = int(row["char_end"])
            if end <= start:
                errors.append(f"empty or reversed chunk span for {trace} analysis_id={row['analysis_id']}")
            if start < previous_end:
                errors.append(f"overlapping chunk spans for {trace} analysis_id={row['analysis_id']}")
            previous_end = end
    return lookup, dict(counts), errors


def _infer_activation_kind(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "empty"
    fieldnames = set(rows[0])
    if "reasoning_split" in fieldnames and "reasoning_step_idx" not in fieldnames:
        return "pre_post_only"
    if "reasoning_step_idx" in fieldnames and (
        "step_last_token_activation_path" in fieldnames
        or "step_mean_activation_path" in fieldnames
        or "step_boundary_window_activation_path" in fieldnames
    ):
        return "step_level"
    if "reasoning_step_idx" in fieldnames and "activation_path" in fieldnames:
        return "step_level_flat"
    return "unknown"


def _activation_paths(row: dict[str, str]) -> dict[str, str]:
    paths: dict[str, str] = {}
    if row.get("step_last_token_activation_path"):
        paths["last_token"] = row["step_last_token_activation_path"]
    if row.get("step_mean_activation_path") and row.get("step_mean_activation_path") != row.get("activation_path"):
        paths["mean_pool"] = row["step_mean_activation_path"]
    if row.get("step_boundary_window_activation_path"):
        paths["boundary_window"] = row["step_boundary_window_activation_path"]
    if row.get("activation_path"):
        paths[row.get("representation_kind") or "activation"] = row["activation_path"]
    return paths


def _tensor_shape(path: str) -> tuple[int, ...] | None:
    if torch is None:
        return None
    tensor = torch.load(path, map_location="cpu")
    shape = getattr(tensor, "shape", None)
    return tuple(int(dim) for dim in shape) if shape is not None else None


def verify_activation_compatibility(
    *,
    activation_rows_path: str,
    chunked_trajectories_path: str = DEFAULT_CHUNKED_TRAJECTORIES,
    output_dir: str,
    required_layers: Sequence[int] = DEFAULT_LAYERS,
    check_tensor_shapes: bool = True,
) -> dict[str, Any]:
    """Verify whether candidate activations can support chunk-level Experiment 1 geometry."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    activation_path = Path(activation_rows_path)
    chunk_lookup, chunk_counts, chunk_errors = _load_chunk_table(Path(chunked_trajectories_path))
    rows = _read_csv(activation_path)
    kind = _infer_activation_kind(rows)
    compatibility_rows: list[dict[str, Any]] = []

    if kind in {"empty", "pre_post_only", "unknown"}:
        reason = {
            "empty": "activation index is empty or missing",
            "pre_post_only": "activation index contains only pre/post reasoning snapshots",
            "unknown": "activation index schema is not recognized as step-level activations",
        }[kind]
        for row in rows:
            compatibility_rows.append({**row, "usable": False, "reason": reason})
        _write_csv(out / "activation_compatibility_rows.csv", compatibility_rows)
        _write_csv(out / "usable_activation_rows.csv", [])
        manifest = {
            "status": "incompatible",
            "activation_rows_path": str(activation_path),
            "activation_kind": kind,
            "reason": reason,
            "n_activation_rows": len(rows),
            "n_usable_rows": 0,
            "chunk_errors": chunk_errors,
        }
        _write_activation_report(out / "activation_compatibility_report.md", manifest, [])
        return manifest

    required = {int(layer) for layer in required_layers}
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        trajectory_id = row.get("trajectory_id", "")
        step_index = _as_int(row.get("step_index"))
        layer = _as_int(row.get("layer"))
        if not trajectory_id or step_index is None or layer is None:
            compatibility_rows.append({**row, "usable": False, "reason": "missing trajectory_id, step_index, or layer"})
            continue
        grouped[(row.get("example_id", ""), trajectory_id, step_index)].append(row)

    tensor_shapes_by_rep: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for (example_id, trajectory_id, step_index), example_rows in grouped.items():
        trace = _trace_id(trajectory_id, step_index)
        has_trace = trace in chunk_counts
        layers_present = {_as_int(row.get("layer")) for row in example_rows}
        missing_layers = sorted(required - {layer for layer in layers_present if layer is not None})
        by_layer_step: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in example_rows:
            step_idx = _as_int(row.get("reasoning_step_idx"))
            if step_idx is not None:
                by_layer_step[int(row.get("layer", 0) or 0)].append(row)
        token_errors_by_layer: dict[int, int] = {}
        for layer, layer_rows in by_layer_step.items():
            ordered = sorted(layer_rows, key=lambda r: _as_int(r.get("reasoning_step_idx")) or -1)
            previous_end = -1
            errors = 0
            for row in ordered:
                start = _as_int(row.get("step_start_token"))
                end = _as_int(row.get("step_end_token"))
                if start is None or end is None or end < start or start < previous_end:
                    errors += 1
                if end is not None:
                    previous_end = end
            token_errors_by_layer[layer] = errors

        for row in example_rows:
            reasons: list[str] = []
            step_idx = _as_int(row.get("reasoning_step_idx"))
            layer = _as_int(row.get("layer"))
            if not has_trace:
                reasons.append("trajectory_id and step_index not found in canonical chunk table")
            if layer not in required:
                reasons.append("layer not requested")
            if missing_layers:
                reasons.append(f"example missing required layers {missing_layers}")
            if token_errors_by_layer.get(layer or -1, 0):
                reasons.append("token spans are not monotonic non-overlapping for this layer")
            paths = _activation_paths(row)
            if not paths:
                reasons.append("no accepted activation path columns")
            missing_paths = [path for path in paths.values() if not Path(path).exists()]
            if missing_paths:
                reasons.append("one or more activation tensor paths are missing")
            if step_idx is None:
                reasons.append("missing reasoning_step_idx")
            elif step_idx == 0:
                pass
            else:
                chunk = chunk_lookup.get((trace, step_idx - 1))
                start = _as_int(row.get("step_start_char_in_analysis"))
                end = _as_int(row.get("step_end_char_in_analysis"))
                if chunk is None:
                    reasons.append("reasoning_step_idx has no matching canonical chunk")
                elif start != int(chunk["char_start"]) or end != int(chunk["char_end"]):
                    reasons.append("activation character span does not match canonical chunk span")
            if check_tensor_shapes and torch is not None and not missing_paths:
                for rep, path in paths.items():
                    try:
                        shape = _tensor_shape(path)
                    except Exception as exc:
                        reasons.append(f"could not load tensor for {rep}: {type(exc).__name__}")
                        continue
                    if shape is not None:
                        tensor_shapes_by_rep[rep].add(shape)
            compatibility_rows.append({**row, "usable": not reasons, "reason": "; ".join(reasons)})

    usable_rows = [row for row in compatibility_rows if row["usable"] is True]
    shape_errors = [rep for rep, shapes in tensor_shapes_by_rep.items() if len(shapes) > 1]
    if shape_errors:
        for row in usable_rows:
            row["usable"] = False
            row["reason"] = (row.get("reason") or "") + f"; inconsistent tensor shapes for {shape_errors}"
        usable_rows = []

    _write_csv(out / "activation_compatibility_rows.csv", compatibility_rows)
    _write_csv(out / "usable_activation_rows.csv", usable_rows)
    status = (
        "compatible"
        if usable_rows
        and len(usable_rows) == len(compatibility_rows)
        and not shape_errors
        and not chunk_errors
        else "incompatible"
    )
    manifest = {
        "status": status,
        "activation_rows_path": str(activation_path),
        "activation_kind": kind,
        "n_activation_rows": len(rows),
        "n_usable_rows": len(usable_rows),
        "n_examples_with_activation_rows": len(grouped),
        "required_layers": sorted(required),
        "chunked_trajectories_path": str(chunked_trajectories_path),
        "n_chunk_table_rows": len(chunk_lookup),
        "chunk_errors": chunk_errors,
        "tensor_shape_errors": shape_errors,
        "top_incompatibility_reasons": Counter(
            reason_part.strip()
            for row in compatibility_rows
            if row.get("reason")
            for reason_part in str(row["reason"]).split(";")
            if reason_part.strip()
        ).most_common(12),
    }
    _write_activation_report(out / "activation_compatibility_report.md", manifest, compatibility_rows)
    return manifest


def _write_activation_report(path: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Activation Compatibility Report",
        "",
        f"- Status: `{manifest.get('status')}`",
        f"- Activation rows: {manifest.get('n_activation_rows', 0)}",
        f"- Usable rows: {manifest.get('n_usable_rows', 0)}",
        f"- Activation kind: `{manifest.get('activation_kind', '')}`",
        f"- Chunk table rows: {manifest.get('n_chunk_table_rows', 0)}",
        "",
    ]
    if manifest.get("reason"):
        lines.extend([f"- Reason: {manifest['reason']}", ""])
    if manifest.get("status") != "compatible":
        lines.extend([
            "## Interpretation",
            "",
            "These activations should not be used for primary Experiment 1 step-level geometry unless they are regenerated or remapped successfully.",
            "",
        ])
    reasons = manifest.get("top_incompatibility_reasons") or []
    if reasons:
        lines.extend(["## Top Incompatibility Reasons", "", "| Reason | Count |", "|---|---:|"])
        lines.extend(f"| {reason} | {count} |" for reason, count in reasons)
        lines.append("")
    if manifest.get("chunk_errors"):
        lines.extend(["## Chunk Table Errors", ""])
        lines.extend(f"- {error}" for error in manifest["chunk_errors"][:20])
        lines.append("")
    path.write_text("\n".join(lines))


def _classify_events(prefix_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_example: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in prefix_rows:
        by_example[row["example_id"]].append(row)
    events: list[dict[str, Any]] = []
    for example_id, rows in by_example.items():
        ordered = sorted(rows, key=lambda r: _as_int(r.get("reasoning_step_idx")) or -1)
        committed_steps = [
            _as_int(row.get("reasoning_step_idx"))
            for row in ordered
            if _as_bool(row.get("action_matches_final")) is True
            and row.get("action_label", "") in {"UP", "DOWN", "LEFT", "RIGHT"}
        ]
        commitment_step = None
        for step in committed_steps:
            if step is None:
                continue
            later = [
                row
                for row in ordered
                if (_as_int(row.get("reasoning_step_idx")) or -1) >= step
            ]
            if later and all(_as_bool(row.get("action_matches_final")) is True for row in later):
                commitment_step = step
                break
        for idx in range(len(ordered) - 1):
            current = ordered[idx]
            following = ordered[idx + 1]
            current_step = _as_int(current.get("reasoning_step_idx"))
            next_step = _as_int(following.get("reasoning_step_idx"))
            if current_step is None or next_step is None:
                continue
            current_action = current.get("action_label", "")
            next_action = following.get("action_label", "")
            current_opt = _as_bool(current.get("action_is_optimal"))
            next_opt = _as_bool(following.get("action_is_optimal"))
            current_entropy = _as_float(current.get("action_logprob_entropy_bits"))
            if current_entropy is None:
                current_entropy = _as_float(current.get("action_mc_entropy"))
            next_entropy = _as_float(following.get("action_logprob_entropy_bits"))
            if next_entropy is None:
                next_entropy = _as_float(following.get("action_mc_entropy"))
            base = {
                "event_id": f"{example_id}:{current_step}->{next_step}",
                "example_id": example_id,
                "trajectory_id": current.get("trajectory_id", ""),
                "step_index": current.get("step_index", ""),
                "failure_category": current.get("failure_category", ""),
                "reasoning_step_idx": current_step,
                "next_reasoning_step_idx": next_step,
                "reasoning_progress": current.get("reasoning_progress", ""),
                "reasoning_character_progress": following.get(
                    "reasoning_character_progress", ""
                ),
                "analysis_unit": following.get("analysis_unit", "packed_chunk"),
                "current_action": current_action,
                "next_action": next_action,
                "current_action_is_optimal": current_opt,
                "next_action_is_optimal": next_opt,
                "current_action_mc_entropy": current.get("action_mc_entropy", ""),
                "next_action_mc_entropy": following.get("action_mc_entropy", ""),
                "action_entropy_method": (
                    "T=0.7 four-action token logprobs"
                    if current.get("action_logprob_entropy_bits") not in {None, ""}
                    else "repeated action samples"
                ),
                "current_action_entropy_bits": (
                    current_entropy if current_entropy is not None else ""
                ),
                "next_action_entropy_bits": (
                    next_entropy if next_entropy is not None else ""
                ),
                "delta_action_entropy_bits": (
                    next_entropy - current_entropy
                    if current_entropy is not None and next_entropy is not None
                    else ""
                ),
                "delta_action_mc_entropy": (
                    next_entropy - current_entropy
                    if current_entropy is not None and next_entropy is not None
                    else ""
                ),
            }
            if current_action and next_action and current_action != next_action:
                events.append({**base, "event_type": "action_change"})
            if next_step == commitment_step and current_step != next_step:
                events.append({**base, "event_type": "commitment_onset"})
            if current_opt is True and next_opt is False:
                later = [_as_bool(row.get("action_is_optimal")) for row in ordered[idx + 1 : idx + 4]]
                valid_later = [flag for flag in later if flag is not None]
                sustained = len(valid_later) >= 2 and sum(flag is False for flag in valid_later) >= 2
                events.append({**base, "event_type": "sustained_optimality_loss" if sustained else "optimality_loss"})
            if current_opt is False and next_opt is True:
                events.append({**base, "event_type": "optimality_recovery"})
    return events


def _build_commitment_summary(trajectory_rows: list[dict[str, str]], prefix_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    event_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for event in _classify_events(prefix_rows):
        event_counts[event["example_id"]][event["event_type"]] += 1
    rows: list[dict[str, Any]] = []
    for row in trajectory_rows:
        counts = event_counts.get(row.get("example_id", ""), Counter())
        example_prefixes = [
            prefix for prefix in prefix_rows if prefix.get("example_id") == row.get("example_id")
        ]
        commitment_step = _as_int(row.get("commitment_step"))
        commitment_prefix = next(
            (
                prefix
                for prefix in example_prefixes
                if _as_int(prefix.get("reasoning_step_idx")) == commitment_step
            ),
            None,
        )
        maximum_chars = max(
            (_as_int(prefix.get("revealed_analysis_chars")) or 0 for prefix in example_prefixes),
            default=0,
        )
        commitment_chars = (
            _as_int(commitment_prefix.get("revealed_analysis_chars"))
            if commitment_prefix is not None
            else None
        )
        rows.append(
            {
                "example_id": row.get("example_id", ""),
                "trajectory_id": row.get("trajectory_id", ""),
                "step_index": row.get("step_index", ""),
                "failure_category": row.get("failure_category", ""),
                "final_action": row.get("final_action", ""),
                "observed_action": row.get("observed_action", ""),
                "optimal_actions_json": row.get("optimal_actions_json", ""),
                "n_reasoning_steps": row.get("n_reasoning_steps", ""),
                "analysis_unit": row.get(
                    "analysis_unit",
                    example_prefixes[0].get("analysis_unit", "packed_chunk")
                    if example_prefixes
                    else "packed_chunk",
                ),
                "commitment_step": row.get("commitment_step", ""),
                "commitment_step_progress": row.get("commitment_step_progress", ""),
                "commitment_character_progress": (
                    commitment_chars / maximum_chars
                    if commitment_chars is not None and maximum_chars
                    else ""
                ),
                "trajectory_remains_optimal": row.get("trajectory_remains_optimal", ""),
                "sustained_optimality_loss_step": row.get("wrong_turn_step_majority", ""),
                "sustained_optimality_loss_progress": row.get("wrong_turn_step_majority_progress", ""),
                "strict_optimality_loss_step": row.get("wrong_turn_step_strict", ""),
                "recovery_step": row.get("recovery_step", ""),
                "n_action_change_events": counts["action_change"],
                "n_optimality_loss_events": counts["optimality_loss"] + counts["sustained_optimality_loss"],
                "n_sustained_optimality_loss_events": counts["sustained_optimality_loss"],
                "n_optimality_recovery_events": counts["optimality_recovery"],
            }
        )
    return rows


def _load_geometry_from_activations(usable_rows: list[dict[str, str]], trajectory_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    if torch is None:
        return []
    geometry_inputs: list[dict[str, Any]] = []
    for row in usable_rows:
        for rep, path in _activation_paths(row).items():
            if rep == "boundary_window":
                # Boundary windows are retained in compatibility outputs but not
                # collapsed into v1 trajectory-geometry curves.
                continue
            geometry_inputs.append({**row, "representation_kind": rep, "activation_path": path})
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in geometry_inputs:
        layer = _as_int(row.get("layer"))
        if layer is None:
            continue
        grouped[(row["example_id"], layer, row["representation_kind"])].append(row)
    trajectory_lookup = {row["example_id"]: row for row in trajectory_rows}
    anchor_vectors = _build_anchor_vectors(geometry_inputs, trajectory_rows)
    output: list[dict[str, Any]] = []
    for (example_id, layer, rep), rows in grouped.items():
        ordered = sorted(rows, key=lambda r: _as_int(r.get("reasoning_step_idx")) or -1)
        vectors = [torch.load(row["activation_path"], map_location="cpu").tolist() for row in ordered]
        anchors = {
            _as_int(row.get("reasoning_step_idx")) or 0: anchor_vectors[(layer, rep, int(row.get("progress_bucket_10") or 0))]
            for row in ordered
            if (layer, rep, int(row.get("progress_bucket_10") or 0)) in anchor_vectors
        }
        metrics = compute_reasoning_geometry_metrics(
            vectors,
            anchor_vectors_by_bucket=anchors,
            step_indices=[_as_int(row.get("reasoning_step_idx")) or 0 for row in ordered],
        )
        traj = trajectory_lookup.get(example_id, {})
        for row, metric in zip(ordered, metrics):
            output.append(
                {
                    "example_id": example_id,
                    "trajectory_id": row.get("trajectory_id", ""),
                    "step_index": row.get("step_index", ""),
                    "failure_category": row.get("failure_category", ""),
                    "trajectory_group": traj.get("trajectory_group", row.get("trajectory_group", "")),
                    "layer": layer,
                    "representation_kind": rep,
                    "reasoning_step_idx": row.get("reasoning_step_idx", ""),
                    "reasoning_progress": row.get("reasoning_progress", ""),
                    "progress_bucket_10": row.get("progress_bucket_10", ""),
                    "activation_path": row.get("activation_path", ""),
                    **metric,
                }
            )
    return output


def _build_anchor_vectors(rows: list[dict[str, Any]], trajectory_rows: list[dict[str, str]]) -> dict[tuple[int, str, int], list[float]]:
    if torch is None:
        return {}
    optimal_ids = {
        row["example_id"]
        for row in trajectory_rows
        if str(row.get("trajectory_remains_optimal", "")).lower() == "true"
    }
    buckets: dict[tuple[int, str, int], list[list[float]]] = defaultdict(list)
    for row in rows:
        if row.get("example_id") not in optimal_ids:
            continue
        layer = _as_int(row.get("layer"))
        if layer is None:
            continue
        key = (layer, row.get("representation_kind", ""), int(row.get("progress_bucket_10") or 0))
        try:
            buckets[key].append(torch.load(row["activation_path"], map_location="cpu").tolist())
        except Exception:
            continue
    anchors: dict[tuple[int, str, int], list[float]] = {}
    for key, vectors in buckets.items():
        if not vectors:
            continue
        dim = len(vectors[0])
        anchors[key] = [sum(vec[idx] for vec in vectors) / len(vectors) for idx in range(dim)]
    return anchors


def _event_aligned_geometry(events: list[dict[str, Any]], geometry_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in geometry_rows:
        layer = _as_int(row.get("layer"))
        if layer is None:
            continue
        grouped[(row["example_id"], layer, row.get("representation_kind", ""))].append(row)
    output: list[dict[str, Any]] = []
    for event in events:
        event_step = int(event["next_reasoning_step_idx"])
        for (example_id, layer, rep), rows in grouped.items():
            if example_id != event["example_id"]:
                continue
            ordered = sorted(rows, key=lambda r: _as_int(r.get("reasoning_step_idx")) or -1)
            event_row = next((row for row in ordered if (_as_int(row.get("reasoning_step_idx")) or -1) == event_step), None)
            if event_row is None:
                continue
            previous = [row for row in ordered if (_as_int(row.get("reasoning_step_idx")) or -1) < event_step][-3:]
            out = {**event, "layer": layer, "representation_kind": rep, "pre_window_size": len(previous)}
            for metric in EVENT_GEOMETRY_METRICS:
                event_value = _as_float(event_row.get(metric))
                previous_values = [_as_float(row.get(metric)) for row in previous]
                previous_values = [value for value in previous_values if value is not None]
                previous_mean = mean(previous_values) if previous_values else None
                out[f"{metric}_at_event"] = event_value if event_value is not None else ""
                out[f"{metric}_previous_3_mean"] = previous_mean if previous_mean is not None else ""
                out[f"{metric}_delta_vs_previous_3"] = (
                    event_value - previous_mean
                    if event_value is not None and previous_mean is not None
                    else ""
                )
            output.append(out)
    return output


def _geometry_class_summary(geometry_rows: list[dict[str, Any]], trajectory_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    trajectory_class = {}
    for row in trajectory_rows:
        if row.get("wrong_turn_step_majority") not in {None, ""}:
            label = "sustained optimality loss"
        elif str(row.get("trajectory_remains_optimal", "")).lower() == "true":
            label = "all valid recommendations optimal"
        else:
            label = "mixed or transient suboptimality"
        trajectory_class[row["example_id"]] = label
    values: dict[tuple[int, str, str, str], list[float]] = defaultdict(list)
    for row in geometry_rows:
        layer = _as_int(row.get("layer"))
        if layer is None:
            continue
        label = trajectory_class.get(row.get("example_id"), "unknown")
        for metric in EVENT_GEOMETRY_METRICS:
            value = _as_float(row.get(metric))
            if value is not None:
                values[(layer, row.get("representation_kind", ""), label, metric)].append(value)
    rows: list[dict[str, Any]] = []
    for (layer, rep, label, metric), vals in sorted(values.items()):
        rows.append(
            {
                "layer": layer,
                "representation_kind": rep,
                "outcome_class": label,
                "metric": metric,
                "n_rows": len(vals),
                "mean": mean(vals) if vals else "",
                "minimum": min(vals) if vals else "",
                "maximum": max(vals) if vals else "",
            }
        )
    return rows


def _event_label(name: str) -> str:
    return str(name).replace("_", " ")


def _set_experiment1_plot_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _plot_outputs(out_dir: Path, event_rows: list[dict[str, Any]], event_geometry: list[dict[str, Any]], commitments: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    _set_experiment1_plot_style(plt)
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    blue = "#1f5f9f"
    light_blue = "#dbeafe"
    grid_color = "#d7e3f0"
    text_color = "#172033"

    progress_by_group: dict[str, list[float]] = defaultdict(list)
    analysis_units = {str(row.get("analysis_unit", "packed_chunk")) for row in commitments}
    analysis_unit = next(iter(analysis_units)) if len(analysis_units) == 1 else "analysis unit"
    for row in commitments:
        progress = _as_float(
            row.get("commitment_character_progress") or row.get("commitment_step_progress")
        )
        if progress is None:
            continue
        group = "failure" if row.get("failure_category") not in {"", "none"} else "control"
        progress_by_group[group].append(progress)
    if progress_by_group:
        fig, ax = plt.subplots(figsize=(5.8, 3.5))
        groups = [group for group in ("control", "failure") if progress_by_group.get(group)]
        labels = [f"{group} (n={len(progress_by_group[group])})" for group in groups]
        box = ax.boxplot([progress_by_group[group] for group in groups], tick_labels=labels, patch_artist=True)
        for patch in box["boxes"]:
            patch.set_facecolor(light_blue)
            patch.set_edgecolor(blue)
        for element in ("whiskers", "caps", "medians"):
            for artist in box[element]:
                artist.set_color(blue)
        ax.set_ylabel("Commitment position\n(fraction of reasoning characters revealed)")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title("Action Commitment Timing by State Group", color=text_color)
        ax.grid(axis="y", color=grid_color, linewidth=0.8)
        fig.tight_layout()
        fig.savefig(fig_dir / "commitment_timing.png", dpi=200)
        plt.close(fig)

    selected = [
        row for row in event_geometry
        if str(row.get("representation_kind")) == "mean_pool" and str(row.get("layer")) == "15"
    ]
    if not selected:
        selected = [
            row for row in event_geometry
            if str(row.get("representation_kind"))
            == "sentence_sampled_token_mean"
            and str(row.get("layer")) == "8"
        ]
    if not selected:
        selected = event_geometry
    if selected:
        previous_unit = "sentence" if analysis_unit == "sentence" else "reasoning unit"
        metric_titles = {
            "aligned_change": "Update direction vs. net trace direction",
            "update_norm": "Activation update size",
            "adjacent_step_cosine": f"Similarity to previous {previous_unit}",
            "optimality_anchor_cosine": "Similarity to always-optimal prefix traces",
        }
        fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.2), sharex=True)
        plotted = False
        names = sorted({str(row.get("event_type", "")) for row in selected})
        for ax, metric in zip(axes.ravel(), EVENT_GEOMETRY_METRICS):
            by_event: dict[str, list[float]] = defaultdict(list)
            for row in selected:
                value = _as_float(row.get(f"{metric}_delta_vs_previous_3"))
                if value is not None:
                    by_event[str(row.get("event_type", ""))].append(value)
            values = [mean(by_event[name]) if by_event.get(name) else 0.0 for name in names]
            ax.bar([_event_label(name) for name in names], values, color=blue)
            ax.axhline(0, color="#64748b", linewidth=1)
            ax.set_title(metric_titles[metric], color=text_color)
            ax.grid(axis="y", color=grid_color, linewidth=0.8)
            ax.tick_params(axis="x", rotation=25)
            plotted = plotted or any(by_event.values())
        if plotted:
            unit_label = "sentences" if analysis_unit == "sentence" else "reasoning units"
            fig.supylabel(
                f"Metric at event minus mean over previous 3 {unit_label}"
            )
            fig.suptitle("Activation Metric Changes at Action Events", fontsize=11, color=text_color)
            fig.tight_layout()
            fig.savefig(fig_dir / "event_aligned_geometry.png", dpi=200)
            plt.close(fig)

    entropy_by_event: dict[str, list[float]] = defaultdict(list)
    for row in event_rows:
        value = _as_float(row.get("delta_action_entropy_bits"))
        if value is None:
            value = _as_float(row.get("delta_action_mc_entropy"))
        if value is not None:
            entropy_by_event[str(row.get("event_type", ""))].append(value)
    if entropy_by_event:
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        label_map = {
            "action_change": "Any action change",
            "commitment_onset": "Commitment onset",
            "optimality_loss": "Transient optimality loss",
            "optimality_recovery": "Recovery to optimal",
            "sustained_optimality_loss": "Sustained optimality loss",
        }
        order = (
            "action_change",
            "commitment_onset",
            "optimality_loss",
            "optimality_recovery",
            "sustained_optimality_loss",
        )
        names = [name for name in order if name in entropy_by_event]
        labels = [f"{label_map[name]}\n(n={len(entropy_by_event[name])})" for name in names]
        ax.bar(labels, [mean(entropy_by_event[name]) for name in names], color=blue)
        ax.axhline(0, color="#64748b", linewidth=1)
        ax.set_ylabel("Mean entropy change (bits)\nevent prefix - previous prefix\nnegative = lower uncertainty")
        entropy_methods = {
            str(row.get("action_entropy_method", "")) for row in event_rows
        }
        entropy_title = (
            "Change in Action-Token Logprob Entropy at Decision Events"
            if entropy_methods == {"T=0.7 four-action token logprobs"}
            else "Change in Action Entropy at Decision Events"
        )
        ax.set_title(entropy_title, color=text_color)
        ax.grid(axis="y", color=grid_color, linewidth=0.8)
        ax.tick_params(axis="x", rotation=18)
        fig.tight_layout()
        fig.savefig(fig_dir / "action_entropy_events.png", dpi=200)
        plt.close(fig)



def _zip_data_members(archive: zipfile.ZipFile) -> list[str]:
    members: list[str] = []
    for name in archive.namelist():
        path = Path(name)
        if "__MACOSX" in path.parts:
            continue
        if path.name in {"", ".DS_Store"} or path.name.startswith("._"):
            continue
        members.append(name)
    return members


def _discover_packed_cot_records(archive: zipfile.ZipFile) -> tuple[str, str, dict[tuple[int, int], tuple[str, str]]]:
    records: dict[tuple[int, int], tuple[str, str]] = {}
    roots: Counter[str] = Counter()
    model_dirs: Counter[str] = Counter()
    for name in _zip_data_members(archive):
        parts = Path(name).parts
        if len(parts) < 5 or not name.endswith("cot.json"):
            continue
        root, model_dir, layer_part, step_part = parts[:4]
        if not layer_part.startswith("layer_") or not step_part.startswith("step_"):
            continue
        layer = _as_int(layer_part.removeprefix("layer_"))
        step = _as_int(step_part.removeprefix("step_"))
        if layer is None or step is None:
            continue
        bf16_name = name[:-5] + ".bf16"
        if bf16_name not in archive.namelist():
            continue
        records[(layer, step)] = (name, bf16_name)
        roots[root] += 1
        model_dirs[model_dir] += 1
    if not records:
        raise ValueError("No layer_*/step_*/cot.json packed BF16 records found in archive")
    return roots.most_common(1)[0][0], model_dirs.most_common(1)[0][0], records


def _load_bf16_matrix_from_zip(archive: zipfile.ZipFile, json_member: str, bf16_member: str):
    if torch is None:
        raise RuntimeError("torch is required to load packed BF16 activations")
    meta = json.loads(archive.read(json_member).decode("utf-8"))
    shape = tuple(int(dim) for dim in meta["shape"])
    raw_bytes = bytearray(archive.read(bf16_member))
    raw = torch.frombuffer(raw_bytes, dtype=torch.uint16)
    expected = math.prod(shape)
    if raw.numel() != expected:
        raise ValueError(f"Packed BF16 size mismatch for {bf16_member}: got {raw.numel()}, expected {expected}")
    return raw.view(torch.bfloat16).reshape(shape), meta


def _discover_packed_cot_records_dir(
    activation_dir: Path,
) -> tuple[str, str, dict[tuple[int, int], tuple[Path, Path]]]:
    records: dict[tuple[int, int], tuple[Path, Path]] = {}
    roots: Counter[str] = Counter()
    model_dirs: Counter[str] = Counter()
    for json_path in activation_dir.glob("*/*/layer_*/step_*/cot.json"):
        relative = json_path.relative_to(activation_dir)
        if len(relative.parts) != 5:
            continue
        root, model_dir, layer_part, step_part, _name = relative.parts
        layer = _as_int(layer_part.removeprefix("layer_"))
        step = _as_int(step_part.removeprefix("step_"))
        bf16_path = json_path.with_suffix(".bf16")
        if layer is None or step is None or not bf16_path.exists():
            continue
        records[(layer, step)] = (json_path, bf16_path)
        roots[root] += 1
        model_dirs[model_dir] += 1
    if not records:
        raise ValueError(f"No extracted layer_*/step_*/cot records found in {activation_dir}")
    return roots.most_common(1)[0][0], model_dirs.most_common(1)[0][0], records


def _load_bf16_matrix_from_files(json_path: Path, bf16_path: Path):
    if torch is None:
        raise RuntimeError("torch is required to load packed BF16 activations")
    meta = json.loads(json_path.read_text())
    shape = tuple(int(dim) for dim in meta["shape"])
    expected = math.prod(shape)
    raw = torch.from_file(str(bf16_path), dtype=torch.uint16, size=expected)
    if raw.numel() != expected:
        raise ValueError(
            f"Packed BF16 size mismatch for {bf16_path}: got {raw.numel()}, expected {expected}"
        )
    return raw.view(torch.bfloat16).reshape(shape), meta


def _load_analysis_texts_by_step(trajectory_path: Path) -> dict[int, str]:
    payload = json.loads(trajectory_path.read_text())
    output: dict[int, str] = {}
    for idx, step in enumerate(payload.get("steps", [])):
        step_idx = _as_int(step.get("step_id", step.get("step_index", idx)))
        if step_idx is None:
            continue
        analysis_text, _final = _extract_analysis_and_final(str(step.get("output_text", "")))
        output[step_idx] = analysis_text
    return output


def _load_raw_analysis_by_step(
    trajectory_path: Path,
) -> tuple[dict[int, str], dict[int, list[int]]]:
    payload = json.loads(trajectory_path.read_text())
    texts: dict[int, str] = {}
    token_ids: dict[int, list[int]] = {}
    for idx, step in enumerate(payload.get("steps", [])):
        step_idx = _as_int(step.get("step_id", step.get("step_index", idx)))
        output_text = str(step.get("output_text", ""))
        if step_idx is None or ANALYSIS_START not in output_text or FINAL_START not in output_text:
            continue
        texts[step_idx] = output_text.split(ANALYSIS_START, 1)[1].split(FINAL_START, 1)[0]
        token_ids[step_idx] = [
            int(token["token_id"])
            for token in step.get("output_tokens", [])
            if "analysis" in token.get("token_groups", [])
        ]
    return texts, token_ids


def _chunk_token_indices(offsets: Sequence[tuple[int, int]], char_start: int, char_end: int) -> list[int]:
    return [idx for idx, (start, end) in enumerate(offsets) if end > start and end > char_start and start < char_end]


def _sampled_rows_for_chunks(
    *,
    chunks: list[dict[str, str]],
    offsets: Sequence[tuple[int, int]],
    token_indices: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not token_indices:
        return [], {"relative_base": "", "sampled_analysis_tokens": 0, "sampled_tokens_outside_chunks": 0}
    relative_base = min(int(tok["relative_idx"]) for tok in token_indices)
    sampled_by_analysis_token = {
        int(tok["relative_idx"]) - relative_base: (row_idx, int(tok["relative_idx"]), int(tok["absolute_idx"]))
        for row_idx, tok in enumerate(token_indices)
    }
    covered_sampled_tokens: set[int] = set()
    rows: list[dict[str, Any]] = []
    for chunk in sorted(chunks, key=lambda r: int(r["analysis_id"])):
        char_start = int(chunk["char_start"])
        char_end = int(chunk["char_end"])
        token_span = _chunk_token_indices(offsets, char_start, char_end)
        sampled = [
            (tok_idx, *sampled_by_analysis_token[tok_idx])
            for tok_idx in token_span
            if tok_idx in sampled_by_analysis_token
        ]
        covered_sampled_tokens.update(tok_idx for tok_idx, *_rest in sampled)
        rows.append(
            {
                **chunk,
                "chunk_token_start": min(token_span) if token_span else "",
                "chunk_token_end_exclusive": (max(token_span) + 1) if token_span else "",
                "n_chunk_tokens": len(token_span),
                "n_sampled_tokens": len(sampled),
                "sampled_analysis_token_indices_json": json.dumps([item[0] for item in sampled]),
                "sampled_relative_indices_json": json.dumps([item[2] for item in sampled]),
                "sampled_absolute_indices_json": json.dumps([item[3] for item in sampled]),
                "sampled_matrix_rows_json": json.dumps([item[1] for item in sampled]),
                "covered_by_sampled_activation": bool(sampled),
            }
        )
    sampled_in_analysis = {tok for tok in sampled_by_analysis_token if 0 <= tok < len(offsets)}
    meta = {
        "relative_base": relative_base,
        "sampled_analysis_tokens": len(sampled_in_analysis),
        "sampled_tokens_outside_chunks": len(sampled_in_analysis - covered_sampled_tokens),
    }
    return rows, meta


def _write_weisheng_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Weisheng Activation Chunk Mapping Report",
        "",
        "## Method",
        "",
        "The source archive stores GPT-OSS-20B COT activations every two generated analysis tokens. For each canonical reasoning chunk, this command averages all sampled token activations whose token offsets overlap that chunk's character span. Chunks with no sampled token are kept in the coverage table but omitted from the chunk-activation tensor table.",
        "",
        "This is suitable for chunk-level analysis, but not for token-level boundary claims. Missing chunks must be reported because one-token or unlucky-boundary chunks can have no sampled activation under stride-2 extraction.",
        "",
        "## Coverage",
        "",
        f"- Source archive: `{manifest.get('archive_path')}`",
        f"- Source trajectory: `{manifest.get('trajectory_id')}`",
        f"- Source model directory: `{manifest.get('model_dir')}`",
        f"- Archive layers used: {manifest.get('layers')}",
        f"- Archive steps used: {manifest.get('steps')}",
        f"- Canonical chunks overall: {manifest.get('n_canonical_chunks_total')}",
        f"- Canonical chunks in archived trajectory steps: {manifest.get('n_archive_scope_chunks')}",
        f"- Chunks with at least one sampled activation: {manifest.get('n_chunks_with_sampled_activation')}",
        f"- Chunks without sampled activation: {manifest.get('n_chunks_without_sampled_activation')}",
        f"- Chunk coverage within archived steps: {manifest.get('chunk_coverage_rate_within_archive_scope'):.3f}",
        f"- Relation to all canonical chunks: `{manifest.get('relation_to_all_canonical_chunks')}`",
        f"- Relation within archived trajectory steps: `{manifest.get('relation_within_archive_scope')}`",
        f"- Token-level relation within archived trajectory steps: `{manifest.get('activation_token_relation_within_archive_scope')}`",
        f"- Activation steps without canonical chunks: {manifest.get('n_activation_steps_without_canonical_chunks')}",
        f"- Sampled analysis tokens outside canonical chunks: {manifest.get('n_sampled_tokens_outside_chunks')}",
        "",
        "## Outputs",
        "",
        "- `chunk_activation_rows.csv`: one row per covered chunk and layer, with a path to the averaged tensor.",
        "- `chunk_coverage_rows.csv`: one row per canonical chunk in the archived trajectory steps, including no-activation chunks.",
        "- `coverage_summary.csv`: compact coverage decision table.",
        "- `chunk_activations/`: averaged chunk tensors saved as `.pt` files.",
        "",
    ]
    path.write_text("\n".join(lines))


def build_weisheng_chunk_averaged_activations(
    *,
    archive_path: str = DEFAULT_WEISHENG_ACTIVATION_ARCHIVE,
    chunked_trajectories_path: str = DEFAULT_CHUNKED_TRAJECTORIES,
    trajectory_dir: str = DEFAULT_TRAJECTORY_DIR,
    output_dir: str = "outputs/experiment1_activation_monitor/weisheng_chunk_averaged_activations",
    tokenizer_name_or_path: str = DEFAULT_GPTOSS20B_TOKENIZER,
    layers: tuple[int, ...] = (),
    save_dtype: str = "float32",
) -> dict[str, Any]:
    """Average Weisheng's stride-2 COT activations inside canonical DoorKey chunks."""
    out = Path(output_dir)
    _validate_output_root(out)
    out.mkdir(parents=True, exist_ok=True)
    activation_out = out / "chunk_activations"
    activation_out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, local_files_only=Path(tokenizer_name_or_path).exists())
    chunk_rows_all = _read_csv(Path(chunked_trajectories_path))
    chunk_rows_by_trace: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in chunk_rows_all:
        chunk_rows_by_trace[row["trace_id"]].append(row)

    coverage_rows: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []
    per_step_meta: list[dict[str, Any]] = []
    hidden_dims: set[int] = set()
    selected_layers: list[int]
    activation_steps_without_chunks = 0

    with zipfile.ZipFile(archive_path) as archive:
        trajectory_id, model_dir, records = _discover_packed_cot_records(archive)
        archive_layers = sorted({layer for layer, _step in records})
        archive_steps = sorted({step for _layer, step in records})
        selected_layers = sorted(set(layers) & set(archive_layers)) if layers else archive_layers
        if layers and len(selected_layers) != len(set(layers)):
            missing = sorted(set(layers) - set(selected_layers))
            raise ValueError(f"Requested layers absent from archive: {missing}. Available layers: {archive_layers}")
        trajectory_path = Path(trajectory_dir) / f"{trajectory_id}.json"
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Missing trajectory JSON for archive root {trajectory_id}: {trajectory_path}")
        analysis_texts = _load_analysis_texts_by_step(trajectory_path)

        for step in archive_steps:
            trace = _trace_id(trajectory_id, step)
            chunks = chunk_rows_by_trace.get(trace, [])
            if not chunks:
                activation_steps_without_chunks += 1
                continue
            analysis_text = analysis_texts.get(step, "")
            if not analysis_text:
                raise ValueError(f"Missing analysis text for {trajectory_id} step {step}")
            encoded = tokenizer(analysis_text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
            first_layer = selected_layers[0]
            matrix, meta = _load_bf16_matrix_from_zip(archive, *records[(first_layer, step)])
            hidden_dims.add(int(matrix.shape[1]))
            step_coverage_rows, step_meta = _sampled_rows_for_chunks(
                chunks=chunks,
                offsets=offsets,
                token_indices=list(meta.get("token_indices", [])),
            )
            for row in step_coverage_rows:
                row.update(
                    {
                        "trajectory_id": trajectory_id,
                        "step_index": step,
                        "trace_id": trace,
                        "source_archive": archive_path,
                        "source_model_dir": model_dir,
                        "sampled_token_stride": 2,
                        "relative_index_base": step_meta["relative_base"],
                    }
                )
                coverage_rows.append(row)
            per_step_meta.append({"step_index": step, "trace_id": trace, **step_meta, "n_chunks": len(chunks)})

            sampled_by_chunk = {
                int(row["analysis_id"]): json.loads(str(row["sampled_matrix_rows_json"]))
                for row in step_coverage_rows
            }
            coverage_by_chunk = {int(row["analysis_id"]): row for row in step_coverage_rows}
            for layer in selected_layers:
                if (layer, step) not in records:
                    continue
                layer_matrix, layer_meta = _load_bf16_matrix_from_zip(archive, *records[(layer, step)])
                hidden_dims.add(int(layer_matrix.shape[1]))
                for analysis_id, row_indices in sampled_by_chunk.items():
                    if not row_indices:
                        continue
                    chunk_meta = coverage_by_chunk[analysis_id]
                    mean_vec = layer_matrix[row_indices].float().mean(dim=0)
                    if save_dtype == "bfloat16":
                        mean_vec = mean_vec.to(torch.bfloat16)
                    elif save_dtype == "float16":
                        mean_vec = mean_vec.to(torch.float16)
                    elif save_dtype != "float32":
                        raise ValueError("save_dtype must be one of: float32, float16, bfloat16")
                    tensor_path = activation_out / trajectory_id / f"step_{step:03d}" / f"layer_{layer}" / f"chunk_{analysis_id:03d}.pt"
                    tensor_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(mean_vec.cpu(), tensor_path)
                    activation_rows.append(
                        {
                            "example_id": trace,
                            "trace_id": trace,
                            "trajectory_id": trajectory_id,
                            "step_index": step,
                            "chunk_id": analysis_id,
                            "reasoning_step_idx": analysis_id + 1,
                            "layer": layer,
                            "representation_kind": "chunk_sampled_token_mean",
                            "activation_path": str(tensor_path),
                            "step_mean_activation_path": str(tensor_path),
                            "step_start_char_in_analysis": chunk_meta["char_start"],
                            "step_end_char_in_analysis": chunk_meta["char_end"],
                            "step_start_token": chunk_meta["chunk_token_start"],
                            "step_end_token": chunk_meta["chunk_token_end_exclusive"],
                            "reasoning_progress": (
                                (analysis_id + 1) / max(1, len(sampled_by_chunk))
                            ),
                            "progress_bucket_10": int(
                                round(((analysis_id + 1) / max(1, len(sampled_by_chunk))) * 10)
                            ),
                            "n_sampled_tokens": len(row_indices),
                            "chunk_token_start": chunk_meta["chunk_token_start"],
                            "chunk_token_end_exclusive": chunk_meta["chunk_token_end_exclusive"],
                            "char_start": chunk_meta["char_start"],
                            "char_end": chunk_meta["char_end"],
                            "sampled_matrix_rows_json": chunk_meta["sampled_matrix_rows_json"],
                            "sampled_relative_indices_json": chunk_meta["sampled_relative_indices_json"],
                            "sampled_absolute_indices_json": chunk_meta["sampled_absolute_indices_json"],
                            "source_archive": archive_path,
                            "source_model_dir": model_dir,
                            "source_category": layer_meta.get("category", "cot"),
                            "sampled_token_stride": 2,
                            "hidden_dim": int(layer_matrix.shape[1]),
                        }
                    )

    covered_chunk_keys = {
        (row["trace_id"], int(row["analysis_id"]))
        for row in coverage_rows
        if row.get("covered_by_sampled_activation") is True
    }
    archive_scope_chunk_keys = {(row["trace_id"], int(row["analysis_id"])) for row in coverage_rows}
    n_scope = len(archive_scope_chunk_keys)
    n_covered = len(covered_chunk_keys)
    n_missing = n_scope - n_covered
    n_outside = sum(int(row["sampled_tokens_outside_chunks"]) for row in per_step_meta)
    relation_to_all = "whole" if n_scope == len(chunk_rows_all) else "subset"
    if n_covered == 0:
        relation_within = "none"
    elif n_missing == 0:
        relation_within = "whole"
    else:
        relation_within = "subset"
    token_relation = "superset" if n_outside else relation_within
    manifest = {
        "archive_path": archive_path,
        "chunked_trajectories_path": chunked_trajectories_path,
        "trajectory_dir": trajectory_dir,
        "output_dir": str(out),
        "trajectory_id": activation_rows[0]["trajectory_id"] if activation_rows else "",
        "model_dir": activation_rows[0]["source_model_dir"] if activation_rows else "",
        "layers": selected_layers,
        "steps": sorted({row["step_index"] for row in coverage_rows}),
        "save_dtype": save_dtype,
        "n_canonical_chunks_total": len(chunk_rows_all),
        "n_archive_scope_chunks": n_scope,
        "n_chunks_with_sampled_activation": n_covered,
        "n_chunks_without_sampled_activation": n_missing,
        "chunk_coverage_rate_within_archive_scope": (n_covered / n_scope) if n_scope else 0.0,
        "relation_to_all_canonical_chunks": relation_to_all,
        "relation_within_archive_scope": relation_within,
        "activation_token_relation_within_archive_scope": token_relation,
        "n_activation_steps_without_canonical_chunks": activation_steps_without_chunks,
        "n_sampled_tokens_outside_chunks": n_outside,
        "n_chunk_activation_rows": len(activation_rows),
        "hidden_dims": sorted(hidden_dims),
        "per_step_meta": per_step_meta,
    }
    summary_rows = [
        {
            "scope": "all_canonical_chunks",
            "n_chunks": len(chunk_rows_all),
            "n_covered_chunks": n_covered,
            "coverage_rate": n_covered / len(chunk_rows_all) if chunk_rows_all else 0.0,
            "relation": relation_to_all,
        },
        {
            "scope": "archived_trajectory_steps",
            "n_chunks": n_scope,
            "n_covered_chunks": n_covered,
            "coverage_rate": n_covered / n_scope if n_scope else 0.0,
            "relation": relation_within,
        },
        {
            "scope": "archived_trajectory_step_sampled_tokens",
            "n_chunks": n_scope,
            "n_covered_chunks": n_covered,
            "coverage_rate": n_covered / n_scope if n_scope else 0.0,
            "relation": token_relation,
        },
    ]
    _write_csv(out / "chunk_coverage_rows.csv", coverage_rows)
    _write_csv(out / "chunk_activation_rows.csv", activation_rows)
    _write_csv(out / "coverage_summary.csv", summary_rows)
    _write_json(out / "run_manifest.json", manifest)
    _write_weisheng_report(out / "coverage_report.md", manifest)
    return manifest


def build_weisheng_sentence_averaged_activations(
    *,
    activation_dir: str = (
        "outputs/experiment1_activation_monitor/weisheng_raw_stride2_activations"
    ),
    sentences_path: str = DEFAULT_SENTENCES,
    trajectory_dir: str = DEFAULT_TRAJECTORY_DIR,
    output_dir: str = (
        "outputs/experiment1_activation_monitor/weisheng_sentence_averaged_activations"
    ),
    tokenizer_name_or_path: str = DEFAULT_GPTOSS20B_TOKENIZER,
    layers: tuple[int, ...] = (),
    save_dtype: str = "bfloat16",
) -> dict[str, Any]:
    """Average extracted stride-2 COT activations within canonical sentences."""
    source = Path(activation_dir)
    out = Path(output_dir)
    _validate_output_root(out)
    out.mkdir(parents=True, exist_ok=True)
    tensor_root = out / "sentence_activations"
    trailing_tensor_root = out / "sentence_trailing_3_activations"
    tensor_root.mkdir(parents=True, exist_ok=True)
    trailing_tensor_root.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        local_files_only=Path(tokenizer_name_or_path).exists(),
    )
    sentence_rows_all = [
        row for row in _read_csv(Path(sentences_path)) if row.get("kind") == "reasoning"
    ]
    sentence_rows_by_trace: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sentence_rows_all:
        sentence_rows_by_trace[row["trace_id"]].append(
            {
                **row,
                "analysis_id": row["sentence_id"],
            }
        )

    trajectory_id, model_dir, records = _discover_packed_cot_records_dir(source)
    archive_layers = sorted({layer for layer, _step in records})
    archive_steps = sorted({step for _layer, step in records})
    selected_layers = sorted(set(layers) & set(archive_layers)) if layers else archive_layers
    if layers and len(selected_layers) != len(set(layers)):
        missing = sorted(set(layers) - set(selected_layers))
        raise ValueError(
            f"Requested layers absent from extracted activations: {missing}. "
            f"Available layers: {archive_layers}"
        )
    trajectory_path = Path(trajectory_dir) / f"{trajectory_id}.json"
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Missing trajectory JSON for {trajectory_id}: {trajectory_path}")
    analysis_texts = _load_analysis_texts_by_step(trajectory_path)
    raw_analysis_texts, stored_analysis_token_ids = _load_raw_analysis_by_step(
        trajectory_path
    )

    coverage_rows: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []
    per_step_meta: list[dict[str, Any]] = []
    hidden_dims: set[int] = set()
    for step in archive_steps:
        trace = _trace_id(trajectory_id, step)
        sentences = sentence_rows_by_trace.get(trace, [])
        if not sentences:
            raise ValueError(f"No canonical reasoning sentences for extracted trace {trace}")
        analysis_text = analysis_texts.get(step, "")
        raw_analysis_text = raw_analysis_texts.get(step, "")
        leading_trim = len(raw_analysis_text) - len(raw_analysis_text.lstrip())
        encoded = tokenizer(
            raw_analysis_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        if list(encoded["input_ids"]) != stored_analysis_token_ids.get(step, []):
            raise ValueError(
                f"Tokenizer IDs do not exactly match stored analysis token IDs for {trace}."
            )
        offsets = [
            (int(start) - leading_trim, int(end) - leading_trim)
            for start, end in encoded["offset_mapping"]
        ]
        first_matrix, first_meta = _load_bf16_matrix_from_files(
            *records[(selected_layers[0], step)]
        )
        hidden_dims.add(int(first_matrix.shape[1]))
        step_coverage, step_meta = _sampled_rows_for_chunks(
            chunks=sentences,
            offsets=offsets,
            token_indices=list(first_meta.get("token_indices", [])),
        )
        for row in step_coverage:
            row.update(
                {
                    "trajectory_id": trajectory_id,
                    "step_index": step,
                    "trace_id": trace,
                    "sentence_id": int(row["sentence_id"]),
                    "analysis_unit": "sentence",
                    "source_activation_dir": str(source),
                    "source_model_dir": model_dir,
                    "sampled_token_stride": 2,
                    "relative_index_base": step_meta["relative_base"],
                }
            )
            coverage_rows.append(row)
        per_step_meta.append(
            {
                "step_index": step,
                "trace_id": trace,
                "n_sentences": len(sentences),
                **step_meta,
            }
        )
        rows_by_sentence = {
            int(row["sentence_id"]): json.loads(str(row["sampled_matrix_rows_json"]))
            for row in step_coverage
        }
        coverage_by_sentence = {
            int(row["sentence_id"]): row for row in step_coverage
        }
        n_sentences = len(step_coverage)
        for layer in selected_layers:
            matrix, layer_meta = _load_bf16_matrix_from_files(*records[(layer, step)])
            hidden_dims.add(int(matrix.shape[1]))
            for sentence_id, matrix_rows in rows_by_sentence.items():
                if not matrix_rows:
                    continue
                sentence_meta = coverage_by_sentence[sentence_id]
                representations = (
                    (
                        "sentence_sampled_token_mean",
                        matrix[matrix_rows].float().mean(dim=0),
                        matrix_rows,
                        tensor_root,
                    ),
                    (
                        "sentence_last_up_to_3_sampled_token_mean",
                        matrix[matrix_rows[-3:]].float().mean(dim=0),
                        matrix_rows[-3:],
                        trailing_tensor_root,
                    ),
                )
                for representation_kind, vector, used_rows, representation_root in representations:
                    if save_dtype == "bfloat16":
                        vector = vector.to(torch.bfloat16)
                    elif save_dtype == "float16":
                        vector = vector.to(torch.float16)
                    elif save_dtype != "float32":
                        raise ValueError(
                            "save_dtype must be one of: float32, float16, bfloat16"
                        )
                    tensor_path = (
                        representation_root
                        / trajectory_id
                        / f"step_{step:03d}"
                        / f"layer_{layer}"
                        / f"sentence_{sentence_id:03d}.pt"
                    )
                    tensor_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(vector.cpu(), tensor_path)
                    activation_rows.append(
                        {
                            "example_id": trace,
                            "trace_id": trace,
                            "trajectory_id": trajectory_id,
                            "step_index": step,
                            "sentence_id": sentence_id,
                            "reasoning_step_idx": sentence_id + 1,
                            "analysis_unit": "sentence",
                            "layer": layer,
                            "representation_kind": representation_kind,
                            "activation_path": str(tensor_path),
                            "step_mean_activation_path": str(tensor_path),
                            "step_start_char_in_analysis": sentence_meta["char_start"],
                            "step_end_char_in_analysis": sentence_meta["char_end"],
                            "step_start_token": sentence_meta["chunk_token_start"],
                            "step_end_token": sentence_meta["chunk_token_end_exclusive"],
                            "reasoning_progress": (sentence_id + 1) / max(1, n_sentences),
                            "reasoning_character_progress": (
                                int(sentence_meta["char_end"]) / len(analysis_text)
                                if analysis_text
                                else 0.0
                            ),
                            "progress_bucket_10": int(
                                round(((sentence_id + 1) / max(1, n_sentences)) * 10)
                            ),
                            "n_sampled_tokens": len(matrix_rows),
                            "n_sampled_tokens_in_representation": len(used_rows),
                            "sentence_token_start": sentence_meta["chunk_token_start"],
                            "sentence_token_end_exclusive": sentence_meta[
                                "chunk_token_end_exclusive"
                            ],
                            "char_start": sentence_meta["char_start"],
                            "char_end": sentence_meta["char_end"],
                            "sampled_matrix_rows_json": json.dumps(used_rows),
                            "sampled_relative_indices_json": sentence_meta[
                                "sampled_relative_indices_json"
                            ],
                            "sampled_absolute_indices_json": sentence_meta[
                                "sampled_absolute_indices_json"
                            ],
                            "source_activation_dir": str(source),
                            "source_model_dir": model_dir,
                            "source_category": layer_meta.get("category", "cot"),
                            "sampled_token_stride": 2,
                            "hidden_dim": int(matrix.shape[1]),
                        }
                    )

    covered = {
        (row["trace_id"], int(row["sentence_id"]))
        for row in coverage_rows
        if row.get("covered_by_sampled_activation") is True
    }
    scope = {(row["trace_id"], int(row["sentence_id"])) for row in coverage_rows}
    n_outside = sum(int(row["sampled_tokens_outside_chunks"]) for row in per_step_meta)
    manifest = {
        "status": "completed",
        "activation_dir": str(source),
        "sentences_path": sentences_path,
        "trajectory_dir": trajectory_dir,
        "output_dir": str(out),
        "trajectory_id": trajectory_id,
        "model_dir": model_dir,
        "layers": selected_layers,
        "steps": archive_steps,
        "save_dtype": save_dtype,
        "analysis_unit": "sentence",
        "aggregations": {
            "sentence_sampled_token_mean": (
                "mean over all available stride-2 sampled positions overlapping each sentence"
            ),
            "sentence_last_up_to_3_sampled_token_mean": (
                "mean over the last min(number available, 3) stride-2 sampled positions "
                "overlapping each sentence"
            ),
        },
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "tokenizer_ids_match_stored_trajectory_tokens": True,
        "n_canonical_reasoning_sentences_total": len(sentence_rows_all),
        "n_archive_scope_sentences": len(scope),
        "n_sentences_with_sampled_activation": len(covered),
        "n_sentences_without_sampled_activation": len(scope - covered),
        "sentence_coverage_rate": len(covered) / len(scope) if scope else 0.0,
        "n_sentence_activation_rows": len(activation_rows),
        "n_sentence_activation_rows_by_representation": {
            kind: sum(row["representation_kind"] == kind for row in activation_rows)
            for kind in (
                "sentence_sampled_token_mean",
                "sentence_last_up_to_3_sampled_token_mean",
            )
        },
        "n_sampled_tokens_outside_sentences": n_outside,
        "hidden_dims": sorted(hidden_dims),
        "per_step_meta": per_step_meta,
    }
    _write_csv(out / "sentence_coverage_rows.csv", coverage_rows)
    _write_csv(out / "sentence_activation_rows.csv", activation_rows)
    _write_json(out / "run_manifest.json", manifest)
    report = [
        "# Weisheng Sentence Activation Mapping",
        "",
        "Two vectors are saved for each covered sentence and layer: the mean of all available stride-2 sampled COT positions, and the mean of the last up to three available sampled positions. Token offsets determine sentence overlap.",
        "Sentences with no sampled token are reported but receive no tensor.",
        "",
        f"- Trajectory: `{trajectory_id}`",
        f"- Environment steps: {archive_steps}",
        f"- Layers: {selected_layers}",
        f"- Reasoning sentences in these steps: {len(scope)}",
        f"- Sentences with a vector: {len(covered)}",
        f"- Sentence-layer-representation rows: {len(activation_rows)}",
        f"- Sentences without a vector: {len(scope - covered)}",
        f"- Coverage: {manifest['sentence_coverage_rate']:.1%}",
        f"- Saved tensor rows: {len(activation_rows)}",
        "",
        "These are sentence means, not exact sentence-final-token activations. The source sampled every second COT token, so an exact final-token vector is unavailable whenever that token was not sampled.",
    ]
    (out / "coverage_report.md").write_text("\n".join(report) + "\n")
    return manifest

def _write_run_report(
    path: Path,
    *,
    compatibility: dict[str, Any],
    events: list[dict[str, Any]],
    event_geometry: list[dict[str, Any]],
    commitments: list[dict[str, Any]],
) -> None:
    event_counts = Counter(row["event_type"] for row in events)
    commitment_values = [_as_float(row.get("commitment_step_progress")) for row in commitments]
    commitment_values = [value for value in commitment_values if value is not None]
    entropy_deltas = [_as_float(row.get("delta_action_mc_entropy")) for row in events]
    entropy_deltas = [value for value in entropy_deltas if value is not None]
    lines = [
        "# Experiment 1 Activation Monitor",
        "",
        "## Status",
        "",
        f"- Activation compatibility: `{compatibility.get('status')}`",
        f"- Usable activation rows: {compatibility.get('n_usable_rows', 0)}",
        f"- Action events: {len(events)}",
        f"- Event-aligned geometry rows: {len(event_geometry)}",
        f"- Trajectories: {len(commitments)}",
        "",
        "## Definitions",
        "",
        "- Recommended action: the action elicited from the model after showing the grid state and a prefix of the existing reasoning trace.",
        "- Action change: the recommended action differs between adjacent reasoning prefixes.",
        "- Optimality loss: the recommendation changes from planner-optimal to planner-suboptimal.",
        "- Sustained optimality loss: after an optimal-to-suboptimal change, at least two of the next three valid recommendations are suboptimal.",
        "- Optimality recovery: the recommendation changes from planner-suboptimal to planner-optimal.",
        "- Commitment onset: the first prefix where the recommendation equals the final full-trace action and all later valid prefix recommendations keep that action.",
        "- Action entropy: Shannon entropy over repeated sampled action recommendations for the same prefix.",
        "",
        "## Action Events",
        "",
        "| Event type | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {event_type} | {count} |" for event_type, count in sorted(event_counts.items()))
    lines.append("")
    if commitment_values:
        lines.extend([
            "## Commitment Timing",
            "",
            f"- Mean commitment progress: {mean(commitment_values):.3f}",
            f"- Number with commitment step: {len(commitment_values)}",
            "",
        ])
    if entropy_deltas:
        lines.extend([
            "## Action Entropy",
            "",
            f"- Events with action entropy delta: {len(entropy_deltas)}",
            f"- Mean action entropy delta at events: {mean(entropy_deltas):.3f}",
            "",
        ])
    if compatibility.get("status") != "compatible":
        lines.extend([
            "## Limitation",
            "",
            "The supplied activation index is not compatible with the canonical chunking scheme, so geometry outputs are absent or should be treated as unavailable for primary Experiment 1 claims. Regenerate activations using the canonical chunks before interpreting geometry.",
            "",
        ])
    path.write_text("\n".join(lines))


def run_experiment1_activation_monitor(
    *,
    run_name: str = "pilot24_gptoss20b",
    output_root: str = DEFAULT_OUTPUT_ROOT,
    drift_run_dir: str = DEFAULT_DRIFT_RUN_DIR,
    activation_rows_path: str | None = None,
    chunked_trajectories_path: str = DEFAULT_CHUNKED_TRAJECTORIES,
    sentences_path: str = DEFAULT_SENTENCES,
    required_layers: tuple[int, ...] = DEFAULT_LAYERS,
    regenerate_if_needed: bool = False,
    local_model_name_or_path: str | None = None,
    candidate_rows_path: str = DEFAULT_CANDIDATE_ROWS_PATH,
    trajectory_dir: str = DEFAULT_TRAJECTORY_DIR,
    slice_mode: str = "trajectory_candidates",
    trajectory_slice_type: str = "balanced_failure_modes_with_controls",
    model_name: str = "together_ai/openai/gpt-oss-20b",
    device: str = "cuda",
    device_map: str | None = "auto",
    torch_dtype: str = "auto",
    forward_chunk_size: int = 256,
    check_tensor_shapes: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build Experiment 1 outputs under outputs/experiment1_activation_monitor."""
    root = Path(output_root) / run_name
    _validate_output_root(Path(output_root))
    root.mkdir(parents=True, exist_ok=True)

    if not Path(sentences_path).exists():
        raise FileNotFoundError(f"Missing canonical sentences table: {sentences_path}")
    if not Path(chunked_trajectories_path).exists():
        raise FileNotFoundError(f"Missing canonical chunked trajectories table: {chunked_trajectories_path}")

    drift = Path(drift_run_dir)
    activation_index = Path(activation_rows_path) if activation_rows_path else drift / "step_activation_rows.csv"
    prefix_rows_path = drift / "prefix_action_rows.csv"
    trajectory_rows_path = drift / "trajectory_wrong_turn_summary.csv"
    if not prefix_rows_path.exists() or not trajectory_rows_path.exists():
        raise FileNotFoundError(
            f"Missing required drift outputs in {drift}: prefix_action_rows.csv and trajectory_wrong_turn_summary.csv"
        )

    compatibility = verify_activation_compatibility(
        activation_rows_path=str(activation_index),
        chunked_trajectories_path=chunked_trajectories_path,
        output_dir=str(root),
        required_layers=required_layers,
        check_tensor_shapes=check_tensor_shapes,
    )

    if compatibility["status"] != "compatible" and regenerate_if_needed:
        if not local_model_name_or_path:
            raise ValueError("regenerate_if_needed=True requires local_model_name_or_path")
        regenerated_dir = root / "regenerated_canonical_drift"
        if verbose:
            print(f"Regenerating canonical drift and activations into {regenerated_dir}")
        run_step_reasoning_drift_experiment(
            candidate_rows_path=candidate_rows_path,
            trajectory_dir=trajectory_dir,
            output_dir=str(regenerated_dir),
            model_name=model_name,
            slice_mode=slice_mode,
            trajectory_slice_type=trajectory_slice_type,
            segmentation_mode="paragraph_or_sentence",
            max_reasoning_steps=32,
            collect_activations=True,
            activation_prompt_mode="original_trace",
            local_model_name_or_path=local_model_name_or_path,
            layers=required_layers,
            device=device,
            device_map=device_map,
            torch_dtype=torch_dtype,
            forward_chunk_size=forward_chunk_size,
            resume=True,
            verbose=verbose,
        )
        drift = regenerated_dir
        activation_index = drift / "step_activation_rows.csv"
        prefix_rows_path = drift / "prefix_action_rows.csv"
        trajectory_rows_path = drift / "trajectory_wrong_turn_summary.csv"
        compatibility = verify_activation_compatibility(
            activation_rows_path=str(activation_index),
            chunked_trajectories_path=chunked_trajectories_path,
            output_dir=str(root),
            required_layers=required_layers,
            check_tensor_shapes=check_tensor_shapes,
        )

    prefix_rows = _read_csv(prefix_rows_path)
    trajectory_rows = _read_csv(trajectory_rows_path)
    events = _classify_events(prefix_rows)
    commitments = _build_commitment_summary(trajectory_rows, prefix_rows)
    _write_csv(root / "prefix_action_rows.csv", prefix_rows)
    _write_csv(root / "event_rows.csv", events)
    _write_csv(root / "trajectory_commitment_summary.csv", commitments)

    usable_rows = _read_csv(root / "usable_activation_rows.csv")
    geometry_rows: list[dict[str, Any]] = []
    if compatibility["status"] == "compatible":
        geometry_rows = _load_geometry_from_activations(usable_rows, trajectory_rows)
    event_geometry = _event_aligned_geometry(events, geometry_rows) if geometry_rows else []
    geometry_summary = _geometry_class_summary(geometry_rows, trajectory_rows) if geometry_rows else []
    _write_csv(root / "event_aligned_geometry_rows.csv", event_geometry)
    _write_csv(root / "geometry_class_summary.csv", geometry_summary)
    _plot_outputs(root, events, event_geometry, commitments)

    manifest = {
        "status": "completed",
        "run_name": run_name,
        "output_dir": str(root),
        "drift_run_dir": str(drift),
        "activation_rows_path": str(activation_index),
        "chunked_trajectories_path": chunked_trajectories_path,
        "sentences_path": sentences_path,
        "required_layers": list(required_layers),
        "activation_compatibility_status": compatibility.get("status"),
        "n_prefix_rows": len(prefix_rows),
        "n_trajectories": len(commitments),
        "n_events": len(events),
        "n_event_aligned_geometry_rows": len(event_geometry),
        "regenerate_if_needed": regenerate_if_needed,
    }
    _write_json(root / "run_manifest.json", manifest)
    _write_run_report(
        root / "run_report.md",
        compatibility=compatibility,
        events=events,
        event_geometry=event_geometry,
        commitments=commitments,
    )
    return manifest


__all__ = [
    "build_weisheng_chunk_averaged_activations",
    "run_experiment1_activation_monitor",
    "verify_activation_compatibility",
]
