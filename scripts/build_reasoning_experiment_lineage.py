#!/usr/bin/env python3
"""Build a checksummed, file-level lineage catalog for recent reasoning experiments."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/reasoning_experiment_lineage.csv"
HF_REPO = "project-telos/reveng-experiment-artifacts"
GROUPS = (
    (
        "reasoning_topology_v1", "scripts/run_reasoning_topology.py",
        "data/behavioral_probes/reasoning_belief_action_cohorts/matched_46_candidates.csv; outputs/activation_collection/gpt_oss_20b_boundary_v1/",
        "outputs/hypothesis_tests/reasoning_topology_v1",
    ),
    (
        "reasoning_regime_coarse_graining_v1", "scripts/run_reasoning_regime_coarse_graining.py",
        "outputs/hypothesis_tests/action_distribution_change_points_v1/position_distribution_metrics.csv; outputs/hypothesis_tests/reasoning_topology_v1/trace_metrics.csv",
        "outputs/hypothesis_tests/reasoning_regime_coarse_graining_v1",
    ),
    (
        "semantic_regime_transition_prediction_v1", "scripts/run_semantic_regime_transition_prediction.py",
        "outputs/hypothesis_tests/reasoning_regime_coarse_graining_v1/transition_rows.csv; outputs/hypothesis_tests/semantic_reasoning_classification_v1/general_corpus_v1/annotations_gpt_oss_20b_multilabel_v3_full.csv; outputs/hypothesis_tests/semantic_reasoning_classification_v1/general_corpus_v1/annotations_gpt_oss_20b_multilabel_v3_full_replicate.csv",
        "outputs/hypothesis_tests/semantic_regime_transition_prediction_v1",
    ),
    (
        "reasoning_dynamics_graph_v1", "scripts/run_reasoning_dynamics_graph.py",
        "outputs/hypothesis_tests/reasoning_regime_coarse_graining_v1/regime_positions.csv; outputs/hypothesis_tests/action_distribution_cpd_beast_v1/detected_change_points.csv; canonical original/replicate semantic annotations",
        "outputs/hypothesis_tests/reasoning_dynamics_graph_v1",
    ),
    (
        "deterministic_reasoning_dag_audit_v1", "scripts/prepare_deterministic_reasoning_dag_audit.py; scripts/serve_reasoning_dag_review.py",
        "outputs/hypothesis_tests/reasoning_regime_coarse_graining_v1/regime_positions.csv; outputs/hypothesis_tests/action_distribution_cpd_beast_v1/detected_change_points.csv; canonical original semantic annotations",
        "outputs/hypothesis_tests/deterministic_reasoning_dag_audit_v1",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def role(path: Path) -> str:
    name = path.name
    if name == "run_manifest.json": return "provenance_manifest"
    if name.endswith("report.md") or name.endswith("GUIDE.md") or name.endswith("RUNBOOK.md"): return "report_or_guide"
    if name.endswith((".png", ".svg")): return "figure"
    if name in {"prediction_rows.csv", "closure_predictions.csv", "predictive_rows.csv", "transition_rows.csv", "regime_positions.csv", "beast_operator_events.csv"}: return "row_level_derived_output"
    if name == "edge_review.csv": return "mutable_human_annotation"
    if name.endswith(".csv"): return "derived_table"
    return "supporting_artifact"


def main() -> None:
    rows = []
    for experiment, producer, inputs, directory in GROUPS:
        for path in sorted((ROOT / directory).rglob("*")):
            if not path.is_file(): continue
            relative = path.relative_to(ROOT).as_posix()
            size = path.stat().st_size
            intended = (
                f"Hugging Face dataset {HF_REPO} at {relative}"
                if size > 50_000_000 else "GitHub origin/feat/counterfactual-patching"
            )
            rows.append({
                "experiment": experiment, "producer": producer,
                "primary_upstream_dependencies": inputs, "artifact_path": relative,
                "artifact_role": role(path), "bytes": size, "sha256": sha256(path),
                "intended_remote": intended,
            })
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
