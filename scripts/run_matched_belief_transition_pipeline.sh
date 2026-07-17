#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export MPLCONFIGDIR=/tmp/matplotlib

DRIFT_DIR="data/behavioral_probes/step_reasoning_drift_matched_46_behavioral"
BELIEF_DIR="data/behavioral_probes/reasoning_belief_action_matched_46_behavioral"
CANDIDATES="data/behavioral_probes/reasoning_belief_action_cohorts/matched_46_candidates.csv"

.venv/bin/reveng-cli run_step_reasoning_drift_experiment \
  --candidate-rows-path "$CANDIDATES" \
  --output-dir "$DRIFT_DIR" \
  --slice-mode trajectory_candidates \
  --trajectory-slice-type selection_candidates \
  --no-collect-activations

.venv/bin/python - <<'PY'
import csv
import json
from pathlib import Path

candidates = Path("data/behavioral_probes/reasoning_belief_action_cohorts/matched_46_candidates.csv")
output_dir = Path("data/behavioral_probes/reasoning_belief_action_matched_46_behavioral")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "belief_rows.jsonl"
sources = [
    Path("data/behavioral_probes/reasoning_belief_action_balanced_v1/belief_rows.jsonl"),
    output_path,
]
matched_ids = {row["example_id"] for row in csv.DictReader(candidates.open())}
rows = {}
for source in sources:
    if not source.exists():
        continue
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("example_id") not in matched_ids or payload.get("query_error"):
            continue
        key = (
            payload.get("example_id"),
            payload.get("reasoning_step_idx"),
            payload.get("question_id"),
        )
        rows[key] = line
output_path.write_text("\n".join(rows.values()) + ("\n" if rows else ""))
print(f"Retained or seeded {len(rows)} completed belief-query rows.")
PY

.venv/bin/reveng-cli run_reasoning_belief_action_observational \
  --drift-run-dir "$DRIFT_DIR" \
  --candidate-rows-path "$CANDIDATES" \
  --output-dir "$BELIEF_DIR"

.venv/bin/reveng-cli build_belief_transition_indicator_analysis \
  --run-dir "$BELIEF_DIR" \
  --event-window 3

.venv/bin/reveng-cli run_belief_transition_models \
  --run-dir "$BELIEF_DIR" \
  --device cuda \
  --epochs 300

date -u +"completed_at=%Y-%m-%dT%H:%M:%SZ" > "$BELIEF_DIR/pipeline_complete.txt"
