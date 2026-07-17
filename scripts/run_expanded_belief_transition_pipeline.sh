#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export MPLCONFIGDIR=/tmp/matplotlib

DRIFT_DIR="data/behavioral_probes/step_reasoning_drift_expanded_185_behavioral"
BELIEF_DIR="data/behavioral_probes/reasoning_belief_action_expanded_185_behavioral"
CANDIDATES="data/behavioral_probes/reasoning_belief_action_cohorts/expanded_185_candidates.csv"

.venv/bin/python - <<'PY'
import csv
import json
from pathlib import Path

candidates = Path("data/behavioral_probes/reasoning_belief_action_cohorts/expanded_185_candidates.csv")
expanded_ids = {row["example_id"] for row in csv.DictReader(candidates.open())}

drift_output = Path(
    "data/behavioral_probes/step_reasoning_drift_expanded_185_behavioral/example_checkpoints.jsonl"
)
drift_sources = [
    Path("data/behavioral_probes/step_reasoning_drift_balanced_v1/example_checkpoints.jsonl"),
    Path("data/behavioral_probes/step_reasoning_drift_matched_46_behavioral/example_checkpoints.jsonl"),
    drift_output,
]
checkpoints = {}
for source in drift_sources:
    if not source.exists():
        continue
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        example_id = payload.get("example_id")
        if example_id in expanded_ids:
            checkpoints[example_id] = line
drift_output.parent.mkdir(parents=True, exist_ok=True)
drift_output.write_text("\n".join(checkpoints.values()) + ("\n" if checkpoints else ""))
print(f"Seeded {len(checkpoints)} completed action-prefix states.")

belief_output = Path(
    "data/behavioral_probes/reasoning_belief_action_expanded_185_behavioral/belief_rows.jsonl"
)
belief_sources = [
    Path("data/behavioral_probes/reasoning_belief_action_balanced_v1/belief_rows.jsonl"),
    Path("data/behavioral_probes/reasoning_belief_action_matched_46_behavioral/belief_rows.jsonl"),
    belief_output,
]
belief_rows = {}
for source in belief_sources:
    if not source.exists():
        continue
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("example_id") not in expanded_ids or payload.get("query_error"):
            continue
        key = (
            payload.get("example_id"),
            payload.get("reasoning_step_idx"),
            payload.get("question_id"),
        )
        belief_rows[key] = line
belief_output.parent.mkdir(parents=True, exist_ok=True)
belief_output.write_text("\n".join(belief_rows.values()) + ("\n" if belief_rows else ""))
print(f"Seeded {len(belief_rows)} completed belief queries.")
PY

.venv/bin/reveng-cli run_step_reasoning_drift_experiment \
  --candidate-rows-path "$CANDIDATES" \
  --output-dir "$DRIFT_DIR" \
  --slice-mode trajectory_candidates \
  --trajectory-slice-type selection_candidates \
  --no-collect-activations

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
