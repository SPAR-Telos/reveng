#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/reveng-matplotlib}"

EXP1_DIR="outputs/experiment1_activation_monitor/weisheng_8_state_sentence_experiment1"
EXP2_DIR="outputs/experiment2_behavioral_beliefs/weisheng_8_state_sentence_v1"
CANDIDATES="outputs/experiment1_activation_monitor/weisheng_8_state_candidates/candidate_rows.csv"
SENTENCES="data/behavioral_probes/doorkey_chunking_validation/sentences.csv"

.venv/bin/python - <<PY
from reveng.experiments.step_reasoning_drift import run_step_reasoning_drift_experiment

run_step_reasoning_drift_experiment(
    candidate_rows_path="${CANDIDATES}",
    output_dir="${EXP1_DIR}",
    slice_mode="trajectory_candidates",
    trajectory_slice_type="all_rows",
    model_name="together_ai/openai/gpt-oss-20b",
    analysis_unit="sentence",
    sentence_boundaries_path="${SENTENCES}",
    max_reasoning_steps=None,
    collect_activations=False,
    action_mc_sample_repeats=10,
    action_mc_temperature=0.7,
    action_mc_max_workers=10,
    resume=True,
    verbose=True,
)
PY

.venv/bin/python scripts/build_experiment1_behavioral_report.py --run-dir "${EXP1_DIR}"

.venv/bin/python - <<PY
from reveng.experiments.reasoning_belief_action import run_reasoning_belief_action_observational

run_reasoning_belief_action_observational(
    drift_run_dir="${EXP1_DIR}",
    candidate_rows_path="${CANDIDATES}",
    output_dir="${EXP2_DIR}",
    model_name="together_ai/openai/gpt-oss-20b",
    event_window=3,
    max_workers=8,
    resume=True,
    verbose=True,
)
PY

.venv/bin/python scripts/build_factorization_descriptive.py --run-dir "${EXP2_DIR}"

.venv/bin/python - <<PY
from reveng.experiments.belief_transition_indicators import build_belief_transition_indicator_analysis
from reveng.experiments.belief_transition_models import run_belief_transition_models

build_belief_transition_indicator_analysis(run_dir="${EXP2_DIR}", event_window=3)
run_belief_transition_models(run_dir="${EXP2_DIR}")
PY

.venv/bin/python scripts/build_state_belief_uncertainty_and_regression.py \
    --output-dir "${EXP2_DIR}" \
    --model-name together_ai/openai/gpt-oss-20b \
    --temperature 0.7 \
    --top-logprobs 20 \
    --seed 0 \
    --max-workers 8

.venv/bin/python scripts/validate_sentence_experiment_outputs.py \
    --exp1-dir "${EXP1_DIR}" \
    --exp2-dir "${EXP2_DIR}" \
    --sentences-path "${SENTENCES}"

.venv/bin/python scripts/build_sentence_packed_comparison.py
