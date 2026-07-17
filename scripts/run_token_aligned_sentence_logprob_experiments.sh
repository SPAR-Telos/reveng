#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/reveng-matplotlib}"
set -a
source .env
set +a

MODE="${1:-gate}"
CANDIDATES="outputs/experiment1_activation_monitor/weisheng_8_state_candidates/candidate_rows.csv"
SENTENCES="data/behavioral_probes/doorkey_chunking_validation/sentences.csv"
MODEL="together_ai/openai/gpt-oss-20b"

run_pair() {
    local unit="$1"
    local exp1_dir="$2"
    local exp2_dir="$3"
    local max_examples="$4"
    local max_prefixes="$5"

    .venv/bin/python - "$unit" "$exp1_dir" "$max_examples" "$max_prefixes" <<'PY'
import sys
from reveng.experiments.step_reasoning_drift import run_step_reasoning_drift_experiment

unit, output_dir, max_examples, max_prefixes = sys.argv[1:]
run_step_reasoning_drift_experiment(
    candidate_rows_path="outputs/experiment1_activation_monitor/weisheng_8_state_candidates/candidate_rows.csv",
    output_dir=output_dir,
    slice_mode="trajectory_candidates",
    trajectory_slice_type="all_rows",
    model_name="together_ai/openai/gpt-oss-20b",
    analysis_unit=unit,
    sentence_boundaries_path="data/behavioral_probes/doorkey_chunking_validation/sentences.csv",
    max_reasoning_steps=None if unit == "sentence" else 32,
    collect_activations=False,
    action_mc_sample_repeats=0,
    action_logprob_temperature=0.7,
    action_top_logprobs=50,
    action_logprob_max_workers=8,
    max_examples=None if max_examples == "none" else int(max_examples),
    max_prefix_positions_per_example=None if max_prefixes == "none" else int(max_prefixes),
    resume=True,
    verbose=True,
)
PY

    .venv/bin/python scripts/build_experiment1_behavioral_report.py --run-dir "$exp1_dir"

    .venv/bin/python - "$exp1_dir" "$exp2_dir" <<'PY'
import sys
from reveng.experiments.reasoning_belief_action import run_reasoning_belief_action_observational

run_reasoning_belief_action_observational(
    drift_run_dir=sys.argv[1],
    candidate_rows_path="outputs/experiment1_activation_monitor/weisheng_8_state_candidates/candidate_rows.csv",
    output_dir=sys.argv[2],
    model_name="together_ai/openai/gpt-oss-20b",
    event_window=3,
    max_workers=16,
    categorical_logprob_temperature=0.7,
    categorical_top_logprobs=50,
    resume=True,
    verbose=True,
)
PY
}

if [[ "$MODE" == "gate" ]]; then
    EXP1="outputs/experiment1_activation_monitor/weisheng_sentence_logprob_gate_v3"
    EXP2="outputs/experiment2_behavioral_beliefs/weisheng_sentence_logprob_gate_v3"
    run_pair sentence "$EXP1" "$EXP2" 1 20
    .venv/bin/python scripts/validate_logprob_candidate_gate.py \
        --exp1-dir "$EXP1" --exp2-dir "$EXP2" --expected-prefixes 20
elif [[ "$MODE" == "full" ]]; then
    .venv/bin/python scripts/validate_logprob_candidate_gate.py \
        --exp1-dir outputs/experiment1_activation_monitor/weisheng_sentence_logprob_gate_v3 \
        --exp2-dir outputs/experiment2_behavioral_beliefs/weisheng_sentence_logprob_gate_v3 \
        --expected-prefixes 20
    run_pair sentence \
        outputs/experiment1_activation_monitor/weisheng_8_state_sentence_logprob_experiment1 \
        outputs/experiment2_behavioral_beliefs/weisheng_8_state_sentence_logprob_v1 \
        none none
    run_pair packed_chunk \
        outputs/experiment1_activation_monitor/weisheng_8_state_packed_logprob_experiment1 \
        outputs/experiment2_behavioral_beliefs/weisheng_8_state_packed_logprob_v1 \
        none none
    for exp2_dir in \
        outputs/experiment2_behavioral_beliefs/weisheng_8_state_sentence_logprob_v1 \
        outputs/experiment2_behavioral_beliefs/weisheng_8_state_packed_logprob_v1
    do
        .venv/bin/python scripts/build_factorization_descriptive.py --run-dir "$exp2_dir"
        .venv/bin/python - "$exp2_dir" <<'PY'
import sys
from reveng.experiments.belief_transition_indicators import build_belief_transition_indicator_analysis
from reveng.experiments.belief_transition_models import run_belief_transition_models

build_belief_transition_indicator_analysis(run_dir=sys.argv[1], event_window=3)
run_belief_transition_models(run_dir=sys.argv[1])
PY
        .venv/bin/python scripts/materialize_integrated_belief_uncertainty.py \
            --run-dir "$exp2_dir"
        .venv/bin/python scripts/build_state_belief_uncertainty_and_regression.py \
            --output-dir "$exp2_dir" \
            --model-name "$MODEL" \
            --temperature 0.7 \
            --top-logprobs 50 \
            --seed 0 \
            --max-workers 16 \
            --skip-queries
    done
    .venv/bin/python scripts/build_sentence_logprob_activation_analysis.py \
        --run-dir outputs/experiment1_activation_monitor/weisheng_8_state_sentence_logprob_experiment1
    .venv/bin/python scripts/validate_sentence_experiment_outputs.py \
        --exp1-dir outputs/experiment1_activation_monitor/weisheng_8_state_sentence_logprob_experiment1 \
        --exp2-dir outputs/experiment2_behavioral_beliefs/weisheng_8_state_sentence_logprob_v1 \
        --sentences-path "$SENTENCES"
    .venv/bin/python scripts/build_sentence_packed_comparison.py \
        --packed-exp1-dir outputs/experiment1_activation_monitor/weisheng_8_state_packed_logprob_experiment1 \
        --sentence-exp1-dir outputs/experiment1_activation_monitor/weisheng_8_state_sentence_logprob_experiment1 \
        --packed-exp2-dir outputs/experiment2_behavioral_beliefs/weisheng_8_state_packed_logprob_v1 \
        --sentence-exp2-dir outputs/experiment2_behavioral_beliefs/weisheng_8_state_sentence_logprob_v1 \
        --output-dir outputs/experiment1_activation_monitor/weisheng_8_state_sentence_vs_packed_logprob
else
    echo "Usage: $0 [gate|full]" >&2
    exit 2
fi
