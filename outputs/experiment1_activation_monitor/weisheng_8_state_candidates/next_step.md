# Next Step: Canonical Prefix Actions For Weisheng Activation States

Current status: activations are already mapped to canonical chunks for 8 GPT-OSS-20B trajectory states. The remaining step is behavioral only: rerun prefix-action elicitation on those same canonical chunks and include action entropy by repeated action sampling.

This requires `TOGETHERAI_API_KEY` or `TOGETHER_API_KEY`. No activation extraction is performed.

## 1. Run canonical prefix-action elicitation with action entropy

```bash
.venv/bin/python - <<'PY'
from reveng.experiments.step_reasoning_drift import run_step_reasoning_drift_experiment

run_step_reasoning_drift_experiment(
    candidate_rows_path='outputs/experiment1_activation_monitor/weisheng_8_state_candidates/candidate_rows.csv',
    output_dir='outputs/experiment1_activation_monitor/weisheng_8_state_canonical_prefix_actions',
    slice_mode='trajectory_candidates',
    trajectory_slice_type='all_rows',
    model_name='together_ai/openai/gpt-oss-20b',
    segmentation_mode='paragraph_or_sentence',
    max_reasoning_steps=32,
    collect_activations=False,
    action_mc_sample_repeats=10,
    action_mc_temperature=0.7,
    action_mc_max_workers=5,
    resume=True,
    verbose=True,
)
PY
```

For a cheaper smoke run, use a separate output directory and set `action_mc_sample_repeats=5`. Use a fresh or incomplete 10-sample output directory for reportable results, because completed-example checkpoints are not upgraded from 5 samples to 10 samples. The command is resumable within a fixed sample count.

## 2. Join canonical prefix rows to existing Weisheng activations

```bash
.venv/bin/python - <<'PY'
from reveng.experiments.experiment1_activation_monitor import run_experiment1_activation_monitor

run_experiment1_activation_monitor(
    run_name='weisheng_8_state_canonical_experiment1',
    drift_run_dir='outputs/experiment1_activation_monitor/weisheng_8_state_canonical_prefix_actions',
    activation_rows_path='outputs/experiment1_activation_monitor/weisheng_chunk_averaged_activations/chunk_activation_rows.csv',
    required_layers=(0,2,4,6,8,10,12,14,16,18,20,22),
    regenerate_if_needed=False,
    check_tensor_shapes=True,
    verbose=True,
)
PY
```

Expected final report:
`outputs/experiment1_activation_monitor/weisheng_8_state_canonical_experiment1/run_report.md`
