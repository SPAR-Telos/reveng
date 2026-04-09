# reveng

## Setup

```bash
source .venv/bin/activate
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Behavioral Probes

Behavioral probes are black-box, single-turn DoorKey queries over rendered grid
states. The current pipeline supports:

- `label3` probes with constrained `A/B/C -> yes/no/unknown`
- `coord_json` probes with exact `{"row": <int>, "col": <int>}` outputs
- readouts:
  - greedy visible answer
  - logprob diagnostics at `T=0.0`, `T=0.7`, `T=1.0`
  - Monte Carlo sampling at `T=0.7`

Important logprob semantics:

- the main logprob diagnostic now explicitly records whether `p(yes) > p(no)`
  rather than relying only on the semantic argmax label
- outputs therefore include both:
  - semantic distributions over `yes/no/unknown`
  - pairwise `yes_gt_no_*` checks for direct comparison with greedy and MC

### Smoke test

Run the expanded smoke test on the manual 9x9 DoorKey states:

```bash
reveng-cli run_behavioral_probe_smoke_test \
  --model-name together_ai/openai/gpt-oss-20b \
  --output-dir data/behavioral_probes/smoke_test \
  --prompt-preset default_observable_state \
  --question-family all \
  --logprob-temperatures 0.0 0.7 1.0
```

Useful smaller slices:

```bash
reveng-cli run_behavioral_probe_smoke_test \
  --model-name together_ai/openai/gpt-oss-20b \
  --output-dir data/behavioral_probes/smoke_test_label3 \
  --prompt-preset cardinal_action_explicit \
  --question-family all_label3 \
  --answer-space label3 \
  --logprob-temperatures 0.0 0.7 1.0
```

```bash
reveng-cli run_behavioral_probe_smoke_test \
  --model-name together_ai/openai/gpt-oss-20b \
  --output-dir data/behavioral_probes/smoke_test_coords \
  --prompt-preset coordinate_output_explicit \
  --question-family coordinates \
  --answer-space coord_json
```

Key outputs:

- `behavioral_probe_rows.csv`
- `behavioral_probe_summary.csv`
- `behavioral_probe_probability_diagnostics.csv`
- `behavioral_probe_raw.json`
- `usage_summary.json`
- `checkpoints/checkpoint_rows.jsonl`
- `checkpoints/checkpoint_raw_rows.jsonl`
- `checkpoints/checkpoint_status.json`
- `figs/behavioral_probe_summary.png`
- `figs/behavioral_probe_directional_heatmap.png`
- `figs/behavioral_probe_coordinate_summary.png`

`usage_summary.json` aggregates Together/LiteLLM request counts, prompt tokens,
completion tokens, total tokens, reasoning tokens (when exposed), and estimated
USD cost by phase (`observed_action`, `logprob_t0`, `logprob_t07`,
`logprob_t1`, `mc`, and coordinate variants when relevant).

Checkpoint files are written incrementally during long runs so a late write
failure does not lose completed rows. `checkpoint_status.json` gives the latest
completed row count plus the current aggregated usage summary.

### Prompt ablation

```bash
reveng-cli run_behavioral_probe_prompt_ablation \
  --model-name together_ai/openai/gpt-oss-20b \
  --output-dir data/behavioral_probes/prompt_ablation \
  --question-family all_label3 \
  --answer-space label3
```

### Merge split behavioral outputs

Useful when a mixed `all` run was split into separate `label3` and `coord_json`
passes, or when you need to repair `behavioral_probe_rows.csv` from the raw
payloads:

```bash
reveng-cli merge_behavioral_probe_outputs \
  --input-dirs data/behavioral_probes/smoke_test data/behavioral_probes/smoke_test_coords \
  --output-dir data/behavioral_probes/smoke_test_all \
  --repair-source-rows-csv True
```

### Door semantics ablation

```bash
reveng-cli run_behavioral_probe_door_semantics_ablation \
  --model-name together_ai/openai/gpt-oss-20b \
  --output-dir data/behavioral_probes/door_semantics_ablation
```

### Mine trajectory-derived probe instances

This expects trace-viewer JSONs produced by `get_trajectory_key_door_env`.

```bash
reveng-cli run_behavioral_probe_trajectory_eval \
  --trajectory-dir path/to/key_door_trajectories \
  --output-dir data/behavioral_probes/trajectory_instances \
  --slice-type all_steps
```

Slice options:

- `all_steps`
- `wall_hit`
- `non_optimal_action`

### Failure-case case studies

```bash
reveng-cli run_behavioral_probe_case_studies \
  --trajectory-dir path/to/key_door_trajectories \
  --model-name together_ai/openai/gpt-oss-20b \
  --output-dir data/behavioral_probes/case_studies \
  --slice-type non_optimal_action
```

## Counterfactual Pipeline

Scope:

`grid pairs -> pair manifest -> artifacts (A/B/patched) -> eval manifest -> evaluation outputs`

Constraints are documented in [`counterfactual_constraints.md`](counterfactual_constraints.md).

### 0) Generate grid pairs

Default backward-compatible mode (`goal_move` only):

```bash
reveng-cli generate_counterfactual_grid_pairs \
  --output-root data/cf \
  --num-pairs 10 \
  --grid-size 7 \
  --grid-complexity 0.4 \
  --seed 42 \
  --overwrite
```

Multi-category mode (10 per category by default):

```bash
reveng-cli generate_counterfactual_grid_pairs \
  --output-root data/cf \
  --categories goal_move start_goal_swap rotate_90 reflect_vertical transpose \
  --num-pairs-per-category 10 \
  --grid-size 7 \
  --grid-complexity 0.4 \
  --seed 42 \
  --overwrite
```

### 1) Generate pair manifest

```bash
reveng-cli generate_counterfactual_pair_manifest \
  --grids-root data/cf \
  --output-path data/cf/pair_manifest.json \
  --overwrite
```

Manifest rows now include:

- `pair_id`
- `category`
- `grid_a_path`, `grid_b_path`
- `goal_a`, `goal_b`

Legacy `goal_orig` / `goal_new` are still accepted.

### 2) Preflight pair manifest

```bash
reveng-cli validate_counterfactual_preflight \
  --pair-manifest-path data/cf/pair_manifest.json \
  --artifacts-output-dir data/cf/artifacts \
  --eval-output-dir data/cf/eval_results
```

If trajectory generation is enabled, set one of:

- `TOGETHERAI_API_KEY`
- `TOGETHER_API_KEY`

### 3) Build artifacts

```bash
reveng-cli build_counterfactual_patch_artifacts \
  --pair-manifest-path data/cf/pair_manifest.json \
  --output-dir data/cf/artifacts \
  --model-name together_ai/openai/gpt-oss-20b \
  --patch-action-source b
```

### 4) Optional: regenerate eval manifest from artifacts

```bash
reveng-cli generate_counterfactual_eval_manifest \
  --pair-manifest-path data/cf/pair_manifest.json \
  --artifacts-dir data/cf/artifacts \
  --output-path data/cf/artifacts/manifest_for_counterfactual_activation_patching.json \
  --overwrite
```

### 5) Preflight eval manifest and run evaluation

```bash
reveng-cli validate_counterfactual_preflight \
  --pair-manifest-path data/cf/pair_manifest.json \
  --eval-manifest-path data/cf/artifacts/manifest_for_counterfactual_activation_patching.json
```

Then run evaluator using manifest row count as `expected_k`:

```bash
reveng-cli counterfactual_activation_patching \
  --manifest-path data/cf/artifacts/manifest_for_counterfactual_activation_patching.json \
  --output-dir data/cf/eval_results \
  --expected-k 50
```

Outputs:

- `data/cf/eval_results/per_pair_results.jsonl`
- `data/cf/eval_results/per_pair_results.csv`
- `data/cf/eval_results/aggregate_summary.json`
- `data/cf/eval_results/report.md`

Metric semantics:

- The current evaluator is the **surrogate trace-substitution baseline**. It is useful as a baseline, but it is not live activation patching.
- `A_base` / `A_target` score the same patched/intervened trace against the true optimal policies on grid A / grid B, respectively.
- These are scored on the actual grids A and B, not on the decoded cognitive map.
- The decoded cognitive map is used for belief readout only.
- Paper comparison: `Acc. GT` in Table 2 is original unpatched behavior scored against the ground-truth grid, so it is not directly equivalent to `A_base`.
- Implementation note: the current repo does not rerun the model after patching layer 15; it takes the saved trace from A and replaces the selected last-3 PRE and last-3 POST layer-15 entries with those from B.

Report highlights:

- invalid summary and disruptive summary are separate
- action summary by category
- separate MLP and linear belief-action 2x2 tables
- probe agreement/disagreement section

### 6) Expanded analysis: layer sweep + threshold sensitivity

Runs a balanced sweep (coarse layers around L15 + local refine), evaluates
action-threshold sensitivity (`0.65`, `0.70`, `0.75`), and writes a consolidated
dashboard for direct review.

```bash
reveng-cli run_counterfactual_expansion \
  --manifest-path data/cf/artifacts/manifest_for_counterfactual_activation_patching.json \
  --output-root data/cf/eval_results/expansion
```

If you have per-layer manifests/artifacts, pass a template path:

```bash
reveng-cli run_counterfactual_expansion \
  --manifest-path data/cf/layer_{layer}/manifest_for_counterfactual_activation_patching.json \
  --output-root data/cf/eval_results/expansion
```

When `--manifest-path` contains `{layer}` and a layer manifest is missing, the
command now auto-generates per-layer artifacts/manifests by default using:

- `--pair-manifest-path` (default: `data/cf/pair_manifest.json`)
- `--base-artifacts-dir` (default: `data/cf/artifacts`, must contain `A.json`/`B.json` per pair)

You can override patch construction settings with:

- `--patch-action-source` (default: `b`)
- `--linear-target` (default: `a`)
- `--synthetic-goal-prob` (default: `0.99`)

Key outputs:

- `data/cf/eval_results/expansion/summary.md` (single dashboard)
- `data/cf/eval_results/expansion/layer_threshold_metrics.csv`
- `data/cf/eval_results/expansion/per_config_aggregate.jsonl`
- `data/cf/eval_results/expansion/figs/layer_threshold_action_true_rate.png`
- `data/cf/eval_results/expansion/figs/layer_threshold_disruptive_rate.png`
- `data/cf/eval_results/expansion/selected_best/` (best config copied outputs)

### 7) Diagnose why the surrogate signal is weak

```bash
reveng-cli diagnose_counterfactual_signal \
  --eval-dir data/cf/eval_results \
  --expansion-dir data/cf/eval_results/expansion \
  --eval-manifest-path data/cf/artifacts/manifest_for_counterfactual_activation_patching.json \
  --output-dir data/cf/eval_results/signal_diagnostics
```

Key outputs:

- `data/cf/eval_results/signal_diagnostics/summary.json`
- `data/cf/eval_results/signal_diagnostics/summary.md`

### 8) Run true live hidden-state patching on a local model

This is a separate path from the surrogate baseline above. It requires a local
hookable Hugging Face causal LM with `torch` installed; it does **not** use the
Together API.

```bash
reveng-cli run_live_patch_curve \
  --model-name-or-path meta-llama/Llama-3.2-1B-Instruct \
  --pair-id pair_goal_move_009 \
  --step-index 0 \
  --output-dir data/cf/live_patch_curve
```

Defaults:

- patch window: final 3 reasoning tokens immediately before the final action JSON
- layer sweep: all layers unless `--layer-end` is set
- tracked actions: `UP`, `DOWN`, `LEFT`, `RIGHT`

Key outputs:

- `data/cf/live_patch_curve/probability_by_layer.csv`
- `data/cf/live_patch_curve/probability_by_layer.png`
- `data/cf/live_patch_curve/probability_delta_by_layer.png`
- `data/cf/live_patch_curve/metadata.json`

## Common Errors

- `Unrecognized options: True`
  - Use `--overwrite`, not `--overwrite True`.
- `No pair directories found under data/cf ...`
  - Generate pairs first with `generate_counterfactual_grid_pairs`.
- `Missing Together API key...`
  - Set `TOGETHERAI_API_KEY` or `TOGETHER_API_KEY`.
- `expected_k mismatch...`
  - Set `--expected-k` to the eval manifest row count.
