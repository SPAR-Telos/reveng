# reveng

## Setup

```bash
source .venv/bin/activate
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
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

## Common Errors

- `Unrecognized options: True`
  - Use `--overwrite`, not `--overwrite True`.
- `No pair directories found under data/cf ...`
  - Generate pairs first with `generate_counterfactual_grid_pairs`.
- `Missing Together API key...`
  - Set `TOGETHERAI_API_KEY` or `TOGETHER_API_KEY`.
- `expected_k mismatch...`
  - Set `--expected-k` to the eval manifest row count.
