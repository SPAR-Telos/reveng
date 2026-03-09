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

Report highlights:

- invalid summary and disruptive summary are separate
- action summary by category
- separate MLP and linear belief-action 2x2 tables
- probe agreement/disagreement section

## Common Errors

- `Unrecognized options: True`
  - Use `--overwrite`, not `--overwrite True`.
- `No pair directories found under data/cf ...`
  - Generate pairs first with `generate_counterfactual_grid_pairs`.
- `Missing Together API key...`
  - Set `TOGETHERAI_API_KEY` or `TOGETHER_API_KEY`.
- `expected_k mismatch...`
  - Set `--expected-k` to the eval manifest row count.
