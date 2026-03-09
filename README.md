# reveng

## Setup

```bash
source .venv/bin/activate
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Counterfactual Pipeline

Scope here is only:

`grid pairs -> pair manifest -> artifacts (A/B/patched) -> eval manifest -> evaluation outputs`

Constraints are documented in [`counterfactual_constraints.md`](counterfactual_constraints.md).

### 0) Generate grid pairs

```bash
reveng-cli generate_counterfactual_grid_pairs \
  --output-root data/cf \
  --num-pairs 10 \
  --grid-size 7 \
  --grid-complexity 0.4 \
  --seed 42 \
  --overwrite
```

This creates:

- `data/cf/pair_000/grid_a.txt` + `grid_b.txt`
- `data/cf/pair_001/grid_a.txt` + `grid_b.txt`
- ...

Manual grids are still supported, but no longer the default path.

### 1) Generate pair manifest

```bash
reveng-cli generate_counterfactual_pair_manifest \
  --grids-root data/cf \
  --output-path data/cf/pair_manifest.json \
  --overwrite
```

The generator scans recursively for `pair_*` directories containing `grid_a.txt` and `grid_b.txt`.

Important: `--overwrite` is a flag (`--overwrite` / `--no-overwrite`). Do not pass `True`.

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

This writes:

- `data/cf/artifacts/<pair_id>/A.json`
- `data/cf/artifacts/<pair_id>/B.json`
- `data/cf/artifacts/<pair_id>/patched.json`
- `data/cf/artifacts/manifest_for_counterfactual_activation_patching.json`

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
  --expected-k 10
```

Outputs:

- `data/cf/eval_results/per_pair_results.jsonl`
- `data/cf/eval_results/per_pair_results.csv`
- `data/cf/eval_results/aggregate_summary.json`
- `data/cf/eval_results/report.md`

## Common Errors

- `Unrecognized options: True`
  - Use `--overwrite`, not `--overwrite True`.
- `No pair directories found under data/cf ...`
  - Create `data/cf/pair_*/grid_a.txt` and `grid_b.txt` first.
- `Missing Together API key...`
  - Set `TOGETHERAI_API_KEY` or `TOGETHER_API_KEY`.
- `expected_k mismatch...`
  - Set `--expected-k` to the eval manifest row count.
