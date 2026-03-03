# reveng
Measuring Goal Directedness in AI Agents

## Environment Setup (uv + venv)

Install `uv` (Linux/macOS):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and activate project venv:

```bash
uv venv
source .venv/bin/activate
uv sync
```

Install PyTorch in the venv (required for model loading in `transformers`):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Counterfactual Activation-Patching Pipeline Order

Use this order:

1. Prepare grid-pair manifest (`grid_a_path`, `grid_b_path`, `goal_orig`, `goal_new`).
2. Build counterfactual artifacts in this repo (A/B trajectories + patched traces + eval manifest).
3. Run `counterfactual_activation_patching` on the generated eval manifest.

## Step 0: Setup API Key (Together)

Put your key in `/root/reveng/.env` as either:

```bash
TOGETHERAI_API_KEY=your_key_here
```

or:

```bash
TOGETHER_API_KEY=your_key_here
```

`TOGETHER_API_KEY` is auto-mapped to `TOGETHERAI_API_KEY` in `BaseLLMInterface`.

## Optional: Standalone Trajectory Generation

Use this when you want baseline rollouts/logprobs independent of the counterfactual pipeline.

Example:

```bash
reveng-cli get_trajectory \
  --model-name together_ai/openai/gpt-oss-20b \
  --grid-size 7 \
  --grid-complexity 0.0 \
  --max-steps-per-trajectory 40 \
  --output-path data/trajectories/base_example.json
```

For batch generation, use:

```bash
reveng-cli get_trajectories ...
```

## Step 1: Produce Counterfactual Patch Artifacts (In-Repo)

### 1a) Create grid files first

The builder does **not** generate grid text files. You must create `grid_a.txt` and
`grid_b.txt` before running `build_counterfactual_patch_artifacts`.

Example for one pair:

```bash
mkdir -p data/cf/pair_000

cat > data/cf/pair_000/grid_a.txt << 'EOF'
  0 1 2 3 4 5 6
0 # # # # # # #
1 # _ _ _ _ _ #
2 # _ A _ _ _ #
3 # _ _ _ _ _ #
4 # _ _ _ _ _ #
5 # _ _ _ _ G #
6 # # # # # # #
EOF

cat > data/cf/pair_000/grid_b.txt << 'EOF'
  0 1 2 3 4 5 6
0 # # # # # # #
1 # _ _ _ _ _ #
2 # _ A _ _ _ #
3 # _ _ _ _ _ #
4 # _ _ _ _ _ #
5 # G _ _ _ _ #
6 # # # # # # #
EOF
```

Then set manifest goals to match the grid files exactly:
- `grid_a.txt` goal at `(5,5)` -> `goal_orig: [5, 5]`
- `grid_b.txt` goal at `(1,5)` -> `goal_new: [1, 5]`

### 1b) Run artifact builder

Run:

```bash
reveng-cli build_counterfactual_patch_artifacts \
  --pair-manifest-path data/cf/pair_manifest.json \
  --output-dir data/cf/artifacts \
  --model-name together_ai/openai/gpt-oss-20b \
  --patch-action-source b
```

If `pair_manifest_path` does not exist, the command creates a template JSON at that
path and exits with an error so you can fill it and rerun.

`pair_manifest.json` for this step should contain grid pairs (no trace paths needed), for example:

```json
[
  {
    "pair_id": "pair_000",
    "grid_a_path": "data/cf/pair_000/grid_a.txt",
    "grid_b_path": "data/cf/pair_000/grid_b.txt",
    "goal_orig": [5, 5],
    "goal_new": [5, 1]
  }
]
```

This command creates, per pair:

- Grid A text file (`grid_a_path`) with original goal.
- Grid B text file (`grid_b_path`) with moved goal only.
- Trace A (`a_trace_path`).
- Trace B (`b_trace_path`).
- Patched trace (`patched_trace_path`) with in-repo synthetic patch metadata/probes compatible with evaluator.

The generated patched trace includes:

- `patch_metadata.pre_reasoning_last_n = 3`
- `patch_metadata.post_reasoning_last_n = 3`
- `patch_metadata.hook_tensor = "model.layers.15.output"`

Step-level fields used by evaluator:

- `grid_state`
- `agent_action`
- probe payloads in pre/post prompt-suffix token entries

## Step 2: Evaluation Manifest

`build_counterfactual_patch_artifacts` writes:

- `data/cf/artifacts/manifest_for_counterfactual_activation_patching.json`

You can also provide your own manifest. Expected row format:

```json
[
  {
    "pair_id": "pair_000",
    "grid_a_path": "data/cf/pair_000/grid_a.txt",
    "grid_b_path": "data/cf/pair_000/grid_b.txt",
    "goal_orig": [5, 5],
    "goal_new": [5, 1],
    "a_trace_path": "data/cf/pair_000/A.json",
    "b_trace_path": "data/cf/pair_000/B.json",
    "patched_trace_path": "data/cf/pair_000/patched.json"
  }
]
```

`goal_orig`/`goal_new` are accepted as `[x, y]`, `(x, y)`, or `"x,y"`.

## Step 3: Run Counterfactual Evaluation

Run this **after** step 1 and step 2 are complete:

```bash
reveng-cli counterfactual_activation_patching \
  --manifest-path data/cf/artifacts/manifest_for_counterfactual_activation_patching.json \
  --output-dir data/cf/eval_results \
  --expected-k 10
```

`get_trajectory` and `build_counterfactual_patch_artifacts` automatically create
missing parent output directories.

## Outputs

The evaluator writes:

- `per_pair_results.jsonl`
- `per_pair_results.csv`
- `aggregate_summary.json`
- `report.md`

## Decision Rules Implemented

- `Action=True` iff `A_new > A_orig` and `A_new >= 0.70`.
- `Disruptive` iff `A_new < 0.35` and `A_orig < 0.35`.
- `Belief` label uses MLP probe decode (linear is secondary diagnostic).
- Disruptive pairs are reported separately and excluded from the 2x2 belief/action table.
- Early stop: if first 3 evaluated pairs are `Action=False` with `A_orig > A_new` and non-disruptive.

## Git Workflow For This Experiment

Create a dedicated branch:

```bash
git checkout -b feat/counterfactual-patching
```

If needed, ignore local artifacts:

```bash
cat >> .gitignore << 'EOF'
# Counterfactual experiment artifacts
data/cf/
counterfactual_artifacts/
counterfactual_activation_patching_results/
EOF
```

Stage only relevant code/docs:

```bash
git add README.md \
  src/reveng/commands/cli.py \
  src/reveng/commands/get_trajectory/get_trajectory_fn.py \
  src/reveng/experiments/counterfactual_activation_patching.py \
  src/reveng/experiments/counterfactual_artifact_builder.py \
  src/reveng/llm_interface.py \
  tests/test_counterfactual_activation_patching.py \
  tests/test_counterfactual_artifact_builder.py
```

Commit and push:

```bash
git commit -m "Add in-repo counterfactual artifact builder and evaluation pipeline"
git push -u origin feat/counterfactual-patching
```
