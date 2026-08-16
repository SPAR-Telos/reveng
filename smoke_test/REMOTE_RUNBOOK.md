# Maze smoke test: remote execution runbook

This runbook implements the compact smoke-test specification. The API and local
preflight are separate gates. Do not use API trajectories as activation data.

## Frozen design

- Regular no-key/no-door mazes.
- Sizes 7, 11, and 15.
- Generator difficulty settings 0.0, 0.6, and 1.0.
- Two fixed grids per size/difficulty cell, shared across every model/condition.
  They are the median-path and 90th-percentile-path grids selected from 32
  deterministically generated candidates, avoiding uninformative one-move cases.
- One temperature-0.7 trajectory on both grids and one temperature-0 trajectory
  on the first grid in each cell.
- Current full-grid context only; no trajectory history.
- 16,000 maximum output tokens per action call.
- Base seed 42. Each action call receives a deterministic seed derived from
  trajectory ID and action index; Together documents seed as best-effort rather
  than guaranteed deterministic across replicas.
- GPT-OSS-20B at low and medium reasoning effort; native/default reasoning for
  Gemma and Qwen.

The exact API IDs, pinned Hugging Face revisions, grid manifest, trajectory
schedule, prices, and decision thresholds are frozen in `config.lock.json`
when `--stage plan` runs.

## 1. API-machine setup

From the repository root:

```bash
uv venv .venv-maze-api --python 3.12
uv pip install --python .venv-maze-api/bin/python \
  -r configs/maze_api_requirements.txt
uv pip install --python .venv-maze-api/bin/python --no-deps -e .
export TOGETHER_API_KEY='...'
```

This API environment does not install Torch, Transformers, or model weights.

The two current serverless model IDs are:

- `openai/gpt-oss-20b`
- `google/gemma-4-31B-it`

The dense `Qwen/Qwen3-32B` checkpoint is not in the current Together serverless
catalog. It needs a dedicated endpoint. First ask Together which hardware is
available:

```bash
tg endpoints hardware --model Qwen/Qwen3-32B
```

If it is deployable, create the endpoint with one of the hardware IDs returned
by that command:

```bash
tg endpoints create \
  --model Qwen/Qwen3-32B \
  --hardware '<HARDWARE_ID>' \
  --display-name 'maze-smoke-qwen3-32b' \
  --wait
```

Copy the returned endpoint **Name**, not its `endpoint-...` management ID. If
Together says the base model is not deployable, follow its custom-model upload
flow or choose another approximately 30B Qwen checkpoint and change both the
API and local checkpoint IDs before freezing the plan.

Enter the endpoint's actual hourly price as `dedicated_hourly_price_usd` in a
private copy of `configs/maze_smoke_test.json`; otherwise Qwen cost remains
explicitly unavailable rather than being guessed.

## 2. Freeze and inspect the plan

The existing `smoke_test/` directory contains only a prepared empty plan, so it
can be safely replaced once with the account-specific Qwen endpoint name:

```bash
.venv-maze-api/bin/python scripts/run_maze_smoke_test.py \
  --stage plan \
  --config configs/maze_smoke_test.json \
  --output-dir smoke_test \
  --api-model-id 'Qwen3-32B=<TOGETHER_ENDPOINT_NAME>' \
  --replace-empty-plan
```

Review the printed trajectory and worst-case call bounds. Then verify that all
three inference names are visible to the account; this makes no generation
calls:

```bash
.venv-maze-api/bin/python scripts/run_maze_smoke_test.py \
  --stage check-api \
  --output-dir smoke_test
```

Do not use `--replace-empty-plan` after `raw_api_calls.jsonl` or
`local_preflight_records.jsonl` contains data. Use a new output directory for a
different design.

## 3. Run the paid API smoke test

Run one model at a time so endpoint or quota failures do not block the other
models. Every call is appended immediately to `raw_api_calls.jsonl`; rerunning
the same command resumes completed trajectories.

```bash
.venv-maze-api/bin/python scripts/run_maze_smoke_test.py \
  --stage query --output-dir smoke_test \
  --model GPT-OSS-20B --confirm-api-run

.venv-maze-api/bin/python scripts/run_maze_smoke_test.py \
  --stage query --output-dir smoke_test \
  --model Gemma-4-31B-IT --confirm-api-run

.venv-maze-api/bin/python scripts/run_maze_smoke_test.py \
  --stage query --output-dir smoke_test \
  --model Qwen3-32B --confirm-api-run
```

Stop the dedicated Qwen endpoint after its trajectories finish so hourly
billing stops. Then rebuild all tables and the report:

```bash
.venv-maze-api/bin/python scripts/run_maze_smoke_test.py \
  --stage analyze --output-dir smoke_test
```

Primary outputs:

- `smoke_test/api_calls.csv`: one row per API action call.
- `smoke_test/api_results.csv`: one row per trajectory.
- `smoke_test/condition_summary.csv`: requested model/size/difficulty/effort summaries.
- `smoke_test/api_summary.md`: compact decisions and resource estimates.

The raw JSONL contains provider response text, retry errors, and reasoning fields
for auditing. Do not put API keys in configuration files or outputs.

## 4. GPU-machine software check without model downloads

Gemma 4 requires Transformers 5.5 or newer. The repository's general-purpose
environment currently pins an older release, so use a dedicated environment on
the GPU machine:

```bash
uv venv .venv-maze --python 3.12
uv pip install --python .venv-maze/bin/python --upgrade \
  -r configs/maze_local_preflight_requirements.txt
uv pip install --python .venv-maze/bin/python --no-deps -e .
```

Install matching CUDA builds of PyTorch and torchvision before those two
commands, using the official PyTorch index for the remote driver. Installing
the repository with `--no-deps` is intentional: the main project pins
Transformers 4.57, while Gemma 4 requires Transformers 5.5 or newer. Before
downloading weights, verify CUDA and required model classes:

```bash
.venv-maze/bin/python scripts/run_maze_local_preflight.py \
  --config smoke_test/config.lock.json \
  --check-environment-only
```

This command does not call `from_pretrained` and does not download a model.

## 5. Minimal local activation preflight

Accept the Gemma Hugging Face license first and set `HF_TOKEN` if the checkpoint
is gated. Run one representative grid per exact local checkpoint:

```bash
.venv-maze/bin/python scripts/run_maze_local_preflight.py \
  --config smoke_test/config.lock.json \
  --output-dir smoke_test \
  --model GPT-OSS-20B \
  --reasoning-setting low \
  --max-new-tokens <GPT_OSS_RECOMMENDED_TOKEN_CAP> \
  --temperature 0.7 --top-p 0.95 --seed 42

.venv-maze/bin/python scripts/run_maze_local_preflight.py \
  --config smoke_test/config.lock.json \
  --output-dir smoke_test \
  --model Gemma-4-31B-IT \
  --reasoning-setting native \
  --max-new-tokens <GEMMA_RECOMMENDED_TOKEN_CAP> \
  --temperature 0.7 --top-p 0.95 --seed 42

.venv-maze/bin/python scripts/run_maze_local_preflight.py \
  --config smoke_test/config.lock.json \
  --output-dir smoke_test \
  --model Qwen3-32B \
  --reasoning-setting native \
  --max-new-tokens <QWEN_RECOMMENDED_TOKEN_CAP> \
  --temperature 0.7 --top-p 0.95 --seed 42
```

Replace each token-cap placeholder with the value reported in `api_summary.md`.
Use `--local-files-only` after the checkpoints are present to guarantee that a
rerun cannot access the network. The preflight saves only reasoning-token
decoder-block outputs for layers 8, 15, and 23 in safetensors files. It also
saves generated token IDs and reasoning-token indices and reports per-device
peak VRAM, tensor shapes, finite-value checks, generation time, bytes per
action, and a trajectory storage projection.

## Hardware and disk planning

- GPT-OSS-20B: one 24 GB GPU is a reasonable minimum for the checkpoint's
  native MXFP4 weights; use 48 GB if the selected token cap is large.
- Gemma-4-31B-it and Qwen3-32B: the pinned checkpoints are BF16 and are roughly
  63-65 GB of weights each. Use one 80 GB GPU at minimum; two 80 GB GPUs are the
  safer configuration for long generations and activation hooks.
- Host RAM: 128 GB recommended for the 31B and 32B preflights.
- Free disk: budget at least 200 GB if all three Hugging Face checkpoints remain
  cached together, plus headroom for package caches. Run models sequentially.

These are planning bounds, not measurements. `local_preflight.md` replaces
them with measured time, VRAM, and activation bytes.

## Expected smoke-test scale

- 18 fixed grids.
- 108 API trajectories: 54 GPT-OSS, 27 Gemma, and 27 Qwen.
- At most 2,480 action calls if every trajectory reaches its cap. This is a
  deliberately pessimistic bound; successful trajectories should terminate
  much earlier.
- Serverless output-only worst-case bound: about $3.97 for GPT-OSS and $4.96
  for Gemma if every call consumes all 16k output tokens. Input charges and the
  Qwen dedicated-endpoint runtime are additional. The plan command prints the
  current bound before any paid request.

Do not begin the full activation-bearing dataset run until both
`api_summary.md` and `local_preflight.md` pass their decision gates.

## Recovery rules

- Interrupted query: rerun the identical model command; completed trajectories
  are skipped and an incomplete trajectory resumes at its next action.
- Malformed JSONL: preserve the file and repair only the final partial line.
- Wrong model ID discovered before calls: rerun `plan` with
  `--replace-empty-plan`.
- Wrong model ID discovered after calls: use a new output directory; never mix
  configurations in one raw ledger.
- OOM or missing activation hooks: keep the failure record, fix the exact local
  inference configuration, and rerun only that model.

Together model names and prices can change. Immediately before the paid run,
compare `configs/maze_smoke_test.json` with the official serverless catalog and
record any justified update by creating a fresh locked output directory.
