# Repository and artifact guide

This is the entry point for understanding the repository, finding raw outputs,
and moving work to another machine. The machine-readable companion is
[`docs/artifact_index.csv`](artifact_index.csv).

## Migration status (2026-08-23)

**Do not delete the current instance yet.** The checked-out branch matches its
GitHub remote at commit `0957af252d6586f59028f36daa060b0c00f6fadd`, but the
working tree contains uncommitted code, environment changes, and semantic-label
artifacts. In particular, the full replicate annotation CSV is only local.

A repository-wide audit also found 92,840 Git-ignored paths occupying about
16 GB, versus about 210 MB of tracked files under the principal code/data/output
directories. These include behavioral-probe records and checkpoints,
counterfactual tensors, intermediate activation products, derived analysis
tables, and reports. Apart from the explicitly verified Hugging Face datasets
below, their presence in another remote store has **not** been established.
Consequently, neither a clean Git push nor the two known Hugging Face datasets
alone constitute a complete backup of every experiment on this instance.

Already remote:

- GitHub `origin/feat/counterfactual-patching`: all committed files at the
  commit above, including the API smoke-test raw JSONL.
- Hugging Face
  [`project-telos/gpt_oss_20b_doorkey_boundary_activations`](https://huggingface.co/datasets/project-telos/gpt_oss_20b_doorkey_boundary_activations),
  revision `67dd0ef403f9dde32d36ef54f746f37226539b5b`: the complete
  1,276-shard, 5.51 GB activation dataset and its checksums/manifests.
- Hugging Face
  [`project-telos/doorkey-semantic-reasoning-labels`](https://huggingface.co/datasets/project-telos/doorkey-semantic-reasoning-labels),
  revision `d815b2bc278c9f073718ba8b2c5a7c2bc1652506`: the sentence
  inventory, original v3 annotation run, taxonomy, audit, and replicate
  diagnostics. **It does not contain the replicate annotation CSV.**

The migration uses the GitHub/Hugging Face split documented in
[`REMOTE_ARTIFACTS.md`](REMOTE_ARTIFACTS.md). Once its uploads are verified,
the consolidated private dataset replaces the persistent-volume requirement
for remaining generated experiment bundles.

Before deletion, preserve and push or copy all current uncommitted files shown
by `git status --short`. The essential semantic additions are approximately
18 MB and include:

- `scripts/run_general_semantic_together_multilabel_labeler.py`
- `scripts/run_local_general_semantic_multilabel_labeler.py`
- `tests/test_general_semantic_labeling.py`
- `pyproject.toml` and `uv.lock`
- `outputs/hypothesis_tests/semantic_reasoning_classification_v1/general_corpus_v1/`
  files containing `multilabel`, `V3_`, `MULTILABEL_TAXONOMY.md`, and the
  updated `RUNBOOK.md`
- this guide and `docs/artifact_index.csv`

The final deletion gate is:

1. `git status --short` contains no wanted local-only files.
2. The intended branch and commit are visible from a fresh clone.
3. The replicate CSV and its manifest/status have either been committed to an
   approved private location or uploaded to an approved dataset repository.
4. Download one remote activation shard and verify it against `SHA256SUMS`.
5. Save credentials separately; never copy `.env`, `.hf_home`, or API tokens
   into Git or an artifact archive.

### Commands to preserve the current semantic work in Git

These commands deliberately omit `.vscode/settings.json` and credential/cache
directories. Review the staged list before committing:

```bash
git add -- \
  README.md \
  docs/REPOSITORY_AND_ARTIFACT_GUIDE.md \
  docs/artifact_index.csv \
  pyproject.toml \
  uv.lock \
  scripts/run_general_semantic_together_multilabel_labeler.py \
  scripts/run_local_general_semantic_multilabel_labeler.py \
  tests/test_general_semantic_labeling.py \
  outputs/hypothesis_tests/semantic_reasoning_classification_v1/general_corpus_v1

git diff --cached --check
git diff --cached --stat
UV_CACHE_DIR=.uv-cache uv run pytest -q tests/test_general_semantic_labeling.py
git commit -m "Add reproducible v3 semantic labels and migration guide"
git push origin HEAD:feat/counterfactual-patching
```

This stages the entire `general_corpus_v1` directory so the full and replicate
CSVs, manifests, statuses, taxonomy, reports, and few-shot examples travel
together. No file in that directory currently exceeds normal GitHub's 100 MB
single-file limit.

### Separate backup for all ignored experiment artifacts

Git deliberately excludes large generated artifacts. Attach a persistent
volume that survives instance deletion, replace `/mnt/persistent/reveng-backup`
with its real mount path, and copy the repository while excluding only
rebuildable environments/caches and credentials:

```bash
mkdir -p /mnt/persistent/reveng-backup
rsync -a --info=progress2 \
  --exclude='/.git/' \
  --exclude='/.venv/' \
  --exclude='/.uv-cache/' \
  --exclude='/.pytest_cache/' \
  --exclude='/.hf_home/' \
  --exclude='/.env' \
  /root/reveng/ /mnt/persistent/reveng-backup/
```

Verify byte-level equality; a successful final dry run prints no file changes:

```bash
rsync -a --dry-run --checksum \
  --exclude='/.git/' \
  --exclude='/.venv/' \
  --exclude='/.uv-cache/' \
  --exclude='/.pytest_cache/' \
  --exclude='/.hf_home/' \
  --exclude='/.env' \
  /root/reveng/ /mnt/persistent/reveng-backup/
```

Do not use an instance-local directory as the destination. Confirm in the
cloud-provider console that the destination volume/object store is persistent
and independently attached before deleting the instance.

## Repository map

| Path | Purpose |
|---|---|
| `src/reveng/` | Reusable experiment, environment, scoring, and CLI code. |
| `scripts/` | Thin run/build/analyse entry points. Start here when reproducing a pipeline. |
| `configs/` | Human-authored experiment configuration and dedicated environment requirements. |
| `data/` | Source collections and experiment-specific raw API records. Some contents are large. |
| `outputs/` | Derived activations, analyses, reports, plots, and semantic annotations. |
| `smoke_test/` | Frozen multi-model API smoke-test design, raw responses, tables, and runbook. |
| `tests/` | Unit and integration tests for pipeline invariants. |
| `docs/` | Experiment specifications, interpretations, and this index. |
| `hf_datasets/` | Local staging packages for Hugging Face datasets; not authoritative by itself. |

## Where the raw outputs are

`smoke_test/api_calls.csv` is a convenient table, but it is not the most raw
record. Use these when provenance matters:

| Data | Raw or closest-to-raw location | Notes |
|---|---|---|
| Multi-model maze smoke test | `smoke_test/raw_api_calls.jsonl` | Provider response text, reasoning fields, retry/error records, tokens, and per-action provenance. `api_calls.csv` and summaries are derived from it. |
| GPT-OSS 120-trajectory pilot | `data/gpt_oss_maze_cot_120_pilot/raw_api_calls.jsonl` | Raw API records. `calls.csv`, `trajectories.csv`, `summary.csv`, and `report.md` are derived views. |
| Semantic judge, original full run | `outputs/hypothesis_tests/semantic_reasoning_classification_v1/general_corpus_v1/annotations_gpt_oss_20b_multilabel_v3_full.csv` | Contains target/context, parsed labels, rationale, `raw_response`, errors, and request provenance. |
| Semantic judge, independent replicate | Same directory, `annotations_gpt_oss_20b_multilabel_v3_full_replicate.csv` | Local-only at the date above. Pair on `annotation_id`. |
| Semantic source sentences | Same directory, `sentence_inventory.csv` | Prepared from the paths and hashes in `preparation_manifest.json`. |
| GPT-OSS activations | `outputs/activation_collection/gpt_oss_20b_boundary_v1/shards/*.safetensors` | Derived raw tensors. Join through `activation_index.parquet`; verify through `SHA256SUMS`. Complete remote copy exists. |

Never treat reports, plots, condition summaries, or diagnostic percentages as
raw records.

## Core pipelines and lineage

### Maze API collection and local replay

1. `scripts/run_maze_smoke_test.py --stage plan` freezes grids, schedule,
   target-model IDs/revisions, generation settings, seed, and price assumptions
   in `smoke_test/config.lock.json`, `grids.jsonl`, and `api_schedule.jsonl`.
2. The same script with `--stage query` appends provider records to
   `smoke_test/raw_api_calls.jsonl`. It is resumable.
3. `--stage analyze` builds `api_calls.csv`, `api_results.csv`,
   `condition_summary.csv`, and `api_summary.md`.
4. `src/reveng/experiments/maze_local_replay.py` teacher-forces saved
   completions through the exact pinned checkpoint and writes activation
   shards plus replay manifests. It does not generate replacement reasoning.

Detailed commands and hardware rules are in `smoke_test/REMOTE_RUNBOOK.md`.

### Semantic classification

1. `scripts/prepare_general_semantic_labeling.py` reads the canonical sentence
   and prior analysis tables listed in `preparation_manifest.json`, then writes
   `sentence_inventory.csv`, calibration files, and preparation metadata.
2. `scripts/run_general_semantic_together_multilabel_labeler.py` is the
   canonical Together AI v3 judge. It reads `sentence_inventory.csv` and
   `few_shot_examples_multilabel.json`, then atomically checkpoints an
   annotation CSV and writes adjacent `.manifest.json` and `.status.json`.
3. `scripts/run_local_general_semantic_multilabel_labeler.py` applies the same
   prompt/schema locally for smoke tests; it is not the producer of the two
   canonical full runs.
4. The canonical outputs are `annotations_gpt_oss_20b_multilabel_v3_full.csv`
   and `_replicate.csv`. `V3_FULL_REPLICATE_DIAGNOSTICS.md` compares their
   7,036 jointly valid IDs.

The taxonomy is `MULTILABEL_TAXONOMY.md`. The exact run commands are in the
adjacent `RUNBOOK.md`.

### GPT-OSS activation extraction

1. `scripts/run_gpt_oss_activation_full.py` calls
   `src/reveng/experiments/gpt_oss_activation_full.py`.
2. Inputs are the stored trajectory corpus and sentence boundaries named and
   hashed in `run_config.json`.
3. Outputs are one safetensors shard per environment state,
   `activation_index.parquet`, timing rows, manifests, and validation reports.
4. Downstream scripts such as `build_information_use_test.py`,
   `build_practical_action_event_monitor.py`, and
   `build_hypothesis_test_plan_outputs.py` consume the activation index rather
   than scanning shard names directly.

## Keeping judge and target-model principles consistent

Do not infer compatibility from filenames. Check manifests before combining
runs.

- Semantic run manifests freeze judge model, prompt version and hash, output
  schema hash, few-shot hash, input hash, reasoning effort, temperature, seed,
  and retry/token settings. The original and replicate v3 manifests match on
  these fields.
- `smoke_test/config.lock.json` separates target models and freezes each API
  model ID, local checkpoint revision, reasoning control, trace format, and
  decoding settings.
- Activation `run_config.json` freezes target model revision, tokenizer-linked
  input hashes, layers, representation definitions, dtype, and seed.
- A serverless seed is best-effort reproducibility, not bitwise determinism.
  Independent judge runs should therefore be analysed as repeatability, not
  duplicates or human accuracy.
- A new judge, prompt/taxonomy hash, target revision, chat template, layer set,
  or input hash requires a new output filename/directory. Never resume into an
  output whose manifest differs; the current scripts reject this where
  implemented.

## Environment reproduction

The project uses Python 3.12, `pyproject.toml`, and `uv.lock`. The lock is
present, resolves 211 packages, and `uv sync --locked --dry-run` reports no
changes as of 2026-08-23.

On a fresh machine:

```bash
uv sync --locked
uv run pytest tests/test_general_semantic_labeling.py
```

Do not copy `.venv`; rebuild it. The current dependency change to
`kernels==0.12.0` and its matching `uv.lock` change are uncommitted and must be
preserved together. CUDA builds of PyTorch are hardware/driver-specific; for
the Gemma/Qwen local preflight use the dedicated requirements and setup in
`smoke_test/REMOTE_RUNBOOK.md` rather than silently changing the main lock.

## GPU, RAM, and disk estimate

Choose by the largest local target model, not by the semantic API judge:

| Work | GPU VRAM | Host RAM | Disk |
|---|---:|---:|---:|
| Together semantic judging / API analysis only | None | 16–32 GB | 40–60 GB |
| GPT-OSS-20B local replay/activations | 48 GB recommended; 24 GB is a measured but tight minimum | 64 GB | 150 GB recommended |
| One BF16 Gemma-4-31B-IT or Qwen3-32B with activation hooks | 80 GB minimum; 2 × 80 GB safer for long traces | 128 GB | 200 GB |
| Keep GPT-OSS, Gemma, and Qwen checkpoints plus outputs together | 80 GB GPU (2 × 80 GB safer) | 128 GB | 300 GB recommended |

The GPT-OSS full extraction actually peaked at 23.36 GB CUDA reserved and
16.26 GB allocated. Its checkpoint cache is about 13 GB and its complete
activation output is 5.51 GB. Gemma/Qwen BF16 weights are approximately
63–65 GB each, so 24/48 GB cards cannot hold those dense checkpoints without
quantisation or multi-GPU sharding. For a generally stronger replacement,
use one A100/H100 80 GB, 128 GB RAM, and a 300 GB persistent disk.
