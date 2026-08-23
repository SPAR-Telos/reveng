# Repository and artifact guide

This is the entry point for understanding the repository, finding raw outputs,
and moving work to another machine. The machine-readable companion is
[`docs/artifact_index.csv`](artifact_index.csv).

## Migration status (2026-08-23)

The repository and research artifacts have been split between GitHub and
Hugging Face. Rebuildable environments, credentials, download caches, model
weights, bytecode, and one redundant activation ZIP are intentionally excluded.

Verified remote state:

- GitHub `origin/feat/counterfactual-patching`: all committed files at the
  migration commit, including implementations, the locked environment,
  semantic raw outputs, API smoke-test raw JSONL, manifests, reports, plans,
  status files, and artifact pointers.
- Hugging Face
  [`project-telos/gpt_oss_20b_doorkey_boundary_activations`](https://huggingface.co/datasets/project-telos/gpt_oss_20b_doorkey_boundary_activations),
  revision `67dd0ef403f9dde32d36ef54f746f37226539b5b`: the complete
  1,276-shard, 5.51 GB activation dataset and its checksums/manifests.
- Hugging Face
  [`project-telos/doorkey-semantic-reasoning-labels`](https://huggingface.co/datasets/project-telos/doorkey-semantic-reasoning-labels),
  revision `5708370a409afdffec083eaf5d8747d02aa10672`: the sentence
  inventory, original and replicate v3 annotation runs, manifests/statuses,
  taxonomy, audit, and replicate diagnostics.
- Private Hugging Face
  `project-telos/reveng-experiment-artifacts`, revision
  `708b1ce4e77351582e364d4581b50485fba61568`: 38,002 files and about
  2.60 GB of remaining behavioral, counterfactual, activation-pilot,
  hypothesis-test, reader-facing, figure, log, and evaluation artifacts.

The complete mapping and restore command are in
[`REMOTE_ARTIFACTS.md`](REMOTE_ARTIFACTS.md). Credentials must still be saved
separately; never commit `.env`, `.hf_home`, or API tokens.

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
| Semantic judge, independent replicate | Same directory, `annotations_gpt_oss_20b_multilabel_v3_full_replicate.csv` | Stored in Git and Hugging Face. Pair on `annotation_id`. |
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
