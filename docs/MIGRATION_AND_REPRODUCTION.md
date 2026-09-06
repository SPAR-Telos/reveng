# Migration and reproduction handoff

Status date: 2026-09-06

## Start here

Use this page for machine migration. Use
[`REPOSITORY_AND_ARTIFACT_GUIDE.md`](REPOSITORY_AND_ARTIFACT_GUIDE.md) for the
repository layout and experimental conventions.

The machine-readable indexes are:

| File | Use |
|---|---|
| [`artifact_index.csv`](artifact_index.csv) | Pipeline stage, producer, inputs, outputs, artifact type, dependencies, and remote location. |
| [`reasoning_experiment_lineage.csv`](reasoning_experiment_lineage.csv) | Per-file size and SHA-256 for the recent reasoning-analysis chain. |
| [`remote_artifact_groups.csv`](remote_artifact_groups.csv) | Directory-level mapping into the private Hugging Face artifact repository. |
| [`REMOTE_ARTIFACTS.md`](REMOTE_ARTIFACTS.md) | Remote repositories and restore commands. |

The experiment directory is the unit of provenance. Read its `run_config.json`,
`query_config.json`, `run_manifest.json`, or preparation manifest before
combining it with another run. Filenames alone do not establish compatibility.

## Current persistence status

The Git branch is `feat/counterfactual-patching`. The last remotely verified
commit before this handoff was `8e7524a`. The handoff is complete only when
the current migration commit is pushed and verified.

The private Hugging Face repository
`project-telos/reveng-experiment-artifacts` was verified at revision
`7541dd2dcf0b9b2faec6c4965e85779b84b36753` with 38,003 files.

The three local files larger than 50 MB match their remote SHA-256 values:

| Local path | Bytes | SHA-256 |
|---|---:|---|
| `data/behavioral_probes/reasoning_belief_action_matched_46_behavioral/belief_rows.jsonl` | 448,689,176 | `53530bcf052ccb787df6423872988fe386d0ed7e3d388243d736349d465ac671` |
| `data/behavioral_probes/reasoning_belief_action_matched_46_behavioral/belief_rows.csv` | 414,601,450 | `0d822a19da707cd78ca7b51fd856071b2abadc94823ed472295efe27406ee2c8` |
| `data/behavioral_probes/doorkey_chunking_validation/sentence_token_boundaries_gpt_oss_20b.csv` | 53,374,107 | `8873752847dc67e9118f54b58bcfb2aa03db3388b0e8dde025cab090b58f2304` |

The local activation download under
`outputs/activation_collection/gpt_oss_20b_boundary_v1/` is incomplete:
`activation_index.parquet` and some temporary files are zero bytes. This does
not lose the completed experiment because the authoritative 5.51 GB copy is in
`project-telos/gpt_oss_20b_doorkey_boundary_activations` at revision
`67dd0ef403f9dde32d36ef54f746f37226539b5b`.

Do not preserve `.venv`, model caches, Hugging Face caches, or credentials.
Manifests pin model revisions and input hashes. Store API and SSH credentials
separately.

## Current new experiment

The matched belief-clause intervention is:

| Stage | Producer | Inputs | Outputs |
|---|---|---|---|
| Prepare | `scripts/run_belief_clause_intervention.py --stage prepare` | matched-46 candidates, final prefix action rows, final belief rows | `cohort.csv`, `preparation_manifest.json` |
| Query | same script with `--stage query` | cohort, pinned GPT-OSS-20B checkpoint | `action_readouts.jsonl`, `query_config.json`, `run_manifest.json` |
| Analyse | same script with `--stage analyze` | cohort and action readouts | analysis tables and `run_report.md` |

The run contains 46 decisions from 31 trajectories and 184 completed
candidate-token readouts. Its raw model output is
`outputs/hypothesis_tests/belief_clause_intervention_matched46_v1/action_readouts.jsonl`.
The manifest freezes the checkpoint revision, temperature 0.7, low reasoning
effort, direct candidate-token scoring, seed 42, and all four conditions.

## Restore on a new instance

```bash
git clone --branch feat/counterfactual-patching git@github.com:SPAR-Telos/reveng.git
cd reveng
uv sync --locked
HF_HOME=.hf_home uv run hf auth login
HF_HOME=.hf_home uv run hf download \
  project-telos/reveng-experiment-artifacts \
  --repo-type dataset --local-dir .
HF_HOME=.hf_home uv run hf download \
  project-telos/gpt_oss_20b_doorkey_boundary_activations \
  --repo-type dataset \
  --revision 67dd0ef403f9dde32d36ef54f746f37226539b5b \
  --local-dir outputs/activation_collection/gpt_oss_20b_boundary_v1
```

Download only the dedicated datasets needed for a run. Their revisions are in
[`REMOTE_ARTIFACTS.md`](REMOTE_ARTIFACTS.md).

Verify the environment:

```bash
uv lock --check
uv sync --locked --dry-run
uv run pytest tests/test_belief_clause_intervention.py
```

## Instance size

For the current GPT-OSS-20B replay and intervention work, use 48 GB VRAM,
64 GB RAM, and at least 150 GB of persistent disk. The measured minimum is
about 24 GB VRAM, but it leaves little room for long traces or concurrent
processes.

For dense BF16 Gemma-4-31B or Qwen3-32B work, use one 80 GB GPU at minimum,
128 GB RAM, and 200 GB persistent disk. Two 80 GB GPUs and 300 GB disk are the
safer general-purpose choice if all checkpoints and outputs must coexist.

On RunPod, `/workspace` is currently a network filesystem. It survives only
if the network volume itself is retained. Deleting the pod and deleting the
volume are different actions. Treat the system disk, `/tmp`, model caches,
and credentials as disposable.

## Before deleting an instance

1. Require a clean `git status`.
2. Verify the migration commit with `git ls-remote origin <branch>`.
3. Verify every file above 50 MB by remote path, size, and SHA-256.
4. Check every completed run has its raw rows, configs, manifests, and report.
5. Copy credentials separately.
6. Delete only the compute instance. Retain the network volume until the new
   checkout and required Hugging Face downloads pass their checksum checks.
