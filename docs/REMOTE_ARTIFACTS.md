---
pretty_name: Reveng Experiment Artifacts
license: apache-2.0
tags:
- interpretability
- activations
- behavioral-evaluation
- doorkey
---

# Remote experiment artifacts

This file maps large or generated local artifacts to their remote storage.
Repository code, setup documents, manifests, reports, status files, and plans
remain in Git. Large raw/derived experiment bundles are stored in Hugging Face
datasets under `project-telos`.

## Consolidated experiment bundle

Repository: `project-telos/reveng-experiment-artifacts` (private)

Verified revision: `708b1ce4e77351582e364d4581b50485fba61568`
(38,002 files; approximately 2.60 GB on the Hub).

The remote paths preserve their repository-relative layout. The contents and
local sizes at migration time are listed in
[`remote_artifact_groups.csv`](remote_artifact_groups.csv).

Restore the bundle into a fresh repository checkout:

```bash
HF_HOME=.hf_home uv run hf download \
  project-telos/reveng-experiment-artifacts \
  --repo-type dataset \
  --local-dir .
```

The dataset intentionally excludes environments, API/Hugging Face credentials,
package/model caches, `.git`, and the redundant 5.2 GB activation ZIP.

## Dedicated datasets reused

| Local artifact | Hugging Face dataset | Verified revision |
|---|---|---|
| `outputs/activation_collection/gpt_oss_20b_boundary_v1/` | `project-telos/gpt_oss_20b_doorkey_boundary_activations` | `67dd0ef403f9dde32d36ef54f746f37226539b5b` |
| `data/hf/trajectories_key_door_100/` | `project-telos/trajectories_key_door_100` | `a8adc8c3ce31f2195af6172b95786ae1d60585a4` |
| `data/hf/trajectories_key_door_env_100_wrong_prompt_suffix/` | `project-telos/trajectories_key_door_env_100_wrong_prompt_suffix` | `39f5c56e7fc43cd3c2e99f75f26ba8273d2718b0` |
| `data/gpt_oss_maze_cot_120_pilot/` and later maze collections | `project-telos/trajectories_maze_120_pilot`, `project-telos/gpt_oss_maze_cot_120_m11_v1`, `project-telos/gemma-4-31b-it_maze_cot_120_m11_v1`, `project-telos/qwen3.5-9b_maze_cot_120_m11_v1` | See each dataset revision on Hugging Face |
| Action-distribution artifacts | `project-telos/gpt_oss_doorkey_action_distributions` | `8072b6d664efa6822efd421bf37b298bbdd202d1` |
| Original and replicate semantic runs, inventory, taxonomy, and diagnostics | `project-telos/doorkey-semantic-reasoning-labels` | `5708370a409afdffec083eaf5d8747d02aa10672` |
| Counterfactual datasets | `project-telos/counterfactual-trajectories`, `project-telos/counterfactual-grids`, `project-telos/counterfactual-activations` | See each dataset revision on Hugging Face |

`data/hf/cache/` is a download cache, not an experiment result, and is excluded.
Model checkpoints are also excluded because the exact model IDs and revisions
are recorded in experiment configuration/manifests and can be downloaded again.

## Consistency rule

Remote location does not determine compatibility. Before combining results,
compare the local or remote manifests for input hashes, model and revision,
judge prompt/schema/few-shot hashes, decoding settings, seeds, layer sets, and
representation definitions. Use a new output directory when any of these
change.
