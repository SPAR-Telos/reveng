# GPT-OSS-20B DoorKey Boundary Activations

This directory contains hidden-state representations extracted from the fixed
Project Telos DoorKey reasoning trajectories.

## Coverage

- Model: `openai/gpt-oss-20b`
- Model revision: `6cee5e81ee83917806bbde320786a8fb61efebee`
- Trajectories: 95
- Environment states: 1,276
- Reasoning sentences: 153,622
- Reasoning tokens: 2,370,850
- Layers: 8, 15, and 23
- Activation dtype: BF16
- Hidden dimension: 2,880

`run_manifest.json` records the complete extraction configuration and counts.
The extraction completed with no pending states, and all stored trajectory
token IDs matched the pinned GPT-OSS tokenizer.

## Representations

Each state has one file in `shards/`. At each layer, it stores:

- `sentence_mean`: mean activation over every token in each reasoning sentence
- `sentence_final`: activation at the final token of each reasoning sentence
- `reasoning_mean`: mean activation over the complete reasoning trace
- `reasoning_final`: activation at the final reasoning token
- `pre_window`: three-token window immediately before reasoning
- `post_window`: three-token window immediately after reasoning
- `action_token`: activation at the final action token

Attention weights and per-token activations are not included.

## Index

`activation_index.parquet` maps trajectory, environment step, sentence, layer,
and token span to a tensor key and row in a shard. Token spans are zero-based,
half-open positions in the exact stored model input.

For sentence rows, `tensor_row` selects the sentence in both
`mean_tensor_key` and `final_tensor_key`. Other record types point to one tensor
through either `mean_tensor_key` or `final_tensor_key`.

## Loading A Sentence Representation

```python
from pathlib import Path

import pandas as pd
from safetensors import safe_open

root = Path("gpt_oss_20b_boundary_v1")
index = pd.read_parquet(root / "activation_index.parquet")

row = index[
    (index["record_kind"] == "sentence")
    & (index["trajectory_id"] == "together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_0")
    & (index["step_index"] == 0)
    & (index["sentence_id"] == 0)
    & (index["layer"] == 15)
].iloc[0]

# The index records the original repository-relative path. Using the basename
# keeps loading valid after downloading or moving this directory.
shard = root / "shards" / Path(row["shard_path"]).name
with safe_open(shard, framework="pt", device="cpu") as handle:
    sentence_mean = handle.get_tensor(row["mean_tensor_key"])[int(row["tensor_row"])]
    sentence_final = handle.get_tensor(row["final_tensor_key"])[int(row["tensor_row"])]

print(sentence_mean.shape, sentence_mean.dtype)
```

Install the required readers with:

```bash
pip install pandas pyarrow safetensors torch
```

## Integrity

From this directory, verify all activation shards with:

```bash
sha256sum -c SHA256SUMS
```

`run_config.json`, `run_manifest.json`, `run_report.md`, and
`validation_report.json` document provenance and validation. The underlying
trajectory text is not duplicated in the shards; join by `trajectory_id` and
`step_index` to the Project Telos DoorKey trajectory dataset.

Before redistributing publicly, confirm that the repository license and terms
cover both the source trajectories and derived GPT-OSS activations.
