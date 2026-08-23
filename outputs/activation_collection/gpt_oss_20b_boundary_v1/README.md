---
language:
- en
tags:
- mechanistic-interpretability
- activations
- reasoning
- gridworld
- doorkey
- gpt-oss-20b
pretty_name: GPT-OSS-20B DoorKey Boundary Activations
size_categories:
- 1K<n<10K
---

# GPT-OSS-20B DoorKey Boundary Activations

This dataset contains GPT-OSS-20B hidden-state activations for fixed Project Telos DoorKey reasoning trajectories. It is intended for analysis of how action recommendations, state beliefs, and reasoning text relate across sentence and environment-step boundaries.

## Source and Scope

- Source trajectories: `project-telos/trajectories_key_door_100`
- Model: `openai/gpt-oss-20b`
- Model revision: `6cee5e81ee83917806bbde320786a8fb61efebee`
- Trajectories: 95
- Environment states: 1,276
- Reasoning sentences: 153,622
- Reasoning tokens: 2,370,850
- Layers: 8, 15, and 23
- Hidden dimension: 2,880
- Activation dtype: BF16
- Packed activation storage: about 5.51 GB

The trajectory text is not duplicated in the activation shards. Join activations to the source trajectories using `trajectory_id` and `step_index`.

## Files

| File or directory | Description |
|---|---|
| `activation_index.parquet` | Searchable index mapping trajectory, environment step, sentence, layer, token span, tensor key, and tensor row to activation shards. |
| `shards/` | Packed `safetensors` activation shards, one per environment state. |
| `state_records/` | Per-state extraction records and metadata. |
| `README_USAGE.md` | Loading example and detailed field descriptions. |
| `run_config.json` | Extraction configuration. |
| `run_manifest.json` | Extraction counts and status. |
| `run_report.md` | Human-readable extraction summary. |
| `validation_report.json` | Validation summary. |
| `SHA256SUMS` | Checksums for integrity verification. |

## Representations

Each state shard stores the following representations at layers 8, 15, and 23:

| Representation | Meaning |
|---|---|
| `sentence_mean` | Mean activation over all tokens in one reasoning sentence. |
| `sentence_final` | Activation at the final token of one reasoning sentence. |
| `reasoning_mean` | Mean activation over the full reasoning trace. |
| `reasoning_final` | Activation at the final reasoning token. |
| `pre_window` | Three-token window immediately before reasoning. |
| `post_window` | Three-token window immediately after reasoning. |
| `action_token` | Activation at the final action token. |

Attention weights and full per-token activations are not included.

## Loading

Use `activation_index.parquet` to locate the shard, tensor key, and tensor row for a sentence. See `README_USAGE.md` for a complete example using `pandas`, `safetensors`, and `torch`.

## Validation

The extraction completed with no pending states. Stored trajectory token IDs matched the pinned GPT-OSS tokenizer. Token spans in `activation_index.parquet` are zero-based and half-open.

To verify downloaded files, run:

```bash
sha256sum -c SHA256SUMS
```

## Intended Use

This dataset is intended for research on language-model reasoning, belief readouts, action commitment, and activation-level monitoring in DoorKey gridworld trajectories. It should not be treated as a general benchmark of GPT-OSS-20B behavior outside this fixed environment and prompt distribution.

## License and Terms

This is a derived activation dataset. Users should comply with the terms for the source Project Telos trajectories and the `openai/gpt-oss-20b` model. No additional trajectory text is included in the activation shards.
