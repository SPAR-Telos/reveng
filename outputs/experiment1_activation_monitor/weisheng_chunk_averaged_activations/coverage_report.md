# Weisheng Activation Chunk Mapping Report

## Method

The source archive stores GPT-OSS-20B COT activations every two generated analysis tokens. For each canonical reasoning chunk, this command averages all sampled token activations whose token offsets overlap that chunk's character span. Chunks with no sampled token are kept in the coverage table but omitted from the chunk-activation tensor table.

This is suitable for chunk-level analysis, but not for token-level boundary claims. Missing chunks must be reported because one-token or unlucky-boundary chunks can have no sampled activation under stride-2 extraction.

## Coverage

- Source archive: `data/together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_0.zip`
- Source trajectory: `together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_0`
- Source model directory: `openai__gpt-oss-20b`
- Chunk boundaries: `data/behavioral_probes/doorkey_chunking_validation/chunked_trajectories_legacy_tail_packed_v1.csv`
- Archive layers used: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
- Archive steps used: [0, 1, 2, 3, 4, 5, 6, 7]
- Canonical chunks overall: 39051
- Canonical chunks in archived trajectory steps: 242
- Chunks with at least one sampled activation: 242
- Chunks without sampled activation: 0
- Chunk coverage within archived steps: 1.000
- Relation to all canonical chunks: `subset`
- Relation within archived trajectory steps: `whole`
- Token-level relation within archived trajectory steps: `superset`
- Activation steps without canonical chunks: 0
- Sampled analysis tokens outside canonical chunks: 5

These activations retain the legacy chunk boundaries used by the completed 8-state pilot. New experiments should use the balanced boundaries in `chunked_trajectories.csv`; the existing chunk averages cannot be split retrospectively where a new boundary falls inside an old chunk.

## Outputs

- `chunk_activation_rows.csv`: one row per covered chunk and layer, with a path to the averaged tensor.
- `chunk_coverage_rows.csv`: one row per canonical chunk in the archived trajectory steps, including no-activation chunks.
- `coverage_summary.csv`: compact coverage decision table.
- `chunk_activations/`: averaged chunk tensors saved as `.pt` files.
