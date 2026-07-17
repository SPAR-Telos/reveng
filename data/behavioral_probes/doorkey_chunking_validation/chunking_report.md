# DoorKey Chunking Validation

- Chunker version: `doorkey_sentence_v1`
- Packing version: `balanced_contiguous_word_count_v2`
- Trajectory files: 95
- Reasoning traces: 1276
- Mean sentences: 120.39
- Median sentences: 97.0
- Max sentences: 631
- Traces over 32 sentences: 86.8%
- Max chunks per row in `chunked_trajectories.csv`: 32
- Validation errors: 0

## Validation

The chunks are usable because every exported sentence is tied to a fixed character span in the original reasoning trace, and every packed trajectory chunk is a deterministic view over adjacent sentence spans. Traces exceeding 32 sentences are divided at approximately equal cumulative word-count boundaries, while preserving sentence order. The same span IDs can therefore be reused for activation extraction, behavioral belief queries, action-prefix evaluation, semantic labels, and later interventions.

| Check | Count | Interpretation |
|---|---:|---|
| Empty chunks | 0 | No empty sentence rows were emitted. |
| Span mismatches | 0 | Exported text matches `raw_trace[char_start:char_end]`. |
| Character-span overlaps | 0 | No overlapping sentence spans were found. |
| Terminal action JSON in reasoning | 0 | Terminal action JSON is not leaked into reasoning chunks. |
| Isolated numbered-list fragments | 0 | No false split such as `1.` or `2.` was found. |
| Punctuation-only fragments | 0 | No meaningless punctuation-only chunks were found. |
| Analysis chunks over cap | 0 | Every packed trajectory has at most 32 chunks. |

## Chunk-Length Balance

For the 1113 traces packed into exactly 32 chunks:

| Position | Median words |
|---|---:|
| First chunk | 24.0 |
| Penultimate chunk | 29.0 |
| Final chunk | 29.0 |

The median final-chunk length divided by the median length of the preceding chunks is 1.00. The previous greedy packer produced a much longer final chunk by forcing all remaining sentences into chunk 32; that legacy table is retained only for reproducing the existing 8-state activation run.

Internal action JSON mentions are allowed when reasoning continues after them; 270 such mentions were flagged, including 54 standalone JSON snippets.

Primary outputs:

- `sentences.csv`
- `chunked_trajectories.csv`
- `chunking_errors.csv`
