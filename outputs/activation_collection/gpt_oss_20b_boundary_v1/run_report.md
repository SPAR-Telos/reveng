# GPT-OSS-20B Full Boundary-Activation Extraction

- Status: completed
- Model revision: `6cee5e81ee83917806bbde320786a8fb61efebee`
- Trajectory files: 95
- Environment states: 1,276
- Reasoning sentences: 153,622
- Reasoning tokens: 2,370,850
- Layers: 8, 15, 23
- Forward chunk size: 512
- Wall-clock time: 0.68 hours
- Model loading time: 6.25 seconds
- Forward time: 0.63 hours
- Mean reasoning-token throughput: 1041.0 tokens/second
- Peak reserved VRAM: 21.76 GiB
- Peak process RAM: 5.90 GiB
- Packed activation storage: 5.51 GB

Each state shard contains BF16 sentence means, sentence-final activations, complete-reasoning means and finals, PRE and POST three-token windows, and the final-action-token activation at layers 8, 15, and 23.

Attention is not included in this dataset. The separate pilot benchmark measured final-action attention to preceding sentences without adding it to the primary extraction.
