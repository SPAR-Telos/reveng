# Attention Analysis Checks

The following requested layers were excluded because their stored sentence-attention tensors were exactly zero: 8.

For GPT-OSS-20B, layer 8 uses sliding-window attention. The original extractor indexed its 128-token attention window using full-sequence token positions, so the stored layer-8 sentence aggregates are invalid. Layers 15 and 23 use full attention and are retained.
