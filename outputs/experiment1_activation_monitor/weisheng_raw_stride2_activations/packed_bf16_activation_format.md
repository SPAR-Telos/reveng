# Packed BF16 Activation Format

This format stores selected activations as raw packed `torch.bfloat16` numbers, not one
`.pt` file per token. It is intended for large extraction runs where the per-token
PyTorch file overhead is too expensive.

## Extraction Command

The key-door packed extraction run uses:

```bash
uv run --extra mlx python -u evaluation_scripts/extract_key_door_activations_mlx.py \
  --storage-format packed_raw_bf16 \
  --output-dir data/activations_key_door_100_mlx_packed_bf16 \
  --no-attention \
  --layer-stride 2 \
  --cot-stride 2
```

This extracts:

- every second resolved MLX layer: `0,2,4,...,22`
- `pre_reasoning`: prompt suffix token indices `-3:-1`
- `post_reasoning`: output token indices `-16:-14`
- `cot`: every second output token tagged `analysis`
- no attention matrices

## Directory Layout

Each raw activation record is one matrix for a trajectory, model, layer, step, and
category:

```text
data/activations_key_door_100_mlx_packed_bf16/
  {trajectory}/
    {model}/
      layer_{N}/
        step_{M}/
          pre_reasoning.bf16
          pre_reasoning.json
          post_reasoning.bf16
          post_reasoning.json
          cot.bf16
          cot.json
```

The `.bf16` file contains only raw little-endian bfloat16 values. There is no header.
The matching `.json` sidecar stores the shape and token-index metadata.

Example sidecar:

```json
{
  "format": "raw_bfloat16",
  "dtype": "torch.bfloat16",
  "shape": [3, 2880],
  "byte_order": "little",
  "category": "pre_reasoning",
  "layer_idx": 0,
  "step_idx": 0,
  "token_indices": [
    {"relative_idx": 10, "absolute_idx": 1998},
    {"relative_idx": 11, "absolute_idx": 1999},
    {"relative_idx": 12, "absolute_idx": 2000}
  ]
}
```

Rows in the activation matrix correspond one-to-one with `token_indices`.

## Reading A Record

Use `torch.from_file` to memory-map or read the raw uint16 payload, then view it as
`torch.bfloat16` and reshape using the metadata.

```python
import json
import math
from pathlib import Path

import torch


def load_packed_bf16(record_path: str | Path) -> tuple[torch.Tensor, dict]:
    record_path = Path(record_path)
    metadata = json.loads(record_path.with_suffix(".json").read_text())
    shape = metadata["shape"]
    numel = math.prod(shape)

    raw = torch.from_file(str(record_path), dtype=torch.uint16, size=numel)
    tensor = raw.view(torch.bfloat16).reshape(shape)
    return tensor, metadata


activation, metadata = load_packed_bf16(
    "data/activations_key_door_100_mlx_packed_bf16/"
    "together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_0/"
    "openai__gpt-oss-20b/layer_0/step_0/cot.bf16"
)

print(activation.shape)
print(activation.dtype)
print(metadata["token_indices"][0])
```

For most analysis code, convert to `float32` after loading only if needed:

```python
activation_f32 = activation.float()
```

## Storage Notes

This format is close to the raw numeric size:

```text
num_selected_positions * num_layers * hidden_size * 2 bytes
```

For the current all-trajectory run with every second layer and every second COT
token plus pre/post reasoning, the estimated raw size is about `82.5 GB`
(`76.8 GiB`), plus small JSON and directory overhead.
