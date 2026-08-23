# GPT-OSS-20B Boundary-Activation Pilot

- Status: completed
- Model revision: `6cee5e81ee83917806bbde320786a8fb61efebee`
- Selected chunk size: 512
- Model load time: 5.37 seconds
- Pilot states completed: 4
- Pilot reasoning tokens: 16,513
- Pilot sentences: 1,089

## Measured states

| Length | Reasoning tokens | Sentences | Forward seconds | Tokens/second | Peak VRAM | Shard size |
|---|---:|---:|---:|---:|---:|---:|
| short | 348 | 28 | 0.61 | 572.0 | 13.47 GiB | 1.07 MiB |
| median | 1,480 | 110 | 1.32 | 1121.3 | 14.35 GiB | 3.78 MiB |
| p95 | 4,952 | 320 | 4.27 | 1160.7 | 21.38 GiB | 10.70 MiB |
| maximum | 9,733 | 631 | 10.32 | 943.2 | 21.75 GiB | 20.95 MiB |

## Full-corpus projection

- Central end-to-end extraction time: 0.80 hours
- Conservative end-to-end extraction time: 0.93 hours
- Exact teacher-forced input tokens: 3,378,242
- Projected packed storage: 5.51 GB
- Projected logical tensor storage: 5.51 GB

## Separate compact-attention benchmark

- Median-state attention pass: 1.44 seconds
- Median-state attention output: 83.0 KiB
- Rough full-corpus incremental time: 0.59 hours
- Rough full-corpus attention storage: 118.58 MB

The attention estimate comes from one median-length state and is less reliable than the four-state residual-stream estimate. It covers attention from the final action token to preceding sentences and is not included in the extraction times above.
