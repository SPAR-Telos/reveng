# Maze API smoke test

Status: complete (480/480 trajectories complete).

This smoke test checks maze-solving behavior, output-token requirements, and runtime before activation collection. Every model sees the same regular no-key/no-door mazes. At each step it sees only the current full grid; trajectory history is not included. The generator setting called `difficulty` controls how many walls are retained; it is not the realized wall density, which is recorded separately.

## Frozen setup

- Grid sizes: 7
- Difficulty settings: 0.0, 0.6, 1.0; two grids per size/difficulty cell
- Grid selection: median and 90th-percentile optimal path length from 32 deterministic candidates per cell
- Main sampling: temperature 0.7 on both grids; one temperature-0 trajectory on the first grid in each cell
- Maximum output per action call: 16,000 tokens
- Context mode: current full grid only (no trajectory history)
- Reasoning effort: low and medium for GPT-OSS; provider-native/default for models without that control

## Results by condition

| Model | Size | Difficulty | Reasoning | T | Goal reached | Optimal actions | Output tokens/call p50 / p95 / max | Total output tokens/trajectory p50 / p95 / max | Trajectory runtime p50 / p95 (s) | Invalid outputs | Truncated or failed |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-OSS-20B | 7 | 0.0 | low | 0.0 | 100% | 99% | 70 / 139 / 189 | 273 / 505 / 517 | 3.3 / 5.5 | 0% | 0% |
| GPT-OSS-20B | 7 | 0.0 | low | 0.7 | 100% | 99% | 84 / 186 / 246 | 300 / 614 / 772 | 3.4 / 6.3 | 0% | 0% |
| GPT-OSS-20B | 7 | 0.2 | low | 0.0 | 100% | 95% | 135 / 411 / 558 | 412 / 1206 / 1556 | 5.1 / 10.1 | 0% | 0% |
| GPT-OSS-20B | 7 | 0.2 | low | 0.7 | 100% | 98% | 128 / 408 / 759 | 472 / 1532 / 1865 | 4.4 / 12.6 | 0% | 0% |
| GPT-OSS-20B | 7 | 0.4 | low | 0.0 | 100% | 95% | 260 / 678 / 960 | 872 / 2667 / 5685 | 8.8 / 19.6 | 0% | 0% |
| GPT-OSS-20B | 7 | 0.4 | low | 0.7 | 100% | 96% | 220 / 673 / 1107 | 976 / 2596 / 3353 | 8.6 / 26.2 | 0% | 0% |
| GPT-OSS-20B | 7 | 0.6 | low | 0.0 | 90% | 78% | 222 / 588 / 931 | 968 / 3946 / 4575 | 8.2 / 28.2 | 0% | 10% |
| GPT-OSS-20B | 7 | 0.6 | low | 0.7 | 100% | 98% | 149 / 611 / 1332 | 751 / 2314 / 3431 | 6.1 / 23.5 | 0% | 0% |
| GPT-OSS-20B | 7 | 0.8 | low | 0.0 | 95% | 92% | 338 / 1127 / 1605 | 1663 / 9217 / 9472 | 11.7 / 62.6 | 0% | 5% |
| GPT-OSS-20B | 7 | 0.8 | low | 0.7 | 97% | 92% | 347 / 1147 / 1664 | 1436 / 9986 / 16086 | 11.9 / 83.8 | 0% | 2% |
| GPT-OSS-20B | 7 | 1.0 | low | 0.0 | 100% | 98% | 296 / 840 / 1236 | 2094 / 5047 / 5328 | 17.3 / 35.7 | 0% | 0% |
| GPT-OSS-20B | 7 | 1.0 | low | 0.7 | 97% | 94% | 321 / 933 / 1337 | 2008 / 4961 / 13600 | 18.2 / 41.6 | 0% | 3% |

## Decisions

Models suitable for full run: GPT-OSS-20B
Recommended reasoning setting/model: GPT-OSS-20B: low
Recommended max tokens/model: GPT-OSS-20B: 1,280 tokens (observed p95 871)
Feasible grid sizes: GPT-OSS-20B: 7
Observed p95 tokens/model: GPT-OSS-20B: 1,280 tokens (observed p95 871)
Observed p95 runtime/model: GPT-OSS-20B: 30.4 s p95/trajectory
Behavioral/API failures: see api_calls.csv for invalid, truncated, retried, and failed calls
Estimated API cost so far: $0.1771 across complete trajectories with known pricing

The low-versus-medium rule deliberately prefers low. Medium is selected only when low fails the usability gate, medium passes it, and medium improves goal success, optimal-action rate, or invalid/failure rate by at least 10 percentage points on the same grid set. The recommended token cap is 120% of the observed per-call p95, rounded up to 256 tokens and never above 16k.

## Limits

These are small-sample engineering estimates, not paper-level capability comparisons. API trajectories contain no activations and cannot replace the local activation-bearing run. The Qwen3-32B candidate requires an account-specific Together dedicated endpoint name. Its monetary estimate remains unavailable until an hourly endpoint price is entered in the config.
