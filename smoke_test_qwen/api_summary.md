# Maze API smoke test

Status: prepared / incomplete (0/108 trajectories complete).

This smoke test checks maze-solving behavior, output-token requirements, and runtime before activation collection. Every model sees the same regular no-key/no-door mazes. At each step it sees only the current full grid; trajectory history is not included. The generator setting called `difficulty` controls how many walls are retained; it is not the realized wall density, which is recorded separately.

## Frozen setup

- Grid sizes: 7, 11, 15
- Difficulty settings: 0.0, 0.6, 1.0; two grids per size/difficulty cell
- Grid selection: median and 90th-percentile optimal path length from 32 deterministic candidates per cell
- Main sampling: temperature 0.7 on both grids; one temperature-0 trajectory on the first grid in each cell
- Maximum output per action call: 16,000 tokens
- Context mode: current full grid only (no trajectory history)
- Reasoning effort: low and medium for GPT-OSS; provider-native/default for models without that control

## Results by condition

No complete API trajectories have been run yet. Run the query stage on the API machine, then run analyze.

## Decisions

Models suitable for full run: pending
Recommended reasoning setting/model: GPT-OSS-20B: pending; Gemma-4-31B-IT: pending; Qwen3-32B: pending
Recommended max tokens/model: GPT-OSS-20B: pending; Gemma-4-31B-IT: pending; Qwen3-32B: pending
Feasible grid sizes: pending
Observed p95 tokens/model: GPT-OSS-20B: pending; Gemma-4-31B-IT: pending; Qwen3-32B: pending
Observed p95 runtime/model: GPT-OSS-20B: pending; Gemma-4-31B-IT: pending; Qwen3-32B: pending
Behavioral/API failures: see api_calls.csv; 108 trajectories remain
Estimated API cost so far: not available

The low-versus-medium rule deliberately prefers low. Medium is selected only when low fails the usability gate, medium passes it, and medium improves goal success, optimal-action rate, or invalid/failure rate by at least 10 percentage points on the same grid set. The recommended token cap is 120% of the observed per-call p95, rounded up to 256 tokens and never above 16k.

## Limits

These are small-sample engineering estimates, not paper-level capability comparisons. API trajectories contain no activations and cannot replace the local activation-bearing run. The Qwen3-32B candidate requires an account-specific Together dedicated endpoint name. Its monetary estimate remains unavailable until an hourly endpoint price is entered in the config.
