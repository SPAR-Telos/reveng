# Maze API smoke test

Status: prepared / incomplete (81/108 trajectories complete).

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

| Model | Size | Difficulty | Reasoning | T | Goal reached | Optimal actions | Output tokens/call p50 / p95 / max | Total output tokens/trajectory p50 / p95 / max | Trajectory runtime p50 / p95 (s) | Invalid outputs | Truncated or failed |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-OSS-20B | 11 | 0.0 | low | 0.0 | 100% | 100% | 47 / 115 / 131 | 312 / 312 / 312 | 2.9 / 2.9 | 0% | 0% |
| GPT-OSS-20B | 11 | 0.0 | low | 0.7 | 100% | 100% | 57 / 83 / 85 | 459 / 634 / 654 | 4.7 / 6.5 | 0% | 0% |
| GPT-OSS-20B | 11 | 0.0 | medium | 0.0 | 100% | 100% | 159 / 335 / 372 | 1011 / 1011 / 1011 | 8.9 / 8.9 | 0% | 0% |
| GPT-OSS-20B | 11 | 0.0 | medium | 0.7 | 100% | 94% | 184 / 261 / 307 | 1596 / 1850 / 1878 | 13.3 / 16.0 | 0% | 0% |
| GPT-OSS-20B | 11 | 0.6 | low | 0.0 | 0% | 53% | 1606 / 3253 / 3929 | 25001 / 25001 / 25001 | 203.1 / 203.1 | 0% | 100% |
| GPT-OSS-20B | 11 | 0.6 | low | 0.7 | 100% | 85% | 441 / 1926 / 2457 | 9011 / 13004 / 13448 | 72.2 / 98.0 | 0% | 0% |
| GPT-OSS-20B | 11 | 0.6 | medium | 0.0 | 100% | 100% | 902 / 2925 / 2950 | 11388 / 11388 / 11388 | 70.0 / 70.0 | 0% | 0% |
| GPT-OSS-20B | 11 | 0.6 | medium | 0.7 | 100% | 84% | 1178 / 5092 / 5921 | 19438 / 20215 / 20301 | 213.9 / 287.9 | 0% | 0% |
| GPT-OSS-20B | 11 | 1.0 | low | 0.0 | 0% | 54% | 1727 / 3395 / 3645 | 39147 / 39147 / 39147 | 339.8 / 339.8 | 0% | 100% |
| GPT-OSS-20B | 11 | 1.0 | low | 0.7 | 0% | 54% | 990 / 2133 / 2763 | 44779 / 60539 / 62290 | 398.8 / 525.7 | 0% | 100% |
| GPT-OSS-20B | 11 | 1.0 | medium | 0.0 | 100% | 100% | 2486 / 4642 / 4879 | 31565 / 31565 / 31565 | 409.7 / 409.7 | 0% | 0% |
| GPT-OSS-20B | 11 | 1.0 | medium | 0.7 | 100% | 82% | 3489 / 7489 / 10119 | 115882 / 192946 / 201509 | 1287.2 / 2076.6 | 0% | 0% |
| GPT-OSS-20B | 15 | 0.0 | low | 0.0 | 100% | 100% | 48 / 73 / 78 | 509 / 509 / 509 | 5.8 / 5.8 | 0% | 0% |
| GPT-OSS-20B | 15 | 0.0 | low | 0.7 | 100% | 100% | 44 / 72 / 143 | 612 / 626 / 627 | 7.2 / 7.5 | 0% | 0% |
| GPT-OSS-20B | 15 | 0.0 | medium | 0.0 | 100% | 100% | 160 / 262 / 278 | 1679 / 1679 / 1679 | 16.9 / 16.9 | 0% | 0% |
| GPT-OSS-20B | 15 | 0.0 | medium | 0.7 | 100% | 100% | 168 / 362 / 571 | 2092 / 2404 / 2439 | 21.4 / 23.4 | 0% | 0% |
| GPT-OSS-20B | 15 | 0.6 | low | 0.0 | 0% | 69% | 215 / 2386 / 4131 | 11733 / 11733 / 11733 | 77.1 / 77.1 | 0% | 100% |
| GPT-OSS-20B | 15 | 0.6 | low | 0.7 | 50% | 74% | 242 / 728 / 1858 | 4982 / 5085 / 5096 | 36.5 / 37.4 | 0% | 50% |
| GPT-OSS-20B | 15 | 0.6 | medium | 0.0 | 0% | 69% | 1264 / 2853 / 3361 | 21940 / 21940 / 21940 | 544.6 / 544.6 | 0% | 100% |
| GPT-OSS-20B | 15 | 0.6 | medium | 0.7 | 100% | 90% | 1612 / 4025 / 4882 | 27332 / 44333 / 46222 | 658.8 / 1174.6 | 0% | 0% |
| GPT-OSS-20B | 15 | 1.0 | low | 0.0 | 0% | 42% | 616 / 1763 / 2296 | 26791 / 26791 / 26791 | 198.2 / 198.2 | 0% | 100% |
| GPT-OSS-20B | 15 | 1.0 | low | 0.7 | 50% | 54% | 442 / 1663 / 2294 | 36711 / 60306 / 62928 | 320.3 / 543.7 | 0% | 50% |
| GPT-OSS-20B | 15 | 1.0 | medium | 0.0 | 100% | 100% | 1938 / 4344 / 4783 | 43681 / 43681 / 43681 | 744.1 / 744.1 | 0% | 0% |
| GPT-OSS-20B | 15 | 1.0 | medium | 0.7 | 50% | 91% | 1776 / 3807 / 4267 | 21334 / 37430 / 39219 | 647.3 / 699.2 | 0% | 50% |
| GPT-OSS-20B | 7 | 0.0 | low | 0.0 | 100% | 100% | 47 / 114 / 121 | 215 / 215 / 215 | 1.9 / 1.9 | 0% | 0% |
| GPT-OSS-20B | 7 | 0.0 | low | 0.7 | 100% | 90% | 81 / 152 / 157 | 477 / 695 / 719 | 5.1 / 6.6 | 0% | 0% |
| GPT-OSS-20B | 7 | 0.0 | medium | 0.0 | 100% | 100% | 211 / 250 / 254 | 647 / 647 / 647 | 7.0 / 7.0 | 0% | 0% |
| GPT-OSS-20B | 7 | 0.0 | medium | 0.7 | 100% | 100% | 232 / 312 / 317 | 977 / 1152 / 1171 | 10.8 / 12.4 | 0% | 0% |
| GPT-OSS-20B | 7 | 0.6 | low | 0.0 | 100% | 100% | 282 / 363 / 370 | 1133 / 1133 / 1133 | 8.7 / 8.7 | 0% | 0% |
| GPT-OSS-20B | 7 | 0.6 | low | 0.7 | 100% | 100% | 362 / 934 / 998 | 2161 / 3336 / 3466 | 16.4 / 25.1 | 0% | 0% |
| GPT-OSS-20B | 7 | 0.6 | medium | 0.0 | 100% | 100% | 482 / 821 / 868 | 2106 / 2106 / 2106 | 19.2 / 19.2 | 0% | 0% |
| GPT-OSS-20B | 7 | 0.6 | medium | 0.7 | 100% | 100% | 761 / 1919 / 1975 | 5166 / 7245 / 7476 | 45.6 / 63.9 | 0% | 0% |
| GPT-OSS-20B | 7 | 1.0 | low | 0.0 | 100% | 100% | 382 / 417 / 419 | 1231 / 1231 / 1231 | 9.8 / 9.8 | 0% | 0% |
| GPT-OSS-20B | 7 | 1.0 | low | 0.7 | 100% | 92% | 272 / 715 / 807 | 2254 / 3447 / 3579 | 17.3 / 26.6 | 0% | 0% |
| GPT-OSS-20B | 7 | 1.0 | medium | 0.0 | 100% | 100% | 476 / 922 / 997 | 2258 / 2258 / 2258 | 22.7 / 22.7 | 0% | 0% |
| GPT-OSS-20B | 7 | 1.0 | medium | 0.7 | 100% | 100% | 805 / 1434 / 1675 | 4910 / 7123 / 7369 | 38.6 / 52.8 | 0% | 0% |
| Gemma-4-31B-IT | 11 | 0.0 | native | 0.0 | 100% | 100% | 536 / 3352 / 3865 | 6538 / 6538 / 6538 | 116.4 / 116.4 | 0% | 0% |
| Gemma-4-31B-IT | 11 | 0.0 | native | 0.7 | 100% | 100% | 1104 / 1618 / 1726 | 8556 / 12540 / 12983 | 169.2 / 260.8 | 0% | 0% |
| Gemma-4-31B-IT | 11 | 0.6 | native | 0.0 | 100% | 100% | 1962 / 2858 / 2887 | 13210 / 13210 / 13210 | 409.2 / 409.2 | 0% | 0% |
| Gemma-4-31B-IT | 11 | 0.6 | native | 0.7 | 50% | 100% | 2017 / 3479 / 3620 | 7794 / 14808 / 15587 | 582.7 / 611.2 | 0% | 50% |
| Gemma-4-31B-IT | 11 | 1.0 | native | 0.0 | 0% | 0% | not available / not available / not available | 0 / 0 / 0 | 551.1 / 551.1 | 0% | 100% |
| Gemma-4-31B-IT | 11 | 1.0 | native | 0.7 | 0% | 0% | not available / not available / not available | 0 / 0 / 0 | 551.4 / 551.6 | 0% | 100% |
| Gemma-4-31B-IT | 15 | 0.0 | native | 0.0 | 100% | 100% | 453 / 1129 / 1245 | 5965 / 5965 / 5965 | 127.6 / 127.6 | 0% | 0% |
| Gemma-4-31B-IT | 15 | 0.0 | native | 0.7 | 100% | 100% | 1425 / 2135 / 2519 | 14752 / 19268 / 19770 | 521.9 / 657.5 | 0% | 0% |
| Gemma-4-31B-IT | 15 | 0.6 | native | 0.0 | 100% | 100% | 844 / 3000 / 3067 | 10080 / 10080 / 10080 | 430.0 / 430.0 | 0% | 0% |
| Gemma-4-31B-IT | 15 | 0.6 | native | 0.7 | 50% | 100% | 1607 / 2846 / 2868 | 6990 / 13280 / 13979 | 401.7 / 536.5 | 0% | 50% |
| Gemma-4-31B-IT | 15 | 1.0 | native | 0.0 | 0% | 100% | 4824 / 4824 / 4824 | 4824 / 4824 / 4824 | 731.2 / 731.2 | 0% | 100% |
| Gemma-4-31B-IT | 15 | 1.0 | native | 0.7 | 0% | 0% | not available / not available / not available | 0 / 0 / 0 | 551.1 / 551.1 | 0% | 100% |
| Gemma-4-31B-IT | 7 | 0.0 | native | 0.0 | 100% | 100% | 366 / 3430 / 3770 | 4430 / 4430 / 4430 | 116.5 / 116.5 | 0% | 0% |
| Gemma-4-31B-IT | 7 | 0.0 | native | 0.7 | 100% | 100% | 573 / 2086 / 2118 | 3970 / 4997 / 5111 | 68.0 / 100.3 | 0% | 0% |
| Gemma-4-31B-IT | 7 | 0.6 | native | 0.0 | 100% | 100% | 1830 / 2537 / 2569 | 6830 / 6830 / 6830 | 98.6 / 98.6 | 0% | 0% |
| Gemma-4-31B-IT | 7 | 0.6 | native | 0.7 | 50% | 100% | 1618 / 2379 / 2419 | 3064 / 5823 / 6129 | 369.0 / 533.5 | 0% | 50% |
| Gemma-4-31B-IT | 7 | 1.0 | native | 0.0 | 100% | 100% | 2104 / 2993 / 3036 | 7853 / 7853 / 7853 | 117.0 / 117.0 | 0% | 0% |
| Gemma-4-31B-IT | 7 | 1.0 | native | 0.7 | 100% | 100% | 2105 / 3639 / 3700 | 13040 / 17429 / 17917 | 295.3 / 408.3 | 0% | 0% |

## Decisions

Models suitable for full run: GPT-OSS-20B
Recommended reasoning setting/model: GPT-OSS-20B: medium; Gemma-4-31B-IT: native; Qwen3-32B: pending
Recommended max tokens/model: GPT-OSS-20B: 7,424 tokens (observed p95 6063); Gemma-4-31B-IT: 3,840 tokens (observed p95 3160); Qwen3-32B: pending
Feasible grid sizes: GPT-OSS-20B: 7, 11, 15; Gemma-4-31B-IT: 7, 11, 15
Observed p95 tokens/model: GPT-OSS-20B: 7,424 tokens (observed p95 6063); Gemma-4-31B-IT: 3,840 tokens (observed p95 3160); Qwen3-32B: pending
Observed p95 runtime/model: GPT-OSS-20B: 1371.7 s p95/trajectory; Gemma-4-31B-IT: 623.1 s p95/trajectory; Qwen3-32B: pending
Behavioral/API failures: see api_calls.csv; 27 trajectories remain
Estimated API cost so far: $0.2993 across complete trajectories with known pricing

The low-versus-medium rule deliberately prefers low. Medium is selected only when low fails the usability gate, medium passes it, and medium improves goal success, optimal-action rate, or invalid/failure rate by at least 10 percentage points on the same grid set. The recommended token cap is 120% of the observed per-call p95, rounded up to 256 tokens and never above 16k.

## Limits

These are small-sample engineering estimates, not paper-level capability comparisons. API trajectories contain no activations and cannot replace the local activation-bearing run. The Qwen3-32B candidate requires an account-specific Together dedicated endpoint name. Its monetary estimate remains unavailable until an hourly endpoint price is entered in the config.
