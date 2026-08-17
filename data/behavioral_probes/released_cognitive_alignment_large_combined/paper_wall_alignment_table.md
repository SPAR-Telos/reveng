# Released Wall Alignment Table

This table summarises the released size-7 wall-alignment slice after joining:
- public cognitive-map probe outputs
- behavioural wall-probe outputs
- A* optimal-action metadata
- wall-hit metadata

| Question | n pre | White-box pre | Black-box pre | Agreement pre | n post | White-box post | Black-box post | Agreement post | Δ white-box | Pre gap cases | Post gap cases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wall_down | 11 | 0.727 | 1.000 | 0.727 | 11 | 0.909 | 1.000 | 0.909 | +0.182 | 0 | 0 |
| wall_left | 11 | 0.636 | 1.000 | 0.636 | 11 | 0.636 | 1.000 | 0.636 | +0.000 | 0 | 0 |
| wall_right | 11 | 0.818 | 1.000 | 0.818 | 11 | 0.909 | 1.000 | 0.909 | +0.091 | 0 | 0 |
| wall_up | 11 | 1.000 | 1.000 | 1.000 | 11 | 0.909 | 1.000 | 0.909 | -0.091 | 0 | 0 |

Note: gap counts are zero on this released slice because these public states are mostly clean trajectory states rather than failure-focused states.