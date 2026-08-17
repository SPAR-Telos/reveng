# Planner-Optimality Audit

## Conclusion

The balanced step-reasoning run uses the correct DoorKey-aware field:
`optimal_actions_json`. It does not use `legacy_optimal_actions_json`, which is
a simplified goal-only calculation that does not model key/door mechanics.

Fresh `DoorKeyStateSolver` recomputation matches the stored action set for all
24 states.

## What The Planner Models

The planner runs BFS over reconstructed environment states and actual DoorKey
transitions. Its state includes:

- agent position
- key-carrying status
- whether the key remains on the grid
- whether a door is closed, open, or removed

For every candidate first action, it computes one transition plus the shortest
remaining distance to the goal. Every first action tied for minimum total
distance is included in `optimal_actions_json`.

This means that the planner:

- accepts multiple equal-length routes
- acquires the key when required by a shortest goal-reaching route
- models a locked door as impassable
- models automatic door removal when a key-carrying agent becomes adjacent
- permits traversal through the removed door cell
- does not force an unnecessary key/door route

## Audit Results

| Check | Result |
|---|---:|
| States audited | 24 |
| Stored action sets matching fresh recomputation | 24/24 |
| States where legacy goal-only and DoorKey-aware action sets differ | 11/24 |
| States with one optimal first action | 21/24 |
| States with two tied optimal first actions | 3/24 |
| Sustained transitions in tied-action states | 0/5 |
| States not carrying the key | 12/24 |
| Not-carrying states where every shortest path acquires the key | 12/12 |
| States with a visible closed door | 22/24 |
| Closed-door states where every shortest path opens the door | 22/22 |
| Closed-door states where every shortest path steps through its cell | 22/22 |

The three tied-action states have these accepted optimal action sets:

| Example | Optimal first actions |
|---|---|
| `together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_0_step_002` | `DOWN`, `LEFT` |
| `together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_20_step_008` | `LEFT`, `UP` |
| `together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_33_step_003` | `DOWN`, `RIGHT` |

## Remaining Sampling Confounds

The labels are correct, but the 24-state sample does not independently vary
all relevant task conditions:

- it contains no state where the key is visible but unnecessary
- it contains no state where a visible closed door can be bypassed on a
  shortest path
- failure states are mostly post-key: 9/12 carry the key
- non-failure controls are mostly pre-key: 9/12 do not carry the key

Within the failure states, sustained-transition rates are identical before and
after key acquisition in this small sample: `1/3` pre-key and `3/9` post-key.
This does not eliminate the phase imbalance from the overall
failure-versus-control comparison.

Future scaling should match or stratify by:

- pre-key versus post-key task phase
- whether a closed door is present and required
- shortest remaining distance
- number of tied optimal first actions

Per-state results are in `planner_optimality_audit.csv`.
