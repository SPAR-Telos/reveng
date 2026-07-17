# Revealed-CoT Metric Glossary

This note clarifies the black-box metrics used in the gradual revealed-CoT experiments.

## What is actually probed?

The current revealed-CoT pipeline probes two different objects:

1. **Adjacent-wall state belief**
- For each direction (`left`, `right`, `up`, `down`), the model is asked whether the adjacent cell in that direction is blocked.

2. **Next action the model would take**
- Given the same partially revealed reasoning trace, the model is asked which action it would take next.

The current pipeline does **not** ask a separate question of the form:
- "Which action is optimal in this state?"

## Column meanings in `gradual_blackbox_reasoning_table.csv`

### `blackbox_wall_accuracy`
- Adjacent-wall state-belief accuracy against the ground-truth environment label.

### `action_greedy_modal_optimality_rate`
- The revealed-CoT **next-action** query asks the model what action it would take.
- This column reports whether that **probed next action** is in the state's optimal action set.
- Clearer name: `probed_next_action_in_optimal_action_set_rate`.

### `local_belief_action_gap_rate`
- This is **not** a general "local" metric.
- It is restricted to rows where the wall question targets the **same direction as the model's chosen action**.
- Among those rows with a valid `yes/no` wall answer, it reports the rate at which the probe says the chosen direction is blocked (`yes`).
- Clearer name: `chosen_direction_wall_belief_contradiction_rate`.

### `astar_conditioned_gap_rate`
- This is also restricted to rows where the wall question targets the **same direction as the model's chosen action**.
- It only counts rows where the wall probe answer is valid **and correct**.
- Among those rows, it reports the rate at which the model's chosen action is still **suboptimal**.
- Clearer name: `chosen_direction_wall_belief_correct_but_action_suboptimal_rate`.

## Mapping to the action-comparison concepts discussed later

Among the concepts:

1. `probed optimal action, evaluated against optimal action`
2. `probed optimal action, evaluated against model action`
3. `probed model/next action, evaluated against model action`

the current gradual revealed-CoT table contains **none of (1) or (2)**.

It contains only:

- **Probed next/model action, evaluated against optimal action**
  - `action_greedy_modal_optimality_rate`

The gap metrics are different:

- `chosen_direction_wall_belief_contradiction_rate`
  - probes **adjacent-wall state belief**, then asks whether that belief says the model's chosen direction is blocked.

- `chosen_direction_wall_belief_correct_but_action_suboptimal_rate`
  - probes **adjacent-wall state belief**, conditions on that belief being correct for the chosen direction, then asks whether the chosen action is suboptimal.

## What would need to be added for the missing concepts?

To get:

- `probed optimal action, evaluated against optimal action`
- `probed optimal action, evaluated against model action`

you would need a new prompt that explicitly asks:

- "Which next action is optimal in this state?"

To get:

- `probed next/model action, evaluated against model action`

you would need an explicit external action reference, for example:

- the model's full-trace action at `100%` reveal, or
- the action in the original stored trajectory.
