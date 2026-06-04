# Activation-Oracle Prompt Families

This file centralizes the prompt families used for the DoorKey activation-oracle
experiments.

Code source:
- [activation_oracle_prompts.py](/root/reveng/src/reveng/experiments/activation_oracle_prompts.py)

## 1. Local Grid-State Readout

Purpose:
- Read out local state information from activations.

Template:

```text
Given these activations, is the cell immediately {DIRECTION} of the agent a wall? Answer with exactly one of: yes, no, unknown.
```

Current use:
- This is the prompt family used for the AO wall-state rows generated from the
  paper slices.

## 2. Model-Action Readout

Purpose:
- Read out the action the model itself would most likely choose from
  activations.

Template:

```text
If the model had to answer now based only on these activations, which next action would it most likely output? Answer with exactly one of: UP, DOWN, LEFT, RIGHT, unknown.
```

Current use:
- This is the prompt family used for the AO action rows generated from the
  failure-focused revealed-CoT slice.
- The current optimal-action comparison reuses this action readout and checks
  whether the decoded action belongs to the stored `optimal_actions_json` set.

## 3. Optimal-Action Readout

Purpose:
- Directly ask whether activations support a shortest-path-optimal next action.

Template:

```text
Given these activations and the current DoorKey state, which next action is shortest-path optimal? Answer with exactly one of: UP, DOWN, LEFT, RIGHT, unknown.
```

Current use:
- Defined centrally for future AO runs.
- Not yet run in the current paper-aligned AO batch.
