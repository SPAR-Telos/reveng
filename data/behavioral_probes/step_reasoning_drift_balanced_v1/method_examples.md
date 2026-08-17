# Step Reasoning Drift: Concrete Method Example

Example state: `together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_15_step_000`

This state has planner-optimal action `RIGHT`. The original trajectory action is `DOWN`.

## Chunk Boundary And Prefix-Elicited Actions

### Chunk 13

```text
But maybe we can bypass door by going to row 4 col 7 (open) and then up to row 1 col 7? Let's check connectivity: from row 1 to goal at (1,7) we need to reach (1,7). We can go from (1,1) down to row 2,3,4,5,6,7 then across to column 7, then up to row 1?
```

After revealing through chunk 13, the independent TogetherAI query returned:

```json
{"action":"RIGHT"}
```

Parsed action: `RIGHT`; planner-optimal: `True`.

### Chunk 14

```text
But there are walls at column 4 (#). Let's examine the grid for columns 5,6,7. In row 1, columns 5,6,7 are open.
```

After revealing through chunk 14, the independent TogetherAI query returned:

```json
{"action":"DOWN"}
```

Parsed action: `DOWN`; planner-optimal: `False`.

The action therefore changes from planner-optimal `RIGHT` after chunk 13 to planner-suboptimal `DOWN` after chunk 14. This is a prefix-query transition; neither chunk is automatically assigned a semantic label such as backtracking or verification.

## Saved Activation Tokens In Balanced Run v1

- Chunk 13: inclusive token span `1350:1442` (93 tokens); saved `last_token.pt` uses token `1442`, decoded as `' But'`; `mean_pool.pt` averages all tokens `1350` through `1442`.
- Chunk 14: inclusive token span `1442:1480` (39 tokens); saved `last_token.pt` uses token `1480`, decoded as `' In'`; `mean_pool.pt` averages all tokens `1442` through `1480`.

The original artifact is retained for provenance, but should not be used for
step-level geometry. It assigned every token whose character offset overlapped
a chunk, so GPT-style leading-whitespace tokens could belong to both adjacent
chunks. Across the original run, 549 of 695 adjacent chunk boundaries had
overlapping token spans.

The corrected artifact is:

`data/behavioral_probes/step_reasoning_drift_balanced_v1_corrected_activations`

It assigns shared boundary tokens to the later chunk. For this example:

- chunk 13 uses inclusive token span `1350:1441` and ends on `?`;
- chunk 14 uses inclusive token span `1442:1479`, begins with `But`, and ends on `.`;
- chunk 15 begins at token `1480` with `In`.

The corrected artifact contains all 24 states and 2,157 activation-index rows
at layers 8, 15, and 23, with zero adjacent-span overlaps.

## Prefix Action Prompt Shape

```text
# Instructions
You are controlling an agent in a 9x9 fully observable DoorKey GridWorld.
...
Choose the next move that best advances toward the goal while respecting the DoorKey mechanics in the current state.

# Inputs

Current grid state:
<the unchanged grid for this state>

Agent status:
- Carrying key: <true|false>

Reasoning trace available so far:
<original reasoning text through chunk k>

Respond with exactly a JSON object of the form {"action": "<UP|DOWN|LEFT|RIGHT>"}.
Do not include any extra text before or after the JSON.
```

Each prefix is queried independently through TogetherAI using `together_ai/openai/gpt-oss-20b`, temperature `0.0`, seed `0`, and low reasoning effort. The returned action is parsed from JSON, with a direction-word regex fallback. It is not executed in the environment.
