# Step Reasoning Drift Experiment

This experiment asks whether revealing progressively more of a generated
reasoning trace changes a newly elicited next-action recommendation from a
planner-optimal action to a planner-suboptimal action, and whether hidden-state
geometry changes near that transition.

The concrete follow-up plan connecting action transitions, belief shifts,
semantic reasoning-move labels, and counterfactual sentence interventions is
in `docs/reasoning_belief_shift_experiment_proposal.md`.

It is an observational diagnostic. It does not establish that a reasoning
chunk caused the original trajectory's action or failure.

## State Sets

The runner uses the same trajectory-derived rows as the revealed-CoT scaling
experiments:

- `focused_failure`: the current focused failure set selected from
  `trajectory_selection_candidates.csv`.
- `trajectory_candidates` with `trajectory_slice_type=primary_failure_mode:short_loop`.
- `trajectory_candidates` with `trajectory_slice_type=non_failure_controls`.
- `trajectory_candidates` with `trajectory_slice_type=balanced_failure_modes`, when a
  mixed failure-category comparison is needed.
- `trajectory_candidates` with
  `trajectory_slice_type=balanced_failure_modes_with_controls`, which adds
  independent non-failure controls to the paired contexts selected with the
  balanced failure states.

These rows come from local GPT-OSS-20B DoorKey trajectories under
`data/hf/trajectories_key_door_100/trajectories_key_door`. They are not the
same trajectories as every public pre/post activation artifact, so uploads
should preserve `trajectory_id`, `step_index`, `example_id`, and
`source_dataset` for alignment.

## Behavioral Prefix Diagnostic

For each trajectory state, the `analysis` text is split into immutable
sentence spans. The runner supports two analysis units. `packed_chunk` packs
adjacent sentences into at most 32 units. `sentence` reads the exact spans
from `sentences.csv` and uses every reasoning sentence without repacking.
For every prefix length `k`, the runner independently asks the black-box model
for a next action after revealing units `1` through `k`.

Use `max_reasoning_steps` only with `analysis_unit=packed_chunk` to
deterministically pack adjacent sentences. With `analysis_unit=sentence`,
`max_reasoning_steps` must be unset and final-action rows are excluded.
Exhausted black-box request failures are
recorded as `INVALID` prefixes with `query_error`; they do not abort the run.

### Exact Chunking Procedure

The current canonical chunker is `doorkey_sentence_v1`. It uses:

- `segmentation_mode=paragraph_or_sentence`
- `max_reasoning_steps=32`

The sentence segmentation procedure is:

1. Split at blank paragraph breaks.
2. Split after sentence-final `.`, `?`, or `!` followed by whitespace.
3. Do not split inside coordinates, decimals, common abbreviations,
   numbered action prefixes such as `1. UP`, or action JSON spans.
4. Separate an action JSON as `final_action` only if it is terminal. Internal
   action JSON mentions remain in reasoning chunks and are flagged.
5. Save fixed character spans for every sentence.
6. Pack adjacent reasoning sentences into at most 32 analysis chunks using
   approximately equal cumulative word-count boundaries. Sentence spans are
   preserved, and no remainder is forced into the final chunk.

The validation command is:

```bash
uv run reveng-cli validate_doorkey_chunking_strategy \
  --trajectory-dir data/hf/trajectories_key_door_100/trajectories_key_door \
  --output-dir data/behavioral_probes/doorkey_chunking_validation
```

The current full local validation covers 95 trajectory files and 1,276
reasoning traces with zero span, overlap, action-JSON, numbered-list, or
analysis-overflow errors. The report is in
`data/behavioral_probes/doorkey_chunking_validation/chunking_report.md`.
The exported sentence table is
`data/behavioral_probes/doorkey_chunking_validation/sentences.csv`, and the
packed analysis-chunk table is
`data/behavioral_probes/doorkey_chunking_validation/chunked_trajectories.csv`.

This is not a field-standard semantic reasoning-step annotation:

| Unit | Boundary rule | Relation to this experiment |
|---|---|---|
| Token-level | Every model token | Finer-grained but much noisier and more expensive; a semantic change can span many tokens. |
| Fixed token window | Every fixed number of tokens | Comparable lengths, but boundaries need not correspond to statements or decisions. |
| Explicit solution step | Newlines, numbering, or model-authored step markers | Common in mathematical/process-supervision work; the DoorKey traces do not consistently provide such markers. |
| Semantic reasoning move | Human or classifier labels such as state reconstruction, route evaluation, correction, backtracking, or commitment | Not currently annotated. This is required before claiming that a detected transition corresponds to backtracking or an "aha" moment. |
| Current sentence | Sentence-level punctuation or paragraph boundary, with protected false-boundary patterns | A reproducible fine-grained text unit for labels and interventions. |
| Current analysis chunk | Deterministic packed view over adjacent sentences, capped at 32 | The unit used for the original 8-state activation-backed analysis. |
| Sentence-prefix analysis | Every canonical reasoning sentence, with the final action stored separately | The finer unit used by the controlled behavioral rerun of Experiments 1 and 2. |

In packed mode, a detected action transition can occur inside a merged chunk, and one chunk
can contain several semantic reasoning moves. A follow-up analysis should
classify the transition chunk and its neighbors into semantic move types,
preferably with human validation. The current task's reasoning is simple, so
correction, route reconsideration, backtracking, and action commitment are
more plausible categories than strong "aha moment" claims.

### How A Prefix-Elicited Action Is Produced

The term **prefix-elicited action** means the output of a new black-box query.
It is not the original trajectory action, and it is not executed in the
environment.

For each prefix `k`, the runner sends TogetherAI
`together_ai/openai/gpt-oss-20b` a fresh user prompt containing:

- the DoorKey instructions and direction definitions
- the unchanged current grid state
- whether the agent carries the key
- the original reasoning text through analysis unit `k`
- an instruction to return exactly
  `{"action": "<UP|DOWN|LEFT|RIGHT>"}`

The completed run used temperature `0.0`, seed `0`, and low reasoning effort.
The response is first parsed as the requested JSON action object. If JSON
parsing fails, the parser falls back to the first standalone
`UP|DOWN|LEFT|RIGHT` direction in the response. An unparseable or exhausted
request is recorded as `INVALID`.

The behavioral outputs are:

- `action_is_optimal(k)`: whether the action newly elicited after prefix `k` is
  in `optimal_actions_json`. Optimality describes the elicited action, not the
  reasoning text itself.
- `action_matches_final(k)`: whether the prefix action matches the original
  full-trace final action.
- `wrong_turn_step_strict`: internal field name for the first optimal-to-suboptimal
  recommendation transition where all valid later prefixes remain suboptimal.
- `wrong_turn_step_majority`: internal field name for the first
  optimal-to-suboptimal recommendation transition where the fraction of valid
  later prefixes that remain suboptimal is at or above `persistence_threshold`.
  The default threshold is `0.5`. Reports call this a **sustained
  optimal-to-suboptimal recommendation transition**.
- `commitment_step`: first prefix whose action matches the final action and
  stays matched.
- `recovery_step`: first later prefix where a non-optimal sequence becomes
  optimal again.

The `0.5` persistence threshold is heuristic. Event-centered geometry uses the
mean of the preceding three selected analysis units as its reference, which is also heuristic.
No threshold is applied to the continuous geometry metrics themselves.

### Sentence-Boundary Rerun

The controlled 8-state sentence rerun uses 654 reasoning sentences and one
empty-prefix baseline per state, for 662 prefix positions. It keeps the model,
prompts, greedy action query, 10 action samples at temperature 0.7, and belief
queries unchanged. Comparisons with packed mode use the fraction of reasoning
characters revealed rather than raw unit indices.

The previously processed Weisheng activations are averages over the original
packed spans and cannot be split into sentence representations.

The raw stride-2 activations are now available under
`outputs/experiment1_activation_monitor/weisheng_raw_stride2_activations`.
They have been mapped to canonical sentence spans under
`outputs/experiment1_activation_monitor/weisheng_sentence_averaged_activations`.
All 654 reasoning sentences in the 8 activation-backed states have a
sentence-mean vector at each of 12 even layers. These are means over available
stride-2 sampled tokens, not guaranteed sentence-final-token activations.

### What Planner-Optimal Means

`optimal_actions_json` is not produced by the older goal-only grid-distance
helper. It is produced by `DoorKeyStateSolver`, which reconstructs the current
rendered state and searches over actual DoorKey environment transitions with
BFS.

The source dataset also preserves `legacy_optimal_actions_json` for comparison.
That legacy field treats every non-wall symbol as passable and does not model
key acquisition or locked-door transitions. It must not be used as the
planner-optimal label for this experiment. The step-reasoning runner loads
`optimal_actions_json`.

The search state includes:

- agent position, represented in the rendered grid
- whether the agent is carrying the key
- whether the key remains on the grid
- whether a door is closed, open, or removed

The transition function implements the environment mechanics: stepping onto
the key acquires it and removes it from the grid; a closed door is impassable;
and carrying the key while becoming adjacent to a locked door removes it.
Agent orientation is not included because experiment actions are absolute
cardinal moves and each action directly sets orientation.

For each possible first action, the solver computes one step plus the BFS
shortest remaining distance. `optimal_actions_json` contains every first
action tied for the minimum total distance. It therefore:

- accepts multiple equal-length first actions
- directs the agent toward the key only when acquiring it is required by a
  shortest route to the goal
- accounts for opening and traversing a required door
- does not force a key/door route when the goal has a shorter bypass

The completed balanced run was audited by freshly recomputing all 24 action
sets:

- stored/recomputed mismatches: `0/24`
- states where the simplified legacy action set differs from the DoorKey-aware
  action set: `11/24`
- states with one optimal first action: `21/24`
- states with two tied optimal first actions: `3/24`
- sustained transitions occurring in tied-action states: `0/5`
- states not carrying the key: `12/24`; every shortest path acquires the key
  in all 12
- states with a visible closed door: `22/24`; every shortest path opens and
  steps through that door in all 22

Thus, the action labels correctly handle ties and DoorKey mechanics. However,
the sample does not include controls where a visible key is unnecessary or a
closed door is bypassable. It is also phase-imbalanced: failure states are
mostly post-key (`9/12` carrying), while non-failure controls are mostly
pre-key (`9/12` not carrying). Future comparisons should match or stratify by
key-carrying status, whether a closed door is present/required, shortest
distance, and the number of tied optimal first actions.

The audit is saved in
`data/behavioral_probes/step_reasoning_drift_balanced_v1/planner_optimality_audit.csv`.

## Activation Extraction

Activation extraction never uses TogetherAI. The local hookable Hugging Face
model performs a teacher-forced causal forward pass and does not generate a
new continuation or action. Forward hooks capture decoder-layer outputs at
only the requested layers. Long prompts are processed with a KV cache in
chunks controlled by `forward_chunk_size`, which avoids full-sequence
eager-attention OOMs.

Two activation prompt modes are available:

- `revealed_prompt`: feeds the same DoorKey prompt used in the behavioral
  revealed-reasoning diagnostic, with the full reasoning trace inserted as text.
- `original_trace`: feeds the original trajectory prompt followed by the
  generated GPT-OSS analysis/final output text. This is the closer match to the
  existing pre/post activation-release convention.

For each reasoning step and layer, the runner saves:

- `last_token.pt`: the decoder-layer output at the inclusive endpoint of the
  chunk's mapped token span.
- `mean_pool.pt`: the mean decoder-layer output over every token in that
  inclusive span.

The runner also saves the final token immediately before reasoning as
`reasoning_step_000`. Saved vectors are cloned before serialization so each
file contains only its compact `(hidden_size,)` tensor rather than the full
backing sequence storage.

The completed balanced run used `activation_prompt_mode=original_trace`.
Consequently, the local input was the original trajectory prompt followed by
the original GPT-OSS output text, including its analysis and final answer.
It captured layers 8, 15, and 23.

### Balanced Run v1 Token-Boundary Limitation

Balanced run v1 mapped a character chunk to every token whose character offset
overlapped that chunk. GPT-style tokens can include leading whitespace plus
the next word, while the textual chunker includes trailing whitespace.
Therefore, adjacent mapped token spans overlap at 549 of 695 boundaries
(79.0%).

For one sustained-transition example:

- chunk 13 maps to tokens `1350:1442`; token `1442`, saved as its
  `last_token.pt`, decodes to ` But`
- chunk 14 also begins at token `1442`

Thus, the chunk-13 last-token vector already includes the lexical start of
chunk 14. Mean-pooled vectors are less dominated by this single boundary
token, but they are also not strictly non-overlapping. Balanced run v1 must be
regenerated with unique, non-overlapping token assignment before last-token
geometry is treated as clean chunk-boundary geometry.

See `data/behavioral_probes/step_reasoning_drift_balanced_v1/method_examples.md`
for the concrete transition chunks, raw action outputs, prompt shape, and
decoded activation tokens.

The index file `step_activation_rows.csv` records:

- `activation_schema_version`
- `activation_prompt_mode`
- `example_id`, `trajectory_id`, `step_index`
- `source_dataset`, `selection_stage`, `failure_category`
- `source_trajectory_path`
- `local_model_name_or_path`
- `prompt_text_path`, `analysis_text_path`
- reasoning-step text and character spans
- token spans
- paths to `last_token.pt` and `mean_pool.pt`

This is intended to be uploadable as a step-indexed activation dataset.

## Completed Balanced Run

The first balanced three-layer run is under
`data/behavioral_probes/step_reasoning_drift_balanced_v1/`.

- `run_report.md`: design, data quality, preliminary interpretation, and limitations
- `method_examples.md`: concrete chunk, action-query, and activation-token example
- `step_activation_rows.csv`: activation index
- `activations/`: compact BF16 activation tensors
- `prefix_action_rows.csv`: step-wise behavioral evaluations
- `trajectory_wrong_turn_summary.csv`: per-state transition labels; column names
  retain the original internal `wrong_turn` terminology
- `state_outcome_accounting.csv`: exhaustive, plain-language outcome
  classification for every state
- `geometry_rows.csv`: step-level geometry metrics; `aligned_change` is the
  internal field name for overall-direction agreement
- `wrong_turn_geometry_events.csv`: event-centered sustained-transition
  comparisons; column names retain the original internal terminology
- `prefix_query_checkpoints.jsonl`: resumable checkpoints after each prefix-action query
- `example_checkpoints.jsonl`: completed per-state checkpoints

## Hardware Notes

OpenAI's gpt-oss-20b is a 21B-parameter, 3.6B-active-parameter model. The
official Transformers guide reports about 16GB VRAM with default MXFP4 and
about 48GB memory for BF16. On the current 1x L4 workspace, PyTorch CUDA sees
the GPU and a CUDA matmul succeeds, but `nvidia-smi`/NVML is not reliable. Use
Transformers' quantized GPT-OSS loading path with `--device-map auto` and
`--torch-dtype auto`; do not use full BF16 loading on this GPU. Keep enough
local Hugging Face cache space for the model snapshot before running extraction.
