Does belief shifts change action along reasoning

Research question
How does commitment to an action emerge during reasoning, and how are action choice, action optimality, state beliefs, and uncertainty related?

This is not only a failure-step zoom-in. We track both cases where the recommended action is planner-optimal and cases where it is suboptimal. Failure states are useful diagnostic cases, but the project-level question is about action commitment in both successful and unsuccessful reasoning.

Subquestions:
1. test whether activation geometry monitors action commitment and action-optimality transitions;
2. test whether interpretable behavioral belief shifts explain or predict the same transitions;
3. test whether action selection is accompanied by lower action uncertainty but higher state-belief uncertainty;
4. intervene on supported belief shifts or reasoning chunks to test causality.

Experiment 1: Activation-Level Monitor of Action Commitment
This experiment asks whether generic hidden-state geometry changes when a prefix-elicited recommendation changes, becomes stable, or changes optimality.

setup:
24 DoorKey pilot states, 46 exact-matched states, and planned 185-state expansion; these are multiple grid states sampled from real on-policy GPT-OSS-20B DoorKey trajectories, not one hand-written grid;
local trajectory source: `data/hf/trajectories_key_door_100/trajectories_key_door`; public Project Telos artifacts can be linked when available, e.g. `https://huggingface.co/datasets/project-telos`;
current primary model: `openai/gpt-oss-20b`;
reasoning traces divided into at most 32 paragraph or sentence-like chunks, zero overlap for activation spans;
segmentation implementation: use `paragraph_or_sentence`; split by blank-line paragraphs, split single-line paragraphs at sentence punctuation, fall back to sentence or newline boundaries, and merge consecutive chunks if there are more than 32;
local teacher-forced inference means the local hookable model receives the fixed prompt plus the existing reasoning text and we extract activations during a forward pass; it does not generate the reasoning trace;
decoder layers 8, 15, and 23;
the activation at the last token of each chunk and the mean activation over the chunk.

The prefix-action dataset comes from fresh queries conditioned on prefixes of real on-policy trajectories. At prefix `t`, the grid state is fixed and the model sees the first `t` reasoning chunks, then we ask it for a next action. The elicited prefix action is not executed in the environment and is not necessarily the original trajectory action.

For each layer and representation type, measure:
update norm between adjacent reasoning chunks;
cosine similarity between step t and step t-1;
cosine between the activation step t and the mean activation avg(step_{0…t-1});
cosine similarity to an empirical anchor built from traces whose prefix-elicited actions remain optimal.

We track action commitment and optimality. Commitment can be defined as the first prefix where the elicited action matches the final full-trace action and remains stable. Optimality uses the DoorKey-aware BFS planner and accepts all tied optimal actions.

Experiment 2: Belief shifts, uncertainty, and action shifts
Given setup in Exp. 1, at every step, we query the model for its next action and beliefs about:

Belief
Measurements
Nearby state
Walls left, right, up, and down; key possession; door state
Locations
Goal, key, door, and agent coordinates
Action consequences
Whether hits a wall, results in holding the key, or leaves a door open
Uncertainty
Action entropy from logprobs or repeated samples; state-belief entropy from belief-probe answer distributions when available

Record every changed belief answer, including error onsets and recoveries. Measure whether each belief shift occurs before, during, or after action selection changes, commitment, optimality loss, or recovery. Compare behavioral and representational probes. If representational probes are temporarily unavailable, use Activation Oracle only as a model-specific auxiliary analysis, not as a substitute for GPT-OSS-20B probes.

Data: Pilot on existing 24 DoorKey states. Match failure and control states by key possession, visible closed door, remaining-distance bin, and number of optimal actions. Scale to the exact-matched 46-state cohort and then the 185-state expanded cohort.

What to measure: Analyze belief accuracy around action shifts, whether beliefs predict the next action's optimality, whether uncertainty changes around commitment, and whether planning, backtracking, checking, or correction sentences coincide with these shifts. This requires extra semantic classification of steps.


Modelling: 
The first run is observational. Use the strongest belief changes that precede sustained action changes as candidates for a later causal study.

Variable
Definition
A_t
Action recommended after reasoning step t
O_t
1 if A_t is planner-optimal
C_t
1 if A_t matches the final full-trace action and remains stable
U_action(t)
Action entropy over UP, DOWN, LEFT, RIGHT
U_state(t)
Average entropy over state-belief probes
t/T
Normalised reasoning progress




Target
Formula
Meaning
Action change
Y_change(t) = 1[A_t != A_(t+1)]
Recommended action changes at the next reasoning step
Optimality loss
Y_loss(t) = 1[O_t = 1 and O_(t+1) = 0]
Action becomes suboptimal
Optimality recovery
Y_recovery(t) = 1[O_t = 0 and O_(t+1) = 1]
Action becomes optimal again
Action uncertainty change
Delta U_action(t) = U_action(t+1) - U_action(t)
Change in action uncertainty
State uncertainty change
Delta U_state(t) = U_state(t+1) - U_state(t)
Change in state-belief uncertainty

Action-event models use `Y` as the dependent variable, where `Y` is one of:
`Y_change`, `Y_loss`, or `Y_recovery`.

`logit P(Y=1) = t/T`
controls for whether action events happen simply because reasoning is early or late.

`logit P(Y=1) = t/T + belief-error answers`
tests whether wrong beliefs about walls, key possession, or state variables predict action change or optimality shifts.

`logit P(Y=1) = t/T + chosen-direction wall error + chosen-action collision error + blocked-action reports`
tests whether recommended actions conflict with the model's own reported beliefs.

Uncertainty models use uncertainty change as the dependent variable:

`Delta U_action(t) = t/T + commitment_onset(t) + O_t + belief-error answers`

`Delta U_state(t) = t/T + commitment_onset(t) + O_t + belief-error answers`

where `commitment_onset(t)=1` if the action first becomes stable between step
`t` and `t+1`. The hypothesis is:

`commitment_onset(t)` predicts lower `Delta U_action(t)` but higher
`Delta U_state(t)`.

## Experiment 3: Counterfactual Sentence Interventions

| Sentence condition | Modification |
|---|---|
| Original | Retain the original sentence |
| Replacement | Substitute a semantically different sampled sentence |
| Deletion | Remove the sentence |
| Belief insertion | Insert a correct or incorrect state-belief statement |

Measure changes in final action optimality, action choice, uncertainty, and downstream beliefs. Begin with about 30 sentences and 10 continuations per condition. For a candidate pair of beliefs `B1` and `B2` that precedes an action transition, use a four-condition factorial intervention:

| Condition | Intervention |
|---|---|
| Control | Preserve both original beliefs |
| Change B1 only | Insert or replace text so only `B1` states the selected correct or counterfactual value |
| Change B2 only | Change only `B2` |
| Change both | Change `B1` and `B2` together |

For example, `B1` could be the reported wall state in the recommended direction and `B2` the predicted consequence of taking that action. The four conditions would preserve both reports, change only the wall report, change only the consequence prediction, or change both.

For each condition, sample continued reasoning and measure downstream belief reports, action identity, action optimality, action uncertainty, and state-belief uncertainty. Let `mu_bc` be the probability of the target downstream outcome when `B1=b` and `B2=c`, where zero means preserved and one means changed. The interaction effect is:

`Delta_interaction = mu_11 - mu_10 - mu_01 + mu_00`.

A nonzero interaction means the joint intervention has an effect not explained by adding the two individual intervention effects. A positive value means the joint intervention increases the target outcome more than expected; a negative value means it suppresses it.

Because inserting text changes the reasoning context rather than directly setting a latent belief, rerun the corresponding belief probes as manipulation checks. If the intended belief report does not change, interpret the result as a sentence intervention rather than a belief intervention.

Alternative to experiment 3: Can we swap the activation at where the behavioural belief about a variable changes to a previous step and ask at that previous step what the action would be? We then observe if the elicited action at that previous step aligns with the transition target, stays the same, or transitions to something else.
