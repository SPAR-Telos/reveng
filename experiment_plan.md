# Do belief shifts change action along reasoning?

Tommy

## Research question

When a reasoning model reads more of its own generated reasoning trace, how does it commit to an action, and how are action commitment, action optimality, belief state, and uncertainty related? Is this relation causal?

Subquestions:  
1\. test whether activation geometry monitors action transitions;  
2\. test whether interpretable behavioral belief shifts explain or predict the  
   same transitions;  
3\. intervene on supported belief shifts or reasoning chunks to test causality.

Deprecated: Can we detect when GPT-OSS-20B's reasoning stops supporting an optimal action,  
and can that change be explained by shifts in its state beliefs?

## Experiment 1: Activation-Level Monitor of Action Commitment and Recommendation Changes

This experiment asks whether generic hidden-state geometry changes (1) when action commitment happens, (2) when a prefix-elicited recommendation action changes, and when (3) the elicited recommendation changes optimality.

setup:

- Models: openai/gpt-oss-20b, and Qwen/Qwen3-8B (has open source LoRA Activation Oracle)  
- 24 DoorKey states: 12 failure states and 12 controls. These are multiple grid states sampled from real on-policy GPT-OSS-20B DoorKey  
-  719 prefix-action evaluations, of which 718 parsed successfully;  
- reasoning traces divided into at most 32 paragraph or sentence-like chunks, zero overlap;  
  - split by blank-line paragraphs, split single-line paragraphs at sentence punctuation, fall back to sentence or newline boundaries, and merge consecutive chunks if there are more than 32\.  
- Offline prefix conditioning-: The reasoning traces and grid states come from on-policy GPT-OSS-20B DoorKey trajectories. The action labels are elicited offline: not executed,   
- do not create new on-policy rollout. For each fixed trajectory state, the local model receives the fixed prompt concatenated with existing reasoning text and we extract activations during a forward pass;  
- decoder layers 8, 15, and 23;  
- the activation at **the last token** of each chunk and **the mean activation** over

  the chunk;

For each layer and representation type, measure:

- update norm between adjacent reasoning chunks;  
- cosine similarity between step t and step t-1 (per-step consistency);  
- cosine between the activation step t and the mean activation avg(step\_{0…t-1});  
- cosine similarity to an empirical anchor built from traces whose prefix-elicited actions remain optimal.

We track action commitment and optimality. Commitment can be defined as the first prefix where the elicited action matches the final full-trace action (oracle) and remains stable. Optimality uses the DoorKey-aware BFS planner and accepts all tied optimal actions.

Deferred representational extensions, including a post-commitment activation
monitor and attention analysis, are indexed in
`docs/deferred_reasoning_ideas.md`. They are not part of the current Experiment
1 evidence.

## Experiment 2: Belief shifts and action shifts

Given setup in Exp. 1, at every step, we query the model for its next action and beliefs about:

| Belief | Measurements |
| :---- | :---- |
| Nearby state | Walls left, right, up, and down; key possession; door state |
| Locations | Goal, key, door, and agent coordinates |
| Action consequences | Whether hits a wall, results in holding the key, or leaves a door open |

Record every changed belief answer, including error onsets and recoveries. Measure whether each belief shift occurs before, during, or after the action shift. Compare behavioral and representational probes. If the representational probes are temporarily unavailable, we resort to activation oracles.

Data: Pilot on the existing 24 DoorKey states. Match failure and control states by key possession, visible closed door, remaining-distance bin, and number of optimal actions.

What to measure: Analyze belief accuracy and uncertainty on states and actions around action shifts, whether beliefs predict the next action's optimality, whether uncertainty changes around commitment and optionally whether planning, backtracking, checking, or correction sentences coincide with these shifts. (This requires extra semantic classification of steps)

### Modelling: 

The first run is observational. Use the strongest belief changes that precede sustained action changes as candidates for a later causal study.

| Variable | Definition |
| :---- | :---- |
| A\_t | Action recommended after reasoning step t |
| O\_t | 1 if A\_t is planner-optimal |
| C\_t | 1 if A\_t matches the final full-trace action and remains stable |
| U\_action(t) | Action entropy over UP, DOWN, LEFT, RIGHT |
| U\_state(t) | Average entropy over state-belief probes |
| t/T | Normalised reasoning progress |

| Target | Formula | Meaning |
| :---- | :---- | :---- |
| Action change | Y\_change(t) \= 1\[A\_t \!= A\_(t+1)\] | Recommended action changes at the next step |
| Optimality loss | Y\_loss(t) \= 1\[O\_t \= 1 and O\_(t+1) \= 0\] | Action becomes suboptimal |
| Optimality recovery | Y\_recovery(t) \= 1\[O\_t \= 0 and O\_(t+1) \= 1\] | Action becomes optimal again |
| Action uncertainty change | Delta U\_action(t) \= U\_action(t+1) \- U\_action(t) | Change in action uncertainty |
| State uncertainty change | Delta U\_state(t) \= U\_state(t+1) \- U\_state(t) | Change in state-belief uncertainty |

Action changes:  
Y can be either  Y\_change, Y\_loss, or Y\_recovery.

Y=t/T controls for if action transitions happen simply because reasoning early or late

t/T \+ belief-error answers if wrong beliefs about walls, key possession, or state variables predict action change.

t/T \+ chosen-direction wall error \+ chosen-action collision error \+ blocked-action reports if model recommend actions conflict with own reported beliefs.

### Factorized belief-action model

The belief probes measure separate task factors rather than one global state
score. We first fit additive models with one variable per belief error. We then
test whether structured combinations of task factors improve prediction of
action changes.

Use only task-motivated combinations:

| Combination | Definition | Question |
| :---- | :---- | :---- |
| Chosen move blocked | chosen-direction wall report \* chosen-action hit-wall report | Does the model recommend an action that its own reports imply is blocked? |
| State transition disagreement | chosen-direction wall error \* chosen-action consequence error | Do inconsistent beliefs about the same transition predict action shifts? |
| Key-door dependency | key-possession error \* door-state error | Do errors in the precondition and door state jointly predict action shifts? |

Compare the additive belief-error model, the action-conditioned model, and the
factorized interaction model on held-out trajectories. This tests a behavioral
version of factorized state representation: whether separately measured belief
factors combine in a task-structured way that explains action commitment,
optimality loss, or recovery. A representational version requires white-box
probes or activation-oracle readouts for the same factors and then tests whether
their activation directions or subspaces are separable.

Hypothesis about uncertainty: commitment reduces action uncertainty while increasing or changing state-belief uncertainty.  We measure

Delta U\_action(t) \= t/T \+ commitment\_onset(t) \+ O\_t \+ belief-error answers

Delta U\_state(t) \= t/T \+ commitment\_onset(t) \+ O\_t \+ belief-error answers

where commitment\_onset(t)=1 if the action first becomes stable between step t and t+1.

Deferred observational extensions ask whether current-state and transition
belief patterns predict how much reasoning continues after commitment. The
prespecified tests and the boundary between a DoorKey-specific monitor and a
cross-task claim are recorded in `docs/deferred_reasoning_ideas.md`.

## (Early Ideation) Experiment 3: Counterfactual sentence interventions

For sentences near an action shift, compare continued reasoning after:

- retaining the original sentence  
- replacing it with a semantically different sampled sentence  
- deleting it  
- inserting a correct or incorrect state belief

Measure changes in final action optimality, action choice, and downstream  
beliefs. Begin with about 30 sentences and 10 continuations per condition.  
For candidate belief pairs, include a factorial intervention that changes each  
belief separately and both together. This tests whether the joint causal effect  
exceeds the sum of the individual effects.

Alternative to experiment 3: Can we swap the activation at where the behavioural belief about a variable changes (which we believe to correlate/cause the action transition) to a previous step and ask at that previous step what the action would be? We then observe if the elicited action at that previous step aligns with the transition target, stays the same, or transitions to something else.

Weisheng:

1) Create Dataset:  
   Use the key-door setting.  
     
   Algo:  
   M \= number of rooms per side, M \= 2, 3   
   N \= number of grids we are going to use for each number of room, N \= 100  
     
   For each grid labeled by (M,N), for each possible starting state (position, has key, door open) generate 5 times one step data. Using temperature \= 0.7.  
     
   Need to make sure each random grid is different from other grids.   
     
   Raise the max number of tokens to have less failure due to token cap.   
     
2) Training probes:  
     
   Train kas\_key and door\_open probes, following the same setting in the previous paper.   
   Train optimal\_action, output\_action probes along CoT.  
     
   For key and door probes: distinguish pre and post reasonings.   
     
   Along CoT means:  
   Use advanced LLM to find mistake positions in CoT, compare behaviors of probes near this mistake position  
     
   Every 20 tokens, train a probe for actions, see how the accuracy changes.  
     
   It turns out that at the end of the CoT the final choice of action is stated. We could look at the attention matrix to see where this action token pays attention to. It should pay more attention to the tokens after the mistake in the CoT for the wrong case. And what is the pattern for the correct case ?   
     
   measure the score in [https://arxiv.org/html/2509.11569v1](https://arxiv.org/html/2509.11569v1) to see whether it differs for the correct case or wrong case. And also before mistake and after mistake in the CoT.

   Try to construct the activation manifold as in [https://arxiv.org/html/2605.12412v1](https://arxiv.org/html/2605.12412v1), using the stream of activations in the CoT to see whether correct and wrong CoT has different low dimension structures. 

   

   

   

   

   

   

**Sıla**   
**—**

**Aims and Experiments Needed:**

- For black-box vs. white-box probe comparison: Work on the white box experiments for the aim of structurally testing for each state variable, do behavioral and representational probes agree, and where they disagree, does the disagreement carry information about action optimality? 

  To do so will first fix the state description

  Then we also need to extend coverage and decide on the concrete grid dimension/representation 

  Train and test representational probes on remaining features

- Provide the one step IRL analyses that are given in the paper based on the grids decided on previous experiments.  
- Also, elaborating more on why one step IRL represents a value function in the paper.  
- Alternative and Interpretable IRL method  
- We may also discuss how to implement/steps for cross-model replication?  
- Testing whether the model assigns a value/preference over states.  
- State uncertainty and gap over reasoning traces  
- Action distribution depending on g\_theta over cot  
- How to divide CoT  
- Uncertainty over trajectory  
- How sure the model is on transition dynamics (can we train a probe for this)

**Defining:**

1) **IRL:**  
   1) Also, elaborating more on why one step IRL represents a value function in the paper.  
   2) Alternative and interpretable IRL method  
2) **How to divide CoT:**

**Concrete Experiment Definitions**

1) **Testing whether the model assigns a value/preference over states:** 

   –Depends on part 1 in defining section

2) **State uncertainty and gap over reasoning traces:**  
   1) Using Tommy’s suggestion for dividing the reasoning steps into chunks apply gap and uncertainty calculations for each chunk

   

3) **Action distribution depending on g\_theta over cot:**  
   1) In addition to tests in 2; calculate how action distribution entropy and the distributions over reasoning steps change

   

4) **Uncertainty over trajectory:**  
   1) Evaluate all the metrics in 2 and 3 for whole trajectory rather than one reasoning trace

   

5) **How sure the model is on transition dynamics**  
   1)  Can we train a probe for this or should we?
