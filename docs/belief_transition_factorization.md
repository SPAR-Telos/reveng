# Belief Indicators and Action Transitions

For the proposed distinction between probe-expressed ignorance, unresolved
belief, confident error, and belief-utilization failure, including threshold
selection and the supporting literature, see
`docs/epistemic_state_thresholding.md`.

## Current Observational Result

The 24-state behavioral pilot contains 25 optimal-to-suboptimal transitions and
29 suboptimal-to-optimal recoveries.

No individual belief is currently a strong standalone indicator:

| Pilot observation | Interpretation |
|---|---|
| Right-wall error onset has the highest precision for an optimal-to-suboptimal transition within three reasoning positions, but occurs only six times. | Potentially precise but too rare for a stable conclusion. |
| Wall-down error onset covers more transitions but has low precision. | Greater coverage, many false alarms. |
| Door-open changes cover many transitions but have very low precision; every environment truth label is closed. | Likely behavioral-probe instability rather than informative state variation. |
| Belief-error recoveries generally occur after action recovery more often than before it. | More consistent with a lagging correlate than a leading indicator. |

Candidate chains are sparse. Only one exact ordered chain repeats across two
events, and the most common primary-belief pair covers three events. The pilot
therefore does not establish a stable belief chain.

Trajectory-held-out predictive models support the same conclusion. For
optimal-to-suboptimal transitions, no learned model exceeds the constant
prevalence null's AUROC of 0.500. Belief-change dynamics is the strongest
learned model at 0.489, while the action-conditioned and belief-action-chain
models reach 0.288 and 0.196. For recoveries, belief-change dynamics reaches
AUROC 0.583, but has worse Brier score and log loss than the constant null.
These are pilot estimates from only 14 complete-case trajectories with 19
optimality-loss events, and six trajectories with 21 recovery events.

The completed exact-matched run contains 46 states, 5,553 reasoning positions,
and 72,165 belief queries. Cross-validation keeps source trajectories and
matched pairs together, leaving 11 independent validation groups.

| Matched-run result | Interpretation |
|---|---|
| The combined model predicts upcoming optimality loss with AUROC 0.603, versus 0.569 for reasoning progress. The paired AUROC difference is 0.034 with a 95 percent interval of -0.013 to 0.057, and calibration does not improve. | Measured beliefs do not reliably warn that an optimal action recommendation is about to become suboptimal. |
| General belief errors predict recovery with AUROC 0.758 and improve Brier score by -0.028 and log loss by -0.053 relative to reasoning progress. Both calibration-difference intervals exclude zero. | Current belief errors contain useful information about when a suboptimal recommendation will recover. |
| Action-conditioned and belief-action-chain models do not outperform general belief errors. | The current results do not support a special action-relevant factorization beyond general state-belief errors. |
| The most frequent ordered belief-change chain appears near three events. | No chain reaches the prespecified recurrence threshold of five independent trajectories. |

## What Factorization Should Mean

Let `t` index reasoning-prefix positions for one fixed environment state.
`A_t` is the action recommended after prefix `t`, and `O_t` indicates whether
`A_t` is in the planner-optimal action set. Fit separate models for:

`Y_t^loss = 1[O_t = 1 and O_(t+1) = 0]`

and:

`Y_t^recovery = 1[O_t = 0 and O_(t+1) = 1]`.

The loss model is evaluated only at positions where `O_t = 1`; the recovery
model is evaluated only where `O_t = 0`.

| Model | Predictors beyond intercept | Question answered |
|---|---|---|
| Constant event-rate null | None | Is any learned model better than predicting the cohort event rate? |
| Reasoning-progress baseline | Normalized reasoning position | Are transitions predictable only from where the model is in its trace? |
| General belief-error | Progress and five state-belief error indicators | Do incorrect state beliefs predict the next transition? |
| Belief dynamics | Progress and counts of changes, error onsets, and recoveries | Do recent belief changes predict the next transition? |
| Action-conditioned | Progress and beliefs specifically about the recommended action | Do the model's beliefs contradict or misdescribe its chosen action? |
| Combined | General and action-conditioned predictors | Does action relevance add information beyond general errors? |
| Belief-action chain | Combined predictors and two task-motivated conjunctions | Do coherent combinations of action-relevant beliefs add predictive value? |

### Reasoning-Progress Baseline

Let `p_t = t / T`, where `T` is the final reasoning position. The baseline is:

`logit P(Y_t = 1) = alpha + beta_p p_t`.

It tests whether transitions are predictable merely because they tend to occur
earlier or later in a trace. A belief model is useful only if it improves on
both this baseline and the constant event-rate null.

### General Belief-Error Model

For belief `i`, let `B_i,t` be the parsed behavioral-probe answer, `G_i` its
fixed environment truth, and define:

`belief_i_is_wrong(t) = 1[B_i,t != G_i]`.

The primary beliefs are:

`I = {wall_left, wall_right, wall_up, wall_down, has_key}`.

The general belief-error model is:

`logit P(Y_t = 1) = alpha + beta_p p_t + sum_(i in I) beta_i belief_i_is_wrong(t)`.

This asks whether being wrong about any measured state factor predicts the
next action-optimality transition. It does not ask whether the error is
relevant to the action currently recommended.

The separate belief-dynamics model uses changes rather than current errors:

`logit P(Y_t = 1) = alpha + beta_p p_t`

`+ beta_1 number_of_belief_changes(t)`

`+ beta_2 number_of_error_onsets(t)`

`+ beta_3 number_of_error_recoveries(t)`.

### Action-Conditioned Model

Define all features relative to the direction recommended by `A_t`:

| Variable | Definition |
|---|---|
| `wall_report_is_wrong(t)` | Chosen-direction wall report is wrong |
| `wall_hit_prediction_is_wrong(t)` | Predicted movement consequence is wrong |
| `reports_chosen_direction_blocked(t)` | Model reports its recommended direction as blocked |
| `predicts_chosen_action_hits_wall(t)` | Model predicts its recommended move will hit a wall |

The action-conditioned model is:

`logit P(Y_t = 1) = alpha + beta_p p_t`

`+ beta_1 wall_report_is_wrong(t)`

`+ beta_2 wall_hit_prediction_is_wrong(t)`

`+ beta_3 reports_chosen_direction_blocked(t)`

`+ beta_4 predicts_chosen_action_hits_wall(t)`.

The first two indicators measure accuracy. The latter two measure direct
belief-action inconsistency: the model recommends an action while reporting
that it is blocked or will collide.

### Task-Motivated Conjunctions

The implemented belief-action-chain model adds only two conjunctions:

| Conjunction | Definition | Interpretation |
|---|---|---|
| Chosen move reported blocked | `reports_chosen_direction_blocked(t) x predicts_chosen_action_hits_wall(t)` | The model reports a wall in the chosen direction and predicts that the chosen move will hit it, yet still recommends that move. |
| Both action-relevant reports wrong | `wall_report_is_wrong(t) x wall_hit_prediction_is_wrong(t)` | Both the chosen-direction wall report and predicted movement consequence are wrong. |

The implemented belief-action-chain model is:

`logit P(Y_t = 1) = general belief-error model + action-conditioned indicators + conjunction indicators`.

It includes all five general belief errors, the chosen-action consequence
error, the chosen-direction wall error, both direct inconsistency reports, and
the two conjunctions. The individual indicators remain in the model whenever
their conjunction is included.

These are deliberately not all possible pairwise interactions. For example,
`wall_left_error * has_key_error` is not included because it has no general
task-semantic interpretation.

A fuller DoorKey action-value factorization is proposed but not implemented:

| Required feature | Role in action value |
|---|---|
| Chosen destination is traversable | Immediate action feasibility |
| A closed door blocks the intended route | Whether door interaction is required |
| Key is held when door passage is required | Whether the door prerequisite is satisfied |
| Action reduces shortest-path distance to the current instrumental target | Whether the action advances the plan |

Such features could define an estimated action-feasibility or action-value
score, but raw categorical answers should not simply be multiplied. Products
of probabilities require calibrated belief estimates and defensible
conditional-independence assumptions.

### Ordered Belief-Change Chains

An ordered chain is a temporal sequence, not an interaction term. For example:

`has_key error onset at t-2 -> wall_right error onset at t-1 -> optimality loss at t`.

Mine these sequences around action transitions, but only report a chain as
recurring when it appears in at least five independent trajectories. Select
candidate chains on training trajectories and evaluate their transition
prediction on held-out trajectories.

## Next Observational Experiment

The matched 46-state behavioral cohort is complete. Use the expanded 185-state
cohort to validate recovery prediction, examine failure-category differences,
and increase the number of independent source trajectories.

| Cohort | Execution script |
|---|---|
| Matched 46-state cohort | `scripts/run_matched_belief_transition_pipeline.sh` |
| Expanded 185-state cohort | `scripts/run_expanded_belief_transition_pipeline.sh` |

Both pipelines are resumable. The expanded pipeline seeds completed
action-prefix evaluations and belief queries from the pilot and matched runs
before issuing new requests.

| Analysis component | Specification |
|---|---|
| Measurements | All primary belief errors at every reasoning position |
| Outcomes | Optimality loss and recovery within horizons of one and three positions |
| Models | General-belief, belief-dynamics, action-conditioned, and task-motivated belief-action-chain logistic models |
| Baselines | Constant event rate and reasoning progress |
| Metrics | Held-out AUROC, Brier score, log loss, event coverage, precision, false-alarm rate, and calibration |
| Ordered chains | Report only when recurring in at least five independent trajectories |

Use grouped cross-validation and regularization. Keep complete source
trajectories together, and keep both members of an exact-matched pair in the
same fold. Do not add unrestricted pairwise interactions. Exclude `door_open`
from primary predictive models until the cohort contains variation in true door
state. Do not select chains and estimate their final effect on the same
trajectories.

## Causal Follow-Up

Select recurring chains that improve held-out transition prediction. At the
first belief change in each chain:

| Intervention | Purpose |
|---|---|
| Insert a correct belief statement | Test whether correcting the belief prevents or reverses the transition |
| Insert an incorrect counterfactual belief statement | Test whether inducing the error promotes the transition |
| Delete or replace the associated reasoning sentence | Test whether the original sentence is necessary |
| Sample multiple continued reasoning traces | Estimate the intervention's effect across stochastic continuations |

Measure whether the intervention changes downstream beliefs and action
optimality. For a candidate pair `B1, B2`, compare four conditions: preserve
both, change only `B1`, change only `B2`, and change both. This tests whether
the joint causal effect differs from the sum of the two individual effects.
If `mu_bc` is the probability of the target outcome under condition
`B1=b, B2=c`, estimate:

`Delta_interaction = mu_11 - mu_10 - mu_01 + mu_00`.

A nonzero value indicates an interaction on the probability scale.

Text insertion manipulates the reasoning context, not the latent belief
directly. Rerun the corresponding belief probes as manipulation checks. If the
target belief report does not change, interpret the result as a sentence
intervention rather than a belief intervention.

Activation interventions should be deferred until position-general white-box
probes are validated. Activation Oracle can generate hypotheses, but the
available Qwen3-8B Oracle cannot provide causal or calibrated GPT-OSS-20B
belief measurements.
