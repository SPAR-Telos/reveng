# Fields-Inspired Reasoning-Regime Coarse-Graining Pilot

## Question

Can sentence-level action dynamics be coarse-grained into a small, reproducible behavioral state process, and do recurrence or unusual transitions distinguish rollout-failure states from exact-matched controls?

## Deterministic state space

Every valid position receives exactly one of four regimes: `unsettled_optimal`, `unsettled_suboptimal`, `stable_optimal`, or `stable_suboptimal`. Stability is the first position in the final constant argmax-action suffix. Optimality uses the stored DoorKey-aware planner label. Probability revision and optimality loss are transition events, not regimes.

Behavioral recurrence is a return to a previously left regime after consecutive repeats have been collapsed. Transition surprisal is `-log2 P_control(next_regime | current_regime)`, estimated from controls outside the held-out trajectory-connected validation group with additive smoothing. To avoid calling every state change unusual merely because self-transitions dominate—and to avoid defining rarity from the final outcome—the descriptive unusual-transition cutoff is the 95th percentile among held-out-control non-self edges that remain unsettled. It is applied only to such edges.

## Data and dependence

- States: 46 in 23 exact matched pairs
- Source trajectories: 31
- Trajectory-connected validation groups: 11
- Sentence-boundary transitions: 7038
- Strict final-suboptimal failure states: 8
- Traces contributing to next-regime prediction: 45 (one trace begins in its stable suffix)

## Predictive closure before stable selection

The held-out target is the next regime at positions that are not yet in the stable suffix. Each source state receives equal total fitting weight, and validation holds out complete trajectory-connected groups.

| model | mean log loss | mean accuracy | log loss improvement vs progress | improvement ci low | improvement ci high | improvement sign flip p | log loss improvement vs first order |
|---|---|---|---|---|---|---|---|
| progress_only | 0.7164 | 0.6442 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.3742 |
| first_order_regime | 0.3422 | 0.9105 | 0.3742 | 0.3273 | 0.4324 | 0.0010 | 0.0000 |
| second_order_regime | 0.3099 | 0.9086 | 0.4065 | 0.3611 | 0.4595 | 0.0000 | 0.0323 |
| full_action_distribution | 0.7034 | 0.6675 | 0.0130 | -0.0465 | 0.0487 | 0.5830 | -0.3612 |

## Failure-control contrasts

| metric | failure mean | control mean | mean difference | ci low | ci high | sign flip p two sided | bh q |
|---|---|---|---|---|---|---|---|
| recurrence_returns_per_100_positions | 4.8682 | 2.7529 | 2.1152 | -1.6345 | 5.3468 | 0.4922 | 0.8203 |
| optimality_losses_per_100_positions | 2.8626 | 1.5886 | 1.2740 | -0.8067 | 2.8919 | 0.3945 | 0.8203 |
| unusual_nonself_transition_fraction | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| maximum_unsettled_transition_surprisal_bits | 3.7782 | 3.3849 | 0.3932 | -1.7199 | 1.5487 | 0.6797 | 0.8496 |
| stable_suffix_fraction | 0.2363 | 0.3495 | -0.1132 | -0.2768 | 0.0493 | 0.2744 | 0.8203 |

## Decision

The first-order regime model changes held-out log loss versus progress alone by +0.3742 (grouped interval [+0.3273, +0.4324]). Adding one more regime of history improves log loss over the first-order model by +0.0323 (grouped interval [+0.0161, +0.0440]). This rejects exact first-order closure in this pilot; an equivalence margin was not prespecified.

Failure minus control recurrence returns per 100 positions: +2.1152 (interval [-1.6345, +5.3468]). Failure minus control optimality losses per 100 positions: +1.2740 (interval [-0.8067, +2.8919]).

In this four-state construction, behavioural recurrence is almost entirely alternation between `unsettled_optimal` and `unsettled_suboptimal`; recurrence and optimality-loss metrics are therefore overlapping descriptions, not independent mechanisms.

Rare unsettled edge types observed above the held-out-control cutoff: 0 aggregated role/edge combinations. With only two possible unsettled non-self edge types, this graph has little capacity to identify unusual transition structure beyond optimality oscillation.

No prespecified failure-control metric survives BH correction at q < 0.05.

This is an offline coarse-graining pilot, not evidence of thermodynamic entropy production or a literal arrow of time. Stable selection is retrospective, self-transitions are structurally common, and the strict final-suboptimal subset is too small for a confirmatory graph comparison. Semantic ReasoningFlow labels remain outside the primary graph because their manual validation is incomplete.

## Next gate

Confirm any supported transition pattern on new repeated continuations. In particular, continue low-stabilization strict failures for controlled additional lengths and test whether they enter `stable_optimal`, remain `stable_suboptimal`, or continue recurrent unsettled transitions.
