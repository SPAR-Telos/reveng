# Reader-Facing Figures

Use these figures in this order:

1. `figs/example_action_probabilities_and_change_points.png` shows what the
   detector receives and how a detected sentence is identified. The deliberately
   selected examples are not evidence for a failure-control difference in action
   confidence.
2. `figs/change_points_near_decision_events.png` is the main result. It tests
   whether detected points occur near recommendation and optimality events more
   often than progress-matched random sentences.
3. `figs/change_point_count_by_state_group.png` shows that failure and control
   states have similar numbers of detected points.
4. `figs/change_point_timing_by_state_group.png` shows where detected points
   occur over reasoning progress; it is secondary because the confidence
   intervals are wide in this 46-state pilot.

The historical figures in
`outputs/hypothesis_tests/action_distribution_change_points_v1/` select the
largest observed change in every state and are not change-point detection
results.
