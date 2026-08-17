# Figure Captions

## `example_action_probabilities_and_change_points.png`

Action probabilities and Bayesian change-point probabilities for one failure state and
one control state. The left panels show temperature 0.7 candidate-token probabilities
over `UP`, `DOWN`, `LEFT`, and `RIGHT` after each reasoning sentence. Vertical lines
mark detected changes. The right panels show the BEAST posterior probability that each
sentence is a change point; the dashed line is the prespecified 0.70
location threshold. Examples are selected by the largest posterior-weighted change in
distance from the initial action distribution within each group. They demonstrate the
detector and are not representative group averages; in particular, their difference in
action confidence should not be interpreted as a failure-control difference.

## `change_point_timing_by_state_group.png`

Timing of detected action-distribution change points across reasoning. The horizontal
axis divides each state trace into ten equal bins by the fraction of reasoning sentences
completed. The vertical axis is the fraction of states with at least one detected point
in each bin. Bands are 95% trajectory-bootstrap intervals. Each group contains 23 states.

## `change_point_count_by_state_group.png`

Number of detected action-distribution change points in each fixed environment state.
Dots are states; diamonds are group means; error bars are 95% trajectory-bootstrap
intervals. A trace must have posterior odds greater than 9:1 for one or more changes,
and each reported location must have posterior probability at least 0.70. The pilot
contains 23 failure states and 23 matched control states.

## `change_points_near_decision_events.png`

Fraction of detected action-distribution change points occurring within 3
sentences before or after each decision event. Error bars for detected points are 95%
trajectory-bootstrap intervals. The comparison samples random sentences from the same
state and same tenth of reasoning progress; its error bars show the 2.5th and 97.5th
percentiles over 2000 randomizations. Recommendation changes include all
changes in the highest-probability action. Optimality is evaluated by the DoorKey planner.
