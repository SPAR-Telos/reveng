# Figure Captions

## `event_aligned_belief_errors.png`
Wall, key, and door belief errors before and after action-optimality changes in the 8-state pilot. Offset zero is the first chunk prefix with the changed optimality status; minus one is the preceding prefix and plus one is the following prefix. The y-axis averages errors in four nearby-wall beliefs, key possession, and whether the door is open. For example, 0.25 means that one quarter of these available belief answers were incorrect on average. This is a descriptive average, not a causal estimate.

## `belief_transition_indicator_heatmap.png`
Coverage of action-optimality transitions by nearby belief changes. Each cell reports the fraction of action-optimality transition events that have the named belief change within a three-prefix window. "Belief leads" means the belief change happens before the action transition. "Belief lags" means the belief change happens after the action transition.

## `primary_belief_transition_indicator_heatmap.png`
Same as `belief_transition_indicator_heatmap.png`, restricted to the six primary state-belief probes.

## `action_transition_heatmap.png`
Counts of prefix-elicited action changes by event type. Rows are the previous recommended action and columns are the current recommended action at the next chunk-prefix position. This describes which action switches underlie action identity changes, optimality losses, and recoveries.

## `uncertainty_predictor_coefficients.png`
Exploratory logistic-regression coefficients for whether an action event occurs after the next reasoning chunk. The y-axis lists the uncertainty predictor and the increase represented by one coefficient unit. The x-axis is the change in predicted log odds: positive values mean that greater uncertainty is associated with a more likely event, zero means no association, and negative values mean a less likely event. Points are coefficients and bars are model-based 95% confidence intervals. The optimality-loss and recovery estimates are unstable because they contain only 4 and 5 events.

## `uncertainty_over_reasoning_progress.png`
Action and state-belief entropy as increasing fractions of each reasoning trace are revealed. Each point first averages positions within a progress decile for each of the 8 environment states, then averages those state-level values; shading is one standard error across states. Action entropy is calculated from 10 sampled immediate actions at temperature 0.7. State-belief entropy is the mean entropy of the wall, key-possession, and door-open answer probabilities at temperature 0. Zero bits means all measured probability is assigned to one answer; one bit corresponds to the uncertainty of two equally likely alternatives.
