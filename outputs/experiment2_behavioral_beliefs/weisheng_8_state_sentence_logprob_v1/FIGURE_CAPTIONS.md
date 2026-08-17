# Figure Captions

## `uncertainty_predictor_coefficients.png`
Exploratory logistic-regression coefficients for whether an action event occurs after the next sentence. The y-axis lists the uncertainty predictor and the increase represented by one coefficient unit. The x-axis is the change in predicted log odds: positive values mean that greater uncertainty is associated with a more likely event, zero means no association, and negative values mean a less likely event. Action entropy is computed from the normalized UP, DOWN, LEFT, and RIGHT token probabilities at temperature 0.7. State-belief entropy uses wall, key, and door answer probabilities at temperature 0.7 and is scaled per 0.01 bits. Points are coefficients and bars are model-based 95% confidence intervals.
