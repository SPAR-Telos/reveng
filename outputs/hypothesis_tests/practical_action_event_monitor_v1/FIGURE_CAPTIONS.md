# Figure Captions

## optimality_loss_precision_recall.png

Precision-recall curves for forecasting a change from a planner-optimal recommendation to a suboptimal recommendation. Predictors use only information available at the current sentence. The one-sentence panel predicts the next sentence boundary; the three-sentence panel predicts any loss within the next three boundaries. Values in parentheses are held-out AUPRC. The dotted horizontal line is the event rate.

## auprc_improvement_by_signal.png

Change in trajectory-held-out AUPRC after adding activation, belief, or sentence-semantic signals to a behavioral baseline. Error bars are trajectory-bootstrap 95% intervals. The baseline uses reasoning progress, current action confidence, and recent change in the four-action probability distribution. Positive values indicate better ranking of future event windows; improvements are considered credible only when held-out log loss also improves.
