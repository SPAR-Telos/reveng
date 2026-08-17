# Figure Captions

## belief_readouts_forecast_optimality_loss.png

Trajectory-held-out prediction of whether the model's recommended action will change
from planner-optimal to suboptimal after the current reasoning sentence. The behavioral
baseline uses reasoning progress, current action confidence, and the recent change in
the probability distribution over UP, DOWN, LEFT, and RIGHT. The belief model adds
behavioral readout probabilities, uncertainty, answer changes, and conflicts between
the recommended action and reported walls or action consequences. The right panel shows
the AUPRC improvement and 95% trajectory-bootstrap interval. Belief readouts improve the
three-sentence forecast but not conclusively the next-sentence forecast.

## action_distribution_changes_near_events.png

Fraction of 160 offline Bayesian change points in the elicited action-probability
sequence that fall within three reasoning sentences before or after each event.
Progress-matched comparisons sample sentences from the same environment state and the
same tenth of reasoning progress. Error bars show 95% trajectory-bootstrap intervals
for detected change points and the central 95% of 2,000 randomizations for comparisons.
The association is strongest for changes in the highest-probability recommended action.
It does not establish that a reasoning sentence caused the action change.

## activation_monitor_baseline_comparison.png

Trajectory-held-out classification of action events at the current sentence position
in 7,038 positions from 46 environment states and 31 trajectories. The activation
result is the best ridge logistic monitor across GPT-OSS-20B layers 8, 15, and 23 and
sentence-mean or sentence-final representations. Behavioral readouts are scalar action
and belief variables available at the same position. Points are AUROC estimates and
bars are 95% trajectory-bootstrap intervals. This is contemporaneous classification,
not advance prediction.

## attention_changes_when_recommended_action_changes.png

Change in attention from the immediate action-readout token to each prompt region when
the replayed recommended action changes, relative to a progress-matched sentence from
the same environment state where the action remains stable. Points show layer-15 means
for 67 matched pairs from 35 states and 28 trajectories; bars are 95% trajectory-
bootstrap intervals. Attention per token increases most for the newest reasoning
sentence. Attention is correlational and does not prove causal use.

## semantic_labels_across_analyses.png

Cross-analysis role of broad sentence-function labels in 152 close, same-state matched
pairs. Panel A shows paired differences in semantic-function rates between BEAST
change-point sentences and matched sentences. Panel B shows the change in
trajectory-held-out change-point AUROC when semantic labels, activation geometry, or
both are added to reasoning progress and sentence length. Panel C shows categorical
behavioral-belief entropy at change points by semantic function, with
trajectory-bootstrap intervals. Panel D shows the semantic breakdown of the
immediate-action attention increase to the newest sentence; bars are 1.96 times the
trajectory-level standard error. Panels C and D are exploratory. BEAST and the
semantic comparisons are retrospective, and none of the panels is a causal test.
