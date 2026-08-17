# Figure Captions

## supervised_activation_monitor_auc.png

Held-out event classification in the matched 46-state run. The figure compares the best ridge logistic activation monitor across GPT-OSS-20B layers 8, 15, and 23 with reasoning progress and scalar behavioral action and belief readouts. PCA is fit only inside each activation monitor's training fold, and validation holds out complete trajectories or matched pairs. Points show AUROC and bars show 95% trajectory-bootstrap intervals; 0.5 indicates chance discrimination. Sample sizes are shown on the y-axis. The evaluated set contains 7,038 sentence positions from 46 environment states and 31 trajectories. This is contemporaneous classification, not advance prediction.

## commitment_boundary_tail_length.png

Retrospective action commitment timing in the matched 46-state run. Each point is one environment state. The x-axis is the fraction of reasoning sentences already revealed at the first sentence where the recommended action equals the full-trace recommendation and remains stable. The y-axis is the number of reasoning sentences after that commitment point. Colors separate states whose full-trace action is optimal from states whose full-trace action is suboptimal. This plot distinguishes early commitment with long remaining reasoning from late commitment with little remaining reasoning.

## post_commitment_length_vs_uncertainty.png

Association between post-commitment reasoning length and residual belief uncertainty in the matched 46-state run. Each point is one environment state. The y-axis is the number of reasoning sentences after retrospective action commitment. Current-state belief entropy is computed from wall, key, and door state-belief readouts. Transition-belief entropy is computed from action-conditioned consequence readouts. The fitted line is descriptive and does not show that uncertainty causes additional reasoning.
