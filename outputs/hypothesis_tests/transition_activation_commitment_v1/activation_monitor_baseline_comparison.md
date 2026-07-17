# Supervised Activation Monitor Baseline Comparison

This supervised monitor uses grouped cross-validation by matched pair or trajectory. PCA is fit inside each training fold before ridge logistic regression. The current result is predictive and not causal.

| Target | Best activation model | Activation AUROC | Reasoning-progress AUROC | Scalar-readout AUROC | Interpretation |
|---|---|---:|---:|---:|---|
| action change | activation_pca_15_sentence_final | 0.597 | 0.532 | 0.816 | beats reasoning progress, but not scalar readouts |
| optimal to suboptimal | activation_pca_15_sentence_mean | 0.628 | 0.603 | 0.812 | does not pass +0.05 over reasoning progress |
| suboptimal to optimal | activation_pca_23_sentence_final | 0.613 | 0.620 | 0.775 | does not pass +0.05 over reasoning progress |
| commitment onset | activation_pca_15_sentence_final | 0.704 | 0.732 | 0.756 | does not pass +0.05 over reasoning progress |
