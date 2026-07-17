# Figure Captions

## predictive_improvement_by_event.png

Held-out AUROC for action-event classification in the matched 46-state GPT-OSS-20B run. The gray diamond is the baseline model using reasoning progress and action confidence. Reasoning progress is the fraction of the reasoning text revealed at the current sentence. Action confidence is the probability assigned to the currently recommended action. Bars show the same baseline plus belief features, layer 15 sentence-mean activation features, or both. The activation features are projected to 32 PCA components inside each training fold. The event label is at the same sentence position as the activation, so this plot tests event classification rather than advance prediction.

## information_score_by_layer.png

Activation-event dependence for action events across GPT-OSS-20B layers 8, 15, and 23. For each layer and sentence representation, activations are standardized, projected to 32 PCA components, and compared with the binary event label using linear CKA. This is a low-cost HSIC-style dependence proxy, not the exact mutual-information estimator used by the MI-peaks paper. Higher values indicate stronger association between activations and the event label, but do not by themselves show that the information is used for action selection.
