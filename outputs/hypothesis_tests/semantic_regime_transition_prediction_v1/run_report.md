# Semantic-Informed Regime-Transition Prediction

## Question

Do semantic multilabels improve prediction of regime persistence, regime-change hazard, and the exact destination conditional on change, beyond progress, behavioral action features, regime duration, and a finite window of regime history?

## Definitions

`history_size = k` is a sliding window containing the current regime and the preceding `k-1` regimes. Every model also includes reasoning progress, current regime duration, the four action probabilities, and action entropy. The prospective semantic model uses the most recently revealed sentence label; the operator sensitivity uses the label of the sentence revealed during the transition and is contemporaneous rather than forecasting.

The hurdle model first predicts `change = (z_next != z_current)`, then predicts one of `switch_unsettled`, `enter_stable_optimal`, or `enter_stable_suboptimal` conditional on change. These probabilities are recombined into one exact four-regime distribution.

## Data

- Prospective transitions: 5179
- Prospective changes: 354
- Change prevalence: 6.84%
- Trajectory-connected validation groups: 11
- Prospective destination support: 309 unsettled switches, 37 stable-optimal entries, and 8 stable-suboptimal entries.
- Change events per validation group range from 2 to 119; inference therefore uses trajectory-level summaries and trajectory-connected groups, but remains data-limited.

## Regime-history window without semantics

| history size | hazard log loss | hazard average precision | destination log loss | overall exact log loss | change exact log loss | balanced exact log loss |
|---|---|---|---|---|---|---|
| 1 | 0.2647 | 0.2026 | 0.5868 | 0.3158 | 2.8677 | 1.4785 |
| 2 | 0.2647 | 0.2014 | 0.5911 | 0.3164 | 2.8698 | 1.4796 |
| 3 | 0.2658 | 0.1998 | 0.5910 | 0.3178 | 2.8834 | 1.4865 |
| 5 | 0.2651 | 0.1940 | 0.5981 | 0.3173 | 2.9039 | 1.4969 |
| 8 | 0.2718 | 0.1855 | 0.6535 | 0.3311 | 3.0035 | 1.5466 |

## Prospective semantic sensitivity at history size 2

| semantic config | hazard log loss | hazard average precision | destination log loss | overall exact log loss | change exact log loss | balanced exact log loss |
|---|---|---|---|---|---|---|
| intersection_all | 0.2644 | 0.1926 | 0.6838 | 0.3176 | 2.9327 | 1.5109 |
| mean_all | 0.2617 | 0.1982 | 0.6436 | 0.3137 | 2.8728 | 1.4809 |
| mean_cues | 0.2648 | 0.1958 | 0.5852 | 0.3144 | 2.8536 | 1.4714 |
| mean_labels | 0.2616 | 0.1979 | 0.6288 | 0.3118 | 2.8713 | 1.4800 |
| none | 0.2647 | 0.2014 | 0.5911 | 0.3164 | 2.8698 | 1.4796 |
| original_all | 0.2601 | 0.2035 | 0.6949 | 0.3160 | 2.9397 | 1.5142 |
| replicate_all | 0.2631 | 0.1932 | 0.6184 | 0.3127 | 2.8451 | 1.4669 |
| union_all | 0.2617 | 0.1962 | 0.6673 | 0.3155 | 2.9111 | 1.4997 |

Positive incremental values below mean lower loss after adding mean original/replicate semantic features to the same nonsemantic model.

| metric | baseline mean | semantic mean | improvement | ci low | ci high | sign flip p | bh q across all tests |
|---|---|---|---|---|---|---|---|
| hazard_log_loss | 0.2647 | 0.2617 | 0.0030 | -0.0039 | 0.0098 | 0.4707 | 0.9893 |
| destination_log_loss | 0.5911 | 0.6436 | -0.0524 | -0.2080 | 0.0548 | 0.3818 | 0.9893 |
| exact_log_loss | 0.3164 | 0.3137 | 0.0026 | -0.0074 | 0.0127 | 0.8672 | 0.9893 |
| change_log_loss | 2.8698 | 2.8728 | -0.0029 | -0.1714 | 0.1033 | 0.9561 | 0.9893 |
| balanced_exact_log_loss | 1.4796 | 1.4809 | -0.0013 | -0.0842 | 0.0505 | 0.9619 | 0.9893 |

## Contemporaneous semantic-operator sensitivity

| semantic config | hazard log loss | hazard average precision | destination log loss | overall exact log loss | change exact log loss |
|---|---|---|---|---|---|
| mean_all | 0.2652 | 0.1995 | 0.7542 | 0.3293 | 3.0063 |
| none | 0.2658 | 0.2045 | 0.5758 | 0.3152 | 2.8325 |

## Conclusions

The best nonsemantic specification uses history size 1. Adding more raw regime lags does not improve change-hazard, conditional-destination, or exact-regime loss; history size 8 is distinctly worse. Thus the current regime plus the continuously valued duration and behavioral state carry more useful information than a longer literal regime window.

The prespecified prospective mean-semantic model does not improve any loss with a confidence interval excluding zero, and no prospective semantic specification has a robust positive incremental result. The labels therefore do not yet provide evidence of predictive information beyond the nonsemantic state variables. This is an informative null, not evidence that the semantic functions are causally irrelevant.

The contemporaneous operator alignment also fails to improve prediction and substantially worsens conditional-destination loss in this high-dimensional linear specification. Treat it as an overfitting warning rather than evidence of a harmful semantic effect.

## Interpretation guardrails

Overall exact-regime loss intentionally retains the natural persistence distribution. Hazard and conditional-destination results diagnose whether apparent performance is only persistence prediction. The balanced exact loss weights persistence and change equally as a diagnostic, not as the natural deployment distribution.

Semantic labels are generated measurements, not randomized treatments. Prospective associations may encode latent reasoning content or annotation artifacts; operator-aligned associations are contemporaneous and cannot by themselves establish that a semantic function caused a regime transition.

All semantic models use fixed L2 regularization rather than nested hyperparameter selection. With only 354 prospective changes—and only 8 stable-suboptimal entries—the conditional destination analysis has limited power, especially for interactions between current regime and semantic features.
