# Multi-Horizon Closure and Reasoning-Dynamics Graph

## Questions

1. Over what future sentence/word/token range do semantic labels add information beyond the current behavioral state?
2. Do semantic operations exhibit ordering beyond their prevalence and coarse progress trends?
3. Do BEAST boundaries form reproducible dynamical operator types and a non-random transition graph?

## Predictive-closure definition

Closure is target- and horizon-specific: semantic closure holds when adding the current sentence's averaged replicate multilabel vector does not improve grouped out-of-sample loss beyond current regime, regime duration, progress, action probabilities, and action entropy. It is not a claim that the state is sufficient for every possible future observable.

Token distances use the local `o200k_harmony` encoding. Word/token horizons select the first future sentence boundary reaching the requested cumulative distance.

## Multi-horizon results

| horizon type | horizon value | metric | n rows | median sentences ahead | median words ahead | median tokens ahead | semantic improvement | ci low | ci high | bh q |
|---|---|---|---|---|---|---|---|---|---|---|
| sentence | 1 | change_log_loss | 6990 | 1.0000 | 7.0000 | 11.0000 | -0.0004 | -0.0011 | 0.0003 | 0.6510 |
| sentence | 1 | exact_log_loss | 6990 | 1.0000 | 7.0000 | 11.0000 | -0.0006 | -0.0014 | 0.0011 | 0.6510 |
| sentence | 2 | change_log_loss | 6944 | 2.0000 | 15.0000 | 25.0000 | -0.0013 | -0.0021 | -0.0004 | 0.1055 |
| sentence | 2 | exact_log_loss | 6944 | 2.0000 | 15.0000 | 25.0000 | -0.0006 | -0.0015 | 0.0013 | 0.6510 |
| sentence | 3 | change_log_loss | 6898 | 3.0000 | 24.0000 | 40.0000 | -0.0005 | -0.0015 | 0.0010 | 0.6510 |
| sentence | 3 | exact_log_loss | 6898 | 3.0000 | 24.0000 | 40.0000 | -0.0011 | -0.0026 | 0.0019 | 0.6510 |
| sentence | 5 | change_log_loss | 6806 | 5.0000 | 41.0000 | 71.0000 | -0.0000 | -0.0027 | 0.0020 | 0.9990 |
| sentence | 5 | exact_log_loss | 6806 | 5.0000 | 41.0000 | 71.0000 | -0.0035 | -0.0055 | -0.0001 | 0.3132 |
| sentence | 8 | change_log_loss | 6668 | 8.0000 | 67.0000 | 117.0000 | 0.0001 | -0.0024 | 0.0023 | 0.9753 |
| sentence | 8 | exact_log_loss | 6668 | 8.0000 | 67.0000 | 117.0000 | -0.0040 | -0.0082 | -0.0009 | 0.2290 |
| sentence | 13 | change_log_loss | 6438 | 13.0000 | 112.0000 | 198.0000 | 0.0001 | -0.0026 | 0.0018 | 0.9753 |
| sentence | 13 | exact_log_loss | 6438 | 13.0000 | 112.0000 | 198.0000 | 0.0003 | -0.0103 | 0.0059 | 0.9363 |
| token | 32 | change_log_loss | 6835 | 3.0000 | 24.0000 | 39.0000 | -0.0007 | -0.0018 | 0.0001 | 0.3132 |
| token | 32 | exact_log_loss | 6835 | 3.0000 | 24.0000 | 39.0000 | -0.0006 | -0.0024 | 0.0029 | 0.8025 |
| token | 64 | change_log_loss | 6701 | 5.0000 | 43.0000 | 72.0000 | -0.0012 | -0.0019 | -0.0005 | 0.1003 |
| token | 64 | exact_log_loss | 6701 | 5.0000 | 43.0000 | 72.0000 | -0.0013 | -0.0039 | 0.0017 | 0.9023 |
| token | 128 | change_log_loss | 6502 | 9.0000 | 80.0000 | 136.0000 | -0.0026 | -0.0037 | -0.0013 | 0.1003 |
| token | 128 | exact_log_loss | 6502 | 9.0000 | 80.0000 | 136.0000 | -0.0041 | -0.0069 | -0.0012 | 0.0820 |
| token | 256 | change_log_loss | 6091 | 17.0000 | 153.0000 | 264.0000 | -0.0031 | -0.0047 | -0.0017 | 0.0820 |
| token | 256 | exact_log_loss | 6091 | 17.0000 | 153.0000 | 264.0000 | -0.0017 | -0.0067 | 0.0033 | 0.7172 |
| word | 32 | change_log_loss | 6770 | 4.0000 | 36.0000 | 63.0000 | -0.0006 | -0.0014 | 0.0002 | 0.4603 |
| word | 32 | exact_log_loss | 6770 | 4.0000 | 36.0000 | 63.0000 | -0.0020 | -0.0039 | 0.0006 | 0.5961 |
| word | 64 | change_log_loss | 6587 | 8.0000 | 68.0000 | 120.0000 | -0.0028 | -0.0036 | -0.0013 | 0.1003 |
| word | 64 | exact_log_loss | 6587 | 8.0000 | 68.0000 | 120.0000 | -0.0047 | -0.0086 | -0.0011 | 0.0820 |
| word | 128 | change_log_loss | 6244 | 15.0000 | 132.0000 | 233.0000 | -0.0022 | -0.0053 | 0.0020 | 0.5961 |
| word | 128 | exact_log_loss | 6244 | 15.0000 | 132.0000 | 233.0000 | -0.0053 | -0.0111 | 0.0010 | 0.5111 |
| word | 256 | change_log_loss | 5574 | 30.0000 | 260.0000 | 459.0000 | -0.0017 | -0.0038 | 0.0001 | 0.3132 |
| word | 256 | exact_log_loss | 5574 | 30.0000 | 260.0000 | 459.0000 | -0.0028 | -0.0078 | 0.0060 | 0.6510 |

Supported positive semantic increments after BH correction: 0 of 28 horizon-metric tests. Therefore this scan detects no violation of semantic predictive closure from 1–30 median sentences, 7–260 median words, or 11–459 median tokens ahead. It does not establish equivalence because no smallest effect of interest was prespecified.

## Semantic process grammar

Primary-label adjacent mutual information is tested against shuffling labels within each trajectory and progress decile. Multilabel edges and three-sentence paths distribute each window's mass across simultaneous labels and use the same null. A motif is declared only when enriched with BH q < .05 in both annotation runs.

| annotation run | observed mutual information bits | null mean bits | excess information bits | permutation p upper |
|---|---|---|---|---|
| original | 0.1670 | 0.0491 | 0.1179 | 0.0002 |
| replicate | 0.1720 | 0.0524 | 0.1195 | 0.0002 |

Replicated enriched semantic paths: 12 edges and 27 three-sentence chains. Same-label bouts account for part, but not all, of this structure. Strong cross-label paths include:

| order | path | mean enrichment ratio | bh q original | bh q replicate |
|---|---|---|---|---|
| 3 | route_planning -> procedural_continuation -> procedural_continuation | 3.9572 | 0.0121 | 0.0112 |
| 3 | new_inference -> procedural_continuation -> procedural_continuation | 3.3493 | 0.0121 | 0.0112 |
| 3 | new_inference -> consolidation -> route_planning | 2.6184 | 0.0121 | 0.0260 |
| 3 | procedural_continuation -> procedural_continuation -> state_readout | 2.5602 | 0.0121 | 0.0112 |
| 3 | action_commitment -> state_readout -> action_commitment | 2.4799 | 0.0121 | 0.0112 |
| 3 | state_readout -> action_commitment -> action_commitment | 2.4505 | 0.0121 | 0.0112 |
| 2 | action_commitment -> procedural_continuation | 2.3822 | 0.0043 | 0.0038 |
| 3 | route_planning -> procedural_continuation -> state_readout | 2.1322 | 0.0205 | 0.0112 |
| 3 | action_commitment -> action_commitment -> state_readout | 2.0409 | 0.0121 | 0.0112 |
| 3 | new_inference -> route_planning -> procedural_continuation | 2.0222 | 0.0121 | 0.0112 |
| 3 | restatement -> restatement -> new_inference | 1.5745 | 0.0287 | 0.0112 |
| 3 | procedural_continuation -> state_readout -> state_readout | 1.5270 | 0.0121 | 0.0112 |

## Unsupervised BEAST operator graph

Each BEAST boundary is represented without semantic or outcome labels by action-identity-invariant pre/post changes in total variation, Jensen-Shannon distance, entropy, confidence, and argmax switching. K-means K is selected by silhouette subject to at least 10 boundaries per cluster. Nodes are resulting operator clusters; directed edges join consecutive BEAST boundaries within a trace.

| n clusters | silhouette | minimum cluster size | eligible | selected |
|---|---|---|---|---|
| 2 | 0.3508 | 55 | True | False |
| 3 | 0.4020 | 40 | True | False |
| 4 | 0.4359 | 18 | True | False |
| 5 | 0.4770 | 15 | True | True |
| 6 | 0.4507 | 15 | True | False |

| operator cluster | n events | total variation | entropy delta bits | confidence delta | argmax change rate | failure fraction |
|---|---|---|---|---|---|---|
| operator_1 | 34 | 0.5706 | -0.7258 | 0.3126 | 1.0000 | 0.5588 |
| operator_2 | 18 | 0.8752 | -0.0848 | 0.0228 | 1.0000 | 0.6111 |
| operator_3 | 37 | 0.2949 | -0.6956 | 0.2918 | 0.0000 | 0.4865 |
| operator_4 | 15 | 0.3006 | 0.6627 | -0.2988 | 0.0000 | 0.4667 |
| operator_5 | 56 | 0.4596 | 0.3393 | -0.1796 | 1.0000 | 0.4821 |

Enriched operator edges after BH correction: 0; enriched three-operator motifs: 0.

Operator-cluster/semantic-label associations after within-trace permutation and BH correction: 0. These associations are post-hoc descriptions and do not define the clusters.

## Decision rules and limitations

Evidence for a grammar requires replicated temporal dependence, not merely common labels. Evidence for graph motifs requires enrichment over a within-trace ordering null. BEAST is retrospective, clusters are exploratory and estimated on this corpus, and semantic-to-cluster prevalence differences are descriptive rather than causal.

The observational causal-intervention proposal is not run: prepared single continuations do not identify the effect of rewriting or removing a sentence. That requires newly sampled matched continuations.

## Literature basis

The tests operationalize network motifs as subgraphs overrepresented relative to randomized networks ([Milo et al., 2002](https://doi.org/10.1126/science.298.5594.824)), temporal motifs as ordered event sequences evaluated with null models ([Kovanen et al., 2011](https://doi.org/10.1088/1742-5468/2011/11/P11005)), process discovery from event logs ([Process Mining Manifesto, 2012](https://doi.org/10.1007/978-3-642-28108-2_19)), and predictive state sufficiency as equivalence of conditional future distributions ([Shalizi and Crutchfield, 2001](https://arxiv.org/abs/cond-mat/9907176)). These works motivate the tests; they do not imply that DoorKey reasoning must contain motifs.
