# Deterministic Reasoning-Graph Decision

## Decision

Use a **typed, left-to-right discourse/logical DAG** as the primary graph. Treat action probabilities, regimes, and BEAST boundaries as node attributes or external event markers—not as the rule that creates nodes or edges.

The current semantic process graph is a deterministic aggregate of adjacent label transitions and is useful for population grammar. It is not a dependency graph of an individual reasoning trace. The BEAST operator graph is exploratory because both its event locations and clustered node types are estimated from the time series.

## Why a DAG

For a trace segmented into elementary spans `s_1,...,s_n`, every edge must satisfy `i < j`. This guarantees acyclicity from autoregressive order while permitting branching, merging, crossing edges, and semantic backtracking. Backtracking is an edge *type*, not a backward-time edge.

Each destination node receives the minimal complete set of earlier predecessors needed to interpret it. Edge labels should be constrained and auditable:

- logical/content: `infer`, `restate`, `support`, `attack`;
- plan: `proceed`, `execute`, `decompose`, `verify`, `backtrack`;
- evaluation: `positive`, `negative`, `uncertain`;
- DoorKey-specific: `state_update`, `route_update`, `action_commit`.

Nodes retain the existing semantic multilabels, sentence/span identifiers, action distribution, regime, optimality, and BEAST indicator. The edge schema—not BEAST—defines topology.

## What is deterministic and what is emergent

- Deterministic/reproducible: segmentation version, allowed labels, `source < destination`, type constraints, minimal-predecessor rule, stored annotation prompt/model/seed, and graph metrics.
- Annotated rather than logically guaranteed: which predecessor and relation apply. Replicate annotation and a human audit are required.
- Emergent: branching, merging, long-range dependencies, verification subgraphs, backtracking depth, motif frequencies, and their relation to success/failure.

## Literature-grounded distinction

[Thought Anchors](https://arxiv.org/abs/2506.19143) estimates mechanistic sentence-to-sentence influence by masking/removing a source sentence and measuring later-token KL; expensive resampling validates that approximation. [ReasoningFlow](https://arxiv.org/abs/2606.05402) instead annotates a typed discourse DAG with eight node and fourteen edge types. It reports that mechanistic scores are not substantially better than selecting nearby nodes for recovering discourse edges. Therefore mechanistic and discourse graphs should be compared, not conflated.

RST motivates hierarchical discourse trees, argument mining motivates support/attack structure, and entailment graphs motivate premise-to-conclusion edges. The DAG is preferred here because DoorKey traces can branch and later merge.

## Staged experiment

1. **Schema audit:** manually annotate the shortest matched control/failure pair; require acceptable edge definitions and predecessor agreement.
2. **Replicated pilot:** independently annotate 6–10 complete traces. Compare edge detection/type agreement and a nearest-predecessor baseline.
3. **Graph hypotheses:** test whether failure traces show more backtracking branches, unresolved plans, attack/correction chains, and lower ancestor coverage of the final commitment. Use matched-pair/grouped inference.
4. **Behavior alignment:** test whether graph events explain action-distribution changes beyond distance/progress. BEAST remains retrospective sensitivity only.
5. **Mechanistic validation:** when compute permits, mask selected high-confidence edges and compare later-token/action KL; reserve resampling for a small calibration subset.

No outcome comparison should proceed until stage 2 passes its annotation-reliability gate.
