# Typed reasoning-DAG edge audit

For each destination node, identify the **minimal complete set** of earlier nodes needed to interpret its contribution. Every edge must point left-to-right. Enter a JSON list such as:

```json
[{"source_node_id":"...__sentence_004","edge_label":"infer"}]
```

Allowed labels:

- `infer`: premise or state information supports a derived claim;
- `restate`: destination repeats/paraphrases the source;
- `support` / `attack`: destination validates or contradicts source content;
- `proceed`: source motivates the next plan;
- `execute`: destination carries out a source plan;
- `decompose`: destination is a subplan of source;
- `verify`: destination initiates checking of source;
- `backtrack`: destination replaces/abandons a source plan;
- `positive` / `negative` / `uncertain`: destination evaluates source;
- `state_update`: destination revises a DoorKey world-state representation;
- `route_update`: destination revises a route or subgoal from source;
- `action_commit`: destination commits to an action derived from source.

Do not use chronological proximity alone. Use `[]` when a reviewed node introduces an independent fact/plan and therefore has no predecessors. A blank cell means **not yet reviewed**. Reviewers A and B annotate independently before adjudication. The BEAST flag and behavioral fields are intentionally absent from `edge_review.csv`; they are joined only after edge reliability is measured.
