# Human-calibrated semantic labels (v3)

These are the canonical names emitted by both the local and Together runners.
Historical v1/v2 outputs are not rewritten.

| Label | Meaning | Main boundary |
|---|---|---|
| `state_readout` | States an observed environment/agent fact, or the coordinate reached by a definite numbered step. | A coordinate in a tentative proposal is `route_planning`. |
| `route_planning` | Explores a possible move, route, subgoal, or unselected sequence. | A selected concrete step is `action_commitment`, not automatically route planning. |
| `verification` | Checks an earlier fact, move, or route. | Merely stating a new fact is `state_readout`; changing a rejected claim is `correction`. |
| `correction` | Rejects, revises, or reverses earlier reasoning. | A question or check without revision is `verification`. |
| `new_inference` | Derives a new consequence, constraint, comparison, or conclusion. | Direct transcription is `state_readout`; proposing movements is `route_planning`. |
| `consolidation` | Combines several earlier findings into a summary or decision. | Repeating one proposition is `restatement`. |
| `restatement` | Repeats the same earlier substantive proposition without changing it. | Repeated format or bookkeeping is `procedural_continuation`. |
| `procedural_continuation` | Setup, formatting, bookkeeping, or connective narration with no substantive claim. | Use only when no substantive label applies. |
| `action_commitment` | Records a selected concrete move or explicitly chooses/recommends the next move. | Possible or alternative moves are `route_planning`. |

## Legitimate overlaps

- `verification` + `state_readout`: reads a cell and uses it to check a route.
- `verification` + `new_inference`: evaluates a route and derives its feasibility.
- `correction` + `state_readout`: a newly read state fact overturns an earlier claim.
- `correction` + `route_planning`: rejects an earlier route and explores a replacement.
- `correction` + `action_commitment`: rejects an earlier choice and selects a replacement move.
- `consolidation` + `action_commitment`: combines earlier evidence into a selected move.
- `action_commitment` + `state_readout`: a definite numbered step states both the move and resulting coordinate.

## Ambiguous boundaries to audit

- A route list can be either exploration (`route_planning`) or an already selected sequence (`action_commitment`). Wording and preceding context decide.
- A coordinate may be observed state (`state_readout`), a definite projected state (`state_readout` + `action_commitment`), or hypothetical (`route_planning`).
- “So” does not determine `restatement`, `new_inference`, or `consolidation`; compare the proposition with the preceding context.
- `verification` can reuse prior content, but `restatement` should be added only if repetition is itself a distinct function.
- `procedural_continuation` is exclusive: if the sentence contains a substantive state, route, check, revision, inference, summary, repetition, or commitment, use that label instead.

