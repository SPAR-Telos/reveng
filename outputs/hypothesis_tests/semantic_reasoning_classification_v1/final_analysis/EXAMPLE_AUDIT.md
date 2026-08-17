# Manual Audit of Representative Semantic Examples

## Procedure

`representative_examples.csv` selects change-point sentences from each broad
semantic class, prioritizing exact optimality-loss/recovery events and high
BEAST posterior probability. I reread each target with its preceding context
and its same-state matched comparison. Selection happened before this review;
the judgments below did not change the primary result.

Of 11 selected change-point examples, seven fine labels are clear, one is
plausible but fragmentary, and three expose genuine taxonomy/judge ambiguity.
The broad matched result should therefore be read with its measured 77.5%
calibration agreement, not as human-ground-truth semantics.

## Clear examples

| ID | Target sentence | Label | Simultaneous action event | Audit |
|---|---|---|---|---|
| `sem_d7568c5e9a73ac92` | “Agent at (3,2).” | state reconstruction | RIGHT → DOWN; optimality loss | Clear direct state readout. |
| `sem_2294acc84d01577a` | “Let's check obstacles: (2,3) open, …, (7,3) goal.” | verification | DOWN → RIGHT; optimality loss and retrospective commitment onset | Clearly checks every cell in a previously proposed route. |
| `sem_be37a7c2af46aa9f` | “Maybe we need to go through row7: …” | route planning | UP → RIGHT; optimality loss | Clearly proposes a new route after two failed candidates. |
| `sem_dc801d2b6c794756` | “So door at (4,2).” | restatement | LEFT → RIGHT; optimality loss | Repeats the location just read from the row. |
| `sem_3db27ebef4630b8e` | “So K at (3,6).” | restatement | UP → DOWN; recovery and retrospective commitment onset | Repeats the key location just read from the row. |
| `sem_fa994f971de1538d` | “So agent at (2,5).” | restatement | RIGHT → UP; recovery | Repeats the agent coordinate just read from the row. |
| `sem_114c50f850628245` | “Let's chart open spaces:” | procedural continuation | Recommendation change and optimality loss | A meta-level heading; the explicit low-confidence adjudication is sensible. |

## Ambiguous or questionable examples

| ID | Assigned label | Audit |
|---|---|---|
| `sem_e8070236d6cf3566` | state reconstruction | “Right (5,4) '#', wall.” is a direct cell readout, but the preceding sentences enumerate candidate moves. Under the guide's precedence, `verification` is at least as plausible and may be preferable. |
| `sem_175cb26020c88933` | state reconstruction | “Row6 col3 is open.” answers the immediately preceding route question. The content is a grid fact, but its discourse function is plausibly `verification`. |
| `sem_65277fcc321e9740` | verification | “So path: …” summarizes a route already established by several checks. `consolidation` is probably preferable; `route_planning` is also plausible. |
| `sem_04a046e54166e913` | procedural continuation | “From agent at (6,7).” is an incomplete connective into a route trace. `procedural_continuation` is plausible, but confidence should remain medium. |

These are not random mistakes: they localize the hardest boundary in the
taxonomy—whether a grid fact functions as a state readout or as verification
of a route currently under discussion. That is why the nine-way labels remain
sensitivity analyses.

## What the examples establish

The examples make the quantitative null interpretable. High-posterior action
change points can occur on:

- a bare coordinate readout;
- an explicit path check;
- a newly proposed route;
- a repeated object location;
- or a meta-level transition.

Matched controls also often perform nearby functions. The semantic diversity
is real enough that the result should not be framed as “change points are
verification sentences” or “commitment happens at summaries.” The defensible
claim is that retrospective action-routing boundaries cut across several
verbal reasoning functions.

For quoted illustrations, prefer the seven clear examples above. Preserve the
three questionable cases as evidence of annotation uncertainty rather than
silently relabeling them after seeing their action events.
