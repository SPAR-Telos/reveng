# Semantic Intervention Candidates

Primary retrospective change points where the recommendation changes from
planner-optimal to suboptimal and all nine categorical belief readouts
(six current-state and three chosen-action consequences) are correct at
that sentence boundary.

| ID | Sentence | Semantic function | Action shift | Belief entropy | BEAST posterior |
|---|---|---|---|---:|---:|
| `sem_114c50f850628245` | “Let's chart open spaces:” | procedural continuation | DOWN → RIGHT | 0.218 | 1.000 |
| `sem_a03a52dbec7aebd5` | “From (2,6) down to (3,6) open, down to (4,6) '#', blocked.” | verification | DOWN → LEFT | 0.668 | 0.921 |
| `sem_8920eb9deed5d1a1` | “But maybe we can go to row2 col3 via top route without passing door: e.g., go up to row1 col3, left to row1...” | verification | RIGHT → UP | 0.742 | 0.990 |
| `sem_2294acc84d01577a` | “Let's check obstacles: (2,3) open, (3,3) open, (4,3) open, (5,3) open, (6,3) open, (7,3) goal.” | verification | DOWN → RIGHT | 0.359 | 1.000 |
| `sem_b1fb5688d53cf752` | “Agent A at (2,6).” | state reconstruction | RIGHT → LEFT | 0.223 | 0.951 |
| `sem_e8070236d6cf3566` | “Right (5,4) '#', wall.” | state reconstruction | RIGHT → LEFT | 0.279 | 0.985 |
| `sem_dc801d2b6c794756` | “So door at (4,2).” | restatement | LEFT → RIGHT | 0.244 | 0.999 |

These are observational candidates, not causal examples. Zero probe
error covers the nine available categorical questions; it does not prove that
every task-relevant latent belief is correct. Fine semantic labels should
also be checked against the context before quoting an individual row.
