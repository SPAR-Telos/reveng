# Evidence Audit and Central Claim

## Defensible conclusion

In these DoorKey trajectories, failures cannot be explained only by absent state
information. Relevant state facts are often recoverable, while action selection changes
with the model's reasoning context. Behavioral belief readouts provide modest advance
warning of optimality loss, abrupt action-probability changes occur near recommendation
changes, and the immediate action readout places more attention on the newest reasoning
sentence when its recommendation changes. Together with the paper's reasoning-trace
interventions, the evidence supports reasoning-mediated action selection rather than a
simple loss-of-state-knowledge account.

This conclusion is specific to GPT-OSS-20B in DoorKey and should use "some failures,"
not "all failures."

## Evidence and limits

| Evidence | Result | Role in the conclusion | Main limitation |
|---|---|---|---|
| State and value readouts in the paper | State facts are decodable, and inferred values favor optimal actions more often than realized actions | Establishes information availability and a belief-action discrepancy | Probe validity and task scope |
| Reasoning-trace intervention in the paper | Patched reasoning determines the final action far more strongly than pre-reasoning activation changes | Strongest evidence that reasoning context controls the final action | The current paper table should report intervention sample sizes and uncertainty |
| Prospective belief monitor | Three-sentence optimality-loss AUPRC rises from 0.273 to 0.334, change +0.061, 95% CI [+0.014, +0.123] | Shows practical short-horizon association between observable beliefs and failure | Next-sentence result is inconclusive; only 31 trajectories |
| Offline action-distribution change points | 86.9% are near recommendation changes versus 65.3% for progress-matched sentences | Identifies candidate reasoning boundaries for intervention | Offline and correlational; earlier probability vectors are not exactly reproducible |
| Immediate action attention | Layer-15 attention to the newest reasoning sentence rises by 0.0838 percentage points per token, 95% CI [0.0442, 0.1314] | Suggests changed recommendations are associated with the newest reasoning content | Attention is not causal evidence; only 67 matched pairs |
| Supervised activation monitors | Best activation AUROC is below behavioral readouts for every event | Rules out a simple claim that generic activation features are the best monitor | Monitors use selected layers and aggregate sentence representations |
| Immediate belief-transition models | Current-state and action-consequence readouts do not robustly improve immediate transition prediction | Prevents claiming that measured belief changes directly explain action changes | Probe set and behavioral elicitation may miss the relevant transition belief |
| Blinded sentence-function labels | No reliable change-point signature or exact action-event profile; route deliberation has the highest belief entropy | Rules out a single verbal-operation account and identifies an uncertainty regime | AI-assisted sentence labels; offline change points; no token or causal localization |

## Claims not supported

- Measured belief changes directly cause recommendation changes.
- Attention proves that grid information was used or ignored.
- Generic activation drift reliably predicts failure.
- Action uncertainty necessarily decreases at commitment.
- Reasoning after retrospective commitment is semantically useless.
- Corrections, verifications, or summaries are a general mechanism for action changes.

## Strong next hypothesis

A reasoning sentence can redirect the action distribution despite stable, recoverable
state information. Test sentences where the recommendation changes from optimal to
suboptimal, state-belief reports remain correct, and attention to the newest sentence
increases. Delete or replace that sentence and re-evaluate the immediate action
distribution. The hypothesis predicts restoration of the previous or planner-optimal
action without a corresponding change in state-belief reports.

This intervention would join the current descriptive results into one causal test:
information remains available, a particular reasoning sentence changes action
selection, and changing that sentence reverses the failure.

## Candidate sentences

Across the matched run, 66 of 161 optimal-to-suboptimal transitions have the same
correct answers immediately before and after the transition for wall-left,
wall-right, wall-up, wall-down, key possession, and door status. Representative
candidates include:

| Environment state and position | Sentence | Elicited action change | Transition |
|---|---|---|---|
| `keepdoor_0`, step 2, sentence 9 | "Row4: # _ # # # D # # #." | LEFT to RIGHT | Sustained optimal to suboptimal |
| `keepdoor_26`, step 8, sentence 76 | "From (5,3) go up to (4,3) which is '#', blocked." | UP to RIGHT | Sustained optimal to suboptimal |
| `keepdoor_33`, step 1, sentence 28 | "Must pick key before door if door blocks path." | DOWN to LEFT | Sustained optimal to suboptimal |

These are observational candidates. The action was elicited after each sentence, so
temporal coincidence does not show that the sentence caused the action change.

The semantic-label join yields a stricter candidate set:
7 of
18 exact optimality losses at primary
BEAST change points have correct answers across all nine categorical belief readouts
at that boundary. See `semantic_intervention_candidates.csv`; semantic functions are
deliberately heterogeneous so a causal follow-up does not select only intuitive
examples.
