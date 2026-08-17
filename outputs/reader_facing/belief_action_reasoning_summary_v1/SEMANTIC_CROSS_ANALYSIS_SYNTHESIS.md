# Semantic Labels Across the Experiment Chain

## Bottom line

The semantic labels sharpen the project mainly by ruling out an overly simple
mechanism. Retrospective action-distribution change points are behaviorally meaningful,
but they are not a stable class of corrections, verifications, summaries, or other
named discourse operations. Broad sentence function also does not reliably explain
which change points are recommendation changes, optimality losses, recoveries, or
commitment onsets, and it does not localize the sentence-level activation transition.

The useful positive result is narrower: route-deliberation sentences mark a
high-uncertainty belief regime, and semantic labels identify concrete optimality-loss
sentences where all nine categorical belief readouts at that boundary are correct. The
labels are therefore better for interpretation and intervention sampling than for
monitoring.

## What changes in the central claim

Recommended wording:

> Task-relevant state information can remain recoverable while reasoning redirects the
> model's action distribution. These redirects occur across several verbal reasoning
> functions rather than at one stereotyped “aha” or correction sentence. Sentence
> function is therefore not the missing conversion mechanism itself; it helps identify
> uncertainty regimes and candidate reasoning content for causal intervention.

This is sharper than saying that action changes are “semantic transitions.” The current
evidence instead supports a **reasoning-mediated conversion failure** whose observable
sentence-level form is heterogeneous.

## Cross-analysis conclusions

| Analysis | What the labels contribute | Conclusion |
|---|---|---|
| Previous belief--action-gap evidence | Interpretive bridge only; the old failure slice was not labeled with this taxonomy | Correct local reports can coexist with inconsistent actions, but current labels do not retrospectively explain those earlier cases |
| BEAST change points | Direct matched test | No reliable four-way semantic signature across 152 close pairs |
| Exact action and optimality events | Direct semantic join | No event association survives correction; smallest q = 0.406 |
| Commitment and post-commitment reasoning | Partial direct join | Commitment onset has no semantic association; the labels do not cover all tail sentences, so they cannot establish that post-commitment text is semantically useless |
| Activation transitions | Direct semantic join | No activation-geometry association survives correction; smallest q = 0.106 |
| Behavioral belief transitions | Direct semantic join | Belief error does not vary reliably, but belief entropy does: route deliberation averages 0.493 bits, q = 0.008 |
| Immediate-action attention | Direct join on 67 matched attention pairs | The increase is largest for summary/restatement sentences (0.224 percentage points per token), but semantic heterogeneity is exploratory (p = 0.053) |
| Prospective event monitor | Not a direct join: its “text semantic” features are embedding novelty, not these labels | Do not attribute its small large-distribution-change result to the four-way taxonomy |

The machine-readable version is `semantic_cross_analysis_decision_table.csv`.

## How this sharpens the experiment chain

**Previous behavioral and causal evidence.** The paper's strongest behavioral example
is that all three eligible wall-hit cases preserve a correct local wall report while
the model still takes the blocked action. Its strongest intervention result is that
patched outputs follow the donor reasoning action in 100% of cases, with a 76% action
flip rate, whereas pre-reasoning activation patching alone has little detectable
effect. The current semantic labels do not re-estimate those effects because they were
collected on a different matched sentence cohort. They sharpen the follow-up question:
which *kind of reasoning content* should be deleted or replaced when correct measured
beliefs coexist with an optimality loss?

**Action distributions and BEAST.** BEAST supplies retrospective candidate boundaries,
not a linguistic mechanism. The semantic null is useful here: it prevents us from
renaming statistical change points as correction, verification, or commitment
sentences. The strongest warranted interpretation remains an action-routing boundary.

**Activation and belief-transition analyses.** Sentence activation geometry is
associated with some same-position events but is not a reliable prospective failure
monitor, and its variation is not explained by broad semantic function. Behavioral
belief readouts do provide modest three-sentence warning of optimality loss, while the
semantic labels instead distinguish the uncertainty regime at the boundary. These are
complementary roles: beliefs carry the prospective signal; semantics interprets what
the model is verbally doing.

**Attention.** The immediate action readout increases attention to the newest sentence
when its recommendation changes. The semantic breakdown suggests that compressed
summaries or restatements may receive the largest increase, but the heterogeneity test
is only borderline. This is a prioritization result for patching, not evidence that
summaries generally cause action changes.

## Expected and counterintuitive observations

Expected:

- Route deliberation dominates both change-point and comparison sentences and is
  numerically more common at change points.
- Belief uncertainty is highest while the model is checking, revising, or constructing
  routes.

Counterintuitive:

- Corrections, checks, and summaries do not form a distinctive change-point class.
- Semantic function does not reliably distinguish harmful action changes from
  recoveries.
- The attention increase is numerically largest for summaries/restatements, whereas
  belief uncertainty is largest for route deliberation. Attention routing and belief
  uncertainty therefore appear to be different parts of the process.
- 7 of 18
  exact optimality losses at primary change points
  (38.9%) occur with correct answers on
  all nine available categorical belief readouts at that boundary. Their semantic
  functions are heterogeneous.

## How the labels should be used

Use them to:

1. prevent post-hoc “aha moment” stories;
2. stratify belief uncertainty by reasoning function;
3. select heterogeneous intervention candidates;
4. choose clear qualitative examples while preserving annotation uncertainty.

Do not use them as:

- a real-time change-point predictor;
- human-ground-truth reasoning states;
- token-level mechanism labels;
- evidence that a discourse function caused an action transition.

## Best causal follow-up

Start with `semantic_intervention_candidates.csv`. These rows are primary BEAST
change points where the recommendation becomes suboptimal while all nine available
categorical belief readouts are correct. Delete, replace, or resample the target
sentence, then re-elicit the immediate action distribution and the same belief probes.
Stratify the sample across state readout, route deliberation, summary or restatement,
and other functions instead of selecting only the most intuitive examples.

If changing a sentence reverses the action without changing the measured beliefs, that
would directly support a failure in converting available information into action. If
the belief reports also change, the result is better described as a belief-mediated
transition. If neither changes, the sentence is a marker rather than a causal driver.

## Scope

The semantic taxonomy is AI-assisted and has stronger reliability at the broad
four-way level than at the fine-label level. BEAST is retrospective, and the unit is a
whole sentence. The results do not rule out sparse token-level forks within otherwise
ordinary sentences.
