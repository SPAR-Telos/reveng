# Semantic Reasoning Annotation Guide

## Question

What function does the target sentence perform relative to the reasoning that
precedes it? Judge the sentence's discourse function, not whether its content
is factually correct or whether the model's action is optimal.

The annotation file is blinded: it does not reveal whether a sentence is a
detected action-distribution change point or a matched comparison sentence.

## Primary Labels

Choose exactly one label.

| Label | Use when |
|---|---|
| `correction` | The sentence explicitly rejects, revises, or reverses an earlier fact, route, or conclusion. Cues such as “wait”, “actually”, or “no” count only when they perform a real revision. |
| `verification` | The sentence checks whether an earlier fact or proposed route is valid, traversable, or efficient without yet replacing it. |
| `state_reconstruction` | The sentence directly reads, records, or locates an environment feature from the grid, such as an object coordinate, wall, or open cell. |
| `route_planning` | The sentence proposes a new action, route, subgoal, or action sequence. |
| `new_inference` | The sentence derives a new consequence, constraint, comparison, or conclusion that is not primarily a route proposal. |
| `consolidation` | The sentence combines multiple earlier facts or conclusions into a summary, plan, or decision without adding a materially new premise. |
| `restatement` | The sentence repeats one earlier fact, route, or conclusion without materially checking, combining, or changing it. |
| `procedural_continuation` | The sentence continues enumeration, arithmetic, bookkeeping, or connective narration and does not fit a more specific category. |
| `unclear` | The available context is insufficient or two labels remain equally plausible after applying the rules below. |

## Decision Order

Apply these questions in order:

1. Does it explicitly revise or reject earlier reasoning? Use `correction`.
2. Does it evaluate an earlier claim or route? Use `verification`.
3. Does it directly transcribe the current grid? Use `state_reconstruction`.
4. Does it construct a new route or action sequence? Use `route_planning`.
5. Does it derive a new consequence or conclusion? Use `new_inference`.
6. Does it combine several earlier results? Use `consolidation`.
7. Does it repeat one earlier result? Use `restatement`.
8. Otherwise use `procedural_continuation` or, if context is inadequate, `unclear`.

This precedence makes the labels mutually exclusive. For example, a sentence
that repeats a route in order to test it is `verification`, not `restatement`.
A sentence that says “Actually, that route is blocked” is `correction`, not
`verification`.

## Observable Cue Columns

Fill each with `yes`, `no`, or `unsure`.

| Column | Criterion |
|---|---|
| `explicitly_revises_prior_reasoning` | The sentence rejects or changes an earlier claim, route, or decision. |
| `evaluates_prior_route_or_claim` | The sentence tests an earlier proposal or factual claim. |
| `repeats_prior_content` | Substantive content already appears in the preceding context. |
| `introduces_new_information_or_plan` | The sentence adds a new fact, inference, route, or decision. |

The cue columns are not labels. A correction may both repeat prior content and
introduce a revised conclusion.

## Confidence

- `high`: the primary label follows directly from the wording and context.
- `medium`: one label is preferable but a nearby category is plausible.
- `low`: the label is tentative; use a short rationale explaining the ambiguity.

## Consistency Rules

- Do not use action correctness, failure status, or change-point status; these
  fields are intentionally hidden.
- Treat “then”, “wait”, and “therefore” as cues, not labels.
- A list of grid cells is `state_reconstruction` unless it evaluates a route.
- A proposed sequence of movements is `route_planning`.
- “This path works” after checking several constraints is `consolidation`.
- Use only preceding context. Do not infer intent from later sentences.
- Give the same label to semantically identical duplicate items. Four hidden
  duplicates in the pilot measure within-annotator consistency.

## Pilot Procedure

1. Label `human_pilot_annotation.csv` without opening the key file.
2. Review all `low`-confidence and `unclear` rows once.
3. Return the completed CSV.
4. We will report duplicate agreement and compare the labels with an independent
   temperature-zero judge before labeling the full set.
5. After the full 320-row template is adjudicated, train the supervised
   classifier with `scripts/train_semantic_reasoning_classifier.py`. The pilot
   is too small for reported classifier performance.
