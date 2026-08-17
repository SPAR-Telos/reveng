# Appendix Cases for Activation Similarity

These cases illustrate optimal-to-suboptimal recommendation changes at sentence
positions with high cosine similarity to the mean of earlier sentence
representations. They are examples for qualitative inspection, not independent
evidence for the aggregate association.

The similarity is computed at GPT-OSS-20B layer 15 from the mean activation over
the current reasoning sentence and the mean of the sentence-mean activations
from all earlier reasoning sentences for the same environment state.

| Environment state | Failure definition | Recommendation change during reasoning | Similarity | Reasoning sentences |
|---|---|---|---:|---|
| `keepdoor_57`, environment step 2 | Wall hit. The trajectory's observed `RIGHT` action is planner-suboptimal; it leaves the agent in the same cell, while `LEFT` is optimal. | Sustained optimal-to-suboptimal change at sentence 6: `LEFT` to `UP`. | 0.963 | Sentences 5–6 below |
| `keepdoor_59`, environment step 2 | Avoidable detour. The observed `RIGHT` action is planner-suboptimal without hitting a wall; `DOWN` is optimal. | Sustained optimal-to-suboptimal change at sentence 84: `DOWN` to `RIGHT`. | 0.966 | Sentences 83–84 below |
| `keepdoor_16`, environment step 11 | Backtrack. The observed `UP` action returns toward the previous state and is planner-suboptimal; `DOWN` and `RIGHT` are tied optimal actions. | Sustained optimal-to-suboptimal change at sentence 33: `RIGHT` to `UP`. | 0.968 | Sentences 32–33 below |

## Exact Reasoning Text

### `keepdoor_57`, environment step 2

Sentence 5:

> The door is at (1,4).

Sentence 6, where the recommended action changes from `LEFT` to `UP`:

> The goal at (1,3).

### `keepdoor_59`, environment step 2

Sentence 83:

> We could go right to (2,3), down to (3,3), down to (4,3), down to (5,3), down to (6,3), down to (7,3).

Sentence 84, where the recommended action changes from `DOWN` to `RIGHT`:

> Let's check obstacles: (2,3) open, (3,3) open, (4,3) open, (5,3) open, (6,3) open, (7,3) goal.

### `keepdoor_16`, environment step 11

Sentence 32:

> Then down to (7,7).

Sentence 33, where the recommended action changes from `RIGHT` to `UP`:

> Need to avoid door at (7,4) which is on the path from (3,7) to (7,7) along column 7?

## Interpretation

The examples are compatible with a consolidation or route-rechecking
interpretation: the current sentence stays representationally close to the
earlier reasoning context just as the recommended action becomes suboptimal.
They do not establish that reiteration causes the action change. The next
confirmatory analysis should compare these sentences with reasoning-progress
matched non-event sentences using blinded semantic labels.
