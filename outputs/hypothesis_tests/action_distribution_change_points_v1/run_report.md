# Sentence-Level Action-Distribution Change Points

## Scope

- Model: GPT-OSS-20B
- States: 46 matched DoorKey states, 23 failure and 23 control
- Trajectories: 31
- Sentence positions: 7,084
- Action distribution: candidate-token logprobs at temperature 0.7, normalized over UP, DOWN, LEFT, and RIGHT
- New model inference: none
- Claim audit: `MAIN_FINDING_ASSESSMENT.md`

## Main Results

- Median final-action jump abruptness ratio: 1.268 (trajectory-bootstrap interval [1.185, 1.410]).
- Median absolute offset between the largest probability jump and stable-action boundary: 16.0 sentences.
- Fraction of states whose boundaries are within three sentences: 0.391.
- A random valid sentence is within three sentences of stable action selection in 0.064 of states on average (permutation p=0.0002).
- The largest probability jump occurs before stable action selection in 29/46 states, at the same sentence in 15/46, and after it in 2/46.
- Mean local post-minus-pre Jensen-Shannon distance to the full-trace distribution: -0.5282 bits. Negative values mean the action distribution becomes more like the full-trace distribution after the jump.
- Mean jump position by reasoning-character progress: failure=0.565, control=0.533.

## Conclusion

The largest rise in final-action probability is temporally related to stable action selection beyond a random-position baseline, and the distribution becomes substantially closer to its full-trace value. However, the largest jump is only modestly larger than competing jumps and is not interchangeable with stable commitment. The two boundaries are more than three sentences apart in most states, and the largest jump usually comes first. The supported interpretation is a two-stage descriptive account: an action becomes much more probable, then later becomes the recommendation that remains stable.

This does not yet identify what semantic operation causes either transition. The analysis uses truncated-prefix action elicitation and retrospective full-trace information, so causal validation requires sentence truncation or replacement around the selected boundaries.

## Internal Correlates

- Jensen-Shannon distance to the full-trace distribution decreases by 0.523 bits more than at same-state progress-matched positions over the five-sentence window (trajectory-bootstrap interval [0.427, 0.610]).
- Cosine similarity to the mean of preceding sentence activations is higher at the probability jump by 0.0119 (interval [0.0052, 0.0194]).
- Mean wall, key, and door belief error differs from matched positions by -0.0036 (interval [-0.0122, 0.0000]).
- At layer 23, final-action attention to the seven-sentence window around stable action selection exceeds its matched window by 0.0015 (interval [0.0003, 0.0028]).
- Adding current-sentence activation similarity to reasoning progress and action confidence changes held-out AUROC for predicting a probability jump at the next sentence by -0.006 (trajectory-bootstrap interval [-0.021, 0.007]; within-state circular-shift p=0.5423).
- For the next sentence's largest change in the four-action distribution, the same added feature improves AUROC by 0.099 (interval [0.008, 0.190]; corrected p=0.0448). The AUROC direction is positive in 4/6 layer-representation checks, but log-loss improvement is approximately zero.

These comparisons use same-state progress-matched controls. They are supporting associations, not causal evidence.

## Failure Versus Control States

- Mean probability-jump position is 0.565 for failure states and 0.533 for control states.
- The paired failure-control estimates and bootstrap intervals are in `matched_group_difference_summary.csv`. The pilot should not be read as establishing a group difference unless the corresponding interval excludes zero.

## Limitation

The stored probabilities are normalized over the four action candidates. They do not report probability mass assigned to non-action text.
The jump is selected from the same probability series used to measure its magnitude. The random-position comparison validates timing agreement, but the local probability change around the selected maximum is descriptive by construction.
