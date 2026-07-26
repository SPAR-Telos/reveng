# Main Finding Assessment

## Candidate Main Finding

GPT-OSS-20B action selection is better described by two reasoning boundaries
than by one commitment point. The probability assigned to the eventual action
often rises sharply before the model reaches the later sentence after which
that action remains its recommendation.

This result connects the existing action-probability, action-confidence,
activation, attention, and post-commitment analyses. It does not validate the
stronger claim that measured current-state belief changes cause action changes.

## Evidence

| Result | Estimate | Interpretation |
|---|---:|---|
| Largest final-action probability jump within 3 sentences of stable selection | 39.1% of 46 states | The boundaries sometimes agree, but usually do not |
| Random valid sentence within 3 sentences of stable selection | 6.4%; permutation p = 0.0002 | The observed relationship is stronger than chance given each trace length |
| Median absolute boundary separation | 16 sentences | Probability revision and stable selection should not be treated as the same event |
| Ordering | jump before stable selection in 29 of 46 states | The major probability revision usually occurs first |
| Change toward the full-trace action distribution | 0.523-bit larger decrease in Jensen-Shannon distance than matched positions; interval [0.427, 0.610] | The selected jump is followed by a distribution closer to the final one |
| Failure minus control jump timing | 0.032 of reasoning characters; interval [-0.143, 0.208] | No reliable failure-control timing difference |
| Wall, key, and door error difference at the jump | -0.0036; interval [-0.0122, 0.0000] | Current-state belief errors do not explain the boundary |
| Layer-15 sentence-mean activation similarity at the jump | +0.0119 versus matched positions; interval [0.0052, 0.0194] | The probability revision has a contemporaneous activation correlate |
| Activation similarity predicting the next largest distribution change | AUROC improvement +0.099; interval [0.008, 0.190]; corrected p = 0.0448 | Exploratory leading signal, but log-loss improvement is approximately zero |
| Activation similarity predicting the next final-action jump | AUROC improvement -0.006; interval [-0.021, 0.007] | No leading signal for the final-action probability boundary |
| Activation similarity predicting the next stable selection | AUROC improvement -0.007; interval [-0.020, 0.006] | No leading signal for stable action selection |
| Final-action attention near stable selection, layer 23 | +0.0015 attention mass; interval [0.0003, 0.0028] | The final action token attends more to reasoning near stable selection |
| Action entropy after stable selection | -0.356 bits relative to the pre-boundary window; interval [-0.471, -0.246] | Confidence continues to increase after the recommendation becomes stable |
| Reasoning remaining after stable selection | mean 39.4 sentences and 27.6% of reasoning characters | Considerable reasoning remains after retrospective stable selection |

## Activation Interpretation

For the prespecified layer-15 sentence-mean representation, similarity to the
mean of preceding sentence activations is lower one sentence before the largest
action-distribution change and higher at the change sentence. The event-minus-
previous difference is 0.0171 with a trajectory-bootstrap interval of
[0.0109, 0.0234].

The direction is clearest for sentence means. AUROC improves in four of six
layer-representation checks, while sentence-final representations are
inconsistent. This may indicate a sentence-level content effect, but it may
also reflect pooling, sentence length, or semantic category. It is not yet an
activation mechanism.

## Relation to Existing Conclusions

- The result strengthens the action-selection account by identifying separate
  probability-revision and stable-selection stages during reasoning.
- It is consistent with post-commitment reasoning because a mean 27.6 percent
  of reasoning characters remains after stable selection, but it does not show
  that this text is causally inert.
- It does not strengthen the claim that current-state belief changes trigger
  action changes. The matched belief-error comparison is null.
- It provides a more specific target for the existing attention result:
  sentence-mean activation dynamics are associated with the earlier
  distribution revision, while final-action attention is clearest near later
  stable selection.

## Confidence

Confidence is moderate for the descriptive two-stage finding because it uses
46 states, a state-length-preserving random-position null, matched controls,
trajectory bootstrap intervals, and exact reproduction of all stable-action
boundaries.

Confidence is low for the proposed internal mechanism. The leading activation
result is representation-dependent, has negligible log-loss improvement, and
both boundaries are defined retrospectively from truncated-prefix action
readouts. Attention and activation associations are not causal.

## Tests Needed for a Stronger Claim

1. Semantically classify the 137 blinded boundary-control sentence pairs in
   `semantic_judge_candidates.jsonl`. Test whether the sentence before the
   distribution change introduces a new inference and whether the change
   sentence performs reiteration, route rechecking, or consolidation.
2. Intervene separately on the sentence before the distribution change, the
   distribution-change sentence, and the stable-selection sentence. Compare
   deletion, semantically different replacement, and same-state control
   conditions using immediate four-action logprobs.
3. Truncate or replace reasoning after both boundaries and generate
   continuations. This distinguishes an early probability revision from the
   later point after which reasoning is behaviorally inert.
4. Replicate the sentence-mean activation result on held-out states and a
   second model before calling it a monitor.

The primary causal prediction is that modifying the distribution-change
sentence should alter the immediate action distribution more than modifying a
progress-matched sentence. If only post-stable text can be removed without
changing the action, the two-stage account can be connected to
post-commitment, behaviorally inert reasoning.
