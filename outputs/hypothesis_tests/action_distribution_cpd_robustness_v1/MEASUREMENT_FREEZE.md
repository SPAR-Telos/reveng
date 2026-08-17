# Frozen Definition of the Change-Point Measure

This file records the change-point definition that subsequent analyses must use. It
prevents us from changing the detector after seeing whether another signal predicts
its output.

## Primary outcome

A change point is one of the 160 sentence positions reported in
`../action_distribution_cpd_beast_v1/detected_change_points.csv`.

For each environment state, BEAST was applied to a one-dimensional time series. At
sentence position *t*, the value is the L2 distance between the current four-action
probability distribution and the distribution before reasoning began. The frozen
settings are:

- minimum segment length: 3 sentences;
- maximum number of change points: 10;
- at least 9-to-1 posterior evidence that the trace contains a change;
- posterior probability of at least 0.70 at a reported position;
- 3 chains, 8,000 samples per chain, 200 burn-in samples, thinning by 5;
- random seed: 42.

The detector is offline. Sentences after a candidate position contribute to the
decision that the candidate is a change point. A model trained to anticipate these
labels can use only current and earlier signals, but the labels themselves are not
available in real time.

## Sensitivity outcome

The sensitivity outcome contains the 127 original locations that were found within
3 sentences in at least 12 of the 16 reruns in
`original_point_stability.csv`. This is a stricter subset, not a replacement outcome
and not a newly tuned detector.

## Prediction windows

At each current sentence *t*, the one-sentence outcome asks whether a frozen change
point occurs at *t + 1*. The three-sentence outcome asks whether one occurs at any of
*t + 1*, *t + 2*, or *t + 3*. A change point at the current sentence is never counted
as a future outcome.

## Validation and reporting

- Entire trajectories stay in the same held-out fold.
- The primary ranking metric is area under the precision-recall curve (AUPRC), because
  change-point windows are uncommon.
- Log loss and Brier score assess probability quality; AUROC is secondary.
- The belief result is the held-out change after adding observable belief-change
  features to the current sentence number, current action confidence, and the change in the
  action distribution at the preceding boundary.
- Results must be reported for both the primary and sensitivity outcomes.
- No new BEAST setting is selected on the basis of these prediction results.

## Scope of the claim

Passing this analysis would show that current belief changes contain information
about an upcoming location later marked by offline BEAST. It would not make BEAST a
real-time detector, establish causality, or show that gradual changes are detected.
