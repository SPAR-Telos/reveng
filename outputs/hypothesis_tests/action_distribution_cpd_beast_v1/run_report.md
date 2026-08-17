# Offline Action-Distribution Change-Point Detection

## Question

Where does the model's elicited next-action distribution change abruptly as more
reasoning sentences are revealed, and do those changes coincide with recommendation
or action-optimality events?

## Method

For each of 46 fixed DoorKey environment states, the input is the sequence of
temperature 0.7 logprob distributions over `UP`, `DOWN`, `LEFT`, and `RIGHT` measured
after each reasoning sentence. Following the scalar reduction used in Forking Paths,
the analyzed series is the Euclidean distance between the action distribution at each
sentence and the distribution before reasoning.

A trend-only BEAST model fits piecewise-linear segments. A state is considered to
contain a change point when the posterior odds of one or more changes versus no change
exceed 9:1. A location is reported when its
posterior change probability is at least 0.70.
The minimum segment length is 3 sentences, the maximum
number of changes is 10, and the MCMC seed is 42.
No continuation resampling and no added Gaussian noise are used.

## Results

- 44 of 46 states contain at least one detected change point.
- 160 change points are detected in total.
- Failure states have 3.57 detected points on average; control states
  have 3.39. This pilot does not show a substantial group difference.
- 86.9% of detected points are within three sentences
  of a recommendation change, compared with 65.3%
  for progress-matched random sentences (randomization p=0.0005).
- Proximity to optimality loss is weaker: 33.1%
  observed versus 26.2% at random
  (p=0.0060).
- 41.9% are near recovery to an optimal recommendation,
  compared with 29.4% at random
  (p=0.0005).

These are change points in immediate action readouts conditioned on observed reasoning
text. They are not causal forks in sampled continuations. The analysis can identify
candidate sentences for later semantic review or intervention, but it cannot show that
the sentence caused the later action.
