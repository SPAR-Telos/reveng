# Replaced Attention Analysis

This directory measures attention from the teacher-forced final action token after
the complete reasoning trace. It is retrospective and should not be used to claim
that attention changed when an intermediate action recommendation changed.

The corrected analysis is:

`outputs/hypothesis_tests/prefix_action_attention_v1/`

It measures attention at the last prompt token immediately before the model scores
`UP`, `DOWN`, `LEFT`, and `RIGHT` at each selected reasoning sentence boundary.
