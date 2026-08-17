# Not a Change-Point Detector

This directory contains descriptive, forced-maximum summaries: it selects the
largest action-probability jump and largest adjacent distribution change in
every state. It therefore reports a boundary even when a time series has no
statistically supported change point.

Use `outputs/hypothesis_tests/action_distribution_cpd_beast_v1/` for the
offline Bayesian change-point analysis. The historical files remain only for
auditability.
