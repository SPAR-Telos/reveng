# Planning Alignment Summary

Hypothesis tested:

> reasoning degrades long-horizon planning information while sharpening local action-readiness

Current released-slice result:

- Activation side: prefix-1 delta (post-pre) = 0.067
- Activation side: prefix-5 delta (post-pre) = 0.267
- Activation side: prefix-10 delta (post-pre) = 0.200
- Textual plan side: prefix-1 delta (post-pre) = 0.000
- Textual plan side: prefix-5 delta (post-pre) = -0.067
- Textual plan side: prefix-10 delta (post-pre) = -0.067

Interpretation:

- On this clean released public slice, the activation-side plan decoder does not support long-horizon degradation; post-reasoning decoding improves at both short and longer prefixes.
- The textual plan analogue is flatter at prefix-1 and slightly worse at longer prefixes.
- This makes the released slice useful for pipeline validation and hypothesis falsification checks, but not yet for a strong failure-focused claim.
