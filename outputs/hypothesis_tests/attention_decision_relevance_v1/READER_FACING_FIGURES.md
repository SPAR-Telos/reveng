# Reader-Facing Attention Results

## Recommended figures

### Final-action attention near reasoning events

Use `figs/attention_event_window_difference.png` as the main result. It compares
the share of attention assigned to seven sentences around an event with a
non-overlapping window from the same state at a similar reasoning position.
The comparison window contains no event of the same type. Intervals are
computed across trajectory-level means.

The main pattern is concentrated near retrospective action commitment. The
all-state estimate is 2.99 percentage points, with a 95% interval from -0.28 to
6.26 at the prespecified layer 15. The direction is the same at layer 23, where
the estimate is 0.15 percentage points with an interval from 0.02 to 0.28. Raw
attention shares are not directly comparable across layers. Changes in
reported action consequences are a weaker candidate. There is
no clear positive signal near recommended-action changes, sustained or
transient changes to a suboptimal action, or recoveries to an optimal action.

### Attention near commitment by state group

Use `figs/commitment_attention_by_state_group.png` only as a subgroup result.
The estimate is larger for failure states, 4.97 percentage points, than for
control states, 1.33 percentage points. This difference is descriptive: the
figure does not test the contrast between groups. Of the 46 states, 43 have a
commitment event and full event and comparison windows suitable for this test.

### Exploratory attention heads

Keep `figs/top_attention_heads.png` in an appendix. It identifies candidate
layer-15 heads for held-out confirmation. The same pilot data were used to
select and estimate these heads, so the plot must not be described as a
replicated circuit result.

## Interpretation

The pilot suggests that the final action token may preferentially retrieve
reasoning near the point at which the recommended action stabilizes. It does
not show preferential retrieval near optimality loss or recovery, and it does
not establish that attention caused the action. This is evidence about where
the final action token attends, not proof that the attended information was
used correctly.

## Signals for the next stage

1. Confirm the commitment result on held-out trajectories. Prespecify layer 15,
   the seven-sentence window, and the candidate heads before evaluation.
2. Compare final-action attention with attention from matched non-action output
   tokens. This tests whether the effect is specific to choosing an action
   rather than a generic property of late output tokens.
3. If the held-out effect survives, ablate or patch the selected heads and
   measure changes in the final action. This is the first test that can support
   a causal information-use claim.
4. Aggregate sentence attention within environment steps as a robustness
   analysis; keep sentence boundaries as the primary resolution.

## Important extraction check

Layer 8 is omitted from the current analysis. GPT-OSS-20B uses 128-token
sliding-window attention at that layer, and the original aggregate extractor
mapped its retained key positions incorrectly. The extractor is corrected for
future runs, but the existing layer-8 shards must be regenerated before use.
Layers 15 and 23 are full-attention layers and remain usable.
