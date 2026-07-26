# Figure Captions

## attention_event_window_difference.png

Attention is measured from the final action token to the event sentence and the three sentences on either side. Each point subtracts attention to the closest non-overlapping seven-sentence window in the same state, at a similar reasoning position, that contains no event of the same type. Values are means of trajectory-level differences at layer 15; error bars are 95% confidence intervals across trajectories. Positive values mean greater attention near the event. This is correlational and does not establish that the attended reasoning caused the final action.

## commitment_attention_by_state_group.png

Final-action attention near retrospective action commitment, shown separately for failure and control states. Commitment is the first sentence boundary at which the recommended action equals the full-trace recommendation and remains unchanged. Values compare the seven-sentence commitment window with a non-overlapping same-state window; error bars are 95% confidence intervals across trajectory-level means. The difference between state groups is descriptive and is not a direct between-group significance test.

## top_attention_heads.png

The ten layer-15 heads with the largest positive attention difference near action commitment among heads showing a positive difference in at least four of five trajectory-grouped subsets. Head selection and effect estimation use the same pilot data, so this is an exploratory screen requiring held-out validation, not evidence of a specialized causal circuit.
