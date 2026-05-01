# Behavioral Probe Case Studies

- Slice type: `balanced_failure_modes`
- Instances: `16`
- `state just before tagged failure`: the immediately preceding distinct state, included only as comparison context; it is not itself a failure-mode label.
- `tagged failure state`: the single trajectory step selected because it matches a deterministic failure-mode rule such as wall hit, backtrack, loop, freeze, or avoidable detour.
- Probes remain Markovian: each row is queried independently as one rendered state, with no trajectory history in the prompt.
- Main figure: tagged failure states only. Appendix figure: tagged failure states plus comparison states immediately before them.

| Example | Row type | Mode | Observed action | Optimal actions | Selected | Reason |
|---|---|---|---|---|---|---|
| together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_0_step_002 | tagged failure state | backtrack | DOWN | DOWN, LEFT | True | backtrack |
| together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_0_step_003 | tagged failure state | short_loop | LEFT | LEFT | True | short_loop_onset |
| together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_10_step_002 | tagged failure state | backtrack | RIGHT | RIGHT | True | backtrack |
| together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_11_step_006 | tagged failure state | short_loop | LEFT | LEFT | True | short_loop_onset |
| together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_15_step_000 | tagged failure state | avoidable_detour | DOWN | RIGHT | True | avoidable_detour,suboptimal_success_failure_step |
| together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_15_step_007 | tagged failure state | backtrack | UP | DOWN | True | avoidable_detour,backtrack,suboptimal_success_failure_step |
| together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_16_step_012 | tagged failure state | oscillation_2cycle | DOWN | DOWN | True | backtrack,oscillation_2cycle_onset,suboptimal_success_failure_step |
| together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_16_step_016 | tagged failure state | wall_hit | RIGHT | DOWN | True | suboptimal_success_failure_step,wall_hit |
| together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_20_step_009 | tagged failure state | wall_hit | UP | LEFT | True | suboptimal_success_failure_step,wall_hit |
| together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_20_step_010 | tagged failure state | freeze_repeat | LEFT | LEFT | True | freeze_repeat_onset,short_loop_onset,suboptimal_success_failure_step |
| together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_33_step_003 | tagged failure state | oscillation_2cycle | RIGHT | DOWN, RIGHT | True | backtrack,oscillation_2cycle_onset,suboptimal_success_failure_step |
| together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_57_step_003 | tagged failure state | freeze_repeat | LEFT | LEFT | True | freeze_repeat_onset,suboptimal_success_failure_step |
