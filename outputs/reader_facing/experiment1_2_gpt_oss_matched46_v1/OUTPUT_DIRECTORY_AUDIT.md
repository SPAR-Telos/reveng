# Output Directory Audit

## Recommended reader-facing location

- `outputs/reader_facing/experiment1_2_gpt_oss_matched46_v1/`: scaled matched-46 figures and tables generated for sharing.

## Keep as source experiment outputs

- `outputs/experiment1_activation_monitor/gpt_oss_local_sentence_matched46_v1/`: source Experiment 1 matched-46 action, entropy, and activation outputs.
- `outputs/experiment2_behavioral_beliefs/gpt_oss_local_sentence_matched46_v1/`: source Experiment 2 matched-46 belief outputs and predictive models.
- `outputs/experiment1_activation_monitor/gpt_oss_local_environment_step_all1276_v1/` and `outputs/experiment2_behavioral_beliefs/gpt_oss_local_environment_step_all1276_v1/`: environment-step robustness outputs, not the same population as matched-46 sentence figures.
- Some belief-related follow-up CSVs also exist in the Experiment 1 matched-46 folder because earlier combined analysis scripts joined action, entropy, and belief rows there. Treat them as source artifacts, not reader-facing figures.

## Potentially confusing legacy or diagnostic outputs

- `outputs/experiment1_activation_monitor/weisheng_8_state_*` and `outputs/experiment2_behavioral_beliefs/weisheng_8_state_*`: earlier 8-state pilot outputs; keep for provenance but do not mix with scaled matched-46 figures.
- `outputs/experiment1_activation_monitor/weisheng_sentence_logprob_gate*` and `outputs/experiment2_behavioral_beliefs/weisheng_sentence_logprob_gate*`: feasibility-gate outputs; move to an archive folder if sharing the repo externally.
- `outputs/reader_facing/experiment1_2_weisheng_8_state_logprob_v1/`: earlier reader-facing 8-state output; superseded for scaled matched-46 reporting.
- Duplicated progress figures inside Experiment 1 and Experiment 2 source folders are acceptable as diagnostics, but the new `outputs/reader_facing/experiment1_2_gpt_oss_matched46_v1/` folder should be the shared location for scaled-run figures.

## Do not delete automatically

No files were deleted. The safest cleanup is to move legacy Weisheng pilot and gate folders under an explicit archive directory after confirming no open notebook references them.
