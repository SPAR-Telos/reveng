# Sentence Experiment Pipeline Status

Status: stopped by the user; resumable.

## Progress

- Canonical prefix positions: 662
- Completed action-prefix checkpoints: 458
- Remaining action-prefix positions: 204
- Completed environment states: steps 0, 1, and 2
- Partially completed state: step 3, with 62 of 112 positions
- Experiment 2 behavioral belief queries: not started
- State-belief uncertainty queries: not started

## Bottleneck

Action uncertainty uses one greedy action query and 10 sampled action queries at temperature 0.7 for every prefix. A complete run therefore requires at most 7,282 TogetherAI completions, of which 6,620 are entropy samples. The sampling accounts for 91 percent of planned action requests.

The runner processes one prefix at a time, although samples within a prefix are concurrent. Long late-trace prefixes increase input length and request latency. The first 396 completed positions also recorded 18 retries and 90 seconds of retry backoff.

## Resume

`prefix_query_checkpoints.jsonl` and `example_checkpoints.jsonl` contain the completed work. Running `scripts/run_weisheng_8_state_sentence_experiments.sh` resumes rather than repeating successful prefixes. The output-directory lock prevents concurrent runners.

The future uncertainty stage is configured to request state-belief logprobs at temperature 0.7, matching the action-sampling temperature. It must not be interpreted together with the older temperature-0 state-belief entropy without an explicit temperature sensitivity comparison.
