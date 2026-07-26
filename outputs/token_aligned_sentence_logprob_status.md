# Token-Aligned Sentence Logprob Run Status

## Completed

- GPT-OSS token boundaries: 153,622 reasoning sentences across 1,276 traces and 95 files.
- Exact tokenizer-ID validation: passed for every trace.
- Sentence activations: 654 of 654 non-empty sentence positions across eight states.
- Activation representations: sentence mean and trailing up-to-three sampled-position mean at even layers 0 through 22.
- Logprob feasibility gate: passed at 100 percent candidate coverage with temperature 0.7 and top 50 logprobs.
- Sentence action run: 662 of 662 prefix positions, eight states, no invalid recommendations, 100 percent four-action candidate coverage.
- Sentence activation join and geometry: completed for 654 non-empty sentence positions with layer 8 sentence means as primary and 15,042 other layer or representation rows as sensitivity analyses.
- Incomplete MC-10 run: retained unchanged at 458 prefixes as a sensitivity analysis.

## Checkpointed

- Sentence categorical and coordinate belief run: 497 of 8,606 queries completed with no recorded query failures.
- Packed action and belief controls: not started.
- Sentence-versus-packed comparison: waiting for behavioral runs to finish.

The managed execution service blocked the resume attempt because its usage limit was reached. The behavioral checkpoint itself is valid and resumable.

## Resume

```bash
bash scripts/run_token_aligned_sentence_logprob_experiments.sh full
```

The command first revalidates the passed gate, restores all 662 sentence action rows, resumes missing sentence belief rows, then runs the packed control and local postprocessing.
