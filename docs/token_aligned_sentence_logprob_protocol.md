# Token-Aligned Sentence Logprob Protocol

## Scope

The reported run evaluates eight consecutive DoorKey environment states from one GPT-OSS-20B trajectory. It compares 662 sentence-prefix positions with 250 packed-prefix positions. The unfinished 458-prefix, ten-sample action run is retained only as a sensitivity analysis.

## Sentence token boundaries

`sentence_token_boundaries_gpt_oss_20b.csv` maps the 153,622 reasoning sentences in 1,276 traces from 95 trajectory files to GPT-OSS-20B token positions. Positions are zero-based and end-exclusive. Analysis positions are relative to the reasoning channel; output positions use each stored output token's index.

The mapper tokenizes the exact untrimmed reasoning-channel text with tokenizer revision `6cee5e81ee83917806bbde320786a8fb61efebee`. Every local token ID must equal the corresponding stored trajectory token ID. Sentence tokens are assigned by overlap between tokenizer character offsets and canonical sentence character spans. Final-action records are excluded. Traces with text mismatches, tokenization disagreement, overlapping spans, or duplicate token assignment fail validation.

## Action uncertainty

At each prefix, GPT-OSS-20B receives the fixed grid state, key-possession status, and the observed reasoning prefix. It is asked to return the next action. One TogetherAI request at temperature 0.7 returns the top 50 token logprobs at the generated action-token position.

The probabilities for `UP`, `DOWN`, `LEFT`, and `RIGHT` are renormalized over those four candidates. The recommended action is the probability argmax. Action uncertainty is Shannon entropy in bits over the same distribution. The output retains candidate probability mass, probability mass represented by the returned top-logprob set, missing candidates, and raw top logprobs.

## Belief uncertainty

At every prefix, nine categorical probes ask about four adjacent walls, key possession, door status, and three consequences of the recommended action. A single temperature-0.7 query supplies the `A`, `B`, and `C` token probabilities corresponding to `yes`, `no`, and `unknown`. The categorical answer is the probability argmax, and uncertainty is Shannon entropy in bits over the same normalized distribution.

Four coordinate probes remain generated answers at temperature 0 and do not receive an entropy estimate.

## Activation mapping

Weisheng's stride-2 GPT-OSS-20B activations cover all 654 non-empty sentence positions in the eight states. Prefix zero has no reasoning activation. For every sentence and even layer from 0 through 22, two representations are saved:

- the mean over all sampled positions overlapping the sentence;
- the mean over the last up to three available sampled positions.

Layer 8 and the all-position sentence mean are fixed as the primary activation analysis. The trailing-position representation and other layers are sensitivity analyses.

## Feasibility gate

The full run starts only after a 20-prefix gate. At least 99 percent of action and categorical-belief queries must expose every candidate label. Top 20 reached only 98 percent because low-probability belief labels were truncated. Top 50 reached 100 percent without changing the model, prompt, or temperature and is therefore used for the reported run.
