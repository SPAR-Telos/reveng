# Untried Follow-up Ideas

## Highest-value next tests

| Test | Question | Evidence it would add |
|---|---|---|
| Held-out attention confirmation | Does final-action attention near commitment generalize beyond the matched 46-state pilot? | A confirmatory association with prespecified layers, windows, and heads |
| Head ablation or activation patching | Does disrupting the candidate commitment signal change the final action? | Causal evidence about information use |
| Post-commitment truncation | Does ending reasoning at commitment preserve the action and task success? | A direct test of whether later reasoning is action-epiphenomenal |
| Sentence-level action-distribution change points | Does one reasoning sentence sharply change the probability distribution over the four actions? | A distributional commitment boundary connecting action entropy, optimality, activations, and attention |
| Grid-information ablation | Does removing or changing a necessary wall, key, or door fact alter the recommendation? | A direct test of whether available state information affects action selection |
| Real-agent transition probes | Do incorrect predictions of the next agent position, key status, or door status predict later action errors? | A cleaner test of transition beliefs than hypothetical-agent prompts |

## Useful extensions after those tests

| Test | Purpose |
|---|---|
| Attention at intermediate action readouts | Measure information routing when each recommendation is elicited, rather than only when the final action is generated |
| Semantic labels from a blinded language-model judge | Test whether commitment-attended sentences are plan conclusions, checks, corrections, summaries, or repetitions |
| Activation-similarity stress test | Test whether cosine similarity predicts the next action event beyond reasoning progress and action confidence |
| Counterfactual sentence replacement | Change a belief-bearing sentence and measure downstream beliefs and actions |
| Position-general representational probes | Track the same current-state and transition beliefs across reasoning positions once compatible probes are available |
| Qwen or Gemma cross-model run | Combine behavioral readouts with a model-supported Activation Oracle or natural-language activation decoder |
| Exact MI-peaks-style analysis | Test token-level dependence on an answer representation; the existing CKA analysis is not this method |
| Cross-task post-commitment monitor | Determine whether the signal generalizes beyond DoorKey before calling it a general reasoning or persona direction |
