# Weisheng Sentence Activation Mapping

Two vectors are saved for each covered sentence and layer: the mean of all available stride-2 sampled COT positions, and the mean of the last up to three available sampled positions. Token offsets determine sentence overlap.
Sentences with no sampled token are reported but receive no tensor.

- Trajectory: `together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_0`
- Environment steps: [0, 1, 2, 3, 4, 5, 6, 7]
- Layers: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
- Reasoning sentences in these steps: 654
- Sentences with a vector: 654
- Sentence-layer-representation rows: 15696
- Sentences without a vector: 0
- Coverage: 100.0%
- Saved tensor rows: 15696

These are sentence means, not exact sentence-final-token activations. The source sampled every second COT token, so an exact final-token vector is unavailable whenever that token was not sampled.
