# Sentence-Level Action Commitment and Activation Results

The recommended action and its uncertainty come from one GPT-OSS-20B request per sentence prefix at temperature 0.7. The action is the argmax over UP, DOWN, LEFT, and RIGHT token probabilities; uncertainty is Shannon entropy over the same normalized distribution.

- Environment states: 8
- Prefix positions: 662
- Activation-backed sentence positions: 654
- Prefix zero activation: unavailable by design
- Primary activation analysis: layer 8, mean over all stride-2 sampled positions in each sentence
- Sensitivity analyses: trailing up to three sampled positions and all other available even layers

| Event | Count |
|---|---:|
| action change | 61 |
| commitment onset | 5 |
| optimality loss | 9 |
| optimality recovery | 15 |
| sustained optimality loss | 3 |
