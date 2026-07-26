# Figure Captions

## final_action_probability_around_jump.png

Probability assigned to the action recommended after the complete reasoning trace, aligned to the sentence producing its largest increase. Position 0 is that sentence. Lines show medians and bands show interquartile ranges across states. Probabilities come from temperature-0.7 candidate-token logprobs normalized over UP, DOWN, LEFT, and RIGHT.

## probability_jump_vs_stable_commitment.png

Comparison of two retrospective action boundaries. The horizontal axis is the sentence producing the largest increase in probability assigned to the full-trace action. The vertical axis is the first sentence after which the highest-probability action remains equal to the full-trace action. Both positions are measured as the fraction of reasoning characters revealed. Points on the diagonal indicate agreement.

## activation_similarity_around_distribution_change.png

Cosine similarity between each layer-15 sentence-mean activation and the mean activation of all preceding reasoning sentences, aligned to the sentence with the largest adjacent Jensen-Shannon divergence in the probability distribution over UP, DOWN, LEFT, and RIGHT. Position 0 is the distribution-change sentence. The line is the median and the band is the interquartile range across 46 states. This association does not establish that the activation change causes the action-distribution change.
