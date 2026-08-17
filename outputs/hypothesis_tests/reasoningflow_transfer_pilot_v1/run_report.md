# ReasoningFlow transfer pilot

## Question

Can a literature-derived, sub-sentence ReasoningFlow schema describe DoorKey reasoning clearly enough to serve as a shared segmentation layer?

## Method

The pilot contains 120 previously unlabelled DoorKey sentences: half randomly sampled and half enriched for multi-clause surface structure. GPT-OSS-20B receives only preceding reasoning and the target sentence. It assigns the eight ReasoningFlow response-node roles. No edge labels, change points, action outcomes, clustering labels, or action-commitment labels are used.

## Current result

- Successfully labelled sentences: 96/120
- Produced nodes: 96
- Sentences split into multiple nodes: 0
- Reviewed boundary acceptability: pending manual review
- Reviewed label acceptability: pending manual review
- Reviewed examples reporting a missing DoorKey function: 0

## Interpretation

The model output is not a validated annotation corpus until the 40-sentence review is completed. If the boundaries and general labels are consistently understandable, ReasoningFlow can be retained as a general discourse layer. Observation, route evaluation and action selection should remain a separate DoorKey-specific layer when reviewers report that the general schema loses those distinctions. Explicit commitment remains a separate annotation joined by the shared sentence and span identifiers.

## Scope relative to other work

This is the literature-supervised branch of reasoning-step classification, not a separate fifth classification project. Clustering should remain label-blind until comparison. The pilot does not operationalize explicit action commitment.
