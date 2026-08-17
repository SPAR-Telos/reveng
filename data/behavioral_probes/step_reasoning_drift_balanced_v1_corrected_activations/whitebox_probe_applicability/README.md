# Corrected Activation White-Box Probe Applicability

No released white-box probe was run on the corrected per-step tensors because no application passed validation.

| Checkpoint family | Intended target | Expected input | Saved input | Result |
|---|---|---|---|---|
| Cognitive-map MLP, pre and post | Grid-tile state beliefs | Three token vectors plus query coordinates, 8,642 features | One 2,880-feature last-token vector and one 2,880-feature sentence mean | Skipped |
| Plan decoder, pre and post | Ten-action sequence | Three token vectors with shape `[3, 2880]` | One 2,880-feature last-token vector and one 2,880-feature sentence mean | Skipped |

The corrected activation spans are non-overlapping and valid for geometry analysis. This does not make the released probes valid at intermediate reasoning positions: those checkpoints were trained only at fixed pre-reasoning or post-reasoning positions. Predictions are deliberately blank in `applicability_rows.csv`.

To obtain per-position white-box results, collect three-token boundary windows and train or validate probes specifically at intermediate reasoning positions. Beliefs without a trained probe should remain missing.
