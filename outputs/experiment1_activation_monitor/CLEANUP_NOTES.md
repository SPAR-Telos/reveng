# Experiment 1 Output Cleanup

Deleted on 2026-06-26:

| Folder | Reason |
|---|---|
| `pilot24_existing_activation_check` | Old compatibility check against non-canonical activation rows; manifest recorded `activation_compatibility_status=incompatible`, so it is not useful for the current Experiment 1 deliverable. |
| `weisheng_chunk_averaged_existing_prefix_actions` | Superseded exploratory join using the older 24-state prefix-action source. The current deliverable is `weisheng_8_state_canonical_experiment1`, which aligns the 8 Weisheng activation-backed states to the canonical prefix-action run. |

Kept:

| Folder | Reason |
|---|---|
| `weisheng_chunk_averaged_activations` | Source chunk-averaged Weisheng activation artifact used by the canonical run. |
| `weisheng_8_state_candidates` | Candidate metadata and programmatic labels for the 8 canonical states. |
| `weisheng_8_state_canonical_prefix_actions` | Prefix-conditioned action and entropy rows used by Experiment 1 and Experiment 2. |
| `weisheng_8_state_canonical_experiment1` | Current canonical Experiment 1 results and figures. |
