# Annotation Provenance

The 44 rows in `human_pilot_annotation.csv` were labeled on 2026-07-29 by the
Codex AI assistant at the user's request. Despite the historical filename,
these are AI-assisted pilot annotations, not human annotations.

The assistant used only `context_before`, `target_sentence`, and
`SEMANTIC_ANNOTATION_GUIDE.md`. It did not inspect `annotation_key.csv` or
`pilot_duplicate_key.csv` until all labels had been written. The hidden keys
were then used by `analyze_semantic_reasoning_classification.py` to validate
duplicate consistency and compare detected change points with matched
sentences.

These labels should be treated as a first-pass judge annotation. Before using
them as training labels or reported evidence, obtain an independent annotation
or adjudicate disagreements on at least a stratified subset.

The current pilot identifiers expose duplicated rows through a historical
`sem_dup_` prefix. Consequently, the observed duplicate agreement is only a
consistency check and not a blinded reliability estimate. Future pilot
generation has been changed to use opaque identifiers.
