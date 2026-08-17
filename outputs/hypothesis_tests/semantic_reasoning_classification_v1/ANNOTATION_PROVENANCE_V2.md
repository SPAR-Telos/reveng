# Semantic Annotation Provenance, Calibrated Full Pass

## Source and blinding

The full annotation source is `judge_items.jsonl`: 320 sentences comprising
160 detected-change-point / matched-comparison pairs. The model judge received
only `context_before` and `target_sentence`. Pair role, trajectory identity,
action probabilities, event labels, activation measurements, behavioral
beliefs, and the hidden matching key were not included in the prompt.

## Judge

- Model: `openai/gpt-oss-20b`
- Model revision: `6cee5e81ee83917806bbde320786a8fb61efebee`
- Prompt: `v2_calibrated`
- Prompt SHA-256:
  `6b871a8822964d14612fdb232d3acc410ad9d321e616a1a39a15b15337890b52`
- Decoding: deterministic (`temperature=0`, seed 42)
- Reasoning effort: `medium`
- Runner: `scripts/run_local_semantic_reasoning_judge.py`
- Full output: `gpt_oss_local_annotations_v2.csv`

The output preserves the rendered source text, parsed cue fields, fine label,
confidence, rationale, raw model response, model revision, prompt hash, and any
parse error. This makes every sentence-level judgment inspectable.

All 18 low-confidence outputs were subsequently reviewed sentence by sentence.
`sentence_label_table.csv` contains the compact final semantic annotations for
the 320 selected sentences. It uses readable compound identifiers such as
`keepdoor_33__step_002__sentence_038`; the component columns identify the
environment, environment step, and one-based sentence number. The opaque
`annotation_id` remains only in the full experimental ledger for compatibility
with the original annotation and adjudication files.

`final_analysis/all_reasoning_sentence_inventory.csv` lists all 7,038 reasoning
sentences in the matched 46-state cohort. It marks the 320 classified sentences
and leaves semantic fields blank for the other 6,718; those rows have not been
semantically classified. The original model fields and the
CPD/action/activation/belief joins remain in
`final_analysis/sentence_label_cpd_analysis_table.csv`; the analysis uses the
explicit replacements in `low_confidence_adjudication_overrides.csv`. This
review corrected direct state readouts, route checks, derived constraints, and
meta-level transitions that the judge had mostly placed in `unclear`.

## Reliability and inferential scope

The nine-way taxonomy is difficult even for a calibrated judge, especially at
the boundaries between direct state readout, route verification, and new
inference. On 40 separately adjudicated pilot items, the calibrated judge
achieved 52.5% exact agreement on the nine-way labels and 77.5% agreement on
the four-way grouping used for primary analysis. The corresponding Cohen
kappas were 0.417 and 0.534. Four repeated pilot items received identical fine
and broad labels.

The adjudication was performed by a Codex AI assistant, not a human expert,
and the prompt was refined after inspecting pilot behavior. These figures are
therefore calibration-set agreement estimates, not an independent human
reliability study. The nine-way results are sensitivity analyses. Primary
inference uses:

- `state_readout`: `state_reconstruction`
- `route_deliberation`: `route_planning`, `verification`, `correction`,
  `new_inference`
- `summary_or_restatement`: `consolidation`, `restatement`
- `other`: `procedural_continuation`, `unclear`

The semantic results are contemporaneous associations at change points found
by a full-trace, retrospective BEAST analysis. They are neither causal tests
nor real-time forecasts.
