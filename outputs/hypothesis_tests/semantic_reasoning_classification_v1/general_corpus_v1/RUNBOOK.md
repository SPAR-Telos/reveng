# General semantic-labelling runbook

Run from the repository root. Both judges are local and require CUDA; neither silently falls back to CPU.

## 1. Smaller few-shot candidate

```bash
.venv/bin/python scripts/run_general_semantic_labeler.py --resume
```

## 2. GPT-OSS-20B continuity judge

```bash
.venv/bin/python scripts/run_general_semantic_labeler.py \
  --model-path /root/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee \
  --model-name openai/gpt-oss-20b \
  --minimum-free-gib 16 \
  --output outputs/hypothesis_tests/semantic_reasoning_classification_v1/general_corpus_v1/annotations_gpt_oss_20b.csv \
  --resume
```

## 3. Compare and decide

```bash
.venv/bin/python scripts/analyze_general_semantic_labeling_pilot.py
```

Do not start the full-corpus pass unless `decision_gate.json` passes and the disagreement review is acceptable.

The cached GPT-OSS-20B snapshot uses native MXFP4 weights and was verified to fit on the 23-GiB L4 with the 16-GiB pre-load guard.
