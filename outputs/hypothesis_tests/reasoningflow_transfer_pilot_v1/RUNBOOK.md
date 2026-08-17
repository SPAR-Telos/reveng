# ReasoningFlow transfer-pilot runbook

Run from the repository root.

```bash
.venv/bin/python scripts/run_reasoningflow_transfer_pilot.py --stage label --resume
.venv/bin/python scripts/run_reasoningflow_transfer_pilot.py --stage analyze
```

After labelling, complete the review fields in `outputs/hypothesis_tests/reasoningflow_transfer_pilot_v1/manual_review.csv` and rerun the analysis stage. Use `yes` or `no` for `boundary_reasonable` and `labels_reasonable`. Record any missing DoorKey-specific role in `missing_doorkey_function`.
