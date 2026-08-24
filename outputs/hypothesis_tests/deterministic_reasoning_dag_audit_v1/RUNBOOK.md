# Runbook

1. Start the browser UI with `uv run python scripts/serve_reasoning_dag_review.py --reviewer a`, then open `http://127.0.0.1:8765`.
2. For an independent second pass, stop the server and rerun it with `--reviewer b`.
3. The initial UI exposes 15 destination nodes per trace (30 total) as a calibration subset and saves directly into `edge_review.csv`.
4. Compare edge detection, source selection, and label agreement before adjudication.
5. Do not inspect behavioral outcomes while annotating edges.
6. Populate `adjudicated_edges_json` only after independent review is complete.
