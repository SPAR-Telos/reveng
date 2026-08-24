# Reasoning experiment and artifact guide

This is the shortest route into the topology, regime, semantic-prediction, graph, and DAG-audit work. The file-level, checksummed companion is [`reasoning_experiment_lineage.csv`](reasoning_experiment_lineage.csv). General repository and remote-storage rules remain in [`REPOSITORY_AND_ARTIFACT_GUIDE.md`](REPOSITORY_AND_ARTIFACT_GUIDE.md) and [`REMOTE_ARTIFACTS.md`](REMOTE_ARTIFACTS.md).

## Pipeline order and usage

| Order | Experiment | Run command | Consumes | Important outputs |
|---:|---|---|---|---|
| 1 | Continuous activation topology | `uv run python scripts/run_reasoning_topology.py` | Matched-46 cohort; GPT-OSS activation index/shards | `reasoning_topology_v1/trace_metrics.csv`, `paired_summary.csv`, `run_report.md`, `run_manifest.json` |
| 2 | Deterministic behavioral regimes | `uv run python scripts/run_reasoning_regime_coarse_graining.py` | Action-distribution position metrics; topology trace metrics | `reasoning_regime_coarse_graining_v1/regime_positions.csv`, `transition_rows.csv`, `graph_edges.csv`, `predictive_rows.csv`, report/manifest |
| 3 | Persistence-aware semantic prediction | `uv run python scripts/run_semantic_regime_transition_prediction.py` | Regime transitions; canonical original and replicate semantic annotations | `semantic_regime_transition_prediction_v1/prediction_rows.csv` (row-level raw model output), summaries/tests, report/manifest |
| 4 | Multi-horizon closure and process graphs | `uv run python scripts/run_reasoning_dynamics_graph.py` | Regime positions; BEAST points; both semantic runs | `reasoning_dynamics_graph_v1/closure_predictions.csv`, semantic paths/edges, BEAST operator events/graphs, report/manifest |
| 5 | Graph figures | `uv run python scripts/plot_reasoning_dynamics_graphs.py` | Experiment 4 graph tables | `reasoning_dynamics_graph_v1/figs/*.png`, `FIGURE_GUIDE.md` |
| 6 | Typed DAG audit preparation | `uv run python scripts/prepare_deterministic_reasoning_dag_audit.py` | Regime positions; BEAST points; canonical semantic annotations | `deterministic_reasoning_dag_audit_v1/nodes.csv`, `edge_review.csv`, guide/runbook |
| 7 | Human DAG calibration | `uv run python scripts/serve_reasoning_dag_review.py --reviewer a` | Audit package | Mutates only the selected reviewer column in `edge_review.csv`; Reviewer B uses `--reviewer b` |
| 8 | Rebuild this catalog | `uv run python scripts/build_reasoning_experiment_lineage.py` | The five output directories above | `docs/reasoning_experiment_lineage.csv` |

All output paths above are below `outputs/hypothesis_tests/`.

## What counts as raw

These experiments reuse existing raw target-model trajectories, action distributions, activations, and raw semantic-judge responses. Their locations and remote datasets are documented in the repository guide. Within this newer pipeline:

- `prediction_rows.csv` is the closest-to-raw out-of-fold output of the semantic transition models.
- `closure_predictions.csv` is the closest-to-raw out-of-fold output of the multi-horizon models.
- `regime_positions.csv`, `transition_rows.csv`, and `beast_operator_events.csv` are row-level deterministic/derived analysis records.
- `edge_review.csv` becomes raw human annotation once review starts; blank means unreviewed and `[]` means reviewed with no predecessor.
- Summaries, reports, and figures are downstream views, never substitutes for these row-level records.

Each completed computational experiment has a `run_manifest.json` recording source paths and hashes, model/annotation inputs, seeds, and analysis settings. The semantic prediction and graph experiments consume both canonical judge runs; they never silently substitute a different judge or target model.

## Dependency graph

```text
stored target trajectories + activations + action distributions
                    |
                    +--> continuous topology
                    |          |
                    +--> behavioral regimes
                               |
canonical semantic runs -------+--> semantic transition prediction
                               |
BEAST points -------------------+--> multi-horizon closure/process graphs
                               |
                               +--> typed discourse-DAG audit
```

BEAST points decorate the proposed discourse DAG; they do not define its topology. The design decision and literature justification are in [`deterministic_reasoning_graph_plan.md`](deterministic_reasoning_graph_plan.md).

## Fresh-machine restoration

1. Clone `origin`, checkout `feat/counterfactual-patching`, and run `uv sync --locked`.
2. Restore private Hugging Face artifacts using the commands in `REMOTE_ARTIFACTS.md`.
3. Verify hashes against `reasoning_experiment_lineage.csv` and each run manifest.
4. Run the focused tests before continuing an experiment.

Never copy `.venv`, model/download caches, `.env`, `.hf_home`, or API credentials. Restore credentials separately.
