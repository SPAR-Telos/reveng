from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_information_use_test.py"
spec = importlib.util.spec_from_file_location("build_information_use_test", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_linear_cka_binary_label_detects_dependence() -> None:
    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * 80)
    dependent = np.column_stack(
        [
            labels + rng.normal(0, 0.05, size=len(labels)),
            rng.normal(size=len(labels)),
        ]
    )
    independent = rng.normal(size=(len(labels), 2))

    dependent_score = module.linear_cka_with_binary_label(dependent, labels)
    independent_score = module.linear_cka_with_binary_label(independent, labels)

    assert dependent_score > independent_score
    assert dependent_score > 10 * independent_score


def test_log_loss_improvement_positive_when_model_log_loss_is_lower() -> None:
    rows = [
        {
            "target": "action_change",
            "model_type": "baseline_progress_action_confidence",
            "roc_auc": 0.55,
            "log_loss": 0.40,
        },
        {
            "target": "action_change",
            "model_type": "baseline_plus_beliefs",
            "roc_auc": 0.60,
            "log_loss": 0.35,
        },
    ]

    enriched = module.add_improvements(rows)
    belief_row = [row for row in enriched if row["model_type"] == "baseline_plus_beliefs"][0]

    assert belief_row["delta_auroc_vs_action_confidence_baseline"] == pytest.approx(0.05)
    assert belief_row["log_loss_improvement_vs_action_confidence_baseline"] == pytest.approx(0.05)
