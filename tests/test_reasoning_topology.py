import numpy as np

from reveng.experiments.reasoning_topology import curve_metrics, paired_inference


def test_looping_curve_has_more_recurrence_and_tortuosity_than_line() -> None:
    progress = np.linspace(0.0, 1.0, 40)
    line = np.column_stack([progress, np.zeros_like(progress)])
    angle = np.linspace(0.0, 4.0 * np.pi, 40)
    loop = np.column_stack([np.cos(angle), np.sin(angle)])
    line_metrics = curve_metrics(line)
    loop_metrics = curve_metrics(loop)
    assert loop_metrics["nonlocal_recurrence"] > line_metrics["nonlocal_recurrence"]
    assert loop_metrics["path_tortuosity"] > line_metrics["path_tortuosity"]


def test_late_consolidation_reduces_dispersion_ratio() -> None:
    rng = np.random.default_rng(7)
    diffuse = rng.normal(size=(40, 6))
    consolidating = diffuse.copy()
    consolidating[-10:] = consolidating[-10:].mean(axis=0) + 0.01 * rng.normal(
        size=(10, 6)
    )
    assert (
        curve_metrics(consolidating)["late_to_early_dispersion_ratio"]
        < curve_metrics(diffuse)["late_to_early_dispersion_ratio"]
    )


def test_paired_inference_detects_consistent_positive_difference() -> None:
    result = paired_inference(
        [0.2, 0.3, 0.4, 0.5], bootstrap_samples=2_000, permutation_samples=5_000
    )
    assert result["mean_difference"] > 0
    assert result["ci_low"] > 0
