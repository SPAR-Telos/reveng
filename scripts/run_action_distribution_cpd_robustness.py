#!/usr/bin/env python3
"""Run the finite robustness study for offline BEAST action change points."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from reveng.experiments.beast_cpd_robustness import (
    default_robustness_analysis_setups,
    event_proximity_rows,
    evaluate_synthetic_trace,
    fit_robustness_analysis_setup,
    generate_synthetic_probability_trace,
    match_detected_locations,
    summarize_event_proximity,
)


BLUE = "#1769AA"
LIGHT_BLUE = "#8CC8E8"
DARK = "#17324D"
ORANGE = "#D97706"
GREEN = "#34805C"
GRID = "#DCEAF3"
EVENT_TYPES = (
    "action_change",
    "optimal_to_suboptimal",
    "suboptimal_to_optimal",
    "commitment_onset",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positions",
        type=Path,
        default=Path(
            "outputs/experiment1_activation_monitor/"
            "gpt_oss_local_sentence_matched46_v1/position_rows.csv"
        ),
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=Path(
            "outputs/experiment1_activation_monitor/"
            "gpt_oss_local_sentence_matched46_v1/event_rows.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/hypothesis_tests/action_distribution_cpd_robustness_v1"
        ),
    )
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--synthetic-samples", type=int, default=2000)
    parser.add_argument("--synthetic-repeats", type=int, default=20)
    parser.add_argument("--original-synthetic-repeats", type=int, default=100)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--event-window", type=int, default=3)
    parser.add_argument("--location-tolerance", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_matplotlib() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update(
        {
            "font.family": "Arial" if "Arial" in available else "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_event_positions(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    frame["position_index"] = frame["event_reasoning_step_idx"].astype(int)
    loss = frame["event_type"].isin(
        ["sustained_optimal_to_suboptimal", "transient_optimal_to_suboptimal"]
    )
    frame.loc[loss, "event_type"] = "optimal_to_suboptimal"
    return frame[frame["event_type"].isin(EVENT_TYPES)].copy()


def run_synthetic_benchmark(
    analysis_setups: list[Any],
    empirical_lengths: np.ndarray,
    *,
    samples: int,
    repeats: int,
    original_repeats: int,
    tolerance: int,
    seed: int,
) -> pd.DataFrame:
    scenarios = (
        "constant",
        "stable_noise",
        "smooth_drift",
        "one_level_shift",
        "one_slope_change",
        "two_level_shifts",
    )
    length_quantiles = np.unique(
        np.maximum(
            21,
            np.rint(np.quantile(empirical_lengths, [0.1, 0.25, 0.5, 0.75, 0.9])),
        ).astype(int)
    )
    rows: list[dict[str, Any]] = []
    for analysis_setup_index, analysis_setup in enumerate(analysis_setups):
        analysis_setup_repeats = (
            original_repeats
            if analysis_setup.analysis_setup_id == "original"
            else repeats
        )
        for scenario_index, scenario in enumerate(scenarios):
            for repeat in range(analysis_setup_repeats):
                length = int(length_quantiles[repeat % len(length_quantiles)])
                trace_seed = (
                    seed
                    + 100_000 * analysis_setup_index
                    + 1_000 * scenario_index
                    + repeat
                )
                distributions, expected = generate_synthetic_probability_trace(
                    scenario,
                    length=length,
                    seed=trace_seed,
                )
                result = evaluate_synthetic_trace(
                    distributions,
                    expected,
                    analysis_setup,
                    samples=samples,
                    tolerance=tolerance,
                    perturbation_seed=trace_seed + 10_000_000,
                )
                rows.append(
                    {
                        **analysis_setup.to_record(),
                        "scenario": scenario,
                        "repeat": repeat,
                        "trace_length": length,
                        **{
                            key: (
                                json.dumps(value)
                                if key == "detected_positions"
                                else value
                            )
                            for key, value in result.items()
                        },
                    }
                )
    return pd.DataFrame(rows)


def summarize_synthetic_benchmark(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for keys, group in rows.groupby(
        ["analysis_setup_id", "analysis_setup_family", "scenario"],
        sort=False,
    ):
        analysis_setup_id, analysis_setup_family, scenario = keys
        localization_errors = group["median_localization_error"].dropna()
        output.append(
            {
                "analysis_setup_id": analysis_setup_id,
                "analysis_setup_family": analysis_setup_family,
                "scenario": scenario,
                "n_traces": len(group),
                "trace_false_positive_rate": (
                    group["trace_false_positive"].mean()
                    if group["n_expected_change_points"].eq(0).all()
                    else np.nan
                ),
                "trace_detection_rate": group["trace_detected"].mean(),
                "change_point_recall": group["change_point_recall"].mean(),
                "median_localization_error": (
                    localization_errors.median()
                    if not localization_errors.empty
                    else np.nan
                ),
                "mean_detected_change_points": group[
                    "n_detected_change_points"
                ].mean(),
            }
        )
    return pd.DataFrame(output)


def build_original_point_stability(
    original_points: pd.DataFrame,
    matches: pd.DataFrame,
    analysis_setup_count: int,
) -> pd.DataFrame:
    output = original_points[
        [
            "example_id",
            "trajectory_id",
            "matched_role",
            "position_index",
            "reasoning_progress",
            "posterior_change_probability_at_position",
        ]
    ].copy()
    recovery_counts = (
        matches.groupby(["example_id", "reference_position"])[
            "analysis_setup_id"
        ]
        .nunique()
        .rename("n_analysis_setups_recovered")
    )
    output = output.merge(
        recovery_counts,
        how="left",
        left_on=["example_id", "position_index"],
        right_index=True,
    )
    output["n_analysis_setups_recovered"] = (
        output["n_analysis_setups_recovered"].fillna(0).astype(int)
    )
    output["recovery_fraction"] = (
        output["n_analysis_setups_recovered"] / analysis_setup_count
    )
    output["found_by_at_least_12_of_16_setups"] = (
        output["n_analysis_setups_recovered"] >= 12
    )
    return output


def stopping_rule_table(
    analysis_setup_summary: pd.DataFrame,
    location_summary: pd.DataFrame,
    point_stability: pd.DataFrame,
    event_summary: pd.DataFrame,
    synthetic_summary: pd.DataFrame,
) -> pd.DataFrame:
    original_synthetic = synthetic_summary[
        synthetic_summary["analysis_setup_id"] == "original"
    ].set_index("scenario")
    null_fpr = float(
        original_synthetic.loc[
            ["stable_noise", "smooth_drift"],
            "trace_false_positive_rate",
        ].max()
    )
    change_rows = original_synthetic.loc[
        ["one_level_shift", "two_level_shifts"]
    ]
    minimum_abrupt_recall = float(change_rows["change_point_recall"].min())
    localization_complete = bool(
        change_rows["median_localization_error"].notna().all()
    )
    maximum_abrupt_localization = (
        float(change_rows["median_localization_error"].max())
        if localization_complete
        else np.nan
    )
    slope_recall = float(
        original_synthetic.loc["one_slope_change", "change_point_recall"]
    )

    seed_rows = analysis_setup_summary[
        analysis_setup_summary["analysis_setup_family"].isin(
            ["original", "random_seed"]
        )
    ]
    original_count = int(
        analysis_setup_summary.set_index("analysis_setup_id").loc[
            "original", "n_detected_change_points"
        ]
    )
    seed_count_deviation = float(
        (seed_rows["n_detected_change_points"] - original_count).abs().max()
        / original_count
    )
    broadly_reproduced_fraction = float(
        point_stability["found_by_at_least_12_of_16_setups"].mean()
    )

    action_effects = event_summary[
        event_summary["event_type"] == "action_change"
    ]
    all_action_effects_positive = bool(
        (action_effects["observed_minus_expected_fraction"] > 0).all()
    )
    original_action = action_effects[
        action_effects["analysis_setup_id"] == "original"
    ].iloc[0]
    original_action_ci_positive = (
        float(original_action["trajectory_bootstrap_ci_low"]) > 0
    )
    distance_locations = location_summary[
        location_summary["analysis_setup_family"] == "distance_calculation"
    ]
    distance_recovery = float(distance_locations["reference_recovery"].min())

    return pd.DataFrame(
        [
            {
                "check_id": "false_alarm_rate_without_abrupt_shift",
                "required_result": (
                    "false alarms in no more than 5% of either kind of "
                    "no-abrupt-shift trace"
                ),
                "measured_result": null_fpr,
                "passed": null_fpr <= 0.05,
                "needed_for_abrupt_shift_claim": True,
                "needed_for_gradual_change_claim": True,
            },
            {
                "check_id": "known_abrupt_shifts_correctly_found",
                "required_result": (
                    "correctly find at least 80% of the known abrupt shifts"
                ),
                "measured_result": minimum_abrupt_recall,
                "passed": minimum_abrupt_recall >= 0.80,
                "needed_for_abrupt_shift_claim": True,
                "needed_for_gradual_change_claim": True,
            },
            {
                "check_id": "known_abrupt_shifts_placed_accurately",
                "required_result": (
                    "typical location error no greater than 3 sentences"
                ),
                "measured_result": maximum_abrupt_localization,
                "passed": (
                    localization_complete and maximum_abrupt_localization <= 3
                ),
                "needed_for_abrupt_shift_claim": True,
                "needed_for_gradual_change_claim": True,
            },
            {
                "check_id": "gradual_slope_changes_correctly_found",
                "required_result": (
                    "correctly locate at least 80% of gradual slope changes"
                ),
                "measured_result": slope_recall,
                "passed": slope_recall >= 0.80,
                "needed_for_abrupt_shift_claim": False,
                "needed_for_gradual_change_claim": True,
            },
            {
                "check_id": "similar_counts_with_different_random_seeds",
                "required_result": (
                    "changing the random seed changes the total count by "
                    "no more than 10%"
                ),
                "measured_result": seed_count_deviation,
                "passed": seed_count_deviation <= 0.10,
                "needed_for_abrupt_shift_claim": True,
                "needed_for_gradual_change_claim": True,
            },
            {
                "check_id": "original_points_found_again_across_reruns",
                "required_result": (
                    "at least 70% of original points are found by "
                    "12 or more of the 16 analysis setups"
                ),
                "measured_result": broadly_reproduced_fraction,
                "passed": broadly_reproduced_fraction >= 0.70,
                "needed_for_abrupt_shift_claim": True,
                "needed_for_gradual_change_claim": True,
            },
            {
                "check_id": "original_points_found_with_every_distance",
                "required_result": (
                    "each distance calculation finds at least 70% of the "
                    "original change points"
                ),
                "measured_result": distance_recovery,
                "passed": distance_recovery >= 0.70,
                "needed_for_abrupt_shift_claim": True,
                "needed_for_gradual_change_claim": True,
            },
            {
                "check_id": "closer_to_recommendation_changes_in_every_rerun",
                "required_result": (
                    "change points are closer to recommendation changes than "
                    "matched sentences in all 16 analysis setups"
                ),
                "measured_result": float(all_action_effects_positive),
                "passed": all_action_effects_positive,
                "needed_for_abrupt_shift_claim": True,
                "needed_for_gradual_change_claim": True,
            },
            {
                "check_id": "original_comparison_remains_above_zero",
                "required_result": (
                    "the entire 95% uncertainty range for the original "
                    "comparison stays above zero"
                ),
                "measured_result": float(original_action["trajectory_bootstrap_ci_low"]),
                "passed": original_action_ci_positive,
                "needed_for_abrupt_shift_claim": True,
                "needed_for_gradual_change_claim": True,
            },
        ]
    )


def plot_dashboard(
    analysis_setup_summary: pd.DataFrame,
    location_summary: pd.DataFrame,
    event_summary: pd.DataFrame,
    synthetic_summary: pd.DataFrame,
    path: Path,
) -> None:
    order = analysis_setup_summary["analysis_setup_id"].tolist()
    label_lookup = {
        "original": "original analysis",
        "distance_total_variation": "total variation distance",
        "distance_hellinger": "Hellinger distance",
        "distance_jensen_shannon": "Jensen–Shannon distance",
        "minimum_segment_5": "minimum segment: 5",
        "minimum_segment_10": "minimum segment: 10",
        "location_cutoff_0_50": "location cutoff: 0.50",
        "location_cutoff_0_90": "location cutoff: 0.90",
        "evidence_cutoff_3": "evidence cutoff: 3",
        "evidence_cutoff_20": "evidence cutoff: 20",
        "random_seed_7": "random seed: 7",
        "random_seed_19": "random seed: 19",
        "random_seed_73": "random seed: 73",
        "random_seed_101": "random seed: 101",
        "small_noise_check_1": "small-noise check: 1",
        "small_noise_check_2": "small-noise check: 2",
    }
    display_labels = [label_lookup[value] for value in order]
    y = np.arange(len(order))
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.2, 10.0),
        constrained_layout=True,
    )

    ax = axes[0, 0]
    ax.scatter(
        analysis_setup_summary["n_detected_change_points"],
        y,
        color=BLUE,
        s=34,
    )
    original_count = int(
        analysis_setup_summary.iloc[0]["n_detected_change_points"]
    )
    ax.axvline(original_count, color=DARK, linestyle="--", linewidth=1)
    ax.set_yticks(y, display_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of detected change points")
    ax.set_title(
        "A. Number found under each analysis setup\n"
        f"(original analysis: {original_count})"
    )
    ax.grid(axis="x", color=GRID)

    ax = axes[0, 1]
    location = location_summary.set_index("analysis_setup_id").reindex(order)
    ax.scatter(100 * location["reference_recovery"], y, color=GREEN, s=34)
    ax.axvline(70, color=DARK, linestyle="--", linewidth=1)
    ax.text(
        70.8,
        len(order) - 0.4,
        "Required: 70%",
        color=DARK,
        fontsize=8,
        rotation=90,
        va="bottom",
    )
    ax.set_yticks(y, display_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 103)
    ax.set_xlabel(
        "Original change points also found within ±3 sentences (%)"
    )
    ax.set_title("B. How often each rerun finds the original change points")
    ax.grid(axis="x", color=GRID)

    ax = axes[1, 0]
    effects = (
        event_summary[event_summary["event_type"] == "action_change"]
        .set_index("analysis_setup_id")
        .reindex(order)
    )
    values = effects["observed_minus_expected_fraction"].to_numpy()
    low = effects["trajectory_bootstrap_ci_low"].to_numpy()
    high = effects["trajectory_bootstrap_ci_high"].to_numpy()
    ax.errorbar(
        100 * values,
        y,
        xerr=np.vstack([100 * (values - low), 100 * (high - values)]),
        fmt="o",
        color=BLUE,
        ecolor=LIGHT_BLUE,
        capsize=3,
    )
    ax.axvline(0, color=DARK, linewidth=1)
    ax.set_yticks(y, display_labels)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Extra percentage near recommendation changes,\n"
        "compared with sentences at the same reasoning stage"
    )
    ax.set_title(
        "C. Change points are more often near recommendation changes"
    )
    ax.grid(axis="x", color=GRID)

    ax = axes[1, 1]
    original = synthetic_summary[
        synthetic_summary["analysis_setup_id"] == "original"
    ].set_index("scenario")
    scenarios = [
        "stable_noise",
        "smooth_drift",
        "one_level_shift",
        "one_slope_change",
        "two_level_shifts",
    ]
    labels = [
        "Random noise\n(no shift)",
        "Smooth drift\n(no shift)",
        "One abrupt\nshift",
        "Gradual\nchange",
        "Two abrupt\nshifts",
    ]
    values = [
        float(original.loc[scenario, "trace_false_positive_rate"])
        if index < 2
        else float(original.loc[scenario, "change_point_recall"])
        for index, scenario in enumerate(scenarios)
    ]
    colors = [ORANGE, ORANGE, GREEN, GREEN, GREEN]
    ax.bar(np.arange(len(scenarios)), 100 * np.asarray(values), color=colors)
    ax.axhline(5, color=ORANGE, linestyle=":", linewidth=1)
    ax.axhline(80, color=GREEN, linestyle="--", linewidth=1)
    ax.text(
        0.05,
        6.5,
        "5% maximum false alarms",
        color=ORANGE,
        fontsize=8,
        ha="left",
    )
    ax.text(
        0.05,
        81.5,
        "80% required",
        color=GREEN,
        fontsize=8,
        ha="left",
    )
    for index, value in enumerate(values):
        ax.text(
            index,
            min(102, 100 * value + 2),
            f"{100 * value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color=DARK,
        )
    ax.set_xticks(np.arange(len(scenarios)), labels)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylabel(
        "False alarms for first two;\n"
        "correctly found changes for last three (%)"
    )
    ax.set_ylim(0, 105)
    ax.set_title("D. Original analysis on traces with known answers")
    ax.grid(axis="y", color=GRID)

    figure.suptitle(
        "Do the Change-Point Results Survive Reasonable Analysis Choices?",
        fontsize=14,
    )
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_report(
    path: Path,
    *,
    analysis_setup_summary: pd.DataFrame,
    location_summary: pd.DataFrame,
    point_stability: pd.DataFrame,
    event_summary: pd.DataFrame,
    synthetic_summary: pd.DataFrame,
    stopping_rules: pd.DataFrame,
    event_window: int,
    location_tolerance: int,
) -> None:
    original_count = int(
        analysis_setup_summary.set_index("analysis_setup_id").loc[
            "original", "n_detected_change_points"
        ]
    )
    count_min = int(analysis_setup_summary["n_detected_change_points"].min())
    count_max = int(analysis_setup_summary["n_detected_change_points"].max())
    broadly_reproduced_count = int(
        point_stability["found_by_at_least_12_of_16_setups"].sum()
    )
    original_action = event_summary[
        (event_summary["analysis_setup_id"] == "original")
        & (event_summary["event_type"] == "action_change")
    ].iloc[0]
    action_differences = event_summary[
        event_summary["event_type"] == "action_change"
    ]["observed_minus_expected_fraction"]
    original_known_answer_checks = synthetic_summary[
        synthetic_summary["analysis_setup_id"] == "original"
    ].set_index("scenario")
    abrupt_rows = stopping_rules[
        stopping_rules["needed_for_abrupt_shift_claim"]
    ]
    broad_rows = stopping_rules[
        stopping_rules["needed_for_gradual_change_claim"]
    ]
    abrupt_passes = bool(abrupt_rows["passed"].all())
    broad_passes = bool(broad_rows["passed"].all())
    abrupt_decision = (
        "STOP: no more robustness checks are needed for the narrow claim that BEAST "
        "finds abrupt shifts in the distance-from-start time series."
        if abrupt_passes
        else "CONTINUE only with the failed checks for abrupt shifts."
    )
    broad_decision = (
        "STOP: the results also support gradual changes."
        if broad_passes
        else "DO NOT CLAIM that this method reliably finds gradual changes."
    )

    check_labels = {
        "false_alarm_rate_without_abrupt_shift": (
            "False alarms when there is no abrupt shift"
        ),
        "known_abrupt_shifts_correctly_found": (
            "Correctly finding known abrupt shifts"
        ),
        "known_abrupt_shifts_placed_accurately": (
            "Placing known abrupt shifts accurately"
        ),
        "gradual_slope_changes_correctly_found": (
            "Correctly finding gradual slope changes"
        ),
        "similar_counts_with_different_random_seeds": (
            "Similar counts with different random seeds"
        ),
        "original_points_found_again_across_reruns": (
            "Original change points found again across reruns"
        ),
        "original_points_found_with_every_distance": (
            "Original change points found with every distance calculation"
        ),
        "closer_to_recommendation_changes_in_every_rerun": (
            "Change points closer to recommendation changes in every rerun"
        ),
        "original_comparison_remains_above_zero": (
            "Original comparison remains above zero after accounting for uncertainty"
        ),
    }

    def display_observed(row: Any) -> str:
        if row.check_id == "known_abrupt_shifts_placed_accurately":
            return f"{float(row.measured_result):.1f} sentences"
        if row.check_id == "closer_to_recommendation_changes_in_every_rerun":
            return "16 of 16 analysis setups"
        if row.check_id == "original_comparison_remains_above_zero":
            return f"{float(row.measured_result):.1%} lower end"
        return f"{float(row.measured_result):.1%}"

    gate_lines = [
        f"| {check_labels[row.check_id]} | {row.required_result} | "
        f"{display_observed(row)} | {'Pass' if row.passed else 'Fail'} | "
        f"{'Yes' if row.needed_for_abrupt_shift_claim else 'No'} |"
        for row in stopping_rules.itertuples()
    ]
    failed_abrupt = abrupt_rows[~abrupt_rows["passed"]]
    if failed_abrupt.empty:
        next_step = (
            "Do not add more alternative BEAST settings for the abrupt-shift claim. "
            "Keep the current detector fixed and proceed to the belief-change and "
            "joint action-change analyses. If a future paper needs to claim that "
            "gradual changes are also detected, that requires a separate study or a "
            "method designed for gradual drift."
        )
    else:
        failed_names = ", ".join(
            failed_abrupt["check_id"].str.replace("_", " ")
        )
        next_step = (
            f"Restrict follow-up work to: {failed_names}. Do not add unrelated "
            "analysis setups."
        )

    report = f"""# How Robust Are the Change-Point Results?

## Short answer

We ran the same BEAST analysis 16 times, changing one choice at a time. We call each
complete version an **analysis setup**. For example, one setup uses the original
choices, another uses Hellinger distance, and another requires longer segments.

The original analysis found {original_count} change points. The 15 alternative setups found between {count_min} and {count_max}. Of the original {original_count} points, {broadly_reproduced_count} were found within {location_tolerance} sentences by at least 12 of the 16 setups.

In the original analysis, {original_action.observed_fraction:.1%} of change points were within {event_window} sentences of a recommendation change. For ordinary sentence positions at the same stage of reasoning, the corresponding rate was {original_action.progress_matched_expected_fraction:.1%}. The difference was {100 * original_action.observed_minus_expected_fraction:.1f} percentage points. The 95% uncertainty range was {100 * original_action.trajectory_bootstrap_ci_low:.1f} to {100 * original_action.trajectory_bootstrap_ci_high:.1f} percentage points; this range was estimated by resampling whole trajectories. Across all 16 setups, the difference stayed between {100 * action_differences.min():.1f} and {100 * action_differences.max():.1f} percentage points.

**Decision for abrupt shifts: {abrupt_decision}**

**Decision for gradual changes: {broad_decision}**

## What an analysis setup means

An analysis setup is one complete set of choices used to run BEAST. The original
setup uses the L2 distance between each four-action probability distribution and the
distribution before reasoning. We reran the analysis while changing:

- total-variation, Hellinger, and Jensen--Shannon distances;
- minimum segment lengths of 3, 5, and 10 sentences;
- the probability cutoff for reporting a location: .50, .70, or .90;
- the amount of evidence required before reporting any change: 3, 9, or 20;
- five random seeds;
- two checks after adding a small amount of noise to the input.

The added noise had a standard deviation equal to 3% of the observed range in each
time series. This is a small-input-change check, not an exact reproduction of the
Forking Paths paper. Every setup remains offline: BEAST uses sentences after a
candidate position when deciding whether that position is a change point.

## Checks using traces with known answers

We also created time series where we knew in advance whether and where a change
occurred. Their lengths matched typical observed traces. The abrupt shift moved 0.18
probability mass between two actions, producing an L2 change of 0.255. This is close
to the lower quarter of abrupt changes in the real data.

- With only random fluctuation and no abrupt shift, BEAST raised a false alarm in {original_known_answer_checks.loc['stable_noise', 'trace_false_positive_rate']:.1%} of traces.
- With smooth drift but no abrupt shift, it raised a false alarm in {original_known_answer_checks.loc['smooth_drift', 'trace_false_positive_rate']:.1%} of traces.
- It correctly found {original_known_answer_checks.loc['one_level_shift', 'change_point_recall']:.1%} of single abrupt shifts.
- It correctly found {original_known_answer_checks.loc['two_level_shifts', 'change_point_recall']:.1%} of the two abrupt shifts in traces containing two shifts.
- It correctly located {original_known_answer_checks.loc['one_slope_change', 'change_point_recall']:.1%} of gradual slope changes within three sentences.

## Checks used to decide when to stop

| Check | What had to happen | What happened | Result | Needed for the abrupt-shift claim? |
|---|---|---:|---|---|
{chr(10).join(gate_lines)}

## Next step

{next_step}

The reruns do not need to return exactly the same sentence every time. The checks ask
whether false alarms are rare, known abrupt shifts are found, most original locations
are found again, and the comparison with recommendation changes remains positive.

## Files

- `MEASUREMENT_FREEZE.md`: the fixed outcome definition for subsequent prediction analyses.
- `analysis_setup_grid.csv`: the 16 complete analysis setups.
- `analysis_setup_summary.csv`: change-point counts from each setup.
- `change_point_location_summary.csv`: how often each setup finds the original points.
- `original_point_stability.csv`: how often each original point is found again.
- `change_point_event_comparison.csv`: direct comparison with matched sentences.
- `known_answer_check_summary.csv`: false alarms and correct detections.
- `robustness_check_results.csv`: the pass/fail decision table.
- `figs/change_point_robustness_summary.png`: self-contained visual summary.
"""
    path.write_text(report)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figs"
    figure_dir.mkdir(exist_ok=True)
    setup_matplotlib()

    source = pd.read_csv(args.positions)
    events = build_event_positions(pd.read_csv(args.events))
    analysis_setups = default_robustness_analysis_setups()
    pd.DataFrame(
        [analysis_setup.to_record() for analysis_setup in analysis_setups]
    ).to_csv(args.output_dir / "analysis_setup_grid.csv", index=False)

    analysis_setup_rows: list[dict[str, Any]] = []
    state_frames: list[pd.DataFrame] = []
    detected_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    for analysis_setup in analysis_setups:
        print(f"Running analysis setup: {analysis_setup.analysis_setup_id}")
        positions, states, modes = fit_robustness_analysis_setup(
            source,
            analysis_setup,
            samples=args.samples,
        )
        detected = modes[modes["detected_change_point"]].copy()
        state_frames.append(states)
        detected_frames.append(detected)
        analysis_setup_rows.append(
            {
                **analysis_setup.to_record(),
                "n_states": states["example_id"].nunique(),
                "n_states_with_change_points": int(
                    (states["n_detected_change_points"] > 0).sum()
                ),
                "n_detected_change_points": len(detected),
                "mean_change_points_per_state": states[
                    "n_detected_change_points"
                ].mean(),
            }
        )
        proximity = event_proximity_rows(
            positions,
            detected,
            events,
            event_types=EVENT_TYPES,
            window=args.event_window,
        )
        summarized = summarize_event_proximity(
            proximity,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=args.seed,
        )
        for column, value in reversed(tuple(analysis_setup.to_record().items())):
            summarized.insert(0, column, value)
        event_frames.append(summarized)

    analysis_setup_summary = pd.DataFrame(analysis_setup_rows)
    states_all = pd.concat(state_frames, ignore_index=True)
    detected_all = pd.concat(detected_frames, ignore_index=True)
    event_summary = pd.concat(event_frames, ignore_index=True)
    original_points = detected_all[
        detected_all["analysis_setup_id"] == "original"
    ].copy()

    location_rows: list[dict[str, Any]] = []
    match_frames: list[pd.DataFrame] = []
    for analysis_setup in analysis_setups:
        candidate = detected_all[
            detected_all["analysis_setup_id"] == analysis_setup.analysis_setup_id
        ]
        matches, summary = match_detected_locations(
            original_points,
            candidate,
            tolerance=args.location_tolerance,
        )
        matches.insert(0, "analysis_setup_id", analysis_setup.analysis_setup_id)
        matches.insert(
            1,
            "analysis_setup_family",
            analysis_setup.analysis_setup_family,
        )
        match_frames.append(matches)
        location_rows.append({**analysis_setup.to_record(), **summary})
    matches_all = pd.concat(match_frames, ignore_index=True)
    location_summary = pd.DataFrame(location_rows)
    point_stability = build_original_point_stability(
        original_points,
        matches_all,
        len(analysis_setups),
    )

    empirical_lengths = source.groupby("example_id").size().to_numpy(dtype=int)
    print("Running synthetic calibration")
    synthetic_rows = run_synthetic_benchmark(
        analysis_setups,
        empirical_lengths,
        samples=args.synthetic_samples,
        repeats=args.synthetic_repeats,
        original_repeats=args.original_synthetic_repeats,
        tolerance=args.location_tolerance,
        seed=args.seed,
    )
    synthetic_summary = summarize_synthetic_benchmark(synthetic_rows)
    stopping_rules = stopping_rule_table(
        analysis_setup_summary,
        location_summary,
        point_stability,
        event_summary,
        synthetic_summary,
    )

    analysis_setup_summary.to_csv(
        args.output_dir / "analysis_setup_summary.csv",
        index=False,
    )
    states_all.rename(
        columns={"primary_step_failure_mode": "main_step_failure_mode"}
    ).to_csv(
        args.output_dir / "state_summaries_by_analysis_setup.csv",
        index=False,
    )
    detected_all.rename(
        columns={"primary_step_failure_mode": "main_step_failure_mode"}
    ).to_csv(
        args.output_dir / "detected_change_points_by_analysis_setup.csv",
        index=False,
    )
    location_summary.rename(
        columns={
            "reference_count": "original_change_point_count",
            "candidate_count": "change_point_count_in_this_setup",
            "matched_within_tolerance": (
                "original_points_found_again_within_tolerance"
            ),
            "reference_recovery": "fraction_of_original_points_found_again",
            "candidate_precision": (
                "fraction_of_this_setups_points_matching_original"
            ),
            "median_absolute_location_difference": (
                "typical_location_difference_sentences"
            ),
        }
    ).to_csv(
        args.output_dir / "change_point_location_summary.csv",
        index=False,
    )
    matches_all.rename(
        columns={
            "reference_position": "original_position",
            "candidate_position": "position_in_this_setup",
            "absolute_location_difference": (
                "location_difference_sentences"
            ),
        }
    ).to_csv(
        args.output_dir / "change_point_location_matches.csv",
        index=False,
    )
    point_stability.rename(
        columns={
            "recovery_fraction": (
                "fraction_of_analysis_setups_finding_this_point"
            )
        }
    ).to_csv(
        args.output_dir / "original_point_stability.csv",
        index=False,
    )
    event_summary.rename(
        columns={
            "observed_fraction": "change_point_fraction_near_event",
            "progress_matched_expected_fraction": (
                "matched_sentence_fraction_near_event"
            ),
            "observed_minus_expected_fraction": (
                "difference_in_fraction_near_event"
            ),
            "trajectory_bootstrap_ci_low": (
                "trajectory_resampling_95pct_low"
            ),
            "trajectory_bootstrap_ci_high": (
                "trajectory_resampling_95pct_high"
            ),
        }
    ).to_csv(
        args.output_dir / "change_point_event_comparison.csv",
        index=False,
    )
    synthetic_rows.rename(
        columns={
            "n_expected_change_points": "known_change_count",
            "n_detected_change_points": "detected_change_count",
            "n_matched_change_points": (
                "known_changes_found_within_tolerance"
            ),
            "trace_false_positive": "raised_false_alarm",
            "trace_detected": "found_any_change",
            "change_point_recall": "fraction_of_known_changes_found",
            "median_localization_error": (
                "typical_location_error_sentences"
            ),
        }
    ).to_csv(
        args.output_dir / "known_answer_check_runs.csv",
        index=False,
    )
    synthetic_summary.rename(
        columns={
            "trace_false_positive_rate": "false_alarm_rate",
            "trace_detection_rate": (
                "fraction_of_traces_with_any_detected_change"
            ),
            "change_point_recall": "fraction_of_known_changes_found",
            "median_localization_error": (
                "typical_location_error_sentences"
            ),
            "mean_detected_change_points": (
                "mean_detected_change_count"
            ),
        }
    ).to_csv(
        args.output_dir / "known_answer_check_summary.csv",
        index=False,
    )
    stopping_rules.to_csv(
        args.output_dir / "robustness_check_results.csv",
        index=False,
    )

    plot_dashboard(
        analysis_setup_summary,
        location_summary,
        event_summary,
        synthetic_summary,
        figure_dir / "change_point_robustness_summary.png",
    )
    write_report(
        args.output_dir / "run_report.md",
        analysis_setup_summary=analysis_setup_summary,
        location_summary=location_summary,
        point_stability=point_stability,
        event_summary=event_summary,
        synthetic_summary=synthetic_summary,
        stopping_rules=stopping_rules,
        event_window=args.event_window,
        location_tolerance=args.location_tolerance,
    )
    manifest = {
        "analysis": "offline_beast_action_distribution_cpd_robustness",
        "input_positions": str(args.positions),
        "input_positions_sha256": sha256(args.positions),
        "input_events": str(args.events),
        "input_events_sha256": sha256(args.events),
        "n_analysis_setups": len(analysis_setups),
        "empirical_beast_samples": args.samples,
        "synthetic_beast_samples": args.synthetic_samples,
        "synthetic_repeats_per_scenario": args.synthetic_repeats,
        "original_synthetic_repeats_per_scenario": (
            args.original_synthetic_repeats
        ),
        "bootstrap_repeats": args.bootstrap_repeats,
        "event_window_sentences": args.event_window,
        "location_tolerance_sentences": args.location_tolerance,
        "all_checks_for_abrupt_shift_claim_pass": bool(
            stopping_rules.loc[
                stopping_rules["needed_for_abrupt_shift_claim"],
                "passed",
            ].all()
        ),
        "all_checks_for_gradual_change_claim_pass": bool(
            stopping_rules.loc[
                stopping_rules["needed_for_gradual_change_claim"],
                "passed",
            ].all()
        ),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Wrote CPD robustness analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
