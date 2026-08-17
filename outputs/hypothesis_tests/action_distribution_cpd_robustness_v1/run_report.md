# How Robust Are the Change-Point Results?

## Short answer

We ran the same BEAST analysis 16 times, changing one choice at a time. We call each
complete version an **analysis setup**. For example, one setup uses the original
choices, another uses Hellinger distance, and another requires longer segments.

The original analysis found 160 change points. The 15 alternative setups found between 74 and 206. Of the original 160 points, 127 were found within 3 sentences by at least 12 of the 16 setups.

In the original analysis, 86.9% of change points were within 3 sentences of a recommendation change. For ordinary sentence positions at the same stage of reasoning, the corresponding rate was 65.3%. The difference was 21.6 percentage points. The 95% uncertainty range was 16.8 to 26.8 percentage points; this range was estimated by resampling whole trajectories. Across all 16 setups, the difference stayed between 18.1 and 22.2 percentage points.

**Decision for abrupt shifts: STOP: no more robustness checks are needed for the narrow claim that BEAST finds abrupt shifts in the distance-from-start time series.**

**Decision for gradual changes: DO NOT CLAIM that this method reliably finds gradual changes.**

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

- With only random fluctuation and no abrupt shift, BEAST raised a false alarm in 0.0% of traces.
- With smooth drift but no abrupt shift, it raised a false alarm in 1.0% of traces.
- It correctly found 100.0% of single abrupt shifts.
- It correctly found 100.0% of the two abrupt shifts in traces containing two shifts.
- It correctly located 0.0% of gradual slope changes within three sentences.

## Checks used to decide when to stop

| Check | What had to happen | What happened | Result | Needed for the abrupt-shift claim? |
|---|---|---:|---|---|
| False alarms when there is no abrupt shift | false alarms in no more than 5% of either kind of no-abrupt-shift trace | 1.0% | Pass | Yes |
| Correctly finding known abrupt shifts | correctly find at least 80% of the known abrupt shifts | 100.0% | Pass | Yes |
| Placing known abrupt shifts accurately | typical location error no greater than 3 sentences | 0.0 sentences | Pass | Yes |
| Correctly finding gradual slope changes | correctly locate at least 80% of gradual slope changes | 0.0% | Fail | No |
| Similar counts with different random seeds | changing the random seed changes the total count by no more than 10% | 3.8% | Pass | Yes |
| Original change points found again across reruns | at least 70% of original points are found by 12 or more of the 16 analysis setups | 79.4% | Pass | Yes |
| Original change points found with every distance calculation | each distance calculation finds at least 70% of the original change points | 78.8% | Pass | Yes |
| Change points closer to recommendation changes in every rerun | change points are closer to recommendation changes than matched sentences in all 16 analysis setups | 16 of 16 analysis setups | Pass | Yes |
| Original comparison remains above zero after accounting for uncertainty | the entire 95% uncertainty range for the original comparison stays above zero | 16.8% lower end | Pass | Yes |

## Next step

Do not add more alternative BEAST settings for the abrupt-shift claim. Keep the current detector fixed and proceed to the belief-change and joint action-change analyses. If a future paper needs to claim that gradual changes are also detected, that requires a separate study or a method designed for gradual drift.

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
