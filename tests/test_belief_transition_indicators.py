import csv
from pathlib import Path

from reveng.experiments.belief_transition_indicators import (
    build_belief_transition_indicator_analysis,
)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_indicator_analysis_distinguishes_leading_loss_and_recovery(tmp_path: Path) -> None:
    _write(
        tmp_path / "action_transition_rows.csv",
        [
            {
                "event_id": "loss",
                "example_id": "e",
                "reasoning_step_idx": 3,
                "event_type": "sustained_optimal_to_suboptimal",
            },
            {
                "event_id": "recovery",
                "example_id": "e",
                "reasoning_step_idx": 6,
                "event_type": "suboptimal_to_optimal_recovery",
            },
        ],
    )
    _write(
        tmp_path / "belief_shift_rows.csv",
        [
            {
                "shift_id": "onset",
                "example_id": "e",
                "reasoning_step_idx": 2,
                "question_id": "wall_right",
                "question_family": "wall_directional",
                "shift_type": "belief_error_onset",
            },
            {
                "shift_id": "corrected",
                "example_id": "e",
                "reasoning_step_idx": 5,
                "question_id": "wall_right",
                "question_family": "wall_directional",
                "shift_type": "belief_error_recovery",
            },
        ],
    )

    build_belief_transition_indicator_analysis(run_dir=str(tmp_path), event_window=1)

    rows = list(csv.DictReader((tmp_path / "belief_transition_indicator_summary.csv").open()))
    loss = next(row for row in rows if row["event_family"] == "optimal_to_suboptimal")
    recovery = next(row for row in rows if row["event_family"] == "suboptimal_to_optimal")
    assert float(loss["leading_event_coverage"]) == 1.0
    assert float(recovery["leading_event_coverage"]) == 1.0
