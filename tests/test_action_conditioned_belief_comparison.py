import csv
import json
from pathlib import Path

from reveng.experiments.action_conditioned_belief_comparison import (
    build_action_conditioned_belief_comparison,
)


def test_build_action_conditioned_belief_comparison_outputs_expected_counts(tmp_path: Path):
    out_dir = tmp_path / "comparison"
    manifest = build_action_conditioned_belief_comparison(output_dir=str(out_dir))

    assert manifest["n_pre_alignment_rows"] == 60
    assert manifest["n_post_alignment_rows"] == 60
    assert manifest["n_acted_direction_rows"] == 120
    assert "Failure-slice white-box comparison not included" in manifest["failure_slice_note"]

    acted_rows = list(csv.DictReader((out_dir / "acted_direction_rows.csv").open(newline="")))
    assert len(acted_rows) == 120

    summary_rows = list(csv.DictReader((out_dir / "summary_table.csv").open(newline="")))
    assert len(summary_rows) == 48

    all_clean_pre = [
        row
        for row in summary_rows
        if row["table_name"] == "directional_belief_accuracy"
        and row["subset_name"] == "all_clean_rows"
        and row["reasoning_split"] == "pre"
        and row["method_key"] == "whitebox_released_probe"
    ]
    assert len(all_clean_pre) == 1
    assert all_clean_pre[0]["accuracy"] == "14/15 (93.3%)"

    non_opt_rows = [
        row
        for row in summary_rows
        if row["subset_name"] == "non_optimal_action_rows"
        and row["reasoning_split"] == "all"
        and row["method_key"] == "whitebox_released_probe"
    ]
    assert len(non_opt_rows) == 2
    assert {row["table_name"] for row in non_opt_rows} == {"directional_belief_accuracy", "consistency_profile"}
    assert all(row["n"] == "0" for row in non_opt_rows)

    whitebox_mismatches = [
        row
        for row in acted_rows
        if row["method_key"] == "whitebox_released_probe"
        and row["acted_direction_belief_correct"] == "False"
    ]
    assert len(whitebox_mismatches) == 1
    assert whitebox_mismatches[0]["reasoning_split"] == "pre"
    assert whitebox_mismatches[0]["action_consistency_label"] == "inconsistent"

    markdown_text = (out_dir / "summary_table.md").read_text()
    assert "Failure slice note:" in markdown_text
    assert "Black-box logprob yes/no (T=0.7)" in markdown_text
