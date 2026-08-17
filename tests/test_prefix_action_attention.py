import math

import torch

from reveng.experiments.prefix_action_attention import (
    REGION_NAMES,
    add_grid_action_regions,
    aggregate_attention,
    grid_cell_character_spans,
    paired_attention_differences,
    trajectory_bootstrap_ci,
)


def test_grid_cell_spans_find_agent_and_task_objects() -> None:
    grid = "  0 1 2\n0 # # #\n1 # A K\n2 # D G"
    cells = grid_cell_character_spans(grid)

    assert cells[(1, 1)][2] == "A"
    assert cells[(1, 2)][2] == "K"
    assert cells[(2, 1)][2] == "D"


def test_action_regions_use_current_and_destination_cells() -> None:
    grid = "  0 1 2\n0 # # #\n1 # A _\n2 # K G"
    cells = grid_cell_character_spans(grid)
    prepared = {
        "grid_cells": cells,
        "grid_start": 10,
        "offsets": [(index, index + 1) for index in range(100)],
        "query_index": 99,
        "region_spans": {"chosen_move_cells": [], "task_objects": []},
        "region_token_indices": {},
    }

    add_grid_action_regions(prepared, current_position="1,1", action="RIGHT")

    assert len(prepared["region_spans"]["chosen_move_cells"]) == 2
    assert len(prepared["region_spans"]["task_objects"]) == 2


def test_attention_aggregation_preserves_mass_and_mean() -> None:
    attention = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    regions = {name: [] for name in REGION_NAMES}
    regions["grid"] = [0, 1]

    result = aggregate_attention(
        attention,
        sentence_token_indices=[[1, 2]],
        region_token_indices=regions,
    )

    assert math.isclose(float(result["sentence_mass"][0, 0]), 0.5)
    assert math.isclose(float(result["sentence_mean"][0, 0]), 0.25)
    assert math.isclose(float(result["region_mass"][0, 0]), 0.3, rel_tol=1e-6)


def test_paired_difference_subtracts_matched_control_change() -> None:
    pair_rows = [
        {
            "pair_id": "p",
            "item_role": "detected_change_point",
            "example_id": "e",
            "trajectory_id": "t",
            "step_index": "1",
            "position_index": "3",
            "match_quality": "primary",
        },
        {
            "pair_id": "p",
            "item_role": "matched_non_change_sentence",
            "example_id": "e",
            "trajectory_id": "t",
            "step_index": "1",
            "position_index": "8",
            "match_quality": "primary",
        },
    ]
    rows = []
    for position, value in ((2, 0.1), (3, 0.4), (7, 0.2), (8, 0.3)):
        rows.append(
            {
                "example_id": "e",
                "position_index": position,
                "layer": 15,
                "region": "grid",
                "available": True,
                "mean_attention_per_token": value,
            }
        )

    result = paired_attention_differences(
        rows, pair_rows, metric="mean_attention_per_token"
    )

    assert len(result) == 1
    assert math.isclose(result[0]["difference_in_differences"], 0.2)


def test_trajectory_bootstrap_weights_trajectories_equally() -> None:
    rows = [
        {"trajectory_id": "a", "x": 1.0},
        {"trajectory_id": "a", "x": 1.0},
        {"trajectory_id": "b", "x": -1.0},
    ]

    mean, low, high = trajectory_bootstrap_ci(rows, value_key="x", repeats=100)

    assert mean == 0.0
    assert low <= mean <= high
