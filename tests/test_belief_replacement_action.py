from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from reveng.experiments.belief_replacement_action import (
    ACTIONS,
    BELIEF_IDS,
    PilotConfig,
    analyze_experiment,
    build_replacement_context,
    jensen_shannon_divergence,
    parse_probabilities,
    parse_grid_text,
    prepare_experiment,
    query_experiment,
    select_smoke_cohort,
    total_variation,
)
from reveng.experiments.local_immediate_behavioral import _cacheable_prefix_length


def _synthetic_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    candidates: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    beliefs: list[dict[str, object]] = []
    for action_index, action in enumerate(ACTIONS):
        for error_index, error_group in enumerate((False, True)):
            number = 2 * action_index + error_index
            example_id = f"example-{number:02d}"
            trajectory_id = f"trajectory-{number:02d}"
            candidates.append(
                {
                    "example_id": example_id,
                    "trajectory_id": trajectory_id,
                    "grid_text": "  0 1 2\n0 # # #\n1 # A G\n2 # # #",
                    "carrying_key": False,
                }
            )
            grid_action = ACTIONS[(action_index + 1) % len(ACTIONS)]
            actions.extend(
                [
                    {
                        "example_id": example_id,
                        "trajectory_id": trajectory_id,
                        "position_index": 0,
                        "action_label": grid_action,
                        "action_probabilities_json": json.dumps(
                            {
                                candidate: 0.7 if candidate == grid_action else 0.1
                                for candidate in ACTIONS
                            }
                        ),
                    },
                    {
                        "example_id": example_id,
                        "trajectory_id": trajectory_id,
                        "position_index": 2,
                        "action_label": action,
                        "action_probabilities_json": json.dumps(
                            {
                                candidate: 0.7 if candidate == action else 0.1
                                for candidate in ACTIONS
                            }
                        ),
                    },
                ]
            )
            for belief_index, question_id in enumerate(BELIEF_IDS):
                is_error = bool(error_group and belief_index == 0)
                probabilities = (
                    {"yes": 0.85, "no": 0.10, "unknown": 0.05}
                    if is_error
                    else {"yes": 0.10, "no": 0.85, "unknown": 0.05}
                )
                beliefs.append(
                    {
                        "example_id": example_id,
                        "trajectory_id": trajectory_id,
                        "position_index": 2,
                        "question_id": question_id,
                        "belief_is_error": is_error,
                        "probabilities_json": json.dumps(probabilities),
                    }
                )
    candidate_path = tmp_path / "candidates.csv"
    action_path = tmp_path / "actions.csv"
    belief_path = tmp_path / "beliefs.csv"
    pd.DataFrame(candidates).to_csv(candidate_path, index=False)
    pd.DataFrame(actions).to_csv(action_path, index=False)
    pd.DataFrame(beliefs).to_csv(belief_path, index=False)
    return candidate_path, action_path, belief_path


def _config(tmp_path: Path) -> PilotConfig:
    candidate_path, action_path, belief_path = _synthetic_sources(tmp_path)
    return PilotConfig(
        candidate_rows_path=candidate_path,
        prefix_action_rows_path=action_path,
        belief_rows_path=belief_path,
        output_dir=tmp_path / "output",
        model_path=tmp_path / "model",
        minimum_free_gib=48.0,
    )


def test_probability_and_distribution_distances() -> None:
    parsed = parse_probabilities('{"UP": 2, "DOWN": 1, "LEFT": 1, "RIGHT": 0}', ACTIONS)
    assert parsed["UP"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        parse_probabilities(
            {"yes": -1, "no": 1, "unknown": 1}, ("yes", "no", "unknown")
        )
    p = {"UP": 1.0, "DOWN": 0.0, "LEFT": 0.0, "RIGHT": 0.0}
    q = {"UP": 0.0, "DOWN": 1.0, "LEFT": 0.0, "RIGHT": 0.0}
    assert total_variation(p, p) == pytest.approx(0.0)
    assert total_variation(p, q) == pytest.approx(1.0)
    assert jensen_shannon_divergence(p, p) == pytest.approx(0.0)
    assert jensen_shannon_divergence(p, q) == pytest.approx(1.0)


def test_single_question_scoring_retains_one_branch_token() -> None:
    assert _cacheable_prefix_length([[1, 2, 3]]) == 2
    assert _cacheable_prefix_length([[1, 2, 3], [1, 2, 4]]) == 2


def test_coordinate_labelled_grid_parsing() -> None:
    grid = "  0 1 2\n0 # # #\n1 # A G\n2 # _ #"
    assert parse_grid_text(grid) == [
        ["#", "#", "#"],
        ["#", "A", "G"],
        ["#", "_", "#"],
    ]
    with pytest.raises(ValueError, match="unsupported cell symbols"):
        parse_grid_text("  0\n0 X")


def test_balanced_deterministic_selection(tmp_path: Path) -> None:
    candidate_path, action_path, belief_path = _synthetic_sources(tmp_path)
    cohort = select_smoke_cohort(
        pd.read_csv(candidate_path), pd.read_csv(action_path), pd.read_csv(belief_path)
    )
    assert len(cohort) == 8
    assert cohort["trajectory_id"].nunique() == 8
    assert cohort["full_cot_action"].value_counts().to_dict() == {
        action: 2 for action in ACTIONS
    }
    assert cohort.groupby(["full_cot_action", "belief_error_group"]).size().eq(1).all()
    assert all(
        len(json.loads(value)) == 6 for value in cohort["canonical_beliefs_json"]
    )


def test_replacement_prompt_contains_only_requested_beliefs() -> None:
    beliefs = {
        question_id: {"yes": 0.1, "no": 0.8, "unknown": 0.1}
        for question_id in BELIEF_IDS
    }
    six = build_replacement_context(
        grid_text="# A G",
        carrying_key=False,
        beliefs=beliefs,
        belief_set_id="six_state_beliefs",
    )
    five = build_replacement_context(
        grid_text="# A G",
        carrying_key=False,
        beliefs=beliefs,
        belief_set_id="five_state_beliefs",
    )
    assert "At least one door is currently open" in six
    assert "At least one door is currently open" not in five
    assert "hit_wall_after" not in six
    assert "ground_truth" not in six
    assert "belief_is_error" not in six
    assert "Reasoning trace available so far" not in six
    assert "yes=0.100000, no=0.800000, unknown=0.100000" in six


class _FakeReader:
    calls = 0

    def __init__(self, model_path: Path, *, temperature: float) -> None:
        self.model_path = model_path
        self.temperature = temperature

    def score_questions(self, *, context: str, questions: object) -> dict[str, object]:
        type(self).calls += 1
        action = "UP" if "At least one door is currently open" in context else "DOWN"
        probabilities = {
            candidate: 0.7 if candidate == action else 0.1 for candidate in ACTIONS
        }
        return {
            "action": {
                "answer": action,
                "probabilities": probabilities,
                "entropy_bits": 1.3568,
            }
        }


def test_gpu_block_and_configuration_lock(tmp_path: Path) -> None:
    config = _config(tmp_path)
    prepare_experiment(config)
    status = query_experiment(
        config,
        gpu_status_fn=lambda _index, _minimum: {
            "status": "blocked_gpu_unavailable",
            "message": "mock unavailable",
        },
    )
    assert status["status"] == "blocked_gpu_unavailable"
    assert (
        "No numerical action-belief result"
        in (config.output_dir / "run_report.md").read_text()
    )
    changed = PilotConfig(
        candidate_rows_path=config.candidate_rows_path,
        prefix_action_rows_path=config.prefix_action_rows_path,
        belief_rows_path=config.belief_rows_path,
        output_dir=config.output_dir,
        model_path=config.model_path,
        temperature=1.0,
    )
    with pytest.raises(ValueError, match="different configuration"):
        query_experiment(changed)


def test_synthetic_query_resume_and_analysis(tmp_path: Path) -> None:
    _FakeReader.calls = 0
    config = _config(tmp_path)
    prepare_experiment(config)
    ready = lambda _index, _minimum: {"status": "ready", "free_gib": 80.0}  # noqa: E731
    first = query_experiment(config, reader_factory=_FakeReader, gpu_status_fn=ready)
    assert first["status"] == "completed"
    assert _FakeReader.calls == 16
    second = query_experiment(config, reader_factory=_FakeReader, gpu_status_fn=ready)
    assert second["status"] == "completed"
    assert _FakeReader.calls == 16
    checkpoint = config.output_dir / "replacement_action_readouts.jsonl"
    assert len(checkpoint.read_text().splitlines()) == 16

    paths = analyze_experiment(config)
    assert all(path.exists() for path in paths.values())
    assert paths["trace_figure"].name == "trace_level_action_disagreement.png"
    rows = pd.read_csv(paths["rows"])
    summary = pd.read_csv(paths["summary"])
    assert len(rows) == 16
    assert set(rows["belief_set_id"]) == {
        "six_state_beliefs",
        "five_state_beliefs",
    }
    assert set(summary["condition"]) == {
        "grid_only",
        "six_state_beliefs",
        "five_state_beliefs",
    }
    assert "eight-trace smoke test" in paths["report"].read_text()
