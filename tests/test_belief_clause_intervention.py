from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from reveng.experiments.belief_clause_intervention import (
    CONDITIONS,
    ClauseExperimentConfig,
    analyze_experiment,
    build_clause_context,
    prepare_experiment,
    query_experiment,
    select_scaled_cohort,
)
from reveng.experiments.belief_replacement_action import ACTIONS, BELIEF_IDS


def _sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    candidates: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    beliefs: list[dict[str, object]] = []
    truths = {
        "wall_left": "yes",
        "wall_right": "no",
        "wall_up": "yes",
        "wall_down": "no",
        "has_key": "no",
        "door_open": "no",
    }
    for index, action in enumerate(ACTIONS):
        example_id = f"example-{index}"
        trajectory_id = f"trajectory-{index // 2}"
        candidates.append(
            {
                "example_id": example_id,
                "trajectory_id": trajectory_id,
                "grid_text": "  0 1 2\n0 # # #\n1 # A _\n2 # # #",
                "carrying_key": False,
                "probe_truths_json": json.dumps(truths),
                "optimal_actions_json": json.dumps([action]),
            }
        )
        actions.extend(
            [
                {
                    "example_id": example_id,
                    "trajectory_id": trajectory_id,
                    "position_index": 0,
                    "action_label": ACTIONS[(index + 1) % len(ACTIONS)],
                    "action_probabilities_json": json.dumps(
                        {
                            candidate: 0.7
                            if candidate == ACTIONS[(index + 1) % len(ACTIONS)]
                            else 0.1
                            for candidate in ACTIONS
                        }
                    ),
                },
                {
                    "example_id": example_id,
                    "trajectory_id": trajectory_id,
                    "position_index": 1,
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
        for question_id in BELIEF_IDS:
            label = truths[question_id]
            probabilities = {
                "yes": 0.8 if label == "yes" else 0.1,
                "no": 0.8 if label == "no" else 0.1,
                "unknown": 0.1,
            }
            beliefs.append(
                {
                    "example_id": example_id,
                    "trajectory_id": trajectory_id,
                    "position_index": 1,
                    "question_id": question_id,
                    "belief_is_error": False,
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


def _config(tmp_path: Path) -> ClauseExperimentConfig:
    candidates, actions, beliefs = _sources(tmp_path)
    return ClauseExperimentConfig(
        candidate_rows_path=candidates,
        prefix_action_rows_path=actions,
        belief_rows_path=beliefs,
        output_dir=tmp_path / "output",
        model_path=tmp_path / "model",
        minimum_free_gib=1.0,
        bootstrap_repetitions=50,
    )


def test_scaled_cohort_uses_all_rows_and_flips_one_wall(tmp_path: Path) -> None:
    candidates, actions, beliefs = _sources(tmp_path)
    cohort = select_scaled_cohort(
        pd.read_csv(candidates),
        pd.read_csv(actions),
        pd.read_csv(beliefs),
    )
    assert len(cohort) == 4
    assert cohort["trajectory_id"].nunique() == 2
    for row in cohort.itertuples(index=False):
        verified = json.loads(row.verified_beliefs_json)
        counterfactual = json.loads(row.counterfactual_beliefs_json)
        changed = [key for key in BELIEF_IDS if verified[key] != counterfactual[key]]
        assert changed == [row.counterfactual_question_id]
        assert row.counterfactual_question_id.startswith("wall_")


def test_clause_context_conditions_do_not_include_an_action_recommendation() -> None:
    model = {
        question_id: {"yes": 0.2, "no": 0.7, "unknown": 0.1}
        for question_id in BELIEF_IDS
    }
    verified = {
        question_id: {"yes": 0.0, "no": 1.0, "unknown": 0.0}
        for question_id in BELIEF_IDS
    }
    counterfactual = json.loads(json.dumps(verified))
    counterfactual["wall_left"] = {"yes": 1.0, "no": 0.0, "unknown": 0.0}
    for condition in CONDITIONS:
        context = build_clause_context(
            grid_text="  0 1\n0 # #\n1 # A",
            carrying_key=False,
            condition=condition,
            model_beliefs=model,
            verified_beliefs=verified,
            counterfactual_beliefs=counterfactual,
        )
        assert "recommended action" not in context.lower()
        assert "full cot action" not in context.lower()
        assert "reasoning trace is unavailable" in context.lower()
    assert "model-generated belief readouts" in build_clause_context(
        grid_text="  0 1\n0 # #\n1 # A",
        carrying_key=False,
        condition="model_reports",
        model_beliefs=model,
        verified_beliefs=verified,
        counterfactual_beliefs=counterfactual,
    )
    assert "trusted external verifier" in build_clause_context(
        grid_text="  0 1\n0 # #\n1 # A",
        carrying_key=False,
        condition="verified_truth",
        model_beliefs=model,
        verified_beliefs=verified,
        counterfactual_beliefs=counterfactual,
    )


class _FakeReader:
    calls = 0

    def __init__(self, model_path: Path, *, temperature: float) -> None:
        self.model_path = model_path
        self.temperature = temperature

    def score_questions(self, *, context: str, questions: object) -> dict[str, object]:
        del context, questions
        action = ACTIONS[type(self).calls % len(ACTIONS)]
        type(self).calls += 1
        probabilities = {
            candidate: 0.7 if candidate == action else 0.1
            for candidate in ACTIONS
        }
        return {
            "action": {
                "answer": action,
                "probabilities": probabilities,
                "entropy_bits": 1.3568,
            }
        }


def test_query_resume_and_analysis(tmp_path: Path) -> None:
    _FakeReader.calls = 0
    config = _config(tmp_path)
    prepare_experiment(config)
    ready = lambda _index, _minimum: {"status": "ready", "free_gib": 20.0}
    first = query_experiment(
        config, reader_factory=_FakeReader, gpu_status_fn=ready
    )
    assert first["status"] == "completed"
    assert _FakeReader.calls == 4 * len(CONDITIONS)
    second = query_experiment(
        config, reader_factory=_FakeReader, gpu_status_fn=ready
    )
    assert second["status"] == "completed"
    assert _FakeReader.calls == 4 * len(CONDITIONS)

    paths = analyze_experiment(config)
    assert all(path.exists() for path in paths.values())
    rows = pd.read_csv(paths["rows"])
    summary = pd.read_csv(paths["summary"])
    contrasts = pd.read_csv(paths["contrasts"])
    assert len(rows) == 4 * len(CONDITIONS)
    assert set(summary["condition"]) == {"full_cot", "grid_only", *CONDITIONS}
    assert len(contrasts) == 7
    assert paths["subgroups"].exists()
    assert "trajectory-clustered" in paths["report"].read_text()
