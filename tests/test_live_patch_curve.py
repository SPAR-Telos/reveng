import csv
import json
from pathlib import Path

from reveng.experiments.live_patch_curve import (
    _build_step_patch_input,
    _extract_final_action,
    _run_live_patch_curve,
)


FINAL_MARKER = "<|end|><|start|>assistant<|channel|>final<|message|>"


def _make_trace(action: str, reasoning: str) -> dict:
    return {
        "prompt": {
            "prompt_template": "Current grid state:\n\n{{grid_state}}",
        },
        "steps": [
            {
                "grid_state": [
                    "  0 1 2",
                    "0 # # #",
                    "1 # A G",
                    "2 # # #",
                ],
                "output_text": (
                    "<|channel|>analysis<|message|>"
                    + reasoning
                    + FINAL_MARKER
                    + '{\n  "action": "'
                    + action
                    + '"\n}<|return|>'
                ),
                "output_tokens": [
                    {"token": action, "token_groups": ["output", "final", "action"]},
                ],
            }
        ],
    }


class FakeBackend:
    def num_layers(self) -> int:
        return 4

    def encode_text(self, text: str) -> list[int]:
        return [idx for idx, _ in enumerate(text.split(), start=1)]

    def action_probabilities(
        self,
        *,
        prefix_a_text: str,
        prefix_b_text: str,
        layer_idx: int,
        patch_positions_a: list[int],
        patch_positions_b: list[int],
        candidate_labels: list[str],
    ) -> dict[str, float]:
        assert prefix_a_text
        assert prefix_b_text
        assert len(patch_positions_a) == 3
        assert len(patch_positions_b) == 3
        base = {
            "UP": 0.05,
            "DOWN": 0.20 + 0.10 * layer_idx,
            "LEFT": 0.55 - 0.08 * layer_idx,
            "RIGHT": 0.10,
        }
        return {label: base[label] for label in candidate_labels}


def _fake_backend_factory(model_name_or_path: str, **_: object) -> FakeBackend:
    assert model_name_or_path == "fake-llama"
    return FakeBackend()


def test_extract_final_action_prefers_output_tokens():
    step = {
        "output_tokens": [
            {"token": "LEFT", "token_groups": ["output", "final", "action"]},
        ],
        "output_text": '{"action": "RIGHT"}',
    }
    assert _extract_final_action(step) == "LEFT"


def test_build_step_patch_input_uses_tail_reasoning_tokens():
    trace = _make_trace("LEFT", "Plan path left twice then finish.")
    backend = FakeBackend()
    patch_input = _build_step_patch_input(
        trace,
        step_index=0,
        backend=backend,
        reasoning_window_size=3,
    )
    expected_last = len(backend.encode_text(patch_input.prompt_text + patch_input.reasoning_text))
    assert patch_input.action_label == "LEFT"
    assert patch_input.reasoning_positions == [expected_last - 3, expected_last - 2, expected_last - 1]
    assert patch_input.prefix_text.endswith('{\n  "action": "')


def test_run_live_patch_curve_writes_expected_outputs(tmp_path: Path):
    pair_dir = tmp_path / "artifacts" / "pair_goal_move_009"
    pair_dir.mkdir(parents=True)
    (pair_dir / "A.json").write_text(
        json.dumps(_make_trace("LEFT", "We reason locally and choose left."), indent=2)
    )
    (pair_dir / "B.json").write_text(
        json.dumps(_make_trace("DOWN", "We reason locally and choose down."), indent=2)
    )

    output_dir = tmp_path / "live_curve"
    _run_live_patch_curve(
        model_name_or_path="fake-llama",
        pair_id="pair_goal_move_009",
        step_index=0,
        output_dir=str(output_dir),
        artifacts_dir=str(tmp_path / "artifacts"),
        backend_factory=_fake_backend_factory,
    )

    csv_path = output_dir / "probability_by_layer.csv"
    png_path = output_dir / "probability_by_layer.png"
    delta_path = output_dir / "probability_delta_by_layer.png"
    metadata_path = output_dir / "metadata.json"

    assert csv_path.exists()
    assert png_path.exists()
    assert delta_path.exists()
    assert metadata_path.exists()

    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["layer"]) for row in rows] == [0, 1, 2, 3]
    assert rows[0]["source_action"] == "LEFT"
    assert rows[0]["target_action"] == "DOWN"
    assert "left_prob" in rows[0]
    assert "down_prob" in rows[0]

    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "ok"
    assert metadata["source_action"] == "LEFT"
    assert metadata["target_action"] == "DOWN"
    assert len(metadata["patch_positions_a"]) == 3
