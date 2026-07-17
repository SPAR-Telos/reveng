from reveng.experiments.plan_reasoning_alignment import _normalize_action_sequence, _prefix_exact


def test_normalize_action_sequence_pads_to_horizon():
    assert _normalize_action_sequence(["down", "RIGHT"], pad_to=5) == [
        "DOWN", "RIGHT", "__PAD__", "__PAD__", "__PAD__"
    ]


def test_prefix_exact_compares_requested_prefix_only():
    predicted = ["DOWN", "RIGHT", "LEFT"]
    target = ["DOWN", "RIGHT", "UP"]
    assert _prefix_exact(predicted, target, 1) == 1
    assert _prefix_exact(predicted, target, 2) == 1
    assert _prefix_exact(predicted, target, 3) == 0
