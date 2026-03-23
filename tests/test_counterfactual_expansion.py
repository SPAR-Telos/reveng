import pandas as pd
import pytest
from pathlib import Path

from reveng.experiments.counterfactual_expansion import (
    _build_layer_plan,
    _ensure_layer_manifest,
    _pick_best_layer,
    _resolve_manifest_path,
)


def test_build_layer_plan_appends_refine_layers_without_duplicates():
    coarse = [9, 12, 15, 18, 21]
    plan = _build_layer_plan(coarse_layers=coarse, best_layer=15, refine_radius=2)
    assert plan == [9, 12, 15, 18, 21, 13, 14, 16, 17]


def test_pick_best_layer_uses_action_true_then_disruptive_tiebreak():
    rows = pd.DataFrame(
        [
            {
                "layer": 12,
                "action_threshold": 0.70,
                "status": "ok",
                "sweep_stage": "coarse",
                "action_true_rate": 0.2,
                "total_pairs_action_true": 4,
                "disruptive_rate": 0.4,
            },
            {
                "layer": 15,
                "action_threshold": 0.70,
                "status": "ok",
                "sweep_stage": "coarse",
                "action_true_rate": 0.2,
                "total_pairs_action_true": 5,
                "disruptive_rate": 0.5,
            },
            {
                "layer": 18,
                "action_threshold": 0.70,
                "status": "ok",
                "sweep_stage": "coarse",
                "action_true_rate": 0.1,
                "total_pairs_action_true": 9,
                "disruptive_rate": 0.1,
            },
        ]
    )
    # Tie on action_true_rate between 12 and 15, pick higher action_true_count.
    best = _pick_best_layer(rows, selection_threshold=0.70, coarse_only=True)
    assert best == 15


def test_resolve_manifest_path_supports_layer_template():
    p = _resolve_manifest_path("data/cf/layer_{layer}/manifest.json", 12)
    assert str(p) == "data/cf/layer_12/manifest.json"


def test_ensure_layer_manifest_returns_existing_without_generation(tmp_path):
    manifest_path = tmp_path / "layer_9" / "manifest_for_counterfactual_activation_patching.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("[]")

    out = _ensure_layer_manifest(
        manifest_path_template=str(tmp_path / "layer_{layer}" / "manifest_for_counterfactual_activation_patching.json"),
        layer=9,
        auto_generate_layer_manifests=True,
        pair_manifest_path=tmp_path / "pair_manifest.json",
        base_artifacts_dir=tmp_path / "base_artifacts",
        patch_action_source="b",
        linear_target="a",
        synthetic_goal_prob=0.99,
        overwrite_layer_artifacts=False,
    )
    assert out == manifest_path


def test_ensure_layer_manifest_autogenerates_for_missing_template(tmp_path, monkeypatch):
    import reveng.experiments.counterfactual_expansion as exp

    template = tmp_path / "layer_{layer}" / "manifest_for_counterfactual_activation_patching.json"
    called = {"copy": 0, "build": 0}

    def fake_copy_base_ab_traces_for_pairs(pair_manifest_path, base_artifacts_dir, layer_output_dir):
        called["copy"] += 1
        layer_output_dir.mkdir(parents=True, exist_ok=True)

    def fake_build_counterfactual_patch_artifacts(**kwargs):
        called["build"] += 1
        output_dir = kwargs["output_dir"]
        manifest = (
            Path(output_dir) / "manifest_for_counterfactual_activation_patching.json"
        )
        manifest.write_text("[]")

    monkeypatch.setattr(exp, "_copy_base_ab_traces_for_pairs", fake_copy_base_ab_traces_for_pairs)
    monkeypatch.setattr(exp, "build_counterfactual_patch_artifacts", fake_build_counterfactual_patch_artifacts)

    out = _ensure_layer_manifest(
        manifest_path_template=str(template),
        layer=12,
        auto_generate_layer_manifests=True,
        pair_manifest_path=tmp_path / "pair_manifest.json",
        base_artifacts_dir=tmp_path / "base_artifacts",
        patch_action_source="b",
        linear_target="a",
        synthetic_goal_prob=0.99,
        overwrite_layer_artifacts=False,
    )
    assert out.exists()
    assert called["copy"] == 1
    assert called["build"] == 1


def test_ensure_layer_manifest_missing_without_template_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _ensure_layer_manifest(
            manifest_path_template=str(tmp_path / "single_manifest.json"),
            layer=9,
            auto_generate_layer_manifests=True,
            pair_manifest_path=tmp_path / "pair_manifest.json",
            base_artifacts_dir=tmp_path / "base_artifacts",
            patch_action_source="b",
            linear_target="a",
            synthetic_goal_prob=0.99,
            overwrite_layer_artifacts=False,
        )
