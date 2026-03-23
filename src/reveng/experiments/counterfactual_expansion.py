"""Expanded layer/threshold analysis for counterfactual evaluation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reveng.experiments.counterfactual_artifact_builder import (
    build_counterfactual_patch_artifacts,
)
from reveng.experiments.counterfactual_activation_patching import (
    _read_manifest,
    counterfactual_activation_patching,
)
from reveng.experiments.counterfactual_manifest_tools import read_pair_manifest_strict


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _threshold_slug(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "p")


def _layer_key(layer: int) -> str:
    return f"model.layers.{layer}.output"


def _load_aggregate(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_manifest_path(template: str, layer: int) -> Path:
    if "{layer}" in template:
        return Path(template.format(layer=layer))
    return Path(template)


def _copy_base_ab_traces_for_pairs(
    pair_manifest_path: Path,
    base_artifacts_dir: Path,
    layer_output_dir: Path,
) -> None:
    pair_specs = read_pair_manifest_strict(pair_manifest_path)
    for spec in pair_specs:
        src_pair_dir = base_artifacts_dir / spec.pair_id
        dst_pair_dir = layer_output_dir / spec.pair_id
        dst_pair_dir.mkdir(parents=True, exist_ok=True)

        for filename in ("A.json", "B.json"):
            src = src_pair_dir / filename
            dst = dst_pair_dir / filename
            if not src.exists():
                raise FileNotFoundError(
                    f"Missing required base trace for auto-generation: {src}. "
                    "Provide --base-artifacts-dir with existing A/B traces."
                )
            if not dst.exists():
                shutil.copy2(src, dst)


def _ensure_layer_manifest(
    manifest_path_template: str,
    layer: int,
    auto_generate_layer_manifests: bool,
    pair_manifest_path: Path,
    base_artifacts_dir: Path,
    patch_action_source: str,
    linear_target: str,
    synthetic_goal_prob: float,
    overwrite_layer_artifacts: bool,
) -> Path:
    manifest_path = _resolve_manifest_path(manifest_path_template, layer)
    if manifest_path.exists():
        return manifest_path

    if not auto_generate_layer_manifests:
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if "{layer}" not in manifest_path_template:
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            "Auto-generation is only supported when --manifest-path includes '{layer}'."
        )

    layer_output_dir = manifest_path.parent
    layer_output_dir.mkdir(parents=True, exist_ok=True)

    _copy_base_ab_traces_for_pairs(
        pair_manifest_path=pair_manifest_path,
        base_artifacts_dir=base_artifacts_dir,
        layer_output_dir=layer_output_dir,
    )

    build_counterfactual_patch_artifacts(
        pair_manifest_path=str(pair_manifest_path),
        output_dir=str(layer_output_dir),
        layer_key=_layer_key(layer),
        patch_action_source=patch_action_source,  # type: ignore[arg-type]
        linear_target=linear_target,  # type: ignore[arg-type]
        synthetic_goal_prob=synthetic_goal_prob,
        skip_trajectory_generation=True,
        overwrite=overwrite_layer_artifacts,
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Auto-generation did not create expected manifest: {manifest_path}"
        )
    return manifest_path


def _sort_config_rows(rows: pd.DataFrame) -> pd.DataFrame:
    sortable = rows.copy()
    sortable["action_true_rate_sort"] = sortable["action_true_rate"].fillna(-1.0)
    sortable["disruptive_rate_sort"] = sortable["disruptive_rate"].fillna(2.0)
    sortable["action_true_count_sort"] = sortable["total_pairs_action_true"].fillna(-1)
    sortable = sortable.sort_values(
        by=["action_true_rate_sort", "action_true_count_sort", "disruptive_rate_sort"],
        ascending=[False, False, True],
    )
    return sortable.drop(
        columns=["action_true_rate_sort", "disruptive_rate_sort", "action_true_count_sort"]
    )


def _pick_best_layer(rows: pd.DataFrame, selection_threshold: float, coarse_only: bool) -> int:
    subset = rows[(rows["status"] == "ok") & (rows["action_threshold"] == selection_threshold)]
    if coarse_only:
        subset = subset[subset["sweep_stage"] == "coarse"]

    if subset.empty:
        raise ValueError("Unable to pick best layer: no successful runs at selection threshold.")

    return int(_sort_config_rows(subset).iloc[0]["layer"])


def _build_layer_plan(
    coarse_layers: list[int],
    best_layer: int,
    refine_radius: int,
) -> list[int]:
    plan = list(dict.fromkeys(coarse_layers))
    refine_layers = [
        layer
        for layer in range(best_layer - refine_radius, best_layer + refine_radius + 1)
        if layer >= 0
    ]
    for layer in refine_layers:
        if layer not in plan:
            plan.append(layer)
    return plan


def _write_heatmap(
    rows: pd.DataFrame,
    metric_col: str,
    title: str,
    output_path: Path,
) -> None:
    ok_rows = rows[rows["status"] == "ok"]
    if ok_rows.empty:
        return

    pivot = ok_rows.pivot_table(
        index="layer",
        columns="action_threshold",
        values=metric_col,
        aggfunc="first",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    arr = pivot.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(x)) for x in pivot.index])
    ax.set_xlabel("Action threshold")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _copy_best_outputs(best_row: pd.Series, selected_dir: Path) -> None:
    selected_dir.mkdir(parents=True, exist_ok=True)
    src = Path(str(best_row["output_dir"]))
    for filename in [
        "aggregate_summary.json",
        "per_pair_results.csv",
        "per_pair_results.jsonl",
        "report.md",
    ]:
        src_path = src / filename
        if src_path.exists():
            shutil.copy2(src_path, selected_dir / filename)

    metadata = {
        "selected_best_layer": int(best_row["layer"]),
        "selected_best_action_threshold": float(best_row["action_threshold"]),
        "selected_best_output_dir": str(src),
    }
    (selected_dir / "best_config.json").write_text(json.dumps(metadata, indent=2))


def _write_dashboard(
    rows: pd.DataFrame,
    output_root: Path,
    selection_threshold: float,
) -> None:
    ok_rows = rows[rows["status"] == "ok"].copy()
    lines: list[str] = [
        "# Counterfactual Expansion Dashboard",
        "",
        "## Configuration",
        f"- selection_threshold_for_best_layer: {selection_threshold:.2f}",
        "- action_thresholds: 0.65, 0.70, 0.75",
        "- disruptive_threshold: 0.35 (fixed)",
        "- early_stop: disabled for comparability",
        "",
    ]

    if ok_rows.empty:
        lines.append("No successful runs were produced.")
        (output_root / "summary.md").write_text("\n".join(lines) + "\n")
        return

    best_rows = _sort_config_rows(
        ok_rows[ok_rows["action_threshold"] == selection_threshold].copy()
    )
    best_row = best_rows.iloc[0]

    lines.extend(
        [
            "## Best Config (by action_true_rate at threshold 0.70)",
            f"- layer: {int(best_row['layer'])}",
            f"- action_threshold: {float(best_row['action_threshold']):.2f}",
            f"- action_true_rate: {_safe_float(best_row['action_true_rate']):.4f}",
            f"- disruptive_rate: {_safe_float(best_row['disruptive_rate']):.4f}",
            f"- output_dir: `{best_row['output_dir']}`",
            "",
            "## Top Configs at Threshold 0.70",
        ]
    )
    top_cols = [
        "layer",
        "action_threshold",
        "action_true_rate",
        "total_pairs_action_true",
        "total_pairs_action_evaluable",
        "disruptive_rate",
        "sweep_stage",
    ]
    lines.append(best_rows[top_cols].head(10).to_markdown(index=False))
    lines.append("")

    # Per-category table from selected best config (if available).
    agg_path = Path(str(best_row["output_dir"])) / "aggregate_summary.json"
    if agg_path.exists():
        agg = _load_aggregate(agg_path)
        per_category = agg.get("per_category", {})
        if per_category:
            cat_rows = []
            for cat, vals in sorted(per_category.items()):
                cat_rows.append(
                    {
                        "category": cat,
                        "action_true_rate": vals.get("action_true_rate"),
                        "disruptive_rate": vals.get("disruptive_rate"),
                        "table_rows_mlp": vals.get("table_rows_mlp"),
                        "table_rows_linear": vals.get("table_rows_linear"),
                    }
                )
            lines.append("## Per-Category (Selected Best Config)")
            lines.append(pd.DataFrame(cat_rows).to_markdown(index=False))
            lines.append("")

    lines.extend(
        [
            "## Figures",
            "- `figs/layer_threshold_action_true_rate.png`",
            "- `figs/layer_threshold_disruptive_rate.png`",
            "",
            "## Machine-Readable Outputs",
            "- `layer_threshold_metrics.csv`",
            "- `per_config_aggregate.jsonl`",
            "- `selected_best/`",
            "",
        ]
    )

    (output_root / "summary.md").write_text("\n".join(lines))


def run_counterfactual_expansion(
    manifest_path: str = "data/cf/artifacts/manifest_for_counterfactual_activation_patching.json",
    output_root: str = "data/cf/eval_results/expansion",
    coarse_layers: tuple[int, ...] = (9, 12, 15, 18, 21),
    refine_radius: int = 2,
    action_thresholds: tuple[float, ...] = (0.65, 0.70, 0.75),
    disruptive_threshold: float = 0.35,
    expected_k: int | None = None,
    selection_threshold: float = 0.70,
    auto_generate_layer_manifests: bool = True,
    pair_manifest_path: str = "data/cf/pair_manifest.json",
    base_artifacts_dir: str = "data/cf/artifacts",
    patch_action_source: str = "b",
    linear_target: str = "a",
    synthetic_goal_prob: float = 0.99,
    overwrite_layer_artifacts: bool = False,
) -> None:
    """Run coarse+refine layer sweep with threshold sensitivity.

    For each layer and action threshold combination this runs the evaluator and
    writes compact, cross-config summaries for easy review.
    """
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    figs_dir = out_root / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    pair_manifest = Path(pair_manifest_path)
    base_artifacts = Path(base_artifacts_dir)

    coarse = list(dict.fromkeys(int(l) for l in coarse_layers))
    action_threshold_values = [float(t) for t in action_thresholds]
    if not coarse:
        raise ValueError("coarse_layers cannot be empty")

    if expected_k is None:
        seed_manifest = _ensure_layer_manifest(
            manifest_path_template=manifest_path,
            layer=coarse[0],
            auto_generate_layer_manifests=auto_generate_layer_manifests,
            pair_manifest_path=pair_manifest,
            base_artifacts_dir=base_artifacts,
            patch_action_source=patch_action_source,
            linear_target=linear_target,
            synthetic_goal_prob=synthetic_goal_prob,
            overwrite_layer_artifacts=overwrite_layer_artifacts,
        )
        expected_k = len(_read_manifest(seed_manifest))

    all_rows: list[dict[str, Any]] = []
    prepared_manifest_by_layer: dict[int, Path] = {}

    def _manifest_for_layer(layer: int) -> Path:
        if layer in prepared_manifest_by_layer:
            return prepared_manifest_by_layer[layer]
        prepared_manifest_by_layer[layer] = _ensure_layer_manifest(
            manifest_path_template=manifest_path,
            layer=layer,
            auto_generate_layer_manifests=auto_generate_layer_manifests,
            pair_manifest_path=pair_manifest,
            base_artifacts_dir=base_artifacts,
            patch_action_source=patch_action_source,
            linear_target=linear_target,
            synthetic_goal_prob=synthetic_goal_prob,
            overwrite_layer_artifacts=overwrite_layer_artifacts,
        )
        return prepared_manifest_by_layer[layer]

    def run_config(layer: int, action_threshold: float, sweep_stage: str) -> None:
        layer_str = f"layer_{layer:02d}"
        thr_slug = _threshold_slug(action_threshold)
        run_dir = out_root / layer_str / f"thr_{thr_slug}"

        manifest_for_layer = None
        row = {
            "layer": layer,
            "layer_key": _layer_key(layer),
            "action_threshold": action_threshold,
            "disruptive_threshold": disruptive_threshold,
            "sweep_stage": sweep_stage,
            "status": "ok",
            "output_dir": str(run_dir),
            "manifest_path": None,
            "error": None,
        }

        try:
            manifest_for_layer = _manifest_for_layer(layer)
            row["manifest_path"] = str(manifest_for_layer)
            counterfactual_activation_patching(
                manifest_path=str(manifest_for_layer),
                output_dir=str(run_dir),
                layer_key=_layer_key(layer),
                expected_k=expected_k,
                action_true_threshold=action_threshold,
                disruptive_threshold=disruptive_threshold,
                enable_early_stop=False,
            )
            aggregate = _load_aggregate(run_dir / "aggregate_summary.json")
            row.update(
                {
                    "action_true_rate": aggregate.get("action_true_rate"),
                    "total_pairs_action_true": aggregate.get("total_pairs_action_true"),
                    "total_pairs_action_evaluable": aggregate.get("total_pairs_action_evaluable"),
                    "disruptive_rate": aggregate.get("disruptive_rate"),
                    "total_pairs_disruptive": aggregate.get("total_pairs_disruptive"),
                    "table_rows_mlp": aggregate.get("table_rows_mlp"),
                    "table_rows_linear": aggregate.get("table_rows_linear"),
                }
            )
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            row.update(
                {
                    "action_true_rate": None,
                    "total_pairs_action_true": None,
                    "total_pairs_action_evaluable": None,
                    "disruptive_rate": None,
                    "total_pairs_disruptive": None,
                    "table_rows_mlp": None,
                    "table_rows_linear": None,
                }
            )

        all_rows.append(row)

    # Coarse sweep first.
    for layer in coarse:
        for action_threshold in action_threshold_values:
            run_config(layer=layer, action_threshold=action_threshold, sweep_stage="coarse")

    coarse_df = pd.DataFrame(all_rows)
    best_layer = _pick_best_layer(coarse_df, selection_threshold=selection_threshold, coarse_only=True)
    final_layers = _build_layer_plan(coarse, best_layer=best_layer, refine_radius=refine_radius)

    for layer in final_layers:
        if layer in coarse:
            continue
        for action_threshold in action_threshold_values:
            run_config(layer=layer, action_threshold=action_threshold, sweep_stage="refine")

    results_df = pd.DataFrame(all_rows)
    results_df = results_df.sort_values(by=["layer", "action_threshold"]).reset_index(drop=True)

    # Write machine-readable outputs.
    results_df.to_csv(out_root / "layer_threshold_metrics.csv", index=False)
    results_df.to_json(
        out_root / "per_config_aggregate.jsonl",
        orient="records",
        lines=True,
    )

    # Select and copy best config outputs.
    ok_at_selection = results_df[
        (results_df["status"] == "ok") & (results_df["action_threshold"] == selection_threshold)
    ]
    if not ok_at_selection.empty:
        best_row = _sort_config_rows(ok_at_selection).iloc[0]
        _copy_best_outputs(best_row, selected_dir=out_root / "selected_best")

    # Figures and dashboard.
    _write_heatmap(
        results_df,
        metric_col="action_true_rate",
        title="Action True Rate by Layer and Threshold",
        output_path=figs_dir / "layer_threshold_action_true_rate.png",
    )
    _write_heatmap(
        results_df,
        metric_col="disruptive_rate",
        title="Disruptive Rate by Layer and Threshold",
        output_path=figs_dir / "layer_threshold_disruptive_rate.png",
    )
    _write_dashboard(results_df, output_root=out_root, selection_threshold=selection_threshold)

    print(f"Wrote expansion outputs to: {out_root}")


__all__ = [
    "run_counterfactual_expansion",
]
