"""Merge behavioral-probe output directories into one combined report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reveng.experiments.behavioral_probe_metrics import summarize_probe_rows
from reveng.experiments.behavioral_probe_plots import (
    plot_behavioral_probe_summary,
    plot_coordinate_probe_summary,
    plot_directional_probe_heatmap,
)
from reveng.experiments.behavioral_probe_runner import _build_probability_diagnostics_rows, _write_csv


def merge_behavioral_probe_outputs(
    *,
    input_dirs: tuple[str, ...],
    output_dir: str,
    repair_source_rows_csv: bool = False,
) -> None:
    """Merge multiple behavioral-probe output dirs using their raw payloads as source of truth."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)

    merged_raw_rows: list[dict[str, Any]] = []
    merged_rows: list[dict[str, Any]] = []
    merged_usage: dict[str, Any] = {
        "source_dirs": list(input_dirs),
        "per_source": {},
    }
    plot_meta: dict[str, Any] = {}

    for input_dir in input_dirs:
        base = Path(input_dir)
        raw_path = base / "behavioral_probe_raw.json"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing behavioral_probe_raw.json in {base}")
        raw_payload = json.loads(raw_path.read_text())
        source_config = raw_payload.get("config", {})
        source_raw_rows = raw_payload.get("rows", [])
        source_rows = [item["parsed_row"] for item in source_raw_rows]
        merged_raw_rows.extend(source_raw_rows)
        merged_rows.extend(source_rows)
        merged_usage["per_source"][str(base)] = raw_payload.get("usage_summary")
        for key in ("mc_sample_repeats", "mc_temperature"):
            if key in source_config and key not in plot_meta:
                plot_meta[key] = source_config[key]

        if repair_source_rows_csv:
            _write_csv(base / "behavioral_probe_rows.csv", source_rows)

    merged_summary = summarize_probe_rows(merged_rows)
    merged_probability_diagnostics = _build_probability_diagnostics_rows(merged_rows)
    merged_payload = {
        "config": {
            "merge_source_dirs": list(input_dirs),
            "repair_source_rows_csv": repair_source_rows_csv,
            **plot_meta,
        },
        "rows": merged_raw_rows,
        "summary": merged_summary,
        "probability_diagnostics": merged_probability_diagnostics,
        "usage_summary": merged_usage,
    }

    _write_csv(out_dir / "behavioral_probe_rows.csv", merged_rows)
    _write_csv(out_dir / "behavioral_probe_summary.csv", merged_summary)
    _write_csv(out_dir / "behavioral_probe_probability_diagnostics.csv", merged_probability_diagnostics)
    (out_dir / "behavioral_probe_raw.json").write_text(json.dumps(merged_payload, indent=2))
    (out_dir / "usage_summary.json").write_text(json.dumps(merged_usage, indent=2))

    label_rows = [row for row in merged_summary if row.get("answer_space") == "label3"]
    coord_rows = [row for row in merged_summary if row.get("answer_space") == "coord_json"]
    if label_rows:
        plot_behavioral_probe_summary(
            label_rows,
            out_dir / "figs" / "behavioral_probe_summary.png",
            mc_sample_repeats=plot_meta.get("mc_sample_repeats"),
            mc_temperature=plot_meta.get("mc_temperature"),
        )
        try:
            plot_directional_probe_heatmap(
                label_rows,
                out_dir / "figs" / "behavioral_probe_directional_heatmap.png",
                mc_sample_repeats=plot_meta.get("mc_sample_repeats"),
                mc_temperature=plot_meta.get("mc_temperature"),
            )
        except ValueError:
            pass
    if coord_rows:
        plot_coordinate_probe_summary(coord_rows, out_dir / "figs" / "behavioral_probe_coordinate_summary.png")


__all__ = ["merge_behavioral_probe_outputs"]
