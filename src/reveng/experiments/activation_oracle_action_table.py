from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _MetricDef:
    key: str
    question_id: str
    match_field: str
    label: str


_METRICS = (
    _MetricDef(
        key="ask_next_action_vs_model_action",
        question_id="model_action",
        match_field="full_sequence_matches_behavioral_label",
        label="Ask next action, compare with model's chosen action",
    ),
    _MetricDef(
        key="ask_next_action_vs_optimal_action",
        question_id="model_action",
        match_field="full_sequence_matches_optimal_action_set",
        label="Ask next action, compare with optimal action set",
    ),
    _MetricDef(
        key="ask_optimal_action_vs_optimal_action",
        question_id="optimal_action",
        match_field="full_sequence_matches_optimal_action_set",
        label="Ask optimal action, compare with optimal action set",
    ),
)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _to_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _format_cell(num: int, den: int) -> str:
    if den == 0:
        return "n/a"
    return f"{num}/{den} ({100.0 * num / den:.1f}%)"


def _aggregate_rows(rows: list[dict[str, str]], slice_name: str) -> list[dict[str, str]]:
    reveal_levels = sorted({int(row["reasoning_reveal_pct"]) for row in rows if row.get("reasoning_reveal_pct", "").strip()})
    output_rows: list[dict[str, str]] = []

    for reveal_pct in reveal_levels + ["Overall"]:
        row_out: dict[str, str] = {
            "slice": slice_name,
            "revealed_reasoning_fraction_pct": str(reveal_pct),
        }
        filtered = rows if reveal_pct == "Overall" else [row for row in rows if int(row["reasoning_reveal_pct"]) == reveal_pct]

        n_state_reveal_pairs = len(
            {
                (row["example_id"], row["step_index"], row["reasoning_reveal_pct"])
                for row in filtered
                if row.get("question_id") == "model_action"
            }
        )
        row_out["n_state_reveal_pairs"] = str(n_state_reveal_pairs)

        for metric in _METRICS:
            metric_rows = [row for row in filtered if row.get("question_id") == metric.question_id]
            den = len(metric_rows)
            num = sum(_to_bool(row.get(metric.match_field, "")) for row in metric_rows)
            row_out[metric.key] = _format_cell(num, den)

        output_rows.append(row_out)

    return output_rows


def _write_csv(path: str | Path, rows: list[dict[str, str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "slice",
        "revealed_reasoning_fraction_pct",
        "n_state_reveal_pairs",
        *[metric.key for metric in _METRICS],
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: str | Path, rows: list[dict[str, str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Slice",
        "Revealed reasoning fraction",
        "State × reveal pairs",
        *[metric.label for metric in _METRICS],
    ]
    lines = [
        "# Activation-Oracle Action Comparison Summary",
        "",
        "This table aggregates only the action-oriented activation-oracle prompts.",
        "",
        "Setup",
        "- Base model: `Qwen/Qwen3-8B`",
        "- Activation-oracle adapter: `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B`",
        "- `clean public slice`: the matched non-failure slice used for the clean revealed-CoT analysis in the paper.",
        "- `failure-focused revealed-CoT slice`: the failure slice used for the revealed-CoT analysis in the paper.",
        "",
        "- `Ask next action, compare with model's chosen action`: the oracle is asked what action the model would take, and that answer is checked against the model's actual emitted action.",
        "- `Ask next action, compare with optimal action set`: the same next-action readout is instead checked against the DoorKey-aware optimal action set for that state.",
        "- `Ask optimal action, compare with optimal action set`: the oracle is asked what action is shortest-path optimal, and that answer is checked against the optimal action set.",
        "- `State × reveal pairs`: the number of distinct `(state, revealed reasoning fraction)` cases contributing to that row.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["slice"],
                    f'{row["revealed_reasoning_fraction_pct"]}%' if row["revealed_reasoning_fraction_pct"] != "Overall" else "Overall",
                    row["n_state_reveal_pairs"],
                    *[row[metric.key] for metric in _METRICS],
                ]
            )
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def build_activation_oracle_action_table(
    clean_comparison_csv: str = "data/activation_oracle/paper_clean_revealed_alignment_qwen_run_v2/ao_behavioral_comparison.csv",
    failure_comparison_csv: str = "data/activation_oracle/paper_failure_revealed_alignment_qwen_run_v2/ao_behavioral_comparison.csv",
    output_csv: str = "data/activation_oracle/ao_action_summary_table.csv",
    output_markdown: str = "data/activation_oracle/ao_action_summary_table.md",
) -> list[dict[str, str]]:
    clean_rows = _read_csv(clean_comparison_csv)
    failure_rows = _read_csv(failure_comparison_csv)

    summary_rows = _aggregate_rows(clean_rows, "clean public slice")
    summary_rows.extend(_aggregate_rows(failure_rows, "failure-focused revealed-CoT slice"))

    _write_csv(output_csv, summary_rows)
    _write_markdown(output_markdown, summary_rows)
    return summary_rows


__all__ = ["build_activation_oracle_action_table"]
