from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from reveng.experiments.behavioral_probe_parse import parse_behavioral_probe_answer

DEFAULT_PRE_ALIGNMENT_PATH = "data/behavioral_probes/released_cognitive_alignment_repeat10_pre/probe_alignment_rows.csv"
DEFAULT_POST_ALIGNMENT_PATH = "data/behavioral_probes/released_cognitive_alignment_repeat10_post/probe_alignment_rows.csv"
DEFAULT_PRE_BEHAVIORAL_ROWS_PATH = "data/behavioral_probes/released_cognitive_alignment_repeat10_pre/behavioral_probe_rows.csv"
DEFAULT_POST_BEHAVIORAL_ROWS_PATH = "data/behavioral_probes/released_cognitive_alignment_repeat10_post/behavioral_probe_rows.csv"
DEFAULT_FAILURE_PAPER_SLICE_PATH = "data/behavioral_probes/gradual_cot_alignment_failure_wall_core_light/gradual_cot_alignment_rows.csv"
DEFAULT_OUTPUT_DIR = "data/behavioral_probes/action_conditioned_belief_comparison"

METHOD_SPECS = (
    ("whitebox_released_probe", "White-box released probe", "whitebox"),
    ("blackbox_repeated_greedy_modal", "Black-box repeated greedy modal", "blackbox"),
    ("blackbox_mc_yes_no", "Black-box MC yes/no", "blackbox"),
    ("blackbox_logprob_yes_no_t07", "Black-box logprob yes/no (T=0.7)", "blackbox"),
)

ACTION_TO_QUESTION = {
    "LEFT": "wall_left",
    "RIGHT": "wall_right",
    "UP": "wall_up",
    "DOWN": "wall_down",
}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _normalize_label(value: Any) -> str:
    return parse_behavioral_probe_answer(None if value is None else str(value))


def _consistency_from_label(label: str) -> str:
    if label == "yes":
        return "inconsistent"
    if label == "no":
        return "potentially_consistent"
    return "unknown_based"


def _state_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("example_id", "")),
        str(row.get("step_index", "")),
        str(row.get("reasoning_split", "")),
    )


def _load_behavioral_t07_index(path: str | Path) -> dict[tuple[str, str, str, str], str]:
    index: dict[tuple[str, str, str, str], str] = {}
    for row in _read_csv(path):
        key = (
            str(row.get("example_id", "")),
            str(row.get("step_index", "")),
            str(row.get("reasoning_split", "")),
            str(row.get("question_id", "")),
        )
        index[key] = _normalize_label(row.get("logprob_yes_no_answer_t07", ""))
    return index


def _build_alignment_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], dict[str, str]]:
    index: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            str(row.get("example_id", "")),
            str(row.get("step_index", "")),
            str(row.get("reasoning_split", "")),
            str(row.get("question_id", "")),
        )
        index[key] = row
    return index


def _compare_pre_post_identity_sets(
    pre_rows: list[dict[str, str]],
    post_rows: list[dict[str, str]],
) -> None:
    def identity_set(rows: list[dict[str, str]]) -> set[tuple[str, str, str]]:
        return {
            (
                str(row.get("example_id", "")),
                str(row.get("step_index", "")),
                str(row.get("question_id", "")),
            )
            for row in rows
        }

    if identity_set(pre_rows) != identity_set(post_rows):
        raise ValueError(
            "Pre/post clean repeat-10 alignment rows do not align on the same (example_id, step_index, question_id) identities."
        )


def _build_acted_direction_rows(
    pre_rows: list[dict[str, str]],
    post_rows: list[dict[str, str]],
    pre_t07: dict[tuple[str, str, str, str], str],
    post_t07: dict[tuple[str, str, str, str], str],
) -> list[dict[str, Any]]:
    all_rows = pre_rows + post_rows
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        grouped[_state_key(row)].append(row)

    acted_rows: list[dict[str, Any]] = []
    for state_key, state_rows in sorted(grouped.items()):
        example_id, step_index, reasoning_split = state_key
        if len(state_rows) != 4:
            raise ValueError(f"Expected 4 wall rows for {state_key}, found {len(state_rows)}")

        observed_action = str(state_rows[0].get("observed_action", "")).upper()
        if observed_action not in ACTION_TO_QUESTION:
            raise ValueError(f"Unsupported observed action {observed_action!r} for {state_key}")
        acted_question_id = ACTION_TO_QUESTION[observed_action]

        by_question = {str(row["question_id"]): row for row in state_rows}
        acted_row = by_question.get(acted_question_id)
        if acted_row is None:
            raise ValueError(f"Missing acted-direction row {acted_question_id} for {state_key}")

        logprob_index = pre_t07 if reasoning_split == "pre" else post_t07
        logprob_key = (example_id, step_index, reasoning_split, acted_question_id)
        logprob_t07 = logprob_index.get(logprob_key, "invalid")

        ground_truth = _normalize_label(acted_row.get("ground_truth_label", ""))
        is_optimal_action = _normalize_bool(acted_row.get("is_optimal_action", ""))

        method_labels = {
            "whitebox_released_probe": _normalize_label(acted_row.get("whitebox_prediction", "")),
            "blackbox_repeated_greedy_modal": _normalize_label(acted_row.get("blackbox_greedy_modal_answer", "")),
            "blackbox_mc_yes_no": _normalize_label(acted_row.get("blackbox_mc_yes_no_answer", "")),
            "blackbox_logprob_yes_no_t07": logprob_t07,
        }

        for method_key, method_label, method_family in METHOD_SPECS:
            acted_belief = method_labels[method_key]
            acted_rows.append(
                {
                    "example_id": example_id,
                    "step_index": step_index,
                    "reasoning_split": reasoning_split,
                    "observed_action": observed_action,
                    "acted_question_id": acted_question_id,
                    "ground_truth_acted_direction": ground_truth,
                    "is_optimal_action": is_optimal_action,
                    "wall_hit": _normalize_bool(acted_row.get("wall_hit", "")),
                    "method_key": method_key,
                    "method_label": method_label,
                    "method_family": method_family,
                    "acted_direction_belief": acted_belief,
                    "acted_direction_belief_correct": acted_belief == ground_truth if acted_belief in {"yes", "no", "unknown"} else False,
                    "action_consistency_label": _consistency_from_label(acted_belief),
                }
            )
    return acted_rows


def _format_pct(num: int, den: int) -> str:
    if den == 0:
        return "n/a"
    return f"{num}/{den} ({100.0 * num / den:.1f}%)"


def _summarize_accuracy(
    rows: list[dict[str, Any]],
    *,
    subset_name: str,
    reasoning_split: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method_key, method_label, _method_family in METHOD_SPECS:
        method_rows = [row for row in rows if row["method_key"] == method_key]
        den = len(method_rows)
        correct = sum(bool(row["acted_direction_belief_correct"]) for row in method_rows)
        unknown = sum(row["acted_direction_belief"] == "unknown" for row in method_rows)
        out.append(
            {
                "table_name": "directional_belief_accuracy",
                "subset_name": subset_name,
                "reasoning_split": reasoning_split,
                "method_key": method_key,
                "method_label": method_label,
                "n": den,
                "accuracy": _format_pct(correct, den),
                "fraction_unknown": _format_pct(unknown, den),
            }
        )
    return out


def _summarize_consistency(
    rows: list[dict[str, Any]],
    *,
    subset_name: str,
    reasoning_split: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method_key, method_label, _method_family in METHOD_SPECS:
        method_rows = [row for row in rows if row["method_key"] == method_key]
        den = len(method_rows)
        inconsistent = sum(row["action_consistency_label"] == "inconsistent" for row in method_rows)
        potentially_consistent = sum(row["action_consistency_label"] == "potentially_consistent" for row in method_rows)
        unknown_based = sum(row["action_consistency_label"] == "unknown_based" for row in method_rows)
        out.append(
            {
                "table_name": "consistency_profile",
                "subset_name": subset_name,
                "reasoning_split": reasoning_split,
                "method_key": method_key,
                "method_label": method_label,
                "n": den,
                "inconsistent": _format_pct(inconsistent, den),
                "potentially_consistent": _format_pct(potentially_consistent, den),
                "unknown_based": _format_pct(unknown_based, den),
            }
        )
    return out


def _build_summary_rows(acted_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    split_options = ["all", "pre", "post"]

    for split_name in split_options:
        split_rows = acted_rows if split_name == "all" else [row for row in acted_rows if row["reasoning_split"] == split_name]
        subsets = [("all_clean_rows", split_rows)]

        non_opt_rows = [row for row in split_rows if row["is_optimal_action"] is False]
        if non_opt_rows:
            subsets.append(("non_optimal_action_rows", non_opt_rows))
        else:
            for table_name in ("directional_belief_accuracy", "consistency_profile"):
                for method_key, method_label, _method_family in METHOD_SPECS:
                    if table_name == "directional_belief_accuracy":
                        summary_rows.append(
                            {
                                "table_name": table_name,
                                "subset_name": "non_optimal_action_rows",
                                "reasoning_split": split_name,
                                "method_key": method_key,
                                "method_label": method_label,
                                "n": 0,
                                "accuracy": "n/a",
                                "fraction_unknown": "n/a",
                                "note": "No non-optimal action rows in the released clean repeat-10 slice.",
                            }
                        )
                    else:
                        summary_rows.append(
                            {
                                "table_name": table_name,
                                "subset_name": "non_optimal_action_rows",
                                "reasoning_split": split_name,
                                "method_key": method_key,
                                "method_label": method_label,
                                "n": 0,
                                "inconsistent": "n/a",
                                "potentially_consistent": "n/a",
                                "unknown_based": "n/a",
                                "note": "No non-optimal action rows in the released clean repeat-10 slice.",
                            }
                        )

        for subset_name, subset_rows in subsets:
            summary_rows.extend(_summarize_accuracy(subset_rows, subset_name=subset_name, reasoning_split=split_name))
            summary_rows.extend(_summarize_consistency(subset_rows, subset_name=subset_name, reasoning_split=split_name))

    return summary_rows


def _failure_whitebox_note(failure_paper_slice_path: str | Path) -> str:
    rows = _read_csv(failure_paper_slice_path)
    nonempty = sum(
        bool((row.get("whitebox_prediction_pre", "") or "").strip()) or bool((row.get("whitebox_prediction_post", "") or "").strip())
        for row in rows
    )
    if nonempty != 0:
        return (
            "Warning: the configured failure slice now contains non-empty white-box predictions; "
            "revisit the clean-only assumption before using this summary."
        )
    return (
        "Failure-slice white-box comparison not included: the exact paper failure slice currently has zero non-empty "
        "`whitebox_prediction_pre/post` values, so a comparable white-box analysis requires backfilling those predictions first."
    )


def _build_markdown(summary_rows: list[dict[str, Any]], failure_note: str) -> str:
    lines = [
        "# Action-Conditioned White-Box vs Black-Box Belief Comparison",
        "",
        "This summary uses the released clean repeat-10 slice only.",
        "",
        "Methods",
        "- White-box released probe",
        "- Black-box repeated greedy modal",
        "- Black-box MC yes/no",
        "- Black-box logprob yes/no (T=0.7)",
        "",
        "Interpretation",
        "- `Directional belief accuracy`: on the direction actually taken by the agent, how often does the readout recover the correct local wall belief?",
        "- `Consistency profile`: on the direction actually taken by the agent, does the readout make that action look locally blocked (`inconsistent`), locally free (`potentially_consistent`), or undecided (`unknown_based`)?",
        "",
        f"Failure slice note: {failure_note}",
        "",
    ]

    for split_name in ("all", "pre", "post"):
        lines.extend(
            [
                f"## Split: {split_name}",
                "",
                "### Directional Belief Accuracy",
                "",
                "| Subset | Method | n | Accuracy | Fraction unknown |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in summary_rows:
            if row["table_name"] != "directional_belief_accuracy" or row["reasoning_split"] != split_name:
                continue
            lines.append(
                f"| {row['subset_name']} | {row['method_label']} | {row['n']} | {row['accuracy']} | {row['fraction_unknown']} |"
            )
        lines.extend(
            [
                "",
                "### Consistency Profile",
                "",
                "| Subset | Method | n | Inconsistent | Potentially consistent | Unknown based |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in summary_rows:
            if row["table_name"] != "consistency_profile" or row["reasoning_split"] != split_name:
                continue
            lines.append(
                f"| {row['subset_name']} | {row['method_label']} | {row['n']} | {row['inconsistent']} | {row['potentially_consistent']} | {row['unknown_based']} |"
            )
        lines.append("")
    return "\n".join(lines)


def build_action_conditioned_belief_comparison(
    pre_alignment_path: str = DEFAULT_PRE_ALIGNMENT_PATH,
    post_alignment_path: str = DEFAULT_POST_ALIGNMENT_PATH,
    pre_behavioral_rows_path: str = DEFAULT_PRE_BEHAVIORAL_ROWS_PATH,
    post_behavioral_rows_path: str = DEFAULT_POST_BEHAVIORAL_ROWS_PATH,
    failure_paper_slice_path: str = DEFAULT_FAILURE_PAPER_SLICE_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    pre_alignment_rows = _read_csv(pre_alignment_path)
    post_alignment_rows = _read_csv(post_alignment_path)
    pre_t07_index = _load_behavioral_t07_index(pre_behavioral_rows_path)
    post_t07_index = _load_behavioral_t07_index(post_behavioral_rows_path)

    _compare_pre_post_identity_sets(pre_alignment_rows, post_alignment_rows)

    acted_rows = _build_acted_direction_rows(pre_alignment_rows, post_alignment_rows, pre_t07_index, post_t07_index)
    summary_rows = _build_summary_rows(acted_rows)
    failure_note = _failure_whitebox_note(failure_paper_slice_path)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    acted_rows_path = output_root / "acted_direction_rows.csv"
    summary_path = output_root / "summary_table.csv"
    markdown_path = output_root / "summary_table.md"
    manifest_path = output_root / "manifest.json"

    _write_csv(acted_rows_path, acted_rows)
    _write_csv(summary_path, summary_rows)
    markdown_path.write_text(_build_markdown(summary_rows, failure_note))

    manifest = {
        "pre_alignment_path": pre_alignment_path,
        "post_alignment_path": post_alignment_path,
        "pre_behavioral_rows_path": pre_behavioral_rows_path,
        "post_behavioral_rows_path": post_behavioral_rows_path,
        "failure_paper_slice_path": failure_paper_slice_path,
        "n_pre_alignment_rows": len(pre_alignment_rows),
        "n_post_alignment_rows": len(post_alignment_rows),
        "n_acted_direction_rows": len(acted_rows),
        "failure_slice_note": failure_note,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return manifest


__all__ = ["build_action_conditioned_belief_comparison"]
