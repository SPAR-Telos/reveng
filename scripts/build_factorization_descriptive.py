"""Build descriptive factorization tables for Experiment 2 behavioral outputs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

DIRECTION_BY_ACTION = {
    "LEFT": "left",
    "RIGHT": "right",
    "UP": "up",
    "DOWN": "down",
}

FEATURES = {
    "blocked_report_and_predicted_hit": (
        "Chosen-direction wall report is yes AND chosen-action hit-wall report is yes."
    ),
    "chosen_wall_error_and_chosen_effect_error": (
        "Chosen-direction wall belief is wrong AND chosen-action hit-wall consequence belief is wrong."
    ),
    "key_error_and_door_error": "Key-possession belief is wrong AND door-open belief is wrong.",
    "chosen_action_conflicts_with_reported_state": (
        "Chosen-direction wall report is yes OR chosen-action hit-wall report is yes."
    ),
}

OUTCOMES = {
    "action_change_next": "Recommended action changes at the next analysis position.",
    "optimality_loss_next": "Planner-optimal recommendation becomes suboptimal at the next analysis position.",
    "optimality_recovery_next": "Suboptimal recommendation becomes planner-optimal at the next analysis position.",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def belief_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, int, str], dict[str, str]]:
    return {
        (row["example_id"], int(row["reasoning_step_idx"]), row["question_id"]): row
        for row in rows
    }


def valid(row: dict[str, str] | None) -> bool:
    return row is not None and as_bool(row.get("answer_valid")) is True


def is_error(row: dict[str, str]) -> bool:
    return as_bool(row.get("belief_is_error")) is True


def answer_yes(row: dict[str, str]) -> bool:
    return str(row.get("answer_key", "")).strip().lower() == "yes"


def safe_rate(num: int, den: int) -> float | str:
    return num / den if den else ""


def odds_ratio(a: int, b: int, c: int, d: int) -> float:
    # Haldane-Anscombe smoothing keeps finite descriptive odds ratios in tiny pilots.
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


def build_rows(run_dir: Path) -> list[dict[str, Any]]:
    positions = read_csv(run_dir / "position_rows.csv")
    beliefs = belief_lookup(read_csv(run_dir / "belief_rows.csv"))
    by_example: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in positions:
        by_example[row["example_id"]].append(row)
    rows: list[dict[str, Any]] = []
    for example_id, example_rows in by_example.items():
        ordered = sorted(example_rows, key=lambda row: int(row["reasoning_step_idx"]))
        for current, following in zip(ordered, ordered[1:]):
            direction = DIRECTION_BY_ACTION.get(str(current["action_label"]).upper())
            if direction is None:
                continue
            step_idx = int(current["reasoning_step_idx"])
            chosen_wall = beliefs.get((example_id, step_idx, f"wall_{direction}"))
            chosen_effect = beliefs.get((example_id, step_idx, f"hit_wall_after_{direction}"))
            has_key = beliefs.get((example_id, step_idx, "has_key"))
            door_open = beliefs.get((example_id, step_idx, "door_open"))
            if not all(valid(row) for row in (chosen_wall, chosen_effect, has_key, door_open)):
                continue
            current_optimal = as_bool(current.get("action_is_optimal"))
            next_optimal = as_bool(following.get("action_is_optimal"))
            if current_optimal is None or next_optimal is None:
                continue
            assert chosen_wall is not None and chosen_effect is not None
            assert has_key is not None and door_open is not None
            rows.append(
                {
                    "example_id": example_id,
                    "trajectory_id": current["trajectory_id"],
                    "reasoning_step_idx": step_idx,
                    "reasoning_progress": current["reasoning_progress"],
                    "reasoning_character_progress": current.get(
                        "reasoning_character_progress", ""
                    ),
                    "analysis_unit": current.get("analysis_unit", "packed_chunk"),
                    "action_label": current["action_label"],
                    "next_action_label": following["action_label"],
                    "current_action_is_optimal": current_optimal,
                    "next_action_is_optimal": next_optimal,
                    "action_change_next": current["action_label"] != following["action_label"],
                    "optimality_loss_next": current_optimal is True and next_optimal is False,
                    "optimality_recovery_next": current_optimal is False and next_optimal is True,
                    "chosen_wall_reports_blocked": answer_yes(chosen_wall),
                    "chosen_effect_reports_hit": answer_yes(chosen_effect),
                    "chosen_wall_error": is_error(chosen_wall),
                    "chosen_effect_error": is_error(chosen_effect),
                    "has_key_error": is_error(has_key),
                    "door_open_error": is_error(door_open),
                    "blocked_report_and_predicted_hit": answer_yes(chosen_wall) and answer_yes(chosen_effect),
                    "chosen_wall_error_and_chosen_effect_error": is_error(chosen_wall) and is_error(chosen_effect),
                    "key_error_and_door_error": is_error(has_key) and is_error(door_open),
                    "chosen_action_conflicts_with_reported_state": answer_yes(chosen_wall) or answer_yes(chosen_effect),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for outcome, outcome_desc in OUTCOMES.items():
        for feature, feature_desc in FEATURES.items():
            with_feature = [row for row in rows if row[feature] is True]
            without_feature = [row for row in rows if row[feature] is False]
            a = sum(row[outcome] is True for row in with_feature)
            b = len(with_feature) - a
            c = sum(row[outcome] is True for row in without_feature)
            d = len(without_feature) - c
            rate_with = safe_rate(a, len(with_feature))
            rate_without = safe_rate(c, len(without_feature))
            summary.append(
                {
                    "outcome": outcome,
                    "outcome_description": outcome_desc,
                    "factorized_feature": feature,
                    "feature_description": feature_desc,
                    "n_with_feature": len(with_feature),
                    "n_without_feature": len(without_feature),
                    "events_with_feature": a,
                    "events_without_feature": c,
                    "event_rate_with_feature": rate_with,
                    "event_rate_without_feature": rate_without,
                    "risk_difference": (
                        rate_with - rate_without
                        if isinstance(rate_with, float) and isinstance(rate_without, float)
                        else ""
                    ),
                    "smoothed_odds_ratio": odds_ratio(a, b, c, d),
                }
            )
    return summary


def write_report(run_dir: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Descriptive Factorized Belief-Action Analysis",
        "",
        "This is a pilot-only descriptive analysis over complete-case prefix positions. It is not a held-out regression and not causal evidence.",
        "",
        f"Complete-case positions: {len(rows)}",
        "",
        "## Features",
        "",
        "| Feature | Definition |",
        "|---|---|",
    ]
    for feature, desc in FEATURES.items():
        lines.append(f"| `{feature}` | {desc} |")
    lines.extend(["", "## Largest Descriptive Risk Differences", "", "| Outcome | Feature | n with feature | Rate with | Rate without | Difference |", "|---|---|---:|---:|---:|---:|"])
    ranked = sorted(
        [row for row in summary if isinstance(row["risk_difference"], float)],
        key=lambda row: abs(float(row["risk_difference"])),
        reverse=True,
    )[:10]
    for row in ranked:
        lines.append(
            f"| {row['outcome']} | `{row['factorized_feature']}` | {row['n_with_feature']} | "
            f"{float(row['event_rate_with_feature']):.3f} | {float(row['event_rate_without_feature']):.3f} | "
            f"{float(row['risk_difference']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Because this run contains 8 states from one trajectory, these rows should be read as sanity checks for the factorization design. The scaled run should compare additive, action-conditioned, and factorized models with trajectory-held-out validation.",
        ]
    )
    (run_dir / "FACTORIZATION_DESCRIPTIVE_SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="outputs/experiment2_behavioral_beliefs/weisheng_8_state_v1")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    rows = build_rows(run_dir)
    summary = summarize(rows)
    write_csv(run_dir / "factorization_position_rows.csv", rows)
    write_csv(run_dir / "factorization_descriptive_summary.csv", summary)
    write_report(run_dir, rows, summary)


if __name__ == "__main__":
    main()
