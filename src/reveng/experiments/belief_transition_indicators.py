"""Descriptive belief indicators and candidate chains around action transitions."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_RUN_DIR = "data/behavioral_probes/reasoning_belief_action_balanced_v1"
LOSS_EVENTS = {"sustained_optimal_to_suboptimal", "transient_optimal_to_suboptimal"}
RECOVERY_EVENTS = {"suboptimal_to_optimal", "suboptimal_to_optimal_recovery"}
PRIMARY_BELIEFS = {"wall_left", "wall_right", "wall_up", "wall_down", "has_key", "door_open"}
ACTION_ORDER = ("UP", "DOWN", "LEFT", "RIGHT")
ACTION_EVENT_LABELS = {
    "action_identity_change_without_optimality_change": "Action change only",
    "suboptimal_to_optimal_recovery": "Recovery",
    "sustained_optimal_to_suboptimal": "Sustained loss",
    "transient_optimal_to_suboptimal": "Transient loss",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _event_family(event_type: str) -> str:
    if event_type in LOSS_EVENTS:
        return "optimal_to_suboptimal"
    if event_type in RECOVERY_EVENTS:
        return "suboptimal_to_optimal"
    return "other"


def _target_shift_type(event_family: str) -> str:
    return "belief_error_onset" if event_family == "optimal_to_suboptimal" else "belief_error_recovery"


def _plot_indicator_heatmap(rows: list[dict[str, Any]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    beliefs = sorted({str(row["question_id"]) for row in rows})
    columns = [
        ("optimal_to_suboptimal", "leading_event_coverage"),
        ("optimal_to_suboptimal", "coincident_event_coverage"),
        ("optimal_to_suboptimal", "lagging_event_coverage"),
        ("suboptimal_to_optimal", "leading_event_coverage"),
        ("suboptimal_to_optimal", "coincident_event_coverage"),
        ("suboptimal_to_optimal", "lagging_event_coverage"),
    ]
    lookup = {(row["event_family"], row["question_id"]): row for row in rows}
    matrix = np.array(
        [
            [float(lookup.get((family, belief), {}).get(metric, 0.0)) for family, metric in columns]
            for belief in beliefs
        ]
    )
    fig, ax = plt.subplots(figsize=(9.0, max(4.5, len(beliefs) * 0.34)))
    image = ax.imshow(matrix, vmin=0.0, vmax=max(0.5, float(matrix.max())), cmap="Blues", aspect="auto")
    ax.set_yticks(range(len(beliefs)), labels=[belief.replace("_", " ") for belief in beliefs])
    ax.set_xticks(
        range(len(columns)),
        labels=[
            "belief leads loss",
            "same prefix as loss",
            "belief lags loss",
            "belief leads recovery",
            "same prefix as recovery",
            "belief lags recovery",
        ],
        rotation=25,
        ha="right",
    )
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            ax.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Belief Changes Near Action-Optimality Transitions")
    fig.colorbar(image, ax=ax, label="Fraction of action events")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _action_transition_rows(events: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    totals: Counter[str] = Counter()
    for event in events:
        event_type = event["event_type"]
        previous_action = str(event.get("previous_action", "")).upper()
        current_action = str(event.get("current_action", "")).upper()
        if previous_action not in ACTION_ORDER or current_action not in ACTION_ORDER:
            continue
        counts[(event_type, previous_action, current_action)] += 1
        totals[event_type] += 1
    rows: list[dict[str, Any]] = []
    for event_type in sorted(totals):
        for previous_action in ACTION_ORDER:
            for current_action in ACTION_ORDER:
                count = counts[(event_type, previous_action, current_action)]
                rows.append(
                    {
                        "event_type": event_type,
                        "previous_action": previous_action,
                        "current_action": current_action,
                        "n_transitions": count,
                        "fraction_within_event_type": count / totals[event_type] if totals[event_type] else 0.0,
                    }
                )
    return rows


def _plot_action_transition_heatmap(rows: list[dict[str, Any]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    event_types = sorted({str(row["event_type"]) for row in rows})
    if not event_types:
        return
    fig, axes = plt.subplots(
        1,
        len(event_types),
        figsize=(max(4.0, 3.2 * len(event_types)), 3.6),
        squeeze=False,
        constrained_layout=True,
    )
    max_count = max(int(row["n_transitions"]) for row in rows) if rows else 1
    for ax, event_type in zip(axes.ravel(), event_types):
        matrix = np.zeros((len(ACTION_ORDER), len(ACTION_ORDER)), dtype=float)
        for row in rows:
            if row["event_type"] != event_type:
                continue
            y = ACTION_ORDER.index(str(row["previous_action"]))
            x = ACTION_ORDER.index(str(row["current_action"]))
            matrix[y, x] = int(row["n_transitions"])
        image = ax.imshow(matrix, vmin=0.0, vmax=max(1, max_count), cmap="Blues")
        ax.set_title(ACTION_EVENT_LABELS.get(event_type, event_type.replace("_", " ")))
        ax.set_xticks(range(len(ACTION_ORDER)), labels=ACTION_ORDER, rotation=35, ha="right")
        ax.set_yticks(range(len(ACTION_ORDER)), labels=ACTION_ORDER)
        ax.set_xlabel("Current action")
        ax.set_ylabel("Previous action")
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                ax.text(x, y, f"{int(matrix[y, x])}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Number of action transitions", shrink=0.82)
    fig.suptitle("Prefix-Elicited Action Transitions by Event Type", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_belief_transition_indicator_analysis(
    *,
    run_dir: str = DEFAULT_RUN_DIR,
    event_window: int = 3,
) -> None:
    """Write per-belief indicator statistics and recurring candidate chains."""
    root = Path(run_dir)
    events = [
        row
        for row in _read_csv(root / "action_transition_rows.csv")
        if _event_family(row["event_type"]) != "other"
    ]
    shifts = [
        row
        for row in _read_csv(root / "belief_shift_rows.csv")
        if row["shift_type"] in {"belief_error_onset", "belief_error_recovery"}
    ]
    shifts_by_example: dict[str, list[dict[str, str]]] = defaultdict(list)
    events_by_example: dict[str, list[dict[str, str]]] = defaultdict(list)
    for shift in shifts:
        shifts_by_example[shift["example_id"]].append(shift)
    for event in events:
        events_by_example[event["example_id"]].append(event)

    questions = sorted({shift["question_id"] for shift in shifts})
    indicator_rows: list[dict[str, Any]] = []
    for family in ("optimal_to_suboptimal", "suboptimal_to_optimal"):
        family_events = [event for event in events if _event_family(event["event_type"]) == family]
        target_shift = _target_shift_type(family)
        for question_id in questions:
            event_relations: dict[str, set[str]] = defaultdict(set)
            for event in family_events:
                center = int(event["reasoning_step_idx"])
                for shift in shifts_by_example[event["example_id"]]:
                    if shift["question_id"] != question_id or shift["shift_type"] != target_shift:
                        continue
                    lag = int(shift["reasoning_step_idx"]) - center
                    if abs(lag) > event_window:
                        continue
                    relation = "leading" if lag < 0 else "coincident" if lag == 0 else "lagging"
                    event_relations[event["event_id"]].add(relation)
            candidate_shifts = [
                shift
                for shift in shifts
                if shift["question_id"] == question_id and shift["shift_type"] == target_shift
            ]
            followed = 0
            for shift in candidate_shifts:
                step = int(shift["reasoning_step_idx"])
                if any(
                    _event_family(event["event_type"]) == family
                    and 0 <= int(event["reasoning_step_idx"]) - step <= event_window
                    for event in events_by_example[shift["example_id"]]
                ):
                    followed += 1
            denominator = len(family_events)
            leading = sum("leading" in relations for relations in event_relations.values())
            coincident = sum("coincident" in relations for relations in event_relations.values())
            lagging = sum("lagging" in relations for relations in event_relations.values())
            any_near = len(event_relations)
            indicator_rows.append(
                {
                    "event_family": family,
                    "target_shift_type": target_shift,
                    "question_id": question_id,
                    "question_family": next(
                        (shift["question_family"] for shift in candidate_shifts), ""
                    ),
                    "n_events": denominator,
                    "n_candidate_belief_shifts": len(candidate_shifts),
                    "events_with_leading_shift": leading,
                    "leading_event_coverage": leading / denominator if denominator else 0.0,
                    "events_with_coincident_shift": coincident,
                    "coincident_event_coverage": coincident / denominator if denominator else 0.0,
                    "events_with_lagging_shift": lagging,
                    "lagging_event_coverage": lagging / denominator if denominator else 0.0,
                    "events_with_any_nearby_shift": any_near,
                    "any_nearby_event_coverage": any_near / denominator if denominator else 0.0,
                    "shifts_followed_by_target_event": followed,
                    "leading_indicator_precision": followed / len(candidate_shifts) if candidate_shifts else 0.0,
                    "leading_indicator_false_alarm_rate": (
                        1.0 - followed / len(candidate_shifts) if candidate_shifts else 0.0
                    ),
                }
            )

    chain_rows: list[dict[str, Any]] = []
    pair_counter: Counter[tuple[str, str, str]] = Counter()
    pair_events: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    chain_counter: Counter[tuple[str, str]] = Counter()
    chain_events: dict[tuple[str, str], set[str]] = defaultdict(set)
    for event in events:
        family = _event_family(event["event_type"])
        target_shift = _target_shift_type(family)
        center = int(event["reasoning_step_idx"])
        relevant = [
            shift
            for shift in shifts_by_example[event["example_id"]]
            if shift["shift_type"] == target_shift
            and -event_window <= int(shift["reasoning_step_idx"]) - center <= 0
        ]
        primary_questions = sorted(
            {
                shift["question_id"]
                for shift in relevant
                if shift["question_id"] in PRIMARY_BELIEFS
            }
        )
        for left_idx, left in enumerate(primary_questions):
            for right in primary_questions[left_idx + 1 :]:
                pair_key = (family, left, right)
                pair_counter[pair_key] += 1
                pair_events[pair_key].add(event["event_id"])
        by_step: dict[int, list[str]] = defaultdict(list)
        for shift in relevant:
            by_step[int(shift["reasoning_step_idx"])].append(shift["question_id"])
        ordered_parts = [
            "+".join(sorted(set(by_step[step])))
            for step in sorted(by_step)
        ]
        if len({question for part in ordered_parts for question in part.split("+")}) < 2:
            continue
        chain = " -> ".join(ordered_parts)
        key = (family, chain)
        chain_counter[key] += 1
        chain_events[key].add(event["event_id"])
    family_event_counts = Counter(_event_family(event["event_type"]) for event in events)
    for (family, chain), count in chain_counter.most_common():
        chain_rows.append(
            {
                "event_family": family,
                "candidate_chain": chain,
                "n_events_with_chain": len(chain_events[(family, chain)]),
                "event_coverage": len(chain_events[(family, chain)]) / family_event_counts[family],
                "n_occurrences": count,
            }
        )
    pair_rows = [
        {
            "event_family": family,
            "belief_a": left,
            "belief_b": right,
            "n_events_with_pair": len(pair_events[(family, left, right)]),
            "event_coverage": len(pair_events[(family, left, right)]) / family_event_counts[family],
            "n_occurrences": count,
        }
        for (family, left, right), count in pair_counter.most_common()
    ]

    _write_csv(root / "belief_transition_indicator_summary.csv", indicator_rows)
    _write_csv(root / "candidate_belief_chains.csv", chain_rows)
    _write_csv(root / "candidate_primary_belief_pairs.csv", pair_rows)
    action_transition_rows = _action_transition_rows(_read_csv(root / "action_transition_rows.csv"))
    _write_csv(root / "action_transition_matrix.csv", action_transition_rows)
    _plot_indicator_heatmap(
        indicator_rows,
        root / "figs" / "belief_transition_indicator_heatmap.png",
    )
    _plot_indicator_heatmap(
        [row for row in indicator_rows if row["question_id"] in PRIMARY_BELIEFS],
        root / "figs" / "primary_belief_transition_indicator_heatmap.png",
    )
    _plot_action_transition_heatmap(
        action_transition_rows,
        root / "figs" / "action_transition_heatmap.png",
    )


__all__ = ["build_belief_transition_indicator_analysis"]
