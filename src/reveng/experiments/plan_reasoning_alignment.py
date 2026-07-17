"""Planning alignment experiment.

Activation side:
- published pre/post plan decoders on released public activations

Textual side:
- direct next-action elicitation (pre analogue)
- deliberative next-action / plan elicitation (post analogue)

The shared unit is one state x split key.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from reveng.agents.llm_templates import ActionResponse
from reveng.experiments.behavioral_probe_runner import _observed_action_prompt_with_state_text
from reveng.experiments.plan_decoder_reasoning_eval import (
    PLAN_ACTION_LABELS,
    PLAN_ACTION_NAME_TO_ID,
    build_public_plan_decoder_activation_rows,
    run_plan_decoder_reasoning_eval,
)
from reveng.llm_interface import BaseLLMInterface

DEFAULT_OUTPUT_DIR = "data/plan_reasoning_alignment"
PLAN_PREFIXES = (1, 2, 3, 5, 10)


class PlanSequenceResponse(BaseModel):
    actions: list[str] = Field(description="Action sequence using LEFT/RIGHT/UP/DOWN and optional __PAD__.")

    @field_validator("actions")
    @classmethod
    def _validate_actions(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            upper = str(item).strip().upper()
            if upper not in set(PLAN_ACTION_LABELS):
                raise ValueError(f"Unsupported action label: {item!r}")
            normalized.append(upper)
        if len(normalized) > 10:
            raise ValueError("Expected at most 10 actions in textual plan response")
        return normalized


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _normalize_action_sequence(actions: list[str], *, pad_to: int = 10) -> list[str]:
    norm = [str(action).strip().upper() for action in actions if str(action).strip()]
    norm = [action for action in norm if action in set(PLAN_ACTION_LABELS)]
    if len(norm) > pad_to:
        norm = norm[:pad_to]
    while len(norm) < pad_to:
        norm.append("__PAD__")
    return norm


def _parse_target_names(row: dict[str, Any]) -> list[str]:
    raw = json.loads(row["target_action_sequence_json"])
    if raw and isinstance(raw[0], int):
        return [PLAN_ACTION_LABELS[int(idx)] for idx in raw]
    return _normalize_action_sequence([str(x) for x in raw])


def _prefix_exact(predicted: list[str], target: list[str], prefix_len: int) -> int:
    return int(predicted[:prefix_len] == target[:prefix_len])


def _next_action_prompt(*, grid_text: str, reasoning_split: str) -> str:
    base = _observed_action_prompt_with_state_text(
        grid_text=grid_text,
        carrying_key=False,
        prompt_preset="cardinal_action_explicit",
        state_description_text=None,
    )
    if reasoning_split == "post":
        base = base.replace(
            "Choose the next move that best advances toward the goal while respecting the DoorKey mechanics in the current state.",
            "Think carefully about the route before answering. Then choose the next move that best advances toward the goal while respecting the DoorKey mechanics in the current state.",
        )
    return base


def _plan_prompt(*, grid_text: str, reasoning_split: str) -> str:
    prefix = (
        "Think carefully about the route before answering. "
        if reasoning_split == "post"
        else "Answer directly from the current state. "
    )
    return (
        "# Instructions\n\n"
        "You are controlling an agent in a fully observable grid-navigation environment. "
        "Interpret RIGHT/LEFT/UP/DOWN as cardinal directions in the rendered grid.\n\n"
        f"{prefix}Return only valid JSON and no extra text.\n\n"
        "Current grid state:\n\n"
        f"{grid_text}\n\n"
        "Return exactly one JSON object of the form "
        '{"actions": ["UP", "DOWN", "LEFT", "RIGHT"]}. '
        "List the best action sequence from this state toward the goal. You may return fewer than 10 actions; missing positions will be treated as __PAD__."
    )


def run_plan_textual_counterpart_eval(
    *,
    activation_rows_path: str,
    output_dir: str,
    model_name: str = "together_ai/openai/gpt-oss-20b",
) -> None:
    rows = _read_csv(Path(activation_rows_path))
    llm = BaseLLMInterface(model_name=model_name, temperature=0.0)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    textual_rows: list[dict[str, Any]] = []
    total_cost = 0.0
    for row in rows:
        target_names = _parse_target_names(row)
        next_prompt = _next_action_prompt(grid_text=row["grid_text"], reasoning_split=row["reasoning_split"])
        next_resp, next_cost, _ = llm._make_completion_request(next_prompt, response_format=ActionResponse)
        plan_prompt = _plan_prompt(grid_text=row["grid_text"], reasoning_split=row["reasoning_split"])
        plan_resp, plan_cost, _ = llm._make_completion_request(plan_prompt, response_format=PlanSequenceResponse)
        total_cost += float(next_cost) + float(plan_cost)

        next_action_name = next_resp.action.name
        predicted_plan = _normalize_action_sequence(plan_resp.actions)
        textual_rows.append(
            {
                "example_id": row["example_id"],
                "trajectory_id": row.get("trajectory_id", ""),
                "step_index": row.get("step_index", ""),
                "reasoning_split": row["reasoning_split"],
                "textual_prompt_mode": "deliberative" if row["reasoning_split"] == "post" else "direct",
                "grid_text": row.get("grid_text", ""),
                "target_action_names_json": json.dumps(target_names),
                "blackbox_next_action": next_action_name,
                "blackbox_next_action_correct": _prefix_exact([next_action_name], target_names, 1),
                "blackbox_plan_action_names_json": json.dumps(predicted_plan),
                "blackbox_plan_prefix_1_exact": _prefix_exact(predicted_plan, target_names, 1),
                "blackbox_plan_prefix_2_exact": _prefix_exact(predicted_plan, target_names, 2),
                "blackbox_plan_prefix_3_exact": _prefix_exact(predicted_plan, target_names, 3),
                "blackbox_plan_prefix_5_exact": _prefix_exact(predicted_plan, target_names, 5),
                "blackbox_plan_prefix_10_exact": _prefix_exact(predicted_plan, target_names, 10),
            }
        )
    _write_csv(out_dir / "plan_textual_counterpart_rows.csv", textual_rows)
    summary_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in textual_rows:
        grouped[str(row["reasoning_split"])].append(row)
    for split, split_rows in sorted(grouped.items()):
        n = len(split_rows)
        summary_rows.append(
            {
                "reasoning_split": split,
                "n_rows": n,
                "textual_next_action_accuracy": sum(int(r["blackbox_next_action_correct"]) for r in split_rows) / n,
                "textual_plan_prefix_1_accuracy": sum(int(r["blackbox_plan_prefix_1_exact"]) for r in split_rows) / n,
                "textual_plan_prefix_2_accuracy": sum(int(r["blackbox_plan_prefix_2_exact"]) for r in split_rows) / n,
                "textual_plan_prefix_3_accuracy": sum(int(r["blackbox_plan_prefix_3_exact"]) for r in split_rows) / n,
                "textual_plan_prefix_5_accuracy": sum(int(r["blackbox_plan_prefix_5_exact"]) for r in split_rows) / n,
                "textual_plan_prefix_10_accuracy": sum(int(r["blackbox_plan_prefix_10_exact"]) for r in split_rows) / n,
            }
        )
    _write_csv(out_dir / "plan_textual_counterpart_summary.csv", summary_rows)
    (out_dir / "usage_summary.json").write_text(json.dumps({"model_name": model_name, "estimated_total_cost": total_cost}, indent=2))


def build_plan_reasoning_alignment(
    *,
    plan_decoder_eval_dir: str,
    textual_counterpart_dir: str,
    output_dir: str,
) -> None:
    activation_rows = _read_csv(Path(plan_decoder_eval_dir) / "plan_decoder_eval_rows.csv")
    textual_rows = _read_csv(Path(textual_counterpart_dir) / "plan_textual_counterpart_rows.csv")
    text_by_key = {(r["example_id"], r["reasoning_split"]): r for r in textual_rows}
    combined_rows: list[dict[str, Any]] = []
    for row in activation_rows:
        key = (row["example_id"], row["reasoning_split"])
        text = text_by_key[key]
        combined_rows.append(
            {
                "example_id": row["example_id"],
                "trajectory_id": row["trajectory_id"],
                "step_index": row["step_index"],
                "reasoning_split": row["reasoning_split"],
                "source_dataset": row.get("source_dataset", ""),
                "observed_action": row.get("observed_action", ""),
                "target_action_names_json": row["target_action_names_json"],
                "whitebox_predicted_action_names_json": row["predicted_action_names_json"],
                "whitebox_next_action_correct": row["next_action_correct"],
                "whitebox_prefix_1_exact": row["prefix_1_exact"],
                "whitebox_prefix_2_exact": row["prefix_2_exact"],
                "whitebox_prefix_3_exact": row["prefix_3_exact"],
                "whitebox_prefix_5_exact": row["prefix_5_exact"],
                "whitebox_prefix_10_exact": row["prefix_10_exact"],
                "textual_prompt_mode": text["textual_prompt_mode"],
                "blackbox_next_action": text["blackbox_next_action"],
                "blackbox_next_action_correct": text["blackbox_next_action_correct"],
                "blackbox_plan_action_names_json": text["blackbox_plan_action_names_json"],
                "blackbox_plan_prefix_1_exact": text["blackbox_plan_prefix_1_exact"],
                "blackbox_plan_prefix_2_exact": text["blackbox_plan_prefix_2_exact"],
                "blackbox_plan_prefix_3_exact": text["blackbox_plan_prefix_3_exact"],
                "blackbox_plan_prefix_5_exact": text["blackbox_plan_prefix_5_exact"],
                "blackbox_plan_prefix_10_exact": text["blackbox_plan_prefix_10_exact"],
            }
        )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "plan_reasoning_alignment_rows.csv", combined_rows)

    def mean_int(rows: list[dict[str, Any]], key: str) -> float:
        return sum(int(r[key]) for r in rows) / len(rows) if rows else 0.0

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in combined_rows:
        grouped[row["reasoning_split"]].append(row)
    summary_rows: list[dict[str, Any]] = []
    for split, split_rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "reasoning_split": split,
                "n_rows": len(split_rows),
                "whitebox_prefix_1_accuracy": mean_int(split_rows, "whitebox_prefix_1_exact"),
                "whitebox_prefix_2_accuracy": mean_int(split_rows, "whitebox_prefix_2_exact"),
                "whitebox_prefix_3_accuracy": mean_int(split_rows, "whitebox_prefix_3_exact"),
                "whitebox_prefix_5_accuracy": mean_int(split_rows, "whitebox_prefix_5_exact"),
                "whitebox_prefix_10_accuracy": mean_int(split_rows, "whitebox_prefix_10_exact"),
                "blackbox_next_action_accuracy": mean_int(split_rows, "blackbox_next_action_correct"),
                "blackbox_plan_prefix_1_accuracy": mean_int(split_rows, "blackbox_plan_prefix_1_exact"),
                "blackbox_plan_prefix_2_accuracy": mean_int(split_rows, "blackbox_plan_prefix_2_exact"),
                "blackbox_plan_prefix_3_accuracy": mean_int(split_rows, "blackbox_plan_prefix_3_exact"),
                "blackbox_plan_prefix_5_accuracy": mean_int(split_rows, "blackbox_plan_prefix_5_exact"),
                "blackbox_plan_prefix_10_accuracy": mean_int(split_rows, "blackbox_plan_prefix_10_exact"),
            }
        )
    _write_csv(out_dir / "plan_reasoning_alignment_summary.csv", summary_rows)

    by_split = {r['reasoning_split']: r for r in summary_rows}
    paper_rows = []
    metric_specs = [
        ("prefix_1", "whitebox_prefix_1_accuracy", "blackbox_plan_prefix_1_accuracy"),
        ("prefix_2", "whitebox_prefix_2_accuracy", "blackbox_plan_prefix_2_accuracy"),
        ("prefix_3", "whitebox_prefix_3_accuracy", "blackbox_plan_prefix_3_accuracy"),
        ("prefix_5", "whitebox_prefix_5_accuracy", "blackbox_plan_prefix_5_accuracy"),
        ("prefix_10", "whitebox_prefix_10_accuracy", "blackbox_plan_prefix_10_accuracy"),
        ("direct_next_action", "whitebox_prefix_1_accuracy", "blackbox_next_action_accuracy"),
    ]
    for metric_name, white_key, black_key in metric_specs:
        pre = by_split.get('pre', {})
        post = by_split.get('post', {})
        paper_rows.append(
            {
                'metric': metric_name,
                'whitebox_pre': pre.get(white_key, ''),
                'whitebox_post': post.get(white_key, ''),
                'whitebox_delta_post_minus_pre': float(post.get(white_key, 0.0) or 0.0) - float(pre.get(white_key, 0.0) or 0.0),
                'textual_pre': pre.get(black_key, ''),
                'textual_post': post.get(black_key, ''),
                'textual_delta_post_minus_pre': float(post.get(black_key, 0.0) or 0.0) - float(pre.get(black_key, 0.0) or 0.0),
            }
        )
    _write_csv(out_dir / 'paper_plan_alignment_table.csv', paper_rows)
    md = [
        '| metric | whitebox_pre | whitebox_post | whitebox_delta_post_minus_pre | textual_pre | textual_post | textual_delta_post_minus_pre |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in paper_rows:
        md.append('| {metric} | {whitebox_pre} | {whitebox_post} | {whitebox_delta_post_minus_pre} | {textual_pre} | {textual_post} | {textual_delta_post_minus_pre} |'.format(**row))
    (out_dir / 'paper_plan_alignment_table.md').write_text('\n'.join(md) + '\n')

    # lightweight hypothesis note
    hypothesis = {
        'hypothesis': 'reasoning degrades long-horizon planning information while sharpening local action-readiness',
        'whitebox_local_delta_prefix_1': next(r['whitebox_delta_post_minus_pre'] for r in paper_rows if r['metric'] == 'prefix_1'),
        'whitebox_long_delta_prefix_5': next(r['whitebox_delta_post_minus_pre'] for r in paper_rows if r['metric'] == 'prefix_5'),
        'whitebox_long_delta_prefix_10': next(r['whitebox_delta_post_minus_pre'] for r in paper_rows if r['metric'] == 'prefix_10'),
        'textual_local_delta_prefix_1': next(r['textual_delta_post_minus_pre'] for r in paper_rows if r['metric'] == 'prefix_1'),
        'textual_long_delta_prefix_5': next(r['textual_delta_post_minus_pre'] for r in paper_rows if r['metric'] == 'prefix_5'),
        'textual_long_delta_prefix_10': next(r['textual_delta_post_minus_pre'] for r in paper_rows if r['metric'] == 'prefix_10'),
        'textual_direct_next_action_delta': next(r['textual_delta_post_minus_pre'] for r in paper_rows if r['metric'] == 'direct_next_action'),
    }
    (out_dir / 'plan_reasoning_hypothesis_summary.json').write_text(json.dumps(hypothesis, indent=2))

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
        x = [1, 2, 3, 5, 10]
        pre = by_split.get('pre', {})
        post = by_split.get('post', {})
        white_pre = [float(pre.get(f'whitebox_prefix_{k}_accuracy', 0.0) or 0.0) for k in x]
        white_post = [float(post.get(f'whitebox_prefix_{k}_accuracy', 0.0) or 0.0) for k in x]
        black_pre = [
            float(pre.get('blackbox_plan_prefix_1_accuracy', 0.0) or 0.0),
            float(pre.get('blackbox_plan_prefix_2_accuracy', 0.0) or 0.0),
            float(pre.get('blackbox_plan_prefix_3_accuracy', 0.0) or 0.0),
            float(pre.get('blackbox_plan_prefix_5_accuracy', 0.0) or 0.0),
            float(pre.get('blackbox_plan_prefix_10_accuracy', 0.0) or 0.0),
        ]
        black_post = [
            float(post.get('blackbox_plan_prefix_1_accuracy', 0.0) or 0.0),
            float(post.get('blackbox_plan_prefix_2_accuracy', 0.0) or 0.0),
            float(post.get('blackbox_plan_prefix_3_accuracy', 0.0) or 0.0),
            float(post.get('blackbox_plan_prefix_5_accuracy', 0.0) or 0.0),
            float(post.get('blackbox_plan_prefix_10_accuracy', 0.0) or 0.0),
        ]
        axes[0].plot(x, white_pre, marker='o', label='White-box pre')
        axes[0].plot(x, white_post, marker='o', label='White-box post')
        axes[0].set_title('Activation-Side Planning Readout')
        axes[0].set_xlabel('Prefix length')
        axes[0].set_ylabel('Exact prefix accuracy')
        axes[0].set_ylim(0.0, 1.0)
        axes[0].legend()
        axes[1].plot(x, black_pre, marker='o', label='Textual pre analogue')
        axes[1].plot(x, black_post, marker='o', label='Textual post analogue')
        axes[1].set_title('Textual Planning Counterpart')
        axes[1].set_xlabel('Prefix length')
        axes[1].set_ylabel('Exact prefix accuracy')
        axes[1].set_ylim(0.0, 1.0)
        axes[1].legend()
        fig.suptitle('Planning Hypothesis Test: local action-readiness vs longer-horizon planning')
        (out_dir / 'figs').mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / 'figs' / 'plan_reasoning_alignment.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
    except Exception:
        pass


def run_plan_reasoning_alignment_eval(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_name: str = 'together_ai/openai/gpt-oss-20b',
    grid_size: int = 7,
    max_trajectories: int = 5,
    max_steps_per_trajectory: int = 5,
) -> None:
    out_dir = Path(output_dir)
    activation_dir = out_dir / 'activation_side'
    textual_dir = out_dir / 'textual_side'
    run_plan_decoder_reasoning_eval(
        output_dir=str(activation_dir),
        grid_size=grid_size,
        max_trajectories=max_trajectories,
        max_steps_per_trajectory=max_steps_per_trajectory,
    )
    activation_rows_path = activation_dir / 'public_activation_rows.csv'
    if not activation_rows_path.exists() or not activation_rows_path.read_text().strip():
        raise ValueError('No plan activation rows were produced for the requested slice.')
    run_plan_textual_counterpart_eval(
        activation_rows_path=str(activation_rows_path),
        output_dir=str(textual_dir),
        model_name=model_name,
    )
    build_plan_reasoning_alignment(
        plan_decoder_eval_dir=str(activation_dir),
        textual_counterpart_dir=str(textual_dir),
        output_dir=str(out_dir),
    )


__all__ = [
    'run_plan_textual_counterpart_eval',
    'build_plan_reasoning_alignment',
    'run_plan_reasoning_alignment_eval',
]
