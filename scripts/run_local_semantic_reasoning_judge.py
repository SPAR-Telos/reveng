#!/usr/bin/env python3
"""Label blinded reasoning sentences with the locally cached GPT-OSS-20B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reveng.experiments.semantic_reasoning_classification import (
    ANNOTATION_COLUMNS,
    validate_annotation_rows,
)
from run_semantic_reasoning_judge import (
    DEFAULT_INPUT,
    SYSTEM_PROMPT,
    normalize_payload,
    parse_json_response,
    user_prompt,
)


DEFAULT_MODEL = Path(
    "/root/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/"
    "6cee5e81ee83917806bbde320786a8fb61efebee"
)
DEFAULT_OUTPUT = Path(
    "outputs/hypothesis_tests/semantic_reasoning_classification_v1/"
    "gpt_oss_local_annotations.csv"
)

CALIBRATED_SYSTEM_PROMPT = """You label the discourse function of one target
sentence in a navigation reasoning trace. Use only the preceding reasoning and
target sentence. Do not judge factual correctness, action optimality, task
success, or whether an action distribution changed.

Choose exactly one primary label. Apply this precedence and the contrastive
rules literally:

1. correction: explicitly rejects, revises, or reverses an earlier claim,
route, coordinate, or conclusion. A contrast word alone is insufficient.
2. verification: evaluates an earlier fact, candidate move, or route for
validity, traversability, consistency, or efficiency. This includes declarative
findings such as "that route is blocked" when they evaluate a route already
under discussion, and questions such as "can we go through row 1?".
3. state_reconstruction: directly transcribes or locates visible grid/state
information without evaluating a route. Cell coordinates, row contents,
adjacent-cell inventories, and carrying-key status belong here.
4. route_planning: proposes a new move, route, subgoal, or action sequence.
Statements of where to go next are route_planning even if they use "so" or
"maybe".
5. new_inference: derives a new consequence or constraint that is neither a
route proposal nor an evaluation of an earlier proposal. Use this only after
excluding labels 1-4.
6. consolidation: combines multiple earlier facts or route checks into a
summary, final plan, or action decision without adding a materially new
premise.
7. restatement: repeats one earlier fact, route, or conclusion without
checking, combining, or changing it.
8. procedural_continuation: meta-level continuation, enumeration, arithmetic,
or bookkeeping with no more specific function.
9. unclear: context is insufficient or two labels remain equally plausible
after applying the rules.

Critical contrasts:
- "Row 4 column 6 is a wall." is state_reconstruction.
- After considering a path through row 4, "Row 4 column 6 is a wall, so that
  path fails." is verification.
- "Go up twice and then left." is route_planning.
- After several checks, "Therefore the final action is UP." is consolidation.
- "The closed door requires the key" is new_inference only when it introduces
  that consequence; it is verification when used to test a route already under
  discussion.
- "Wait, row 4 column 6 is a wall, not open" is correction.

Before choosing, contrast the two most plausible labels and apply the
precedence. Return only one JSON object with:
explicitly_revises_prior_reasoning, evaluates_prior_route_or_claim,
repeats_prior_content, introduces_new_information_or_plan (each yes, no, or
unsure); primary_label; confidence (high, medium, or low); and a one-sentence
rationale that states why the chosen label outranks the nearest alternative."""

STRICT_SYSTEM_PROMPT = """Classify the discourse function of one target
sentence in a navigation reasoning trace. Use only the preceding reasoning and
target sentence. Ignore factual correctness, action optimality, task success,
and action-distribution changes.

Apply this decision tree in order.

A. Does the target explicitly replace or reject a prior assertion? If yes:
`correction`.
B. Does it test, confirm, or rule out a route, move, factual claim, or answer
already under discussion? If yes: `verification`.
C. Does it directly read the grid or state--coordinates, row contents, cell
contents, adjacent cells, object locations, or key-carrying status? If yes:
`state_reconstruction`.
D. Does it propose where to move, a route, a subgoal, or an action sequence?
If yes: `route_planning`.
E. Does it derive a consequence or constraint that is not a direct grid
observation, route proposal, or test of an earlier proposal? If yes:
`new_inference`.
F. Does it combine several prior findings into a summary, complete plan, or
final action decision? If yes: `consolidation`.
G. Does it repeat one prior finding or route without checking or combining it?
If yes: `restatement`.
H. Is it only a meta-level transition, enumeration, arithmetic, or bookkeeping?
If yes: `procedural_continuation`.
Otherwise: `unclear`.

Hard exclusions:
- `new_inference` MUST NOT be used for a direct coordinate, row transcription,
cell observation, adjacency inventory, or object location, even when that
observed information is new. Those are `state_reconstruction`.
- `new_inference` MUST NOT be used for a proposed move, route, or subgoal.
Those are `route_planning`.
- A route that reaches a wall or is declared blocked is `verification` unless
the sentence explicitly says an earlier factual assertion was wrong, in which
case it is `correction`.
- "Let's analyze connectivity" alone is `procedural_continuation`; a sentence
that actually asks whether a particular connection works is `verification`.
- "Thus the answer is UP" after route analysis is `consolidation`. Repeating
an already stated answer is `restatement`.

Before answering, identify which decision-tree branch applies and check every
hard exclusion. Return only one JSON object with:
explicitly_revises_prior_reasoning, evaluates_prior_route_or_claim,
repeats_prior_content, introduces_new_information_or_plan (each yes, no, or
unsure); primary_label; confidence (high, medium, or low); and a one-sentence
rationale naming the applicable branch."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=260)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prompt-version",
        choices=("v1", "v2_calibrated", "v3_strict"),
        default="v1",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="low",
    )
    return parser.parse_args()


def render_prompt(
    tokenizer: AutoTokenizer,
    row: object,
    *,
    system_prompt: str,
    reasoning_effort: str,
) -> str:
    rendered = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": system_prompt + "\n\n" + user_prompt(row),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=reasoning_effort,
    )
    return rendered + "<|channel|>final<|message|>"


def main() -> None:
    args = parse_args()
    system_prompt = {
        "v1": SYSTEM_PROMPT,
        "v2_calibrated": CALIBRATED_SYSTEM_PROMPT,
        "v3_strict": STRICT_SYSTEM_PROMPT,
    }[args.prompt_version]
    system_prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    source = pd.read_csv(args.input).fillna("")
    if args.limit is not None:
        source = source.head(args.limit)

    completed: dict[str, dict[str, object]] = {}
    if args.resume and args.output.exists():
        previous = pd.read_csv(args.output).fillna("")
        completed = {
            str(row["annotation_id"]): row
            for row in previous.to_dict("records")
            if str(row.get("primary_label", "")).strip()
        }
    output = list(completed.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    ).eval()
    device = model.model.embed_tokens.weight.device
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    for index, row in enumerate(source.itertuples(index=False), start=1):
        annotation_id = str(row.annotation_id)
        if annotation_id in completed:
            continue
        rendered = render_prompt(
            tokenizer,
            row,
            system_prompt=system_prompt,
            reasoning_effort=args.reasoning_effort,
        )
        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)
        raw = ""
        error = ""
        parsed: dict[str, str] = {}
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )[0]
            new_ids = output_ids[inputs["input_ids"].shape[1] :]
            raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            parsed = normalize_payload(parse_json_response(raw))
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        record = {
            "annotation_id": annotation_id,
            "context_before": row.context_before,
            "target_sentence": row.target_sentence,
            "target_sentence_characters": row.target_sentence_characters,
            **{column: parsed.get(column, "") for column in ANNOTATION_COLUMNS},
            "judge_model": "openai/gpt-oss-20b",
            "judge_revision": args.model_path.name,
            "judge_temperature": 0.0,
            "judge_seed": 42,
            "judge_prompt_version": args.prompt_version,
            "judge_prompt_sha256": system_prompt_hash,
            "judge_reasoning_effort": args.reasoning_effort,
            "raw_response": raw,
            "error": error,
        }
        output.append(record)
        if parsed:
            completed[annotation_id] = record
        pd.DataFrame(output).drop_duplicates(
            "annotation_id", keep="last"
        ).to_csv(args.output, index=False)
        if index % 10 == 0 or error:
            print(
                f"processed={index}/{len(source)} completed={len(completed)} "
                f"error={error or 'none'}",
                flush=True,
            )

    final = pd.read_csv(args.output).fillna("")
    labeled = final[final["primary_label"].astype(str).str.strip().ne("")]
    validate_annotation_rows(labeled, allow_partial=False)
    print(f"output={args.output} labeled={len(labeled)}/{len(source)}")


if __name__ == "__main__":
    main()
