"""Live hidden-state patching utilities for local Hugging Face causal LMs.

This module is intentionally separate from the existing counterfactual
surrogate evaluator. The existing evaluator operates on saved traces; this
module runs a local hookable model and patches hidden states live during the
forward pass so later layers are recomputed.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import matplotlib.pyplot as plt

FINAL_CHANNEL_MARKER = "<|end|><|start|>assistant<|channel|>final<|message|>"
ACTION_LABELS = ("UP", "DOWN", "LEFT", "RIGHT")

try:  # Optional dependency in the current workspace.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - exercised by import-time environment only.
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


class LivePatchBackend(Protocol):
    """Minimal backend interface used by the curve runner."""

    def num_layers(self) -> int: ...

    def encode_text(self, text: str) -> list[int]: ...

    def action_probabilities(
        self,
        *,
        prefix_a_text: str,
        prefix_b_text: str,
        layer_idx: int,
        patch_positions_a: list[int],
        patch_positions_b: list[int],
        candidate_labels: list[str],
    ) -> dict[str, float]: ...


@dataclass
class StepPatchInput:
    prompt_text: str
    reasoning_text: str
    prefix_text: str
    action_label: str
    reasoning_positions: list[int]


@dataclass
class CurveMetadata:
    status: str
    pair_id: str
    step_index: int
    model_name_or_path: str
    a_trace_path: str
    b_trace_path: str
    source_action: str
    target_action: str
    patch_window_mode: str
    reasoning_window_size: int
    patch_positions_a: list[int]
    patch_positions_b: list[int]
    tracked_actions: list[str]
    layer_start: int
    layer_end: int
    note: str


def _extract_final_action(step: dict[str, Any]) -> str:
    for token in step.get("output_tokens", []):
        groups = set(token.get("token_groups") or [])
        tok = token.get("token")
        if {"output", "final", "action"} <= groups and isinstance(tok, str):
            cleaned = tok.strip().upper()
            if cleaned in ACTION_LABELS:
                return cleaned

    text = step.get("output_text", "")
    match = re.search(r'"action"\s*:\s*"(UP|DOWN|LEFT|RIGHT)"', text)
    if match:
        return match.group(1)
    raise ValueError("Could not recover final action token from step output.")


def _render_prompt(trace: dict[str, Any], step_index: int) -> str:
    template = trace.get("prompt", {}).get("prompt_template")
    if not isinstance(template, str):
        raise ValueError("Trace is missing prompt.prompt_template.")

    steps = trace.get("steps") or []
    if step_index >= len(steps):
        raise IndexError(f"step_index={step_index} out of range for {len(steps)} steps.")

    grid_state = steps[step_index].get("grid_state")
    if not isinstance(grid_state, list) or not all(isinstance(x, str) for x in grid_state):
        raise ValueError("Trace step is missing grid_state text.")

    return template.replace("{{grid_state}}", "\n".join(grid_state))


def _split_output_prefix(output_text: str, action_label: str) -> tuple[str, str]:
    if FINAL_CHANNEL_MARKER not in output_text:
        raise ValueError("Step output does not contain the expected final-channel marker.")

    reasoning_text, final_tail = output_text.split(FINAL_CHANNEL_MARKER, 1)
    action_match = re.search(
        rf'^(?P<prefix>.*?"action"\s*:\s*")(?P<action>{action_label})(?P<suffix>".*)$',
        final_tail,
        re.DOTALL,
    )
    if action_match is None:
        raise ValueError("Unable to isolate the final JSON action span in output_text.")

    prefix_after_reasoning = FINAL_CHANNEL_MARKER + action_match.group("prefix")
    return reasoning_text, prefix_after_reasoning


def _build_step_patch_input(
    trace: dict[str, Any],
    step_index: int,
    backend: LivePatchBackend,
    reasoning_window_size: int,
) -> StepPatchInput:
    steps = trace.get("steps") or []
    if step_index >= len(steps):
        raise IndexError(f"step_index={step_index} out of range for {len(steps)} steps.")
    step = steps[step_index]

    prompt_text = _render_prompt(trace, step_index)
    action_label = _extract_final_action(step)
    output_text = step.get("output_text")
    if not isinstance(output_text, str):
        raise ValueError("Trace step is missing output_text.")

    reasoning_text, final_prefix_tail = _split_output_prefix(output_text, action_label)
    reasoning_prefix_text = prompt_text + reasoning_text
    reasoning_token_ids = backend.encode_text(reasoning_prefix_text)
    if len(reasoning_token_ids) < reasoning_window_size:
        raise ValueError(
            "No clean reasoning span exists: fewer model tokens than the requested "
            f"reasoning_window_size={reasoning_window_size}."
        )

    reasoning_positions = list(
        range(
            len(reasoning_token_ids) - reasoning_window_size,
            len(reasoning_token_ids),
        )
    )
    return StepPatchInput(
        prompt_text=prompt_text,
        reasoning_text=reasoning_text,
        prefix_text=reasoning_prefix_text + final_prefix_tail,
        action_label=action_label,
        reasoning_positions=reasoning_positions,
    )


class HuggingFaceLivePatchBackend:
    """Hook-based backend for local Hugging Face causal language models."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cpu",
        trust_remote_code: bool = False,
    ) -> None:
        if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ModuleNotFoundError(
                "run_live_patch_curve requires local torch + transformers model "
                "weights; the current environment does not have torch installed."
            )

        self.model_name_or_path = model_name_or_path
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()
        self.layers = self._resolve_layers()

    def _resolve_layers(self) -> Any:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        raise ValueError(
            "Unsupported model architecture for live patching. "
            "Expected model.model.layers or model.transformer.h."
        )

    def num_layers(self) -> int:
        return len(self.layers)

    def encode_text(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        return list(encoded["input_ids"])

    def _candidate_token_ids(self, prefix_text: str, candidate_label: str) -> list[int]:
        prefix_ids = self.encode_text(prefix_text)
        full_ids = self.encode_text(prefix_text + candidate_label)
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                "Candidate label tokenization does not preserve the prefix boundary. "
                "Use a model/tokenizer where the final action string tokenizes cleanly "
                "after the JSON quote boundary."
            )
        suffix = full_ids[len(prefix_ids) :]
        if not suffix:
            raise ValueError(f"Candidate label {candidate_label!r} produced no suffix tokens.")
        return suffix

    def _forward(
        self,
        input_ids: list[int],
        *,
        patch_layer: Optional[int] = None,
        patch_positions_a: Optional[list[int]] = None,
        patch_values: Optional[Any] = None,
        output_hidden_states: bool = False,
    ) -> Any:
        hook = None
        if patch_layer is not None:
            if patch_positions_a is None or patch_values is None:
                raise ValueError("patch_positions_a and patch_values are required when patching.")
            positions = list(patch_positions_a)
            expected_width = patch_values.shape[1]
            if len(positions) != expected_width:
                raise ValueError("Patch position width does not match cached patch values width.")

            def _patch_hook(_module: Any, _inputs: Any, output: Any) -> Any:
                if isinstance(output, tuple):
                    hidden = output[0]
                    remainder = output[1:]
                else:
                    hidden = output
                    remainder = None

                hidden = hidden.clone()
                hidden[:, positions, :] = patch_values.to(hidden.device, dtype=hidden.dtype)
                if remainder is None:
                    return hidden
                return (hidden, *remainder)

            hook = self.layers[patch_layer].register_forward_hook(_patch_hook)

        try:
            with torch.no_grad():
                tensor_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                return self.model(
                    input_ids=tensor_ids,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )
        finally:
            if hook is not None:
                hook.remove()

    def _sequence_probability(
        self,
        *,
        prefix_ids: list[int],
        candidate_token_ids: list[int],
        layer_idx: int,
        patch_positions_a: list[int],
        patch_values: Any,
    ) -> float:
        scoring_input_ids = prefix_ids + candidate_token_ids[:-1]
        outputs = self._forward(
            scoring_input_ids,
            patch_layer=layer_idx,
            patch_positions_a=patch_positions_a,
            patch_values=patch_values,
            output_hidden_states=False,
        )
        logits = outputs.logits[0]
        prob = 1.0
        prefix_last = len(prefix_ids) - 1
        for offset, token_id in enumerate(candidate_token_ids):
            row_idx = prefix_last + offset
            token_prob = torch.softmax(logits[row_idx], dim=-1)[token_id].item()
            prob *= float(token_prob)
        return prob

    def action_probabilities(
        self,
        *,
        prefix_a_text: str,
        prefix_b_text: str,
        layer_idx: int,
        patch_positions_a: list[int],
        patch_positions_b: list[int],
        candidate_labels: list[str],
    ) -> dict[str, float]:
        prefix_a_ids = self.encode_text(prefix_a_text)
        prefix_b_ids = self.encode_text(prefix_b_text)
        outputs_b = self._forward(prefix_b_ids, output_hidden_states=True)
        hidden_states = outputs_b.hidden_states
        if hidden_states is None:
            raise ValueError("Model did not return hidden_states.")

        patch_values = hidden_states[layer_idx + 1][:, patch_positions_b, :].detach()

        probs: dict[str, float] = {}
        for label in candidate_labels:
            candidate_token_ids = self._candidate_token_ids(prefix_a_text, label)
            probs[label] = self._sequence_probability(
                prefix_ids=prefix_a_ids,
                candidate_token_ids=candidate_token_ids,
                layer_idx=layer_idx,
                patch_positions_a=patch_positions_a,
                patch_values=patch_values,
            )
        return probs


def _default_backend_factory(
    model_name_or_path: str,
    *,
    device: str = "cpu",
    trust_remote_code: bool = False,
) -> LivePatchBackend:
    return HuggingFaceLivePatchBackend(
        model_name_or_path,
        device=device,
        trust_remote_code=trust_remote_code,
    )


def _resolve_trace_paths(
    *,
    pair_id: str,
    artifacts_dir: Path,
    a_trace_path: Optional[str],
    b_trace_path: Optional[str],
) -> tuple[Path, Path]:
    if a_trace_path and b_trace_path:
        return Path(a_trace_path), Path(b_trace_path)

    pair_dir = artifacts_dir / pair_id
    return pair_dir / "A.json", pair_dir / "B.json"


def _write_curve_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["layer"]
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_probability_curves(
    *,
    rows: list[dict[str, Any]],
    source_action: str,
    target_action: str,
    output_path: Path,
) -> None:
    layers = [int(row["layer"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    for action in ACTION_LABELS:
        probs = [float(row[f"{action.lower()}_prob"]) for row in rows]
        if action == source_action:
            ax.plot(layers, probs, marker="o", linewidth=2.2, label=f"{action} (A)")
        elif action == target_action:
            ax.plot(layers, probs, marker="o", linewidth=2.2, label=f"{action} (B)")
        else:
            ax.plot(layers, probs, linewidth=1.2, alpha=0.35, label=action)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Probability")
    ax.set_title("Activation Patching: Action Probability by Layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_probability_delta(
    *,
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    layers = [int(row["layer"]) for row in rows]
    deltas = [float(row["target_minus_source"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, deltas, marker="o", linewidth=2.0)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Target - Source probability")
    ax.set_title("Activation Patching: Target-vs-Source Probability Delta")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _run_live_patch_curve(
    *,
    model_name_or_path: str,
    pair_id: str,
    step_index: int,
    output_dir: str,
    artifacts_dir: str = "data/cf/artifacts",
    a_trace_path: Optional[str] = None,
    b_trace_path: Optional[str] = None,
    layer_start: int = 0,
    layer_end: Optional[int] = None,
    patch_window_mode: str = "tail_reasoning_3",
    reasoning_window_size: int = 3,
    device: str = "cpu",
    trust_remote_code: bool = False,
    backend_factory: Optional[Callable[..., LivePatchBackend]] = None,
) -> None:
    if patch_window_mode != "tail_reasoning_3":
        raise ValueError(
            "Only patch_window_mode='tail_reasoning_3' is implemented in the first pass."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a_path, b_path = _resolve_trace_paths(
        pair_id=pair_id,
        artifacts_dir=Path(artifacts_dir),
        a_trace_path=a_trace_path,
        b_trace_path=b_trace_path,
    )
    if not a_path.exists():
        raise FileNotFoundError(f"Missing A trace: {a_path}")
    if not b_path.exists():
        raise FileNotFoundError(f"Missing B trace: {b_path}")

    backend_builder = backend_factory or _default_backend_factory
    backend = backend_builder(
        model_name_or_path,
        device=device,
        trust_remote_code=trust_remote_code,
    )

    trace_a = json.loads(a_path.read_text())
    trace_b = json.loads(b_path.read_text())
    step_a = _build_step_patch_input(trace_a, step_index, backend, reasoning_window_size)
    step_b = _build_step_patch_input(trace_b, step_index, backend, reasoning_window_size)

    source_action = step_a.action_label
    target_action = step_b.action_label

    num_layers = backend.num_layers()
    resolved_layer_end = (num_layers - 1) if layer_end is None else layer_end
    if layer_start < 0 or resolved_layer_end >= num_layers or layer_start > resolved_layer_end:
        raise ValueError(
            f"Invalid layer range [{layer_start}, {resolved_layer_end}] for num_layers={num_layers}."
        )

    rows: list[dict[str, Any]] = []
    for layer_idx in range(layer_start, resolved_layer_end + 1):
        probs = backend.action_probabilities(
            prefix_a_text=step_a.prefix_text,
            prefix_b_text=step_b.prefix_text,
            layer_idx=layer_idx,
            patch_positions_a=step_a.reasoning_positions,
            patch_positions_b=step_b.reasoning_positions,
            candidate_labels=list(ACTION_LABELS),
        )
        row: dict[str, Any] = {
            "layer": layer_idx,
            "pair_id": pair_id,
            "step_index": step_index,
            "source_action": source_action,
            "target_action": target_action,
            "target_minus_source": probs[target_action] - probs[source_action],
        }
        for action in ACTION_LABELS:
            row[f"{action.lower()}_prob"] = probs[action]
        rows.append(row)

    metadata = CurveMetadata(
        status="ok",
        pair_id=pair_id,
        step_index=step_index,
        model_name_or_path=model_name_or_path,
        a_trace_path=str(a_path),
        b_trace_path=str(b_path),
        source_action=source_action,
        target_action=target_action,
        patch_window_mode=patch_window_mode,
        reasoning_window_size=reasoning_window_size,
        patch_positions_a=step_a.reasoning_positions,
        patch_positions_b=step_b.reasoning_positions,
        tracked_actions=list(ACTION_LABELS),
        layer_start=layer_start,
        layer_end=resolved_layer_end,
        note=(
            "This is the live hidden-state patching path. It is separate from the "
            "existing surrogate trace-substitution baseline."
        ),
    )

    _write_curve_csv(out_dir / "probability_by_layer.csv", rows)
    _plot_probability_curves(
        rows=rows,
        source_action=source_action,
        target_action=target_action,
        output_path=out_dir / "probability_by_layer.png",
    )
    _plot_probability_delta(rows=rows, output_path=out_dir / "probability_delta_by_layer.png")
    (out_dir / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2))


def run_live_patch_curve(
    model_name_or_path: str,
    pair_id: str = "pair_goal_move_009",
    step_index: int = 0,
    output_dir: str = "data/cf/live_patch_curve",
    artifacts_dir: str = "data/cf/artifacts",
    a_trace_path: Optional[str] = None,
    b_trace_path: Optional[str] = None,
    layer_start: int = 0,
    layer_end: Optional[int] = None,
    patch_window_mode: str = "tail_reasoning_3",
    reasoning_window_size: int = 3,
    device: str = "cpu",
    trust_remote_code: bool = False,
) -> None:
    """Run live hidden-state patching and write Mario-style layer curves.

    This command requires a local hookable Hugging Face causal LM. It does not
    use the Together API path.
    """
    _run_live_patch_curve(
        model_name_or_path=model_name_or_path,
        pair_id=pair_id,
        step_index=step_index,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
        a_trace_path=a_trace_path,
        b_trace_path=b_trace_path,
        layer_start=layer_start,
        layer_end=layer_end,
        patch_window_mode=patch_window_mode,
        reasoning_window_size=reasoning_window_size,
        device=device,
        trust_remote_code=trust_remote_code,
    )


__all__ = [
    "ACTION_LABELS",
    "FINAL_CHANNEL_MARKER",
    "CurveMetadata",
    "HuggingFaceLivePatchBackend",
    "StepPatchInput",
    "_build_step_patch_input",
    "_extract_final_action",
    "_render_prompt",
    "_run_live_patch_curve",
    "run_live_patch_curve",
]
