"""GPU residual-stream capture and patch runner for Phase 2B (exploratory causal pilot)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


def _layer_module(model: Any, layer_index: int) -> torch.nn.Module:
    return model.model.text_model.layers[layer_index]


def capture_last_residual(
    model: Any,
    inputs: Dict[str, Any],
    *,
    layer_index: int,
    device: str = "cuda:0",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Capture post-block residual row at final prompt token for one layer."""
    captured: Dict[str, Any] = {}

    def _hook(module, args, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        # last prompt token of this forward
        row = hidden[0, -1, :].detach().float().cpu().numpy()
        captured["row"] = row
        captured["native_dtype"] = str(hidden.dtype)
        captured["seq_len"] = int(hidden.shape[1])
        return output

    handle = _layer_module(model, layer_index).register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            out = model(**inputs)
            logits = out.logits[0, -1, :].detach().float()
    finally:
        handle.remove()
    if "row" not in captured:
        raise RuntimeError(f"Layer {layer_index} hook did not fire")
    meta = {
        "layer_index": layer_index,
        "native_dtype": captured["native_dtype"],
        "stored_dtype": "float32",
        "shape": list(captured["row"].shape),
        "seq_len": captured["seq_len"],
        "device": device,
    }
    return captured["row"], {"logits": logits, "meta": meta}


class residual_patch_context:
    """Install a post-block residual replacement hook for the duration of a block.

    Ensures candidate scoring re-forwards also observe the intervention.
    """

    def __init__(
        self,
        model: Any,
        *,
        layer_index: int,
        token_position: int,
        replacement_row: Optional[np.ndarray],
        no_op: bool = False,
    ):
        self.model = model
        self.layer_index = layer_index
        self.token_position = token_position
        self.replacement_row = replacement_row
        self.no_op = no_op
        self.applied = 0
        self._handle = None

    def __enter__(self):
        def _hook(module, args, output):
            if self.no_op or self.replacement_row is None:
                return output
            if isinstance(output, (tuple, list)):
                hidden = output[0]
                rest = output[1:]
                is_tuple = True
            else:
                hidden = output
                rest = ()
                is_tuple = False
            if self.token_position < 0 or self.token_position >= hidden.shape[1]:
                raise IndexError("token_position out of range during patch context")
            vec = torch.as_tensor(self.replacement_row, dtype=hidden.dtype, device=hidden.device)
            if vec.numel() != hidden.shape[-1]:
                raise ValueError("replacement dim mismatch")
            hidden = hidden.clone()
            hidden[0, self.token_position, :] = vec
            self.applied += 1
            if is_tuple:
                return (hidden,) + rest
            return hidden

        self._handle = _layer_module(self.model, self.layer_index).register_forward_hook(_hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


def run_with_residual_patch(
    model: Any,
    inputs: Dict[str, Any],
    *,
    layer_index: int,
    token_position: int,
    replacement_row: Optional[np.ndarray],
    no_op: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Forward pass that optionally replaces residual row at (layer, token_position).

    Hook is post-block (decoder layer output). replacement_row is [H] float32 CPU array.
    """
    applied = {"count": 0, "shape": None, "dtype": None}

    def _hook(module, args, output):
        if isinstance(output, (tuple, list)):
            hidden = output[0]
            rest = output[1:]
            is_tuple = True
        else:
            hidden = output
            rest = ()
            is_tuple = False
        if no_op or replacement_row is None:
            return output
        if token_position < 0 or token_position >= hidden.shape[1]:
            raise IndexError(f"token_position {token_position} out of range T={hidden.shape[1]}")
        vec = torch.as_tensor(replacement_row, dtype=hidden.dtype, device=hidden.device)
        if vec.numel() != hidden.shape[-1]:
            raise ValueError(f"replacement dim {vec.numel()} != hidden {hidden.shape[-1]}")
        hidden = hidden.clone()
        hidden[0, token_position, :] = vec
        applied["count"] += 1
        applied["shape"] = list(hidden.shape)
        applied["dtype"] = str(hidden.dtype)
        if is_tuple:
            return (hidden,) + rest
        return hidden

    handle = _layer_module(model, layer_index).register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            out = model(**inputs)
            logits = out.logits[0, -1, :].detach().float()
    finally:
        handle.remove()

    if (not no_op) and applied["count"] != 1:
        raise RuntimeError(f"Patch hook applied {applied['count']} times (expected 1)")
    meta = {
        "layer_index": layer_index,
        "token_position": token_position,
        "no_op": no_op,
        "hook_applied": applied["count"],
        "hidden_shape_during_hook": applied["shape"],
        "hidden_dtype_during_hook": applied["dtype"],
    }
    return logits, meta


def candidate_loglikelihoods(
    model: Any,
    processor: Any,
    base_inputs: Dict[str, Any],
    candidate_strings: List[str],
    *,
    logits_override: Optional[torch.Tensor] = None,
    device: str = "cuda:0",
) -> Dict[str, float]:
    """Score complete candidate strings under the same visual/prompt prefix.

    If logits_override is provided, it is the *prompt-final* logits from a single
    forward; subsequent candidate tokens still use model forward for multi-token
    sequences. For multi-token candidates we re-run with concatenated ids.
    """
    from src.evaluation.candidate_scorer import score_candidate_answers

    if logits_override is not None:
        # First-token only from override; full sequence still needs scorer
        pass
    result = score_candidate_answers(
        model=model,
        processor=processor,
        base_inputs=base_inputs,
        candidate_strings=candidate_strings,
        device=device,
    )
    return {k: v.total_log_likelihood for k, v in result.candidates.items()}


def score_margins_from_logprobs(
    logprobs: Dict[str, float],
    source_label: str,
    target_label: str,
) -> Dict[str, float]:
    """m = logP(source_label) - logP(target_label) under the given run."""
    sl = logprobs[source_label]
    tl = logprobs[target_label]
    return {
        "logP_source": sl,
        "logP_target": tl,
        "margin_source_minus_target": sl - tl,
    }


def delta_margin(
    baseline_m: float,
    patched_m: float,
) -> Dict[str, Any]:
    """Δm = m_patched - m_baseline (positive ⇒ shift toward source label)."""
    raw = patched_m - baseline_m
    denom = abs(baseline_m)
    nie = raw / denom if denom >= 1e-3 else None
    return {
        "baseline_margin": baseline_m,
        "patched_margin": patched_m,
        "delta_m": raw,
        "normalized_delta_m": nie,
        "nie_stability": "stable" if denom >= 1e-3 else "unstable_suppressed",
        "nie_note": "Raw delta_m is primary; normalized secondary with |m_baseline|>=1e-3 gate.",
    }
