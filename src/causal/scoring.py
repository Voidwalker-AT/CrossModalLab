"""Complete-answer candidate scoring with optional residual patch context.

Ensures the intervention hook is active for every candidate re-forward.
Token position refers to the original prompt-final token index.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.causal.patch_runner import residual_patch_context


def score_complete_candidates(
    model: Any,
    processor: Any,
    base_inputs: Dict[str, Any],
    candidate_strings: List[str],
    *,
    layer_index: Optional[int] = None,
    token_position: Optional[int] = None,
    replacement_row: Optional[np.ndarray] = None,
    no_op: bool = False,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Score candidates with the same prefix; optionally keep patch hook active.

    The patched token index is the *original prompt length - 1* of base_inputs.
    Candidate continuation tokens are appended after the prompt; the hook still
    replaces the absolute position token_position (prompt-final), not continuation
    positions. If token_position >= prompt_len, raise.
    """
    from src.evaluation.candidate_scorer import score_candidate_answers

    prompt_len = int(base_inputs["input_ids"].shape[1])
    if layer_index is not None:
        if token_position is None:
            raise ValueError("token_position required when layer_index set")
        if token_position < 0 or token_position >= prompt_len:
            raise ValueError(
                f"token_position {token_position} not in original prompt range [0,{prompt_len})"
            )

    def _score():
        return score_candidate_answers(
            model=model,
            processor=processor,
            base_inputs=base_inputs,
            candidate_strings=candidate_strings,
            device=device,
        )

    if layer_index is None:
        cand = _score()
    else:
        with residual_patch_context(
            model,
            layer_index=layer_index,
            token_position=token_position,
            replacement_row=replacement_row,
            no_op=no_op,
        ) as ctx:
            cand = _score()
            hook_applied = ctx.applied

    logprobs = {k: v.total_log_likelihood for k, v in cand.candidates.items()}
    return {
        "logprobs": logprobs,
        "prompt_len": prompt_len,
        "hook_applied": hook_applied if layer_index is not None else 0,
        "candidate_keys": list(cand.candidates.keys()),
    }
