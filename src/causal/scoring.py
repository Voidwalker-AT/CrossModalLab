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
        "scorer": "full_forward",
    }


def score_candidates_hybrid(
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
    """Optimized complete-answer scorer (CPU-verifiable).

    Strategy:
    1) One prompt-only forward (with optional patch) to obtain prompt-final logits
       for first-token scoring of every candidate.
    2) For multi-token candidates, continue with full forwards under the same
       patch context so intervention semantics match the reference scorer.

    For single-token candidates this is numerically equivalent to the full-forward
    scorer by construction (same prompt logits, same first-token log-softmax).
    For multi-token candidates, continuation uses the same full-forward path as
    the reference (no KV-cache reuse), so equivalence holds by construction as well;
    the only savings is avoiding one redundant prompt-only pass per candidate when
    all candidates share the first token (not assumed here).

    Returns logprobs with the same key set as ``score_complete_candidates``.
    """
    from src.evaluation.candidate_scorer import compute_sequence_log_likelihood_from_logits

    tokenizer = processor.tokenizer
    prompt_ids = base_inputs["input_ids"]
    prompt_len = int(prompt_ids.shape[1])
    if layer_index is not None:
        if token_position is None:
            raise ValueError("token_position required when layer_index set")
        if token_position < 0 or token_position >= prompt_len:
            raise ValueError("token_position not in original prompt range")

    def _run(ids_batch):
        with residual_patch_context(
            model,
            layer_index=layer_index if layer_index is not None else 0,
            token_position=token_position if token_position is not None else 0,
            replacement_row=replacement_row,
            no_op=(layer_index is None) or no_op,
        ) if layer_index is not None else _nullctx():
            with torch.inference_mode():
                return model(**ids_batch)

    class _nullctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    # Prompt-only forward for first-token logits
    if layer_index is None:
        with torch.inference_mode():
            prompt_out = model(**base_inputs)
        prompt_logits = prompt_out.logits
        hook_applied = 0
    else:
        with residual_patch_context(
            model,
            layer_index=layer_index,
            token_position=token_position,
            replacement_row=replacement_row,
            no_op=no_op,
        ) as ctx:
            with torch.inference_mode():
                prompt_out = model(**base_inputs)
            prompt_logits = prompt_out.logits
            hook_applied = ctx.applied

    logprobs: Dict[str, float] = {}
    for cand_str in candidate_strings:
        cand_ids = tokenizer.encode(cand_str, add_special_tokens=False)
        if len(cand_ids) == 0:
            raise ValueError("empty candidate")
        # First token from prompt-final logits
        first_lp, first_p, total_ll, _ = compute_sequence_log_likelihood_from_logits(
            prompt_logits, cand_ids[:1], prompt_len
        )
        total = total_ll
        if len(cand_ids) > 1:
            # Full forward under same patch semantics for remaining tokens
            cand_tensor = torch.tensor([cand_ids], dtype=prompt_ids.dtype, device=prompt_ids.device)
            full_ids = torch.cat([prompt_ids, cand_tensor], dim=1)
            cand_inputs = dict(base_inputs)
            cand_inputs["input_ids"] = full_ids
            if "attention_mask" in base_inputs:
                cand_inputs["attention_mask"] = torch.ones_like(full_ids)
            if layer_index is None:
                with torch.inference_mode():
                    outs = model(**cand_inputs)
                logits = outs.logits
            else:
                with residual_patch_context(
                    model,
                    layer_index=layer_index,
                    token_position=token_position,
                    replacement_row=replacement_row,
                    no_op=no_op,
                ) as ctx2:
                    with torch.inference_mode():
                        outs = model(**cand_inputs)
                    logits = outs.logits
                    hook_applied += ctx2.applied
            _, _, total_ll2, _ = compute_sequence_log_likelihood_from_logits(
                logits, cand_ids, prompt_len
            )
            total = total_ll2
        logprobs[cand_str] = total

    return {
        "logprobs": logprobs,
        "prompt_len": prompt_len,
        "hook_applied": hook_applied,
        "candidate_keys": list(logprobs.keys()),
        "scorer": "hybrid",
    }
