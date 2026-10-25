"""Differentiable complete-candidate sequence scoring for evidence-aware losses."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

CANDIDATES = (" Yes", " No", " Undetermined")
ABSTAIN_INDEX = 2


def encode_candidates(tokenizer, candidates: Sequence[str] = CANDIDATES) -> List[List[int]]:
    return [tokenizer.encode(c, add_special_tokens=False) for c in candidates]


def candidate_class_loglikelihoods(
    model: Any,
    base_inputs: Dict[str, Any],
    candidate_ids: Sequence[Sequence[int]],
    *,
    candidates: Sequence[str] = CANDIDATES,
) -> torch.Tensor:
    """Return differentiable class log-likelihoods [B, C] (summed over candidate tokens).

    Pre-registered rule: **summed** sequence log-likelihood (not length-normalized).
    Gradients flow through model forward; no detach/NumPy before return.
    """
    prompt_ids = base_inputs.get("input_ids")
    if not torch.is_tensor(prompt_ids) or prompt_ids.dim() != 2:
        raise ValueError("base_inputs.input_ids must be [B,T]")
    if prompt_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must contain integer token IDs")
    if not candidates or len(candidate_ids) != len(candidates):
        raise ValueError("candidate_ids/candidates length mismatch")

    prompt_len = prompt_ids.shape[1]
    batch = prompt_ids.shape[0]
    if batch != 1 or prompt_len == 0:
        raise ValueError("Phase 2I-B scoring requires one nonempty prompt")
    for key in ("past_key_values", "inputs_embeds", "labels", "cache_position"):
        if base_inputs.get(key) is not None:
            raise ValueError(f"prompt-only scoring does not accept {key}")
    if base_inputs.get("logits_to_keep", 0) != 0:
        raise ValueError("scoring requires all sequence logits")
    for key in ("attention_mask", "token_type_ids", "mm_token_type_ids"):
        value = base_inputs.get(key)
        if value is not None and (not torch.is_tensor(value) or value.shape != prompt_ids.shape):
            raise ValueError(f"{key} must match input_ids [B,T]")
    mask = base_inputs.get("attention_mask")
    if mask is not None:
        if not ((mask == 0) | (mask == 1)).all() or mask[0, -1] != 1:
            raise ValueError("attention_mask must be binary with an unmasked final prompt token")
    positions = base_inputs.get("position_ids")
    if positions is not None and (
        not torch.is_tensor(positions)
        or positions.ndim not in (2, 3)
        or positions.shape[-2:] != prompt_ids.shape
    ):
        raise ValueError("position_ids must end in [B,T]")
    multimodal_rope = base_inputs.get("mm_token_type_ids") is not None and any(
        base_inputs.get(key) is not None for key in ("image_grid_thw", "video_grid_thw")
    )
    # Validate every continuation before any forward (no partially scored invalid batch).
    for ids in candidate_ids:
        if len(ids) == 0 or any(not isinstance(t, Integral) or isinstance(t, bool) or t < 0 for t in ids):
            raise ValueError("candidates must be nonempty sequences of nonnegative integer IDs")

    scores = []
    for ids in candidate_ids:
        cand_t = torch.tensor([ids], dtype=prompt_ids.dtype, device=prompt_ids.device)
        full_ids = torch.cat([prompt_ids, cand_t], dim=1)
        inputs = dict(base_inputs)
        inputs["input_ids"] = full_ids
        for key in ("attention_mask", "token_type_ids", "mm_token_type_ids"):
            value = base_inputs.get(key)
            if value is not None:
                # Qwen modality IDs: text=0, image=1, video=2. Candidates are text.
                fill = 1 if key == "attention_mask" else 0
                suffix = value.new_full((batch, len(ids)), fill)
                inputs[key] = torch.cat([value, suffix], dim=-1)
        if multimodal_rope:
            # Recompute M-RoPE from the complete modality sequence and unchanged grids.
            inputs.pop("position_ids", None)
        elif positions is not None:
            offsets = torch.arange(1, len(ids) + 1, device=positions.device, dtype=positions.dtype)
            inputs["position_ids"] = torch.cat([positions, positions[..., -1:] + offsets], dim=-1)
        inputs["use_cache"] = False
        outputs = model(**inputs)
        logits = outputs.logits  # [B, T_full, V]
        # causal shift: token k of candidate predicted from position prompt_len-1+k
        total = torch.zeros((), dtype=torch.float32, device=logits.device)
        for k, tid in enumerate(ids):
            pos = prompt_len - 1 + k
            log_probs = F.log_softmax(logits[0, pos, :].to(torch.float32), dim=-1)
            total = total + log_probs[tid]
        scores.append(total)
        del outputs, logits

    stacked = torch.stack(scores, dim=0).unsqueeze(0)  # [1, C]
    return stacked


def evidence_sufficiency_from_class_scores(
    class_ll: torch.Tensor,
    gold_index: torch.Tensor,
    *,
    sufficiency: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """ES hinge on candidate class log-likelihoods [B,C].

    sufficiency 0: want abstain LL > others + margin (or penalize otherwise)
    sufficiency 1: want gold non-abstain LL > abstain + margin when gold != abstain
    """
    if class_ll.ndim != 2 or class_ll.shape[1] != 3:
        raise ValueError("class_ll must be [B,3]")
    if margin < 0:
        raise ValueError("margin must be >= 0")
    if not torch.isfinite(class_ll).all():
        raise ValueError("non-finite class log-likelihoods")

    B = class_ll.shape[0]
    abst = class_ll[:, ABSTAIN_INDEX]
    # for sufficient examples, target is gold_index (Yes/No), not abstain
    gold_ll = class_ll.gather(1, gold_index.view(-1, 1)).squeeze(1)
    # other max excluding abstain for insufficient preference of abstain over best definite
    definite = class_ll.clone()
    definite[:, ABSTAIN_INDEX] = -torch.inf
    best_def = definite.max(dim=1).values

    # insufficient: want abst >= best_def + margin  → hinge = relu(m + best_def - abst)
    h_ins = F.relu(margin + best_def - abst)
    # sufficient & gold is definite: want gold >= abst + margin → relu(m + abst - gold)
    h_suff = F.relu(margin + abst - gold_ll)
    # sufficient & gold is abstain (shouldn't happen for our labels): no extra penalty
    is_abst_gold = gold_index == ABSTAIN_INDEX
    hinge = torch.where(sufficiency == 0, h_ins, torch.where(is_abst_gold, torch.zeros_like(h_suff), h_suff))
    if B == 0:
        raise ValueError("empty batch")
    return hinge.mean()


def total_loss(
    ce: torch.Tensor,
    es: torch.Tensor,
    *,
    mu_es: float,
    ce_weight: float = 1.0,
) -> torch.Tensor:
    if mu_es < 0 or ce_weight < 0:
        raise ValueError("coefficients must be >= 0")
    total = ce_weight * ce + mu_es * es
    if not torch.isfinite(total):
        raise ValueError("non-finite total loss")
    return total
