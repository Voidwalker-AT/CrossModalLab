"""CPU-safe training objective utilities for Phase 2E design validation.

No VLM training. Pure tensor math for tests and future authorized runs.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard CE. logits [B,C], targets [B] long."""
    if logits.shape[0] == 0:
        raise ValueError("empty batch")
    if not torch.isfinite(logits).all():
        raise ValueError("non-finite logits")
    return F.cross_entropy(logits, targets)


def evidence_aware_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sufficiency: torch.Tensor,
    *,
    abstain_index: int,
    margin: float = 1.0,
    mode: str = "ce",
) -> torch.Tensor:
    """sufficiency: 1 = evidence complete (definite), 0 = insufficient (should abstain).

    mode='ce': upweight abstain class CE when sufficiency==0 (soft mask).
    mode='margin': hinge forcing z_abstain - max(z_other) >= margin when sufficiency==0.
    """
    if mode == "ce":
        ce = F.cross_entropy(logits, targets, reduction="none")
        # When insufficient and target is not abstain, emphasize fixing to abstain
        weight = torch.where(
            sufficiency == 0,
            torch.where(targets == abstain_index, 1.0, 3.0),
            torch.ones_like(ce),
        )
        return (ce * weight).mean()
    if mode == "margin":
        z_abs = logits[:, abstain_index]
        other = logits.clone()
        other[:, abstain_index] = -1e9
        z_other = other.max(dim=1).values
        hinge = F.relu(margin + z_other - z_abs)
        # only when insufficient
        mask = (sufficiency == 0).float()
        if float(mask.sum()) == 0:
            return logits.sum() * 0.0
        return (hinge * mask).sum() / mask.sum()
    raise ValueError(mode)


def counterfactual_consistency_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Symmetric CE toward each member's own label (pairwise consistency regularizer)."""
    if logits_a.shape != logits_b.shape:
        raise ValueError("pair shape mismatch")
    if not torch.isfinite(logits_a).all() or not torch.isfinite(logits_b).all():
        raise ValueError("non-finite pair logits")
    loss_a = F.cross_entropy(logits_a / temperature, target_a)
    loss_b = F.cross_entropy(logits_b / temperature, target_b)
    return 0.5 * (loss_a + loss_b)


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sufficiency: torch.Tensor,
    *,
    abstain_index: int,
    logits_a: torch.Tensor | None = None,
    logits_b: torch.Tensor | None = None,
    target_a: torch.Tensor | None = None,
    target_b: torch.Tensor | None = None,
    lambda_cf: float = 0.0,
    lambda_ea: float = 0.25,
    ce_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    ce = cross_entropy_loss(logits, targets)
    ea = evidence_aware_loss(logits, targets, sufficiency, abstain_index=abstain_index)
    if lambda_cf > 0:
        if logits_a is None:
            raise ValueError("lambda_cf>0 requires pair tensors")
        cf = counterfactual_consistency_loss(logits_a, logits_b, target_a, target_b)
    else:
        cf = logits.sum() * 0.0
    total = ce_weight * ce + lambda_ea * ea + lambda_cf * cf
    # stability
    if not torch.isfinite(total):
        raise ValueError("non-finite total loss")
    return {"total": total, "ce": ce, "ea": ea, "cf": cf}
