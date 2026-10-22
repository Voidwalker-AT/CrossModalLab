"""Evidence-aware and counterfactual loss objectives (CPU, design-validated).

Not wired into the live QLoRA runner in Phase 2H. Synthetic tests only —
no claim of improved model performance.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

LABEL_TO_IDX = {"Yes": 0, "No": 1, "Undetermined": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
VALID_LABELS = frozenset(LABEL_TO_IDX)


def encode_labels(labels: Sequence[str]) -> torch.Tensor:
    if not labels:
        raise ValueError("empty label batch")
    bad = [l for l in labels if l not in VALID_LABELS]
    if bad:
        raise ValueError(f"invalid labels: {bad}")
    return torch.tensor([LABEL_TO_IDX[l] for l in labels], dtype=torch.long)


def validate_coeffs(lambda_cf: float, mu_es: float) -> None:
    if lambda_cf < 0 or mu_es < 0:
        raise ValueError("loss coefficients must be >= 0")


def standard_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Objective A: supervised CE on class logits [B, C] vs long targets [B]."""
    if logits.ndim != 2 or targets.ndim != 1:
        raise ValueError("expected logits [B,C] and targets [B]")
    if logits.shape[0] == 0:
        raise ValueError("empty batch")
    if not torch.isfinite(logits).all():
        raise ValueError("non-finite logits")
    return F.cross_entropy(logits, targets)


def counterfactual_consistency(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Objective B: mean of CE(logits_a, y) and CE(logits_b, y_cf).

    Uses actual symbolic labels for each member. Same-label pairs still train
    toward their shared label (do not force a flip). Differentiable wrt logits.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if logits_a.shape != logits_b.shape:
        raise ValueError("pair logit shape mismatch")
    if logits_a.shape[0] != targets_a.shape[0]:
        raise ValueError("batch mismatch")
    if not torch.isfinite(logits_a).all() or not torch.isfinite(logits_b).all():
        raise ValueError("non-finite pair logits")
    la = F.cross_entropy(logits_a / temperature, targets_a)
    lb = F.cross_entropy(logits_b / temperature, targets_b)
    return 0.5 * (la + lb)


def evidence_sufficiency(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    abstain_index: int = 2,
    sufficiency: torch.Tensor,
    margin: float = 1.0,
    mode: str = "margin",
) -> torch.Tensor:
    """Objective C: selective-prediction style term on 3-way logits.

    sufficiency[t] == 1 → evidence complete (Yes/No expected); prefer non-abstain.
    sufficiency[t] == 0 → insufficient (Undetermined expected); prefer abstain.

    mode='margin' uses dual hinges on raw logits (not calibrated confidence).
    mode='ce' upweights misaligned CE with fixed factor 3.0.
    """
    if logits.shape[0] != sufficiency.shape[0]:
        raise ValueError("sufficiency batch mismatch")
    if mode not in {"margin", "ce"}:
        raise ValueError(mode)
    if margin < 0:
        raise ValueError("margin must be >= 0")
    if not torch.isfinite(logits).all():
        raise ValueError("non-finite logits")

    z_abs = logits[:, abstain_index]
    other = logits.clone()
    other[:, abstain_index] = -torch.finfo(logits.dtype).max
    z_other = other.max(dim=1).values

    if mode == "margin":
        h_insuff = F.relu(margin + z_other - z_abs)
        h_suff = F.relu(margin + z_abs - z_other)
        return torch.where(sufficiency == 0, h_insuff, h_suff).mean()

    ce = F.cross_entropy(logits, targets, reduction="none")
    w = torch.where(
        sufficiency == 0,
        torch.where(targets == abstain_index, 1.0, 3.0),
        torch.where(targets == abstain_index, 3.0, 1.0),
    )
    return (ce * w).mean()


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    logits_cf: Optional[torch.Tensor] = None,
    targets_cf: Optional[torch.Tensor] = None,
    sufficiency: Optional[torch.Tensor] = None,
    lambda_cf: float = 0.0,
    mu_es: float = 0.0,
    margin: float = 1.0,
    ce_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Objective D: L = ce_weight·CE + λ·CF + μ·ES (λ, μ ≥ 0)."""
    validate_coeffs(lambda_cf, mu_es)
    ce = standard_ce(logits, targets)
    zero = logits.sum() * 0.0
    cf = zero
    es = zero
    if lambda_cf > 0:
        if logits_cf is None or targets_cf is None:
            raise ValueError("lambda_cf>0 requires counterfactual tensors")
        cf = counterfactual_consistency(logits, logits_cf, targets, targets_cf)
    if mu_es > 0:
        if sufficiency is None:
            raise ValueError("mu_es>0 requires sufficiency flags")
        es = evidence_sufficiency(logits, targets, sufficiency=sufficiency, margin=margin)
    total = ce_weight * ce + lambda_cf * cf + mu_es * es
    if not torch.isfinite(total):
        raise ValueError("non-finite total loss")
    return {"total": total, "ce": ce, "cf": cf, "es": es}
