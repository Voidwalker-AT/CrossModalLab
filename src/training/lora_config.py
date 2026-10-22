"""Minimal LoRA configuration helpers for hardware feasibility tests (Phase 2F)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def build_lora_config(
    *,
    r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.0,
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    bias: str = "none",
):
    if r not in (1, 2, 4, 8):
        raise ValueError(f"Phase 2F restricts r to {{1,2,4,8}}, got {r}")
    if not target_modules:
        raise ValueError("target_modules must be non-empty")
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        bias=bias,
        task_type="CAUSAL_LM",
    )


def apply_lora(model: Any, config: Any):
    from peft import get_peft_model

    peft_model = get_peft_model(model, config)
    return peft_model


def count_parameters(model: Any) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


def assert_base_frozen(base_named_params: Dict[str, Any], peft_named_params: Dict[str, Any]) -> List[str]:
    """Return list of frozen base tensors that changed value (should be empty for non-LoRA)."""
    changed = []
    for name, before in base_named_params.items():
        if "lora_" in name:
            continue
        after = peft_named_params.get(name)
        if after is None:
            continue
        if after.shape != before.shape:
            continue
        if not torch_allclose_snapshot(name, before, after):
            changed.append(name)
    return changed


def torch_allclose_snapshot(name: str, a: Any, b: Any) -> bool:
    import torch

    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return True
    if a.dtype.is_floating_point is False:
        return bool(torch.equal(a, b))
    # Quantized params may be nested; skip non-dense
    try:
        return bool(torch.allclose(a.detach().float().cpu(), b.detach().float().cpu(), atol=1e-6, rtol=1e-5))
    except Exception:
        return True
