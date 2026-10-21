"""Passive residual-stream capture hooks.

Scientific limits:
- Passive observation only; no activation replacement/patching.
- Probe decodability of captured states is NOT causal usage evidence.
- Do not equate activation coordinates with human-interpretable neurons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class CaptureSpec:
    """Specification for a single passive residual capture."""

    layer_index: int
    token_positions: Sequence[str]  # e.g. ["last", "first_image", "mean_image"]
    store_dtype: str = "float32"  # CPU storage dtype
    include_attention_output: bool = False
    include_mlp_output: bool = False
    notes: str = ""


@dataclass
class PassiveCaptureResult:
    layer_index: int
    native_dtype: str
    stored_dtype: str
    shapes: Dict[str, List[int]]
    stats: Dict[str, Dict[str, float]]
    logits_max_abs_diff: float
    logits_identical: bool
    has_nonfinite: bool
    peak_vram_mib_before: Optional[float]
    peak_vram_mib_after: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "native_dtype": self.native_dtype,
            "stored_dtype": self.stored_dtype,
            "shapes": self.shapes,
            "stats": self.stats,
            "logits_max_abs_diff": self.logits_max_abs_diff,
            "logits_identical": self.logits_identical,
            "has_nonfinite": self.has_nonfinite,
            "peak_vram_mib_before": self.peak_vram_mib_before,
            "peak_vram_mib_after": self.peak_vram_mib_after,
            "metadata": self.metadata,
        }


def _resolve_positions(
    hidden: "Any",
    input_ids: "Any",
    image_token_id: int,
    requested: Sequence[str],
) -> Dict[str, "Any"]:
    """Select token positions from hidden [1, T, H]. Returns CPU float tensors."""
    assert hidden.dim() == 3 and hidden.shape[0] == 1
    seq = hidden[0]  # [T, H]
    ids = input_ids[0]
    out: Dict[str, torch.Tensor] = {}
    t = seq.shape[0]
    for name in requested:
        if name == "last":
            out[name] = seq[-1].detach().to("cpu", torch.float32)
        elif name == "first":
            out[name] = seq[0].detach().to("cpu", torch.float32)
        elif name == "last_image":
            img_pos = (ids == image_token_id).nonzero(as_tuple=False)
            if img_pos.numel() == 0:
                raise ValueError("No image tokens found for last_image position")
            out[name] = seq[int(img_pos[-1])].detach().to("cpu", torch.float32)
        elif name == "first_image":
            img_pos = (ids == image_token_id).nonzero(as_tuple=False)
            if img_pos.numel() == 0:
                raise ValueError("No image tokens found for first_image position")
            out[name] = seq[int(img_pos[0])].detach().to("cpu", torch.float32)
        elif name == "mean_image":
            img_pos = (ids == image_token_id).nonzero(as_tuple=False).flatten()
            if img_pos.numel() == 0:
                raise ValueError("No image tokens found for mean_image position")
            out[name] = seq[img_pos].mean(dim=0).detach().to("cpu", torch.float32)
        elif name == "last_text_non_image":
            text_pos = (ids != image_token_id).nonzero(as_tuple=False).flatten()
            if text_pos.numel() == 0:
                raise ValueError("No non-image tokens found")
            out[name] = seq[int(text_pos[-1])].detach().to("cpu", torch.float32)
        else:
            raise ValueError(f"Unknown token position selector: {name}")
        if t <= 0:
            raise ValueError("Empty sequence")
    return out


def capture_passive_residual(
    *,
    model: Any,
    processor: Any,
    inputs: Dict[str, Any],
    spec: CaptureSpec,
    baseline_logits: Optional[torch.Tensor] = None,
    image_token_id: Optional[int] = None,
) -> PassiveCaptureResult:
    """Run one forward pass with a passive residual hook at ``spec.layer_index``.

    The hook only reads and clones selected rows; it does not modify the
    residual stream. Logits are compared to ``baseline_logits`` when provided.
    """
    import torch

    if image_token_id is None:
        image_token_id = model.config.image_token_id

    layer = model.model.text_model.layers[spec.layer_index]
    captured: Dict[str, torch.Tensor] = {}
    native_dtype_holder = {"dtype": None}

    def _hook(module, args, output):
        # output may be tensor or tuple(hidden, ...)
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        native_dtype_holder["dtype"] = str(hidden.dtype)
        captured.update(
            _resolve_positions(hidden, inputs["input_ids"], image_token_id, spec.token_positions)
        )
        return output  # passive: unchanged

    handle = layer.register_forward_hook(_hook)
    peak_before = torch.cuda.max_memory_allocated(0) / (1024 * 1024) if torch.cuda.is_available() else None
    try:
        torch.cuda.reset_peak_memory_stats(0)
        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :].detach().to(torch.float32)
    finally:
        handle.remove()

    peak_after = torch.cuda.max_memory_allocated(0) / (1024 * 1024) if torch.cuda.is_available() else None

    has_nonfinite = False
    shapes: Dict[str, List[int]] = {}
    stats: Dict[str, Dict[str, float]] = {}
    for k, v in captured.items():
        shapes[k] = list(v.shape)
        has_nonfinite = has_nonfinite or bool(torch.isnan(v).any() or torch.isinf(v).any())
        stats[k] = {
            "mean": float(v.mean()),
            "std": float(v.std(unbiased=False)) if v.numel() > 1 else 0.0,
            "min": float(v.min()),
            "max": float(v.max()),
            "l2_norm": float(torch.linalg.vector_norm(v)),
        }
    has_nonfinite = has_nonfinite or bool(torch.isnan(logits).any() or torch.isinf(logits).any())

    max_diff = 0.0
    identical = True
    if baseline_logits is not None:
        diff = (logits - baseline_logits.detach().to(logits.device).to(torch.float32)).abs().max()
        max_diff = float(diff)
        identical = max_diff == 0.0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return PassiveCaptureResult(
        layer_index=spec.layer_index,
        native_dtype=native_dtype_holder["dtype"] or "unknown",
        stored_dtype=spec.store_dtype,
        shapes=shapes,
        stats=stats,
        logits_max_abs_diff=max_diff,
        logits_identical=identical,
        has_nonfinite=has_nonfinite,
        peak_vram_mib_before=peak_before,
        peak_vram_mib_after=peak_after,
        metadata={
            "token_positions": list(spec.token_positions),
            "passive_only": True,
            "causal_intervention": False,
            "notes": spec.notes,
        },
    )
