"""Multi-layer sparse residual capture (passive; non-causal).

Records selected token rows and their indices for descriptive analysis.
Never patches activations. Mean image vectors are aggregates, not neurons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

TOKEN_SELECTORS = (
    "last",
    "first_image",
    "last_image",
    "mean_image",
    "last_text_non_image",
)


@dataclass(frozen=True)
class MultiLayerCaptureSpec:
    layer_indices: Tuple[int, ...]
    token_positions: Tuple[str, ...] = TOKEN_SELECTORS
    store_dtype: str = "float32"
    notes: str = ""


@dataclass
class TokenActivation:
    selector: str
    token_index: Optional[int]  # None for aggregates like mean_image
    vector: List[float]  # CPU float32 list for JSON
    native_dtype: str
    l2_norm: float
    mean: float
    std: float
    finite: bool

    def stats_only(self) -> Dict[str, Any]:
        return {
            "selector": self.selector,
            "token_index": self.token_index,
            "native_dtype": self.native_dtype,
            "shape": [len(self.vector)],
            "l2_norm": self.l2_norm,
            "mean": self.mean,
            "std": self.std,
            "finite": self.finite,
        }


@dataclass
class LayerActivationBundle:
    layer_index: int
    native_dtype: str
    tokens: Dict[str, TokenActivation] = field(default_factory=dict)


@dataclass
class MultiLayerCaptureResult:
    sequence_length: int
    image_token_count: int
    input_condition: str  # IMAGE_MULTIMODAL | BLANK_IMAGE_MULTIMODAL | TRUE_TEXT_ONLY
    layers: Dict[int, LayerActivationBundle]
    logits_max_abs_diff: float
    logits_identical: bool
    has_nonfinite: bool
    peak_vram_mib: Optional[float]
    host_ram_used_mib: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary_dict(self, include_vectors: bool = False) -> Dict[str, Any]:
        layer_summaries = {}
        for li, bundle in self.layers.items():
            tokens = {}
            for name, tok in bundle.tokens.items():
                if include_vectors:
                    tokens[name] = {
                        **tok.stats_only(),
                        "vector": tok.vector,
                    }
                else:
                    tokens[name] = tok.stats_only()
            layer_summaries[str(li)] = {
                "native_dtype": bundle.native_dtype,
                "tokens": tokens,
            }
        return {
            "sequence_length": self.sequence_length,
            "image_token_count": self.image_token_count,
            "input_condition": self.input_condition,
            "layers": layer_summaries,
            "logits_max_abs_diff": self.logits_max_abs_diff,
            "logits_identical": self.logits_identical,
            "has_nonfinite": self.has_nonfinite,
            "peak_vram_mib": self.peak_vram_mib,
            "host_ram_used_mib": self.host_ram_used_mib,
            "metadata": self.metadata,
        }


def classify_input_condition(image_token_count: int, has_pixel_values: bool) -> str:
    if image_token_count == 0 and not has_pixel_values:
        return "TRUE_TEXT_ONLY"
    if image_token_count > 0 and has_pixel_values:
        return "IMAGE_MULTIMODAL"
    if image_token_count > 0 and not has_pixel_values:
        return "IMAGE_TOKENS_WITHOUT_PIXELS"
    return "BLANK_OR_UNKNOWN"


def _select_rows(hidden, input_ids, image_token_id: int, selectors: Sequence[str]):
    import torch

    assert hidden.dim() == 3 and hidden.shape[0] == 1
    seq = hidden[0]
    ids = input_ids[0]
    img_pos = (ids == image_token_id).nonzero(as_tuple=False).flatten()
    text_pos = (ids != image_token_id).nonzero(as_tuple=False).flatten()
    results: Dict[str, Tuple[Optional[int], Any, str]] = {}
    native = str(hidden.dtype)

    for name in selectors:
        if name == "last":
            vec = seq[-1].detach().to("cpu", torch.float32)
            results[name] = (seq.shape[0] - 1, vec, native)
        elif name == "first_image":
            if img_pos.numel() == 0:
                raise ValueError("first_image requires image tokens")
            idx = int(img_pos[0])
            results[name] = (idx, seq[idx].detach().to("cpu", torch.float32), native)
        elif name == "last_image":
            if img_pos.numel() == 0:
                raise ValueError("last_image requires image tokens")
            idx = int(img_pos[-1])
            results[name] = (idx, seq[idx].detach().to("cpu", torch.float32), native)
        elif name == "mean_image":
            if img_pos.numel() == 0:
                raise ValueError("mean_image requires image tokens")
            vec = seq[img_pos].mean(dim=0).detach().to("cpu", torch.float32)
            results[name] = (None, vec, native)
        elif name == "last_text_non_image":
            if text_pos.numel() == 0:
                raise ValueError("last_text_non_image requires non-image tokens")
            idx = int(text_pos[-1])
            results[name] = (idx, seq[idx].detach().to("cpu", torch.float32), native)
        else:
            raise ValueError(f"Unknown selector {name}")
    return results


def capture_multi_layer_passive(
    *,
    model: Any,
    inputs: Dict[str, Any],
    spec: MultiLayerCaptureSpec,
    baseline_logits: Optional[Any] = None,
    image_token_id: Optional[int] = None,
    input_condition: str = "IMAGE_MULTIMODAL",
) -> MultiLayerCaptureResult:
    """Single forward with simultaneous passive hooks on selected layers."""
    import torch
    import psutil

    if image_token_id is None:
        image_token_id = model.config.image_token_id

    captured: Dict[int, Dict[str, Tuple[Optional[int], Any, str]]] = {
        li: {} for li in spec.layer_indices
    }
    handles = []

    def make_hook(layer_idx: int):
        def _hook(module, args, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captured[layer_idx].update(
                _select_rows(hidden, inputs["input_ids"], image_token_id, spec.token_positions)
            )
            return output

        return _hook

    text_layers = model.model.text_model.layers
    for li in spec.layer_indices:
        handles.append(text_layers[li].register_forward_hook(make_hook(li)))

    proc = psutil.Process()
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)
        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :].detach().to(torch.float32)
        peak = torch.cuda.max_memory_allocated(0) / (1024 * 1024) if torch.cuda.is_available() else None
        host_ram = proc.memory_info().rss / (1024 * 1024)
    finally:
        for h in handles:
            h.remove()

    has_nonfinite = bool(torch.isnan(logits).any() or torch.isinf(logits).any())
    layers: Dict[int, LayerActivationBundle] = {}
    for li, token_map in captured.items():
        native = next(iter(token_map.values()))[2] if token_map else "unknown"
        bundle = LayerActivationBundle(layer_index=li, native_dtype=native)
        for name, (tidx, vec, nat) in token_map.items():
            finite = bool(torch.isfinite(vec).all())
            has_nonfinite = has_nonfinite or (not finite)
            bundle.tokens[name] = TokenActivation(
                selector=name,
                token_index=tidx,
                vector=[round(float(x), 6) for x in vec.tolist()],
                native_dtype=nat,
                l2_norm=round(float(torch.linalg.vector_norm(vec)), 6),
                mean=round(float(vec.mean()), 6),
                std=round(float(vec.std(unbiased=False)), 6),
                finite=finite,
            )
        layers[li] = bundle

    max_diff = 0.0
    identical = True
    if baseline_logits is not None:
        max_diff = float((logits - baseline_logits.detach().to(logits.device).to(torch.float32)).abs().max())
        identical = max_diff == 0.0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    seq_len = int(inputs["input_ids"].shape[1])
    n_img = int((inputs["input_ids"] == image_token_id).sum())

    return MultiLayerCaptureResult(
        sequence_length=seq_len,
        image_token_count=n_img,
        input_condition=input_condition,
        layers=layers,
        logits_max_abs_diff=max_diff,
        logits_identical=identical,
        has_nonfinite=has_nonfinite,
        peak_vram_mib=peak,
        host_ram_used_mib=host_ram,
        metadata={
            "layer_indices": list(spec.layer_indices),
            "token_positions": list(spec.token_positions),
            "passive_only": True,
            "causal_intervention": False,
            "notes": spec.notes,
        },
    )


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("Vectors must be non-empty and same length")
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


def l2_norm(v: Sequence[float]) -> float:
    return sum(x * x for x in v) ** 0.5


def true_text_only_inputs(processor: Any, prompt_text: str) -> Dict[str, Any]:
    """Build pure-text chat inputs with no image tokens / pixel_values."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if "pixel_values" in inputs:
        raise AssertionError("TRUE_TEXT_ONLY path unexpectedly contains pixel_values")
    return inputs
