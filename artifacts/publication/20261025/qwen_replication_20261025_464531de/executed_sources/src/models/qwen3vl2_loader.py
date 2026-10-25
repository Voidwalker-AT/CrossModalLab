"""Pinned Qwen3-VL-2B loader with NF4 quantization (no BF16 full load)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from src.config.paths import get_hf_home, get_smolvlm2_snapshot_dir  # noqa: F401  keep paths module warm
from src.config.experiment import require_gpu_capabilities

QWEN_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
QWEN_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
QWEN_IMAGE_TOKEN_ID = 151655


def load_qwen3vl2_nf4(*, local_files_only: bool = True) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load Qwen3-VL-2B with 4-bit NF4 on CUDA. Raises if hardware insufficient."""
    import time

    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    require_gpu_capabilities()

    # Resolve snapshot path
    from pathlib import Path

    hf_home = get_hf_home()
    repo_name = f"models--{QWEN_MODEL_ID.replace('/', '--')}"
    candidates = [
        hf_home / repo_name / "snapshots" / QWEN_REVISION,
        hf_home / "hub" / repo_name / "snapshots" / QWEN_REVISION,
    ]
    snapshot = next((c for c in candidates if c.is_dir()), None)
    if snapshot is None:
        raise FileNotFoundError(
            f"Qwen snapshot not found for {QWEN_MODEL_ID}@{QWEN_REVISION}. "
            "Run scripts/acquire_qwen3vl2.py first."
        )

    # Prefer native Qwen3VL class if available
    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    except Exception:
        from transformers import AutoModelForImageTextToText as ModelCls

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=local_files_only)
    model = ModelCls.from_pretrained(
        str(snapshot),
        quantization_config=bnb,
        device_map="cuda:0",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    model.eval()
    load_sec = time.perf_counter() - t0

    # Inspect language layers
    text_model = None
    for attr in ("model", "language_model", "llm"):
        if hasattr(model, attr):
            inner = getattr(model, attr)
            if hasattr(inner, "layers"):
                text_model = inner
                break
            if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
                text_model = inner.language_model
                break
    n_layers = len(text_model.layers) if text_model is not None else None
    hidden = None
    if text_model is not None and hasattr(text_model, "config"):
        hidden = getattr(text_model.config, "hidden_size", None)

    meta = {
        "model_id": QWEN_MODEL_ID,
        "revision": QWEN_REVISION,
        "snapshot_dir": str(snapshot),
        "load_sec": round(load_sec, 2),
        "device": "cuda:0",
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
        },
        "attn_implementation": "sdpa",
        "torch_allocated_mib": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2),
        "torch_reserved_mib": round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2),
        "text_num_layers": n_layers,
        "text_hidden_size": hidden,
        "image_token_id": getattr(model.config, "image_token_id", QWEN_IMAGE_TOKEN_ID),
        "model_class": type(model).__name__,
    }
    return processor, model, meta
