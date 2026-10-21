#!/usr/bin/env python3
"""Phase 0E passive activation-capture smoke test (single layer, batch size 1).

Does NOT patch activations, train probes, or cache full layers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image

from src.config.experiment import get_experiment_config
from src.config.paths import apply_default_cache_environment, get_artifacts_dir
from src.evaluation.behavioral_metrics import build_prompt
from src.instrumentation.passive_capture import CaptureSpec, capture_passive_residual
from src.models.model_loader import load_pinned_smolvlm2

apply_default_cache_environment()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=12)
    p.add_argument(
        "--sample-id",
        default="p0d2_ctrl_forward_chain_0019_forward_chain",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    args = p.parse_args()

    cfg = get_experiment_config()
    man = json.loads((REPO_ROOT / "configs/phase_0d2/behavioral_manifest_v1.json").read_text(encoding="utf-8"))
    sample = next(s for s in man["samples"] if s["sample_id"] == args.sample_id)
    img_path = REPO_ROOT / sample["image_file"]
    image = Image.open(img_path).convert("RGB")

    processor, model, model_meta = load_pinned_smolvlm2(require_gpu=True)
    device = model_meta["device"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["prompt"]},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    # Baseline logits without hooks
    torch.cuda.reset_peak_memory_stats(0)
    with torch.inference_mode():
        baseline = model(**inputs).logits[0, -1, :].detach().to(torch.float32)
    base_peak = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
    torch.cuda.empty_cache()

    spec = CaptureSpec(
        layer_index=args.layer,
        token_positions=("last", "first_image", "last_image", "mean_image", "last_text_non_image"),
        notes="Phase 0E passive smoke; residual stream after decoder layer.",
    )
    result = capture_passive_residual(
        model=model,
        processor=processor,
        inputs=inputs,
        spec=spec,
        baseline_logits=baseline,
        image_token_id=model.config.image_token_id,
    )

    # Memory estimate for full residual if stored for this sequence
    t = int(inputs["input_ids"].shape[1])
    h = int(model.model.text_model.config.hidden_size)
    full_fp32_bytes = t * h * 4
    selected_bytes = 0
    for shape in result.shapes.values():
        n = 1
        for dim in shape:
            n *= int(dim)
        selected_bytes += n * 4

    payload = {
        "phase": "0E",
        "test": "passive_residual_smoke",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample_id": args.sample_id,
        "model": model_meta,
        "generation_config": {
            "do_sample": cfg.generation["do_sample"],
            "max_new_tokens": 0,
            "note": "forward-only logits; no decode loop",
        },
        "sequence_length": t,
        "hidden_size": h,
        "baseline_peak_vram_mib": round(base_peak, 3),
        "capture": result.to_dict(),
        "memory_planning": {
            "full_residual_fp32_bytes": full_fp32_bytes,
            "full_residual_fp32_mib": round(full_residual_mib := full_fp32_bytes / (1024 * 1024), 3),
            "selected_positions_fp32_mib": round(selected_bytes / (1024 * 1024), 6),
            "cpu_offload": "selected rows cloned to CPU float32 inside hook; GPU tensors not retained",
        },
        "verdict": {
            "pass": (
                result.logits_identical
                and not result.has_nonfinite
                and result.native_dtype in {"torch.bfloat16", "torch.float32", "torch.float16"}
            ),
            "logits_identical": result.logits_identical,
            "has_nonfinite": result.has_nonfinite,
            "native_dtype": result.native_dtype,
        },
        "scientific_limits": [
            "Passive capture does not establish causal mediation.",
            "Activation coordinates are not assumed to be human-interpretable neurons.",
            "No probe training or patching performed in this smoke test.",
        ],
    }

    out_dir = args.out_dir or (get_artifacts_dir() / "instrumentation" / "phase_0e")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "passive_layer_smoke.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Phase 0E Passive Capture Smoke")
    print(f"Sample: {args.sample_id}")
    print(f"Layer: {args.layer} module=model.model.text_model.layers[{args.layer}]")
    print(f"Native dtype: {result.native_dtype}")
    print(f"Shapes: {result.shapes}")
    print(f"Logits identical: {result.logits_identical} (max|Δ|={result.logits_max_abs_diff})")
    print(f"Has nonfinite: {result.has_nonfinite}")
    print(f"Baseline peak VRAM: {base_peak:.1f} MiB")
    print(f"Capture peak VRAM: {result.peak_vram_mib_after:.1f} MiB")
    print(f"Full residual FP32 would be {payload['memory_planning']['full_residual_fp32_mib']:.2f} MiB")
    print(f"Selected rows FP32: {payload['memory_planning']['selected_positions_fp32_mib']:.6f} MiB")
    print(f"Output: {out_path}")
    print(f"VERDICT: {'PASS' if payload['verdict']['pass'] else 'FAIL'}")
    print("=" * 72)

    del model, processor, inputs, baseline
    torch.cuda.empty_cache()
    return 0 if payload["verdict"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
