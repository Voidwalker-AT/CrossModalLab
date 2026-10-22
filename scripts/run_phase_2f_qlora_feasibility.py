#!/usr/bin/env python3
"""Phase 2F — Qwen3-VL-2B QLoRA single-step hardware feasibility smoke.

Maximum: 1 text-only optimizer step + 1 multimodal optimizer step.
Not a scientific fine-tuning result. Discards adapter after checks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image

from src.causal.telemetry import reset_torch_peaks, snapshot_memory
from src.config.paths import apply_default_cache_environment, get_artifacts_dir
from src.instrumentation.capture_integrity import atomic_write_json
from src.models.qwen3vl2_loader import QWEN_REVISION, load_qwen3vl2_nf4
from src.training.lora_config import apply_lora, build_lora_config, count_parameters

apply_default_cache_environment()

SYNTHETIC_FIXTURE = True  # hardware test only — not research data


def lora_target_modules(model) -> List[str]:
    """Discover language attention projection names; never target vision tower."""
    names = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf not in {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"}:
            continue
        lname = name.lower()
        if "vision" in lname or "visual" in lname or "image" in lname:
            continue
        # Prefer language backbone path when identifiable
        if "text_model" in lname or "language_model" in lname or "llm" in lname or "layers." in lname:
            names.add(leaf)
        else:
            # still allow bare projections if not vision-related
            names.add(leaf)
    return sorted(names)


def freeze_non_lora(model: Any) -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False


def one_step(
    model,
    processor,
    inputs: Dict[str, Any],
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    stage: str,
) -> Dict[str, Any]:
    model.train()
    reset_torch_peaks()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    mem_fwd = snapshot_memory()

    optimizer.zero_grad(set_to_none=True)
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    t_fwd = time.perf_counter()
    mem_before_bwd = snapshot_memory()
    loss.backward()
    t_bwd = time.perf_counter()
    mem_after_bwd = snapshot_memory()

    # grad stats on trainable
    gnorm = 0.0
    n_with_grad = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            n_with_grad += 1
            gnorm += float(p.grad.detach().float().norm(2) ** 2)
    gnorm = gnorm ** 0.5

    t_opt0 = time.perf_counter()
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_opt = time.perf_counter() - t_opt0
    mem_after_opt = snapshot_memory()
    total = time.perf_counter() - t0

    return {
        "stage": stage,
        "loss": float(loss.detach().cpu()),
        "grad_tensors": n_with_grad,
        "grad_global_l2": gnorm,
        "duration_sec": round(total, 4),
        "forward_sec": round(t_fwd - t0, 4),
        "backward_sec": round(t_bwd - t_fwd, 4),
        "optimizer_sec": round(t_opt, 4),
        "memory_forward": mem_fwd.to_dict(),
        "memory_before_backward": mem_before_bwd.to_dict(),
        "memory_after_backward": mem_after_bwd.to_dict(),
        "memory_after_optimizer": mem_after_opt.to_dict(),
        "loss_is_finite": bool(torch.isfinite(loss.detach()).item()),
    }


def snapshot_named_float_params(model, prefix_check) -> Dict[str, torch.Tensor]:
    snap = {}
    for n, p in model.named_parameters():
        if prefix_check(n) and p.is_floating_point():
            snap[n] = p.detach().float().cpu().clone()
    return snap


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-multimodal", action="store_true")
    p.add_argument("--r", type=int, default=2)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    out_dir = args.out_dir or (get_artifacts_dir() / "training" / "phase_2f")
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "phase": "2F",
        "experiment": "qlora_single_step_feasibility",
        "scientific_claims": "NONE_hardware_feasibility_only",
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "revision": QWEN_REVISION,
        "synthetic_fixture_text_only": SYNTHETIC_FIXTURE,
    }

    # Load NF4 base
    reset_torch_peaks()
    t_load = time.perf_counter()
    processor, model, load_meta = load_qwen3vl2_nf4(local_files_only=True)
    report["load_meta"] = load_meta
    report["load_sec"] = round(time.perf_counter() - t_load, 2)

    targets = lora_target_modules(model)
    report["lora_target_modules"] = targets
    if not targets:
        raise SystemExit("No LoRA target modules found")

    # QLoRA prep: keep base quantized; enable input grads for checkpointing
    from peft import prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    report["prepared_for_kbit"] = True

    cfg = build_lora_config(r=args.r, lora_alpha=args.r * 2, target_modules=tuple(targets))
    model = apply_lora(model, cfg)
    freeze_non_lora(model)
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n
    # Required when gradient-checkpointing + LoRA
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    # Disable KV cache during training
    if hasattr(model, "config"):
        model.config.use_cache = False
    report["use_cache"] = False

    params = count_parameters(model)
    report["lora_config"] = {
        "r": args.r,
        "lora_alpha": args.r * 2,
        "target_modules": targets,
        "bias": "none",
        "vision_frozen": True,
        "projector_frozen": True,
        "optimizer": "AdamW(lr=1e-4, betas=(0.9,0.999))",
        "batch_size": 1,
        "grad_accum": 1,
        "gradient_checkpointing": True,
        "kv_cache_during_training": False,
    }
    report["parameter_counts"] = params
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    report["trainable_param_names_sample"] = trainable[:20]
    report["n_trainable_tensors"] = len(trainable)
    if len(trainable) == 0:
        raise SystemExit("No trainable parameters")

    # Freeze snapshot of a few base weights (embed_tokens if present)
    base_before = snapshot_named_float_params(
        model, lambda n: ("embed_tokens" in n or n.endswith("layers.0.self_attn.q_proj.weight") or "norm.weight" in n) and "lora_" not in n
    )
    lora_before = snapshot_named_float_params(model, lambda n: "lora_A" in n or "lora_B" in n)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)

    try:
        model.gradient_checkpointing_enable()
        report["gradient_checkpointing_enabled"] = True
    except Exception as exc:
        report["gradient_checkpointing_enabled"] = False
        report["gradient_checkpointing_error"] = str(exc)

    # ---- TEXT-ONLY single step (synthetic fixture) ----
    text = (
        "The cyan box is to the left of the green cylinder.\n"
        "Question: is cyan box to the left of green cylinder?\n"
        "Answer with Yes, No, or Undetermined."
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    enc = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to("cuda:0")
    # labels: supervise only the answer continuation tokens (simple: last 4 positions)
    input_ids = enc["input_ids"]
    labels_t = input_ids.clone()
    labels_t[:, :-4] = -100  # mask prompt; short synthetic span

    report["text_only"] = one_step(model, processor, enc, labels_t, optimizer, stage="text_only")

    # adapter weights changed?
    lora_after = snapshot_named_float_params(model, lambda n: "lora_A" in n or "lora_B" in n)
    lora_changed = []
    for k, v0 in lora_before.items():
        v1 = lora_after.get(k)
        if v1 is None:
            continue
        if not torch.equal(v0, v1):
            lora_changed.append(k)
    report["text_only"]["lora_tensors_changed"] = len(lora_changed)
    report["text_only"]["lora_tensors_total"] = len(lora_before)
    report["text_only"]["adapter_weights_changed"] = len(lora_changed) > 0

    # frozen base unchanged?
    base_after = snapshot_named_float_params(
        model, lambda n: ("embed_tokens" in n or n.endswith("layers.0.self_attn.q_proj.weight") or "norm.weight" in n) and "lora_" not in n
    )
    base_changed = []
    for k, v0 in base_before.items():
        v1 = base_after.get(k)
        if v1 is None:
            continue
        if not torch.allclose(v0, v1, atol=1e-6, rtol=0):
            base_changed.append(k)
    report["text_only"]["frozen_base_sample_changed"] = base_changed
    report["text_only"]["frozen_base_ok"] = len(base_changed) == 0
    optimizer.zero_grad(set_to_none=True)

    # ---- MULTIMODAL single step ----
    if not args.skip_multimodal:
        try:
            img_path = REPO_ROOT / "datasets/controlled/images/ctrl_forward_chain_0001.png"
            image = Image.open(img_path).convert("RGB")
            messages_m = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text + " Answer: Yes"},
                    ],
                }
            ]
            enc_m = processor.apply_chat_template(
                messages_m, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            ).to("cuda:0")
            n_img = int((enc_m["input_ids"] == getattr(model.config, "image_token_id", 151655)).sum())
            report["multimodal_meta"] = {
                "image_file": str(img_path.relative_to(REPO_ROOT)),
                "n_image_tokens": n_img,
                "seq_len": int(enc_m["input_ids"].shape[1]),
                "genuinely_multimodal": n_img > 0 and "pixel_values" in enc_m,
            }
            if n_img == 0 or "pixel_values" not in enc_m:
                raise RuntimeError("image not actually in model inputs")
            labels_m = enc_m["input_ids"].clone()
            labels_m[:, :-4] = -100
            report["multimodal"] = one_step(
                model, processor, enc_m, labels_m, optimizer, stage="multimodal"
            )
            report["multimodal"]["n_image_tokens"] = n_img
        except Exception as exc:
            report["multimodal"] = {
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            }
            if "out of memory" in str(exc).lower() or "OutOfMemory" in type(exc).__name__:
                report["multimodal"]["oom"] = True
                torch.cuda.empty_cache()

    # Final memory
    report["final_memory"] = snapshot_memory().to_dict()
    report["status"] = "OK" if report["text_only"].get("adapter_weights_changed") else "FAIL"

    # Quarantine: do not save adapter
    report["adapter_saved"] = False
    report["adapter_policy"] = "one-step adapter discarded; not a research model"

    del model, processor, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    atomic_write_json(out_dir / "qlora_feasibility_report_v1.json", report)
    print(json.dumps({
        "status": report["status"],
        "trainable": params,
        "text_loss": report["text_only"].get("loss"),
        "text_adapter_changed": report["text_only"].get("adapter_weights_changed"),
        "text_frozen_ok": report["text_only"].get("frozen_base_ok"),
        "text_peak_torch_alloc": report["text_only"].get("memory_after_backward", {}).get("torch_max_allocated_mib"),
        "mm": {k: report.get("multimodal", {}).get(k) for k in ("status", "loss", "adapter_weights_changed", "oom", "error", "n_image_tokens")},
        "out": str(out_dir / "qlora_feasibility_report_v1.json"),
    }, indent=2))
    return 0 if report["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
