#!/usr/bin/env python3
"""Phase 2C-B — Bounded Qwen3-VL-2B NF4 load + smoke tests.

No training, no causal patching, no holdout. Max 4 visual + 3 cross-modal + 1 hook.
"""

from __future__ import annotations

import argparse
import hashlib
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
from src.evaluation.behavioral_metrics import parse_model_answer
from src.instrumentation.capture_integrity import atomic_write_json
from src.models.qwen3vl2_loader import QWEN_REVISION, load_qwen3vl2_nf4

apply_default_cache_environment()

CANDIDATES = [" Yes", " No", " Undetermined"]


def build_qwen_inputs(processor, prompt: str, image: Image.Image | None, device: str = "cuda:0"):
    if image is None:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    else:
        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs.to(device)


def score_candidates_qwen(model, processor, inputs, candidates, device="cuda:0"):
    from transformers import AutoModelForImageTextToText  # noqa: F401
    from src.evaluation.candidate_scorer import score_candidate_answers

    # Reuse generic scorer if it only needs tokenizer + model forward
    return score_candidate_answers(
        model=model, processor=processor, base_inputs=inputs, candidate_strings=candidates, device=device
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--skip-hook", action="store_true")
    args = p.parse_args()

    out_dir = args.out_dir or (get_artifacts_dir() / "qwen" / "phase_2c_b")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen3-VL-2B NF4...")
    reset_torch_peaks()
    processor, model, meta = load_qwen3vl2_nf4(local_files_only=True)
    load_mem = snapshot_memory()
    print(json.dumps({**meta, "load_memory": load_mem.to_dict()}, indent=2))

    # Smoke images from controlled set
    img_dir = REPO_ROOT / "datasets" / "controlled" / "images"
    prompts_cases = [
        {
            "id": "object_identity",
            "image": img_dir / "ctrl_forward_chain_0001.png",
            "prompt": "Is there a colored geometric shape in the image? Answer with Yes or No.",
            "expected": None,
        },
        {
            "id": "left_of",
            "image": img_dir / "ctrl_forward_chain_0001.png",
            "prompt": "The cyan box is to the left of the green cylinder. Based on the image, is the cyan box to the left of the green cylinder? Answer with: Yes, No, or Undetermined.",
            "expected": "Yes",
        },
        {
            "id": "right_of_reversed",
            "image": img_dir / "ctrl_forward_chain_0001.png",
            "prompt": "Based on the image, is the green cylinder to the left of the cyan box? Answer with: Yes, No, or Undetermined.",
            "expected": "No",
        },
        {
            "id": "missing_visual_text_only",
            "image": None,
            "prompt": "There is no image provided. Is the cyan box to the left of the green cylinder? Answer with: Yes, No, or Undetermined.",
            "expected": "Undetermined",
        },
    ]

    visual_records = []
    for case in prompts_cases[:4]:
        print(f"\n=== VISUAL {case['id']} ===", flush=True)
        img = Image.open(case["image"]).convert("RGB") if case["image"] else None
        img_sha = None
        if case["image"] and case["image"].is_file():
            img_sha = hashlib.sha256(case["image"].read_bytes()).hexdigest()
        reset_torch_peaks()
        inputs = build_qwen_inputs(processor, case["prompt"], img)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        lat = time.perf_counter() - t0
        new_ids = out[0, inputs["input_ids"].shape[1] :].tolist()
        raw = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        parsed = parse_model_answer(raw)
        n_img_tok = int((inputs["input_ids"] == meta["image_token_id"]).sum()) if img is not None else 0
        mem = snapshot_memory()
        rec = {
            "id": case["id"],
            "image_sha256": img_sha,
            "prompt": case["prompt"],
            "raw_completion": raw,
            "parsed_answer": parsed["parsed_label"],
            "expected_loose": case["expected"],
            "seq_len": int(inputs["input_ids"].shape[1]),
            "n_image_tokens": n_img_tok,
            "latency_sec": round(lat, 4),
            "memory": mem.to_dict(),
            "revision": QWEN_REVISION,
        }
        visual_records.append(rec)
        print(f"  raw={raw!r} parsed={parsed['parsed_label']} lat={lat:.2f}s img_tok={n_img_tok}", flush=True)
        del inputs, out
        torch.cuda.empty_cache()

    # Cross-modal development examples (from contrast v2 train split - not holdout)
    cross_records = []
    ds = REPO_ROOT / "datasets" / "cross_modal_contrast_v2" / "cross_modal_contrast_v2.jsonl"
    wanted = {"entailed", "contradicted", "underdetermined"}
    chosen = {}
    for line in ds.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["split"] != "train":
            continue
        if r["variant_id"] in wanted and r["variant_id"] not in chosen:
            chosen[r["variant_id"]] = r
        if len(chosen) == 3:
            break

    for var, rec in chosen.items():
        print(f"\n=== CROSS-MODAL {var} ===", flush=True)
        img_path = REPO_ROOT / rec["image_file"]
        img = Image.open(img_path).convert("RGB")
        reset_torch_peaks()
        inputs = build_qwen_inputs(processor, rec["prompt"], img)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        lat = time.perf_counter() - t0
        new_ids = out[0, inputs["input_ids"].shape[1] :].tolist()
        raw = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        parsed = parse_model_answer(raw)
        # candidate scoring
        try:
            cand = score_candidates_qwen(model, processor, inputs, CANDIDATES)
            logprobs = {k: v.total_log_likelihood for k, v in cand.candidates.items()}
        except Exception as exc:
            logprobs = {"error": str(exc)}
        mem = snapshot_memory()
        cross_records.append(
            {
                "variant_id": var,
                "sample_id": rec["sample_id"],
                "ground_truth_label": rec["ground_truth_label"],
                "raw_completion": raw,
                "parsed_answer": parsed["parsed_label"],
                "is_correct": parsed["parsed_label"] == rec["ground_truth_label"],
                "image_sha256": rec["image_sha256"],
                "prompt_sha256": rec["prompt_sha256"],
                "logprobs": logprobs,
                "latency_sec": round(lat, 4),
                "memory": mem.to_dict(),
                "n_image_tokens": int((inputs["input_ids"] == meta["image_token_id"]).sum()),
            }
        )
        print(f"  GT={rec['ground_truth_label']} pred={parsed['parsed_label']} correct={parsed['parsed_label']==rec['ground_truth_label']}", flush=True)
        del inputs, out
        torch.cuda.empty_cache()

    # Passive hook on one language layer
    hook_result = {"skipped": True}
    if not args.skip_hook and meta.get("text_num_layers"):
        try:
            layer_idx = min(12, meta["text_num_layers"] - 1)
            # locate layers
            text_model = None
            inner = getattr(model, "model", None)
            if inner is not None and hasattr(inner, "layers"):
                text_model = inner
            elif inner is not None and hasattr(inner, "language_model"):
                text_model = inner.language_model
            if text_model is None:
                hook_result = {"skipped": True, "reason": "layers_not_found"}
            else:
                captured = {}

                def _hook(module, args, output):
                    h = output[0] if isinstance(output, (tuple, list)) else output
                    captured["shape"] = list(h.shape)
                    captured["dtype"] = str(h.dtype)
                    captured["last"] = h[0, -1, :].detach().float().cpu()
                    return output

                img = Image.open(chosen["entailed"]["image_file"]).convert("RGB")
                inputs = build_qwen_inputs(processor, chosen["entailed"]["prompt"], img)
                # baseline logits
                with torch.inference_mode():
                    base = model(**inputs).logits[0, -1, :].detach().float()
                handle = text_model.layers[layer_idx].register_forward_hook(_hook)
                try:
                    reset_torch_peaks()
                    with torch.inference_mode():
                        hooked = model(**inputs).logits[0, -1, :].detach().float()
                finally:
                    handle.remove()
                max_diff = float((hooked - base).abs().max())
                finite = bool(torch.isfinite(captured["last"]).all())
                hook_result = {
                    "skipped": False,
                    "layer_index": layer_idx,
                    "module_path": f"text_model.layers[{layer_idx}]",
                    "shape": captured.get("shape"),
                    "dtype": captured.get("dtype"),
                    "logits_max_abs_diff": max_diff,
                    "finite": finite,
                    "memory": snapshot_memory().to_dict(),
                }
                del inputs
                torch.cuda.empty_cache()
        except Exception as exc:
            hook_result = {"skipped": False, "error": str(exc)}

    del model, processor
    torch.cuda.empty_cache()

    summary = {
        "phase": "2C-B",
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "revision": QWEN_REVISION,
        "load_meta": meta,
        "load_memory": load_mem.to_dict(),
        "visual_smoke": visual_records,
        "cross_modal_smoke": cross_records,
        "hook_test": hook_result,
        "final_holdout_touched": False,
        "causal_patching": False,
        "limitations": [
            "Smoke only; not general accuracy.",
            "Different tokenization from SmolVLM2; logits not cross-calibrated.",
            "NF4 load measured; not BF16.",
        ],
    }
    atomic_write_json(out_dir / "qwen_smoke_summary_v1.json", summary)
    print("\nSUMMARY", json.dumps({
        "load_torch_mib": meta["torch_allocated_mib"],
        "visual_n": len(visual_records),
        "cross_n": len(cross_records),
        "hook": {k: hook_result.get(k) for k in ("skipped", "logits_max_abs_diff", "error", "layer_index")},
    }, indent=2))
    print(f"Wrote {out_dir / 'qwen_smoke_summary_v1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
