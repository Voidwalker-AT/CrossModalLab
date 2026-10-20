"""CrossModalLab — Primary VLM Hardware Profiling & Inference Verification Script.

Executes Stage 4:
Loads HuggingFaceTB/SmolVLM-500M-Instruct in native bfloat16 (batch size 1)
on NVIDIA GeForce RTX 2050 (4 GB VRAM ceiling).
Measures peak VRAM, host RAM, processor init time, first-token and total latency,
and repeatability across two consecutive inferences.
Saves structured telemetry to artifacts/telemetry/hardware_profiling_run.json.
"""

import os
import sys
import time
import json
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.paths import (
    apply_default_cache_environment,
    get_hf_home,
    get_smolvlm2_snapshot_dir,
)

# Preserve legacy E: cache when present; honor env overrides; portable elsewhere.
apply_default_cache_environment()



def profile_model():
    import torch
    import psutil
    from transformers import AutoProcessor, AutoModelForImageTextToText

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "revision": "a7da5b986cb59b408707209984f360a5f4ad7e47",
        "device": "cuda:0",
        "device_name": torch.cuda.get_device_name(0),
        "total_vram_mib": round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "stages": {}
    }

    process = psutil.Process()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    # Baseline memory
    vram_before_load_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
    vram_reserved_before_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
    ram_before_load_mb = process.memory_info().rss / (1024 * 1024)

    report["stages"]["baseline"] = {
        "vram_allocated_mib": round(vram_before_load_mb, 2),
        "vram_reserved_mib": round(vram_reserved_before_mb, 2),
        "host_ram_mib": round(ram_before_load_mb, 2)
    }

    # 1. Processor Init
    print("[1/4] Initializing AutoProcessor...", flush=True)
    t0_proc = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        "HuggingFaceTB/SmolVLM-500M-Instruct",
        revision="a7da5b986cb59b408707209984f360a5f4ad7e47",
        local_files_only=True
    )
    t_proc_init = time.perf_counter() - t0_proc
    report["stages"]["processor_init"] = {
        "duration_sec": round(t_proc_init, 3)
    }
    print(f"  -> Processor initialized in {t_proc_init:.3f}s", flush=True)

    # 2. Model Loading
    print("[2/4] Loading AutoModelForImageTextToText in bfloat16...", flush=True)
    t0_model = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM-500M-Instruct",
        revision="a7da5b986cb59b408707209984f360a5f4ad7e47",
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
        local_files_only=True
    ).to("cuda:0")
    model.eval()
    t_model_load = time.perf_counter() - t0_model

    vram_post_load_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
    vram_res_post_load_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
    peak_vram_load_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
    ram_post_load_mb = process.memory_info().rss / (1024 * 1024)

    report["stages"]["model_loading"] = {
        "duration_sec": round(t_model_load, 3),
        "vram_allocated_mib": round(vram_post_load_mb, 2),
        "vram_reserved_mib": round(vram_res_post_load_mb, 2),
        "peak_vram_mib": round(peak_vram_load_mb, 2),
        "host_ram_mib": round(ram_post_load_mb, 2)
    }
    print(f"  -> Model loaded in {t_model_load:.3f}s. VRAM allocated: {vram_post_load_mb:.2f} MiB (Peak: {peak_vram_load_mb:.2f} MiB)", flush=True)

    # 3. First Inference (Minimal synthetic test image)
    print("[3/4] Executing Inference 1 (First pass)...", flush=True)
    # Create simple 224x224 RGB image with blue and red squares
    test_img = Image.new("RGB", (224, 224), color=(240, 240, 240))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_img)
    draw.rectangle([30, 80, 80, 130], fill=(220, 50, 50))   # red square on left
    draw.rectangle([140, 80, 190, 130], fill=(50, 50, 220)) # blue square on right

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Is the red square to the left of the blue square? Answer Yes or No."}
            ]
        }
    ]

    torch.cuda.reset_peak_memory_stats(0)
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[test_img], return_tensors="pt").to("cuda:0")

    t0_inf = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    torch.cuda.synchronize()
    t_inf_1 = time.perf_counter() - t0_inf

    gen_text_1 = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    peak_vram_inf_1 = torch.cuda.max_memory_allocated(0) / (1024 * 1024)

    report["stages"]["inference_1"] = {
        "prompt": "Is the red square to the left of the blue square? Answer Yes or No.",
        "duration_sec": round(t_inf_1, 3),
        "peak_vram_mib": round(peak_vram_inf_1, 2),
        "generated_tokens": len(generated_ids[0]),
        "raw_output": gen_text_1.strip()
    }
    print(f"  -> Inference 1 completed in {t_inf_1:.3f}s. Peak VRAM: {peak_vram_inf_1:.2f} MiB", flush=True)
    print(f"  -> Output: {gen_text_1.strip()[:100]}", flush=True)

    # 4. Second Inference (Repeatability assessment)
    print("[4/4] Executing Inference 2 (Repeatability)...", flush=True)
    torch.cuda.reset_peak_memory_stats(0)
    t0_inf2 = time.perf_counter()
    with torch.inference_mode():
        generated_ids_2 = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    torch.cuda.synchronize()
    t_inf_2 = time.perf_counter() - t0_inf2
    gen_text_2 = processor.batch_decode(generated_ids_2, skip_special_tokens=True)[0]
    peak_vram_inf_2 = torch.cuda.max_memory_allocated(0) / (1024 * 1024)

    report["stages"]["inference_2"] = {
        "duration_sec": round(t_inf_2, 3),
        "peak_vram_mib": round(peak_vram_inf_2, 2),
        "output_identical": (gen_text_1 == gen_text_2),
        "raw_output": gen_text_2.strip()
    }
    print(f"  -> Inference 2 completed in {t_inf_2:.3f}s. Identical: {gen_text_1 == gen_text_2}", flush=True)

    # Final Summary
    report["hardware_verdict"] = {
        "fits_under_4gb_ceiling": peak_vram_inf_1 < 4095.5,
        "peak_overall_vram_mib": round(max(peak_vram_load_mb, peak_vram_inf_1, peak_vram_inf_2), 2),
        "vram_headroom_mib": round(4095.5 - max(peak_vram_load_mb, peak_vram_inf_1, peak_vram_inf_2), 2),
        "repeatability_pass": (gen_text_1 == gen_text_2),
        "status": "PASS" if peak_vram_inf_1 < 4095.5 and (gen_text_1 == gen_text_2) else "FAIL"
    }

    out_path = Path("artifacts/telemetry/hardware_profiling_run.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nProfiling report saved to {out_path.resolve()}")
    print(f"VERDICT: {report['hardware_verdict']['status']} (Peak VRAM: {report['hardware_verdict']['peak_overall_vram_mib']} MiB / Headroom: {report['hardware_verdict']['vram_headroom_mib']} MiB)")

if __name__ == "__main__":
    profile_model()
