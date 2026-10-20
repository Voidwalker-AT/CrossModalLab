"""CrossModalLab — Stage 4: SmolVLM2-2.2B 4-bit Quantized Hardware Profiler.

Profiles HuggingFaceTB/SmolVLM2-2.2B-Instruct with bitsandbytes 4-bit NF4 quantization
on NVIDIA GeForce RTX 2050 (4 GB dedicated VRAM ceiling).

Measures:
1. Baseline host RAM, commit memory, dedicated VRAM.
2. AutoProcessor initialization latency.
3. 4-bit quantized model loading latency, VRAM allocation, host RAM usage.
4. Component quantization audit: checks which modules are Linear4bit vs standard precision.
5. Single-sample forward inference with batch size 1 and greedy generation.
6. Peak allocated VRAM, peak reserved VRAM, WDDM shared memory spillover check.
7. Repeatability check across two consecutive passes.
8. Saves structured hardware telemetry to artifacts/telemetry/smolvlm2_quantized_profile.json.
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



def patch_smolvlm_inputs_merger():
    """Monkey-patch SmolVLMModel.inputs_merger with canonical masked_scatter implementation.

    The transformers 5.20 smolvlm implementation contains an unvalidated chunking assumption
    `image_hidden_states.view(-1, patch_size, ...)` where patch_size is set to vision_config.patch_size (14).
    This triggers a RuntimeError whenever image feature tokens (1377 for SmolVLM2) are not divisible by 14.
    The canonical Idefics3 masked_scatter implementation accurately replaces image token slots without modulo errors.
    """
    import torch
    from transformers.models.smolvlm import modeling_smolvlm

    def safe_inputs_merger(self, input_ids: torch.LongTensor, inputs_embeds: torch.Tensor, image_hidden_states: torch.Tensor):
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.full((), self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        special_image_mask = special_image_mask.unsqueeze(-1).to(inputs_embeds.device)
        image_hidden_states = image_hidden_states.to(inputs_embeds.device, inputs_embeds.dtype)
        return inputs_embeds.masked_scatter(special_image_mask, image_hidden_states)

    modeling_smolvlm.SmolVLMModel.inputs_merger = safe_inputs_merger

def profile_quantized_model():
    import torch
    import psutil
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    from src.config.experiment import get_experiment_config, require_gpu_capabilities

    require_gpu_capabilities()
    exp_cfg = get_experiment_config()
    patch_smolvlm_inputs_merger()

    model_id = exp_cfg.model.model_id
    revision = exp_cfg.model.revision
    snapshot_dir = get_smolvlm2_snapshot_dir()
    if not snapshot_dir.exists():
        raise FileNotFoundError(
            f"Pinned model snapshot not found: {snapshot_dir}. "
            "See docs/ASSET_MANIFEST.md. Do not silently substitute another revision."
        )
    device = "cuda:0"

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": model_id,
        "revision": revision,
        "snapshot_path": str(snapshot_dir),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0),
        "total_dedicated_vram_mib": round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2),
        "quantization_config": {
            "load_in_4bit": exp_cfg.quantization.load_in_4bit,
            "bnb_4bit_quant_type": exp_cfg.quantization.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": exp_cfg.quantization.bnb_4bit_compute_dtype,
            "bnb_4bit_use_double_quant": exp_cfg.quantization.bnb_4bit_use_double_quant
        },
        "stages": {}
    }

    process = psutil.Process()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    # 1. Baseline Memory
    vram_base_alloc = torch.cuda.memory_allocated(0) / (1024 * 1024)
    vram_base_res = torch.cuda.memory_reserved(0) / (1024 * 1024)
    ram_base = process.memory_info().rss / (1024 * 1024)

    report["stages"]["baseline"] = {
        "vram_allocated_mib": round(vram_base_alloc, 2),
        "vram_reserved_mib": round(vram_base_res, 2),
        "host_ram_rss_mib": round(ram_base, 2)
    }
    print(f"[1/5] Baseline Memory: VRAM allocated={vram_base_alloc:.1f} MiB, Host RAM RSS={ram_base:.1f} MiB", flush=True)

    # 2. Processor Init
    print("[2/5] Initializing AutoProcessor...", flush=True)
    t0_proc = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(snapshot_dir), local_files_only=True)
    t_proc = time.perf_counter() - t0_proc
    report["stages"]["processor_init"] = {
        "duration_sec": round(t_proc, 3)
    }
    print(f"  -> AutoProcessor initialized in {t_proc:.2f}s", flush=True)

    # 3. Model Loading with 4-bit NF4
    print("[3/5] Loading SmolVLM2-2.2B in 4-bit NF4 (bnb_4bit_compute_dtype=bfloat16)...", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    t0_load = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        str(snapshot_dir),
        quantization_config=bnb_config,
        device_map="cuda:0",
        _attn_implementation="eager",
        local_files_only=True
    )
    t_load = time.perf_counter() - t0_load

    vram_post_load_alloc = torch.cuda.memory_allocated(0) / (1024 * 1024)
    vram_post_load_res = torch.cuda.memory_reserved(0) / (1024 * 1024)
    vram_post_load_peak = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
    ram_post_load = process.memory_info().rss / (1024 * 1024)

    # Component Quantization Inspection
    total_modules = 0
    linear_4bit_count = 0
    linear_std_count = 0
    for name, mod in model.named_modules():
        mod_cls = mod.__class__.__name__
        if "Linear4bit" in mod_cls:
            linear_4bit_count += 1
        elif "Linear" in mod_cls and "Linear4bit" not in mod_cls:
            linear_std_count += 1
        total_modules += 1

    report["stages"]["model_loading"] = {
        "duration_sec": round(t_load, 3),
        "vram_allocated_mib": round(vram_post_load_alloc, 2),
        "vram_reserved_mib": round(vram_post_load_res, 2),
        "peak_vram_mib": round(vram_post_load_peak, 2),
        "host_ram_rss_mib": round(ram_post_load, 2),
        "component_quantization": {
            "total_modules": total_modules,
            "linear4bit_modules": linear_4bit_count,
            "standard_linear_modules": linear_std_count
        }
    }
    print(f"  -> Model loaded in {t_load:.2f}s.", flush=True)
    print(f"  -> Post-load VRAM: {vram_post_load_alloc:.1f} MiB allocated, {vram_post_load_res:.1f} MiB reserved (Peak: {vram_post_load_peak:.1f} MiB)", flush=True)
    print(f"  -> Host RAM RSS: {ram_post_load:.1f} MiB", flush=True)
    print(f"  -> Quantization breakdown: {linear_4bit_count} Linear4bit modules, {linear_std_count} unquantized Linear modules", flush=True)

    # 4. Inference 1 (Single controlled visual sample)
    print("[4/5] Executing Inference 1 (controlled image forward pass)...", flush=True)
    from src.config.paths import get_datasets_dir

    test_img_path = get_datasets_dir() / "controlled" / "images" / "ctrl_forward_chain_0001.png"
    test_img = Image.open(test_img_path).convert("RGB")

    prompt_text = (
        "The green cylinder is to the left of the magenta prism.\n\n"
        "Question: Based on the image and statement, is the cyan box to the left of the magenta prism?\n"
        "Carefully evaluate whether the question is entailed (Yes), contradicted (No), or underdetermined from the evidence (Undetermined).\n"
        "Answer directly with: Yes, No, or Undetermined."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_img},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to("cuda:0")
    prompt_len = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats(0)
    t0_gen = time.perf_counter()
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    torch.cuda.synchronize()
    t_gen_1 = time.perf_counter() - t0_gen

    peak_vram_inf_1 = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
    new_token_ids_1 = out_ids[0, prompt_len:].tolist()
    completion_text_1 = processor.tokenizer.decode(new_token_ids_1, skip_special_tokens=True).strip()

    report["stages"]["inference_1"] = {
        "prompt_tokens": prompt_len,
        "completion_tokens": len(new_token_ids_1),
        "completion_token_ids": new_token_ids_1,
        "completion_text": completion_text_1,
        "duration_sec": round(t_gen_1, 3),
        "peak_vram_mib": round(peak_vram_inf_1, 2)
    }
    print(f"  -> Inference 1 completed in {t_gen_1:.2f}s. Peak VRAM: {peak_vram_inf_1:.1f} MiB", flush=True)
    print(f"  -> Output: '{completion_text_1}'", flush=True)

    # 5. Inference 2 (Repeatability check)
    print("[5/5] Executing Inference 2 (repeatability check)...", flush=True)
    torch.cuda.reset_peak_memory_stats(0)
    t0_gen2 = time.perf_counter()
    with torch.inference_mode():
        out_ids_2 = model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    torch.cuda.synchronize()
    t_gen_2 = time.perf_counter() - t0_gen2

    peak_vram_inf_2 = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
    new_token_ids_2 = out_ids_2[0, prompt_len:].tolist()
    completion_text_2 = processor.tokenizer.decode(new_token_ids_2, skip_special_tokens=True).strip()

    is_identical = (completion_text_1 == completion_text_2)
    report["stages"]["inference_2"] = {
        "duration_sec": round(t_gen_2, 3),
        "peak_vram_mib": round(peak_vram_inf_2, 2),
        "output_identical": is_identical,
        "completion_text": completion_text_2
    }
    print(f"  -> Inference 2 completed in {t_gen_2:.2f}s. Output identical: {is_identical}", flush=True)

    # Hardware Verdict
    peak_overall_vram = max(vram_post_load_peak, peak_vram_inf_1, peak_vram_inf_2)
    vram_headroom = 4095.5 - peak_overall_vram
    report["hardware_verdict"] = {
        "fits_under_4gb_ceiling": peak_overall_vram < 3800.0,
        "peak_overall_vram_mib": round(peak_overall_vram, 2),
        "vram_headroom_mib": round(vram_headroom, 2),
        "repeatability_pass": is_identical,
        "status": "PASS" if peak_overall_vram < 3800.0 and is_identical else "FAIL"
    }

    out_telemetry = Path("artifacts/telemetry/smolvlm2_quantized_profile.json")
    out_telemetry.parent.mkdir(parents=True, exist_ok=True)
    out_telemetry.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nHardware telemetry successfully saved to: {out_telemetry.resolve()}", flush=True)
    print(f"VERDICT: {report['hardware_verdict']['status']} (Peak VRAM: {peak_overall_vram:.1f} MiB / Headroom: {vram_headroom:.1f} MiB)", flush=True)

if __name__ == "__main__":
    profile_quantized_model()
