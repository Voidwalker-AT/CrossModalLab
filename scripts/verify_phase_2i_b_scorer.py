#!/usr/bin/env python3
"""Verify local Qwen metadata, or bounded LoRA gradients; never optimize.

CPU mode calls the installed Qwen positional methods with real processor data,
without loading model weights. It does not produce model likelihoods.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import MethodType
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.paths import apply_default_cache_environment, get_hf_home


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="One example, backward only; requires 4 GiB free RAM")
    args = parser.parse_args()
    apply_default_cache_environment()
    os.environ["HF_HUB_OFFLINE"] = "1"
    import psutil
    import torch
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    from src.models.qwen3vl2_loader import QWEN_MODEL_ID, QWEN_REVISION, load_qwen3vl2_nf4
    from src.training.differentiable_scoring import candidate_class_loglikelihoods, encode_candidates
    from scripts.run_phase_2i_a_ce_control import attach_lora
    from src.causal.telemetry import snapshot_memory, reset_torch_peaks

    out = ROOT / "artifacts/training/phase_2i_b_recovery"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"{'gpu' if args.gpu else 'positions'}_{uuid4().hex}.json"
    report = {"mode": "gpu" if args.gpu else "cpu_positions_only", "holdout_used": False,
              "optimizer_steps": 0, "model_id": QWEN_MODEL_ID, "revision": QWEN_REVISION,
              "versions": {k: importlib.metadata.version(k) for k in
                           ("torch", "torchvision", "transformers", "peft", "bitsandbytes")}}
    t0 = time.perf_counter()
    try:
        report["available_ram_gib"] = psutil.virtual_memory().available / 2**30
        if args.gpu and report["available_ram_gib"] < 4:
            raise RuntimeError("RAM_GATE_FAIL: at least 4 GiB available host RAM required before loading")
        hub = get_hf_home()
        repo = f"models--{QWEN_MODEL_ID.replace('/', '--')}"
        snapshot = next(p for p in (hub / repo / "snapshots" / QWEN_REVISION,
                                    hub / "hub" / repo / "snapshots" / QWEN_REVISION) if p.is_dir())
        config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        dataset = ROOT / "datasets/cross_modal_contrast_v2/cross_modal_contrast_v2.jsonl"
        with dataset.open(encoding="utf-8") as stream:
            rec = next(r for line in stream if (r := json.loads(line))["split"] == "train")
        report["sample_id"] = rec["sample_id"]
        report["split"] = rec["split"]
        from PIL import Image
        with Image.open(ROOT / rec["image_file"]) as image_file:
            image = image_file.convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                      {"type": "text", "text": rec["prompt"]}]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        ids = encode_candidates(processor.tokenizer)
        report["input_shapes"] = {k: list(v.shape) for k, v in inputs.items() if torch.is_tensor(v)}
        report["candidate_ids"] = ids
        report["candidate_lengths"] = [len(x) for x in ids]
        report["image_grid_thw"] = inputs["image_grid_thw"].tolist()
        report["position_ids_present"] = "position_ids" in inputs
        report["token_type_ids_present"] = "token_type_ids" in inputs
        report["config"] = {"spatial_merge_size": config.vision_config.spatial_merge_size,
                            "image_token_id": config.image_token_id,
                            "text_hidden_size": config.text_config.hidden_size}
        report["forward_shapes"] = []
        class PositionsValidated(Exception):
            pass

        class PositionOnly:
            def __init__(self):
                self.config = config
                self.rope_deltas = None
                self.get_rope_index = MethodType(Qwen3VLModel.get_rope_index, self)
                self.get_vision_position_ids = MethodType(Qwen3VLModel.get_vision_position_ids, self)

            def __call__(self, **kw):
                report["forward_shapes"].append({k: list(v.shape) for k, v in kw.items() if torch.is_tensor(v)})
                positions = Qwen3VLModel.compute_3d_position_ids(
                    self, kw["input_ids"], None, image_grid_thw=kw["image_grid_thw"],
                    attention_mask=kw["attention_mask"], mm_token_type_ids=kw["mm_token_type_ids"])
                assert positions.shape == (3, 1, kw["input_ids"].shape[1])
                raise PositionsValidated()

        for candidate in ids:
            try:
                candidate_class_loglikelihoods(PositionOnly(), inputs, [candidate], candidates=("candidate",))
            except PositionsValidated:
                pass
        report["positional_validation"] = "PASSED"
        if args.gpu:
            if psutil.virtual_memory().available < 4 * 2**30:
                raise RuntimeError("RAM_GATE_FAIL immediately before model load")
            assert torch.cuda.is_available(), "CUDA unavailable"
            x = torch.ones((16, 16), device="cuda")
            assert (x @ x).sum().item() == 4096
            del x
            free_vram, total_vram = torch.cuda.mem_get_info()
            # Bound PyTorch allocation below dedicated VRAM; fail rather than spill.
            torch.cuda.set_per_process_memory_fraction(min(0.9, free_vram / total_vram))
            report["memory_before_load"] = snapshot_memory().to_dict()
            report["host_before_load"] = psutil.virtual_memory()._asdict()
            report["swap_before_load"] = psutil.swap_memory()._asdict()
            reset_torch_peaks()
            processor, model, report["load_meta"] = load_qwen3vl2_nf4(local_files_only=True)
            model, report["lora_targets"] = attach_lora(model, r=2)
            model.train()
            # Exact hashes one tensor at a time avoid full model copies in host RAM.
            def parameter_hashes():
                return {n: hashlib.sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                        for n, p in model.named_parameters()}
            before = parameter_hashes()
            trainable = [n for n, p in model.named_parameters() if p.requires_grad]
            assert trainable and all("lora_" in n for n in trainable)
            report["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            inputs = inputs.to("cuda:0")
            ll = candidate_class_loglikelihoods(model, inputs, ids)
            assert torch.isfinite(ll).all()
            report["complete_answer_scores"] = ll.detach().cpu().tolist()
            (-ll.sum()).backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert any(g.count_nonzero().item() for g in grads)
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            report["gradient_tensors"] = len(grads)
            report["changed_parameters"] = [n for n, digest in parameter_hashes().items() if digest != before[n]]
            assert not report["changed_parameters"]
            report["memory"] = snapshot_memory().to_dict()
            report["host_after_backward"] = psutil.virtual_memory()._asdict()
            report["swap_after_backward"] = psutil.swap_memory()._asdict()
            del ll, grads, inputs, model, processor
            torch.cuda.empty_cache()
        report["status"] = "PASSED"
        code = 0
    except Exception:
        report["status"] = "FAILED"
        report["traceback"] = traceback.format_exc()
        code = 1
    report["elapsed_sec"] = time.perf_counter() - t0
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"REPORT {report_path}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
