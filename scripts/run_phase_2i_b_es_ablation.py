#!/usr/bin/env python3
"""Phase 2I-B — Matched CE+ES QLoRA ablation vs Phase 2I-A CE control.

Adds evidence-sufficiency hinge on differentiable candidate sequence
log-likelihoods. Same schedule/seed/steps as 2I-A. ES mu/margin pre-registered.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.causal.telemetry import reset_torch_peaks, snapshot_memory, should_stop_on_memory
from src.config.paths import apply_default_cache_environment, get_artifacts_dir
from src.models.qwen3vl2_loader import QWEN_REVISION, load_qwen3vl2_nf4
from src.training.answer_parser import evaluation_accounting, strict_parse_answer
from src.training.phase_2i_b_protocol import CANDIDATES, build_plan, compose_objective
from src.training.run_state import RunStore, digest, isolated_rng, write_once
from src.training.lora_config import apply_lora, build_lora_config
from src.training.sampling_and_masking import (
    build_supervised_example,
    maximize_family_coverage_schedule,
)

def attach_lora(model, r: int = 2):
    from peft import prepare_model_for_kbit_training

    names = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        lname = name.lower()
        if leaf in {"q_proj", "k_proj", "v_proj", "o_proj"} and not any(
            x in lname for x in ("vision", "visual", "image")
        ):
            names.add(leaf)
    cfg = build_lora_config(r=r, lora_alpha=r * 2, target_modules=tuple(sorted(names)))
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = apply_lora(model, cfg)
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model, sorted(names)


def eval_one(processor, model, rec, *, abstention_prompt: bool, device="cuda:0"):
    if rec.get("split") != "validation":
        raise ValueError("evaluation accepts development-validation records only")
    import torch
    from PIL import Image

    prompt = rec["prompt"]
    if abstention_prompt and "You may answer Undetermined" not in prompt:
        prompt = (
            prompt.rstrip()
            + "\nYou may answer Undetermined when evidence is insufficient."
            + "\nAnswer directly with: Yes, No, or Undetermined."
        )
    with Image.open(REPO_ROOT / rec["image_file"]) as source:
        image = source.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    lat = time.perf_counter() - t0
    raw = processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1] :].tolist(), skip_special_tokens=True)
    parsed = strict_parse_answer(raw)
    return {
        "sample_id": rec["sample_id"],
        "family_id": rec["family_id"],
        "ground_truth_label": rec["ground_truth_label"],
        "raw_completion": raw,
        "parsed_answer": parsed["parsed_label"],
        "parse_status": parsed["parse_status"],
        "is_correct": parsed["parsed_label"] == rec["ground_truth_label"],
        "latency_sec": round(lat, 4),
        "abstention_prompt": abstention_prompt,
        "prompt_sha256": rec["prompt_sha256"],
        "image_sha256": rec["image_sha256"],
    }


def build_prompt_only_inputs(processor, rec, device="cuda:0"):
    """Prompt-only inputs (assistant generation prefix) for candidate scoring."""
    if rec.get("split") != "train":
        raise ValueError("candidate training accepts train records only")
    from PIL import Image

    with Image.open(REPO_ROOT / rec["image_file"]) as source:
        image = source.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": rec["prompt"]}],
        }
    ]
    return processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(device)


def train_step_es(
    model,
    processor,
    rec,
    optimizer,
    *,
    step: int,
    cand_ids,
    es_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    if rec.get("split") != "train":
        raise ValueError("training accepts train records only")
    import torch
    model.train()
    inputs, labels, span_meta = build_supervised_example(
        processor, rec, abstention_prompt=False, device="cuda:0"
    )
    # prompt-only view for candidate scoring (same images/prompt prefix)
    prompt_inputs = build_prompt_only_inputs(processor, rec, device="cuda:0")

    reset_torch_peaks()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    # CE on gold answer span
    outputs = model(**inputs, labels=labels)
    ce = outputs.loss

    # Differentiable candidate class LLs on prompt-only prefix
    t_sc0 = time.perf_counter()
    loss, es, class_ll = compose_objective(ce, model, prompt_inputs, cand_ids, rec["ground_truth_label"], es_cfg)
    score_sec = time.perf_counter() - t_sc0

    if not (torch.isfinite(ce) and torch.isfinite(es) and torch.isfinite(loss)):
        raise RuntimeError(f"non-finite loss ce={ce} es={es} total={loss}")

    loss.backward()
    gnorm = 0.0
    n_g = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            if not torch.isfinite(p.grad).all():
                raise RuntimeError("nonfinite LoRA gradient")
            n_g += 1
            gnorm += float(p.grad.detach().double().norm(2) ** 2)
        elif not p.requires_grad and p.grad is not None:
            raise RuntimeError("unexpected frozen gradient")
    gnorm = gnorm ** 0.5
    if n_g == 0 or gnorm == 0 or not math.isfinite(gnorm):
        raise RuntimeError("no LoRA gradients")
    guard_resources()  # AdamW has not allocated its moments or updated parameters yet.
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dur = time.perf_counter() - t0
    mem = snapshot_memory()

    row = {
        "step": step,
        "sample_id": rec["sample_id"],
        "family_id": rec["family_id"],
        "label": rec["ground_truth_label"],
        "ce_loss": float(ce.detach().cpu()),
        "es_loss": float(es.detach().cpu()),
        "total_loss": float(loss.detach().cpu()),
        "grad_l2": gnorm,
        "grad_tensors": n_g,
        "n_supervised": span_meta["n_supervised"],
        "candidate_score_sec": round(score_sec, 4),
        "duration_sec": round(dur, 4),
        "memory": mem.to_dict(),
        "class_ll": [float(x) for x in class_ll.detach().cpu().reshape(-1)],
    }
    print(
        f"  step {step} fam={row['family_id']} label={row['label']} "
        f"ce={row['ce_loss']:.4f} es={row['es_loss']:.4f} tot={row['total_loss']:.4f} "
        f"t={row['duration_sec']:.2f}s",
        flush=True,
    )
    del inputs, labels, prompt_inputs, outputs, class_ll, es, loss
    torch.cuda.empty_cache()
    return row


def completed_steps(metrics_path: Path) -> int:
    """Journal diagnostic only: never used as proof of restored model state."""
    if not metrics_path.is_file():
        return 0
    rows = [json.loads(l) for l in metrics_path.read_text(encoding="utf-8").splitlines()]
    if [r["step"] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("duplicate or noncontiguous metric steps")
    return len(rows)


def guard_resources():
    verdict = should_stop_on_memory(snapshot_memory())
    if verdict["stop"]:
        raise RuntimeError(f"resource safety gate: {verdict['reasons']}")


def adapter_restorers(model, lora):
    """Restore into the existing adapter, preserving optimizer parameter bindings."""
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    cached = {}

    def validate(checkpoint):
        config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
        if (config["r"] != lora["r"] or config["lora_alpha"] != lora["lora_alpha"]
                or config["bias"] != lora["bias"]
                or sorted(config["target_modules"]) != sorted(lora["target_modules"])):
            raise ValueError("adapter configuration mismatch")
        weights = load_file(str(checkpoint / "adapter_model.safetensors"), device="cpu")
        expected = get_peft_model_state_dict(model)
        if set(weights) != set(expected):
            raise ValueError("adapter parameter keys mismatch")
        if any(weights[k].shape != expected[k].shape or weights[k].dtype != expected[k].dtype
               or not torch.isfinite(weights[k]).all() for k in weights):
            raise ValueError("adapter shape/dtype/value mismatch")
        cached.clear()
        cached.update(weights)

    def load(checkpoint):
        result = set_peft_model_state_dict(model, cached, adapter_name="default")
        if result.unexpected_keys:
            raise ValueError("unexpected adapter keys on restoration")
        after = get_peft_model_state_dict(model)
        if any(not torch.equal(after[k].detach().cpu(), value) for k, value in cached.items()):
            raise ValueError("adapter restoration verification failed")
        cached.clear()
    return validate, load


def evaluate(processor, model, plan, store, step, *, baseline=False):
    path = store.path / ("baseline.json" if baseline else f"evaluation_step_{step:04d}.json")
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if (previous["run_id"] != store.run_id or previous["identity_hash"] != store.identity_hash
                or previous["step"] != step):
            raise ValueError("evaluation identity mismatch")
        return previous
    def pack(rows):
        return {"metrics": evaluation_accounting([r["ground_truth_label"] for r in rows],
                                                 [r["parsed_answer"] for r in rows]), "raw": rows}
    previous_mode = model.training
    try:
        model.eval()
        # Evaluation cannot consume training RNG or change the resumed schedule.
        with isolated_rng():
            if baseline:
                with model.disable_adapter():
                    standard = [eval_one(processor, model, r, abstention_prompt=False) for r in plan["validation"]]
                    abstention = [eval_one(processor, model, r, abstention_prompt=True) for r in plan["validation"]]
            else:
                standard = [eval_one(processor, model, r, abstention_prompt=False) for r in plan["validation"]]
                abstention = [eval_one(processor, model, r, abstention_prompt=True) for r in plan["validation"]]
    finally:
        model.train(previous_mode)
    payload = {"run_id": store.run_id, "identity_hash": store.identity_hash, "step": step,
               "holdout_used": False, "standard": pack(standard), "abstention_prompt": pack(abstention)}
    write_once(path, payload)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--new-run", action="store_true", help="Create a unique run; never reuse old artifacts")
    mode.add_argument("--resume", type=Path, help="Restore the last fully committed checkpoint of this run")
    parser.add_argument("--out-dir", type=Path, help="Parent for new runs, inside the workspace")
    parser.add_argument("--stop-after", type=int, default=5, help="Absolute step budget; frozen schedule still has 50 steps")
    parser.add_argument("--recover-metrics", action="store_true", help="Explicitly finish an interrupted metric append")
    parser.add_argument("--plan", action="store_true", help="CPU-only identity/asset checks; no run creation or model loading")
    parser.add_argument("--execute-training", action="store_true", help="Required for future explicitly authorized GPU execution")
    args = parser.parse_args(argv)
    if args.plan and (args.execute_training or args.new_run or args.resume or args.recover_metrics):
        parser.error("--plan is read-only and cannot be combined with execution modes")
    if not args.plan and (not args.execute_training or not (args.new_run or args.resume)):
        parser.error("choose --plan, or explicitly --execute-training with --new-run/--resume")
    if args.recover_metrics and not args.resume:
        parser.error("--recover-metrics requires --resume")
    if args.resume and args.out_dir:
        parser.error("--out-dir applies to new runs only")
    plan = build_plan()
    cfg = plan["config"]
    if not 1 <= args.stop_after <= cfg["matched_to_2i_a"]["max_steps"]:
        parser.error("stop-after must be between 1 and the frozen 50-step horizon")
    if args.plan:
        print(json.dumps({"identity_hash": digest(plan["identity"]), "schedule_steps": len(plan["schedule"]),
                          "validation_cases": len(plan["validation"]), "holdout_used": False,
                          "optimizer_steps": 0, "model_loaded": False}, indent=2))
        return 0

    # These imports and operations are unreachable in CPU planning mode.
    temp = Path(os.environ.get("CROSSMODAL_TEMP_DIR", REPO_ROOT / ".cache/training-temp")).resolve()
    if not temp.is_relative_to(REPO_ROOT):
        raise ValueError("temporary files must remain inside workspace")
    temp.mkdir(parents=True, exist_ok=True)
    for key in ("TMP", "TEMP", "CROSSMODAL_TEMP_DIR"):
        os.environ[key] = str(temp)
    os.environ["TORCH_HOME"] = str(REPO_ROOT / ".cache/torch")
    os.environ["HF_HUB_OFFLINE"] = "1"
    apply_default_cache_environment()
    import psutil
    import torch
    from src.training.differentiable_scoring import encode_candidates
    if psutil.virtual_memory().available < 4 * 1024**3:
        raise RuntimeError("RAM_GATE_FAIL: at least 4 GiB available physical RAM required")
    if not torch.cuda.is_available():
        raise RuntimeError("pinned NF4 training requires CUDA; no CPU substitution")
    free_vram, total_vram = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(0.9, free_vram / total_vram))
    output = (args.resume or args.out_dir or (get_artifacts_dir() / "training/phase_2i_b/runs")).resolve()
    if not output.is_relative_to(REPO_ROOT):
        raise ValueError("run files must remain inside workspace")
    store = RunStore(output, plan["identity"]) if args.resume else RunStore.fresh(output, plan["identity"])
    model = optimizer = processor = None
    execution = {"run_id": store.run_id, "identity_hash": store.identity_hash,
                 "holdout_used": False, "requested_stop_after": args.stop_after}
    with store:
        code = 1
        try:
            if args.resume:
                execution["resume"] = store.inspect(recover_metrics=args.recover_metrics)
                if args.stop_after < store.step:
                    raise ValueError("requested budget is below restored step")
            store.event("LOADING", resume=bool(args.resume), stop_after=args.stop_after)
            seed = cfg["matched_to_2i_a"]["seed"]
            torch.manual_seed(seed)
            random.seed(seed)
            import numpy as np
            np.random.seed(seed % 2**32)
            reset_torch_peaks()
            processor, model, execution["load_meta"] = load_qwen3vl2_nf4(local_files_only=True)
            if (execution["load_meta"]["revision"] != QWEN_REVISION
                    or execution["load_meta"]["quantization"] != plan["identity"]["quantization"]):
                raise ValueError("loaded model identity mismatch")
            model, targets = attach_lora(model, r=cfg["matched_to_2i_a"]["lora_r"])
            if sorted(targets) != sorted(plan["identity"]["lora"]["target_modules"]):
                raise ValueError("LoRA target mismatch")
            guard_resources()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                         lr=cfg["matched_to_2i_a"]["learning_rate"])
            cand_ids = encode_candidates(processor.tokenizer, CANDIDATES)
            if args.resume:
                store.restore(model, optimizer, *adapter_restorers(model, plan["identity"]["lora"]))
            else:
                evaluate(processor, model, plan, store, 0, baseline=True)
                store.commit(0, None, model, optimizer, lambda path: model.save_pretrained(
                    str(path), safe_serialization=True, save_embedding_layers=False))
            store.event("RUNNING", restored_step=store.step, stop_after=args.stop_after)
            for index in range(store.step, args.stop_after):
                step = index + 1
                guard_resources()
                store.event("STEP_ATTEMPT", step=step, sample_id=plan["schedule"][index]["sample_id"])
                row = train_step_es(model, processor, plan["schedule"][index], optimizer,
                                    step=step, cand_ids=cand_ids, es_cfg=cfg["es_objective"])
                guard_resources()
                store.commit(step, row, model, optimizer, lambda path: model.save_pretrained(
                    str(path), safe_serialization=True, save_embedding_layers=False))
                if step in {cfg["matched_to_2i_a"]["sanity_steps"], args.stop_after}:
                    evaluate(processor, model, plan, store, step)
            status = "COMPLETE" if store.step == len(plan["schedule"]) else "PAUSED"
            store.event(status, completed_step=store.step)
            execution.update(status=status, completed_step=store.step)
            code = 0
        except BaseException as exc:
            execution.update(status="INCOMPLETE", committed_step=store.step,
                             error_type=type(exc).__name__, traceback=traceback.format_exc())
            code = 130 if isinstance(exc, KeyboardInterrupt) else 1
            try:
                store.event("INCOMPLETE", committed_step=store.step, error_type=type(exc).__name__)
            except Exception:
                execution["event_write_error"] = traceback.format_exc()
        finally:
            execution["memory_before_cleanup"] = snapshot_memory().to_dict()
            model = optimizer = processor = None
            gc.collect()
            try:
                torch.cuda.empty_cache()
                execution["cleanup_status"] = "PASSED"
            except Exception:
                execution["cleanup_status"] = "FAILED"
                execution["cleanup_error"] = traceback.format_exc()
                code = 1
            execution["memory_after_cleanup"] = snapshot_memory().to_dict()
            from uuid import uuid4
            write_once(store.path / f"execution_{uuid4().hex}.json", execution)
    print(json.dumps({"run_id": store.run_id, "run_path": str(store.path), **execution}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
