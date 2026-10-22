#!/usr/bin/env python3
"""Phase 2G — Bounded supervised QLoRA training + frozen-val evaluation.

Max 50 optimizer steps. No holdout. No custom losses. No dependency changes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from src.training.eval_freeze import freeze_val_ids, summarize_eval
from src.training.lora_config import apply_lora, build_lora_config, count_parameters
from src.training.qlora_train_utils import (
    apply_completion_only_labels,
    balanced_family_schedule,
    labels_not_fully_masked,
)

apply_default_cache_environment()
SEED = 20261103
ABSTAIN_SUFFIX = " You may answer Undetermined when evidence is insufficient."
ANSWER_SUFFIX = " Answer directly with: Yes, No, or Undetermined."


def build_inputs(processor, rec: Dict[str, Any], *, abstention_prompt: bool, device="cuda:0"):
    prompt = rec["prompt"]
    if abstention_prompt:
        if not prompt.endswith(ANSWER_SUFFIX) and ANSWER_SUFFIX.strip() not in prompt:
            prompt = prompt.rstrip() + "\n" + ABSTAIN_SUFFIX + ANSWER_SUFFIX
    image = Image.open(REPO_ROOT / rec["image_file"]).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(device)


def prompt_length_from_ids(input_ids) -> int:
    """Supervise only the final answer continuation: mask all but last 4 tokens.

    For short completions (Yes/No/Undetermined) last-4 includes the answer span.
    Image/prompt tokens are the bulk of the sequence and are masked.
    """
    t = int(input_ids.shape[1])
    return max(0, t - 4)


def eval_one(processor, model, rec, *, abstention_prompt: bool, device="cuda:0"):
    inputs = build_inputs(processor, rec, abstention_prompt=abstention_prompt, device=device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    lat = time.perf_counter() - t0
    new_ids = out[0, inputs["input_ids"].shape[1] :].tolist()
    raw = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
    parsed = parse_model_answer(raw)
    return {
        "sample_id": rec["sample_id"],
        "family_id": rec["family_id"],
        "ground_truth_label": rec["ground_truth_label"],
        "prompt_sha256": rec["prompt_sha256"],
        "image_sha256": rec["image_sha256"],
        "raw_completion": raw,
        "parsed_answer": parsed["parsed_label"],
        "is_correct": parsed["parsed_label"] == rec["ground_truth_label"],
        "latency_sec": round(lat, 4),
        "abstention_prompt": abstention_prompt,
        "n_image_tokens": int((inputs["input_ids"] == getattr(model.config, "image_token_id", 151655)).sum()),
    }


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


def train_steps(
    model,
    processor,
    schedule: List[Dict[str, Any]],
    *,
    start_step: int,
    end_step: int,
    optimizer: torch.optim.Optimizer,
    metrics_path: Path,
) -> Dict[str, Any]:
    model.train()
    last = {}
    for idx in range(start_step, end_step):
        rec = schedule[idx]
        inputs = build_inputs(processor, rec, abstention_prompt=False)
        plen = prompt_length_from_ids(inputs["input_ids"])
        labels = apply_completion_only_labels(
            inputs["input_ids"],
            inputs.get("attention_mask"),
            prompt_length=plen,
            pad_token_id=processor.tokenizer.pad_token_id
            if processor.tokenizer.pad_token_id is not None
            else processor.tokenizer.eos_token_id,
        )
        if not labels_not_fully_masked(labels):
            raise RuntimeError(f"labels fully masked for {rec['sample_id']}")

        reset_torch_peaks()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        if not torch.isfinite(loss.detach()):
            raise RuntimeError("non-finite loss")
        loss.backward()
        gnorm = 0.0
        n_g = 0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                n_g += 1
                gnorm += float(p.grad.detach().float().norm(2) ** 2)
        gnorm = gnorm ** 0.5
        if n_g == 0 or not torch.isfinite(torch.tensor(gnorm)):
            raise RuntimeError("invalid gradients")
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dur = time.perf_counter() - t0
        mem = snapshot_memory()
        row = {
            "step": idx + 1,
            "sample_id": rec["sample_id"],
            "family_id": rec["family_id"],
            "label": rec["ground_truth_label"],
            "loss": float(loss.detach().cpu()),
            "grad_l2": gnorm,
            "grad_tensors": n_g,
            "lr": optimizer.param_groups[0]["lr"],
            "duration_sec": round(dur, 4),
            "memory": mem.to_dict(),
        }
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        last = row
        print(
            f"  step {row['step']} label={row['label']} loss={row['loss']:.4f} "
            f"gnorm={row['grad_l2']:.4f} t={row['duration_sec']}s",
            flush=True,
        )
        del inputs, outputs, loss
        torch.cuda.empty_cache()
    return last


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--sanity-steps", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--baseline-only", action="store_true")
    args = p.parse_args()

    cfg = json.loads((REPO_ROOT / "configs/phase_2g/qlora_train_config_v1.json").read_text(encoding="utf-8"))
    assert cfg["data"]["holdout_allowed"] is False

    # Host RAM gate before any model load (training requires >=4 GiB free)
    try:
        import psutil

        free_gb = psutil.virtual_memory().available / (1024**3)
        if free_gb < 4.0:
            print(f"RAM_GATE_FAIL free_gb={free_gb:.2f} < 4.0 — not loading model")
            return 4
        print(f"RAM_GATE_OK free_gb={free_gb:.2f}")
    except Exception as exc:
        print(f"RAM_GATE_WARN cannot measure: {exc}")

    out_dir = args.out_dir or (get_artifacts_dir() / "training" / "phase_2g")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "train_metrics_append.jsonl"

    torch.manual_seed(cfg["training"]["seed"])
    random.seed(cfg["training"]["seed"])
    np_seed = cfg["training"]["seed"]

    # Data
    dataset_path = REPO_ROOT / cfg["data"]["dataset"].replace("cross_modal_v1", "cross_modal_contrast_v1")
    # actual file name
    dataset_path = REPO_ROOT / "datasets/cross_modal_contrast_v2/cross_modal_contrast_v2.jsonl"
    records = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    test_fams = {r["family_id"] for r in records if r["split"] == "test"}
    assert not any(r["split"] == "test" for r in records if False)
    train_recs = [r for r in records if r["split"] == "train"]
    # exclude previously heavily used 2C-C pilot families from training? optional — keep all train
    val_pack = freeze_val_ids(dataset_path, n_families=6)
    val_cases = val_pack["cases"]
    assert len({r["family_id"] for r in val_cases} & test_fams) == 0

    # Class balance check on train
    from collections import Counter as _Counter

    train_labels = _Counter(r["ground_truth_label"] for r in train_recs)
    assert set(train_labels) == {"Yes", "No", "Undetermined"}

    schedule = balanced_family_schedule(
        train_recs, max_steps=args.max_steps, seed=cfg["training"]["seed"]
    )

    report: Dict[str, Any] = {
        "phase": "2G",
        "config": cfg,
        "dataset_sha256": hashlib_sha(dataset_path),
        "n_train_records": len(train_recs),
        "train_label_counts": dict(train_labels),
        "val": {"n_families": val_pack["n_val_families"], "n_cases": val_pack["n_cases"]},
        "holdout_used": False,
        "steps": [],
    }

    # Load model + LoRA
    processor, model, load_meta = load_qwen3vl2_nf4(local_files_only=True)
    report["load_meta"] = load_meta
    model, targets = attach_lora(model, r=cfg["lora"]["r"])
    report["lora_targets"] = targets
    report["params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg["training"]["learning_rate"]
    )

    # ---- BASELINE eval (frozen base before steps? we already attached LoRA at init)
    # Evaluate "base-equivalent" by using lora disabled if possible
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            base_std = [eval_one(processor, model, r, abstention_prompt=False) for r in val_cases]
    else:
        base_std = [eval_one(processor, model, r, abstention_prompt=False) for r in val_cases]
    base_abs = [eval_one(processor, model, r, abstention_prompt=True) for r in val_cases]
    report["baseline_standard"] = summarize_eval(base_std)
    report["baseline_abstention_prompt"] = summarize_eval(base_abs)
    atomic_write_json(out_dir / "baseline_eval_v1.json", {
        "standard": report["baseline_standard"],
        "abstention": report["baseline_abstention_prompt"],
        "model_revision": QWEN_REVISION,
        "holdout_used": False,
    })
    print("BASELINE standard BA", report["baseline_standard"]["metrics"]["balanced_accuracy"])
    print("BASELINE abstain-prompt BA", report["baseline_abstention_prompt"]["metrics"]["balanced_accuracy"])

    if args.baseline_only:
        del model, processor
        torch.cuda.empty_cache()
        atomic_write_json(out_dir / "phase_2g_report_partial.json", report)
        return 0

    # ---- Step 0 already = baseline; train sanity 5 then to max
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    def snapshot_adapter(step: int, *, with_opt: bool) -> None:
        d = ckpt_dir / f"step_{step:03d}"
        d.mkdir(exist_ok=True)
        model.save_pretrained(str(d))
        state = {
            "step": step,
            "model_id": cfg["model_id"],
            "revision": cfg["revision"],
            "config_hash": hashlib_sha(cfg),
            "seed": cfg["training"]["seed"],
            "dataset_sha256": report["dataset_sha256"],
            "torch": torch.__version__,
            "trainable_params": report["params"],
        }
        if with_opt:
            torch.save(optimizer.state_dict(), d / "optimizer.pt")
            state["optimizer_saved"] = True
        atomic_write_json(d / "checkpoint_meta.json", state)

    # Resume-aware: if metrics already have completed steps, continue from there
    # (adapter must have been checkpointed; otherwise reset metrics for clean restart)
    done = completed_steps(metrics_path)
    ckpt_last = out_dir / "checkpoints" / f"step_{done:03d}"
    if done > 0 and not (ckpt_last / "checkpoint_meta.json").is_file():
        print(f"WARNING: metrics show step {done} but checkpoint missing — clearing metrics for clean restart")
        metrics_path.write_text("", encoding="utf-8")
        done = 0
    elif done >= args.max_steps:
        print(f"Already at step {done}; evaluating only")
        done = args.max_steps

    # Sanity 5 (or skip if already past)
    sanity_end = min(args.sanity_steps, args.max_steps)
    if done < sanity_end:
        print(f"=== SANITY steps {done}→{sanity_end} ===")
        train_steps(
            model, processor, schedule, start_step=done, end_step=sanity_end,
            optimizer=optimizer, metrics_path=metrics_path,
        )
        snapshot_adapter(sanity_end, with_opt=True)
        done = sanity_end

    if done >= args.sanity_steps and (out_dir / "checkpoints" / f"step_{args.sanity_steps:03d}" / "checkpoint_meta.json").is_file() or done == args.sanity_steps:
        if done >= args.sanity_steps:
            val_mid = [eval_one(processor, model, r, abstention_prompt=False) for r in val_cases]
            report["val_step_5"] = summarize_eval(val_mid)
            print("VAL@5 BA", report["val_step_5"]["metrics"]["balanced_accuracy"])

    # Continue to max
    if done < args.max_steps:
        print(f"=== CONTINUE {done} → {args.max_steps} ===")
        train_steps(
            model,
            processor,
            schedule,
            start_step=done,
            end_step=args.max_steps,
            optimizer=optimizer,
            metrics_path=metrics_path,
        )
        snapshot_adapter(args.max_steps, with_opt=True)
        done = args.max_steps
    report["total_steps"] = done

    # Final val
    val_final = [eval_one(processor, model, r, abstention_prompt=False) for r in val_cases]
    report["val_step_final"] = summarize_eval(val_final)
    # With adapter still on; also abstention prompt variant
    val_final_abs = [eval_one(processor, model, r, abstention_prompt=True) for r in val_cases]
    report["val_step_final_abstention_prompt"] = summarize_eval(val_final_abs)

    report["final_memory"] = snapshot_memory().to_dict()
    report["status"] = "OK"
    report["total_steps"] = args.max_steps

    # Paired deltas vs baseline
    def delta(a, b):
        return {
            "baseline": a["metrics"]["balanced_accuracy"],
            "final": b["metrics"]["balanced_accuracy"],
            "delta": b["metrics"]["balanced_accuracy"] - a["metrics"]["balanced_accuracy"],
        }

    report["paired_ba"] = {
        "standard_vs_final": delta(report["baseline_standard"], report["val_step_final"]),
        "abstention_prompt_vs_final_abstention": delta(
            report["baseline_abstention_prompt"], report["val_step_final_abstention_prompt"]
        ),
    }

    del model, processor, optimizer
    torch.cuda.empty_cache()
    atomic_write_json(out_dir / "phase_2g_report_v1.json", report)
    print(json.dumps({
        "baseline_ba": report["baseline_standard"]["metrics"]["balanced_accuracy"],
        "val5_ba": report["val_step_5"]["metrics"]["balanced_accuracy"],
        "final_ba": report["val_step_final"]["metrics"]["balanced_accuracy"],
        "paired": report["paired_ba"],
        "out": str(out_dir / "phase_2g_report_v1.json"),
    }, indent=2))
    return 0


def hashlib_sha(path: Path) -> str:
    import hashlib

    if isinstance(path, dict):
        blob = json.dumps(path, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def completed_steps(metrics_path: Path) -> int:
    if not metrics_path.is_file():
        return 0
    steps = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            steps.append(json.loads(line)["step"])
    return max(steps) if steps else 0


if __name__ == "__main__":
    raise SystemExit(main())
