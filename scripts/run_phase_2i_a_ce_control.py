#!/usr/bin/env python3
"""Phase 2I-A — Fair CE-QLoRA control with family-max sampling and gold-span masking.

CE only. Max 50 steps. No holdout. Strict parser. Raw completions stored.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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

from src.causal.telemetry import reset_torch_peaks, snapshot_memory
from src.config.paths import apply_default_cache_environment, get_artifacts_dir
from src.instrumentation.capture_integrity import atomic_write_json
from src.models.qwen3vl2_loader import QWEN_REVISION, load_qwen3vl2_nf4
from src.training.answer_parser import evaluation_accounting, strict_parse_answer
from src.training.eval_freeze import freeze_val_ids, summarize_eval
from src.training.lora_config import apply_lora, build_lora_config
from src.training.sampling_and_masking import (
    build_supervised_example,
    maximize_family_coverage_schedule,
)

apply_default_cache_environment()


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
    from PIL import Image

    prompt = rec["prompt"]
    if abstention_prompt and "You may answer Undetermined" not in prompt:
        prompt = (
            prompt.rstrip()
            + "\nYou may answer Undetermined when evidence is insufficient."
            + "\nAnswer directly with: Yes, No, or Undetermined."
        )
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
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    lat = time.perf_counter() - t0
    new_ids = out[0, inputs["input_ids"].shape[1] :].tolist()
    raw = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
    parsed = strict_parse_answer(raw)
    return {
        "sample_id": rec["sample_id"],
        "family_id": rec["family_id"],
        "ground_truth_label": rec["ground_truth_label"],
        "prompt_sha256": rec["prompt_sha256"],
        "image_sha256": rec["image_sha256"],
        "raw_completion": raw,
        "parsed_answer": parsed["parsed_label"],
        "parse_status": parsed["parse_status"],
        "is_correct": parsed["parsed_label"] == rec["ground_truth_label"],
        "latency_sec": round(lat, 4),
        "abstention_prompt": abstention_prompt,
    }


def train_steps(model, processor, schedule, *, start_step, end_step, optimizer, metrics_path):
    model.train()
    last: Dict[str, Any] = {}
    for idx in range(start_step, end_step):
        rec = schedule[idx]
        inputs, labels, span_meta = build_supervised_example(
            processor, rec, abstention_prompt=False, device="cuda:0"
        )
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
        if n_g == 0:
            raise RuntimeError("no LoRA gradients")
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
            "n_supervised": span_meta["n_supervised"],
            "answer": span_meta["answer"],
            "duration_sec": round(dur, 4),
            "memory": mem.to_dict(),
        }
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        last = row
        print(
            f"  step {row['step']} fam={row['family_id']} label={row['label']} "
            f"loss={row['loss']:.4f} sup={row['n_supervised']} t={row['duration_sec']:.2f}s",
            flush=True,
        )
        del inputs, labels, outputs, loss
        torch.cuda.empty_cache()
    return last


def completed_steps(metrics_path: Path) -> int:
    if not metrics_path.is_file():
        return 0
    steps = [json.loads(l)["step"] for l in metrics_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return max(steps) if steps else 0


def sha256_file(path: Path) -> str:
    import hashlib

    if isinstance(path, dict):
        return hashlib.sha256(json.dumps(path, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--sanity-steps", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    cfg = json.loads((REPO_ROOT / "configs/phase_2i_a/ce_control_protocol_v1.json").read_text(encoding="utf-8"))
    assert cfg["data"]["holdout_allowed"] is False
    assert cfg["training"]["custom_losses"] is False

    try:
        import psutil

        free = psutil.virtual_memory().available / (1024**3)
        if free < 4.0:
            print(f"RAM_GATE_FAIL {free:.2f}")
            return 4
        print(f"RAM_GATE_OK {free:.2f}")
    except Exception as exc:
        print("RAM_GATE_WARN", exc)

    out_dir = args.out_dir or (get_artifacts_dir() / "training" / "phase_2i_a")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "train_metrics_append.jsonl"

    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    random.seed(seed)

    ds = REPO_ROOT / cfg["data"]["dataset"]
    records = [json.loads(line) for line in ds.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_recs = [r for r in records if r["split"] == "train"]
    val_pack = freeze_val_ids(ds, n_families=6)
    val_cases = val_pack["cases"]
    assert not ({r["family_id"] for r in val_cases} & {r["family_id"] for r in train_recs})

    schedule = maximize_family_coverage_schedule(train_recs, max_steps=args.max_steps, seed=seed)
    exp_families = sorted({r["family_id"] for r in schedule})
    print(f"schedule_families={len(exp_families)} unique_samples={len({r['sample_id'] for r in schedule})}")

    report: Dict[str, Any] = {
        "phase": "2I-A",
        "protocol": cfg,
        "dataset_sha256": sha256_file(ds),
        "schedule_family_count": len(exp_families),
        "schedule_families": exp_families,
        "schedule_label_counts": dict(Counter := __import__("collections").Counter(r["ground_truth_label"] for r in schedule)),
        "val": {"n_families": val_pack["n_val_families"], "n_cases": val_pack["n_cases"]},
        "holdout_used": False,
    }

    processor, model, load_meta = load_qwen3vl2_nf4(local_files_only=True)
    report["load_meta"] = load_meta
    model, targets = attach_lora(model, r=cfg["lora"]["r"])
    report["lora_targets"] = targets
    report["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg["training"]["learning_rate"]
    )

    # Baseline (LoRA disabled if possible)
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            base_std = [eval_one(processor, model, r, abstention_prompt=False) for r in val_cases]
    else:
        base_std = [eval_one(processor, model, r, abstention_prompt=False) for r in val_cases]
    base_abs = [eval_one(processor, model, r, abstention_prompt=True) for r in val_cases]

    def pack(rows):
        y_t = [r["ground_truth_label"] for r in rows]
        y_p = [r["parsed_answer"] for r in rows]
        m = evaluation_accounting(y_t, y_p)
        return {
            "metrics": m,
            "raw": rows,
            "families": sorted({r["family_id"] for r in rows}),
        }

    report["baseline_standard"] = pack(base_std)
    report["baseline_abstention_prompt"] = pack(base_abs)
    atomic_write_json(
        out_dir / "baseline_eval_v1.json",
        {
            "standard": {"metrics": report["baseline_standard"]["metrics"], "raw": base_std},
            "abstention": {"metrics": report["baseline_abstention_prompt"]["metrics"], "raw": base_abs},
            "revision": QWEN_REVISION,
            "holdout_used": False,
        },
    )
    print("BASELINE BA", report["baseline_standard"]["metrics"]["balanced_accuracy"],
          "invalid", report["baseline_standard"]["metrics"]["invalid_output_rate"])

    done = completed_steps(metrics_path)
    ckpt = out_dir / "checkpoints" / f"step_{done:03d}"
    if done > 0 and not (ckpt / "checkpoint_meta.json").is_file():
        print(f"clearing metrics (no ckpt at step {done})")
        metrics_path.write_text("", encoding="utf-8")
        done = 0

    def snapshot(step: int) -> None:
        d = out_dir / "checkpoints" / f"step_{step:03d}"
        d.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(d))
        atomic_write_json(
            d / "checkpoint_meta.json",
            {
                "step": step,
                "revision": QWEN_REVISION,
                "seed": seed,
                "dataset_sha256": report["dataset_sha256"],
                "protocol_sha": sha256_file(cfg),
                "trainable_params": report["trainable_params"],
                "objective": "ce_only",
            },
        )
        torch.save(optimizer.state_dict(), d / "optimizer.pt")

    sanity_end = min(args.sanity_steps, args.max_steps)
    if done < sanity_end:
        print(f"=== SANITY {done}->{sanity_end} ===")
        train_steps(model, processor, schedule, start_step=done, end_step=sanity_end, optimizer=optimizer, metrics_path=metrics_path)
        snapshot(sanity_end)
        done = sanity_end
    if done >= args.sanity_steps:
        val_mid = [eval_one(processor, model, r, abstention_prompt=False) for r in val_cases]
        report["val_step_5"] = pack(val_mid)
        print("VAL@5 BA", report["val_step_5"]["metrics"]["balanced_accuracy"],
              "invalid", report["val_step_5"]["metrics"]["invalid_output_rate"])

    if done < args.max_steps:
        print(f"=== CONTINUE {done}->{args.max_steps} ===")
        train_steps(model, processor, schedule, start_step=done, end_step=args.max_steps, optimizer=optimizer, metrics_path=metrics_path)
        snapshot(args.max_steps)
        done = args.max_steps
    report["total_steps"] = done

    val_final = [eval_one(processor, model, r, abstention_prompt=False) for r in val_cases]
    report["val_final"] = pack(val_final)
    val_abs = [eval_one(processor, model, r, abstention_prompt=True) for r in val_cases]
    report["val_final_abstention_prompt"] = pack(val_abs)

    def ba(block):
        return block["metrics"]["balanced_accuracy"]

    report["paired"] = {
        "baseline_to_final": {
            "baseline": ba(report["baseline_standard"]),
            "final": ba(report["val_final"]),
            "delta": ba(report["val_final"]) - ba(report["baseline_standard"]),
        }
    }
    report["status"] = "OK"
    report["final_memory"] = snapshot_memory().to_dict()

    del model, processor, optimizer
    torch.cuda.empty_cache()
    atomic_write_json(out_dir / "phase_2i_a_report_v1.json", report)
    # save final raw eval
    atomic_write_json(out_dir / "val_final_raw_v1.json", val_final)
    print(json.dumps({
        "baseline_ba": ba(report["baseline_standard"]),
        "val5_ba": report.get("val_step_5", {}).get("metrics", {}).get("balanced_accuracy"),
        "final_ba": ba(report["val_final"]),
        "final_invalid_rate": report["val_final"]["metrics"]["invalid_output_rate"],
        "schedule_families": len(exp_families),
        "steps": done,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
