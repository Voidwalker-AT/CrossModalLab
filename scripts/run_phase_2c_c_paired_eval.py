#!/usr/bin/env python3
"""Phase 2C-C — Paired SmolVLM2 vs Qwen3-VL behavioral evaluation.

One model at a time; max 18 cases each; resume-safe; no holdout; no patching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

apply_default_cache_environment()

CANDIDATES = [" Yes", " No", " Undetermined"]
SMOL_REVISION = "482adb537c021c86670beed01cd58990d01e72e4"
QWEN_REVISION_PIN = QWEN_REVISION


def load_manifest() -> Dict[str, Any]:
    path = REPO_ROOT / "configs/phase_2c_c/paired_eval_manifest_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def eval_id(model_tag: str, sample_id: str) -> str:
    return f"{model_tag}::{sample_id}"


def build_inputs_smol(processor, prompt: str, image: Image.Image, device="cuda:0"):
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


def build_inputs_qwen(processor, prompt: str, image: Image.Image, device="cuda:0"):
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


def run_case(model, processor, inputs, device, image_token_id) -> Dict[str, Any]:
    reset_torch_peaks()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    lat = time.perf_counter() - t0
    new_ids = out[0, inputs["input_ids"].shape[1] :].tolist()
    raw = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
    parsed = parse_model_answer(raw)
    mem = snapshot_memory()
    return {
        "raw_completion": raw,
        "parsed_answer": parsed["parsed_label"],
        "parse_status": parsed["parse_status"],
        "latency_sec": round(lat, 4),
        "seq_len": int(inputs["input_ids"].shape[1]),
        "n_image_tokens": int((inputs["input_ids"] == image_token_id).sum()),
        "memory": mem.to_dict(),
    }


def append_record(path: Path, rec: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_done(path: Path) -> Dict[str, Dict[str, Any]]:
    done = {}
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") == "ok":
            done[r["evaluation_id"]] = r
    return done


def evaluate_model(
    model_tag: str,
    loader,
    image_token_id_fn,
    manifest: Dict[str, Any],
    jsonl_path: Path,
    *,
    max_new_cases: Optional[int] = None,
) -> int:
    done = load_done(jsonl_path)
    cases = []
    for fam in manifest["families"]:
        for case in fam["cases"]:
            cases.append({**case, "family_id": fam["family_id"]})

    pending = [c for c in cases if eval_id(model_tag, c["sample_id"]) not in done]
    if max_new_cases is not None:
        pending = pending[:max_new_cases]
    print(f"[{model_tag}] done={len(done)} pending={len(pending)} of {len(cases)}")
    if not pending:
        return 0

    processor, model, meta = loader()
    image_token_id = image_token_id_fn(model, meta)
    try:
        for i, case in enumerate(pending, 1):
            iid = eval_id(model_tag, case["sample_id"])
            print(f"  [{i}/{len(pending)}] {iid}", flush=True)
            img_path = REPO_ROOT / case["image_file"]
            image = Image.open(img_path).convert("RGB")
            img_sha = hashlib.sha256(img_path.read_bytes()).hexdigest()
            if img_sha != case["image_sha256"]:
                raise RuntimeError(f"image hash mismatch for {case['sample_id']}")
            inputs = (
                build_inputs_smol(processor, case["prompt"], image)
                if model_tag == "smolvlm2"
                else build_inputs_qwen(processor, case["prompt"], image)
            )
            res = run_case(model, processor, inputs, "cuda:0", image_token_id)
            rec = {
                "evaluation_id": iid,
                "status": "ok",
                "model_tag": model_tag,
                "model_id": meta.get("model_id"),
                "revision": meta.get("revision") or (SMOL_REVISION if model_tag == "smolvlm2" else QWEN_REVISION_PIN),
                "quantization": meta.get("quantization", {"load_in_4bit": True}),
                "family_id": case["family_id"],
                "sample_id": case["sample_id"],
                "variant_id": case["variant_id"],
                "ground_truth_label": case["ground_truth_label"],
                "image_sha256": img_sha,
                "prompt_sha256": case["prompt_sha256"],
                "prompt": case["prompt"],
                "is_correct": res["parsed_answer"] == case["ground_truth_label"],
                **res,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            append_record(jsonl_path, rec)
            print(f"    pred={rec['parsed_answer']} GT={rec['ground_truth_label']} correct={rec['is_correct']} lat={rec['latency_sec']}s", flush=True)
            del inputs
            torch.cuda.empty_cache()
    finally:
        del model, processor
        torch.cuda.empty_cache()
    return len(pending)


def load_smol():
    from src.models.model_loader import load_pinned_smolvlm2

    processor, model, meta = load_pinned_smolvlm2(require_gpu=True)
    meta = dict(meta)
    meta["model_id"] = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    meta["revision"] = SMOL_REVISION
    return processor, model, meta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["smolvlm2", "qwen3vl2", "both"], default="both")
    p.add_argument("--smoke", action="store_true", help="1 case per model first")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    manifest = load_manifest()
    out_dir = args.out_dir or (get_artifacts_dir() / "paired" / "phase_2c_c")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "paired_evaluations_v1.jsonl"

    models = ["smolvlm2", "qwen3vl2"] if args.model == "both" else [args.model]
    if args.smoke:
        models = models  # still one at a time

    for tag in models:
        print(f"\n========== MODEL {tag} ==========")
        if tag == "smolvlm2":
            n = evaluate_model(tag, load_smol, lambda m, meta: m.config.image_token_id, manifest, jsonl_path, max_new_cases=1 if args.smoke else None)
        else:
            def _qload():
                processor, model, meta = load_qwen3vl2_nf4(local_files_only=True)
                return processor, model, meta

            n = evaluate_model(
                tag,
                _qload,
                lambda m, meta: meta.get("image_token_id", 151655),
                manifest,
                jsonl_path,
                max_new_cases=1 if args.smoke else None,
            )
        print(f"{tag} evaluated this invocation: {n}")

    # Analysis
    rows = []
    if jsonl_path.is_file():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    latest = {}
    for r in rows:
        if r.get("status") == "ok":
            latest[r["evaluation_id"]] = r
    smol = {r["sample_id"]: r for r in latest.values() if r["model_tag"] == "smolvlm2"}
    qwen = {r["sample_id"]: r for r in latest.values() if r["model_tag"] == "qwen3vl2"}

    def metrics(by_sid):
        from src.instrumentation.capture_integrity import compute_probe_metrics

        if not by_sid:
            return {}
        y_true = [r["ground_truth_label"] for r in by_sid.values()]
        y_pred = [r["parsed_answer"] for r in by_sid.values()]
        m = compute_probe_metrics(y_true, y_pred, ["Yes", "No", "Undetermined"])
        und = [r for r in by_sid.values() if r["ground_truth_label"] == "Undetermined"]
        abstain = sum(1 for r in und if r["parsed_answer"] == "Undetermined")
        definite = sum(1 for r in und if r["parsed_answer"] in {"Yes", "No"})
        return {
            **m,
            "underdetermined_n": len(und),
            "underdetermined_abstention": round(abstain / len(und), 6) if und else None,
            "unsupported_definite_rate": round(definite / len(und), 6) if und else None,
            "mean_latency_sec": round(sum(r["latency_sec"] for r in by_sid.values()) / len(by_sid.values()), 4),
            "max_memory_torch_alloc": max((r.get("memory") or {}).get("torch_allocated_mib") or 0 for r in by_sid.values()),
            "max_nvidia_smi": max((r.get("memory") or {}).get("nvidia_smi_used_mib") or 0 for r in by_sid.values()),
        }

    shared_ids = sorted(set(smol) & set(qwen))
    pair = {
        "both_correct": [],
        "both_incorrect": [],
        "only_smol_correct": [],
        "only_qwen_correct": [],
        "shared_undetermined_failure": [],
        "disagreement": [],
    }
    for sid in shared_ids:
        s, q = smol[sid], qwen[sid]
        entry = {
            "sample_id": sid,
            "family_id": s["family_id"],
            "variant_id": s["variant_id"],
            "gt": s["ground_truth_label"],
            "smol_pred": s["parsed_answer"],
            "qwen_pred": q["parsed_answer"],
        }
        if s["is_correct"] and q["is_correct"]:
            pair["both_correct"].append(entry)
        elif (not s["is_correct"]) and (not q["is_correct"]):
            pair["both_incorrect"].append(entry)
            if s["ground_truth_label"] == "Undetermined":
                pair["shared_undetermined_failure"].append(entry)
        elif s["is_correct"] and not q["is_correct"]:
            pair["only_smol_correct"].append(entry)
        else:
            pair["only_qwen_correct"].append(entry)
        if s["parsed_answer"] != q["parsed_answer"]:
            pair["disagreement"].append(entry)

    summary = {
        "phase": "2C-C",
        "manifest": "configs/phase_2c_c/paired_eval_manifest_v1.json",
        "n_smolvlm2": len(smol),
        "n_qwen3vl2": len(qwen),
        "n_paired": len(shared_ids),
        "smolvlm2_metrics": metrics(smol),
        "qwen3vl2_metrics": metrics(qwen),
        "paired_comparison": {k: len(v) for k, v in pair.items()},
        "paired_examples": pair,
        "final_holdout_touched": False,
        "notes": [
            "Behavioral only; not mechanistic.",
            "Different architectures; logits not cross-calibrated.",
            "Small pilot; family is independent unit.",
        ],
    }
    atomic_write_json(out_dir / "paired_eval_summary_v1.json", summary)
    print(json.dumps({
        "n_smol": len(smol),
        "n_qwen": len(qwen),
        "paired": summary["paired_comparison"],
        "smol_undet": summary["smolvlm2_metrics"].get("underdetermined_abstention"),
        "qwen_undet": summary["qwen3vl2_metrics"].get("underdetermined_abstention"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
