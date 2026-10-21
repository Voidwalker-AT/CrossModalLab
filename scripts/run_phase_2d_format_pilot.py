#!/usr/bin/env python3
"""Phase 2D — Bounded format/evidence-sufficiency pilot (≤24 evals per model)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
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
from src.models.qwen3vl2_loader import load_qwen3vl2_nf4

apply_default_cache_environment()
SMOL_REVISION = "482adb537c021c86670beed01cd58990d01e72e4"


def parse_with_format(text: str, choices: List[str], abstain_labels: List[str]) -> str:
    t = text.strip().lower()
    for lab in sorted(abstain_labels, key=len, reverse=True):
        if lab.lower() in t:
            return "Undetermined"
    if not t:
        return "Invalid"
    first = t.split()[0].strip(".,;:!?")
    if first in {"yes", "true", "entailed"} or t.startswith("yes") or t.startswith("true"):
        return "Yes"
    if first in {"no", "false", "contradicted"} or t.startswith("no") or t.startswith("false"):
        return "No"
    if any(x in t for x in ("insufficient", "cannot", "undetermined", "undecidable", "determined")):
        return "Undetermined"
    # order fallback
    low = [c.lower() for c in choices]
    if low[0] in t:
        return "Yes"
    if low[1] in t:
        return "No"
    if len(low) > 2 and low[2] in t:
        return "Undetermined"
    return "Invalid"


def build_inputs(processor, prompt: str, image: Optional[Image.Image], model_tag: str, device="cuda:0"):
    if image is None:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    else:
        content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(device)


def timed_generate(model, inputs, max_new_tokens=24) -> Dict[str, Any]:
    reset_torch_peaks()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    lat = time.perf_counter() - t0
    new_ids = out[0, inputs["input_ids"].shape[1] :].tolist()
    return {"latency_sec": lat, "new_ids": new_ids, "memory": snapshot_memory().to_dict()}


def append(path: Path, rec: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_done(path: Path) -> set:
    done = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "ok":
                    done.add(r["evaluation_id"])
    return done


def eval_smol_case(processor, model, case, image_token_id):
    image = None
    if case.get("image_file"):
        image = Image.open(REPO_ROOT / case["image_file"]).convert("RGB")
    inputs = build_inputs(processor, case["prompt"], image, "smolvlm2")
    t = timed_generate(model, inputs)
    raw = processor.tokenizer.decode(t["new_ids"], skip_special_tokens=True)
    parsed = parse_with_format(raw, case["choices"], case["parse_abstain_labels"])
    return {
        "raw_completion": raw,
        "parsed_answer": parsed,
        "latency_sec": round(t["latency_sec"], 4),
        "seq_len": int(inputs["input_ids"].shape[1]),
        "n_image_tokens": int((inputs["input_ids"] == image_token_id).sum()),
        "memory": t["memory"],
    }


def eval_qwen_case(processor, model, case, image_token_id):
    image = None
    if case.get("image_file"):
        image = Image.open(REPO_ROOT / case["image_file"]).convert("RGB")
    inputs = build_inputs(processor, case["prompt"], image, "qwen3vl2")
    t = timed_generate(model, inputs)
    raw = processor.tokenizer.decode(t["new_ids"], skip_special_tokens=True)
    parsed = parse_with_format(raw, case["choices"], case["parse_abstain_labels"])
    return {
        "raw_completion": raw,
        "parsed_answer": parsed,
        "latency_sec": round(t["latency_sec"], 4),
        "seq_len": int(inputs["input_ids"].shape[1]),
        "n_image_tokens": int((inputs["input_ids"] == image_token_id).sum()),
        "memory": t["memory"],
    }


def run_model(model_tag: str, cases: List[Dict[str, Any]], jsonl: Path, max_new: int) -> int:
    done = load_done(jsonl)
    pending = [c for c in cases if f"{model_tag}::{c['case_id']}" not in done][:max_new]
    print(f"[{model_tag}] done={len(done)} pending={len(pending)}")
    if not pending:
        return 0

    if model_tag == "smolvlm2":
        from src.models.model_loader import load_pinned_smolvlm2

        processor, model, meta = load_pinned_smolvlm2(require_gpu=True)
        meta = {**meta, "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct", "revision": SMOL_REVISION}
        image_token_id = model.config.image_token_id
        evaluator = eval_smol_case
    else:
        processor, model, meta = load_qwen3vl2_nf4(local_files_only=True)
        image_token_id = meta.get("image_token_id", 151655)
        evaluator = eval_qwen_case

    n = 0
    try:
        for i, case in enumerate(pending, 1):
            eid = f"{model_tag}::{case['case_id']}"
            print(f"  [{i}/{len(pending)}] {eid}", flush=True)
            res = evaluator(processor, model, case, image_token_id)
            rec = {
                "evaluation_id": eid,
                "status": "ok",
                "model_tag": model_tag,
                "model_revision": meta.get("revision"),
                **{k: case[k] for k in (
                    "case_id", "family_id", "kind", "format_id", "logical_label",
                    "mapped_answer_in_format", "prompt_sha256", "image_sha256",
                )},
                "is_abstain_correct": (
                    (res["parsed_answer"] == "Undetermined")
                    if case["logical_label"] == "Undetermined"
                    else (res["parsed_answer"] == case["logical_label"])
                ),
                "parsed_answer": res["parsed_answer"],
                "raw_completion": res["raw_completion"],
                "latency_sec": res["latency_sec"],
                "seq_len": res["seq_len"],
                "n_image_tokens": res["n_image_tokens"],
                "memory": res["memory"],
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            # generalize correctness for undetermined mapped labels
            rec["is_correct"] = rec["parsed_answer"] == case["logical_label"]
            append(jsonl, rec)
            print(f"    pred={rec['parsed_answer']} GT={rec['logical_label']} ok={rec['is_correct']} lat={rec['latency_sec']}s", flush=True)
            n += 1
            torch.cuda.empty_cache()
    finally:
        del model, processor
        torch.cuda.empty_cache()
    return n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["smolvlm2", "qwen3vl2", "both"], default="both")
    p.add_argument("--max-per-model", type=int, default=24)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    man = json.loads((REPO_ROOT / "configs/phase_2d/evidence_format_manifest_v1.json").read_text(encoding="utf-8"))
    cases = man["cases"]
    out_dir = args.out_dir or (get_artifacts_dir() / "evidence" / "phase_2d")
    jsonl = out_dir / "format_pilot_v1.jsonl"

    tags = ["smolvlm2", "qwen3vl2"] if args.model == "both" else [args.model]
    for tag in tags:
        print(f"\n===== {tag} =====")
        run_model(tag, cases, jsonl, args.max_per_model)

    # Analysis
    rows = []
    if jsonl.is_file():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    latest = {r["evaluation_id"]: r for r in rows if r.get("status") == "ok"}

    def agg(subset):
        from collections import defaultdict
        from src.instrumentation.capture_integrity import compute_probe_metrics

        if not subset:
            return {}
        by_kind = defaultdict(list)
        for r in subset:
            by_kind[r["kind"]].append(r)
        out = {"n": len(subset)}
        y_t = [r["logical_label"] for r in subset]
        y_p = [r["parsed_answer"] for r in subset]
        out["overall"] = compute_probe_metrics(y_t, y_p, ["Yes", "No", "Undetermined"])
        for kind, rs in by_kind.items():
            und = [r for r in rs if r["logical_label"] == "Undetermined"]
            abst = sum(1 for r in und if r["parsed_answer"] == "Undetermined")
            out[kind] = {
                "n": len(rs),
                "mean_latency": sum(r["latency_sec"] for r in rs) / len(rs),
                "undetermined_n": len(und),
                "abstention_rate": round(abst / len(und), 6) if und else None,
                "pred_dist": dict(Counter_p(r["parsed_answer"] for r in rs)),
            }
        return out

    def Counter_p(gen):
        from collections import Counter

        return Counter(gen)

    by_model = defaultdict(list)
    for r in latest.values():
        by_model[r["model_tag"]].append(r)

    # Format sensitivity: abstention rate by format_id among format_sensitivity kind
    fmt_abs = {}
    for tag, rs in by_model.items():
        fmt_abs[tag] = {}
        for r in rs:
            if r["kind"] != "format_sensitivity":
                continue
            fmt_abs[tag].setdefault(r["format_id"], []).append(
                r["parsed_answer"] == "Undetermined"
            )
        for fid, flags in fmt_abs[tag].items():
            fmt_abs[tag][fid] = {
                "n": len(flags),
                "abstention_rate": sum(flags) / len(flags),
            }

    summary = {
        "phase": "2D",
        "n_unique": len(latest),
        "per_model": {k: agg(v) for k, v in by_model.items()},
        "format_abstention": fmt_abs,
        "final_holdout_touched": False,
        "causal_patching": False,
        "timing_note": "latency uses torch.cuda.synchronize before/after generate",
    }
    atomic_write_json(out_dir / "format_pilot_summary_v1.json", summary)
    print(json.dumps({"per_model": {k: {kk: vv for kk, vv in v.items() if kk in ("n", "format_sensitivity", "text_only_insufficient")} for k, v in summary["per_model"].items()}, "format_abstention": fmt_abs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
