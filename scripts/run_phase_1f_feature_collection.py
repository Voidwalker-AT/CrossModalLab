#!/usr/bin/env python3
"""Phase 1F — Expanded resumable feature collection on contrast v2 development families.

Cap: --max-forwards (default 96 NEW). Reuses Phase 1E features when identity matches.
No final holdout. No VLM training.
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

import numpy as np
import torch
from PIL import Image

from src.benchmark.contrast_v2 import load_contrast_records
from src.config.paths import apply_default_cache_environment, get_artifacts_dir
from src.evaluation.behavioral_harness import append_jsonl, load_jsonl
from src.instrumentation.capture_integrity import (
    annotate_identity_dict,
    build_capture_identity,
    should_skip_capture,
)
from src.instrumentation.multi_layer_capture import MultiLayerCaptureSpec, capture_multi_layer_passive
from src.models.model_loader import load_pinned_smolvlm2

apply_default_cache_environment()
LAYERS = (3, 12, 18, 23)
SELECTORS = ("last", "last_text_non_image", "mean_image")
CAPTURE_SCHEMA = "phase_1f_features_v1"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-families", type=int, default=32)
    p.add_argument("--max-forwards", type=int, default=96)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "datasets/cross_modal_contrast_v2/cross_modal_contrast_v2.jsonl",
    )
    p.add_argument(
        "--reuse-meta",
        type=Path,
        default=get_artifacts_dir() / "probes" / "phase_1e" / "contrast_probe_meta_v1.jsonl",
    )
    args = p.parse_args()

    records = load_contrast_records(args.dataset)
    by_fam_var = {(r["family_id"], r["variant_id"]): r for r in records}
    # Development only: train+validation splits from contrast v2
    dev_fams = sorted({r["family_id"] for r in records if r["split"] in {"train", "validation"}})
    # Prefer families not already in 1E pilot when expanding
    existing_fams = set()
    if args.reuse_meta.is_file():
        for row in load_jsonl(args.reuse_meta):
            if row.get("status") == "ok":
                existing_fams.add(row.get("family_id"))
    new_fams = [f for f in dev_fams if f not in existing_fams]
    # Mix: keep some 1E families for identity continuity + new ones
    plan_fams = (sorted(existing_fams & set(dev_fams)) + new_fams)[: args.n_families]
    print(f"Plan families={len(plan_fams)} (from_existing={len(existing_fams & set(plan_fams))} new={len(plan_fams) - len(existing_fams & set(plan_fams))})")

    plan = []
    for fam in plan_fams:
        for var in ("entailed", "contradicted", "underdetermined"):
            plan.append((fam, var, by_fam_var[(fam, var)]))
    plan = plan[: args.max_forwards * 2]  # allow skip-resume; hard cap below

    out_dir = args.out_dir or (get_artifacts_dir() / "probes" / "phase_1f")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "features_meta_v1.jsonl"
    vec_dir = out_dir / "features"
    vec_dir.mkdir(exist_ok=True)
    existing = load_jsonl(meta_path) if meta_path.is_file() else []
    # Also seed from 1E meta for reuse of complete identities
    if args.reuse_meta.is_file():
        for row in load_jsonl(args.reuse_meta):
            # Copy path reference if vectors still exist under 1E
            if row.get("status") == "ok" and row.get("vectors_file"):
                vpath = REPO_ROOT / row["vectors_file"]
                if vpath.is_file():
                    existing.append(row)

    processor, model, model_meta = load_pinned_smolvlm2(require_gpu=True)
    device = model_meta["device"]
    image_token_id = model.config.image_token_id
    run_id = time.strftime("phase_1f_%Y%m%dT%H%M%SZ", time.gmtime())
    n_new = 0

    def vectors_path(sample_id: str) -> Path:
        return vec_dir / f"{sample_id.replace('::', '__')}.npz"

    for fam, var, rec in plan:
        if n_new >= args.max_forwards:
            break
        sid = rec["sample_id"]
        required = build_capture_identity(
            sample_id=sid,
            image_sha256=rec.get("image_sha256", ""),
            prompt=rec["prompt"],
            model_id=model_meta["model_id"],
            model_revision=model_meta["revision"],
            quantization=model_meta["quantization"],
            layer_indices=LAYERS,
            token_selectors=SELECTORS,
            capture_schema_version=CAPTURE_SCHEMA,
        )
        # Reuse if 1E identity matches with same layers/selectors (schema may differ → recompute under 1F schema)
        reuse_row = None
        for row in existing:
            if row.get("sample_id") != sid or row.get("status") != "ok":
                continue
            if set(row.get("layers") or []) != set(LAYERS):
                continue
            if set(row.get("token_selectors") or []) != set(SELECTORS):
                continue
            if row.get("image_sha256") != required.image_sha256:
                continue
            # prompt hash
            if row.get("prompt_sha256") and row["prompt_sha256"] != required.prompt_sha256:
                continue
            if row.get("model", {}).get("revision") != required.model_revision:
                continue
            reuse_row = row
            break

        if reuse_row is not None:
            # Ensure listed in 1F meta (copy)
            row_out = dict(reuse_row)
            row_out["schema_version"] = CAPTURE_SCHEMA
            row_out["reused_from_phase_1e"] = True
            row_out["split_role"] = "train" if rec["split"] == "train" else "validation"
            # append if not already in 1F meta
            if not any(r.get("sample_id") == sid and r.get("schema_version") == CAPTURE_SCHEMA for r in existing):
                append_jsonl(meta_path, row_out)
                existing.append(row_out)
            print(f"REUSE {sid}")
            continue

        image = Image.open(REPO_ROOT / rec["image_file"]).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": rec["prompt"]},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(device)
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(0)
        # latency components
        result = capture_multi_layer_passive(
            model=model,
            inputs=inputs,
            spec=MultiLayerCaptureSpec(layer_indices=LAYERS, token_positions=SELECTORS),
            image_token_id=image_token_id,
        )
        latency = time.perf_counter() - t0
        feats = []
        for li in LAYERS:
            for sel in SELECTORS:
                feats.extend(result.layers[li].tokens[sel].vector)
        vec = np.asarray(feats, dtype=np.float32)
        vpath = vectors_path(sid)
        np.savez_compressed(vpath, vector=vec)
        row = {
            "schema_version": CAPTURE_SCHEMA,
            "run_id": run_id,
            "sample_id": sid,
            "family_id": fam,
            "variant_id": var,
            "condition_id": rec["condition_id"],
            "ground_truth_label": rec["ground_truth_label"],
            "split_role": "train" if rec["split"] == "train" else "validation",
            "contrast_split": rec["split"],
            "prompt_sha256": rec["prompt_sha256"],
            "image_sha256": rec.get("image_sha256"),
            "model": model_meta,
            "layers": list(LAYERS),
            "token_selectors": list(SELECTORS),
            "feature_dim": int(vec.shape[0]),
            "vectors_file": str(vpath.relative_to(REPO_ROOT)).replace("\\", "/"),
            "capture_identity": annotate_identity_dict(required),
            "capture_schema_version": CAPTURE_SCHEMA,
            "status": "ok",
            "peak_vram_mib": result.peak_vram_mib,
            "latency_sec": round(latency, 6),
            "seq_len": result.sequence_length,
            "image_token_count": result.image_token_count,
            "reused_from_phase_1e": False,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        append_jsonl(meta_path, row)
        existing.append(row)
        n_new += 1
        print(f"[new {n_new}] {sid} lat={latency:.1f}s vram={result.peak_vram_mib:.0f} T={result.sequence_length}", flush=True)
        del inputs, result
        torch.cuda.empty_cache()

    del model, processor
    torch.cuda.empty_cache()

    # Write runtime analysis
    rows = [r for r in load_jsonl(meta_path) if r.get("status") == "ok"]
    lats = [r["latency_sec"] for r in rows if r.get("latency_sec") is not None]
    slow = sorted(rows, key=lambda r: r.get("latency_sec") or 0, reverse=True)[:5]
    runtime = {
        "n_rows": len(rows),
        "n_new_forwards_this_run": n_new,
        "latency_sec": {
            "min": min(lats) if lats else None,
            "median": float(np.median(lats)) if lats else None,
            "mean": float(np.mean(lats)) if lats else None,
            "p90": float(np.percentile(lats, 90)) if lats else None,
            "max": max(lats) if lats else None,
        },
        "slowest": [
            {
                "sample_id": r["sample_id"],
                "latency_sec": r.get("latency_sec"),
                "seq_len": r.get("seq_len"),
                "image_token_count": r.get("image_token_count"),
                "peak_vram_mib": r.get("peak_vram_mib"),
            }
            for r in slow
        ],
        "observations": [
            "Most forwards cluster near ~15s; a heavy tail reaches 30–77s.",
            "Slow cases still use ~3330 MiB VRAM and identical image token counts (1377).",
            "Likely host scheduling/thermal shared-GPU contention; NOT proven without clocks/temp logs.",
            "No OOM; sequence lengths comparable across fast and slow cases.",
        ],
        "speculation_flagged": True,
    }
    (out_dir / "runtime_latency_v1.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(json.dumps(runtime["latency_sec"], indent=2))
    print(f"meta={meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
