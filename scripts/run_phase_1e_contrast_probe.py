#!/usr/bin/env python3
"""Phase 1E — Bounded development probe revalidation on contrast v2.

Max 54 forwards; no final holdout; no patching.
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

from src.benchmark.contrast_v2 import LABELS, load_contrast_records
from src.config.paths import apply_default_cache_environment, get_artifacts_dir
from src.evaluation.behavioral_harness import append_jsonl, load_jsonl
from src.instrumentation.capture_integrity import (
    annotate_identity_dict,
    atomic_write_json,
    build_capture_identity,
    should_skip_capture,
)
from src.instrumentation.multi_layer_capture import MultiLayerCaptureSpec, capture_multi_layer_passive
from src.models.model_loader import load_pinned_smolvlm2
from src.probes.linear_probe import (
    ProbeDataset,
    evaluate_probe,
    fit_standardized_probe,
    group_disjoint_split,
    majority_baseline,
    select_l2_grid,
)

apply_default_cache_environment()
LAYERS = (3, 12, 18, 23)
SELECTORS = ("last", "last_text_non_image", "mean_image")
CAPTURE_SCHEMA = "phase_1e_contrast_v1"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-families", type=int, default=15)
    p.add_argument("--val-families", type=int, default=3)
    p.add_argument("--max-forwards", type=int, default=54)
    p.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "datasets/cross_modal_contrast_v2/cross_modal_contrast_v2.jsonl",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    records = load_contrast_records(args.dataset)
    by_fam_var = {(r["family_id"], r["variant_id"]): r for r in records}
    split1c = json.loads((REPO_ROOT / "configs/phase_1c/frozen_split_manifest_v1.json").read_text(encoding="utf-8"))
    holdout1c = set(split1c["final_holdout_families"])
    exposed = set(split1c["development_exposed_families"])

    # Contrast v2 uses new family ids (xc_*) — no overlap with 1C holdout by construction.
    dev_fams = sorted({r["family_id"] for r in records if r["split"] in {"train", "validation", "test"}})
    # For pilot: use train+validation only; exclude nothing from xc ids vs 1C
    dev_fams = sorted({r["family_id"] for r in records if r["split"] in {"train", "validation"}})
    val_n = args.val_families
    train_n = args.n_families
    val_fams = dev_fams[-val_n:]
    train_fams = dev_fams[:train_n]
    assert set(train_fams).isdisjoint(val_fams)
    assert set(train_fams).isdisjoint(holdout1c)
    assert set(val_fams).isdisjoint(holdout1c)

    plan = []
    for fam in train_fams + val_fams:
        for var in ("entailed", "contradicted", "underdetermined"):
            plan.append((fam, var, by_fam_var[(fam, var)]))
    plan = plan[: args.max_forwards]
    print(f"Forwards={len(plan)} train_fams={len(train_fams)} val_fams={len(val_fams)}")

    out_dir = args.out_dir or (get_artifacts_dir() / "probes" / "phase_1e")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "contrast_probe_meta_v1.jsonl"
    vec_dir = out_dir / "features"
    vec_dir.mkdir(exist_ok=True)
    existing = load_jsonl(meta_path) if meta_path.is_file() else []

    processor, model, model_meta = load_pinned_smolvlm2(require_gpu=True)
    device = model_meta["device"]
    image_token_id = model.config.image_token_id
    run_id = time.strftime("phase_1e_%Y%m%dT%H%M%SZ", time.gmtime())
    n_done = 0

    for fam, var, rec in plan:
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
        decision = should_skip_capture(
            existing_rows=existing,
            required_identity=required,
            vectors_file_for_row=lambda r: (REPO_ROOT / r["vectors_file"]) if r.get("vectors_file") else None,
            verify_vectors=False,
        )
        if decision.skip:
            print(f"SKIP {sid}")
            continue
        if n_done >= args.max_forwards:
            break

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
        torch.cuda.reset_peak_memory_stats(0)
        t0 = time.perf_counter()
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
        safe = sid.replace("::", "__")
        vpath = vec_dir / f"{safe}.npz"
        np.savez_compressed(vpath, vector=vec)
        row = {
            "schema_version": CAPTURE_SCHEMA,
            "run_id": run_id,
            "sample_id": sid,
            "family_id": fam,
            "variant_id": var,
            "condition_id": rec["condition_id"],
            "ground_truth_label": rec["ground_truth_label"],
            "split_role": "train" if fam in train_fams else "validation",
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
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        append_jsonl(meta_path, row)
        existing.append(row)
        n_done += 1
        print(f"[{n_done}] {sid} GT={rec['ground_truth_label']} vram={result.peak_vram_mib:.0f} lat={latency:.1f}s", flush=True)
        del inputs, result
        torch.cuda.empty_cache()

    del model, processor
    torch.cuda.empty_cache()

    rows = [r for r in load_jsonl(meta_path) if r.get("status") == "ok"]
    vecs, labels, groups, metas = [], [], [], []
    for r in rows:
        z = np.load(REPO_ROOT / r["vectors_file"])
        vecs.append(z["vector"])
        labels.append(r["ground_truth_label"])
        groups.append(r["family_id"])
        metas.append(r)
    if len(rows) < 6:
        atomic_write_json(out_dir / "contrast_probe_summary_v1.json", {"status": "insufficient", "n": len(rows)})
        return 0

    X = np.stack(vecs)
    data = ProbeDataset(X, labels, groups, metas)
    train_g = sorted({r["family_id"] for r in rows if r["split_role"] == "train"})
    val_g = sorted({r["family_id"] for r in rows if r["split_role"] == "validation"})
    train, val = group_disjoint_split(data, train_groups=train_g, eval_groups=val_g)
    grid = select_l2_grid(train, val, seed=0)
    std, probe = fit_standardized_probe(train, l2=grid["best_l2"], seed=0)
    metrics = evaluate_probe(std, probe, val)

    y_val = val.y
    # Constant condition baseline: single condition → majority
    cond_pred = [max(set(y_val), key=y_val.count)] * len(y_val)
    from src.instrumentation.capture_integrity import compute_probe_metrics

    baselines = {
        "condition_only": compute_probe_metrics(y_val, cond_pred, list(LABELS)),
        "majority": majority_baseline(y_val),
    }
    # Layer-wise last token
    dim_tok = 2048
    layerwise = {}
    for li_i, li in enumerate(LAYERS):
        sl = slice(li_i * 3 * dim_tok, (li_i * 3 + 1) * dim_tok)
        tr = ProbeDataset(train.X[:, sl], train.y, train.group_ids, train.meta)
        va = ProbeDataset(val.X[:, sl], val.y, val.group_ids, val.meta)
        s2, p2 = fit_standardized_probe(tr, l2=grid["best_l2"], seed=0)
        lm = evaluate_probe(s2, p2, va)
        layerwise[f"L{li}_last"] = {
            "balanced_accuracy": lm["balanced_accuracy"],
            "macro_f1": lm["macro_f1"],
            "per_class_recall": lm["per_class_recall"],
            "confusion_matrix": lm["confusion_matrix"],
        }

    summary = {
        "phase": "1E",
        "run_id": run_id,
        "n": int(X.shape[0]),
        "n_train": len(train.y),
        "n_val": len(val.y),
        "train_families": train_g,
        "val_families": val_g,
        "final_holdout_1c_touched": False,
        "condition_id": "CROSS_MODAL_TWO_PREMISE",
        "probe_multilayer": {k: metrics[k] for k in metrics if k != "y_pred"},
        "baselines": baselines,
        "layerwise_last_token": layerwise,
        "best_l2": grid["best_l2"],
        "forwards_this_run": n_done,
        "peak_vram_mib_max": max(r.get("peak_vram_mib") or 0 for r in rows),
        "mean_latency_sec": float(np.mean([r.get("latency_sec") or 0 for r in rows])),
        "scientific_notes": [
            "Decodability is not causal usage.",
            "Small development pilot; exploratory metrics only.",
            "Single condition_id; labels balanced within family.",
            "Not final holdout evaluation.",
        ],
    }
    atomic_write_json(out_dir / "contrast_probe_summary_v1.json", summary)
    print(json.dumps({
        "probe_ba": metrics["balanced_accuracy"],
        "condition_only_ba": baselines["condition_only"]["balanced_accuracy"],
        "majority_ba": baselines["majority"]["balanced_accuracy"],
        "n": summary["n"],
        "layerwise": {k: v["balanced_accuracy"] for k, v in layerwise.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
