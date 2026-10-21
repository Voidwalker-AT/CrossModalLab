#!/usr/bin/env python3
"""Phase 1D — Bounded development-only activation + probe pilot.

Caps at --max-forwards (default 54). Uses frozen 1C development families only.
Never touches final holdout. Passive capture only (no patching).
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

from src.config.experiment import get_experiment_config
from src.config.paths import apply_default_cache_environment, get_artifacts_dir
from src.evaluation.behavioral_harness import append_jsonl, load_jsonl
from src.instrumentation.capture_integrity import (
    annotate_identity_dict,
    atomic_write_json,
    build_capture_identity,
    should_skip_capture,
)
from src.instrumentation.multi_layer_capture import (
    MultiLayerCaptureSpec,
    capture_multi_layer_passive,
)
from src.models.model_loader import load_pinned_smolvlm2
from src.probes.confounding import CONDITION_TO_LABEL
from src.probes.linear_probe import (
    ProbeDataset,
    condition_lookup_baseline,
    evaluate_probe,
    fit_standardized_probe,
    group_disjoint_split,
    majority_baseline,
    permutation_test_balanced_accuracy,
    select_l2_grid,
)

apply_default_cache_environment()

LAYERS = (3, 12, 18, 23)
SELECTORS = ("last", "last_text_non_image", "mean_image")
CAPTURE_SCHEMA = "phase_1d_probe_v1"
BALANCED_CONDITIONS = ("both_premises", "both_modified", "underdetermined_fork")


def load_dev_families(manifest_path: Path, n_families: int) -> List[str]:
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    exposed = set(man["development_exposed_families"])
    holdout = set(man["final_holdout_families"])
    # Prefer development families that are not exposed and not holdout
    dev = [f for f in man["development_families"] if f not in exposed and f not in holdout]
    return dev[:n_families]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-families", type=int, default=15)
    p.add_argument("--max-forwards", type=int, default=54)
    p.add_argument("--val-families", type=int, default=4)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    cfg = get_experiment_config()
    split_manifest = json.loads(
        (REPO_ROOT / "configs/phase_1c/frozen_split_manifest_v1.json").read_text(encoding="utf-8")
    )
    exposed = set(split_manifest["development_exposed_families"])
    holdout = set(split_manifest["final_holdout_families"])
    assert args.n_families + args.val_families <= 18

    # Balanced records from dataset v1
    records = [
        json.loads(line)
        for line in (REPO_ROOT / "datasets/cross_modal_dependency_v1/cross_modal_dependency_v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    by_fam_cond = {(r["family_id"], r["condition"]): r for r in records}

    train_fams = load_dev_families(
        REPO_ROOT / "configs/phase_1c/frozen_split_manifest_v1.json", args.n_families + args.val_families
    )
    val_fams = train_fams[-args.val_families :]
    train_fams = train_fams[: args.n_families]
    assert not (set(train_fams) & set(val_fams))
    assert not (set(train_fams) & holdout) and not (set(val_fams) & holdout)
    assert not (set(train_fams) & exposed) and not (set(val_fams) & exposed)

    plan = []
    for fam in train_fams + val_fams:
        for cond in BALANCED_CONDITIONS:
            plan.append((fam, cond, by_fam_cond[(fam, cond)]))
    plan = plan[: args.max_forwards]
    print(f"Plan forwards: {len(plan)} | train_fams={len(train_fams)} val_fams={len(val_fams)}")

    out_dir = args.out_dir or (get_artifacts_dir() / "probes" / "phase_1d")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "probe_pilot_meta_v1.jsonl"
    vec_dir = out_dir / "features"
    vec_dir.mkdir(exist_ok=True)
    existing = load_jsonl(meta_path) if meta_path.is_file() else []

    def vectors_path(sample_id: str) -> Path:
        safe = sample_id.replace("::", "__").replace("/", "_")
        return vec_dir / f"{safe}.npz"

    def row_vectors_path(row: Dict[str, Any]) -> Path | None:
        vf = row.get("vectors_file")
        return (REPO_ROOT / vf) if vf else None

    processor, model, model_meta = load_pinned_smolvlm2(require_gpu=True)
    device = model_meta["device"]
    image_token_id = model.config.image_token_id
    run_id = time.strftime("phase_1d_%Y%m%dT%H%M%SZ", time.gmtime())
    n_completed = 0

    for fam, cond, rec in plan:
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
            verify_vectors=True,
        )
        if decision.skip:
            print(f"SKIP {sid}")
            continue
        if n_completed >= args.max_forwards:
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
        spec = MultiLayerCaptureSpec(layer_indices=LAYERS, token_positions=SELECTORS, notes="phase_1d probe pilot")
        result = capture_multi_layer_passive(
            model=model,
            inputs=inputs,
            spec=spec,
            baseline_logits=None,
            image_token_id=image_token_id,
            input_condition="IMAGE_MULTIMODAL",
        )
        latency = time.perf_counter() - t0

        # Build feature vector: concat last + last_text + mean_image for selected layers
        feats = []
        for li in LAYERS:
            for sel in SELECTORS:
                tok = result.layers[li].tokens[sel]
                feats.extend(tok.vector)
        vec = np.asarray(feats, dtype=np.float32)
        vpath = vectors_path(sid)
        np.savez_compressed(vpath, vector=vec, layers=np.array(LAYERS), selectors=np.array(SELECTORS))

        row = {
            "schema_version": CAPTURE_SCHEMA,
            "run_id": run_id,
            "sample_id": sid,
            "family_id": fam,
            "condition": cond,
            "ground_truth_label": rec["ground_truth_label"],
            "symbolic_status": rec["symbolic_status"],
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
            "has_nonfinite": bool(result.has_nonfinite),
            "logits_max_abs_diff": result.logits_max_abs_diff,
            "peak_vram_mib": result.peak_vram_mib,
            "latency_sec": round(latency, 6),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        append_jsonl(meta_path, row)
        existing.append(row)
        n_completed += 1
        print(
            f"[{n_completed}] {sid} {cond} GT={rec['ground_truth_label']} "
            f"vram={result.peak_vram_mib:.0f} lat={latency:.1f}s dim={vec.shape[0]}",
            flush=True,
        )
        del inputs, result
        torch.cuda.empty_cache()

    del model, processor
    torch.cuda.empty_cache()

    # -------- Probe pilot on collected development features --------
    rows = [r for r in load_jsonl(meta_path) if r.get("status") == "ok"]
    # Only train/val roles; never holdout (should not be present)
    rows = [r for r in rows if r.get("split_role") in {"train", "validation"}]
    # Rebuild arrays
    vecs = []
    labels = []
    groups = []
    metas = []
    for r in rows:
        z = np.load(REPO_ROOT / r["vectors_file"])
        vecs.append(z["vector"])
        labels.append(r["ground_truth_label"])
        groups.append(r["family_id"])
        metas.append(r)
    if len(rows) < 6:
        summary = {"status": "insufficient_samples", "n": len(rows)}
        atomic_write_json(out_dir / "probe_pilot_summary_v1.json", summary)
        print("Insufficient samples for probe pilot")
        return 0

    X = np.stack(vecs, axis=0)
    data = ProbeDataset(X, labels, groups, metas)
    train_g = sorted({r["family_id"] for r in rows if r["split_role"] == "train"})
    val_g = sorted({r["family_id"] for r in rows if r["split_role"] == "validation"})
    if not val_g:
        # engineering fallback: hold out 20% groups deterministically
        all_g = sorted(set(groups))
        val_g = all_g[-max(1, len(all_g) // 5) :]
        train_g = [g for g in all_g if g not in set(val_g)]
    train, val = group_disjoint_split(data, train_groups=train_g, eval_groups=val_g)

    grid = select_l2_grid(train, val, seed=0)
    std, probe = fit_standardized_probe(train, l2=grid["best_l2"], seed=0)
    probe_metrics = evaluate_probe(std, probe, val)
    # Drop bulky fields for JSON
    probe_metrics.pop("y_pred", None)

    y_val = val.y
    baselines = {
        "majority": majority_baseline(y_val),
        "condition_lookup": condition_lookup_baseline(val.meta, y_val),
        "probe_layerwise_note": "single concatenated multi-layer feature vector for pilot",
    }
    # Layer-wise last-token only probes at each layer
    layerwise = {}
    # feature layout: for each layer: last(2048)+last_text(2048)+mean_image(2048)
    dim_tok = 2048
    for li_i, li in enumerate(LAYERS):
        sl = slice(li_i * 3 * dim_tok, (li_i * 3 + 1) * dim_tok)  # last only
        Xs_tr = train.X[:, sl]
        Xs_va = val.X[:, sl]
        tr = ProbeDataset(Xs_tr, train.y, train.group_ids, train.meta)
        va = ProbeDataset(Xs_va, val.y, val.group_ids, val.meta)
        s, pr = fit_standardized_probe(tr, l2=grid["best_l2"], seed=0)
        lm = evaluate_probe(s, pr, va)
        lm.pop("y_pred", None)
        layerwise[f"L{li}_last"] = {
            "balanced_accuracy": lm["balanced_accuracy"],
            "macro_f1": lm["macro_f1"],
        }

    perm = permutation_test_balanced_accuracy(
        train.X, train.y, train.group_ids, n_perm=100, seed=0, l2=grid["best_l2"]
    )

    confounded = baselines["condition_lookup"]["balanced_accuracy"] >= 0.999
    summary = {
        "phase": "1D",
        "run_id": run_id,
        "n_features_total": int(X.shape[0]),
        "n_train": len(train.y),
        "n_val": len(val.y),
        "train_families": train_g,
        "val_families": val_g,
        "final_holdout_touched": False,
        "best_l2": grid["best_l2"],
        "l2_grid": grid["grid"],
        "probe_metrics_multilayer": probe_metrics,
        "baselines": baselines,
        "layerwise_last_token": layerwise,
        "permutation_train": perm,
        "label_condition_confounded": confounded,
        "scientific_verdict": (
            "BLOCKED_FOR_GENUINE_RELATIONAL_INTERPRETATION"
            if confounded
            else "ENGINEERING_PILOT_ONLY_SMALL_N"
        ),
        "interpretation_notes": [
            "Linear decodability is not causal usage.",
            "Condition-label lookup perfect accuracy means 3-way labels are template-confounded.",
            "Small development pilot; not final holdout; not publication metrics.",
        ],
        "peak_vram_mib_max": max((r.get("peak_vram_mib") or 0) for r in rows) if rows else None,
        "mean_latency_sec": float(np.mean([r.get("latency_sec") or 0 for r in rows])) if rows else None,
        "forwards_completed_this_run": n_completed,
    }
    atomic_write_json(out_dir / "probe_pilot_summary_v1.json", summary)
    print(json.dumps({
        "probe_ba": probe_metrics["balanced_accuracy"],
        "condition_lookup_ba": baselines["condition_lookup"]["balanced_accuracy"],
        "majority_ba": baselines["majority"]["balanced_accuracy"],
        "verdict": summary["scientific_verdict"],
        "n": summary["n_features_total"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
