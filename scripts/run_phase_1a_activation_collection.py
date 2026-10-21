#!/usr/bin/env python3
"""Phase 1A — Sparse multi-layer passive activation collection.

Passive only: no patching, no probes, no training.
Modes:
  --smoke : 1 multimodal sample, layers 3 and 12 (memory preflight)
  default : full frozen case set, layers 3/12/18/23 (if preflight ok)
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

import torch
from PIL import Image

from src.config.experiment import get_experiment_config
from src.config.paths import apply_default_cache_environment, get_artifacts_dir
from src.evaluation.behavioral_metrics import build_prompt
from src.instrumentation.multi_layer_capture import (
    MultiLayerCaptureSpec,
    capture_multi_layer_passive,
    classify_input_condition,
    true_text_only_inputs,
)
from src.models.model_loader import load_pinned_smolvlm2

apply_default_cache_environment()

PASSIVE_LOGIT_TOL = 1e-5  # documented numerical tolerance for multi-hook path


def load_manifest_and_preds():
    man = json.loads((REPO_ROOT / "configs/phase_0d2/behavioral_manifest_v1.json").read_text(encoding="utf-8"))
    preds = {}
    path = REPO_ROOT / "artifacts/evaluations/phase_0d2/behavioral_predictions_v1.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("prediction_status") == "ok":
                preds[row["sample_id"]] = row
    return man, preds


def build_image_inputs(processor, sample) -> Dict[str, Any]:
    img = Image.open(REPO_ROOT / sample["image_file"]).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": sample["prompt"]},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="1 sample, layers 3+12 only")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    cfg = get_experiment_config()
    sel = json.loads((REPO_ROOT / "configs/phase_1a/activation_case_selection.json").read_text(encoding="utf-8"))
    man, preds = load_manifest_and_preds()
    man_by = {s["sample_id"]: s for s in man["samples"]}
    layers = tuple(sel["layers"])
    if args.smoke:
        layers = (3, 12)
        cases = [sel["cases"][0]]  # correct forward 0019
    else:
        cases = sel["cases"]

    processor, model, model_meta = load_pinned_smolvlm2(require_gpu=True)
    device = model_meta["device"]
    image_token_id = model.config.image_token_id
    run_id = time.strftime("phase_1a_%Y%m%dT%H%M%SZ", time.gmtime())

    out_dir = args.out_dir or (get_artifacts_dir() / "instrumentation" / "phase_1a")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "activations_meta_v1.jsonl"
    vectors_dir = out_dir / "vectors"
    vectors_dir.mkdir(exist_ok=True)

    # resume: skip completed sample_ids with ok status
    done = set()
    if jsonl_path.is_file():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "ok":
                    done.add(row["sample_id"])

    records: List[Dict[str, Any]] = []
    smoke_ok = True

    for case in cases:
        sid = case["sample_id"]
        if sid in done:
            print(f"SKIP already ok: {sid}")
            continue
        sample = man_by[sid]
        pred = preds[sid]
        print(f"\n=== CAPTURE {sid} layers={layers} ===", flush=True)

        inputs = build_image_inputs(processor, sample).to(device)
        cond = classify_input_condition(
            image_token_count=int((inputs["input_ids"] == image_token_id).sum()),
            has_pixel_values="pixel_values" in inputs,
        )
        # Override taxonomy for known blank-image controls
        if case.get("analytical_label") == "BLANK_IMAGE_MULTIMODAL" or sample["condition"] == "text_premise_control":
            cond = "BLANK_IMAGE_MULTIMODAL"

        with torch.inference_mode():
            baseline = model(**inputs).logits[0, -1, :].detach().to(torch.float32)
        torch.cuda.empty_cache()

        spec = MultiLayerCaptureSpec(
            layer_indices=layers,
            token_positions=tuple(sel["token_selectors"]),
            notes="Phase 1A sparse multi-layer passive capture",
        )
        result = capture_multi_layer_passive(
            model=model,
            inputs=inputs,
            spec=spec,
            baseline_logits=baseline,
            image_token_id=image_token_id,
            input_condition=cond,
        )

        passive_pass = (
            result.logits_identical
            or result.logits_max_abs_diff <= PASSIVE_LOGIT_TOL
        ) and not result.has_nonfinite
        if not passive_pass:
            smoke_ok = False
            print(f"  FAIL passivity/nonfinite for {sid}: diff={result.logits_max_abs_diff}")

        # save vectors
        vec_path = vectors_dir / f"{sid}.json"
        vec_path.write_text(
            json.dumps(result.summary_dict(include_vectors=True), indent=2),
            encoding="utf-8",
        )

        meta = {
            "run_id": run_id,
            "phase": "1A",
            "sample_id": sid,
            "status": "ok" if passive_pass else "failed_passivity",
            "role": case["role"],
            "genuinely_paired": case.get("genuinely_paired"),
            "matched_group_id": case.get("matched_group_id"),
            "reason_for_selection": case["reason"],
            "condition_0d2": sample["condition"],
            "analytical_input_condition": cond,
            "ground_truth_label": pred["ground_truth_label"],
            "parsed_answer": pred["parsed_answer"],
            "is_correct_0d2": pred.get("is_correct"),
            "image_sha256": sample.get("image_sha256"),
            "model": model_meta,
            "layers": list(layers),
            "sequence_length": result.sequence_length,
            "image_token_count": result.image_token_count,
            "logits_max_abs_diff": result.logits_max_abs_diff,
            "logits_identical": result.logits_identical,
            "logit_tolerance": PASSIVE_LOGIT_TOL,
            "has_nonfinite": result.has_nonfinite,
            "peak_vram_mib": result.peak_vram_mib,
            "host_ram_used_mib": result.host_ram_used_mib,
            "vectors_file": str(vec_path.relative_to(REPO_ROOT)),
            "descriptive_non_causal": True,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        records.append(meta)
        print(
            f"  cond={cond} T={result.sequence_length} img_tok={result.image_token_count} "
            f"diff={result.logits_max_abs_diff} peak_vram={result.peak_vram_mib:.1f} "
            f"pass={passive_pass}",
            flush=True,
        )

        del inputs, baseline, result
        torch.cuda.empty_cache()

    # True text-only diagnostics (not mixed into multimodal aggregates)
    if not args.smoke:
        tt_results = []
        for prompt in sel["true_text_only_diagnostic"]["prompts"]:
            inputs = true_text_only_inputs(processor, prompt["text"]).to(device)
            n_img = int((inputs["input_ids"] == image_token_id).sum())
            assert n_img == 0 and "pixel_values" not in inputs
            with torch.inference_mode():
                baseline = model(**inputs).logits[0, -1, :].detach().to(torch.float32)
            spec = MultiLayerCaptureSpec(layer_indices=(3, 12, 18, 23), token_positions=("last", "last_text_non_image"))
            result = capture_multi_layer_passive(
                model=model,
                inputs=inputs,
                spec=spec,
                baseline_logits=baseline,
                image_token_id=image_token_id,
                input_condition="TRUE_TEXT_ONLY",
            )
            rec = {
                "run_id": run_id,
                "phase": "1A",
                "sample_id": f"true_text_only::{prompt['id']}",
                "status": "ok",
                "role": "true_text_only_diagnostic",
                "analytical_input_condition": "TRUE_TEXT_ONLY",
                "expected_logic_label": prompt["expected_logic_label"],
                "image_token_count": result.image_token_count,
                "sequence_length": result.sequence_length,
                "logits_max_abs_diff": result.logits_max_abs_diff,
                "has_nonfinite": result.has_nonfinite,
                "peak_vram_mib": result.peak_vram_mib,
                "vectors_file": None,
                "descriptive_non_causal": True,
                "summary": result.summary_dict(include_vectors=False),
            }
            tt_path = out_dir / f"true_text_only_{prompt['id']}.json"
            tt_path.write_text(json.dumps({**rec, "vectors": result.summary_dict(include_vectors=True)}, indent=2), encoding="utf-8")
            tt_results.append(rec)
            del inputs, baseline, result
            torch.cuda.empty_cache()
        (out_dir / "true_text_only_diagnostics.json").write_text(json.dumps(tt_results, indent=2), encoding="utf-8")

    # run summary
    summary = {
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model_meta,
        "layers": list(layers),
        "n_captured": len(records),
        "all_passivity_ok": all(r["status"] == "ok" for r in records),
        "passive_logit_tolerance": PASSIVE_LOGIT_TOL,
        "artifact_jsonl": str(jsonl_path),
        "descriptive_non_causal": True,
    }
    (out_dir / "run_summary_v1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    del model, processor
    torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print(f"Captured {summary['n_captured']} multimodal cases")
    print(f"All passivity OK: {summary['all_passivity_ok']}")
    print(f"JSONL: {jsonl_path}")
    print(f"Summary: {out_dir / 'run_summary_v1.json'}")
    print("=" * 72)
    return 0 if summary["all_passivity_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
