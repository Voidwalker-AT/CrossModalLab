#!/usr/bin/env python3
"""Phase 0D.2 — Reproducible behavioral evaluation runner.

Loads the pinned SmolVLM2-2.2B NF4 model once, evaluates a fixed manifest
sequentially (batch size 1), and writes append-only JSONL records plus a
metrics summary. Supports resume without duplicate ok-records.

Examples:
  # 3-case smoke
  ulvo run . -- python scripts/run_phase_0d2_behavioral_benchmark.py --smoke

  # full 36-case
  ulvo run . -- python scripts/run_phase_0d2_behavioral_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

# Reduce noisy HF progress bars / non-fatal processor warnings in long runs.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.experiment import get_experiment_config
from src.config.paths import apply_default_cache_environment, get_artifacts_dir
from src.evaluation.behavioral_harness import (
    append_jsonl,
    evaluate_one_sample,
    load_completed_sample_ids,
    load_jsonl,
    sha256_file,
    validate_resume_compatibility,
    write_progress_log,
)
from src.evaluation.behavioral_metrics import compute_behavioral_metrics
from PIL import Image

apply_default_cache_environment()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 0D.2 behavioral benchmark")
    p.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "configs" / "phase_0d2" / "behavioral_manifest_v1.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to artifacts/evaluations/phase_0d2",
    )
    p.add_argument("--smoke", action="store_true", help="Evaluate only first 3 manifest samples")
    p.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Optional explicit sample_id subset from the manifest",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--force", action="store_true", help="Re-run samples already marked ok")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = get_experiment_config()

    if not args.manifest.is_file():
        print(f"ERROR: manifest missing: {args.manifest}")
        print("Run: python scripts/build_phase_0d2_manifest.py")
        return 2

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples: List[Dict[str, Any]] = manifest["samples"]
    manifest_sha = sha256_file(args.manifest)

    if args.sample_ids:
        want = set(args.sample_ids)
        samples = [s for s in samples if s["sample_id"] in want]
    if args.smoke:
        samples = samples[:3]
    # Note: --limit bounds pending samples after resume filtering (see below).

    out_dir = args.out_dir or (get_artifacts_dir() / "evaluations" / "phase_0d2")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("phase_0d2_%Y%m%dT%H%M%SZ", time.gmtime())
    jsonl_path = out_dir / "behavioral_predictions_v1.jsonl"
    metrics_path = out_dir / "behavioral_metrics_v1.json"
    meta_path = out_dir / "run_metadata_v1.json"

    progress_log = out_dir / "progress_v1.log"
    existing_rows = load_jsonl(jsonl_path) if jsonl_path.is_file() else []
    resume_check = validate_resume_compatibility(
        existing_rows=existing_rows,
        manifest_sha256=manifest_sha,
        expected_revision=cfg.model.revision,
        expected_model_id=cfg.model.model_id,
    )
    if existing_rows and not resume_check["safe_to_resume"] and not args.force:
        print("ERROR: existing predictions are not safely resumable:")
        for issue in resume_check["issues"]:
            print(f"  - {issue}")
        print("Refusing to continue. Preserve artifacts and investigate.")
        return 4

    completed = set() if args.force else load_completed_sample_ids(jsonl_path)
    pending = [s for s in samples if s["sample_id"] not in completed]
    # --limit bounds the number of *pending* samples this invocation will run.
    if args.limit is not None:
        pending = pending[: args.limit]

    print("=" * 72)
    print("Phase 0D.2 Behavioral Benchmark")
    print(f"Manifest: {args.manifest}")
    print(f"Manifest SHA-256: {manifest_sha}")
    print(f"Resume check: safe={resume_check['safe_to_resume']} prior_ok={resume_check['n_ok']} issues={len(resume_check['issues'])}")
    print(f"Samples requested: {len(samples)} | already ok: {len(samples) - len(pending)} | pending: {len(pending)}")
    print(f"Output JSONL: {jsonl_path}")
    print(f"Progress log: {progress_log}")
    print(f"Model: {cfg.model.model_id} @ {cfg.model.revision}")
    print("=" * 72)
    write_progress_log(progress_log, f"START run_id={run_id} pending={len(pending)} completed_prior={len(completed)}")

    if not pending:
        print("Nothing pending; recomputing metrics from existing JSONL.")
        write_progress_log(progress_log, f"NOTHING_PENDING run_id={run_id}")
    else:
        from src.models.model_loader import load_pinned_smolvlm2

        print("Loading pinned model (once)...")
        write_progress_log(progress_log, "LOAD_MODEL_START")
        processor, model, model_meta = load_pinned_smolvlm2(require_gpu=True)
        print(f"Model ready on {model_meta['device']} ({model_meta['gpu_name']})")
        print(f"Merger patch: {model_meta['inputs_merger_patch']}")
        write_progress_log(progress_log, f"LOAD_MODEL_OK patch={model_meta['inputs_merger_patch']}")

        max_new = args.max_new_tokens or int(cfg.generation["max_new_tokens_default"])
        device = model_meta["device"]

        for i, sample in enumerate(pending, 1):
            print(f"\n[{i}/{len(pending)}] {sample['sample_id']} | {sample['condition']} | GT={sample['ground_truth_label']}", flush=True)
            write_progress_log(
                progress_log,
                f"SAMPLE_START {sample['sample_id']} idx={i}/{len(pending)}",
            )
            image = None
            if sample.get("modality") != "text_only" and sample.get("image_file"):
                img_path = REPO_ROOT / sample["image_file"]
                if not img_path.is_file():
                    raise FileNotFoundError(f"Missing image: {img_path}")
                image = Image.open(img_path).convert("RGB")

            try:
                rec = evaluate_one_sample(
                    model=model,
                    processor=processor,
                    sample=sample,
                    image=image,
                    candidate_strings=list(cfg.candidate_answer_strings),
                    device=device,
                    max_new_tokens=max_new,
                    do_sample=bool(cfg.generation["do_sample"]),
                    model_meta=model_meta,
                    manifest_sha256=manifest_sha,
                    run_id=run_id,
                )
            except Exception as exc:
                print(f"  FATAL on {sample['sample_id']}: {exc}", flush=True)
                traceback.print_exc()
                fail_rec = {
                    "schema_version": 1,
                    "run_id": run_id,
                    "phase": "0D.2",
                    "sample_id": sample["sample_id"],
                    "manifest_sha256": manifest_sha,
                    "condition": sample["condition"],
                    "ground_truth_label": sample["ground_truth_label"],
                    "prediction_status": "failed",
                    "prediction_error": f"{type(exc).__name__}: {exc}",
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                append_jsonl(jsonl_path, fail_rec)
                write_progress_log(progress_log, f"SAMPLE_FAIL {sample['sample_id']} err={exc}")
                if "out of memory" in str(exc).lower():
                    print("Aborting due to OOM.")
                    write_progress_log(progress_log, "ABORT_OOM")
                    return 3
                continue

            append_jsonl(jsonl_path, rec)
            print(
                f"  pred={rec['parsed_answer']} correct={rec.get('is_correct')} "
                f"latency={rec.get('latency_sec')}s peak_vram={rec.get('peak_vram_mib')}MiB",
                flush=True,
            )
            write_progress_log(
                progress_log,
                f"SAMPLE_OK {sample['sample_id']} pred={rec['parsed_answer']} correct={rec.get('is_correct')} "
                f"lat={rec.get('latency_sec')} vram={rec.get('peak_vram_mib')}",
            )

        # free GPU before metrics
        del model, processor
        import torch

        torch.cuda.empty_cache()

    # Metrics from full JSONL (ok records only for accuracy; all for failure count)
    all_rows = load_jsonl(jsonl_path)
    # Prefer unique latest ok per sample_id
    latest_ok: Dict[str, Dict[str, Any]] = {}
    failures = []
    for row in all_rows:
        sid = row.get("sample_id")
        if row.get("prediction_status") == "ok":
            latest_ok[sid] = row
        elif row.get("prediction_status") in {"failed", "failed_oom"}:
            failures.append(row)

    metrics = compute_behavioral_metrics(latest_ok.values())
    metrics["run_id"] = run_id
    metrics["manifest_path"] = str(args.manifest)
    metrics["manifest_sha256"] = manifest_sha
    metrics["n_failure_records"] = len(failures)
    metrics["candidate_score_semantics"] = "uncalibrated_candidate_normalized_softmax"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    run_meta = {
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_sha256": manifest_sha,
        "n_samples_in_manifest": manifest.get("n_samples"),
        "n_evaluated_ok": metrics["n_scored"],
        "n_failed_or_missing": metrics["n_failed_or_missing"],
        "smoke": bool(args.smoke),
        "sample_ids": [s["sample_id"] for s in samples],
        "output_jsonl": str(jsonl_path),
        "output_metrics": str(metrics_path),
    }
    meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"Scored cases: {metrics['n_scored']}")
    print(f"Overall accuracy (95% bootstrap CI): {metrics['overall_accuracy']}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']}")
    print(f"Underdetermined abstention acc: {metrics['underdetermined_abstention_accuracy']}")
    print(f"Underdetermined definite rate: {metrics['underdetermined_definite_answer_rate']}")
    print(f"Metrics: {metrics_path}")
    print(f"JSONL:   {jsonl_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
