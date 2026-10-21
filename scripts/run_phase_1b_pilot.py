#!/usr/bin/env python3
"""Phase 1B — Bounded multimodal pilot on cross-modal dependency conditions.

Caps at 36 VLM inferences by default (12 families × 3 conditions).
"""

from __future__ import annotations

import argparse
import hashlib
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
from src.evaluation.behavioral_harness import append_jsonl, load_completed_sample_ids
from src.evaluation.behavioral_metrics import parse_model_answer
from src.models.model_loader import load_pinned_smolvlm2

apply_default_cache_environment()

PILOT_CONDITIONS = (
    "both_premises",
    "visual_removed",
    "text_removed",
    "underdetermined_fork",
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_pilot_records(records: List[Dict[str, Any]], n_families: int) -> List[Dict[str, Any]]:
    """Select n_families families × 3 primary conditions (capped later at 36)."""
    families: Dict[str, Dict[str, Any]] = {}
    for r in records:
        families.setdefault(r["family_id"], {})[r["condition"]] = r

    # Prefer test split families first, then validation, for pilot honesty
    ordered = []
    for split in ("test", "validation", "train"):
        fids = sorted({r["family_id"] for r in records if r["split"] == split})
        ordered.extend(fids)

    selected: List[Dict[str, Any]] = []
    used = 0
    for fid in ordered:
        if used >= n_families:
            break
        fam = families[fid]
        for cond in ("both_premises", "visual_removed", "text_removed"):
            if cond in fam:
                selected.append(fam[cond])
        used += 1
    return selected


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-families", type=int, default=12)
    p.add_argument("--max-inferences", type=int, default=36)
    p.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "datasets" / "cross_modal_dependency_v1" / "cross_modal_dependency_v1.jsonl",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    cfg = get_experiment_config()
    records = load_jsonl(args.dataset)
    pilot = select_pilot_records(records, args.n_families)[: args.max_inferences]
    print(f"Pilot size: {len(pilot)} inferences from {args.n_families} families")

    out_dir = args.out_dir or (get_artifacts_dir() / "evaluations" / "phase_1b")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "phase_1b_pilot_predictions_v1.jsonl"
    done = load_completed_sample_ids(jsonl_path)
    pending = [r for r in pilot if r["sample_id"] not in done]
    print(f"Already ok: {len(pilot) - len(pending)} | pending: {len(pending)}")

    processor, model, model_meta = load_pinned_smolvlm2(require_gpu=True)
    device = model_meta["device"]
    run_id = time.strftime("phase_1b_%Y%m%dT%H%M%SZ", time.gmtime())

    from src.evaluation.candidate_scorer import score_candidate_answers

    for i, rec in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] {rec['sample_id']} | {rec['condition']} | GT={rec['ground_truth_label']}", flush=True)
        img_path = REPO_ROOT / rec["image_file"]
        image = Image.open(img_path).convert("RGB")
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
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=int(cfg.generation["max_new_tokens_default"]),
                do_sample=bool(cfg.generation["do_sample"]),
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - t0
        new_ids = out_ids[0, inputs["input_ids"].shape[1] :].tolist()
        raw = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        parsed = parse_model_answer(raw)
        cand = score_candidate_answers(
            model=model,
            processor=processor,
            base_inputs=inputs,
            candidate_strings=list(cfg.candidate_answer_strings),
            device=device,
        )
        peak = torch.cuda.max_memory_allocated(0) / (1024 * 1024)

        row = {
            "schema_version": "phase_1b_pilot_v1",
            "run_id": run_id,
            "sample_id": rec["sample_id"],
            "family_id": rec["family_id"],
            "condition": rec["condition"],
            "evidence_type": rec["evidence_type"],
            "split": rec["split"],
            "ground_truth_label": rec["ground_truth_label"],
            "symbolic_status": rec["symbolic_status"],
            "parsed_answer": parsed["parsed_label"],
            "raw_completion": raw,
            "is_correct": parsed["parsed_label"] == rec["ground_truth_label"],
            "prompt_sha256": rec["prompt_sha256"],
            "image_sha256": rec["image_sha256"],
            "model": model_meta,
            "candidate_ranking_length_norm": cand.length_normalized_ranking,
            "candidate_probabilities_length_norm_uncalibrated": cand.candidate_probabilities_normalized,
            "latency_sec": round(latency, 6),
            "peak_vram_mib": round(peak, 3),
            "prediction_status": "ok",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        append_jsonl(jsonl_path, row)
        print(f"  pred={row['parsed_answer']} correct={row['is_correct']} lat={latency:.2f}s vram={peak:.0f}", flush=True)
        del inputs, out_ids, cand
        torch.cuda.empty_cache()

    del model, processor
    torch.cuda.empty_cache()

    # Summary metrics
    all_rows = load_jsonl(jsonl_path)
    latest = {r["sample_id"]: r for r in all_rows if r.get("prediction_status") == "ok"}
    by_cond: Dict[str, List[bool]] = {}
    for r in latest.values():
        by_cond.setdefault(r["condition"], []).append(bool(r["is_correct"]))
    summary = {
        "run_id": run_id,
        "n_predictions": len(latest),
        "n_families": len({r["family_id"] for r in latest.values()}),
        "accuracy_by_condition": {
            k: {"n": len(v), "acc": sum(v) / len(v)} for k, v in sorted(by_cond.items())
        },
        "output_jsonl": str(jsonl_path),
        "descriptive_behavioral": True,
        "notes": [
            "Bounded pilot; not publication performance.",
            "Answer changes alone do not prove internal cross-modal composition.",
        ],
    }
    (out_dir / "phase_1b_pilot_summary_v1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["accuracy_by_condition"], indent=2))
    print(f"Wrote {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
