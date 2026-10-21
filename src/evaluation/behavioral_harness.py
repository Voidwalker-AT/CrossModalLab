"""Append-only behavioral evaluation harness for Phase 0D.2.

Design:
- Manifest is fixed before inference and referenced by SHA-256.
- Records are append-only JSONL keyed by sample_id for safe resume.
- No ground-truth mutation; parsing is separate from labels.
- Candidate-normalized scores are labeled uncalibrated.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from src.evaluation.behavioral_metrics import build_prompt, parse_model_answer


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(64 * 1024):
            h.update(chunk)
    return h.hexdigest()


def sha256_image_bytes(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_completed_sample_ids(path: Path) -> Set[str]:
    if not path.is_file():
        return set()
    ids: Set[str] = set()
    for row in load_jsonl(path):
        sid = row.get("sample_id")
        status = row.get("prediction_status")
        if sid and status == "ok":
            ids.add(sid)
    return ids


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append one JSONL record and flush durably to disk."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def validate_resume_compatibility(
    *,
    existing_rows: List[Dict[str, Any]],
    manifest_sha256: str,
    expected_revision: str,
    expected_model_id: str,
) -> Dict[str, Any]:
    """Validate partial JSONL can be safely resumed against the frozen manifest/model."""
    ok_ids: List[str] = []
    issues: List[str] = []
    for row in existing_rows:
        sid = row.get("sample_id", "<missing>")
        if row.get("prediction_status") != "ok":
            continue
        ok_ids.append(sid)
        if row.get("manifest_sha256") != manifest_sha256:
            issues.append(f"{sid}: manifest_sha256 mismatch")
        model = row.get("model") or {}
        if model.get("revision") != expected_revision:
            issues.append(f"{sid}: model revision mismatch")
        if model.get("model_id") != expected_model_id:
            issues.append(f"{sid}: model_id mismatch")
        scores = row.get("candidate_scores") or {}
        for cand, payload in scores.items():
            for key in ("total_log_likelihood", "length_normalized_log_likelihood"):
                val = payload.get(key)
                if val is None or val != val:  # NaN check
                    issues.append(f"{sid}: non-finite {key} for {cand}")
                    break

    dupes = sorted({sid for sid in ok_ids if ok_ids.count(sid) > 1})
    if dupes:
        issues.append(f"duplicate ok sample_ids: {dupes}")

    return {
        "n_ok": len(set(ok_ids)),
        "ok_ids": sorted(set(ok_ids)),
        "duplicate_ok_ids": dupes,
        "issues": issues,
        "safe_to_resume": len(issues) == 0,
    }


def write_progress_log(path: Path, message: str) -> None:
    """Append a timestamped progress line to a local artifact log."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def build_messages_for_sample(sample: Dict[str, Any], image: Any) -> List[Dict[str, Any]]:
    modality = sample.get("modality", "visual")
    if modality == "text_only" or image is None:
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": sample["prompt"]}],
            }
        ]
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["prompt"]},
            ],
        }
    ]


def evaluate_one_sample(
    *,
    model: Any,
    processor: Any,
    sample: Dict[str, Any],
    image: Any,
    candidate_strings: List[str],
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    model_meta: Dict[str, Any],
    manifest_sha256: str,
    run_id: str,
) -> Dict[str, Any]:
    """Run greedy decode + candidate scoring for one manifest sample."""
    import torch

    record: Dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "phase": "0D.2",
        "sample_id": sample["sample_id"],
        "manifest_sha256": manifest_sha256,
        "condition": sample["condition"],
        "topology": sample.get("topology"),
        "modality": sample.get("modality"),
        "relation": sample.get("relation"),
        "matched_group_id": sample.get("matched_group_id"),
        "ground_truth_label": sample["ground_truth_label"],
        "symbolic_epistemic_status": sample.get("symbolic_epistemic_status"),
        "text_premise": sample.get("text_premise"),
        "query": sample.get("query"),
        "image_file": sample.get("image_file"),
        "image_sha256": sample.get("image_sha256"),
        "is_synthetic_demo": sample.get("is_synthetic_demo", False),
        "prompt_template_id": sample.get("prompt_template_id"),
        "prompt": sample["prompt"],
        "model": model_meta,
        "generation_config": {
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "seed": sample.get("seed"),
        },
        "prediction_status": "ok",
        "prediction_error": None,
        "raw_completion": None,
        "parsed_answer": None,
        "parse_status": None,
        "candidate_scores": None,
        "candidate_ranking_length_norm": None,
        "candidate_ranking_raw": None,
        "candidate_probabilities_length_norm_uncalibrated": None,
        "candidate_probabilities_raw_uncalibrated": None,
        "latency_sec": None,
        "peak_vram_mib": None,
        "reproduction_command": (
            "ulvo run . -- python scripts/run_phase_0d2_behavioral_benchmark.py "
            f"--sample-ids {sample['sample_id']}"
        ),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    try:
        from src.evaluation.candidate_scorer import score_candidate_answers

        messages = build_messages_for_sample(sample, image)
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(0)

        prompt_len = inputs["input_ids"].shape[1]
        t0 = time.perf_counter()
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - t0

        new_ids = out_ids[0, prompt_len:].tolist()
        raw_completion = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        parsed = parse_model_answer(raw_completion)

        cand = score_candidate_answers(
            model=model,
            processor=processor,
            base_inputs=inputs,
            candidate_strings=candidate_strings,
            device=device,
        )

        peak = None
        if device.startswith("cuda"):
            peak = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 3)
            torch.cuda.empty_cache()

        record.update(
            {
                "raw_completion": raw_completion,
                "parsed_answer": parsed["parsed_label"],
                "parse_status": parsed["parse_status"],
                "completion_token_ids": new_ids,
                "prompt_token_length": prompt_len,
                "completion_token_length": len(new_ids),
                "candidate_scores": {
                    k: {
                        "total_log_likelihood": v.total_log_likelihood,
                        "length_normalized_log_likelihood": v.length_normalized_log_likelihood,
                        "sequence_length": v.sequence_length,
                        "token_strings": v.token_strings,
                        "token_log_probs": v.token_log_probs,
                    }
                    for k, v in cand.candidates.items()
                },
                "candidate_ranking_length_norm": cand.length_normalized_ranking,
                "candidate_ranking_raw": cand.raw_ranking,
                "candidate_probabilities_length_norm_uncalibrated": cand.candidate_probabilities_normalized,
                "candidate_probabilities_raw_uncalibrated": cand.candidate_probabilities_raw,
                "length_rank_flip": cand.length_rank_flip,
                "latency_sec": round(latency, 6),
                "peak_vram_mib": peak,
                "is_correct": parsed["parsed_label"] == sample["ground_truth_label"],
            }
        )
    except Exception as exc:  # keep run going; mark failure explicitly
        record["prediction_status"] = "failed"
        record["prediction_error"] = f"{type(exc).__name__}: {exc}"
        record["is_correct"] = False
        if str(type(exc)).find("OutOfMemory") >= 0 or "out of memory" in str(exc).lower():
            record["prediction_status"] = "failed_oom"
            raise

    return record
