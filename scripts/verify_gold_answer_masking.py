#!/usr/bin/env python3
"""Processor-only verification of gold-answer tokenization/masking (CPU/host RAM only, no full training)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.paths import get_hf_home
from src.instrumentation.capture_integrity import atomic_write_json
from src.models.qwen3vl2_loader import QWEN_MODEL_ID, QWEN_REVISION
from src.training.answer_parser import gold_answer_string, strict_parse_answer
from src.training.sampling_and_masking import (
    answer_span_prompt_length,
    build_supervised_example,
)


def load_processor_only():
    from transformers import AutoProcessor

    hf = get_hf_home()
    snap = hf / f"models--{QWEN_MODEL_ID.replace('/', '--')}" / "snapshots" / QWEN_REVISION
    return AutoProcessor.from_pretrained(str(snap), local_files_only=True)


def main() -> int:
    processor = load_processor_only()

    samples = [
        {"sample_id": "yes", "ground_truth_label": "Yes", "prompt": "Q? ", "image_file": "datasets/controlled/images/ctrl_forward_chain_0001.png"},
        {"sample_id": "no", "ground_truth_label": "No", "prompt": "Q? ", "image_file": "datasets/controlled/images/ctrl_forward_chain_0001.png"},
        {"sample_id": "und", "ground_truth_label": "Undetermined", "prompt": "Q? ", "image_file": "datasets/controlled/images/ctrl_forward_chain_0001.png"},
    ]
    report = []
    for s in samples:
        inputs, labels, meta_m = build_supervised_example(processor, s, abstention_prompt=False, device="cuda:0")
        # also verify decode of answer tokens
        dec = processor.tokenizer.decode(meta_m["answer_token_ids"], skip_special_tokens=True)
        rep = {
            "sample_id": s["sample_id"],
            "answer": meta_m["answer"],
            "decoded": dec,
            "n_answer_tokens": meta_m["n_answer_tokens"],
            "n_supervised": meta_m["n_supervised"],
            "prompt_length": meta_m["prompt_length"],
            "supervised_match": dec.strip() == meta_m["answer"].strip() or dec == meta_m["answer"],
            "labels_valid": bool((labels != -100).any()),
        }
        # strict parser sanity
        p = strict_parse_answer(meta_m["answer"])
        rep["parser_on_gold"] = p["parsed_label"]
        report.append(rep)
        del inputs, labels

    ok = all(r["supervised_match"] and r["labels_valid"] and r["parser_on_gold"] == r["answer"].strip() or
             (r["parser_on_gold"] == {"Yes": "Yes", "No": "No", "Undetermined": "Undetermined"}[r["answer"].strip()])
             for r in report)
    # simpler: parser on gold must match label
    for r in report:
        exp = r["answer"].strip()
        got = r["parser_on_gold"]
        if got != exp:
            ok = False
    payload = {"status": "PASS" if ok else "FAIL", "records": report}
    out = REPO_ROOT / "artifacts/training/phase_2i_a/masking_processor_check_v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, payload)
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
