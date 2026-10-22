#!/usr/bin/env python3
"""Phase 2G — Frozen development-validation baseline + optional adapter eval (CPU-safe eval helpers used by GPU runner)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.training.qlora_train_utils import eval_metrics


def summarize_eval(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    y_true = [r["ground_truth_label"] for r in rows]
    y_pred = [r["parsed_answer"] for r in rows]
    fam = sorted({r["family_id"] for r in rows})
    return {
        "n": len(rows),
        "n_families": len(fam),
        "families": fam,
        "sample_ids": [r["sample_id"] for r in rows],
        "metrics": eval_metrics(y_true, y_pred),
        "confusion": {
            t: dict(Counter_p([r["parsed_answer"] for r in rows if r["ground_truth_label"] == t]))
            for t in ("Yes", "No", "Undetermined")
        },
    }


def Counter_p(vals):
    from collections import Counter

    return Counter(vals)


def freeze_val_ids(manifest_path: Path, n_families: int = 8) -> List[Dict[str, Any]]:
    """Pick development-validation families from contrast-v2 (never test split)."""
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    test_fams = {r["family_id"] for r in records if r["split"] == "test"}
    by_fam: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        if r["split"] == "test":
            continue  # never train/eval on holdout
        by_fam.setdefault(r["family_id"], []).append(r)
    val_fams = sorted({r["family_id"] for r in records if r["split"] == "validation"} - test_fams)
    train_fams = sorted({r["family_id"] for r in records if r["split"] == "train"} - test_fams)
    selected = val_fams[:n_families]
    if len(selected) < n_families:
        raise RuntimeError(f"not enough validation families: {len(val_fams)}")
    cases = []
    for fid in selected:
        if fid in test_fams:
            raise RuntimeError(f"selected family {fid} is holdout")
        for r in sorted(by_fam[fid], key=lambda x: x["variant_id"]):
            if r["variant_id"] not in {"entailed", "contradicted", "underdetermined"}:
                continue
            cases.append(
                {
                    "sample_id": r["sample_id"],
                    "family_id": fid,
                    "variant_id": r["variant_id"],
                    "ground_truth_label": r["ground_truth_label"],
                    "prompt": r["prompt"],
                    "prompt_sha256": r["prompt_sha256"],
                    "image_file": r["image_file"],
                    "image_sha256": r["image_sha256"],
                }
            )
    return {"val_families": selected, "n_val_families": len(selected), "n_cases": len(cases), "cases": cases,
            "train_families_sample": train_fams[:5], "n_train_families": len(train_fams)}
