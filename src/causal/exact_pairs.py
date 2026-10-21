"""Generate a small exact-image / exact-text counterfactual pair dataset (v3).

Uses the contrast_v2 visual layouts but reuses a single image file for
EXACT_IMAGE_CHANGED_TEXT pairs (entailed vs underdetermined share one canvas).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from src.benchmark.contrast_v2 import (
    CONDITION_ID,
    LABELS,
    VARIANTS,
    build_prompt_with_text_pair,
    render_canvas,
    variant_premises,
)
from src.config.paths import REPO_ROOT
from src.symbolic.relations import DirectionalRelation, Modality, Premise, Query
from src.symbolic.solver import SymbolicSolver
import random


def build_exact_image_pair_records(
    family_id: str,
    a: str,
    b: str,
    c: str,
    *,
    out_images: Path,
    seed: int = 20261110,
) -> List[Dict[str, Any]]:
    """Entailed vs underdetermined with identical image bytes."""
    rng = random.Random(seed)
    # Render once for visual premise R(A,B)
    img_name = f"{family_id}__exact_shared.png"
    img_path = out_images / img_name
    meta = render_canvas(a, b, img_path, rng, bg_jitter=0, gap_scale=1.0)

    records = []
    for variant in ("entailed", "underdetermined"):
        premises, _ = variant_premises(variant, a, b, c)
        text_p = [p for p in premises if p.modality == Modality.TEXTUAL][0]
        res = SymbolicSolver.solve(
            premises, Query(a, c, DirectionalRelation.LEFT_OF), verify_with_oracle=True
        )
        status = res.epistemic_status.value
        label = {"entailed": "Yes", "underdetermined": "Undetermined"}[variant]
        status_label = {"entailed": "Yes", "contradicted": "No", "underdetermined": "Undetermined"}[status]
        assert status_label == label, (variant, status, label)
        prompt = build_prompt_with_text_pair(
            a, c, text_p.subject, text_p.object, paraphrase_idx=0, query_idx=0
        )
        records.append(
            {
                "schema_version": "exact_pairs_v1",
                "family_id": family_id,
                "sample_id": f"{family_id}::{variant}",
                "condition_id": CONDITION_ID,
                "variant_id": variant,
                "pair_bucket": "exact_image_changed_text",
                "entities": {"A": a, "B": b, "C": c},
                "query": {"subject": a, "object": c, "relation": "left_of"},
                "premises": [
                    {"subject": p.subject, "object": p.object, "modality": p.modality.value}
                    for p in premises
                ],
                "ground_truth_label": label,
                "symbolic_status": status,
                "oracle_agreement": res.algorithm_agreement,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "image_file": meta["image_file"],
                "image_sha256": meta["image_sha256"],
                "visual_entities_shown": sorted(meta["entities_shown"].keys()),
                "third_entity_hidden": c not in meta["entities_shown"],
                "is_synthetic_demo": False,
            }
        )
    # Both records must share identical image hash
    assert records[0]["image_sha256"] == records[1]["image_sha256"]
    return records


def build_dataset(out_dir: Path, n_families: int = 3) -> Dict[str, Any]:
    from src.benchmark.contrast_v2 import plan_families

    out_dir.mkdir(parents=True, exist_ok=True)
    images = out_dir / "images"
    images.mkdir(exist_ok=True)
    plans = [p for p in plan_families(n_families=240, seed=20261031) if p.split in {"train", "validation"}][:n_families]
    all_recs: List[Dict[str, Any]] = []
    for plan in plans:
        all_recs.extend(
            build_exact_image_pair_records(
                plan.family_id, plan.a, plan.b, plan.c, out_images=images, seed=plan.layout_seed
            )
        )
    jsonl = out_dir / "exact_pairs_v1.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        for r in all_recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "schema_version": "exact_pairs_v1",
        "n_families": len(plans),
        "n_records": len(all_recs),
        "pair_bucket": "exact_image_changed_text",
        "labels": dict(Counter_labels(all_recs)),
        "image_hash_shared_per_family": True,
        "jsonl": str(jsonl.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    (out_dir / "exact_pairs_meta_v1.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def Counter_labels(recs):
    from collections import Counter

    return Counter(r["ground_truth_label"] for r in recs)


if __name__ == "__main__":
    out = REPO_ROOT / "datasets/exact_pairs_v1"
    m = build_dataset(out, n_families=3)
    print(json.dumps(m, indent=2))
