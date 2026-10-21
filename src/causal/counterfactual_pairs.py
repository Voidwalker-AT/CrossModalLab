"""Counterfactual pair construction from contrast v2 families (development design)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config.paths import REPO_ROOT


def load_contrast(jsonl_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = jsonl_path or (REPO_ROOT / "datasets/cross_modal_contrast_v2/cross_modal_contrast_v2.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_family_pairs(
    records: Sequence[Dict[str, Any]],
    family_id: str,
) -> Dict[str, Any]:
    """Document clean counterfactual structure within one family."""
    fam = [r for r in records if r["family_id"] == family_id]
    by_var = {r["variant_id"]: r for r in fam}
    needed = {"entailed", "contradicted", "underdetermined"}
    if set(by_var) != needed:
        return {"family_id": family_id, "ok": False, "reason": f"variants={set(by_var)}"}

    e, c, u = by_var["entailed"], by_var["contradicted"], by_var["underdetermined"]
    pairs = []

    def _img_same(a, b):
        return a.get("image_sha256") == b.get("image_sha256")

    def _text_premise(r):
        return [p for p in r["premises"] if p["modality"] == "textual"]

    def _vis_premise(r):
        return [p for p in r["premises"] if p["modality"] == "visual"]

    # Entailed vs Underdetermined: same visual image; text premise changes
    if _img_same(e, u):
        pairs.append(
            {
                "pair_id": f"{family_id}:entailed_vs_underdetermined",
                "pair_type": "same_image_changed_text",
                "clean_sample_id": e["sample_id"],
                "counterfactual_sample_id": u["sample_id"],
                "clean_label": e["ground_truth_label"],
                "counterfactual_label": u["ground_truth_label"],
                "evidence_change": "textual premise direction (B<C) → (C<B)",
                "visual_consistent": True,
                "text_consistent": False,
                "single_variable": True,
            }
        )
    # Contradicted vs Underdetermined: same text premise; visual flips
    if _text_premise(c) == _text_premise(u) and not _img_same(c, u):
        pairs.append(
            {
                "pair_id": f"{family_id}:contradicted_vs_underdetermined",
                "pair_type": "same_text_changed_image",
                "clean_sample_id": c["sample_id"],
                "counterfactual_sample_id": u["sample_id"],
                "clean_label": c["ground_truth_label"],
                "counterfactual_label": u["ground_truth_label"],
                "evidence_change": "visual left/right order of A,B flipped",
                "visual_consistent": False,
                "text_consistent": True,
                "single_variable": True,
            }
        )
    # Entailed vs Contradicted: multi-factor
    pairs.append(
        {
            "pair_id": f"{family_id}:entailed_vs_contradicted",
            "pair_type": "multi_factor",
            "clean_sample_id": e["sample_id"],
            "counterfactual_sample_id": c["sample_id"],
            "clean_label": e["ground_truth_label"],
            "counterfactual_label": c["ground_truth_label"],
            "evidence_change": "both visual order and text premise differ",
            "visual_consistent": False,
            "text_consistent": False,
            "single_variable": False,
            "warning": "NOT a single-variable intervention pair",
        }
    )

    # Token alignment metadata
    def _tok_meta(r):
        return {
            "sample_id": r["sample_id"],
            "prompt_sha256": r["prompt_sha256"],
            "image_sha256": r["image_sha256"],
            "entities": r["entities"],
            "ground_truth_label": r["ground_truth_label"],
            "premises": r["premises"],
        }

    return {
        "family_id": family_id,
        "ok": True,
        "entities": e["entities"],
        "split": e["split"],
        "samples": {v: _tok_meta(by_var[v]) for v in needed},
        "pairs": pairs,
    }


def audit_development_pairs(
    family_ids: Sequence[str],
    records: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    recs = list(records) if records is not None else load_contrast()
    audited = [audit_family_pairs(recs, fid) for fid in family_ids]
    ok = [a for a in audited if a.get("ok")]
    single_var = []
    for a in ok:
        for p in a["pairs"]:
            if p.get("single_variable"):
                single_var.append(p)
    return {
        "n_families_audited": len(family_ids),
        "n_families_ok": len(ok),
        "n_single_variable_pairs": len(single_var),
        "single_variable_pairs": single_var,
        "families": ok,
        "notes": [
            "Only development families; final holdout sealed.",
            "Multi-factor pairs excluded from single-variable causal claims.",
            "Token alignment: same image ⇒ same visual token count; text length may differ — patch last token only after independent forward alignment check at execution time.",
        ],
    }


if __name__ == "__main__":
    # Default: Phase 1F validation families
    val = [
        "xc_ho_0013",
        "xc_ho_0016",
        "xc_ho_0024",
        "xc_ho_0028",
        "xc_ho_0029",
        "xc_ho_0032",
        "xc_ho_0033",
        "xc_ho_0040",
        "xc_id_0191",
    ]
    report = audit_development_pairs(val)
    out = REPO_ROOT / "artifacts/causal/phase_2a/counterfactual_pair_audit_v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"families_ok={report['n_families_ok']} single_var_pairs={report['n_single_variable_pairs']}")
