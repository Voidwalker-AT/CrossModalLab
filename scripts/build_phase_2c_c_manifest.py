#!/usr/bin/env python3
"""Freeze Phase 2C-C paired cross-model comparison manifest (CPU)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.instrumentation.capture_integrity import atomic_write_json as awj
from src.symbolic.relations import DirectionalRelation, Modality, Premise, Query
from src.symbolic.solver import SymbolicSolver

# Development-exposed from prior pilots (not holdout)
DEVELOPMENT_EXPOSED = {
    "xc_ho_0000", "xc_ho_0001", "xc_ho_0003", "xc_ho_0005", "xc_ho_0006",
    "xc_ho_0007", "xc_ho_0008", "xc_ho_0009", "xc_ho_0011", "xc_ho_0013",
    "xc_ho_0014", "xc_ho_0015", "xc_ho_0016", "xc_ho_0024", "xc_ho_0028",
    "xc_ho_0029", "xc_ho_0032", "xc_ho_0033", "xc_ho_0040",
    "xc_id_0189", "xc_id_0190", "xc_id_0191",
}
# Contrast-v2 test split = final holdout for this comparison (do not touch)
HOLDOUT = set()  # filled from dataset in main()

# Prefer train-split families not in holdout; mark exposed
CANDIDATE_FAMILIES = [
    "xc_ho_0017",
    "xc_ho_0019",
    "xc_ho_0020",
    "xc_ho_0021",
    "xc_ho_0022",
    "xc_ho_0023",
]


def verify_modality_dependency(rec) -> dict:
    premises = [
        Premise(
            p["subject"],
            p["object"],
            DirectionalRelation.LEFT_OF,
            Modality.VISUAL if p["modality"] == "visual" else Modality.TEXTUAL,
        )
        for p in rec["premises"]
    ]
    q = Query(rec["entities"]["A"], rec["entities"]["C"], DirectionalRelation.LEFT_OF)
    status = SymbolicSolver.solve(premises, q, verify_with_oracle=True)
    vis = [p for p in premises if p.modality == Modality.VISUAL]
    txt = [p for p in premises if p.modality == Modality.TEXTUAL]
    vis_only = SymbolicSolver.solve(vis, q, verify_with_oracle=True).epistemic_status.value if vis else "underdetermined"
    txt_only = SymbolicSolver.solve(txt, q, verify_with_oracle=True).epistemic_status.value if txt else "underdetermined"
    return {
        "solver_status": status.epistemic_status.value,
        "oracle_agreement": status.algorithm_agreement,
        "vis_only_status": vis_only,
        "txt_only_status": txt_only,
        "label_match": {
            "entailed": "Yes",
            "contradicted": "No",
            "underdetermined": "Undetermined",
        }[status.epistemic_status.value]
        == rec["ground_truth_label"],
        "third_entity_hidden": rec.get("third_entity_hidden", True),
        "single_modality_sufficient": (
            status.epistemic_status.value in {"entailed", "contradicted"}
            and (vis_only == status.epistemic_status.value or txt_only == status.epistemic_status.value)
        ),
    }


def main() -> int:
    src = REPO_ROOT / "datasets/cross_modal_contrast_v2/cross_modal_contrast_v2.jsonl"
    records = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_fam = {}
    holdout = set()
    for r in records:
        by_fam.setdefault(r["family_id"], {})[r["variant_id"]] = r
        if r["split"] == "test":
            holdout.add(r["family_id"])

    selected = []
    for fam in CANDIDATE_FAMILIES:
        if fam in holdout:
            raise SystemExit(f"ERROR: {fam} is contrast-v2 test holdout")
        if fam not in by_fam:
            raise SystemExit(f"ERROR: {fam} missing")
        fam_recs = by_fam[fam]
        cases = []
        for var in ("entailed", "contradicted", "underdetermined"):
            rec = fam_recs[var]
            audit = verify_modality_dependency(rec)
            if not audit["label_match"]:
                raise SystemExit(f"label mismatch {fam} {var}")
            if audit["single_modality_sufficient"]:
                raise SystemExit(f"single-modality sufficient {fam} {var}")
            if not audit["third_entity_hidden"]:
                raise SystemExit(f"C leaked {fam} {var}")
            cases.append(
                {
                    "sample_id": rec["sample_id"],
                    "variant_id": var,
                    "ground_truth_label": rec["ground_truth_label"],
                    "symbolic_status": rec["symbolic_status"],
                    "image_file": rec["image_file"],
                    "image_sha256": rec["image_sha256"],
                    "prompt": rec["prompt"],
                    "prompt_sha256": rec["prompt_sha256"],
                    "condition_id": rec["condition_id"],
                    "modality_audit": audit,
                }
            )
        selected.append(
            {
                "family_id": fam,
                "split": fam_recs["entailed"]["split"],
                "development_exposed": fam in DEVELOPMENT_EXPOSED,
                "holdout": False,
                "cases": cases,
            }
        )

    n_cases = sum(len(f["cases"]) for f in selected)
    assert n_cases == 18
    manifest = {
        "schema_version": "phase_2c_c_manifest_v1",
        "n_families": len(selected),
        "n_cases": n_cases,
        "n_evaluations_planned": n_cases * 2,  # two models
        "models": {
            "smolvlm2": {
                "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                "revision": "482adb537c021c86670beed01cd58990d01e72e4",
            },
            "qwen3vl2": {
                "model_id": "Qwen/Qwen3-VL-2B-Instruct",
                "revision": "89644892e4d85e24eaac8bacfd4f463576704203",
            },
        },
        "final_holdout_touched": False,
        "families": selected,
        "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "notes": [
            "Development-only; all six families marked development_exposed where prior probe use occurred.",
            "Identical image bytes and logical prompts for both models.",
            "Model-specific chat templates only for serialization.",
        ],
    }
    out = REPO_ROOT / "configs/phase_2c_c/paired_eval_manifest_v1.json"
    awj(out, manifest)
    print(f"families={len(selected)} cases={n_cases} exposed={sum(f['development_exposed'] for f in selected)}")
    print(f"out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
