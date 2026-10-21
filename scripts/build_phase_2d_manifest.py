#!/usr/bin/env python3
"""Freeze Phase 2D format-sensitivity + evidence-sufficiency manifest (CPU)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.instrumentation.capture_integrity import atomic_write_json
from src.symbolic.relations import DirectionalRelation, Modality, Premise, Query
from src.symbolic.solver import SymbolicSolver

# Same 6 development families as Phase 2C-C (train split, not contrast-v2 test)
FAMILIES = [
    "xc_ho_0017",
    "xc_ho_0019",
    "xc_ho_0020",
    "xc_ho_0021",
    "xc_ho_0022",
    "xc_ho_0023",
]

# Answer format templates (semantic equivalents)
FORMATS = {
    "fmt_yn_undetermined": {
        # baseline already measured in 2C-C; still included for manifest completeness
        "choices": ["Yes", "No", "Undetermined"],
        "instruction": "Answer directly with: Yes, No, or Undetermined.",
        "abstain_labels": ["Undetermined"],
        "run_in_pilot": False,
        "reason": "baseline already in Phase 2C-C",
    },
    "fmt_entailed_contradicted_insufficient": {
        "choices": ["Entailed", "Contradicted", "Insufficient Evidence"],
        "instruction": "Answer directly with one of: Entailed, Contradicted, or Insufficient Evidence.",
        "abstain_labels": ["Insufficient Evidence"],
        "run_in_pilot": True,
    },
    "fmt_true_false_cannot": {
        "choices": ["True", "False", "Cannot Be Determined"],
        "instruction": "Answer directly with one of: True, False, or Cannot Be Determined.",
        "abstain_labels": ["Cannot Be Determined"],
        "run_in_pilot": True,
    },
    "fmt_neutral_abstain_permitted": {
        "choices": ["Yes", "No", "Undetermined"],
        "instruction": (
            "You may answer Undetermined when the evidence does not settle the question. "
            "Answer directly with: Yes, No, or Undetermined."
        ),
        "abstain_labels": ["Undetermined"],
        "run_in_pilot": True,
    },
}


def map_label_to_format(gt: str, fmt: dict) -> str:
    if gt == "Yes":
        return fmt["choices"][0] if fmt["choices"][0] in {"Yes", "True", "Entailed"} else fmt["choices"][0]
    if gt == "No":
        # second choice
        return fmt["choices"][1]
    if gt == "Undetermined":
        return fmt["choices"][2]
    raise ValueError(gt)


def parse_format_answer(text: str, fmt: dict) -> str:
    t = text.strip().lower()
    # try longest abstain labels first
    for lab in sorted(fmt["abstain_labels"], key=len, reverse=True):
        if lab.lower() in t:
            return "Undetermined"
    first = t.split()[0].strip(".,;:!?") if t.split() else ""
    choices_l = [c.lower() for c in fmt["choices"]]
    # map first token
    if first in ("yes", "true", "entailed") or t.startswith("yes") or t.startswith("true"):
        return "Yes"
    if first in ("no", "false", "contradicted") or t.startswith("no") or t.startswith("false"):
        return "No"
    if "insufficient" in t or "cannot" in t or "undetermined" in t or "undecidable" in t:
        return "Undetermined"
    for i, cl in enumerate(choices_l):
        if cl in t:
            return ["Yes", "No", "Undetermined"][i]
    return "Invalid"


def apply_format_to_prompt(base_prompt: str, fmt: dict) -> str:
    """Replace the fixed instruction block with a format-specific one."""
    marker = "Answer directly with:"
    if marker in base_prompt:
        head = base_prompt.split(marker)[0].rstrip()
        # also strip prior carefull-evaluate line if present? keep evaluation sentence
        return head + "\n" + fmt["instruction"]
    return base_prompt + "\n" + fmt["instruction"]


def verify_solver(rec) -> bool:
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
    status = SymbolicSolver.solve(premises, q).epistemic_status.value
    expected = {"entailed": "Yes", "contradicted": "No", "underdetermined": "Undetermined"}[status]
    return expected == rec["ground_truth_label"]


def main() -> int:
    src = REPO_ROOT / "datasets/cross_modal_contrast_v2/cross_modal_contrast_v2.jsonl"
    records = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    by = {}
    for r in records:
        by[(r["family_id"], r["variant_id"])] = r

    cases = []
    # Format variants on underdetermined forks
    for fam in FAMILIES:
        base = by[(fam, "underdetermined")]
        if not verify_solver(base):
            raise SystemExit(f"solver fail {fam} underdetermined")
        for fmt_id, fmt in FORMATS.items():
            if not fmt.get("run_in_pilot", True):
                continue
            prompt = apply_format_to_prompt(base["prompt"], fmt)
            cases.append(
                {
                    "case_id": f"{fam}::underdetermined::{fmt_id}",
                    "family_id": fam,
                    "kind": "format_sensitivity",
                    "format_id": fmt_id,
                    "source_sample_id": base["sample_id"],
                    "logical_label": "Undetermined",
                    "mapped_answer_in_format": map_label_to_format("Undetermined", fmt),
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "image_file": base["image_file"],
                    "image_sha256": base["image_sha256"],
                    "choices": fmt["choices"],
                    "parse_abstain_labels": fmt["abstain_labels"],
                }
            )

    # Text-only insufficient evidence: only ONE textual premise; solver => underdetermined
    for fam in FAMILIES[:6]:
        base = by[(fam, "underdetermined")]
        # build text with only textual premise, prompt says statements only, no image
        text_premises = [p for p in base["premises"] if p["modality"] == "textual"]
        if not text_premises:
            raise SystemExit("missing text premise")
        tp = text_premises[0]
        premise_line = f"The {tp['subject'].replace('_',' ')} is to the left of the {tp['object'].replace('_', ' ')}."
        query = f"is {base['entities']['A'].replace('_',' ')} to the left of {base['entities']['C'].replace('_',' ')}?"
        prompt = (
            f"{premise_line}\n\n"
            f"Question: Based on the statement (no image is provided), {query[0].upper()+query[1:]}\n"
            "Carefully evaluate whether the question is entailed (Yes), contradicted (No), "
            "or underdetermined from the evidence (Undetermined).\n"
            "Answer directly with: Yes, No, or Undetermined."
        )
        # verify with only text premises
        premises = [
            Premise(p["subject"], p["object"], DirectionalRelation.LEFT_OF, Modality.TEXTUAL)
            for p in text_premises
        ]
        q = Query(base["entities"]["A"], base["entities"]["C"], DirectionalRelation.LEFT_OF)
        status = SymbolicSolver.solve(premises, q, verify_with_oracle=True)
        if status.epistemic_status.value != "underdetermined":
            raise SystemExit(f"text-only expected underdetermined got {status.epistemic_status.value}")
        cases.append(
            {
                "case_id": f"{fam}::text_only_insufficient",
                "family_id": fam,
                "kind": "text_only_insufficient",
                "format_id": "fmt_neutral_abstain_permitted",
                "source_sample_id": base["sample_id"],
                "logical_label": "Undetermined",
                "mapped_answer_in_format": "Undetermined",
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "image_file": None,
                "image_sha256": None,
                "choices": ["Yes", "No", "Undetermined"],
                "parse_abstain_labels": ["Undetermined"],
                "modality": "true_text_only",
            }
        )

    # Balance budget note: format cases = 6*3=18, text_only=6 => 24 per model
    n = len(cases)
    if n > 24:
        raise SystemExit(f"too many cases {n}")

    manifest = {
        "schema_version": "phase_2d_manifest_v1",
        "n_cases_per_model": n,
        "max_new_evals_per_model": 24,
        "families": FAMILIES,
        "final_holdout_touched": False,
        "models": {
            "smolvlm2": "482adb537c021c86670beed01cd58990d01e72e4",
            "qwen3vl2": "89644892e4d85e24eaac8bacfd4f463576704203",
        },
        "formats": FORMATS,
        "cases": cases,
        "notes": [
            "Answers mapped back to Yes/No/Undetermined for metric comparison.",
            "Baseline Yes/No/Undetermined not re-run (see 2C-C).",
            "Text-only cases use neutral abstain-permitted instruction.",
        ],
    }
    out = REPO_ROOT / "configs/phase_2d/evidence_format_manifest_v1.json"
    atomic_write_json(out, manifest)
    kinds = Counter_kinds(cases)
    print(f"n_cases={n} kinds={kinds} out={out}")
    return 0


def Counter_kinds(cases):
    from collections import Counter

    return dict(Counter(c["kind"] for c in cases))


if __name__ == "__main__":
    raise SystemExit(main())
