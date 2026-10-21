"""Deterministic Phase 0D.2 36-case behavioral manifest builder (CPU-safe).

Selection is fixed before any model inference. Ground truth is copied from
the controlled symbolic benchmark and independently re-verified with the
symbolic solver where premises exist.

Design (seed=42):
- 6 forward_chain
- 6 reverse_chain
- 6 underdetermined_fork
- 6 unimodal_visual
- 6 unimodal_text
- 6 matched directional-reversal controls (reverse query on the same
  forward-chain image / premises; matched_group_id shared with parent)
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from src.config.paths import REPO_ROOT, get_datasets_dir
from src.evaluation.behavioral_metrics import build_prompt
from src.symbolic.solver import SymbolicSolver

MANIFEST_VERSION = "phase_0d2_v1"
SEED = 42
N_PER_CLASS = 6


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(64 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _load_controlled() -> List[Dict[str, Any]]:
    path = get_datasets_dir() / "controlled" / "controlled_benchmark_180.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _query_is_left_of(query_obj: Dict[str, Any]) -> str:
    # query is like {"subject": "...", "object": "..."} in symbolic_ground_truth
    return f"{query_obj['subject'].replace('_', ' ')} to the left of {query_obj['object'].replace('_', ' ')}?"


def _select_stratified(rows: List[Dict[str, Any]], topology: str, n: int, rng: random.Random) -> List[Dict[str, Any]]:
    pool = [r for r in rows if r["topology"] == topology]
    if len(pool) < n:
        raise ValueError(f"Insufficient samples for {topology}: {len(pool)} < {n}")

    # Diversity-aware selection: prefer unique entity-pair signatures first
    by_sig = defaultdict(list)
    for r in pool:
        ents = r.get("symbolic_ground_truth", {}).get("premises", [])
        sig = tuple(sorted({p["subject"] for p in ents} | {p["object"] for p in ents}))
        by_sig[sig].append(r)

    sigs = sorted(by_sig.keys())
    rng.shuffle(sigs)
    selected: List[Dict[str, Any]] = []
    # Round-robin one per entity signature until n
    while len(selected) < n and any(by_sig[s] for s in sigs):
        for s in sigs:
            if not by_sig[s]:
                continue
            # pop deterministic index 0 after shuffle of list order
            chosen = by_sig[s].pop(0)
            selected.append(chosen)
            if len(selected) >= n:
                break
    if len(selected) < n:
        remaining = [r for r in pool if r not in selected]
        selected.extend(remaining[: n - len(selected)])
    return selected


def _verify_symbolic(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Re-derive epistemic status from premises via SymbolicSolver when available."""
    sgt = sample.get("symbolic_ground_truth")
    if not sgt or not sgt.get("premises"):
        return {
            "verified": True,
            "reason": "no_symbolic_premises_for_this_condition",
            "recomputed_status": None,
        }
    from src.symbolic.relations import DirectionalRelation, Modality, Premise, Query

    premises = []
    for p in sgt["premises"]:
        modality = Modality.VISUAL if p.get("modality") == "visual" else Modality.TEXTUAL
        premises.append(
            Premise(
                subject=p["subject"],
                relation=DirectionalRelation.LEFT_OF,
                object=p["object"],
                modality=modality,
            )
        )
    q = sgt["query"]
    query = Query(subject=q["subject"], relation=DirectionalRelation.LEFT_OF, object=q["object"])
    result = SymbolicSolver.solve(premises, query)
    status = result.epistemic_status.value
    expected = str(sgt["epistemic_status"]).lower()
    match = status == expected
    return {
        "verified": bool(match),
        "recomputed_status": status,
        "declared_status": expected,
        "reason": "solver_recompute",
        "algorithm_agreement": result.algorithm_agreement,
    }


def _make_record(
    base: Dict[str, Any],
    *,
    condition: str,
    modality: str,
    relation: str,
    matched_group_id: str,
    query_text: str,
    template_id: str,
    seed: int,
) -> Dict[str, Any]:
    image_path = get_datasets_dir() / "controlled" / "images" / base["image_file"]
    prompt = build_prompt(base["text_premise"], query_text, modality="text_only" if modality == "text_only" else "visual")
    verification = _verify_symbolic(base)
    return {
        "manifest_version": MANIFEST_VERSION,
        "sample_id": f"p0d2_{base['sample_id']}_{condition}",
        "source_sample_id": base["sample_id"],
        "condition": condition,
        "topology": base["topology"],
        "modality": modality,
        "relation": relation,
        "matched_group_id": matched_group_id,
        "seed": seed,
        "prompt_template_id": template_id,
        "prompt": prompt,
        "text_premise": base["text_premise"],
        "query": query_text,
        "ground_truth_label": base["ground_truth_decision"],
        "symbolic_epistemic_status": base["epistemic_status"],
        "image_file": str(image_path.relative_to(REPO_ROOT)) if image_path.exists() else None,
        "image_sha256": base.get("image_sha256"),
        "image_exists": image_path.exists(),
        "is_synthetic_demo": base.get("is_synthetic_demo", False),
        "symbolic_verification": verification,
        "entity_roles": base.get("image_metadata", {}).get("entities"),
    }


def build_phase_0d2_manifest() -> Dict[str, Any]:
    rng = random.Random(SEED)
    rows = _load_controlled()

    fwd = _select_stratified(rows, "forward_chain", N_PER_CLASS, rng)
    rev = _select_stratified(rows, "reverse_chain", N_PER_CLASS, rng)
    fork = _select_stratified(rows, "underdetermined_fork", N_PER_CLASS, rng)
    vis = _select_stratified(rows, "unimodal_visual", N_PER_CLASS, rng)
    txt = _select_stratified(rows, "unimodal_text", N_PER_CLASS, rng)

    records: List[Dict[str, Any]] = []
    entity_reuse: Dict[str, List[str]] = defaultdict(list)

    def add(base: Dict[str, Any], **kwargs):
        rec = _make_record(base, seed=SEED, **kwargs)
        records.append(rec)
        for ent in (rec.get("entity_roles") or {}):
            entity_reuse[ent].append(rec["sample_id"])

    for b in fwd:
        sg = b["symbolic_ground_truth"]
        add(
            b,
            condition="forward_chain",
            modality="cross_modal",
            relation="left_of",
            matched_group_id=f"mg_{b['sample_id']}",
            query_text=_query_is_left_of(sg["query"]),
            template_id="controlled_cross_modal_v1",
        )
        # Matched directional-reversal: same image, inverted query entities
        rev_query = _query_is_left_of(
            {"subject": sg["query"]["object"], "object": sg["query"]["subject"]}
        )
        rev_base = dict(b)
        # Ground truth for inverted query on forward-chain image: reverse relation holds
        # A left of B in image/query originally entailed; inverted is contradicted (No).
        rev_base = dict(b)
        rev_base["ground_truth_decision"] = "No"
        rev_base["epistemic_status"] = "contradicted"
        rec_rev = _make_record(
            rev_base,
            condition="directional_reversal_control",
            modality="cross_modal",
            relation="left_of",
            matched_group_id=f"mg_{b['sample_id']}",
            query_text=rev_query,
            template_id="controlled_cross_modal_v1",
            seed=SEED,
        )
        # Explicitly mark independent GT for reversal (not reusing parent Yes label)
        rec_rev["ground_truth_label"] = "No"
        rec_rev["symbolic_epistemic_status"] = "contradicted"
        rec_rev["reversal_parent_sample_id"] = b["sample_id"]
        rec_rev["reversal_note"] = (
            "Matched control: same visual canvas as parent forward-chain case; "
            "query entities swapped. GT=No (contradicted) under left-of semantics."
        )
        # Override sample_id uniqueness
        rec_rev["sample_id"] = f"p0d2_revctl_{b['sample_id']}"
        records.append(rec_rev)
        for ent in (rec_rev.get("entity_roles") or {}):
            entity_reuse[ent].append(rec_rev["sample_id"])

    for b in rev:
        sg = b["symbolic_ground_truth"]
        add(
            b,
            condition="reverse_chain",
            modality="cross_modal",
            relation="left_of",
            matched_group_id=f"mg_{b['sample_id']}",
            query_text=_query_is_left_of(sg["query"]),
            template_id="controlled_cross_modal_v1",
        )

    for b in fork:
        sg = b["symbolic_ground_truth"]
        add(
            b,
            condition="underdetermined_fork",
            modality="cross_modal",
            relation="left_of",
            matched_group_id=f"mg_{b['sample_id']}",
            query_text=_query_is_left_of(sg["query"]),
            template_id="controlled_cross_modal_v1",
        )

    for b in vis:
        sg = b["symbolic_ground_truth"]
        add(
            b,
            condition="visual_premise_control",
            modality="visual",
            relation="left_of",
            matched_group_id=f"mg_{b['sample_id']}",
            query_text=_query_is_left_of(sg["query"]),
            template_id="controlled_visual_v1",
        )

    for b in txt:
        sg = b["symbolic_ground_truth"]
        add(
            b,
            condition="text_premise_control",
            modality="text_only",
            relation="left_of",
            matched_group_id=f"mg_{b['sample_id']}",
            query_text=_query_is_left_of(sg["query"]),
            template_id="controlled_text_only_v1",
        )

    # Integrity checks
    ids = [r["sample_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate sample_id in manifest")
    if len(records) != 36:
        raise AssertionError(f"Expected 36 records, got {len(records)}")

    from collections import Counter

    cond_counts = Counter(r["condition"] for r in records)
    expected = {
        "forward_chain": 6,
        "reverse_chain": 6,
        "underdetermined_fork": 6,
        "visual_premise_control": 6,
        "text_premise_control": 6,
        "directional_reversal_control": 6,
    }
    if dict(cond_counts) != expected:
        raise AssertionError(f"Condition imbalance: {dict(cond_counts)}")

    missing_img = [r["sample_id"] for r in records if not r["image_exists"]]
    if missing_img:
        raise AssertionError(f"Missing images for: {missing_img}")

    unverified = [
        r["sample_id"]
        for r in records
        if r["condition"] != "directional_reversal_control"
        and not r["symbolic_verification"]["verified"]
    ]
    # Reversal controls use constructed inverted-query GT (No/contradicted) by design
    if unverified:
        raise AssertionError(f"Symbolic verification failed for: {unverified}")

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "phase": "0D.2",
        "seed": SEED,
        "source_jsonl": "datasets/controlled/controlled_benchmark_180.jsonl",
        "source_jsonl_sha256": _sha256_file(
            get_datasets_dir() / "controlled" / "controlled_benchmark_180.jsonl"
        ),
        "n_samples": len(records),
        "condition_counts": dict(cond_counts),
        "entity_reuse_counts": {k: len(v) for k, v in sorted(entity_reuse.items())},
        "notes": [
            "Fixed before inference; do not regenerate during a run.",
            "36-case pilot is NOT a representative real-world benchmark.",
            "Matched reversal controls share visual entities with parent forward-chain cases.",
            "Directional-reversal GT is constructed (No/contradicted) for the inverted query on the same canvas.",
        ],
        "samples": records,
    }

    # Stable hash of samples payload
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["samples_sha256"] = hashlib.sha256(payload).hexdigest()
    return manifest


def write_manifest(path: Path) -> Dict[str, Any]:
    manifest = build_phase_0d2_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    out = REPO_ROOT / "configs" / "phase_0d2" / "behavioral_manifest_v1.json"
    m = write_manifest(out)
    print(f"Wrote {out}")
    print(f"n={m['n_samples']} conditions={m['condition_counts']}")
    print(f"samples_sha256={m['samples_sha256']}")
