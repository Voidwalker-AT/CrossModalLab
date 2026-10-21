#!/usr/bin/env python3
"""Phase 0E scientific integrity audit of Phase 0D.2 artifacts (CPU-only)."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.symbolic.relations import DirectionalRelation, Modality, Premise, Query
from src.symbolic.solver import SymbolicSolver


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    man = load_json(REPO_ROOT / "configs/phase_0d2/behavioral_manifest_v1.json")
    ctrl_path = REPO_ROOT / "datasets/controlled/controlled_benchmark_180.jsonl"
    ctrl = {row["sample_id"]: row for row in load_jsonl(ctrl_path)}
    preds = load_jsonl(REPO_ROOT / "artifacts/evaluations/phase_0d2/behavioral_predictions_v1.jsonl")
    ok = {r["sample_id"]: r for r in preds if r.get("prediction_status") == "ok"}
    print(f"N_OK={len(ok)}")

    # Confusion from raw
    y_true = [r["ground_truth_label"] for r in ok.values()]
    y_pred = [r["parsed_answer"] for r in ok.values()]
    labels = ["Yes", "No", "Undetermined"]
    print("CONFUSION")
    for t in labels:
        row = {p: sum(1 for a, b in zip(y_true, y_pred) if a == t and b == p) for p in labels + ["Invalid"]}
        print(f"  {t}: {row}")

    # Independent reversal GT verification
    print("REVERSAL_GT_AUDIT")
    rev_ok = 0
    rev_total = 0
    for s in man["samples"]:
        if s["condition"] != "directional_reversal_control":
            continue
        rev_total += 1
        parent_src = s["sample_id"].replace("p0d2_revctl_", "")
        parent = next(
            x for x in man["samples"] if x["condition"] == "forward_chain" and x["source_sample_id"] == parent_src
        )
        _ = parent
        base = ctrl[parent_src]
        sgt = base["symbolic_ground_truth"]
        premises = []
        all_ents = set()
        for p in sgt["premises"]:
            modality = Modality.VISUAL if p["modality"] == "visual" else Modality.TEXTUAL
            premises.append(Premise(p["subject"], p["object"], DirectionalRelation.LEFT_OF, modality))
            all_ents.add(p["subject"])
            all_ents.add(p["object"])
        all_ents.add(sgt["query"]["subject"])
        all_ents.add(sgt["query"]["object"])
        qtext = s["query"]
        subject_phrase = qtext.split("to the left of")[0].strip()
        object_phrase = qtext.split("to the left of")[1].strip().rstrip("?")

        def match_entity(phrase: str) -> str:
            for e in all_ents:
                if e.replace("_", " ") == phrase:
                    return e
            for e in all_ents:
                parts = e.split("_")
                if all(p in phrase for p in parts):
                    return e
            raise KeyError(f"Cannot map phrase {phrase!r} to entities {sorted(all_ents)}")

        subj = match_entity(subject_phrase)
        obj = match_entity(object_phrase)
        res = SymbolicSolver.solve(premises, Query(subj, obj, DirectionalRelation.LEFT_OF))
        status = res.epistemic_status.value
        expected = {"entailed": "Yes", "contradicted": "No", "underdetermined": "Undetermined"}.get(status, status)
        match = expected == s["ground_truth_label"]
        if match:
            rev_ok += 1
        print(f"  {s['sample_id']}: {subj}-left_of-{obj} solver={status} declared={s['ground_truth_label']} match={match}")
    print(f"REVERSAL_MATCH {rev_ok}/{rev_total}")

    # Forward/reverse/fork declared vs solver recompute on source samples
    print("SOURCE_GT_REAUDIT")
    src_mismatch = 0
    for s in man["samples"]:
        if s["condition"] in {"directional_reversal_control", "text_premise_control"}:
            continue
        base = ctrl[s["source_sample_id"]]
        sgt = base["symbolic_ground_truth"]
        if not sgt.get("premises"):
            print(f"  SKIP no premises {s['sample_id']}")
            continue
        premises = []
        for p in sgt["premises"]:
            modality = Modality.VISUAL if p["modality"] == "visual" else Modality.TEXTUAL
            premises.append(Premise(p["subject"], p["object"], DirectionalRelation.LEFT_OF, modality))
        q = sgt["query"]
        res = SymbolicSolver.solve(premises, Query(q["subject"], q["object"], DirectionalRelation.LEFT_OF))
        status = res.epistemic_status.value
        if status != sgt["epistemic_status"] or status != s["symbolic_epistemic_status"]:
            src_mismatch += 1
            print(f"  MISMATCH {s['sample_id']} solver={status} declared={s['symbolic_epistemic_status']}")
    print(f"SOURCE_GT_MISMATCHES={src_mismatch}")

    # Text-only controls: blank images?
    print("TEXT_ONLY_AUDIT")
    for s in man["samples"]:
        if s["condition"] != "text_premise_control":
            continue
        base = ctrl[s["source_sample_id"]]
        # blank canvas images share identical sha if truly blank
        print(
            f"  {s['sample_id']} topology={base['topology']} entities={s.get('entity_roles')} "
            f"img={s['image_sha256'][:12]} prompt_modality_word={'image' in s['prompt'].lower()} "
            f"raw={ok[s['sample_id']]['raw_completion'][:40]!r}"
        )
    blank_shas = Counter(
        s["image_sha256"] for s in man["samples"] if s["condition"] == "text_premise_control"
    )
    print(f"TEXT_ONLY_IMAGE_SHA_COUNTS={dict(blank_shas)}")
    print(
        "TEXT_ONLY_NOTE=Blank-canvas image still enters vision tower; not a pure no-image text path."
    )

    # Prompt leakage: GT label tokens outside instructional answer list
    print("PROMPT_LEAKAGE")
    leaks = []
    for s in man["samples"]:
        # Instruction always lists Yes/No/Undetermined — expected task vocabulary.
        # Check premise/query for label words (unexpected leakage).
        for field in ("text_premise", "query"):
            val = s[field]
            for lab in labels:
                if lab.lower() in val.lower():
                    leaks.append((s["sample_id"], field, lab, val))
    print(f"LABEL_IN_PREMISE_OR_QUERY={len(leaks)}")
    for item in leaks[:10]:
        print(" ", item)

    # Matched group structure
    groups = defaultdict(list)
    for s in man["samples"]:
        groups[s.get("matched_group_id")].append(
            {"sample_id": s["sample_id"], "condition": s["condition"], "image_sha256": s["image_sha256"]}
        )
    print("MATCHED_GROUP_SIZES", Counter(len(v) for v in groups.values()))
    # True matched pairs: forward_chain + directional_reversal_control share image_sha256
    pair_image_match = 0
    pair_image_mismatch = 0
    for gid, members in groups.items():
        conds = {m["condition"] for m in members}
        if conds == {"forward_chain", "directional_reversal_control"}:
            shas = {m["image_sha256"] for m in members}
            if len(shas) == 1:
                pair_image_match += 1
            else:
                pair_image_mismatch += 1
                print("PAIR_IMAGE_MISMATCH", gid, shas)
    print(f"TRUE_MATCHED_IMAGE_PAIRS image_shared={pair_image_match} mismatch={pair_image_mismatch}")

    # Behavior summary for mechanistic case selection
    print("CASE_SELECTION_CANDIDATES")
    correct_fwd = [
        r["sample_id"]
        for r in ok.values()
        if r["condition"] == "forward_chain" and r.get("is_correct")
    ]
    incorrect_fwd = [
        r["sample_id"]
        for r in ok.values()
        if r["condition"] == "forward_chain" and not r.get("is_correct")
    ]
    incorrect_fork = [
        r["sample_id"]
        for r in ok.values()
        if r["condition"] == "underdetermined_fork" and not r.get("is_correct")
    ]
    incorrect_revctl = [
        r["sample_id"]
        for r in ok.values()
        if r["condition"] == "directional_reversal_control" and not r.get("is_correct")
    ]
    correct_vis = [
        r["sample_id"] for r in ok.values() if r["condition"] == "visual_premise_control" and r.get("is_correct")
    ]
    print("correct_fwd", correct_fwd)
    print("incorrect_fwd", incorrect_fwd)
    print("incorrect_fork", incorrect_fork)
    print("incorrect_revctl", incorrect_revctl)
    print("correct_vis_controls", correct_vis)

    # Matched pair outcomes for composition failures
    print("MATCHED_PAIR_OUTCOMES")
    for gid, members in groups.items():
        if len(members) != 2:
            continue
        rows = [ok.get(m["sample_id"]) for m in members]
        if any(r is None for r in rows):
            continue
        print(
            gid,
            [(r["sample_id"], r["ground_truth_label"], r["parsed_answer"], r["is_correct"]) for r in rows],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
