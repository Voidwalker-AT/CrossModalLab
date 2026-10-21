#!/usr/bin/env python3
"""Phase 1C — Frozen split manifests, contamination audit, and balanced sampling protocol.

Does not alter cross_modal_dependency_v1 ground truth.
Does not run GPU inference or train probes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.instrumentation.capture_integrity import atomic_write_json, compute_probe_metrics

SEED = 20261022
# Families used in Phase 1B pilot (development-exposed)
PILOT_FAMILIES = {
    "xm_ho_0000",
    "xm_ho_0013",
    "xm_ho_0014",
    "xm_ho_0021",
    "xm_ho_0025",
    "xm_ho_0029",
    "xm_ho_0038",
    "xm_id_0003",
    "xm_id_0015",
    "xm_id_0021",
    "xm_id_0024",
    "xm_id_0028",
}

# Map conditions to 3-way probe evaluation strata (inconsistent kept separate)
THREE_WAY_CONDITIONS = {
    "both_premises": "Yes",
    "both_modified": "No",
    "visual_removed": "Undetermined",
    "text_removed": "Undetermined",
    "underdetermined_fork": "Undetermined",
    "visual_reversed": "Undetermined",
    "text_reversed": "Undetermined",
}
SEPARATE_CONDITION = "contradictory_evidence"  # symbolic inconsistent


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def family_map(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    fams: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        fams[r["family_id"]].append(r)
    return fams


def build_frozen_manifests(records: List[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    fams = family_map(records)
    all_fids = sorted(fams.keys())
    exposed = sorted(PILOT_FAMILIES & set(all_fids))
    unexposed = sorted(set(all_fids) - PILOT_FAMILIES)

    # Contamination: original test split family membership
    orig_test = sorted({fid for fid, rows in fams.items() if rows[0]["split"] == "test"})
    contaminated_orig_test = sorted(set(orig_test) & PILOT_FAMILIES)

    # New protocol over unexposed families only:
    # development = former train/validation unexposed
    # final_holdout = unexposed families formerly in test PLUS a carved subset from unexposed train
    # to restore a clean holdout without reusing pilot families.
    unexposed_orig_split = {
        fid: fams[fid][0]["split"] for fid in unexposed
    }
    # Holdout: all unexposed former-test (36-12=24) + 12 families sampled from unexposed train (seeded)
    holdout_from_test = [fid for fid in unexposed if unexposed_orig_split[fid] == "test"]
    train_pool = [fid for fid in unexposed if unexposed_orig_split[fid] == "train"]
    val_pool = [fid for fid in unexposed if unexposed_orig_split[fid] == "validation"]
    rng = random.Random(SEED)
    rng.shuffle(train_pool)
    carved_holdout = train_pool[:12]
    remaining_train = train_pool[12:]

    development_families = sorted(set(remaining_train) | set(val_pool))
    final_holdout_families = sorted(set(holdout_from_test) | set(carved_holdout))
    assert not (set(development_families) & set(final_holdout_families))
    assert not (set(final_holdout_families) & PILOT_FAMILIES)
    assert not (set(development_families) & PILOT_FAMILIES)

    # Balanced sampling protocol (family-level): one Yes, one No, one Undetermined stratum record per family
    # Undetermined condition chosen deterministically: underdetermined_fork
    def balanced_records_for_families(fids: Sequence[str]) -> List[Dict[str, Any]]:
        out = []
        for fid in fids:
            by_cond = {r["condition"]: r for r in fams[fid]}
            out.append(by_cond["both_premises"])
            out.append(by_cond["both_modified"])
            out.append(by_cond["underdetermined_fork"])
        return out

    def all_records_for_families(fids: Sequence[str]) -> List[Dict[str, Any]]:
        out = []
        for fid in fids:
            out.extend(fams[fid])
        return out

    dev_balanced = balanced_records_for_families(development_families)
    hold_balanced = balanced_records_for_families(final_holdout_families)

    # Metrics demo on trivial majority baseline for protocol definition
    y_true = [r["ground_truth_label"] for r in hold_balanced]
    majority = Counter(y_true).most_common(1)[0][0]
    baseline_metrics = compute_probe_metrics(y_true, [majority] * len(y_true), ["Yes", "No", "Undetermined"])

    # Checksums of source v1 jsonl
    src = REPO_ROOT / "datasets/cross_modal_dependency_v1/cross_modal_dependency_v1.jsonl"
    src_sha = hashlib.sha256(src.read_bytes()).hexdigest()

    manifest = {
        "schema_version": "phase_1c_splits_v1",
        "seed": SEED,
        "source_dataset": "datasets/cross_modal_dependency_v1/cross_modal_dependency_v1.jsonl",
        "source_sha256": src_sha,
        "ground_truth_altered": False,
        "n_families_total": len(all_fids),
        "development_exposed_families": exposed,
        "n_development_exposed": len(exposed),
        "original_test_family_count": len(orig_test),
        "original_test_contaminated_count": len(contaminated_orig_test),
        "original_test_contaminated_families": contaminated_orig_test,
        "contamination_verdict": (
            "ENTIRE_ORIGINAL_TEST_SPLIT_CONTAMINATED"
            if set(orig_test) <= PILOT_FAMILIES
            else "PARTIAL_ORIGINAL_TEST_CONTAMINATION"
        ),
        "development_families": development_families,
        "final_holdout_families": final_holdout_families,
        "n_development_families": len(development_families),
        "n_final_holdout_families": len(final_holdout_families),
        "disjoint_checks": {
            "dev_holdout_disjoint": True,
            "pilot_excluded_from_dev_and_holdout": True,
            "family_intact_within_split": True,
        },
        "balanced_sampling_protocol": {
            "description": (
                "Per family, include exactly one record from each 3-way stratum: "
                "Yes=both_premises, No=both_modified, Undetermined=underdetermined_fork. "
                "contradictory_evidence (inconsistent) is excluded from 3-way metrics and tracked separately."
            ),
            "three_way_conditions": THREE_WAY_CONDITIONS,
            "separate_condition": SEPARATE_CONDITION,
            "n_dev_balanced": len(dev_balanced),
            "n_holdout_balanced": len(hold_balanced),
            "holdout_label_counts": dict(Counter(r["ground_truth_label"] for r in hold_balanced)),
            "do_not_oversample_related_records_as_independent": True,
        },
        "probe_metrics_definition": {
            "balanced_accuracy": "mean per-class recall over classes with support>0",
            "macro_f1": "unweighted mean of per-class F1 over classes with support>0",
            "per_class_recall": "TP/(TP+FN) per label",
            "majority_baseline_holdout": baseline_metrics,
        },
        "final_evaluation_rules": [
            "Do not tune prompts, models, or probes using final_holdout labels.",
            "All members of a matched family stay in one partition.",
            "Probe train/eval must be group-disjoint by family_id.",
            "Inconsistent (contradictory_evidence) is not forced into Yes/No/Undetermined metrics.",
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "frozen_split_manifest_v1.json", manifest)

    # Write sample id lists for tooling
    def write_ids(name: str, fids: Sequence[str]) -> None:
        ids = []
        for fid in fids:
            for r in fams[fid]:
                ids.append(r["sample_id"])
        (out_dir / name).write_text("\n".join(ids) + "\n", encoding="utf-8")

    write_ids("development_family_ids.txt", development_families)
    write_ids("final_holdout_family_ids.txt", final_holdout_families)

    # Balanced JSONL extracts (metadata only references; ground truth unchanged)
    with (out_dir / "balanced_development_v1.jsonl").open("w", encoding="utf-8") as fh:
        for r in dev_balanced:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "balanced_final_holdout_v1.jsonl").open("w", encoding="utf-8") as fh:
        for r in hold_balanced:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Checksums of derived manifests
    checksums = {}
    for p in out_dir.glob("*"):
        if p.is_file() and p.name != "checksums_v1.json":
            checksums[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    atomic_write_json(out_dir / "checksums_v1.json", checksums)

    return manifest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "datasets/cross_modal_dependency_v1/cross_modal_dependency_v1.jsonl",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "configs/phase_1c",
    )
    args = p.parse_args()
    records = load_jsonl(args.dataset)
    manifest = build_frozen_manifests(records, args.out_dir)
    print(json.dumps({k: manifest[k] for k in [
        "contamination_verdict",
        "n_development_families",
        "n_final_holdout_families",
        "n_development_exposed",
    ]}, indent=2))
    print("balanced holdout labels", manifest["balanced_sampling_protocol"]["holdout_label_counts"])
    print("Wrote manifests to", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
