"""Dataset schema, split integrity, and balance utilities (CPU)."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REQUIRED_FIELDS = (
    "sample_id",
    "family_id",
    "split",
    "ground_truth_label",
    "prompt",
    "image_sha256",
    "symbolic_status",
)


def validate_record_schema(rec: Dict[str, Any]) -> List[str]:
    issues = []
    for f in REQUIRED_FIELDS:
        if f not in rec:
            issues.append(f"missing field {f}")
    if rec.get("ground_truth_label") not in {"Yes", "No", "Undetermined"}:
        if "ground_truth_label" in rec:
            issues.append(f"bad label {rec.get('ground_truth_label')}")
    return issues


def check_split_leakage(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """All members of a family must share one split."""
    fam_splits: Dict[str, set] = defaultdict(set)
    for r in records:
        fam_splits[r["family_id"]].add(r["split"])
    leaks = {f: sorted(s) for f, s in fam_splits.items() if len(s) > 1}
    return {"n_families": len(fam_splits), "leaking_families": leaks, "ok": not leaks}


def check_label_balance(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    c = Counter(r["ground_truth_label"] for r in records)
    n = sum(c.values()) or 1
    return {
        "counts": dict(c),
        "fractions": {k: v / n for k, v in c.items()},
        "balanced": len(set(c.values())) == 1,
    }


def check_counterfactual_groups(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Each family should contain one record per label if designed that way."""
    by_fam: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        by_fam[r["family_id"]].append(r["ground_truth_label"])
    incomplete = {
        f: labs
        for f, labs in by_fam.items()
        if sorted(labs) != ["No", "Undetermined", "Yes"]
    }
    return {
        "n_families": len(by_fam),
        "incomplete_families": incomplete,
        "ok": not incomplete,
    }


def holdout_disjoint(development_families: Iterable[str], holdout_families: Iterable[str]) -> bool:
    return not (set(development_families) & set(holdout_families))


def full_audit(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    schema_issues = []
    for r in records:
        iss = validate_record_schema(r)
        if iss:
            schema_issues.append({"sample_id": r.get("sample_id"), "issues": iss})
    return {
        "n": len(records),
        "schema_ok": not schema_issues,
        "schema_issues": schema_issues[:20],
        "split": check_split_leakage(records),
        "balance": check_label_balance(records),
        "counterfactual": check_counterfactual_groups(records),
    }
