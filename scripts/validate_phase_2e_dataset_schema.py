#!/usr/bin/env python3
"""Validate cross_modal_contrast_v2 (and optional custom paths) with Phase 2E schema rules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.instrumentation.capture_integrity import atomic_write_json
from src.training.schema import full_audit, holdout_disjoint


def main() -> int:
    path = REPO_ROOT / "datasets/cross_modal_contrast_v2/cross_modal_contrast_v2.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    audit = full_audit(records)
    # holdout check against contrast-v2 test
    test_f = {r["family_id"] for r in records if r["split"] == "test"}
    dev_f = {r["family_id"] for r in records if r["split"] != "test"}
    audit["holdout_disjoint"] = holdout_disjoint(dev_f, test_f)
    out = REPO_ROOT / "artifacts/datasets/phase_2e_schema_audit_v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, audit)
    print(json.dumps({
        "n": audit["n"],
        "schema_ok": audit["schema_ok"],
        "split_ok": audit["split"]["ok"],
        "cf_ok": audit["counterfactual"]["ok"],
        "balance": audit["balance"]["counts"],
        "holdout_disjoint": audit["holdout_disjoint"],
    }, indent=2))
    return 0 if audit["schema_ok"] and audit["split"]["ok"] and audit["counterfactual"]["ok"] and audit["holdout_disjoint"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
