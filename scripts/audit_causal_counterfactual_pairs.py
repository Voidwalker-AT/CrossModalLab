#!/usr/bin/env python3
"""Audit contrast-v2 counterfactual pairs for Phase 2A design (CPU)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json

from src.causal.counterfactual_pairs import audit_development_pairs

# Phase 1F validation families (development-only)
VAL_FAMILIES = [
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


def main() -> int:
    report = audit_development_pairs(VAL_FAMILIES)
    out = REPO_ROOT / "artifacts/causal/phase_2a/counterfactual_pair_audit_v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "n_families_ok": report["n_families_ok"],
        "n_single_variable_pairs": report["n_single_variable_pairs"],
        "out": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
