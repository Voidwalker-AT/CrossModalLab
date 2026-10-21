#!/usr/bin/env python3
"""CPU-only Phase 2B evidence recovery audit (does not modify the JSONL)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.causal.recovery import (
    compare_control_vs_targeted,
    missing_interventions,
    peak_vram_schema_note,
    validate_intervention_jsonl,
)
from src.instrumentation.capture_integrity import atomic_write_json


def main() -> int:
    path = REPO_ROOT / "artifacts/causal/phase_2b/intervention_records_v1.jsonl"
    report = validate_intervention_jsonl(path)
    rows = report.pop("rows")
    families = ["xc_ho_0013", "xc_ho_0016", "xc_ho_0024"]
    pair_types = [
        "entailed_vs_underdetermined",
        "contradicted_vs_underdetermined",
    ]
    directions = [
        "source_to_target",
        "target_to_source",
        "same_label",
        "unrelated_token",
        "random_source",
    ]
    missing = missing_interventions(rows, families, pair_types, directions)
    effects = compare_control_vs_targeted(rows)
    eng = REPO_ROOT / "artifacts/causal/phase_2b/engineering_validation_v1.json"
    eng_data = json.loads(eng.read_text(encoding="utf-8")) if eng.is_file() else None

    recovery = {
        "phase": "2B_recovery",
        "status": "AUDIT_COMPLETE_NO_GPU",
        "process_state": "no_phase_2b_python_process_running",
        "evidence_file": report,
        "missing_intervention_count": len(missing),
        "missing_interventions_sample": missing[:20],
        "effect_audit": effects,
        "engineering_validation": eng_data,
        "memory_telemetry": peak_vram_schema_note(),
        "historical_limitations": [
            "Earlier session used Remove-Item on partial intervention_records_v1.jsonl before a scoring-hook fix; those discarded lines are unrecoverable.",
            "Current file (sha256 in evidence_file) is the surviving post-fix record set and must be preserved byte-for-byte.",
            "Pair 'same image' in contrast_v2 is same visual *structure* with independent canvas jitter; image_sha256 differs across variants.",
        ],
        "scientific_flags": [
            "EXPLORATORY / POST-HOC MODIFIED: scoring hook lifetime fixed after first zero-delta run was discarded.",
            "Negative controls (unrelated_token, random_source) currently show larger |Δm| than targeted patches on completed pairs.",
            "Do not interpret current deltas as confirmatory causal evidence.",
        ],
        "safe_reduced_cost_strategy": [
            "Do not resume full 6-pair × 5-direction matrix on GPU in this recovery task.",
            "If later authorized: only source_to_target + no-op baseline on 3 families × 1 pair type (≤12 forwards), with per-forward peak reset and dual VRAM telemetry.",
            "Require identical image_sha256 for 'same image' pairs or relabel as same_visual_structure.",
        ],
        "recommendation": "REDESIGN_BEFORE_RESUME",
    }
    out = REPO_ROOT / "artifacts/causal/phase_2b/recovery_audit_v1.json"
    atomic_write_json(out, recovery)
    print(json.dumps({
        "valid_records": report["n_valid"],
        "last": report["last_intervention_id"],
        "missing": len(missing),
        "effect_flag": effects["scientific_flag"],
        "out": str(out),
    }, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
