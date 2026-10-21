"""Phase 2B recovery helpers: memory telemetry schema and resume validation (CPU)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def validate_intervention_jsonl(path: Path) -> Dict[str, Any]:
    """Parse and validate intervention records without modifying the file."""
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    rows: List[Dict[str, Any]] = []
    bad_lines: List[Dict[str, Any]] = []
    for i, ln in enumerate(lines, 1):
        try:
            rows.append(json.loads(ln))
        except Exception as exc:
            bad_lines.append({"line": i, "error": str(exc), "tail": ln[-80:]})

    ids = [r.get("intervention_id") for r in rows]
    dupes = [k for k, v in Counter(ids).items() if v and v > 1]
    expected_model_rev = "482adb537c021c86670beed01cd58990d01e72e4"
    model_mismatches = [
        r.get("intervention_id")
        for r in rows
        if (r.get("model") or {}).get("revision") not in (None, expected_model_rev)
    ]
    return {
        "path": str(path),
        "byte_length": len(raw),
        "n_lines": len(lines),
        "n_valid": len(rows),
        "n_bad_lines": len(bad_lines),
        "bad_lines": bad_lines,
        "duplicate_ids": dupes,
        "model_revision_mismatches": model_mismatches,
        "last_intervention_id": ids[-1] if ids else None,
        "roles": dict(Counter(r.get("role") for r in rows)),
        "directions": dict(Counter(r.get("direction") for r in rows if r.get("role") == "intervention")),
        "families": dict(Counter(r.get("family_id") for r in rows)),
        "pairs": dict(Counter(r.get("pair_id") for r in rows)),
        "ok": len(bad_lines) == 0 and not dupes and not model_mismatches,
        "rows": rows,
    }


def missing_interventions(
    rows: List[Dict[str, Any]],
    family_ids: List[str],
    pair_types: List[str],
    directions: List[str],
) -> List[str]:
    """Compute missing intervention IDs for a reduced completion matrix."""
    have = {r.get("intervention_id") for r in rows if r.get("status") == "ok"}
    missing: List[str] = []
    for fam in family_ids:
        for pt in pair_types:
            prefix = f"{fam}:{pt}"
            for role in ("baseline_source", "baseline_target"):
                iid = f"{prefix}::{role}"
                if iid not in have:
                    missing.append(iid)
            for d in directions:
                iid = f"{prefix}::{d}"
                if iid not in have:
                    missing.append(iid)
    return missing


def peak_vram_schema_note() -> Dict[str, Any]:
    return {
        "field": "peak_vram_mib",
        "intended_semantics": "torch.cuda.max_memory_allocated(0) since last reset, in MiB",
        "observed_issue": "Phase 2B runner did not call reset_peak_memory_stats before each intervention; values ~5076–5142 MiB are process-lifetime peaks and may include shared/system spill accounting.",
        "nvidia_dedicated_typical": "RTX 2050 dedicated=4096 MiB; values above 4096 indicate allocator accounting including shared GPU memory on Windows WDDM, not fabricated numbers.",
        "recommendation": "Future runs must reset peak stats per intervention and record torch.cuda.memory_allocated, memory_reserved, and nvidia-smi dedicated separately.",
    }


def compare_control_vs_targeted(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_dir: Dict[str, List[float]] = {}
    for r in rows:
        if r.get("role") == "intervention" and isinstance(r.get("delta_m"), dict):
            by_dir.setdefault(r["direction"], []).append(float(r["delta_m"]["delta_m"]))
    means = {d: (sum(v) / len(v) if v else None) for d, v in by_dir.items()}
    abs_means = {d: (sum(abs(x) for x in v) / len(v) if v else None) for d, v in by_dir.items()}
    targeted = abs_means.get("source_to_target")
    controls = [abs_means.get(k) for k in ("random_source", "unrelated_token", "same_label")]
    controls = [c for c in controls if c is not None]
    max_control = max(controls) if controls else None
    return {
        "mean_delta_m_by_direction": means,
        "mean_abs_delta_m_by_direction": abs_means,
        "max_control_abs_mean": max_control,
        "targeted_abs_mean": targeted,
        "targeted_exceeds_controls": (targeted is not None and max_control is not None and targeted > max_control),
        "scientific_flag": (
            "GENERIC_INTERVENTION_SENSITIVITY"
            if targeted is not None and max_control is not None and targeted <= max_control
            else "TARGETED_MAY_EXCEED_CONTROLS_INSUFFICIENT_N"
        ),
        "note": "N is tiny; do not treat as inferential statistics.",
    }
