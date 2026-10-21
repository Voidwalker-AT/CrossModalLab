#!/usr/bin/env python3
"""Inventory SmolVLM-500M cache for Phase 2C-A cleanup eligibility (read-only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.instrumentation.capture_integrity import atomic_write_json


def dir_stats(p: Path):
    total = 0
    n = 0
    newest = None
    oldest = None
    if not p.exists():
        return 0, 0, None, None
    for f in p.rglob("*"):
        if not f.is_file():
            continue
        try:
            st = f.stat()
        except OSError:
            continue
        total += st.st_size
        n += 1
        m = st.st_mtime
        if newest is None or m > newest:
            newest = m
        if oldest is None or m < oldest:
            oldest = m
    return total, n, newest, oldest


def main() -> int:
    import time

    root = Path("E:/cache/huggingface")
    m500_hub = root / "hub" / "models--HuggingFaceTB--SmolVLM-500M-Instruct"
    m500_flat = root / "models--HuggingFaceTB--SmolVLM-500M-Instruct"
    m220 = root / "models--HuggingFaceTB--SmolVLM2-2.2B-Instruct"
    home_hub = Path.home() / ".cache" / "huggingface" / "hub"
    home_500 = home_hub / "models--HuggingFaceTB--SmolVLM-500M-Instruct"

    candidates = []
    for label, path in [
        ("e_cache_hub_500m", m500_hub),
        ("e_cache_flat_500m", m500_flat),
        ("user_cache_500m", home_500),
    ]:
        if not path.exists():
            continue
        total, nfiles, newest, oldest = dir_stats(path)
        refs = {}
        ref_dir = path / "refs"
        if ref_dir.exists():
            for r in ref_dir.glob("*"):
                refs[r.name] = r.read_text(encoding="utf-8", errors="ignore").strip()
        snapshots = []
        snap_dir = path / "snapshots"
        if snap_dir.exists():
            snapshots = [s.name for s in snap_dir.iterdir() if s.is_dir()]
        # blob files (hash-named under blobs/)
        blobs = []
        blob_dir = path / "blobs"
        if blob_dir.exists():
            for b in blob_dir.glob("*"):
                if b.is_file():
                    blobs.append({"name": b.name, "size": b.stat().st_size})
        candidates.append(
            {
                "label": label,
                "path": str(path),
                "total_bytes": total,
                "total_mib": round(total / (1024 * 1024), 2),
                "n_files": nfiles,
                "newest_mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(newest)) if newest else None,
                "oldest_mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(oldest)) if oldest else None,
                "refs": refs,
                "snapshots": snapshots,
                "blobs": blobs,
                "blob_total_bytes": sum(b["size"] for b in blobs),
            }
        )

    # 2.2B integrity
    t220, n220, _, _ = dir_stats(m220)
    m220_present = m220.exists()

    # Filename overlap (sharing risk)
    names500 = set()
    names220 = set()
    if m500_hub.exists():
        names500 = {p.name for p in m500_hub.rglob("*") if p.is_file()}
    if m220.exists():
        names220 = {p.name for p in m220.rglob("*") if p.is_file()}
    shared = sorted(names500 & names220)

    # Scripts still requiring 500M?
    scripts = []
    for py in (REPO_ROOT / "scripts").glob("*.py"):
        txt = py.read_text(encoding="utf-8", errors="ignore")
        if "SmolVLM-500M" in txt or "SmolVLM_500M" in txt:
            scripts.append(py.name)

    # Process check is external; note no python experiment running from preflight
    report = {
        "phase": "2C-A",
        "candidates": candidates,
        "smolvlm2_2_2b_present": m220_present,
        "smolvlm2_2_2b_total_bytes": t220,
        "smolvlm2_2_2b_total_mib": round(t220 / (1024 * 1024), 2) if m220_present else 0,
        "shared_filenames_with_2_2b": shared[:50],
        "shared_filename_count": len(shared),
        "scripts_mentioning_500m": scripts,
        "historical_manifest": "datasets/metadata/model_acquisition_manifest.json",
        "notes": [
            "Never delete SmolVLM2-2.2B.",
            "500M historical results remain valid via pinned repo id/revision in manifests.",
            "Shared blob risk assessed by filename overlap and hub layout.",
        ],
    }
    out = REPO_ROOT / "artifacts/cache/phase_2c_a/model_inventory_v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, report)
    print(json.dumps({
        "n_candidates": len(candidates),
        "mib_500m": [c["total_mib"] for c in candidates],
        "mib_2_2b": report["smolvlm2_2_2b_total_mib"],
        "shared_filenames": len(shared),
        "scripts": scripts,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
