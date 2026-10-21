#!/usr/bin/env python3
"""Build the fixed Phase 0D.2 36-case behavioral manifest (CPU-safe, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.phase_0d2_manifest import write_manifest


def main() -> int:
    out = REPO_ROOT / "configs" / "phase_0d2" / "behavioral_manifest_v1.json"
    manifest = write_manifest(out)
    print(f"Wrote manifest: {out}")
    print(f"n_samples: {manifest['n_samples']}")
    print(f"conditions: {manifest['condition_counts']}")
    print(f"samples_sha256: {manifest['samples_sha256']}")
    print(f"source_jsonl_sha256: {manifest['source_jsonl_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
