#!/usr/bin/env python3
"""Build exact-image counterfactual pair dataset for Phase 2B-R (CPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.causal.exact_pairs import build_dataset


def main() -> int:
    out = REPO_ROOT / "datasets/exact_pairs_v1"
    meta = build_dataset(out, n_families=3)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
