#!/usr/bin/env python3
"""Generate the Phase 1B cross-modal dependency dataset (CPU; no VLM inference)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.cross_modal_dependency import generate_dataset


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-families", type=int, default=240)
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "cross_modal_dependency_v1",
    )
    p.add_argument("--no-images", action="store_true")
    args = p.parse_args()

    result = generate_dataset(
        n_families=args.n_families,
        seed=args.seed,
        out_dir=args.out_dir,
        render_images=not args.no_images,
    )
    print(f"families={result['n_families']} records={result['n_records']}")
    print(f"labels={result['label_counts']}")
    print(f"splits={result['split_family_counts']}")
    print(f"solver_mismatches={len(result['solver_mismatches'])}")
    print(f"group_disjoint_splits_ok={result['group_disjoint_splits_ok']}")
    return 0 if result["group_disjoint_splits_ok"] and not result["solver_mismatches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
