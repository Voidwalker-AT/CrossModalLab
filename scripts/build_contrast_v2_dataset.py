#!/usr/bin/env python3
"""Build Phase 1E confound-controlled contrast dataset v2 (CPU; no VLM)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.contrast_v2 import generate_contrast_dataset


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-families", type=int, default=240)
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "cross_modal_contrast_v2",
    )
    p.add_argument("--no-images", action="store_true")
    args = p.parse_args()
    meta = generate_contrast_dataset(
        n_families=args.n_families,
        seed=args.seed,
        out_dir=args.out_dir,
        render_images=not args.no_images,
    )
    print(f"families={meta['n_families']} records={meta['n_records']}")
    print(f"labels={meta['label_counts']}")
    print(f"splits={meta['split_family_counts']}")
    print(f"cf_ok={meta['counterfactual_complete_families']} family_split_ok={meta['family_split_ok']}")
    print(f"C_leak={meta['third_entity_leak_count']} single_mod_sufficient={meta['single_modality_sufficient_count']}")
    print(f"condition_id={meta['condition_id']} condition_only_ba={meta['condition_only_balanced_accuracy']}")
    ok = (
        meta["counterfactual_complete_families"]
        and meta["family_split_ok"]
        and meta["third_entity_leak_count"] == 0
        and meta["label_counts"].get("Yes", 0) == meta["n_families"]
        and meta["label_counts"].get("No", 0) == meta["n_families"]
        and meta["label_counts"].get("Undetermined", 0) == meta["n_families"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
