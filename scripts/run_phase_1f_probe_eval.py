#!/usr/bin/env python3
"""Phase 1F — Reliable multi-spec probe evaluation + shortcut comparison on development features.

Does not touch final holdout. Does not train the VLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.benchmark.contrast_v2 import LABELS, load_contrast_records
from src.config.paths import get_artifacts_dir
from src.instrumentation.capture_integrity import atomic_write_json, compute_probe_metrics
from src.probes.eval_protocol import (
    LAYERS,
    TOKEN_SELECTORS,
    FeatureSpec,
    evaluate_feature_spec,
    extract_token_slice,
    frozen_dev_split,
    group_bootstrap_ci,
    load_feature_rows,
    spec_hash,
)
from src.probes.linear_probe import (
    ProbeDataset,
    fit_standardized_probe,
    group_disjoint_split,
    majority_baseline,
    random_label_baseline,
)


def text_lexical_baseline(train_rows, val_rows) -> Dict[str, Any]:
    """Train-only majority per premise first-line signature."""
    from collections import defaultdict

    def sig(r):
        return r["prompt"].split("\n")[0][:80]

    by = defaultdict(list)
    for r in train_rows:
        by[sig(r)].append(r["ground_truth_label"])
    maj = {k: Counter(v).most_common(1)[0][0] for k, v in by.items()}
    default = Counter([r["ground_truth_label"] for r in train_rows]).most_common(1)[0][0]
    y_true = [r["ground_truth_label"] for r in val_rows]
    y_pred = [maj.get(sig(r), default) for r in val_rows]
    return compute_probe_metrics(y_true, y_pred, list(LABELS))


def entity_baseline(train_rows, val_rows) -> Dict[str, Any]:
    from collections import defaultdict

    def key(r):
        return (r["entities"]["A"], r["entities"]["C"])

    by = defaultdict(list)
    for r in train_rows:
        by[key(r)].append(r["ground_truth_label"])
    maj = {k: Counter(v).most_common(1)[0][0] for k, v in by.items()}
    default = Counter([r["ground_truth_label"] for r in train_rows]).most_common(1)[0][0]
    y_true = [r["ground_truth_label"] for r in val_rows]
    y_pred = [maj.get(key(r), default) for r in val_rows]
    return compute_probe_metrics(y_true, y_pred, list(LABELS))


def layout_baseline(train_rows, val_rows) -> Dict[str, Any]:
    from collections import defaultdict

    def key(r):
        lay = r.get("layout") or {}
        return (
            tuple(int(x) // 16 for x in (lay.get("bg_rgb") or [0, 0, 0])),
            tuple(int(x) // 16 for x in (lay.get("object_size") or [0, 0])),
            int(lay.get("gap_px") or 0) // 24,
        )

    by = defaultdict(list)
    for r in train_rows:
        by[key(r)].append(r["ground_truth_label"])
    maj = {k: Counter(v).most_common(1)[0][0] for k, v in by.items()}
    default = Counter([r["ground_truth_label"] for r in train_rows]).most_common(1)[0][0]
    y_true = [r["ground_truth_label"] for r in val_rows]
    y_pred = [maj.get(key(r), default) for r in val_rows]
    return compute_probe_metrics(y_true, y_pred, list(LABELS))


def relation_direction_baseline(train_rows, val_rows) -> Dict[str, Any]:
    from collections import defaultdict

    def key(r):
        return tuple(r.get("visual_entities_shown") or [])

    by = defaultdict(list)
    for r in train_rows:
        by[key(r)].append(r["ground_truth_label"])
    maj = {k: Counter(v).most_common(1)[0][0] for k, v in by.items()}
    default = Counter([r["ground_truth_label"] for r in train_rows]).most_common(1)[0][0]
    y_true = [r["ground_truth_label"] for r in val_rows]
    y_pred = [maj.get(key(r), default) for r in val_rows]
    return compute_probe_metrics(y_true, y_pred, list(LABELS))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--meta-jsonl",
        type=Path,
        default=get_artifacts_dir() / "probes" / "phase_1f" / "features_meta_v1.jsonl",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "datasets/cross_modal_contrast_v2/cross_modal_contrast_v2.jsonl",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    if not args.meta_jsonl.is_file():
        # fallback to 1E meta for reproduction
        alt = get_artifacts_dir() / "probes" / "phase_1e" / "contrast_probe_meta_v1.jsonl"
        print(f"NOTE: {args.meta_jsonl} missing; using {alt}")
        args.meta_jsonl = alt

    X, rows = load_feature_rows(args.meta_jsonl, REPO_ROOT)
    ds = {r["sample_id"]: r for r in load_contrast_records(args.dataset)}

    # Join layout/entity from dataset records
    for r in rows:
        src = ds.get(r["sample_id"])
        if src:
            r["layout"] = src.get("layout")
            r["entities"] = src.get("entities")
            r["visual_entities_shown"] = src.get("visual_entities_shown")
            r["prompt"] = src.get("prompt", "")

    train_f, val_f = frozen_dev_split(rows)
    print(f"Frozen dev split: train_families={len(train_f)} val_families={len(val_f)}")

    train_idx = [i for i, r in enumerate(rows) if r["family_id"] in set(train_f)]
    val_idx = [i for i, r in enumerate(rows) if r["family_id"] in set(val_f)]
    labels = [rows[i]["ground_truth_label"] for i in range(len(rows))]
    groups = [rows[i]["family_id"] for i in range(len(rows))]
    metas = rows
    data = ProbeDataset(X, labels, groups, metas)
    tr = ProbeDataset(X[train_idx], [labels[i] for i in train_idx], [groups[i] for i in train_idx], [metas[i] for i in train_idx])
    va = ProbeDataset(X[val_idx], [labels[i] for i in val_idx], [groups[i] for i in val_idx], [metas[i] for i in val_idx])

    # Sanity: no overlap
    assert set(tr.group_ids).isdisjoint(va.group_ids)

    specs = []
    for layer in LAYERS:
        for sel in TOKEN_SELECTORS:
            specs.append(FeatureSpec(layer=layer, token_selector=sel, use_pca=False))
    # One PCA control
    specs.append(FeatureSpec(layer=12, token_selector="last", use_pca=True, pca_dim=64))

    results = []
    for spec in specs:
        print(f"Evaluating {spec.key()} pca={spec.use_pca}...", flush=True)
        results.append(evaluate_feature_spec(tr, va, spec))

    # Baselines on same val families
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    y_val = [r["ground_truth_label"] for r in val_rows]
    y_train = [r["ground_truth_label"] for r in train_rows]
    default = Counter(y_train).most_common(1)[0][0]
    baselines = {
        "majority_train_default": compute_probe_metrics(y_val, [default] * len(y_val), list(LABELS)),
        "condition_only_single_id": compute_probe_metrics(y_val, [default] * len(y_val), list(LABELS)),
        "random_label_seed0": random_label_baseline(y_val, seed=0),
        "text_lexical": text_lexical_baseline(train_rows, val_rows),
        "entity_identity": entity_baseline(train_rows, val_rows),
        "relation_direction": relation_direction_baseline(train_rows, val_rows),
        "image_layout": layout_baseline(train_rows, val_rows),
    }

    # Rank neural specs
    neural_sorted = sorted(results, key=lambda r: r["balanced_accuracy"], reverse=True)
    best = neural_sorted[0]
    best_ci = best["group_bootstrap_ci"]
    best_ba = best["balanced_accuracy"]
    max_baseline_ba = max(v["balanced_accuracy"] for v in baselines.values())
    exceeds = best_ba > max_baseline_ba + 0.05
    # CI excludes 0.4 (above chance-ish band)
    ci_excludes_chance = best_ci["ci_low"] > 1 / 3

    interpretation = {
        "best_spec": best["spec_key"],
        "best_ba": best_ba,
        "best_group_bootstrap_ci": best_ci,
        "max_shortcut_baseline_ba": max_baseline_ba,
        "exceeds_baselines_by_gt_0.05": exceeds,
        "ci_low_above_chance": ci_excludes_chance,
        "n_val_families": len(val_f),
        "scientific_status": (
            "EXPLORATORY_POSITIVE_DECODE"
            if exceeds and ci_excludes_chance
            else (
                "EXPLORATORY_WEAK_OR_INCONCLUSIVE"
                if best_ba > max_baseline_ba
                else "NO_ADVANTAGE_OVER_SHORTCUTS"
            )
        ),
        "causal_claim": False,
        "notes": [
            "Linear decodability is not causal model usage.",
            "Validation families are development-only; not final holdout.",
            "Group bootstrap resamples families.",
            "Best-spec selection uses development validation (not holdout).",
        ],
    }

    out_dir = args.out_dir or (get_artifacts_dir() / "probes" / "phase_1f")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": "1F",
        "spec_version_hash": spec_hash(),
        "meta_jsonl": str(args.meta_jsonl),
        "n_features": int(X.shape[0]),
        "n_train_families": len(train_f),
        "n_val_families": len(val_f),
        "train_families": train_f,
        "val_families": val_f,
        "final_holdout_touched": False,
        "probe_results": results,
        "baselines": baselines,
        "interpretation": interpretation,
        "runtime_note": "See separate latency artifact if present.",
    }
    atomic_write_json(out_dir / "probe_reliability_summary_v1.json", summary)
    print(json.dumps({
        "best": interpretation["best_spec"],
        "best_ba": best_ba,
        "max_baseline_ba": max_baseline_ba,
        "status": interpretation["scientific_status"],
        "ci": best_ci,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
