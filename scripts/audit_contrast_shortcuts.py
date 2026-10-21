#!/usr/bin/env python3
"""Shortcut baselines for confound-controlled contrast dataset (CPU)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.contrast_v2 import LABELS, load_contrast_records
from src.instrumentation.capture_integrity import compute_probe_metrics
from src.probes.linear_probe import majority_baseline, random_label_baseline


def _ba(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, Any]:
    return compute_probe_metrics(y_true, y_pred, list(LABELS))


def _family_disjoint_maps(records: Sequence[Dict[str, Any]]):
    """Split families 80/20 for honest shortcut evaluation (no singleton full-set lookup)."""
    fams = sorted({r["family_id"] for r in records})
    n_train = max(1, int(0.8 * len(fams)))
    train_f = set(fams[:n_train])
    train = [r for r in records if r["family_id"] in train_f]
    evalr = [r for r in records if r["family_id"] not in train_f]
    return train, evalr


def _map_predict(
    train: Sequence[Dict[str, Any]],
    evalr: Sequence[Dict[str, Any]],
    key_fn,
) -> Dict[str, Any]:
    by = defaultdict(list)
    for r in train:
        by[key_fn(r)].append(r["ground_truth_label"])
    maj = {k: Counter(v).most_common(1)[0][0] for k, v in by.items()}
    default = Counter([r["ground_truth_label"] for r in train]).most_common(1)[0][0]
    y_true = [r["ground_truth_label"] for r in evalr]
    y_pred = [maj.get(key_fn(r), default) for r in evalr]
    return _ba(y_true, y_pred)


def condition_only_baseline(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    # Constant condition: majority on train, predict on eval
    train, evalr = _family_disjoint_maps(records)
    default = Counter([r["ground_truth_label"] for r in train]).most_common(1)[0][0]
    y_true = [r["ground_truth_label"] for r in evalr]
    return _ba(y_true, [default] * len(y_true))


def template_only_baseline(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def sig(p: str) -> str:
        return p.split("\n")[0][:80]

    return _map_predict(*_family_disjoint_maps(records), key_fn=lambda r: sig(r["prompt"]))


def entity_identity_baseline(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return _map_predict(
        *_family_disjoint_maps(records),
        key_fn=lambda r: (r["entities"]["A"], r["entities"]["C"]),
    )


def relation_direction_baseline(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return _map_predict(*_family_disjoint_maps(records), key_fn=lambda r: tuple(r["visual_entities_shown"]))


def color_shape_baseline(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def key(r):
        from scripts.generate_controlled_benchmark import ENTITY_PALETTE

        cols = []
        for e in r["visual_entities_shown"]:
            cols.append((ENTITY_PALETTE[e]["color"], ENTITY_PALETTE[e]["shape"]))
        return tuple(sorted(cols))

    return _map_predict(*_family_disjoint_maps(records), key_fn=key)


def lexical_bow_baseline(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def key(r):
        import re

        line = r["prompt"].split("\n")[0].lower()
        return tuple(sorted(re.findall(r"[a-z]+", line)))

    return _map_predict(*_family_disjoint_maps(records), key_fn=key)


def layout_baseline(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def key(r):
        lay = r.get("layout") or {}
        # Coarse bins to avoid unique-key lookup
        return (
            tuple(int(x) // 16 for x in (lay.get("bg_rgb") or [0, 0, 0])),
            tuple(int(x) // 16 for x in (lay.get("object_size") or [0, 0])),
            int(lay.get("gap_px") or 0) // 24,
        )

    return _map_predict(*_family_disjoint_maps(records), key_fn=key)


def audit_shortcuts(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    y_true = [r["ground_truth_label"] for r in records]
    results = {
        "n": len(records),
        "label_counts": dict(Counter(y_true)),
        "condition_only": condition_only_baseline(records),
        "template_only": template_only_baseline(records),
        "entity_identity": entity_identity_baseline(records),
        "relation_direction": relation_direction_baseline(records),
        "color_shape": color_shape_baseline(records),
        "lexical_bow": lexical_bow_baseline(records),
        "layout_metadata": layout_baseline(records),
        "majority": majority_baseline([r["ground_truth_label"] for r in records]),
        "random_label_seed0": random_label_baseline([r["ground_truth_label"] for r in records], seed=0),
        "notes": [
            "Baselines use full-dataset majority maps as diagnostic upper bounds for leakage.",
            "Chance for balanced 3-way = 1/3 balanced accuracy.",
            "Finite-sample variation expected; not all baselines equal exactly chance.",
        ],
    }
    # Flag suspicious shortcut
    suspicious = []
    for name in ("template_only", "entity_identity", "relation_direction", "color_shape", "lexical_bow", "layout_metadata"):
        ba = results[name]["balanced_accuracy"]
        if ba >= 0.85:
            suspicious.append({name: ba})
    results["suspicious_shortcuts"] = suspicious
    results["condition_only_ok"] = results["condition_only"]["balanced_accuracy"] <= 0.40
    results["verdict"] = (
        "SHORTCUTS_ACCEPTABLE" if not suspicious and results["condition_only_ok"] else "SHORTCUTS_SUSPICIOUS"
    )
    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "datasets/cross_modal_contrast_v2/cross_modal_contrast_v2.jsonl",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts/probes/phase_1e/shortcut_audit_v2.json",
    )
    args = p.parse_args()
    records = load_contrast_records(args.dataset)
    report = audit_shortcuts(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "verdict": report["verdict"],
        "condition_only_ba": report["condition_only"]["balanced_accuracy"],
        "suspicious": report["suspicious_shortcuts"],
        "n": report["n"],
    }, indent=2))
    return 0 if report["verdict"] == "SHORTCUTS_ACCEPTABLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
