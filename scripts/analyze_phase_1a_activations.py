#!/usr/bin/env python3
"""Phase 1A descriptive (non-causal) activation analysis from saved vectors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.instrumentation.multi_layer_capture import cosine_similarity, l2_norm


def load_meta(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("status") == "ok":
                rows.append(row)
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "instrumentation" / "phase_1a",
    )
    args = p.parse_args()
    meta_path = args.out_dir / "activations_meta_v1.jsonl"
    rows = load_meta(meta_path)

    # Load vectors
    vectors: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        vec_file = REPO_ROOT / row["vectors_file"]
        vectors[row["sample_id"]] = json.loads(vec_file.read_text(encoding="utf-8"))

    analysis: Dict[str, Any] = {
        "phase": "1A",
        "label": "DESCRIPTIVE / NON-CAUSAL",
        "n_samples": len(rows),
        "sample_ids": [r["sample_id"] for r in rows],
        "input_conditions": defaultdict(int),
        "l2_norm_by_layer_selector": defaultdict(list),
        "cosine_matched_pairs": [],
        "correct_vs_incorrect_last_token_l2": {},
        "layer_depth_last_token_l2": defaultdict(list),
    }

    for r in rows:
        analysis["input_conditions"][r["analytical_input_condition"]] += 1

    # L2 norms
    for sid, vec in vectors.items():
        for layer, bundle in vec["layers"].items():
            for sel, tok in bundle["tokens"].items():
                analysis["l2_norm_by_layer_selector"][f"L{layer}:{sel}"].append(
                    {"sample_id": sid, "l2_norm": tok["l2_norm"]}
                )
                if sel == "last":
                    analysis["layer_depth_last_token_l2"][f"L{layer}"].append(tok["l2_norm"])

    # Matched pair cosines (true pairs only)
    by_id = {r["sample_id"]: r for r in rows}
    pair_groups = defaultdict(list)
    for r in rows:
        if r.get("genuinely_paired") and r.get("matched_group_id"):
            pair_groups[r["matched_group_id"]].append(r["sample_id"])

    for gid, sids in pair_groups.items():
        if len(sids) != 2:
            continue
        a, b = sids
        if a not in vectors or b not in vectors:
            continue
        for layer in vectors[a]["layers"]:
            for sel in ("last", "mean_image", "last_text_non_image"):
                if sel not in vectors[a]["layers"][layer]["tokens"]:
                    continue
                if sel not in vectors[b]["layers"][layer]["tokens"]:
                    continue
                va = vectors[a]["layers"][layer]["tokens"][sel]["vector"]
                vb = vectors[b]["layers"][layer]["tokens"][sel]["vector"]
                analysis["cosine_matched_pairs"].append(
                    {
                        "matched_group_id": gid,
                        "sample_a": a,
                        "sample_b": b,
                        "layer": int(layer),
                        "selector": sel,
                        "cosine": round(cosine_similarity(va, vb), 6),
                        "l2_a": l2_norm(va),
                        "l2_b": l2_norm(vb),
                        "note": "Same image for forward/reversal pairs; query text differs.",
                    }
                )

    # Correct vs incorrect last-token L2 at L12
    for layer in ("3", "12", "18", "23"):
        correct = []
        incorrect = []
        for r in rows:
            sid = r["sample_id"]
            if sid not in vectors or layer not in vectors[sid]["layers"]:
                continue
            tok = vectors[sid]["layers"][layer]["tokens"].get("last")
            if not tok:
                continue
            if r.get("is_correct_0d2"):
                correct.append(tok["l2_norm"])
            else:
                incorrect.append(tok["l2_norm"])
        analysis["correct_vs_incorrect_last_token_l2"][f"L{layer}"] = {
            "n_correct": len(correct),
            "n_incorrect": len(incorrect),
            "mean_l2_correct": round(sum(correct) / len(correct), 6) if correct else None,
            "mean_l2_incorrect": round(sum(incorrect) / len(incorrect), 6) if incorrect else None,
        }

    # CSV summary
    csv_path = args.out_dir / "activation_l2_summary_v1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "role", "condition", "layer", "selector", "token_index", "l2_norm", "native_dtype"])
        for r in rows:
            sid = r["sample_id"]
            vec = vectors[sid]
            for layer, bundle in vec["layers"].items():
                for sel, tok in bundle["tokens"].items():
                    w.writerow([sid, r["role"], r["analytical_input_condition"], layer, sel, tok["token_index"], tok["l2_norm"], tok["native_dtype"]])

    # JSON-ify defaultdicts
    out = {
        "phase": analysis["phase"],
        "label": analysis["label"],
        "n_samples": analysis["n_samples"],
        "sample_ids": analysis["sample_ids"],
        "input_conditions": dict(analysis["input_conditions"]),
        "l2_norm_by_layer_selector": {k: v for k, v in analysis["l2_norm_by_layer_selector"].items()},
        "cosine_matched_pairs": analysis["cosine_matched_pairs"],
        "correct_vs_incorrect_last_token_l2": analysis["correct_vs_incorrect_last_token_l2"],
        "layer_depth_last_token_l2": {k: v for k, v in analysis["layer_depth_last_token_l2"].items()},
        "limitations": [
            "Descriptive only; not causal.",
            "Mean image vector is an aggregate, not a neuron.",
            "Matched-pair cosine on same image reflects query-conditioned residual differences, not 'understanding'.",
            "Small N; not statistically representative.",
        ],
    }
    out_path = args.out_dir / "descriptive_analysis_v1.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {csv_path}")
    print(f"Matched pair cosine rows: {len(analysis['cosine_matched_pairs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
