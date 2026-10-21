"""Label–condition confounding analysis for Phase 1C balanced sampling (CPU)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.config.paths import REPO_ROOT

# Deterministic mapping in balanced protocol
CONDITION_TO_LABEL = {
    "both_premises": "Yes",
    "both_modified": "No",
    "underdetermined_fork": "Undetermined",
}
LABEL_TO_CONDITION = {v: k for k, v in CONDITION_TO_LABEL.items()}


def load_balanced(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def analyze_confounding(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Quantify whether condition identity determines the 3-way label."""
    n = len(records)
    pair_counts = Counter((r["condition"], r["ground_truth_label"]) for r in records)
    cond_counts = Counter(r["condition"] for r in records)
    label_counts = Counter(r["ground_truth_label"] for r in records)

    # Condition→label purity
    purity = {}
    for cond in cond_counts:
        labs = [lab for (c, lab), k in pair_counts.items() if c == cond for _ in range(k)]
        if not labs:
            continue
        top = Counter(labs).most_common(1)[0]
        purity[cond] = {
            "n": cond_counts[cond],
            "majority_label": top[0],
            "purity": top[1] / cond_counts[cond],
        }

    # Condition-label lookup accuracy (perfect if bijective)
    lookup_correct = 0
    for r in records:
        expected = CONDITION_TO_LABEL.get(r["condition"])
        if expected == r["ground_truth_label"]:
            lookup_correct += 1
    lookup_acc = lookup_correct / n if n else 0.0

    # Unique (condition, label) pairs — bijectivity check
    bijective = all(p["purity"] == 1.0 for p in purity.values()) and len(CONDITION_TO_LABEL) == len(
        set(CONDITION_TO_LABEL.values())
    )

    return {
        "n_records": n,
        "condition_counts": dict(cond_counts),
        "label_counts": dict(label_counts),
        "condition_label_pairs": {f"{c}|{l}": k for (c, l), k in pair_counts.items()},
        "condition_label_purity": purity,
        "condition_label_lookup_accuracy": round(lookup_acc, 6),
        "condition_label_bijective": bool(bijective),
        "verdict": "CONDITION-CONFOUNDED" if bijective and lookup_acc >= 0.999 else "PARTIAL_OR_UNRESOLVED",
        "interpretation": (
            "On the Phase 1C balanced three-way sample, ground-truth labels are a deterministic "
            "function of condition templates (both_premises/both_modified/underdetermined_fork). "
            "A three-way probe accuracy cannot distinguish logical representation from condition identity."
            if bijective
            else "Condition and label are not fully bijective; residual confounding may remain."
        ),
        "smallest_defensible_extension": [
            "Add multiple logical realizations per label with different surface templates and premise structures.",
            "E.g. include underdetermined_fork AND visual_removed AND text_removed as Undetermined under distinct prompts.",
            "Include multiple contradicted constructions (both_modified AND visual_reversed if solver says contradicted).",
            "Evaluate composition probes on contrast sets where condition text is held constant but entity order changes.",
            "Minimum: 2+ independent condition templates per 3-way label with family-disjoint evaluation.",
        ],
    }


def main() -> int:
    path = REPO_ROOT / "configs/phase_1c/balanced_development_v1.jsonl"
    records = load_balanced(path)
    report = analyze_confounding(records)
    out = REPO_ROOT / "artifacts/probes/phase_1d/label_condition_confounding_v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["n_records", "condition_label_lookup_accuracy", "verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
