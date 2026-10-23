"""Deterministic stored-data tables; stdlib only, no inference or training.

Family-cluster percentile intervals describe repeatedly used development data.
They do not account for adaptive selection or establish generalization.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training.answer_parser import strict_parse_answer

RUN = ROOT / "artifacts/training/phase_2i_b/runs/56e277e421e442b08d33b107fc934494"
LABELS = ("Yes", "No", "Undetermined")


def quantile(values, q):
    a = sorted(values)
    x = (len(a) - 1) * q
    lo = int(x)
    return a[lo] + (a[min(lo + 1, len(a) - 1)] - a[lo]) * (x - lo)


def bootstrap(family_scores):
    rng = random.Random(20261023)
    a = list(family_scores.values())
    draws = [sum(rng.choices(a, k=len(a))) / len(a) for _ in range(2000)]
    return {"ci_low": quantile(draws, .025), "ci_high": quantile(draws, .975),
            "n_boot": 2000, "seed": 20261023, "independent_units": len(a),
            "interpretation": "descriptive development family-cluster percentile CI; selection bias uncorrected"}


def summarize(rows):
    assert len({r["sample_id"] for r in rows}) == len(rows)
    assert all(strict_parse_answer(r["raw_completion"])["parsed_label"] == r["parsed_answer"] for r in rows)
    groups = sorted({r["family_id"] for r in rows})
    assert all(Counter(r["ground_truth_label"] for r in rows if r["family_id"] == f)
               == Counter(LABELS) for f in groups)
    scores = {f: sum(r["parsed_answer"] == r["ground_truth_label"] for r in rows if r["family_id"] == f) / 3
              for f in groups}
    recall = {c: sum(r["parsed_answer"] == c for r in rows if r["ground_truth_label"] == c) /
              sum(r["ground_truth_label"] == c for r in rows) for c in LABELS}
    return {"n": len(rows), "families": len(groups), "balanced_accuracy": sum(recall.values()) / 3,
            "per_class_recall": recall, "invalid_output_rate": sum(r["parsed_answer"] not in LABELS for r in rows) / len(rows),
            "abstention_rate": sum(r["parsed_answer"] == "Undetermined" for r in rows) / len(rows),
            "over_abstention_on_sufficient": sum(r["parsed_answer"] == "Undetermined" and r["ground_truth_label"] != "Undetermined" for r in rows) / (2 * len(groups)),
            "false_definitive_on_underdetermined": sum(r["parsed_answer"] in ("Yes", "No") and r["ground_truth_label"] == "Undetermined" for r in rows) / len(groups),
            "exact_label_output_count": sum(r["raw_completion"].strip() in LABELS for r in rows),
            "prediction_counts": dict(Counter(r["parsed_answer"] for r in rows)),
            "family_accuracy": scores, "balanced_accuracy_interval": bootstrap(scores)}


def main():
    ce_path = ROOT / "artifacts/training/phase_2i_a/phase_2i_a_report_v1.json"
    before_path, after_path = RUN / "baseline.json", RUN / "evaluation_step_0005.json"
    ce, before, after = [json.loads(p.read_text(encoding="utf-8")) for p in (ce_path, before_path, after_path)]
    conditions = {"ce_frozen_standard": ce["baseline_standard"]["raw"],
                  "ce_step5_standard": ce["val_step_5"]["raw"],
                  "ce_step50_standard": ce["val_final"]["raw"],
                  "pilot_frozen_standard": before["standard"]["raw"],
                  "pilot_frozen_abstention_prompt": before["abstention_prompt"]["raw"],
                  "pilot_step5_standard": after["standard"]["raw"],
                  "pilot_step5_abstention_prompt": after["abstention_prompt"]["raw"]}
    ref = {r["sample_id"]: (r["family_id"], r["ground_truth_label"], r["prompt_sha256"], r["image_sha256"])
           for r in conditions["ce_step5_standard"]}
    # Abstention wording changes deliberately; stored prompt hash is the original
    # record hash in both historical runners. It is not the augmented-prompt hash.
    for rows in conditions.values():
        assert {r["sample_id"]: (r["family_id"], r["ground_truth_label"], r["prompt_sha256"], r["image_sha256"])
                for r in rows} == ref
    summary = {name: summarize(rows) for name, rows in conditions.items()}
    a, b = summary["ce_step5_standard"], summary["pilot_step5_standard"]
    delta = {f: b["family_accuracy"][f] - a["family_accuracy"][f] for f in a["family_accuracy"]}
    payload = {"classification": "PRELIMINARY", "split": "validation", "holdout_evaluated": False,
               "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (ce_path, before_path, after_path)}, "conditions": summary,
               "paired_step5_delta": {"point": sum(delta.values()) / len(delta), **bootstrap(delta)},
               "comparability": "same 18 IDs, six families, source prompt/image hashes, strict_answer_v1, greedy 16-token generation; separate processes and minor numerical nondeterminism possible",
               "caveats": ["Repeated development-set inspection; no independent test",
                           "Historical parser accepts trailing garbage; exact-label counts shown separately",
                           "CE step-five abstention-prompt evaluation unavailable",
                           "Source prompt hashes for augmented condition identify original records only"]}
    out = Path(__file__).resolve().parent
    (out / "development_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out / "family_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["condition", "sample_id", "family_id", "ground_truth_label", "parsed_answer", "raw_completion", "prompt_sha256", "image_sha256"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for condition, rows in conditions.items():
            for row in sorted(rows, key=lambda r: r["sample_id"]):
                writer.writerow({"condition": condition, **{k: row[k] for k in fields if k != "condition"}})
    print(json.dumps({k: {x: v[x] for x in ("balanced_accuracy", "per_class_recall", "invalid_output_rate", "abstention_rate", "exact_label_output_count")} for k, v in summary.items()}, indent=2))
    print(json.dumps(payload["paired_step5_delta"], indent=2))


if __name__ == "__main__":
    main()
