"""Answer parsing and behavioral metrics for Phase 0D.2 (CPU-safe, pure Python)."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

VALID_LABELS = ("Yes", "No", "Undetermined")
INVALID = "Invalid"
MISSING = "Missing"


def parse_model_answer(raw_completion: Optional[str]) -> Dict[str, str]:
    """Parse a raw completion into {parsed_label, parse_status, first_token}.

    Does not alter ground truth. Returns parsed_label in
    {Yes, No, Undetermined, Invalid}.
    """
    if raw_completion is None:
        return {"parsed_label": MISSING, "parse_status": "missing", "first_token": ""}

    text = raw_completion.strip()
    if not text:
        return {"parsed_label": INVALID, "parse_status": "empty", "first_token": ""}

    if "Assistant:" in text:
        text = text.split("Assistant:")[-1].strip()

    lowered = text.lower()
    tokens = [t.strip(".,;:!?\"' \n\t") for t in lowered.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return {"parsed_label": INVALID, "parse_status": "empty", "first_token": ""}

    first = tokens[0]
    if first in {"yes", "yeah", "yep", "correct", "true"}:
        return {"parsed_label": "Yes", "parse_status": "parsed", "first_token": first}
    if first in {"no", "nope", "incorrect", "false"}:
        return {"parsed_label": "No", "parse_status": "parsed", "first_token": first}
    if first in {"undetermined", "unknown", "uncertain", "unclear", "undecidable"}:
        return {"parsed_label": "Undetermined", "parse_status": "parsed", "first_token": first}
    # Multi-word fallback: search first three tokens for an answer keyword
    head = " ".join(tokens[:3])
    if "undetermined" in head or "undecidable" in head:
        return {"parsed_label": "Undetermined", "parse_status": "parsed_later", "first_token": first}
    if head.startswith("yes"):
        return {"parsed_label": "Yes", "parse_status": "parsed_later", "first_token": first}
    if head.startswith("no"):
        return {"parsed_label": "No", "parse_status": "parsed_later", "first_token": first}

    return {"parsed_label": INVALID, "parse_status": "unparseable", "first_token": first}


def build_prompt(text_premise: str, query: str, modality: str) -> str:
    """Fixed prompt template for controlled relational queries."""
    if modality == "text_only":
        return (
            f"{text_premise}\n\n"
            f"Question: Based on the statements, is {query}\n"
            "Carefully evaluate whether the question is entailed (Yes), contradicted (No), "
            "or underdetermined from the evidence (Undetermined).\n"
            "Answer directly with: Yes, No, or Undetermined."
        )
    return (
        f"{text_premise}\n\n"
        f"Question: Based on the image and statement, is {query}\n"
        "Carefully evaluate whether the question is entailed (Yes), contradicted (No), "
        "or underdetermined from the evidence (Undetermined).\n"
        "Answer directly with: Yes, No, or Undetermined."
    )


def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def bootstrap_ci_binary(
    outcomes: Sequence[bool],
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Percentile bootstrap CI for a binary accuracy mean (B>=1000)."""
    if not outcomes:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0, "n_boot": n_boot}

    rng = random.Random(seed)
    n = len(outcomes)
    vals = [1.0 if o else 0.0 for o in outcomes]
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo_idx = int((alpha / 2.0) * n_boot)
    hi_idx = min(n_boot - 1, int((1.0 - alpha / 2.0) * n_boot))
    mean = sum(vals) / n
    return {
        "mean": round(mean, 6),
        "ci_low": round(means[lo_idx], 6),
        "ci_high": round(means[hi_idx], 6),
        "n": n,
        "n_boot": n_boot,
        "seed": seed,
    }


def confusion_3x3(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = VALID_LABELS,
) -> Dict[str, Dict[str, int]]:
    matrix = {t: {p: 0 for p in list(labels) + [INVALID, MISSING]} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t not in matrix:
            matrix[t] = {lab: 0 for lab in list(labels) + [INVALID, MISSING]}
        if p not in matrix[t]:
            matrix[t][p] = 0
        matrix[t][p] += 1
    return matrix


def balanced_accuracy(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = VALID_LABELS,
) -> float:
    recalls = []
    for lab in labels:
        idx = [i for i, t in enumerate(y_true) if t == lab]
        if not idx:
            continue
        correct = sum(1 for i in idx if y_pred[i] == lab)
        recalls.append(_safe_div(correct, len(idx)))
    return _safe_div(sum(recalls), len(recalls)) if recalls else 0.0


def unsupported_definite_rate(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> Dict[str, float]:
    """Among underdetermined cases, rate of definite Yes/No answers."""
    idx = [i for i, t in enumerate(y_true) if t == "Undetermined"]
    if not idx:
        return {"rate": 0.0, "n_underdetermined": 0, "n_definite": 0}
    definite = sum(1 for i in idx if y_pred[i] in {"Yes", "No"})
    return {
        "rate": round(_safe_div(definite, len(idx)), 6),
        "n_underdetermined": len(idx),
        "n_definite": definite,
    }


def compute_behavioral_metrics(records: Iterable[Dict]) -> Dict:
    """Compute Phase 0D.2 metrics from prediction records.

    Matched counterfactual pairs are not treated as independent for pair
    consistency; bootstrap CIs are computed over case-level outcomes.
    """
    recs = [r for r in records if r.get("prediction_status") in {"ok", "failed"}]
    # Primary scored labels: use length-normalized top candidate when available
    scored = []
    for r in recs:
        if r.get("prediction_status") != "ok":
            continue
        gt = r["ground_truth_label"]
        pred = r["parsed_answer"]
        cond = r["condition"]
        scored.append(
            {
                "sample_id": r["sample_id"],
                "gt": gt,
                "pred": pred,
                "cond": cond,
                "correct": pred == gt,
                "modality": r.get("modality"),
                "relation": r.get("relation"),
                "matched_group": r.get("matched_group_id"),
                "image_sha256": r.get("image_sha256"),
            }
        )

    y_true = [s["gt"] for s in scored]
    y_pred = [s["pred"] for s in scored]

    by_cond = defaultdict(list)
    for s in scored:
        by_cond[s["cond"]].append(s["correct"])

    conf = confusion_3x3(y_true, y_pred)

    entailed = [s for s in scored if s["gt"] == "Yes"]
    contradicted = [s for s in scored if s["gt"] == "No"]
    underdet = [s for s in scored if s["gt"] == "Undetermined"]

    # Matched counterfactual consistency: within matched_group, both members correct
    groups = defaultdict(list)
    for s in scored:
        g = s.get("matched_group")
        if g:
            groups[g].append(s)
    pair_both_correct = 0
    pair_total = 0
    for g, members in groups.items():
        if len(members) < 2:
            continue
        pair_total += 1
        if all(m["correct"] for m in members):
            pair_both_correct += 1

    left = [s for s in scored if s.get("relation") == "left_of"]
    right = [s for s in scored if s.get("relation") == "right_of"]
    visual = [s for s in scored if s.get("modality") == "visual"]
    text_only = [s for s in scored if s.get("modality") == "text_only"]

    latencies = [r.get("latency_sec") for r in recs if isinstance(r.get("latency_sec"), (int, float))]
    peak_vram = [r.get("peak_vram_mib") for r in recs if isinstance(r.get("peak_vram_mib"), (int, float))]

    overall_ci = bootstrap_ci_binary([s["correct"] for s in scored])
    abstain_ci = bootstrap_ci_binary([s["pred"] == "Undetermined" for s in underdet])

    return {
        "n_scored": len(scored),
        "n_failed_or_missing": len(recs) - len(scored),
        "overall_accuracy": overall_ci,
        "balanced_accuracy": round(balanced_accuracy(y_true, y_pred), 6),
        "confusion_matrix": conf,
        "accuracy_by_condition": {
            cond: {
                "n": len(vals),
                "n_correct": int(sum(vals)),
                "accuracy": round(_safe_div(sum(vals), len(vals)), 6),
                "ci": bootstrap_ci_binary(vals),
            }
            for cond, vals in sorted(by_cond.items())
        },
        "entailed_accuracy": bootstrap_ci_binary([s["correct"] for s in entailed]),
        "contradicted_accuracy": bootstrap_ci_binary([s["correct"] for s in contradicted]),
        "underdetermined_abstention_accuracy": abstain_ci,
        "underdetermined_definite_answer_rate": unsupported_definite_rate(y_true, y_pred),
        "matched_counterfactual_consistency": {
            "n_groups": pair_total,
            "n_groups_all_correct": pair_both_correct,
            "rate": round(_safe_div(pair_both_correct, pair_total), 6),
            "note": "Group-level joint correctness; not independent Bernoulli trials.",
        },
        "relation_breakdown": {
            "left_of_n": len(left),
            "left_of_accuracy": round(_safe_div(sum(s["correct"] for s in left), len(left)), 6),
            "right_of_n": len(right),
            "right_of_accuracy": round(_safe_div(sum(s["correct"] for s in right), len(right)), 6),
        },
        "modality_breakdown": {
            "visual_n": len(visual),
            "visual_accuracy": round(_safe_div(sum(s["correct"] for s in visual), len(visual)), 6),
            "text_only_n": len(text_only),
            "text_only_accuracy": round(_safe_div(sum(s["correct"] for s in text_only), len(text_only)), 6),
        },
        "latency_sec": {
            "n": len(latencies),
            "mean": round(sum(latencies) / len(latencies), 6) if latencies else None,
            "max": round(max(latencies), 6) if latencies else None,
            "min": round(min(latencies), 6) if latencies else None,
        },
        "peak_vram_mib": {
            "n": len(peak_vram),
            "max": round(max(peak_vram), 3) if peak_vram else None,
            "mean": round(sum(peak_vram) / len(peak_vram), 3) if peak_vram else None,
        },
        "label_distribution": {
            "true": dict(Counter(y_true)),
            "pred": dict(Counter(y_pred)),
        },
        "epistemic_notes": [
            "Candidate-normalized scores are uncalibrated ranking scores, not confidence.",
            "Behavioral errors do not establish internal neural mechanisms.",
            "Bootstrap CIs treat scored cases as the resampling unit; matched pairs share entities.",
        ],
    }
