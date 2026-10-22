"""Phase 2G training utilities: completion-only masking, sampling, metrics (CPU-safe)."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.instrumentation.capture_integrity import compute_probe_metrics


def apply_completion_only_labels(
    input_ids,
    attention_mask,
    *,
    prompt_length: int,
    pad_token_id: Optional[int] = None,
):
    """Mask prompt (and padding) so only completion tokens contribute to CE.

    input_ids: [B, T] long tensor. prompt_length: number of supervised-ineligible
    prefix positions (including BOS/chat template prefix).
    Returns labels [B, T] with -100 on prompt/pad.
    """
    labels = input_ids.clone()
    if prompt_length < 0:
        raise ValueError("prompt_length must be >= 0")
    labels[:, :prompt_length] = -100
    if pad_token_id is not None:
        labels[input_ids == pad_token_id] = -100
    return labels


def labels_not_fully_masked(labels) -> bool:
    return bool((labels != -100).any())


def balanced_family_schedule(
    records: Sequence[Dict[str, Any]],
    *,
    max_steps: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Deterministic family-aware schedule covering Yes/No/Undetermined.

    Only uses records whose split == train by caller filtering.
    """
    by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_class[r["ground_truth_label"]].append(r)
    for k in ("Yes", "No", "Undetermined"):
        if not by_class.get(k):
            raise ValueError(f"class {k} missing from training records")
        by_class[k].sort(key=lambda r: (r["family_id"], r["sample_id"]))

    rng = random.Random(seed)
    order: List[Dict[str, Any]] = []
    pools = {k: list(v) for k, v in by_class.items()}
    keys = ["Yes", "No", "Undetermined"]
    i = 0
    while len(order) < max_steps:
        k = keys[i % 3]
        i += 1
        if not pools[k]:
            rng.shuffle(pools[k])
            pools[k] = list(by_class[k])
        order.append(pools[k].pop(0))
    return order[:max_steps]


def group_disjoint(train_families: Sequence[str], eval_families: Sequence[str]) -> bool:
    return not (set(train_families) & set(eval_families))


def extract_completion_span(
    input_ids_list: List[int],
    *,
    prompt_length: int,
    processor,
) -> str:
    ids = input_ids_list[prompt_length:]
    return processor.tokenizer.decode(ids, skip_special_tokens=True)


def config_hash(obj: Any) -> str:
    blob = json_dumps_stable(obj).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def json_dumps_stable(obj: Any) -> str:
    import json

    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def eval_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, Any]:
    labels = ["Yes", "No", "Undetermined"]
    m = compute_probe_metrics(y_true, y_pred, labels)
    und_idx = [i for i, t in enumerate(y_true) if t == "Undetermined"]
    definite = sum(1 for i in und_idx if y_pred[i] in {"Yes", "No"})
    ans_idx = [i for i, t in enumerate(y_true) if t != "Undetermined"]
    over_abstain = sum(1 for i in ans_idx if y_pred[i] == "Undetermined")
    m["unsupported_definite_rate"] = round(definite / len(und_idx), 6) if und_idx else None
    m["over_abstention_rate"] = round(over_abstain / len(ans_idx), 6) if ans_idx else None
    return m
