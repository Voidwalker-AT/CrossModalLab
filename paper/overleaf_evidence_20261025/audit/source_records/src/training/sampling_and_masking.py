"""Family-maximizing balanced schedule and gold-span label builders (CPU + processor)."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.training.answer_parser import gold_answer_string


def maximize_family_coverage_schedule(
    records: Sequence[Dict[str, Any]],
    *,
    max_steps: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Balanced Yes/No/Undetermined schedule maximizing unique families.

    - Randomizes family order under fixed seed.
    - Round-robins classes, preferring families that still have unused samples
      for the requested class, then any unused family, then reshuffles.
    - Avoids repeating the same sample_id before exhausting unique samples
      per class (finite pool).
    """
    by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_class[r["ground_truth_label"]].append(r)
    for k in ("Yes", "No", "Undetermined"):
        if not by_class.get(k):
            raise ValueError(f"class {k} missing")
        by_class[k].sort(key=lambda r: r["sample_id"])

    rng = random.Random(seed)
    families = sorted({r["family_id"] for r in records})
    rng.shuffle(families)

    pools = {k: list(v) for k, v in by_class.items()}
    for k in pools:
        rng.shuffle(pools[k])
    used_sample_ids: set = set()
    used_families: set = set()
    schedule: List[Dict[str, Any]] = []
    keys = ["Yes", "No", "Undetermined"]
    guard = 0
    while len(schedule) < max_steps and guard < max_steps * 80:
        guard += 1
        lab = keys[len(schedule) % 3]
        chosen = None
        # Pass 1: unused sample from a family never yet used
        for fid in families:
            if fid in used_families:
                continue
            for i, cand in enumerate(pools[lab]):
                if cand["family_id"] == fid and cand["sample_id"] not in used_sample_ids:
                    chosen = pools[lab].pop(i)
                    break
            if chosen:
                break
        # Pass 2: unused sample from any family (prefer fewer prior uses)
        if chosen is None:
            candidates = [
                (families.index(cand["family_id"]) if cand["family_id"] in families else 10**9, i, cand)
                for i, cand in enumerate(pools[lab])
                if cand["sample_id"] not in used_sample_ids
            ]
            # prefer families with lowest prior exposure
            prior = {fid: 0 for fid in families}
            for s in schedule:
                prior[s["family_id"]] = prior.get(s["family_id"], 0) + 1
            if candidates:
                candidates.sort(key=lambda t: (prior.get(t[2]["family_id"], 0), t[0], t[1]))
                chosen = candidates[0][2]
                pools[lab] = [c for c in pools[lab] if c["sample_id"] != chosen["sample_id"]]
        if chosen is None:
            # reshuffle class pool: clear used for this class
            for cand in by_class[lab]:
                used_sample_ids.discard(cand["sample_id"])
            continue
        used_sample_ids.add(chosen["sample_id"])
        used_families.add(chosen["family_id"])
        schedule.append(chosen)

    if len(schedule) < max_steps:
        raise RuntimeError(f"could not build schedule length {max_steps} got {len(schedule)}")
    return schedule


def answer_span_prompt_length(input_ids, answer_token_ids: List[int]) -> int:
    """Return prompt_length = start index of gold answer tokens in sequence.

    Requires answer_token_ids to appear as a contiguous span in input_ids.
    """
    if input_ids.ndim == 1:
        ids = input_ids.tolist()
    else:
        ids = input_ids[0].tolist()
    if not answer_token_ids:
        raise ValueError("empty answer ids")
    L = len(answer_token_ids)
    start = None
    for i in range(len(ids) - L, -1, -1):
        if ids[i : i + L] == list(answer_token_ids):
            start = i
            break
    if start is None:
        raise ValueError("gold answer token span not found in input_ids")
    return start


def build_supervised_example(
    processor,
    rec: Dict[str, Any],
    *,
    abstention_prompt: bool,
    device: str = "cuda:0",
) -> Tuple[Dict[str, Any], "Any", Dict[str, Any]]:
    """Build inputs + labels supervising only the gold answer span.

    Returns (inputs, labels, meta).
    """
    import torch
    from PIL import Image

    from src.training.qlora_train_utils import apply_completion_only_labels  # local import cycle-safe

    answer = gold_answer_string(rec["ground_truth_label"])
    prompt = rec["prompt"]
    if abstention_prompt:
        extra = (
            "\nYou may answer Undetermined when evidence is insufficient."
            "\nAnswer directly with: Yes, No, or Undetermined."
        )
        if "You may answer Undetermined" not in prompt:
            prompt = prompt.rstrip() + extra

    image = Image.open(REPO_ROOT / rec["image_file"]).convert("RGB")
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
        {"role": "assistant", "content": answer},
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    ans_ids = processor.tokenizer.encode(answer, add_special_tokens=False)
    plen = answer_span_prompt_length(inputs["input_ids"], ans_ids)
    # Supervise EXACTLY the gold answer span; everything else -100
    labels = torch.full_like(inputs["input_ids"], -100)
    labels[0, plen : plen + len(ans_ids)] = inputs["input_ids"][0, plen : plen + len(ans_ids)]
    n_sup = int((labels != -100).sum())
    if n_sup != len(ans_ids):
        raise RuntimeError(f"expected {len(ans_ids)} supervised tokens, got {n_sup}")
    meta = {
        "prompt_length": plen,
        "n_supervised": n_sup,
        "n_answer_tokens": len(ans_ids),
        "answer": answer,
        "answer_token_ids": ans_ids,
        "masking": "gold_answer_span_exact",
    }
    return inputs, labels, meta


# late import path helpers
from src.config.paths import REPO_ROOT  # noqa: E402
