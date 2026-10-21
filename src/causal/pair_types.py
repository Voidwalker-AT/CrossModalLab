"""Strict counterfactual pair classification for causal interventions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def classify_pair(source: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a (source, target) pair for causal eligibility.

    Returns pair_type among:
      EXACT_IMAGE_CHANGED_TEXT | EXACT_TEXT_CHANGED_IMAGE | MULTI_FACTOR | INVALID
    """
    issues: List[str] = []
    if source.get("family_id") != target.get("family_id"):
        issues.append("family_id mismatch")

    img_same = source.get("image_sha256") == target.get("image_sha256")
    img_bytes_same = True  # byte identity implied by hash match when hashes present
    if not source.get("image_sha256") or not target.get("image_sha256"):
        issues.append("missing image_sha256")
        img_bytes_same = False

    text_premise_s = [p for p in source.get("premises", []) if p.get("modality") == "textual"]
    text_premise_t = [p for p in target.get("premises", []) if p.get("modality") == "textual"]
    vis_premise_s = [p for p in source.get("premises", []) if p.get("modality") == "visual"]
    vis_premise_t = [p for p in target.get("premises", []) if p.get("modality") == "visual"]

    text_same = text_premise_s == text_premise_t
    vis_same = vis_premise_s == vis_premise_t
    query_same = source.get("query") == target.get("query")
    entities_same = source.get("entities") == target.get("entities")

    prompt_same = source.get("prompt") == target.get("prompt")

    if img_same and text_same and query_same:
        pair_type = "MULTI_FACTOR" if not (vis_same and entities_same) else "INVALID_IDENTICAL"
        issues.append("source and target appear identical on primary fields")
    elif img_same and not text_same and entities_same and query_same:
        # textual relation change only (premises text differ); visual premises may be same
        if vis_same:
            pair_type = "EXACT_IMAGE_CHANGED_TEXT"
        else:
            pair_type = "MULTI_FACTOR"
            issues.append("image identical but visual premise metadata differs")
    elif text_same and not img_same and entities_same and query_same:
        if not vis_same:
            pair_type = "EXACT_TEXT_CHANGED_IMAGE"
        else:
            pair_type = "MULTI_FACTOR"
            issues.append("text identical but visual premise metadata same while image differs")
    else:
        pair_type = "MULTI_FACTOR"
        if not img_same and not text_same:
            issues.append("both image and text changed")

    # Layout jitter warning when image hash differs but visual entities match
    if not img_same and set(source.get("visual_entities_shown") or []) == set(
        target.get("visual_entities_shown") or []
    ):
        issues.append("same_visual_entities_different_image_hash_jitter_possible")

    return {
        "pair_type": pair_type,
        "eligible_for_primary_causal": pair_type in {"EXACT_IMAGE_CHANGED_TEXT", "EXACT_TEXT_CHANGED_IMAGE"},
        "image_sha256_equal": img_same,
        "visual_premise_equal": vis_same,
        "text_premise_equal": text_same,
        "query_equal": query_same,
        "entities_equal": entities_same,
        "prompt_equal": prompt_same,
        "issues": issues,
        "source_sample_id": source.get("sample_id"),
        "target_sample_id": target.get("sample_id"),
        "source_label": source.get("ground_truth_label"),
        "target_label": target.get("ground_truth_label"),
    }
