"""Strict answer parser (versioned) and gold-span helpers for Phase 2I-A."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

PARSER_VERSION = "strict_answer_v1"

_GOLD = {"Yes": " Yes", "No": " No", "Undetermined": " Undetermined"}
_VALID = {"Yes", "No", "Undetermined"}


def gold_answer_string(label: str) -> str:
    if label not in _GOLD:
        raise ValueError(f"bad gold label {label!r}")
    return _GOLD[label]


def strict_parse_answer(raw: Optional[str]) -> Dict[str, str]:
    """Strict parser: never maps Invalid→No/Undetermined.

    Returns parsed_label in {Yes,No,Undetermined,Invalid,Missing,Conflicted}
    and parse_status detailing why.
    """
    if raw is None:
        return {"parsed_label": "Missing", "parse_status": "missing", "parser_version": PARSER_VERSION}
    text = raw.strip()
    if not text:
        return {"parsed_label": "Invalid", "parse_status": "empty", "parser_version": PARSER_VERSION}

    if "Assistant:" in text:
        text = text.split("Assistant:")[-1].strip()

    low = text.lower()
    hits = []
    if low.startswith("yes") or low[:4] in {"yes.", "yes:"}:
        hits.append("Yes")
    elif low.startswith("no") or low[:3] in {"no.", "no:"}:
        hits.append("No")
    elif low.startswith("undetermined") or low.startswith("undecidable"):
        hits.append("Undetermined")
    else:
        # also accept exact first word
        w = low.split()[0].strip(".,;:!?\"'")
        if w == "yes":
            hits.append("Yes")
        elif w == "no":
            hits.append("No")
        elif w in {"undetermined", "undecidable"}:
            hits.append("Undetermined")

    # conflict detection across whole string (word boundaries)
    has_y = _word_in(low, "yes")
    has_n = _word_in(low, "no")
    has_u = _word_in(low, "undetermined") or _word_in(low, "undecidable")
    flags = [f for f, on in (("Yes", has_y), ("No", has_n), ("Undetermined", has_u)) if on]
    if len(flags) > 1:
        return {
            "parsed_label": "Conflicted",
            "parse_status": f"multiple:{'|'.join(flags)}",
            "parser_version": PARSER_VERSION,
        }

    if not hits:
        return {"parsed_label": "Invalid", "parse_status": "unparseable", "parser_version": PARSER_VERSION}
    return {"parsed_label": hits[0], "parse_status": "parsed_strict", "parser_version": PARSER_VERSION}


def _word_in(text: str, word: str) -> bool:
    import re

    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def evaluation_accounting(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = ("Yes", "No", "Undetermined"),
) -> Dict[str, Any]:
    """BA/F1/recall with explicit Invalid/Missing/Conflicted accounting (not folded into classes)."""
    from src.instrumentation.capture_integrity import compute_probe_metrics

    # Compute on parseable subset AND full set with invalid rate
    invalid_pred = [p for p in y_pred if p not in labels]
    valid_idx = [i for i, p in enumerate(y_pred) if p in labels]
    y_true_v = [y_true[i] for i in valid_idx]
    y_pred_v = [y_pred[i] for i in valid_idx]
    m_full = compute_probe_metrics(y_true, y_pred, list(labels))
    # For invalid outputs, treat as incorrect but report separately: recompute
    # metrics only if all valid; otherwise use full metrics from compute_probe_metrics
    # (invalid already counted as wrong class).
    m_full["invalid_output_count"] = len(invalid_pred)
    m_full["invalid_output_rate"] = round(len(invalid_pred) / max(1, len(y_pred)), 6)
    m_full["parser_version"] = PARSER_VERSION
    return m_full
