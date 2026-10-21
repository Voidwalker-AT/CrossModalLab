"""Causal intervention interfaces (CPU synthetic tests only; no VLM patching in 2A)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class InterventionSpec:
    """Specification for a single residual-stream replacement intervention."""

    layer_index: int
    token_position: int  # absolute index in sequence
    direction: str  # clean_to_cf | cf_to_clean | no_op | random_source | same_label | unrelated_token
    source_kind: str  # clean | counterfactual | random | none
    seed: int = 0
    notes: str = ""

    def identity_hash(self) -> str:
        payload = {
            "layer_index": self.layer_index,
            "token_position": self.token_position,
            "direction": self.direction,
            "source_kind": self.source_kind,
            "seed": self.seed,
            "notes": self.notes,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


VALID_DIRECTIONS = {
    "clean_to_cf",
    "cf_to_clean",
    "no_op",
    "random_source",
    "same_label",
    "unrelated_token",
}


def validate_intervention_spec(spec: InterventionSpec) -> List[str]:
    issues: List[str] = []
    if spec.direction not in VALID_DIRECTIONS:
        issues.append(f"invalid direction {spec.direction}")
    if spec.token_position < 0:
        issues.append("token_position must be >= 0")
    if spec.layer_index < 0:
        issues.append("layer_index must be >= 0")
    if spec.direction == "no_op" and spec.source_kind != "none":
        issues.append("no_op must use source_kind=none")
    if spec.direction in {"clean_to_cf"} and spec.source_kind != "counterfactual":
        issues.append("clean_to_cf requires source_kind=counterfactual")
    if spec.direction in {"cf_to_clean"} and spec.source_kind != "clean":
        issues.append("cf_to_clean requires source_kind=clean")
    if spec.direction == "random_source" and spec.source_kind != "random":
        issues.append("random_source requires source_kind=random")
    return issues


def apply_residual_replacement(
    target: np.ndarray,
    source: Optional[np.ndarray],
    *,
    token_position: int,
    spec: InterventionSpec,
) -> np.ndarray:
    """Replace target[token_position] with source vector (or no-op).

    target/source: shape [T, H] float arrays. Returns a new array (copy).
    Rejects shape/dtype/device (as finite) mismatches.
    """
    if target.ndim != 2:
        raise ValueError("target must be [T,H]")
    if token_position < 0 or token_position >= target.shape[0]:
        raise IndexError("token_position out of range")

    out = target.copy()
    if spec.direction == "no_op":
        return out

    if source is None:
        raise ValueError(f"direction {spec.direction} requires source activations")
    if source.ndim != 1 or source.shape[0] != target.shape[1]:
        raise ValueError(
            f"source shape {source.shape} incompatible with hidden dim {target.shape[1]}"
        )
    if not np.isfinite(source).all():
        raise ValueError("source contains non-finite values")
    if not np.isfinite(target[token_position]).all():
        raise ValueError("target token row contains non-finite values")

    if spec.direction == "clean_to_cf":
        if source.shape[0] != target.shape[1]:
            raise ValueError("hidden size mismatch")
    out[token_position, :] = source.astype(out.dtype, copy=False)
    return out


def resolve_source_vector(
    spec: InterventionSpec,
    *,
    clean_row: np.ndarray,
    cf_row: np.ndarray,
    unrelated_row: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Optional[np.ndarray]:
    """Choose the source row for an intervention without touching a VLM."""
    if spec.direction == "no_op":
        return None
    if spec.direction == "clean_to_cf":
        return cf_row
    if spec.direction == "cf_to_clean":
        return clean_row
    if spec.direction == "same_label":
        return clean_row  # placeholder: caller supplies same-label row as clean_row
    if spec.direction == "unrelated_token":
        if unrelated_row is None:
            raise ValueError("unrelated_token requires unrelated_row")
        return unrelated_row
    if spec.direction == "random_source":
        rng = rng or np.random.default_rng(spec.seed)
        return rng.normal(0, 1, size=clean_row.shape[0]).astype(np.float32)
    raise ValueError(spec.direction)


def config_hash(specs: Sequence[InterventionSpec]) -> str:
    blob = "|".join(s.identity_hash() for s in specs)
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class MarginOutcomes:
    """Complete-candidate Yes/No/Undetermined margin outcomes."""

    logprobs: Dict[str, float]
    target: str
    alternative: str

    def raw_margin(self) -> float:
        return self.logprobs[self.target] - self.logprobs[self.alternative]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logprobs": self.logprobs,
            "target": self.target,
            "alternative": self.alternative,
            "raw_margin": self.raw_margin(),
        }


def margin_change(baseline: MarginOutcomes, patched: MarginOutcomes) -> Dict[str, Any]:
    """Primary outcome: change in target-vs-alternative raw log-likelihood margin."""
    if baseline.target != patched.target or baseline.alternative != patched.alternative:
        raise ValueError("target/alternative mismatch between baseline and patched")
    b = baseline.raw_margin()
    p = patched.raw_margin()
    change = p - b
    # Pre-registered stability rule for normalized indirect effect (optional secondary)
    denom = abs(b)
    if denom < 1e-6:
        nie = None
        nie_note = "normalized effect suppressed: |baseline margin| < 1e-6"
    else:
        nie = change / denom
        nie_note = "raw change preferred; NIE secondary only"
    return {
        "baseline_margin": b,
        "patched_margin": p,
        "raw_margin_change": change,
        "normalized_indirect_effect": nie,
        "nie_note": nie_note,
        "baseline_logprobs": baseline.logprobs,
        "patched_logprobs": patched.logprobs,
    }
