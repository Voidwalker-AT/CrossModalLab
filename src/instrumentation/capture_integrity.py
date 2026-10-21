"""Specification-aware capture identity and resume integrity for activation captures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CAPTURE_SCHEMA_VERSION = "phase_1a_capture_v2"


def _stable_hash(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class CaptureIdentity:
    """Identity of a single-sample multi-layer capture specification."""

    sample_id: str
    image_sha256: str
    prompt_sha256: str
    model_id: str
    model_revision: str
    quantization: Dict[str, Any]
    layer_indices: Tuple[int, ...]
    token_selectors: Tuple[str, ...]
    capture_schema_version: str = CAPTURE_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["layer_indices"] = list(self.layer_indices)
        d["token_selectors"] = list(self.token_selectors)
        return d

    def identity_hash(self) -> str:
        return _stable_hash(self.to_dict())


def build_capture_identity(
    *,
    sample_id: str,
    image_sha256: str,
    prompt: str,
    model_id: str,
    model_revision: str,
    quantization: Dict[str, Any],
    layer_indices: Sequence[int],
    token_selectors: Sequence[str],
    capture_schema_version: str = CAPTURE_SCHEMA_VERSION,
) -> CaptureIdentity:
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return CaptureIdentity(
        sample_id=sample_id,
        image_sha256=image_sha256 or "",
        prompt_sha256=prompt_sha,
        model_id=model_id,
        model_revision=model_revision,
        quantization=dict(quantization),
        layer_indices=tuple(int(x) for x in layer_indices),
        token_selectors=tuple(token_selectors),
        capture_schema_version=capture_schema_version,
    )


def identity_from_meta_row(row: Dict[str, Any], prompt: Optional[str] = None) -> Optional[CaptureIdentity]:
    """Reconstruct identity from a stored metadata row when fields are present."""
    if "capture_identity" in row and isinstance(row["capture_identity"], dict):
        c = row["capture_identity"]
        try:
            return CaptureIdentity(
                sample_id=c["sample_id"],
                image_sha256=c.get("image_sha256", ""),
                prompt_sha256=c.get("prompt_sha256", ""),
                model_id=c["model_id"],
                model_revision=c["model_revision"],
                quantization=c["quantization"],
                layer_indices=tuple(c["layer_indices"]),
                token_selectors=tuple(c["token_selectors"]),
                capture_schema_version=c.get("capture_schema_version", CAPTURE_SCHEMA_VERSION),
            )
        except Exception:
            return None

    # Legacy Phase 1A rows: reconstruct from top-level fields + optional prompt
    if row.get("status") != "ok":
        return None
    required = ("sample_id", "image_sha256", "layers", "model")
    if any(k not in row for k in required):
        return None
    model = row["model"]
    quant = model.get("quantization") or {}
    if prompt is None:
        # Without prompt we cannot form a complete identity; mark incomplete via empty prompt hash only if caller supplies
        return None
    return build_capture_identity(
        sample_id=row["sample_id"],
        image_sha256=row.get("image_sha256", ""),
        prompt=prompt,
        model_id=model.get("model_id", ""),
        model_revision=model.get("revision", ""),
        quantization=quant,
        layer_indices=row.get("layers", []),
        token_selectors=row.get("token_selectors") or (
            "last",
            "first_image",
            "last_image",
            "mean_image",
            "last_text_non_image",
        ),
    )


@dataclass
class ResumeDecision:
    skip: bool
    reason: str
    identity_hash: Optional[str] = None
    issues: List[str] = field(default_factory=list)


def verify_vector_artifact(
    *,
    vectors_file: Path,
    expected_layers: Sequence[int],
    expected_selectors: Sequence[str],
) -> List[str]:
    issues: List[str] = []
    if not vectors_file.is_file():
        return [f"missing vector file: {vectors_file}"]
    try:
        data = json.loads(vectors_file.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"unreadable vector file: {exc}"]
    layers = data.get("layers") or {}
    for li in expected_layers:
        if str(li) not in layers:
            issues.append(f"missing layer {li} in vectors")
            continue
        tokens = layers[str(li)].get("tokens") or {}
        for sel in expected_selectors:
            if sel not in tokens:
                issues.append(f"missing token {sel} at layer {li}")
                continue
            vec = tokens[sel].get("vector")
            if not isinstance(vec, list) or len(vec) == 0:
                issues.append(f"empty vector {sel}@L{li}")
            elif any((x is None) or (isinstance(x, float) and (x != x)) for x in vec[:3]):
                issues.append(f"non-finite-like vector head {sel}@L{li}")
    return issues


def should_skip_capture(
    *,
    existing_rows: Iterable[Dict[str, Any]],
    required_identity: CaptureIdentity,
    vectors_file_for_row=None,
    verify_vectors: bool = True,
) -> ResumeDecision:
    """Decide whether a capture can be skipped for resume.

    Skips only if a *complete* capture with matching identity exists and
    optional vector integrity checks pass.
    """
    required_hash = required_identity.identity_hash()
    candidates: List[Dict[str, Any]] = []
    for row in existing_rows:
        if row.get("sample_id") != required_identity.sample_id:
            continue
        if row.get("status") != "ok":
            continue
        candidates.append(row)

    if not candidates:
        return ResumeDecision(skip=False, reason="no_ok_capture")

    issues_all: List[str] = []
    for row in candidates:
        # Prefer stored identity hash
        stored_hash = None
        if isinstance(row.get("capture_identity"), dict):
            stored_hash = _stable_hash(
                {
                    **row["capture_identity"],
                    "layer_indices": list(row["capture_identity"].get("layer_indices", [])),
                    "token_selectors": list(row["capture_identity"].get("token_selectors", [])),
                }
            )
            # identity_hash stored field is authoritative if present
            if row["capture_identity"].get("identity_hash"):
                stored_hash = row["capture_identity"]["identity_hash"]

        layers = tuple(row.get("layers") or [])
        selectors = tuple(row.get("token_selectors") or required_identity.token_selectors)
        model = row.get("model") or {}
        layer_ok = set(layers) == set(required_identity.layer_indices)
        selector_ok = set(selectors) == set(required_identity.token_selectors)
        model_ok = (
            model.get("model_id") == required_identity.model_id
            and model.get("revision") == required_identity.model_revision
        )
        quant_ok = (model.get("quantization") or {}) == required_identity.quantization
        image_ok = (row.get("image_sha256") or "") == required_identity.image_sha256

        row_issues: List[str] = []
        if not layer_ok:
            row_issues.append(f"layer set mismatch: stored={layers} required={required_identity.layer_indices}")
        if not selector_ok:
            row_issues.append("token selector mismatch")
        if not model_ok:
            row_issues.append("model id/revision mismatch")
        if not quant_ok:
            row_issues.append("quantization mismatch")
        if not image_ok:
            row_issues.append("image_sha256 mismatch")
        if stored_hash is not None and stored_hash != required_hash:
            row_issues.append("capture identity hash mismatch")
        if row.get("prompt_sha256") and row.get("prompt_sha256") != required_identity.prompt_sha256:
            row_issues.append("prompt_sha256 mismatch")
        if row.get("capture_schema_version") and row.get("capture_schema_version") != required_identity.capture_schema_version:
            row_issues.append("schema version mismatch")

        if verify_vectors and vectors_file_for_row is not None:
            vpath = vectors_file_for_row(row)
            if vpath is not None:
                row_issues.extend(
                    verify_vector_artifact(
                        vectors_file=vpath,
                        expected_layers=required_identity.layer_indices,
                        expected_selectors=required_identity.token_selectors,
                    )
                )

        if not row_issues:
            return ResumeDecision(skip=True, reason="complete_matching_capture", identity_hash=required_hash)
        issues_all.extend(row_issues)

    return ResumeDecision(
        skip=False,
        reason="incomplete_or_mismatched_capture",
        identity_hash=required_hash,
        issues=issues_all,
    )


def annotate_identity_dict(identity: CaptureIdentity) -> Dict[str, Any]:
    d = identity.to_dict()
    d["identity_hash"] = identity.identity_hash()
    return d
