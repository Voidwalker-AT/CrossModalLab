"""Specification-aware capture identity and resume integrity for activation captures."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CAPTURE_SCHEMA_VERSION = "phase_1a_capture_v2"


def _stable_hash(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _finite_number(x: Any) -> bool:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    return math.isfinite(float(x))


@dataclass(frozen=True)
class CaptureIdentity:
    """Identity of a single-sample multi-layer capture specification.

    ``identity_hash`` is always recomputed from canonical fields only;
    it is never an input to its own hash.
    """

    sample_id: str
    image_sha256: str
    prompt_sha256: str
    model_id: str
    model_revision: str
    quantization: Dict[str, Any]
    layer_indices: Tuple[int, ...]
    token_selectors: Tuple[str, ...]
    capture_schema_version: str = CAPTURE_SCHEMA_VERSION

    def canonical_fields(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "image_sha256": self.image_sha256,
            "prompt_sha256": self.prompt_sha256,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "quantization": dict(self.quantization),
            "layer_indices": [int(x) for x in self.layer_indices],
            "token_selectors": list(self.token_selectors),
            "capture_schema_version": self.capture_schema_version,
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.canonical_fields()

    def identity_hash(self) -> str:
        return _stable_hash(self.canonical_fields())


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


def identity_from_stored_fields(row: Dict[str, Any]) -> Optional[CaptureIdentity]:
    """Rebuild identity from stored metadata fields only (never trusts stored hash)."""
    if row.get("status") != "ok":
        return None
    if "sample_id" not in row or "model" not in row:
        return None
    model = row.get("model") or {}
    ci = row.get("capture_identity") or {}
    prompt_sha = row.get("prompt_sha256") or ci.get("prompt_sha256")
    if not prompt_sha:
        return None
    layers = row.get("layers") or ci.get("layer_indices")
    selectors = row.get("token_selectors") or ci.get("token_selectors")
    if not layers or not selectors:
        return None
    return CaptureIdentity(
        sample_id=row["sample_id"],
        image_sha256=row.get("image_sha256", ci.get("image_sha256", "")),
        prompt_sha256=prompt_sha,
        model_id=model.get("model_id", ci.get("model_id", "")),
        model_revision=model.get("revision", ci.get("model_revision", "")),
        quantization=model.get("quantization", ci.get("quantization", {})),
        layer_indices=tuple(int(x) for x in layers),
        token_selectors=tuple(selectors),
        capture_schema_version=row.get("capture_schema_version")
        or ci.get("capture_schema_version", CAPTURE_SCHEMA_VERSION),
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
    reject_nonfinite: bool = True,
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
                continue
            if reject_nonfinite:
                for idx, x in enumerate(vec):
                    if not _finite_number(x):
                        issues.append(f"non-finite value at {sel}@L{li}[{idx}]")
                        break
    return issues


def should_skip_capture(
    *,
    existing_rows: Iterable[Dict[str, Any]],
    required_identity: CaptureIdentity,
    vectors_file_for_row=None,
    verify_vectors: bool = True,
) -> ResumeDecision:
    """Skip only if a complete capture matches the required identity.

    Stored ``identity_hash`` values are ignored for trust; identity is
    recomputed from canonical stored fields and compared to the required hash.
    Forged hashes cannot bypass metadata verification.
    """
    required_hash = required_identity.identity_hash()
    issues_all: List[str] = []

    for row in existing_rows:
        if row.get("sample_id") != required_identity.sample_id or row.get("status") != "ok":
            continue

        rebuilt = identity_from_stored_fields(row)
        if rebuilt is None:
            issues_all.append("incomplete stored identity fields")
            continue

        rebuilt_hash = rebuilt.identity_hash()
        stored_hash = (row.get("capture_identity") or {}).get("identity_hash")

        row_issues: List[str] = []
        if rebuilt_hash != required_hash:
            row_issues.append("rebuilt identity hash mismatch vs required")
        if stored_hash is not None and stored_hash != required_hash:
            # Informational: stored hash disagrees with required (even if fields match)
            row_issues.append("stored identity_hash field mismatch (ignored for trust)")
        # Prefer field-based equality; if fields match required, ignore bogus stored hash alone
        field_match = rebuilt.canonical_fields() == required_identity.canonical_fields()
        if not field_match:
            # already captured via rebuilt_hash mismatch; keep detailed field diffs
            if set(rebuilt.layer_indices) != set(required_identity.layer_indices):
                row_issues.append("layer set mismatch")
            if set(rebuilt.token_selectors) != set(required_identity.token_selectors):
                row_issues.append("token selector mismatch")
            if rebuilt.model_revision != required_identity.model_revision:
                row_issues.append("model revision mismatch")
            if rebuilt.image_sha256 != required_identity.image_sha256:
                row_issues.append("image_sha256 mismatch")
            if rebuilt.prompt_sha256 != required_identity.prompt_sha256:
                row_issues.append("prompt_sha256 mismatch")

        # If fields fully match required, forged stored hash alone must not block;
        # but we require rebuilt==required (already checked). Clear noise issues when field_match.
        if field_match:
            row_issues = [i for i in row_issues if "stored identity_hash field mismatch" not in i]
            if verify_vectors and vectors_file_for_row is not None:
                vpath = vectors_file_for_row(row)
                if vpath is not None:
                    row_issues.extend(
                        verify_vector_artifact(
                            vectors_file=vpath,
                            expected_layers=required_identity.layer_indices,
                            expected_selectors=required_identity.token_selectors,
                            reject_nonfinite=True,
                        )
                    )
            if not row_issues:
                return ResumeDecision(
                    skip=True, reason="complete_matching_capture", identity_hash=required_hash
                )
        issues_all.extend(row_issues)

    if not issues_all:
        return ResumeDecision(skip=False, reason="no_ok_capture", identity_hash=required_hash)
    return ResumeDecision(
        skip=False,
        reason="incomplete_or_mismatched_capture",
        identity_hash=required_hash,
        issues=issues_all,
    )


def annotate_identity_dict(identity: CaptureIdentity) -> Dict[str, Any]:
    d = identity.canonical_fields()
    d["identity_hash"] = identity.identity_hash()
    return d


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically via temp file + replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass


def compute_probe_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, Any]:
    """Balanced accuracy, macro-F1, per-class recall (pure Python)."""
    from collections import Counter

    support = Counter(y_true)
    recalls = {}
    precisions = {}
    f1s = {}
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        recalls[lab] = tp / (tp + fn) if (tp + fn) else 0.0
        precisions[lab] = tp / (tp + fp) if (tp + fp) else 0.0
        f1s[lab] = (
            2 * precisions[lab] * recalls[lab] / (precisions[lab] + recalls[lab])
            if (precisions[lab] + recalls[lab])
            else 0.0
        )
    present = [lab for lab in labels if support.get(lab, 0) > 0]
    balanced = sum(recalls[lab] for lab in present) / len(present) if present else 0.0
    macro_f1 = sum(f1s[lab] for lab in present) / len(present) if present else 0.0
    return {
        "n": len(y_true),
        "support": dict(support),
        "per_class_recall": recalls,
        "per_class_precision": precisions,
        "per_class_f1": f1s,
        "balanced_accuracy": round(balanced, 6),
        "macro_f1": round(macro_f1, 6),
    }
