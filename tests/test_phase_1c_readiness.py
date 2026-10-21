"""Phase 1C tests: splits, contamination, metrics, capture integrity hardening."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from src.config.paths import REPO_ROOT
from src.instrumentation.capture_integrity import (
    atomic_write_json,
    build_capture_identity,
    compute_probe_metrics,
    identity_from_stored_fields,
    should_skip_capture,
    verify_vector_artifact,
)


class TestCaptureIntegrityHardening(unittest.TestCase):
    def _ident(self, **kw):
        base = dict(
            sample_id="s1",
            image_sha256="img",
            prompt="p",
            model_id="m",
            model_revision="r1",
            quantization={"load_in_4bit": True},
            layer_indices=(3, 12),
            token_selectors=("last",),
        )
        base.update(kw)
        return build_capture_identity(**base)

    def test_hash_excludes_self_field(self):
        ident = self._ident()
        fields = ident.canonical_fields()
        self.assertNotIn("identity_hash", fields)
        self.assertEqual(ident.identity_hash(), build_capture_identity(
            sample_id="s1", image_sha256="img", prompt="p", model_id="m", model_revision="r1",
            quantization={"load_in_4bit": True}, layer_indices=(3, 12), token_selectors=("last",)
        ).identity_hash())

    def test_forged_stored_hash_cannot_bypass(self):
        ident = self._ident()
        # Row with WRONG layers but forged correct-looking hash
        row = {
            "sample_id": "s1",
            "status": "ok",
            "layers": [3],  # incomplete
            "token_selectors": ["last"],
            "image_sha256": "img",
            "prompt_sha256": ident.prompt_sha256,
            "model": {"model_id": "m", "revision": "r1", "quantization": {"load_in_4bit": True}},
            "capture_schema_version": ident.capture_schema_version,
            "capture_identity": {**ident.canonical_fields(), "identity_hash": ident.identity_hash()},
        }
        d = should_skip_capture(existing_rows=[row], required_identity=ident, verify_vectors=False)
        self.assertFalse(d.skip)

    def test_prompt_change_invalidates(self):
        ident = self._ident()
        other = self._ident(prompt="different")
        row = {
            "sample_id": "s1",
            "status": "ok",
            "layers": [3, 12],
            "token_selectors": ["last"],
            "image_sha256": "img",
            "prompt_sha256": other.prompt_sha256,
            "model": {"model_id": "m", "revision": "r1", "quantization": {"load_in_4bit": True}},
            "capture_schema_version": ident.capture_schema_version,
        }
        d = should_skip_capture(existing_rows=[row], required_identity=ident, verify_vectors=False)
        self.assertFalse(d.skip)

    def test_image_change_invalidates(self):
        ident = self._ident()
        other = self._ident(image_sha256="other")
        row = {
            "sample_id": "s1",
            "status": "ok",
            "layers": [3, 12],
            "token_selectors": ["last"],
            "image_sha256": other.image_sha256,
            "prompt_sha256": ident.prompt_sha256,
            "model": {"model_id": "m", "revision": "r1", "quantization": {"load_in_4bit": True}},
            "capture_schema_version": ident.capture_schema_version,
        }
        d = should_skip_capture(existing_rows=[row], required_identity=ident, verify_vectors=False)
        self.assertFalse(d.skip)

    def test_nonfinite_vector_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            vp = Path(tmp) / "v.json"
            payload = {
                "layers": {
                    "3": {
                        "tokens": {
                            "last": {"vector": [1.0, float("nan"), 3.0]},
                        }
                    }
                }
            }
            vp.write_text(json.dumps(payload), encoding="utf-8")
            issues = verify_vector_artifact(
                vectors_file=vp, expected_layers=(3,), expected_selectors=("last",)
            )
            self.assertTrue(any("non-finite" in i for i in issues))

    def test_atomic_write_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            atomic_write_json(path, {"a": 1})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 1})
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_identity_from_stored_ignores_hash_field(self):
        ident = self._ident()
        row = {
            "sample_id": "s1",
            "status": "ok",
            "layers": [3, 12],
            "token_selectors": ["last"],
            "image_sha256": "img",
            "prompt_sha256": ident.prompt_sha256,
            "model": {"model_id": "m", "revision": "r1", "quantization": {"load_in_4bit": True}},
            "capture_schema_version": ident.capture_schema_version,
            "capture_identity": {"identity_hash": "deadbeef", **ident.canonical_fields()},
        }
        rebuilt = identity_from_stored_fields(row)
        self.assertIsNotNone(rebuilt)
        self.assertEqual(rebuilt.identity_hash(), ident.identity_hash())


class TestProbeMetrics(unittest.TestCase):
    def test_balanced_accuracy_and_macro_f1(self):
        y_true = ["Yes", "No", "Undetermined", "Yes"]
        y_pred = ["Yes", "No", "Yes", "No"]
        m = compute_probe_metrics(y_true, y_pred, ["Yes", "No", "Undetermined"])
        self.assertEqual(m["n"], 4)
        self.assertIn("balanced_accuracy", m)
        self.assertIn("macro_f1", m)
        self.assertAlmostEqual(m["per_class_recall"]["Yes"], 0.5)


class TestFrozenSplitsArtifact(unittest.TestCase):
    def test_manifest_exists_and_disjoint(self):
        path = REPO_ROOT / "configs/phase_1c/frozen_split_manifest_v1.json"
        if not path.is_file():
            self.skipTest("frozen splits not built yet")
        man = json.loads(path.read_text(encoding="utf-8"))
        dev = set(man["development_families"])
        hold = set(man["final_holdout_families"])
        exposed = set(man["development_exposed_families"])
        self.assertFalse(dev & hold)
        self.assertFalse(dev & exposed)
        self.assertFalse(hold & exposed)
        self.assertEqual(man["ground_truth_altered"], False)
        self.assertIn("CONTAMINATION", man["contamination_verdict"])
        self.assertGreaterEqual(man["original_test_contaminated_count"], 1)
        labels = man["balanced_sampling_protocol"]["holdout_label_counts"]
        self.assertEqual(labels.get("Yes"), man["n_final_holdout_families"])
        self.assertEqual(labels.get("No"), man["n_final_holdout_families"])
        self.assertEqual(labels.get("Undetermined"), man["n_final_holdout_families"])


if __name__ == "__main__":
    unittest.main()
