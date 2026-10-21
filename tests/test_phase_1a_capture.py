"""Phase 1A unit tests (CPU-safe)."""

from __future__ import annotations

import json
import unittest

from src.config.paths import REPO_ROOT
from src.instrumentation.multi_layer_capture import (
    TOKEN_SELECTORS,
    MultiLayerCaptureSpec,
    classify_input_condition,
    cosine_similarity,
    l2_norm,
)


class TestLayerAndTokenSpec(unittest.TestCase):
    def test_default_selectors(self):
        self.assertEqual(
            set(TOKEN_SELECTORS),
            {"last", "first_image", "last_image", "mean_image", "last_text_non_image"},
        )

    def test_spec_layers(self):
        spec = MultiLayerCaptureSpec(layer_indices=(3, 12, 18, 23))
        self.assertEqual(spec.layer_indices, (3, 12, 18, 23))


class TestConditionTaxonomy(unittest.TestCase):
    def test_true_text_only(self):
        self.assertEqual(classify_input_condition(0, False), "TRUE_TEXT_ONLY")

    def test_image_multimodal(self):
        self.assertEqual(classify_input_condition(1377, True), "IMAGE_MULTIMODAL")

    def test_blank_image_still_has_tokens(self):
        # Blank canvas still produces image tokens + pixels — NOT true text-only
        self.assertEqual(classify_input_condition(1377, True), "IMAGE_MULTIMODAL")


class TestCaseSelection(unittest.TestCase):
    def test_phase_1a_selection_valid(self):
        path = REPO_ROOT / "configs/phase_1a/activation_case_selection.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        man = json.loads((REPO_ROOT / "configs/phase_0d2/behavioral_manifest_v1.json").read_text(encoding="utf-8"))
        ids = {s["sample_id"] for s in man["samples"]}
        self.assertGreaterEqual(len(data["cases"]), 8)
        self.assertLessEqual(len(data["cases"]), 12)
        for case in data["cases"]:
            self.assertIn(case["sample_id"], ids)
        # blank image control must be labeled as such
        blank = next(c for c in data["cases"] if c["role"] == "blank_image_multimodal_control")
        self.assertEqual(blank["analytical_label"], "BLANK_IMAGE_MULTIMODAL")
        taxonomy = data["control_taxonomy"]
        self.assertIn("BLANK_IMAGE_MULTIMODAL", taxonomy)
        self.assertIn("TRUE_TEXT_ONLY", taxonomy)


class TestMathHelpers(unittest.TestCase):
    def test_cosine_identical(self):
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=6)

    def test_l2(self):
        self.assertAlmostEqual(l2_norm([3.0, 4.0]), 5.0, places=6)


class TestAnalysisArtifactIfPresent(unittest.TestCase):
    def test_optional_analysis_schema(self):
        path = REPO_ROOT / "artifacts/instrumentation/phase_1a/descriptive_analysis_v1.json"
        if not path.is_file():
            self.skipTest("analysis not generated yet")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["label"], "DESCRIPTIVE / NON-CAUSAL")


if __name__ == "__main__":
    unittest.main()
