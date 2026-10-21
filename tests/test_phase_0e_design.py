"""Tests for Phase 0E instrumentation helpers (CPU-only; no GPU required)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.config.paths import REPO_ROOT
from src.instrumentation.passive_capture import CaptureSpec


class TestCaptureSpec(unittest.TestCase):
    def test_frozen_spec(self):
        spec = CaptureSpec(layer_index=12, token_positions=("last", "mean_image"))
        self.assertEqual(spec.layer_index, 12)
        self.assertEqual(spec.store_dtype, "float32")
        self.assertFalse(spec.include_attention_output)


class TestMatchedCaseSelection(unittest.TestCase):
    def test_selection_file_valid(self):
        path = REPO_ROOT / "configs" / "phase_0e" / "matched_case_selection.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["phase"], "0E")
        man = json.loads(
            (REPO_ROOT / "configs" / "phase_0d2" / "behavioral_manifest_v1.json").read_text(encoding="utf-8")
        )
        ids = {s["sample_id"] for s in man["samples"]}
        for group, values in data["cases"].items():
            if group in {"true_matched_pairs", "not_matched_merely_similar"}:
                continue
            for sid in values:
                self.assertIn(sid, ids, f"{sid} missing from Phase 0D.2 manifest ({group})")
        for pair in data["cases"]["true_matched_pairs"]:
            self.assertIn(pair["forward"], ids)
            self.assertIn(pair["reversal"], ids)


class TestDesignDocPresent(unittest.TestCase):
    def test_design_doc_exists(self):
        path = REPO_ROOT / "docs" / "phase_0e_mechanistic_design.md"
        text = path.read_text(encoding="utf-8")
        self.assertIn("Probe decodability", text)
        self.assertIn("blank-image multimodal", text.lower())
        self.assertIn("CONDITIONAL GO", text)


if __name__ == "__main__":
    unittest.main()
