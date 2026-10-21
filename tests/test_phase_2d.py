"""Phase 2D tests: format manifest and optional pilot artifacts."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.config.paths import REPO_ROOT


class TestFormatManifest(unittest.TestCase):
    def test_manifest_budget_and_families(self):
        path = REPO_ROOT / "configs/phase_2d/evidence_format_manifest_v1.json"
        man = json.loads(path.read_text(encoding="utf-8"))
        self.assertLessEqual(man["n_cases_per_model"], 24)
        self.assertEqual(man["n_cases_per_model"], len(man["cases"]))
        self.assertFalse(man["final_holdout_touched"])
        self.assertEqual(len(man["families"]), 6)
        # kinds
        from collections import Counter

        kinds = Counter(c["kind"] for c in man["cases"])
        self.assertEqual(kinds["format_sensitivity"], 18)
        self.assertEqual(kinds["text_only_insufficient"], 6)
        # all logical labels undetermined in this pilot
        self.assertTrue(all(c["logical_label"] == "Undetermined" for c in man["cases"]))


class TestPilotIfPresent(unittest.TestCase):
    def test_pilot_complete(self):
        path = REPO_ROOT / "artifacts/evidence/phase_2d/format_pilot_summary_v1.json"
        if not path.is_file():
            self.skipTest("pilot not run")
        s = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(s["per_model"]["smolvlm2"]["n"], 24)
        self.assertEqual(s["per_model"]["qwen3vl2"]["n"], 24)
        self.assertFalse(s["final_holdout_touched"])
        # text-only abstention perfect for both
        self.assertEqual(s["per_model"]["smolvlm2"]["text_only_insufficient"]["abstention_rate"], 1.0)
        self.assertEqual(s["per_model"]["qwen3vl2"]["text_only_insufficient"]["abstention_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
