"""Phase 2C-A tests: adapter specs, cleanup manifest, 2.2B presence."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.config.paths import REPO_ROOT
from src.models.model_adapter import QWEN3_VL_2B, SMOLVLM2_2_2B, behavioral_task_battery, describe_model_specs


class TestModelSpecs(unittest.TestCase):
    def test_specs_distinct(self):
        self.assertNotEqual(SMOLVLM2_2_2B.model_id, QWEN3_VL_2B.model_id)
        self.assertEqual(SMOLVLM2_2_2B.hidden_size, 2048)
        self.assertEqual(QWEN3_VL_2B.hidden_size, 2048)
        self.assertEqual(QWEN3_VL_2B.text_num_layers, 28)
        self.assertFalse(QWEN3_VL_2B.requires_trust_remote_code)

    def test_battery_has_required_tasks(self):
        ids = {t["task_id"] for t in behavioral_task_battery()}
        for t in [
            "object_identity",
            "left_right_relation",
            "image_swap",
            "text_premise_reversal",
            "entailed_chain",
            "contradicted_chain",
            "underdetermined_fork",
            "missing_visual_evidence",
            "missing_textual_evidence",
        ]:
            self.assertIn(t, ids)


class TestCleanupManifest(unittest.TestCase):
    def test_manifest_and_2_2b(self):
        man = json.loads(
            (REPO_ROOT / "artifacts/cache/phase_2c_a/cleanup_manifest_v1.json").read_text(encoding="utf-8")
        )
        self.assertTrue(man["smolvlm2_intact"])
        self.assertTrue(man["smolvlm2_snapshot_ok"])
        self.assertGreater(man["eligible_bytes"], 900 * 1024 * 1024)
        # 500M path should not exist
        self.assertFalse(Path(man["deleted_path"]).exists())
        # 2.2B path exists
        snap = Path("E:/cache/huggingface/models--HuggingFaceTB--SmolVLM2-2.2B-Instruct/snapshots/482adb537c021c86670beed01cd58990d01e72e4")
        self.assertTrue(snap.exists())


if __name__ == "__main__":
    unittest.main()
