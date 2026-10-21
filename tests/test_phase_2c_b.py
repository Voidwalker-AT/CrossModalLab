"""Phase 2C-B tests: Qwen specs, acquisition manifest, smoke summary."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.config.paths import REPO_ROOT
from src.models.model_adapter import QWEN3_VL_2B
from src.models.qwen3vl2_loader import QWEN_IMAGE_TOKEN_ID, QWEN_REVISION


class TestQwenSpec(unittest.TestCase):
    def test_pinned_revision_format(self):
        self.assertEqual(len(QWEN_REVISION), 40)
        self.assertTrue(all(c in "0123456789abcdef" for c in QWEN_REVISION))
        self.assertEqual(QWEN_IMAGE_TOKEN_ID, 151655)
        self.assertEqual(QWEN3_VL_2B.architecture, "qwen3_vl")


class TestAcquisitionManifest(unittest.TestCase):
    def test_manifest_complete(self):
        path = REPO_ROOT / "datasets/metadata/qwen3vl2_acquisition_manifest.json"
        self.assertTrue(path.is_file())
        man = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(man["status"], "COMPLETED")
        self.assertEqual(man["revision"], QWEN_REVISION)
        self.assertGreater(man["total_bytes"], 4_000_000_000)
        self.assertEqual(man["missing_required"], [])
        # snapshot exists
        snap = Path(man["snapshot_path"])
        self.assertTrue(snap.is_dir())
        self.assertTrue((snap / "model.safetensors").is_file())


class TestSmokeArtifactIfPresent(unittest.TestCase):
    def test_smoke_summary(self):
        path = REPO_ROOT / "artifacts/qwen/phase_2c_b/qwen_smoke_summary_v1.json"
        if not path.is_file():
            self.skipTest("smoke not run")
        s = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(s["revision"], QWEN_REVISION)
        self.assertFalse(s["causal_patching"])
        self.assertFalse(s["final_holdout_touched"])
        self.assertEqual(len(s["visual_smoke"]), 4)
        self.assertEqual(len(s["cross_modal_smoke"]), 3)
        self.assertFalse(s["hook_test"].get("skipped", True))
        self.assertEqual(s["hook_test"]["logits_max_abs_diff"], 0.0)


class TestSmolvlmUntouched(unittest.TestCase):
    def test_22b_snapshot_exists(self):
        p = Path(
            "E:/cache/huggingface/models--HuggingFaceTB--SmolVLM2-2.2B-Instruct/snapshots/482adb537c021c86670beed01cd58990d01e72e4"
        )
        self.assertTrue(p.exists())
        self.assertTrue((p / "model-00001-of-00002.safetensors").is_file())


if __name__ == "__main__":
    unittest.main()
