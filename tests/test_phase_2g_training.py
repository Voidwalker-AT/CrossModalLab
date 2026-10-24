"""Phase 2G CPU tests: masking, schedule, metrics, freeze splits, LoRA targets."""

from __future__ import annotations

import json
import os
import unittest

try:
    import torch
    _TORCH_OK = True
except ImportError:
    torch = None
    _TORCH_OK = False

from src.config.paths import REPO_ROOT
from src.training.eval_freeze import freeze_val_ids
from src.training.qlora_train_utils import (
    apply_completion_only_labels,
    balanced_family_schedule,
    eval_metrics,
    group_disjoint,
    labels_not_fully_masked,
)


@unittest.skipUnless(_TORCH_OK, "torch required for masking tests")
class TestLabelMasking(unittest.TestCase):
    def test_completion_only_masks_prompt(self):
        ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
        labels = apply_completion_only_labels(ids, None, prompt_length=4)
        self.assertEqual(labels[0, :4].tolist(), [-100, -100, -100, -100])
        self.assertTrue((labels[0, 4:] != -100).all())
        self.assertTrue(labels_not_fully_masked(labels))

    def test_fully_masked_rejected(self):
        ids = torch.tensor([[1, 2, 3]])
        labels = apply_completion_only_labels(ids, None, prompt_length=3)
        self.assertFalse(labels_not_fully_masked(labels))

    def test_pad_masking(self):
        ids = torch.tensor([[1, 2, 0, 0]])
        labels = apply_completion_only_labels(ids, None, prompt_length=2, pad_token_id=0)
        self.assertTrue((labels[0, 2:] == -100).all())


class TestSchedule(unittest.TestCase):
    def test_balanced_three_class(self):
        recs = []
        for lab in ("Yes", "No", "Undetermined"):
            for i in range(10):
                recs.append(
                    {
                        "ground_truth_label": lab,
                        "family_id": f"f{i}",
                        "sample_id": f"{lab}-{i}",
                    }
                )
        sched = balanced_family_schedule(recs, max_steps=30, seed=1)
        self.assertEqual(len(sched), 30)
        labs = [r["ground_truth_label"] for r in sched]
        self.assertEqual(labs.count("Yes"), 10)
        self.assertEqual(labs.count("No"), 10)
        self.assertEqual(labs.count("Undetermined"), 10)

    def test_missing_class_raises(self):
        recs = [{"ground_truth_label": "Yes", "family_id": "f", "sample_id": "s"}]
        with self.assertRaises(ValueError):
            balanced_family_schedule(recs, max_steps=3, seed=0)


class TestMetrics(unittest.TestCase):
    def test_over_abstention(self):
        y_t = ["Yes", "No", "Undetermined"]
        y_p = ["Undetermined", "Undetermined", "Undetermined"]
        m = eval_metrics(y_t, y_p)
        self.assertEqual(m["over_abstention_rate"], 1.0)
        self.assertEqual(m["unsupported_definite_rate"], 0.0)


class TestFreezeVal(unittest.TestCase):
    @unittest.skipIf(os.environ.get("CROSSMODAL_EXCLUDE_MIXED_DATASETS") == "1",
                     "Phase 3 isolation: mixed train/validation/test record file kept closed")
    def test_no_holdout(self):
        path = REPO_ROOT / "datasets/cross_modal_contrast_v2/cross_modal_contrast_v2.jsonl"
        if not path.is_file():
            self.skipTest("dataset missing")
        pack = freeze_val_ids(path, n_families=4)
        self.assertEqual(pack["n_val_families"], 4)
        self.assertEqual(pack["n_cases"], 12)  # 4×3
        self.assertTrue(group_disjoint(pack["val_families"], []))
        # all classes present
        labs = {c["ground_truth_label"] for c in pack["cases"]}
        self.assertEqual(labs, {"Yes", "No", "Undetermined"})


class TestTrainConfig(unittest.TestCase):
    def test_frozen_config(self):
        cfg = json.loads((REPO_ROOT / "configs/phase_2g/qlora_train_config_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["training"]["max_steps"], 50)
        self.assertEqual(cfg["lora"]["r"], 2)
        self.assertFalse(cfg["data"]["holdout_allowed"])
        self.assertTrue(cfg["training"]["no_custom_losses"])


if __name__ == "__main__":
    unittest.main()
