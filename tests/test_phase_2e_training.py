"""Phase 2E tests: losses, schema, split leakage (CPU)."""

from __future__ import annotations

import json
import unittest

try:
    import torch
    from src.training.losses import (
        combined_loss,
        counterfactual_consistency_loss,
        cross_entropy_loss,
        evidence_aware_loss,
    )
    _TORCH_OK = True
except ImportError:
    torch = None
    combined_loss = counterfactual_consistency_loss = cross_entropy_loss = evidence_aware_loss = None
    _TORCH_OK = False

from src.config.paths import REPO_ROOT
from src.training.schema import (
    check_label_balance,
    check_split_leakage,
    holdout_disjoint,
    validate_record_schema,
)


@unittest.skipUnless(_TORCH_OK, "torch required for loss tests")
class TestLosses(unittest.TestCase):
    def test_ce_stable(self):
        logits = torch.tensor([[2.0, 1.0, 0.1]], dtype=torch.float32)
        targets = torch.tensor([0])
        loss = cross_entropy_loss(logits, targets)
        self.assertTrue(torch.isfinite(loss))

    def test_ea_margin_insufficient(self):
        # insufficient, abstain index 2 should get margin push
        logits = torch.tensor([[3.0, 2.0, 0.0]], dtype=torch.float32)
        targets = torch.tensor([0])
        suff = torch.tensor([0])
        loss = evidence_aware_loss(logits, targets, suff, abstain_index=2, margin=1.0, mode="margin")
        self.assertGreater(float(loss), 0.0)

    def test_ea_ce_nonfinite_rejected(self):
        bad = torch.tensor([[float("nan"), 0.0, 0.0]])
        with self.assertRaises(ValueError):
            cross_entropy_loss(bad, torch.tensor([0]))

    def test_cf_pair(self):
        a = torch.tensor([[5.0, 0.0, 0.0]])
        b = torch.tensor([[0.0, 5.0, 0.0]])
        loss = counterfactual_consistency_loss(a, b, torch.tensor([0]), torch.tensor([1]))
        self.assertTrue(torch.isfinite(loss))

    def test_combined(self):
        logits = torch.tensor([[1.0, 0.0, 0.0]])
        out = combined_loss(
            logits,
            torch.tensor([0]),
            torch.tensor([1]),
            abstain_index=2,
            lambda_cf=0.0,
            lambda_ea=0.5,
        )
        self.assertTrue(torch.isfinite(out["total"]))


class TestSchema(unittest.TestCase):
    def test_valid_record(self):
        rec = {
            "sample_id": "a",
            "family_id": "f",
            "split": "train",
            "ground_truth_label": "Yes",
            "prompt": "p",
            "image_sha256": "x",
            "symbolic_status": "entailed",
        }
        self.assertEqual(validate_record_schema(rec), [])

    def test_split_leakage_detected(self):
        recs = [
            {"family_id": "f1", "split": "train"},
            {"family_id": "f1", "split": "test"},
        ]
        out = check_split_leakage(recs)
        self.assertFalse(out["ok"])

    def test_balance(self):
        recs = [
            {"ground_truth_label": "Yes"},
            {"ground_truth_label": "No"},
            {"ground_truth_label": "Undetermined"},
        ]
        self.assertTrue(check_label_balance(recs)["balanced"])


class TestContrastV2AuditIfPresent(unittest.TestCase):
    def test_repo_dataset_passes(self):
        path = REPO_ROOT / "datasets/cross_modal_contrast_v2/cross_modal_contrast_v2.jsonl"
        if not path.is_file():
            self.skipTest("no dataset")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertTrue(check_split_leakage(records)["ok"])
        self.assertTrue(holdout_disjoint(
            {r["family_id"] for r in records if r["split"] != "test"},
            {r["family_id"] for r in records if r["split"] == "test"},
        ))


if __name__ == "__main__":
    unittest.main()
