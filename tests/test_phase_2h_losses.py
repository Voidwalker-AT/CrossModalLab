"""Phase 2H tests: evidence losses, gradients, provenance (CPU, synthetic)."""

from __future__ import annotations

import json
import unittest

try:
    import torch
    from src.training.evidence_losses import (
        LABEL_TO_IDX,
        combined_loss,
        counterfactual_consistency,
        encode_labels,
        evidence_sufficiency,
        standard_ce,
        validate_coeffs,
    )
    _TORCH_OK = True
except ImportError:
    torch = None
    combined_loss = counterfactual_consistency = encode_labels = None
    evidence_sufficiency = standard_ce = validate_coeffs = None
    LABEL_TO_IDX = {}
    _TORCH_OK = False

from src.config.paths import REPO_ROOT


@unittest.skipUnless(_TORCH_OK, "torch required for loss tests")
class TestEncoding(unittest.TestCase):
    def test_encode_labels(self):
        t = encode_labels(["Yes", "No", "Undetermined"])
        self.assertEqual(t.tolist(), [0, 1, 2])

    def test_invalid_label(self):
        with self.assertRaises(ValueError):
            encode_labels(["Maybe"])

    def test_empty_batch(self):
        with self.assertRaises(ValueError):
            encode_labels([])


@unittest.skipUnless(_TORCH_OK, "torch required")
class TestCE(unittest.TestCase):
    def test_known_logits_finite(self):
        logits = torch.tensor([[4.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
        targets = torch.tensor([0])
        loss = standard_ce(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        # gradient should decrease wrong class mass / increase correct
        self.assertLess(logits.grad[0, 0].item(), 0.0)
        self.assertGreater(logits.grad[0, 1].item(), 0.0)

    def test_extreme_logits_stable(self):
        logits = torch.tensor([[100.0, -100.0, -100.0]], dtype=torch.float32)
        loss = standard_ce(logits, torch.tensor([0]))
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss), 1e-3)

    def test_nonfinite_rejected(self):
        bad = torch.tensor([[float("nan"), 0.0, 0.0]])
        with self.assertRaises(ValueError):
            standard_ce(bad, torch.tensor([0]))

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            standard_ce(torch.zeros(0, 3), torch.zeros(0, dtype=torch.long))


@unittest.skipUnless(_TORCH_OK, "torch required")
class TestCounterfactual(unittest.TestCase):
    def test_pair_loss_finite_and_grad(self):
        a = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
        b = torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32, requires_grad=True)
        ta = torch.tensor([0])
        tb = torch.tensor([2])
        loss = counterfactual_consistency(a, b, ta, tb)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)

    def test_same_label_ok(self):
        a = torch.randn(2, 3, dtype=torch.float32)
        t = torch.tensor([0, 1])
        loss = counterfactual_consistency(a, a, t, t)
        self.assertTrue(torch.isfinite(loss))

    def test_bad_temperature(self):
        a = torch.randn(1, 3)
        t = torch.tensor([0])
        with self.assertRaises(ValueError):
            counterfactual_consistency(a, a, t, t, temperature=0.0)


@unittest.skipUnless(_TORCH_OK, "torch required")
class TestEvidenceSufficiency(unittest.TestCase):
    def test_margin_prefers_abstain_when_insufficient(self):
        # logits favor Yes strongly but sufficiency=0 → hinge > 0
        logits = torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
        suff = torch.tensor([0])
        loss = evidence_sufficiency(logits, torch.tensor([0]), sufficiency=suff, margin=1.0, mode="margin")
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_margin_zero_when_sufficient_and_definite(self):
        logits = torch.tensor([[5.0, 0.0, -10.0]], dtype=torch.float32)
        suff = torch.tensor([1])
        loss = evidence_sufficiency(logits, torch.tensor([0]), sufficiency=suff, margin=1.0, mode="margin")
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_ce_mode_finite(self):
        logits = torch.randn(4, 3, dtype=torch.float32, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 0])
        suff = torch.tensor([1, 1, 0, 0])
        loss = evidence_sufficiency(logits, targets, sufficiency=suff, mode="ce")
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_negative_margin_rejected(self):
        logits = torch.randn(1, 3)
        with self.assertRaises(ValueError):
            evidence_sufficiency(logits, torch.tensor([0]), sufficiency=torch.tensor([0]), margin=-1.0)


@unittest.skipUnless(_TORCH_OK, "torch required")
class TestCombined(unittest.TestCase):
    def test_zero_coeffs_equal_ce(self):
        logits = torch.randn(4, 3, dtype=torch.float32)
        targets = torch.tensor([0, 1, 2, 0])
        out = combined_loss(logits, targets)
        self.assertTrue(torch.allclose(out["total"], out["ce"]))
        self.assertEqual(float(out["cf"]), 0.0)

    def test_negative_coeff_rejected(self):
        logits = torch.randn(1, 3)
        targets = torch.tensor([0])
        with self.assertRaises(ValueError):
            combined_loss(logits, targets, lambda_cf=-0.1)

    def test_missing_cf_tensors(self):
        logits = torch.randn(1, 3)
        with self.assertRaises(ValueError):
            combined_loss(logits, torch.tensor([0]), lambda_cf=1.0)

    def test_deterministic(self):
        torch.manual_seed(0)
        logits = torch.randn(8, 3, dtype=torch.float32)
        targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        suff = torch.tensor([1, 1, 0, 0, 1, 0, 1, 0])
        a = combined_loss(logits, targets, sufficiency=suff, lambda_cf=0.0, mu_es=0.5)
        b = combined_loss(logits, targets, sufficiency=suff, lambda_cf=0.0, mu_es=0.5)
        self.assertEqual(float(a["total"]), float(b["total"]))


class TestProvenance(unittest.TestCase):
    def test_phase_2g_artifacts_untouched_ids(self):
        r = json.loads(
            (REPO_ROOT / "artifacts/training/phase_2g/phase_2g_report_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(r["total_steps"], 50)
        self.assertFalse(r["holdout_used"])
        self.assertEqual(r["val_step_final"]["metrics"]["balanced_accuracy"], 0.222222)


if __name__ == "__main__":
    unittest.main()
