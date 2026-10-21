"""Phase 2A CPU tests for intervention interfaces (no VLM)."""

from __future__ import annotations

import unittest

import numpy as np

from src.causal.interventions import (
    InterventionSpec,
    MarginOutcomes,
    apply_residual_replacement,
    config_hash,
    margin_change,
    resolve_source_vector,
    validate_intervention_spec,
)


class TestSpecValidation(unittest.TestCase):
    def test_valid_clean_to_cf(self):
        s = InterventionSpec(18, 5, "clean_to_cf", "counterfactual")
        self.assertEqual(validate_intervention_spec(s), [])

    def test_invalid_direction(self):
        s = InterventionSpec(18, 5, "teleport", "none")
        self.assertTrue(validate_intervention_spec(s))

    def test_no_op_source_kind(self):
        s = InterventionSpec(3, 0, "no_op", "clean")
        self.assertTrue(validate_intervention_spec(s))


class TestReplacement(unittest.TestCase):
    def setUp(self):
        self.T, self.H = 10, 8
        self.target = np.zeros((self.T, self.H), dtype=np.float32)
        self.target[5] = 1.0
        self.source = np.full(self.H, 2.0, dtype=np.float32)

    def test_no_op(self):
        spec = InterventionSpec(18, 5, "no_op", "none")
        out = apply_residual_replacement(self.target, None, token_position=5, spec=spec)
        self.assertTrue(np.allclose(out, self.target))

    def test_replacement_at_position(self):
        spec = InterventionSpec(18, 5, "clean_to_cf", "counterfactual")
        out = apply_residual_replacement(self.target, self.source, token_position=5, spec=spec)
        self.assertTrue(np.allclose(out[5], 2.0))
        self.assertTrue(np.allclose(out[4], 0.0))

    def test_shape_mismatch(self):
        spec = InterventionSpec(18, 5, "clean_to_cf", "counterfactual")
        bad = np.zeros(self.H + 1, dtype=np.float32)
        with self.assertRaises(ValueError):
            apply_residual_replacement(self.target, bad, token_position=5, spec=spec)

    def test_out_of_range(self):
        spec = InterventionSpec(18, 99, "clean_to_cf", "counterfactual")
        with self.assertRaises(IndexError):
            apply_residual_replacement(self.target, self.source, token_position=99, spec=spec)

    def test_nonfinite_source(self):
        spec = InterventionSpec(18, 5, "clean_to_cf", "counterfactual")
        bad = self.source.copy()
        bad[0] = np.nan
        with self.assertRaises(ValueError):
            apply_residual_replacement(self.target, bad, token_position=5, spec=spec)


class TestSourceResolution(unittest.TestCase):
    def test_directions(self):
        clean = np.ones(4, dtype=np.float32)
        cf = np.full(4, 2.0, dtype=np.float32)
        s = InterventionSpec(18, 0, "clean_to_cf", "counterfactual")
        self.assertTrue(np.allclose(resolve_source_vector(s, clean_row=clean, cf_row=cf), cf))
        s2 = InterventionSpec(18, 0, "cf_to_clean", "clean")
        self.assertTrue(np.allclose(resolve_source_vector(s2, clean_row=clean, cf_row=cf), clean))
        s3 = InterventionSpec(18, 0, "no_op", "none")
        self.assertIsNone(resolve_source_vector(s3, clean_row=clean, cf_row=cf))


class TestMargins(unittest.TestCase):
    def test_margin_change_and_nie_guard(self):
        base = MarginOutcomes({" Yes": -1.0, " No": -2.0, " Undetermined": -3.0}, " Yes", " Undetermined")
        patch = MarginOutcomes({" Yes": -0.5, " No": -2.0, " Undetermined": -3.0}, " Yes", " Undetermined")
        d = margin_change(base, patch)
        self.assertAlmostEqual(d["raw_margin_change"], 0.5)
        # tiny baseline margin suppresses NIE
        base2 = MarginOutcomes({" Yes": -1e-9, " No": 0.0, " Undetermined": 0.0}, " Yes", " Undetermined")
        patch2 = MarginOutcomes({" Yes": -0.1, " No": 0.0, " Undetermined": 0.0}, " Yes", " Undetermined")
        d2 = margin_change(base2, patch2)
        self.assertIsNone(d2["normalized_indirect_effect"])

    def test_config_hash_stable(self):
        s1 = InterventionSpec(18, 0, "no_op", "none")
        s2 = InterventionSpec(18, 0, "clean_to_cf", "counterfactual")
        self.assertEqual(config_hash([s1]), config_hash([s1]))
        self.assertNotEqual(config_hash([s1]), config_hash([s2]))


class TestPreregArtifact(unittest.TestCase):
    def test_prereg_exists_and_unauthorized(self):
        from pathlib import Path
        from src.config.paths import REPO_ROOT
        import json

        path = REPO_ROOT / "configs/phase_2a/causal_preregistration_v1.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(data["execution_authorized"])
        self.assertTrue(data["probe_context"]["not_a_proven_logical_representation"])


if __name__ == "__main__":
    unittest.main()
