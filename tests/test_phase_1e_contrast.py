"""Phase 1E tests: contrast v2 schema, labels, splits, shortcuts, fixture."""
from __future__ import annotations

import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from src.benchmark.contrast_v2 import (
    CONDITION_ID,
    LABELS,
    VARIANTS,
    generate_contrast_dataset,
    label_for_variant,
    plan_families,
    variant_premises,
)
from src.config.paths import REPO_ROOT
from src.symbolic.relations import DirectionalRelation, Query
from src.symbolic.solver import SymbolicSolver


class TestVariantLogic(unittest.TestCase):
    def test_three_labels_same_condition(self):
        self.assertEqual(CONDITION_ID, "CROSS_MODAL_TWO_PREMISE")
        self.assertEqual(len(VARIANTS), 3)
        labels = [label_for_variant(v) for v in VARIANTS]
        self.assertEqual(set(labels), set(LABELS))

    def test_solver_agreement(self):
        a, b, c = "cyan_box", "green_cylinder", "magenta_prism"
        q = Query(a, c, DirectionalRelation.LEFT_OF)
        for var, expected in [("entailed", "Yes"), ("contradicted", "No"), ("underdetermined", "Undetermined")]:
            premises, _ = variant_premises(var, a, b, c)
            status = SymbolicSolver.solve(premises, q).epistemic_status.value
            mapping = {"entailed": "Yes", "contradicted": "No", "underdetermined": "Undetermined"}
            self.assertEqual(mapping[status], expected, var)


class TestDatasetGeneration(unittest.TestCase):
    def test_small_generation(self):
        with tempfile_dir() as tmp:
            meta = generate_contrast_dataset(n_families=6, seed=5, out_dir=tmp, render_images=True)
            self.assertEqual(meta["n_records"], 18)
            self.assertEqual(meta["label_counts"]["Yes"], 6)
            self.assertEqual(meta["label_counts"]["No"], 6)
            self.assertEqual(meta["label_counts"]["Undetermined"], 6)
            self.assertEqual(meta["third_entity_leak_count"], 0)
            self.assertTrue(meta["counterfactual_complete_families"])
            self.assertTrue(meta["family_split_ok"])
            self.assertEqual(meta["condition_only_balanced_accuracy"], 1 / 3)
            # single condition id
            self.assertEqual(meta["condition_counts"], {CONDITION_ID: 18})


def tempfile_dir():
    import tempfile

    class _Ctx:
        def __enter__(self):
            self._t = tempfile.TemporaryDirectory()
            return Path(self._t.name)

        def __exit__(self, *a):
            self._t.cleanup()
            return False

    return _Ctx()


class TestBuiltDatasetIfPresent(unittest.TestCase):
    def test_full_meta_if_present(self):
        path = REPO_ROOT / "datasets/cross_modal_contrast_v2/cross_modal_contrast_meta_v2.json"
        if not path.is_file():
            self.skipTest("contrast v2 not built")
        meta = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(meta["n_families"], 240)
        self.assertEqual(meta["n_records"], 720)
        self.assertEqual(meta["condition_only_balanced_accuracy"], 1 / 3)
        self.assertFalse(meta.get("condition_label_correlated", True))


class TestSyntheticFixtureDesign(unittest.TestCase):
    def test_fixture_threshold_documented(self):
        # Ensure we did not leave a 0.7 threshold in the fixture script
        text = (REPO_ROOT / "scripts/verify_probe_synthetic_fixtures.py").read_text(encoding="utf-8")
        self.assertIn("0.85", text)
        self.assertNotIn("< 0.7:", text)


if __name__ == "__main__":
    unittest.main()
