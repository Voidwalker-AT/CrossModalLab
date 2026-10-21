"""Phase 1D unit tests: confounding audit, linear probes, synthetic rules."""

from __future__ import annotations

import json
import unittest

import numpy as np

from src.config.paths import REPO_ROOT
from src.probes.confounding import CONDITION_TO_LABEL, analyze_confounding
from src.probes.linear_probe import (
    ProbeDataset,
    assert_finite_features,
    assert_group_disjoint,
    evaluate_probe,
    fit_standardized_probe,
    group_disjoint_split,
)


class TestConfounding(unittest.TestCase):
    def test_balanced_dev_is_condition_confounded(self):
        path = REPO_ROOT / "configs/phase_1c/balanced_development_v1.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        report = analyze_confounding(records)
        self.assertEqual(report["verdict"], "CONDITION-CONFOUNDED")
        self.assertGreaterEqual(report["condition_label_lookup_accuracy"], 0.999)
        self.assertTrue(report["condition_label_bijective"])


class TestLinearProbe(unittest.TestCase):
    def _data(self, n=48, d=8, seed=1):
        rng = np.random.default_rng(seed)
        X, y, g = [], [], []
        for i in range(n):
            lab = ["Yes", "No", "Undetermined"][i % 3]
            vec = rng.normal(0, 0.2, size=d)
            vec[0] += {"Yes": 3.0, "No": -3.0, "Undetermined": 0.0}[lab]
            X.append(vec)
            y.append(lab)
            g.append(f"f{i // 3:03d}")
        return ProbeDataset(np.asarray(X), y, g)

    def test_group_split_and_fit(self):
        data = self._data()
        tr = [f"f{i:03d}" for i in range(12)]
        ev = [f"f{i:03d}" for i in range(12, 16)]
        assert_group_disjoint(tr, ev)
        train, val = group_disjoint_split(data, train_groups=tr, eval_groups=ev)
        std, probe = fit_standardized_probe(train, seed=0)
        m = evaluate_probe(std, probe, val)
        self.assertGreater(m["balanced_accuracy"], 0.5)
        self.assertIn("confusion_matrix", m)

    def test_nonfinite_rejected(self):
        X = np.zeros((3, 4))
        X[0, 0] = np.inf
        with self.assertRaises(ValueError):
            assert_finite_features(X)

    def test_leakage_rejected(self):
        with self.assertRaises(ValueError):
            assert_group_disjoint(["a"], ["a"])

    def test_mapping_bijective(self):
        self.assertEqual(len(set(CONDITION_TO_LABEL.values())), 3)


class TestHoldoutUntouched(unittest.TestCase):
    def test_probe_summary_if_present_excludes_holdout(self):
        path = REPO_ROOT / "artifacts/probes/phase_1d/probe_pilot_summary_v1.json"
        if not path.is_file():
            self.skipTest("pilot not run yet")
        s = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(s.get("final_holdout_touched", True))
        man = json.loads((REPO_ROOT / "configs/phase_1c/frozen_split_manifest_v1.json").read_text(encoding="utf-8"))
        hold = set(man["final_holdout_families"])
        for f in s.get("train_families", []) + s.get("val_families", []):
            self.assertNotIn(f, hold)


if __name__ == "__main__":
    unittest.main()
