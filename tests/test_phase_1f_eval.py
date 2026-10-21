"""Phase 1F tests: eval protocol, group bootstrap, feature slices."""

from __future__ import annotations

import unittest

import numpy as np

from src.probes.eval_protocol import (
    LAYERS,
    TOKEN_SELECTORS,
    FeatureSpec,
    PCAFit,
    extract_token_slice,
    group_bootstrap_ci,
    spec_hash,
)
from src.probes.linear_probe import ProbeDataset, group_disjoint_split


class TestFeatureSlices(unittest.TestCase):
    def test_slice_layout(self):
        # 4 layers * 3 selectors * 8 dims
        d = 4 * 3 * 8
        X = np.arange(6 * d, dtype=np.float32).reshape(6, d)
        sl = extract_token_slice(X, 12, "last")
        self.assertEqual(sl.shape, (6, 8))
        # layer 12 is index 1 in LAYERS
        self.assertEqual(LAYERS.index(12), 1)
        self.assertEqual(TOKEN_SELECTORS.index("last"), 0)
        start = (1 * 3 + 0) * 8
        self.assertTrue(np.allclose(sl[0], X[0, start : start + 8]))


class TestBootstrap(unittest.TestCase):
    def test_bootstrap_bounds(self):
        y_true = ["Yes"] * 6 + ["No"] * 6 + ["Undetermined"] * 6
        y_pred = y_true[:]
        g = [f"g{i}" for i in range(18)]
        ci = group_bootstrap_ci(y_true, y_pred, g, n_boot=200, seed=1)
        self.assertEqual(ci["point"], 1.0)
        self.assertLessEqual(ci["ci_low"], ci["point"])
        self.assertGreaterEqual(ci["ci_high"], ci["point"])


class TestPCA(unittest.TestCase):
    def test_pca_shapes(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 32))
        pca = PCAFit(n_components=5).fit(X)
        Z = pca.transform(X)
        self.assertEqual(Z.shape, (20, 5))


class TestSpecHashStable(unittest.TestCase):
    def test_hash_stable(self):
        self.assertEqual(spec_hash(), spec_hash())
        self.assertEqual(len(spec_hash()), 64)


class TestGroupSplitSafety(unittest.TestCase):
    def test_no_leak(self):
        X = np.zeros((9, 4))
        y = ["Yes", "No", "Undetermined"] * 3
        g = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
        data = ProbeDataset(X, y, g)
        tr, va = group_disjoint_split(data, train_groups=["a", "b"], eval_groups=["c"])
        self.assertEqual(set(tr.group_ids), {"a", "b"})
        self.assertEqual(set(va.group_ids), {"c"})


if __name__ == "__main__":
    unittest.main()
