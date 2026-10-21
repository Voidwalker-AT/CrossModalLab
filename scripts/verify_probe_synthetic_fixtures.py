#!/usr/bin/env python3
"""Synthetic fixture verification for the Phase 1D probe pipeline (CPU, no VLM)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.probes.linear_probe import (
    ProbeDataset,
    assert_finite_features,
    assert_group_disjoint,
    evaluate_probe,
    fit_standardized_probe,
    group_disjoint_split,
    majority_baseline,
)


def _mk(n=90, d=16, seed=0):
    """Clearly separable synthetic classes.

    Phase 1D initial BA≈0.722 occurred because (a) only dim0 carried signal with
    moderate noise, (b) eval had few groups, and (c) the optimizer stopped early.
    We restore a strict fixture with stronger separation and more samples, and
    require BA≥0.85 (not a lowered bar).
    """
    rng = np.random.default_rng(seed)
    y = []
    X = []
    g = []
    for i in range(n):
        lab = ["Yes", "No", "Undetermined"][i % 3]
        base = {"Yes": 4.0, "No": -4.0, "Undetermined": 0.0}[lab]
        vec = rng.normal(0, 0.25, size=d)
        vec[0] += base
        vec[1] += 0.5 * base
        X.append(vec)
        y.append(lab)
        g.append(f"fam{i // 3:03d}")
    return ProbeDataset(np.asarray(X), y, g, [{"i": i} for i in range(n)])


def main() -> int:
    failures = []

    # Separable
    data = _mk()
    all_g = sorted(set(data.group_ids))
    tr_groups = all_g[: max(1, int(0.7 * len(all_g)))]
    ev_groups = all_g[len(tr_groups) :]
    try:
        assert_group_disjoint(tr_groups, ev_groups)
        train, ev = group_disjoint_split(data, train_groups=tr_groups, eval_groups=ev_groups)
        std, probe = fit_standardized_probe(train, seed=0, max_iter=800, l2=1e-3)
        m = evaluate_probe(std, probe, ev)
        if m["balanced_accuracy"] < 0.85:
            failures.append(f"separable probe too weak: {m['balanced_accuracy']}")
    except Exception as exc:
        failures.append(f"separable failed: {exc}")

    # Constant features → near chance
    Xc = np.zeros((30, 8))
    yc = (["Yes", "No", "Undetermined"] * 10)
    gc = [f"g{i:02d}" for i in range(30)]
    try:
        data_c = ProbeDataset(Xc, yc, gc)
        train_c, ev_c = group_disjoint_split(
            data_c, train_groups=gc[:20], eval_groups=gc[20:]
        )
        std, probe = fit_standardized_probe(train_c, seed=0)
        m = evaluate_probe(std, probe, ev_c)
        if m["balanced_accuracy"] > 0.7:
            failures.append(f"constant features should be near chance: {m['balanced_accuracy']}")
    except Exception as exc:
        failures.append(f"constant failed: {exc}")

    # Non-finite rejected
    try:
        bad = np.asarray(Xc)
        bad[0, 0] = np.nan
        assert_finite_features(bad)
        failures.append("non-finite not rejected")
    except ValueError:
        pass

    # Group leakage rejected
    try:
        assert_group_disjoint(["a", "b"], ["b", "c"])
        failures.append("leakage not rejected")
    except ValueError:
        pass

    # Imbalanced majority baseline
    y_imb = ["Yes"] * 20 + ["No"] * 5 + ["Undetermined"] * 5
    mb = majority_baseline(y_imb)
    if abs(mb["per_class_recall"]["Yes"] - 1.0) > 1e-9:
        failures.append("majority baseline recall wrong")

    print("SYNTHETIC_FIXTURES", "PASS" if not failures else "FAIL")
    for f in failures:
        print(" ", f)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
