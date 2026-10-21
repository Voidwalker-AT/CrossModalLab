"""CPU-only regularized linear probes (NumPy). No sklearn dependency.

Linear probes here measure linear accessibility only — not causal usage.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.instrumentation.capture_integrity import compute_probe_metrics


@dataclass
class ProbeDataset:
    X: np.ndarray  # [N, D]
    y: List[str]
    group_ids: List[str]
    meta: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.X) != len(self.y) or len(self.y) != len(self.group_ids):
            raise ValueError("X, y, group_ids length mismatch")
        if self.X.ndim != 2:
            raise ValueError("X must be 2-D")


def assert_finite_features(X: np.ndarray) -> None:
    if not np.isfinite(X).all():
        raise ValueError("Non-finite values in feature matrix")


def assert_group_disjoint(a: Sequence[str], b: Sequence[str]) -> None:
    overlap = set(a) & set(b)
    if overlap:
        raise ValueError(f"Group leakage between splits: {sorted(overlap)[:5]}")


@dataclass
class Standardizer:
    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_ = np.where(self.scale_ < 1e-8, 1.0, self.scale_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.scale_ is not None
        return (X - self.mean_) / self.scale_


@dataclass
class LinearProbe:
    """Multinomial L2 logistic regression (softmax + SGD/Adam-free full-batch LBFGS-lite via gradient descent)."""

    classes_: List[str]
    W: np.ndarray  # [C, D]
    b: np.ndarray  # [C]
    learning_rate: float = 0.5
    max_iter: int = 400
    l2: float = 1e-2
    tol: float = 1e-6
    seed: int = 0

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: Sequence[str],
        *,
        classes: Optional[Sequence[str]] = None,
        l2: float = 1e-2,
        max_iter: int = 400,
        learning_rate: float = 0.5,
        seed: int = 0,
        tol: float = 1e-6,
    ) -> "LinearProbe":
        assert_finite_features(X)
        classes = list(classes) if classes is not None else sorted(set(y))
        class_to_i = {c: i for i, c in enumerate(classes)}
        Y = np.zeros((len(y), len(classes)), dtype=np.float64)
        for i, lab in enumerate(y):
            Y[i, class_to_i[lab]] = 1.0

        rng = np.random.default_rng(seed)
        D = X.shape[1]
        C = len(classes)
        W = rng.normal(0, 0.01, size=(C, D))
        b = np.zeros(C, dtype=np.float64)
        n = X.shape[0]
        prev_loss = float("inf")
        for it in range(max_iter):
            logits = X @ W.T + b
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / exp.sum(axis=1, keepdims=True)
            loss = -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1)) + 0.5 * l2 * float(np.sum(W * W))
            grad_logits = (probs - Y) / n
            gW = grad_logits.T @ X + l2 * W
            gb = grad_logits.sum(axis=0)
            W -= learning_rate * gW
            b -= learning_rate * gb
            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss
        return cls(classes_=classes, W=W, b=b, l2=l2, max_iter=max_iter, learning_rate=learning_rate, seed=seed)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert_finite_features(X)
        logits = X @ self.W.T + self.b
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> List[str]:
        probs = self.predict_proba(X)
        idx = probs.argmax(axis=1)
        return [self.classes_[i] for i in idx]


def fit_standardized_probe(
    train: ProbeDataset,
    *,
    l2: float = 1e-2,
    max_iter: int = 400,
    seed: int = 0,
    classes: Optional[Sequence[str]] = None,
) -> Tuple[Standardizer, LinearProbe]:
    assert_finite_features(train.X)
    std = Standardizer().fit(train.X)
    Xs = std.transform(train.X)
    probe = LinearProbe.fit(
        Xs, train.y, classes=classes, l2=l2, max_iter=max_iter, seed=seed
    )
    return std, probe


def evaluate_probe(
    std: Standardizer,
    probe: LinearProbe,
    data: ProbeDataset,
    labels: Sequence[str] = ("Yes", "No", "Undetermined"),
) -> Dict[str, Any]:
    y_pred = probe.predict(std.transform(data.X))
    metrics = compute_probe_metrics(data.y, y_pred, list(labels))
    conf = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(data.y, y_pred):
        conf[t][p] += 1
    metrics["confusion_matrix"] = conf
    return metrics


def condition_lookup_baseline(meta: Sequence[Dict[str, Any]], y_true: Sequence[str]) -> Dict[str, Any]:
    """Baseline that predicts label from condition template alone."""
    from src.probes.confounding import CONDITION_TO_LABEL

    y_pred = [CONDITION_TO_LABEL.get(m.get("condition"), "Yes") for m in meta]
    return compute_probe_metrics(y_true, y_pred, ["Yes", "No", "Undetermined"])


def majority_baseline(y_true: Sequence[str]) -> Dict[str, Any]:
    from collections import Counter

    if not y_true:
        return {"balanced_accuracy": 0.0, "macro_f1": 0.0}
    top = Counter(y_true).most_common(1)[0][0]
    return compute_probe_metrics(y_true, [top] * len(y_true), ["Yes", "No", "Undetermined"])


def random_label_baseline(y_true: Sequence[str], seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)
    labs = sorted(set(y_true))
    y_pred = [rng.choice(labs) for _ in y_true]
    return compute_probe_metrics(y_true, y_pred, ["Yes", "No", "Undetermined"])


def permutation_test_balanced_accuracy(
    X: np.ndarray,
    y: Sequence[str],
    groups: Sequence[str],
    *,
    n_perm: int = 200,
    seed: int = 0,
    l2: float = 1e-2,
    train_frac: float = 0.7,
) -> Dict[str, Any]:
    """Group-aware train/eval split permutation test (engineering diagnostic)."""
    rng = random.Random(seed)
    uniq = sorted(set(groups))
    rng.shuffle(uniq)
    n_train = max(1, int(len(uniq) * train_frac))
    train_g = set(uniq[:n_train])
    test_g = set(uniq[n_train:])
    train_idx = [i for i, g in enumerate(groups) if g in train_g]
    test_idx = [i for i, g in enumerate(groups) if g in test_g]
    if not train_idx or not test_idx:
        return {"error": "empty split"}

    def _score(labels: Sequence[str]) -> float:
        y_tr = [labels[i] for i in train_idx]
        y_te = [labels[i] for i in test_idx]
        Xtr, Xte = X[train_idx], X[test_idx]
        std = Standardizer().fit(Xtr)
        probe = LinearProbe.fit(std.transform(Xtr), y_tr, l2=l2, seed=seed)
        pred = probe.predict(std.transform(Xte))
        return compute_probe_metrics(y_te, pred, sorted(set(labels)))["balanced_accuracy"]

    observed = score_labels = list(y)
    obs = _score(score_labels)
    nulls = []
    y_list = list(y)
    for _ in range(n_perm):
        perm = y_list[:]
        rng.shuffle(perm)
        nulls.append(_score(perm))
    nulls_sorted = sorted(nulls)
    p = (1 + sum(1 for v in nulls if v >= obs)) / (1 + n_perm)
    return {
        "observed_balanced_accuracy": obs,
        "null_mean": sum(nulls) / len(nulls),
        "null_p95": nulls_sorted[int(0.95 * (len(nulls) - 1))],
        "permutation_p_value": p,
        "n_perm": n_perm,
        "seed": seed,
    }


def group_disjoint_split(
    data: ProbeDataset,
    *,
    train_groups: Sequence[str],
    eval_groups: Sequence[str],
) -> Tuple[ProbeDataset, ProbeDataset]:
    assert_group_disjoint(train_groups, eval_groups)
    tr_idx = [i for i, g in enumerate(data.group_ids) if g in set(train_groups)]
    ev_idx = [i for i, g in enumerate(data.group_ids) if g in set(eval_groups)]
    if not tr_idx or not ev_idx:
        raise ValueError("Empty train/eval after group filter")
    meta_tr = [data.meta[i] for i in tr_idx] if data.meta else [{} for _ in tr_idx]
    meta_ev = [data.meta[i] for i in ev_idx] if data.meta else [{} for _ in ev_idx]
    return (
        ProbeDataset(data.X[tr_idx], [data.y[i] for i in tr_idx], [data.group_ids[i] for i in tr_idx], meta_tr),
        ProbeDataset(data.X[ev_idx], [data.y[i] for i in ev_idx], [data.group_ids[i] for i in ev_idx], meta_ev),
    )


def select_l2_grid(
    train: ProbeDataset,
    val: ProbeDataset,
    grid: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0),
    seed: int = 0,
) -> Dict[str, Any]:
    """Choose L2 on development validation only."""
    best = None
    rows = []
    for l2 in grid:
        std, probe = fit_standardized_probe(train, l2=l2, seed=seed)
        m = evaluate_probe(std, probe, val)
        rows.append({"l2": l2, "balanced_accuracy": m["balanced_accuracy"], "macro_f1": m["macro_f1"]})
        if best is None or m["balanced_accuracy"] > best["balanced_accuracy"]:
            best = {"l2": l2, "balanced_accuracy": m["balanced_accuracy"], "macro_f1": m["macro_f1"], "std": std, "probe": probe}
    return {"grid": rows, "best_l2": best["l2"], "best": best}
