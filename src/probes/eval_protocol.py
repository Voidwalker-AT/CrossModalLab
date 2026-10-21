"""Frozen Phase 1F probe evaluation specification and metrics."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.instrumentation.capture_integrity import compute_probe_metrics
from src.probes.linear_probe import (
    ProbeDataset,
    evaluate_probe,
    fit_standardized_probe,
    group_disjoint_split,
    select_l2_grid,
)

SPEC_VERSION = "phase_1f_probe_spec_v1"
LAYERS = (3, 12, 18, 23)
TOKEN_SELECTORS = ("last", "last_text_non_image", "mean_image")
LABELS = ("Yes", "No", "Undetermined")
L2_GRID = (1e-3, 1e-2, 1e-1, 1.0)
SEED = 20261101
BOOTSTRAP_B = 2000


@dataclass(frozen=True)
class FeatureSpec:
    layer: int
    token_selector: str
    use_pca: bool = False
    pca_dim: int = 64

    def key(self) -> str:
        return f"L{self.layer}_{self.token_selector}"


def spec_hash() -> str:
    payload = {
        "version": SPEC_VERSION,
        "layers": list(LAYERS),
        "token_selectors": list(TOKEN_SELECTORS),
        "l2_grid": list(L2_GRID),
        "seed": SEED,
        "labels": list(LABELS),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def extract_token_slice(X: np.ndarray, layer_index: int, selector: str) -> np.ndarray:
    """Feature layout from Phase 1E/1F: for each layer in LAYERS: last, last_text, mean_image."""
    li = LAYERS.index(layer_index)
    si = TOKEN_SELECTORS.index(selector)
    n_blocks = len(LAYERS) * len(TOKEN_SELECTORS)
    if X.shape[1] % n_blocks != 0:
        raise ValueError(f"Feature dim {X.shape[1]} not divisible by {n_blocks} blocks")
    dim = X.shape[1] // n_blocks
    start = (li * len(TOKEN_SELECTORS) + si) * dim
    end = start + dim
    return X[:, start:end]


class PCAFit:
    """Train-only PCA (SVD on centered train matrix)."""

    def __init__(self, n_components: int, seed: int = 0):
        self.n_components = n_components
        self.seed = seed
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PCAFit":
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # economy SVD
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = vt[: self.n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.components_ is not None
        return (X - self.mean_) @ self.components_.T


def group_bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    groups: Sequence[str],
    *,
    n_boot: int = BOOTSTRAP_B,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Bootstrap CI by resampling families (groups), not individual variants."""
    by_g: Dict[str, List[int]] = {}
    for i, g in enumerate(groups):
        by_g.setdefault(g, []).append(i)
    gids = sorted(by_g.keys())
    rng = random.Random(seed)
    bas = []
    for _ in range(n_boot):
        sample_g = [gids[rng.randrange(len(gids))] for _ in range(len(gids))]
        idxs = [i for g in sample_g for i in by_g[g]]
        yt = [y_true[i] for i in idxs]
        yp = [y_pred[i] for i in idxs]
        bas.append(compute_probe_metrics(yt, yp, list(LABELS))["balanced_accuracy"])
    bas.sort()
    point = compute_probe_metrics(y_true, y_pred, list(LABELS))["balanced_accuracy"]
    return {
        "point": point,
        "ci_low": bas[int(0.025 * (n_boot - 1))],
        "ci_high": bas[int(0.975 * (n_boot - 1))],
        "n_boot": n_boot,
        "n_groups": len(gids),
        "seed": seed,
    }


def evaluate_feature_spec(
    train: ProbeDataset,
    val: ProbeDataset,
    spec: FeatureSpec,
    *,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Fit and evaluate one layer×selector probe; hyperparams on val only."""
    Xtr = extract_token_slice(train.X, spec.layer, spec.token_selector)
    Xva = extract_token_slice(val.X, spec.layer, spec.token_selector)
    pca = None
    if spec.use_pca:
        pca = PCAFit(spec.pca_dim, seed=seed).fit(Xtr)
        Xtr = pca.transform(Xtr)
        Xva = pca.transform(Xva)

    tr = ProbeDataset(Xtr, train.y, train.group_ids, train.meta)
    va = ProbeDataset(Xva, val.y, val.group_ids, val.meta)
    grid = select_l2_grid(tr, va, grid=L2_GRID, seed=seed)
    std, probe = fit_standardized_probe(tr, l2=grid["best_l2"], seed=seed)
    metrics = evaluate_probe(std, probe, va)
    ci = group_bootstrap_ci(va.y, probe.predict(std.transform(Xva)), va.group_ids, seed=seed)
    param_count = int(probe.W.size + probe.b.size)
    return {
        "spec": asdict(spec),
        "spec_key": spec.key(),
        "best_l2": grid["best_l2"],
        "l2_grid": grid["grid"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class_recall": metrics["per_class_recall"],
        "confusion_matrix": metrics["confusion_matrix"],
        "group_bootstrap_ci": ci,
        "n_train": len(train.y),
        "n_val": len(va.y),
        "n_train_groups": len(set(train.group_ids)),
        "n_val_groups": len(set(va.group_ids)),
        "feature_dim_raw": int(Xtr.shape[1]),
        "feature_dim_used": int(std.transform(Xtr[:1]).shape[1]),
        "probe_parameter_count": param_count,
        "effective_sample_size_groups": len(set(val.group_ids)),
    }


def frozen_dev_split(
    records_meta: Sequence[Dict[str, Any]],
    *,
    val_fraction: float = 0.25,
    seed: int = SEED,
    exclude_families: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Deterministic family split over provided feature-meta rows."""
    exclude = set(exclude_families or [])
    fams = sorted({r["family_id"] for r in records_meta if r["family_id"] not in exclude})
    rng = random.Random(seed)
    order = fams[:]
    rng.shuffle(order)
    n_val = max(2, int(len(order) * val_fraction))
    val = sorted(order[:n_val])
    train = sorted(order[n_val:])
    return train, val


def load_feature_rows(meta_jsonl: Path, repo_root: Path) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    rows = []
    vecs = []
    for line in meta_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        z = np.load(repo_root / r["vectors_file"])
        vecs.append(z["vector"])
        rows.append(r)
    if not vecs:
        raise ValueError(f"No feature rows in {meta_jsonl}")
    return np.stack(vecs), rows
