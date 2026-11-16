# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Late fusion: one model per feature block, posteriors combined (ml06 §4).

Combiners, in increasing complexity:

* ``mean``      mean of the block posteriors (no parameters);
* ``weighted``  ``w * P_1 + (1 - w) * P_2 ...`` with weights chosen on *grouped
                out-of-fold* posteriors of the training data by macro-F1 (grid over
                the simplex in steps of ``step``, default 0.1; ties go to the first
                grid point, i.e. the most weight on the *last* block);
* ``stack``     multinomial logistic regression on the concatenated grouped
                out-of-fold posteriors, so the meta-classifier never sees in-fold
                predictions of its base models.

Use it through the registry as ``late_mean:lda``, ``late_weighted:svm``,
``late_stack:mlp`` with a multi-block feature set (``emg+imu``). ``block_params``
(one dict per block) gives each block model its own hyper-parameters on top of
the shared ones (the ``late_fusion`` variant, :mod:`ml.protocols.variants`).
"""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from ml.config.classes import CLASS_COUNT
from ml.evaluation.metrics import macro_f1
from ml.models.base import _Base, align_proba

METHODS = ("mean", "weighted", "stack")


def late_fusion_posteriors(probas: list[np.ndarray], *, method: str = "mean",
                           weights: np.ndarray | None = None) -> np.ndarray:
    """Combine (n, K) posteriors with ``mean`` or ``weighted`` (weights sum to 1)."""
    P = np.stack([np.asarray(p, dtype=np.float64) for p in probas])
    if method == "mean":
        return P.mean(axis=0)
    if method == "weighted":
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(probas),) or not np.isclose(w.sum(), 1.0) or np.any(w < 0):
            raise ValueError(f"weights must be {len(probas)} non-negative values summing to 1")
        return np.tensordot(w, P, axes=1)
    raise ValueError(f"method must be mean or weighted here, got {method!r}")


def _simplex(n: int, step: float = 0.1) -> list[np.ndarray]:
    k = int(round(1 / step))
    return [np.array(c) / k for c in itertools.product(range(k + 1), repeat=n) if sum(c) == k]


class LateFusionClassifier(_Base):
    _OWN = ("base", "method", "inner", "seed", "slices", "probability", "step", "block_params")

    def __init__(self, base: str = "lda", method: str = "mean", inner: int = 3, seed: int = 42,
                 slices: list[slice] | None = None, probability: bool = False,
                 step: float = 0.1, block_params: list[dict] | None = None,
                 **base_params) -> None:
        self.base = base
        self.method = method
        self.inner = inner
        self.seed = seed
        self.slices = slices
        self.probability = probability
        self.step = step
        self.block_params = block_params
        self.base_params = base_params

    def get_params(self, deep: bool = True):
        return {**{k: getattr(self, k) for k in self._OWN}, **self.base_params}

    def set_params(self, **params):
        for k in self._OWN:
            if k in params:
                setattr(self, k, params.pop(k))
        self.base_params.update(params)
        return self

    def _make(self, b: int = 0):
        from ml.models.registry import get

        own = (self.block_params or [{}] * (b + 1))[b] or {}
        return get(self.base).make(self.seed, probability=True, **{**self.base_params, **own})

    def _cols(self, X: np.ndarray, s: slice) -> np.ndarray:
        return X[..., s]

    def oof_splits(self, y, groups) -> list[tuple[np.ndarray, np.ndarray]] | None:
        """The grouped splits of the out-of-fold posteriors (None with < 2 groups)."""
        if groups is None or np.unique(groups).size < 2:
            return None
        k = min(int(self.inner), np.unique(groups).size)
        return list(GroupKFold(k).split(np.zeros((len(y), 1)), y, groups))

    def fit_block(self, b: int, X, y, groups):
        """Block ``b``'s model on the rows of ``X`` (one independent piece of :meth:`fit`)."""
        return self._make(b).fit(self._cols(X, self.slices[b]), y, groups=groups)

    def oof_piece(self, b: int, X, y, groups, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
        """Posteriors of block ``b`` on ``te`` from a model fitted on ``tr``."""
        m = self.fit_block(b, X[tr], y[tr], groups[tr])
        return m.predict_proba(self._cols(X[te], self.slices[b]))

    def _oof(self, X, y, groups) -> list[np.ndarray] | None:
        splits = self.oof_splits(y, groups)
        if splits is None:
            return None
        oof = [np.zeros((len(y), CLASS_COUNT)) for _ in self.slices]
        for tr, te in splits:
            for b in range(len(self.slices)):
                oof[b][te] = self.oof_piece(b, X, y, groups, tr, te)
        return oof

    def _check(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"late fusion method must be one of {METHODS}")
        if not self.slices or len(self.slices) < 2:
            raise ValueError("late fusion needs a multi-block feature set such as emg+imu")
        if self.block_params is not None and len(self.block_params) != len(self.slices):
            raise ValueError("block_params needs one dict per block")

    def combine(self, y, oof: list[np.ndarray] | None) -> "LateFusionClassifier":
        """Set the combiner from the out-of-fold posteriors (``weights_`` / ``meta_``)."""
        self._check()
        y = np.asarray(y)
        self.classes_ = np.arange(CLASS_COUNT)
        self.weights_ = np.full(len(self.slices), 1.0 / len(self.slices))
        self.meta_ = None
        if self.method != "mean":
            if oof is None:
                raise ValueError(f"late_{self.method} needs >= 2 groups for out-of-fold posteriors")
            if self.method == "weighted":
                grid = _simplex(len(oof), float(self.step))
                scores = [macro_f1(y, late_fusion_posteriors(oof, method="weighted", weights=w)
                                   .argmax(1)) for w in grid]
                self.weights_ = grid[int(np.argmax(scores))]
                self.weight_scores_ = dict(zip((tuple(w) for w in grid), scores))
            else:
                self.meta_ = LogisticRegression(max_iter=1000, C=1.0).fit(np.hstack(oof), y)
        return self

    def fit(self, X, y, groups=None):
        """Out-of-fold block models (:meth:`oof_piece`), combiner (:meth:`combine`), block
        models on all rows (:meth:`fit_block`): every piece is independent, so a caller may
        run them in parallel and assemble with the same methods (same model)."""
        self._check()
        y = np.asarray(y)
        oof = self._oof(X, y, groups) if self.method != "mean" else None
        self.combine(y, oof)
        self.models_ = [self.fit_block(b, X, y, groups) for b in range(len(self.slices))]
        return self

    def proba_from_blocks(self, P: list[np.ndarray]) -> np.ndarray:
        """Fused posteriors from the block models' posteriors (``predict_proba``)."""
        if self.meta_ is not None:
            return align_proba(self.meta_.predict_proba(np.hstack(P)), self.meta_.classes_)
        return late_fusion_posteriors(P, method="weighted", weights=self.weights_)

    def block_proba(self, b: int, model, X) -> np.ndarray:
        return model.predict_proba(self._cols(X, self.slices[b]))

    def predict_proba(self, X):
        return self.proba_from_blocks([self.block_proba(b, m, X)
                                       for b, m in enumerate(self.models_)])


__all__ = ["METHODS", "LateFusionClassifier", "late_fusion_posteriors"]
