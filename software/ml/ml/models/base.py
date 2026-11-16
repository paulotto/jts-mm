# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Shared pieces of all classifiers: the fit/predict contract and grouped splits."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GroupShuffleSplit

from ml.config.classes import CLASS_COUNT


def align_proba(proba: np.ndarray, classes: np.ndarray, n_classes: int = CLASS_COUNT) -> np.ndarray:
    """Map estimator class columns onto the global class axis 0..6."""
    out = np.zeros((proba.shape[0], n_classes))
    for col, c in enumerate(np.asarray(classes, dtype=int)):
        if 0 <= c < n_classes:
            out[:, c] = proba[:, col]
    return out


def grouped_val_split(groups: np.ndarray | None, seed: int, frac: float = 0.2
                      ) -> tuple[np.ndarray, np.ndarray] | None:
    """Train/validation rows split by whole groups; None with fewer than 2 groups."""
    if groups is None:
        return None
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if uniq.size < 2:
        return None
    size = min(max(frac, 1.0 / uniq.size), 0.5)
    tr, va = next(GroupShuffleSplit(1, test_size=size, random_state=seed)
                  .split(np.zeros(len(groups)), groups=groups))
    return tr, va


class _Base(ClassifierMixin, BaseEstimator):
    input = "features"

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
