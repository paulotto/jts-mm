# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Early fusion: per-block standardization and concatenation (ml06 §4).

Every model pipeline starts with :class:`BlockScaler`. It standardizes each
column on the training data of the fold and, when the input concatenates several
blocks (``emg+imu``), weights block b by ``sqrt(D / (B * d_b))`` so every block
carries the same total variance ``D / B``. Without the weight the 66-column EMG
block would dominate the 25-column IMU block in an RBF kernel's distance just by
having more columns. A single block is left at unit variance (weight 1), so
single-modality models are unchanged. Works on (n, d) and on (n, S, d) sequences
(statistics over windows and sub-frames).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BlockScaler(TransformerMixin, BaseEstimator):
    def __init__(self, slices: list[slice] | None = None) -> None:
        self.slices = slices

    def fit(self, X: np.ndarray, y=None) -> "BlockScaler":
        X = np.asarray(X, dtype=np.float64)
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0)
        sd = flat.std(axis=0)
        self.scale_ = np.where(sd > 1e-12, sd, 1.0)
        d = X.shape[-1]
        sl = self.slices or [slice(0, d)]
        w = np.ones(d)
        if len(sl) > 1:
            for s in sl:
                width = len(range(*s.indices(d)))
                w[s] = np.sqrt(d / (len(sl) * width))
        self.weight_ = w
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_ * self.weight_


def concat_blocks(blocks: list[np.ndarray]) -> tuple[np.ndarray, list[slice]]:
    """Concatenate blocks on the last axis and return the column slice of each."""
    widths = np.cumsum([0] + [b.shape[-1] for b in blocks])
    return (np.concatenate(blocks, axis=-1),
            [slice(int(a), int(b)) for a, b in zip(widths[:-1], widths[1:])])


__all__ = ["BlockScaler", "concat_blocks"]
