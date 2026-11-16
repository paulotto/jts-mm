# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Exact fast ``SVC.predict`` for the RBF kernel (speed only, identical labels).

libsvm predicts one row at a time: one kernel value per support vector in a scalar loop,
then the one-vs-one vote. :func:`predict` computes the same decision values for a batch of
rows with one BLAS kernel matrix (float64), in row chunks, and votes as libsvm does
(``svm_predict_values``): for the class pair (i, j), i < j, in libsvm's pair order,
``dec = sum(coef * K) - rho``; ``dec > 0`` votes for i, otherwise for j; the predicted
class is the *first* class with the most votes (lowest class index on a tie).

The batched decision values differ from libsvm's by rounding only. A row is handed to
``SVC.predict`` (libsvm itself) when any of its decision values lies within a rigorous
rounding bound of zero (at least 1e-9), so its sign could differ, and -- conservatively,
although the tie rule above is libsvm's -- when its vote is tied. The labels are therefore
those of ``SVC.predict``; tests check this on random problems and on the study's fold and
grid combinations.

``ML_SVM_FAST_PREDICT=0`` in the environment (inherited by worker processes) switches the
fast path off; batches below :data:`MIN_ROWS` rows (e.g. the one-window in-loop decoding of
the pseudo-online replay, whose compute time is a result) always use libsvm.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

ENV = "ML_SVM_FAST_PREDICT"
MIN_ROWS = 64                 # smaller batches: libsvm (and the replay's timing) unchanged
CHUNK_ELEMS = 2_000_000       # kernel block per chunk (16 MB float64)
TIE_FALLBACK = True           # tied votes go to libsvm too (its tie rule is reproduced)
_EPS = float(np.finfo(np.float64).eps)


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in ("0", "false", "off", "no")


@contextmanager
def disabled() -> Iterator[None]:
    """Fast path off in this process and in workers started inside the block."""
    old = os.environ.get(ENV)
    os.environ[ENV] = "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = old


def supported(svc: Any) -> bool:
    return (getattr(svc, "kernel", None) == "rbf" and getattr(svc, "_impl", None) == "c_svc"
            and not getattr(svc, "_sparse", True) and not getattr(svc, "break_ties", False)
            and hasattr(svc, "_dual_coef_") and hasattr(svc, "support_vectors_"))


def _pairs(k: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(k) for j in range(i + 1, k)]


def predict(svc: Any, X: np.ndarray, *, stats: dict[str, int] | None = None) -> np.ndarray:
    """``svc.predict(X)`` (fitted RBF ``sklearn.svm.SVC``), batched; see the module doc."""
    if not supported(svc):
        return svc.predict(X)
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != svc.support_vectors_.shape[1] or not np.isfinite(X).all():
        return svc.predict(X)                       # libsvm's own validation / errors
    SV = np.ascontiguousarray(svc.support_vectors_, dtype=np.float64)
    coef = np.asarray(svc._dual_coef_, dtype=np.float64)
    if hasattr(coef, "toarray"):                    # pragma: no cover - dense fits only
        return svc.predict(X)
    b = np.asarray(svc._intercept_, dtype=np.float64)         # = -rho, libsvm pair order
    nsup = np.asarray(svc._n_support, dtype=np.int64)
    gamma = float(svc._gamma)
    k, (n, d), m = nsup.size, X.shape, SV.shape[0]
    pairs = _pairs(k)
    if k < 2 or b.size != len(pairs) or coef.shape != (k - 1, m) or int(nsup.sum()) != m:
        return svc.predict(X)
    start = np.r_[0, np.cumsum(nsup)]
    blk = [slice(int(start[c]), int(start[c + 1])) for c in range(k)]
    ns = np.einsum("ij,ij->i", SV, SV)
    # Rounding bound of a decision value (both computations against the exact one):
    # squared distance |err| <= c u (|x|^2 + |s|^2) with c = 2 (d + 4) (Gram form here,
    # difference form in libsvm), K = exp(-gamma D) then |dK| <= K gamma |dD| + 4 u K
    # <= gamma c u (|x|^2 + max|s|^2) + 4 u (K <= 1), summed with |coef| over the pair's
    # support vectors, plus the summation rounding of both (2 m u sum|coef|) and of rho.
    abs_coef = [float(np.abs(coef[j - 1, blk[i]]).sum() + np.abs(coef[i, blk[j]]).sum())
                for i, j in pairs]
    c_d = 2.0 * (d + 4)
    ns_max = float(ns.max()) if m else 0.0
    out = np.empty(n, dtype=np.int64)
    bad = np.zeros(n, dtype=bool)
    n_amb = n_tie = 0
    step = max(1, CHUNK_ELEMS // max(m, 1))
    for r0 in range(0, n, step):
        Xc = X[r0:r0 + step]
        r = Xc.shape[0]
        nx = np.einsum("ij,ij->i", Xc, Xc)
        K = Xc @ SV.T
        K *= -2.0
        K += nx[:, None]
        K += ns[None, :]
        np.maximum(K, 0.0, out=K)
        K *= -gamma
        np.exp(K, out=K)
        part = [K[:, blk[c]] @ coef[:, blk[c]].T for c in range(k)]      # (r, k - 1)
        k_err = gamma * c_d * _EPS * (nx + ns_max) + 4.0 * _EPS
        votes = np.zeros((r, k), dtype=np.int64)
        amb = np.zeros(r, dtype=bool)
        for p, (i, j) in enumerate(pairs):
            dec = part[i][:, j - 1] + part[j][:, i] + b[p]
            tol = 4.0 * (abs_coef[p] * (k_err + 2.0 * m * _EPS) + abs(b[p]) * _EPS)
            amb |= np.abs(dec) <= np.maximum(tol, 1e-9)
            win_i = dec > 0
            votes[win_i, i] += 1
            votes[~win_i, j] += 1
        top = votes.max(axis=1)
        tie = (votes == top[:, None]).sum(axis=1) > 1
        out[r0:r0 + r] = votes.argmax(axis=1)                 # first max = libsvm's rule
        n_amb += int(amb.sum())
        n_tie += int((tie & ~amb).sum())
        bad[r0:r0 + r] = amb | (tie if TIE_FALLBACK else False)
    y = np.asarray(svc.classes_).take(out)
    if bad.any():
        y[bad] = svc.predict(X[bad])
    if stats is not None:
        stats["rows"] = stats.get("rows", 0) + n
        stats["ambiguous"] = stats.get("ambiguous", 0) + n_amb
        stats["tied"] = stats.get("tied", 0) + n_tie
        stats["fallback"] = stats.get("fallback", 0) + int(bad.sum())
    return y


__all__ = ["ENV", "MIN_ROWS", "disabled", "enabled", "predict", "supported"]
