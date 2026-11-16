# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Threshold-free EMG onset: the variance changepoint of the AGLR family (C2-E, X6 P6).

Staude et al. (2001, IEEE TBME 48:1292) estimate an EMG onset as the change time of a
generalized likelihood ratio (GLR) for a change in the signal's variance (zero-mean Gaussian
model). The detection step of their approximated GLR (a threshold on the GLR) is replaced here
by the window's maximum-likelihood change time, so no threshold enters (the single-cycle
detectors of :mod:`biomech.timing.lead` all threshold against the rest or the window start,
which removes channel-cycles active at the window start, E34):

    LLR(k) = 1/2 [ n log s0^2 - k log s1^2(k) - (n - k) log s2^2(k) ],

s1^2 / s2^2 the mean squares before / from sample k, s0^2 over the window; the onset is the k
maximising LLR among the changes that *increase* the variance, at least ``min_seg_s`` from
both window edges. Masked samples (NaN) are left out of every mean square (counts per
segment). ``ratio_db`` = 10 log10(s2^2 / s1^2) (amplitude dB), the usability gate as
``burst_min_db`` of the threshold detectors.
"""

from __future__ import annotations

import numpy as np


def variance_changepoints(X: np.ndarray, fs: float, min_seg_s: float = 0.05
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise ML variance-increase changepoints of windows ``X`` (R, L) of zero-mean EMG
    (µV; NaN = masked). Returns ``k`` (window sample of the change, -1 = none), ``llr`` and
    ``ratio_db`` (NaN without a change)."""
    X = np.atleast_2d(np.asarray(X, float))
    R, n = X.shape
    v = np.isfinite(X)
    c = np.zeros((R, n + 1))
    np.cumsum(np.where(v, X * X, 0.0), axis=1, out=c[:, 1:])
    cn = np.zeros((R, n + 1))
    np.cumsum(v, axis=1, out=cn[:, 1:])
    m = max(int(round(min_seg_s * fs)), 2)
    k_out = np.full(R, -1, np.int64)
    llr_out = np.full(R, np.nan)
    rdb = np.full(R, np.nan)
    if n < 2 * m + 1:
        return k_out, llr_out, rdb
    ks = np.arange(m, n - m + 1)
    n1, n2 = cn[:, ks], cn[:, [n]] - cn[:, ks]
    with np.errstate(divide="ignore", invalid="ignore"):
        s1 = c[:, ks] / n1
        s2 = (c[:, [n]] - c[:, ks]) / n2
        s0 = c[:, [n]] / cn[:, [n]]
        llr = 0.5 * (cn[:, [n]] * np.log(s0) - n1 * np.log(s1) - n2 * np.log(s2))
    ok = (n1 >= m // 2) & (n2 >= m // 2) & (s2 > s1) & np.isfinite(llr) & (s1 > 0)
    llr = np.where(ok, llr, -np.inf)
    j = np.argmax(llr, axis=1)
    has = np.isfinite(llr[np.arange(R), j])
    k_out[has] = ks[j[has]]
    llr_out[has] = llr[np.arange(R), j][has]
    with np.errstate(divide="ignore", invalid="ignore"):
        rdb[has] = 10 * np.log10(s2[np.arange(R), j][has] / s1[np.arange(R), j][has])
    return k_out, llr_out, rdb
