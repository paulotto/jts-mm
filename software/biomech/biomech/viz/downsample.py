# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Downsampling for plotting long signals: LTTB and min/max decimation."""

from __future__ import annotations

import numpy as np


def lttb(x: np.ndarray, y: np.ndarray, n_out: int) -> np.ndarray:
    """Indices kept by Largest-Triangle-Three-Buckets (Steinarsson 2013).

    Preserves the visual shape of a line with ``n_out`` points; NaN samples are
    kept as line breaks only if they fall on bucket representatives (callers
    should split at gaps first).
    """
    n = len(x)
    if n_out >= n or n_out < 3:
        return np.arange(n)
    idx = [0]
    edges = np.linspace(1, n - 1, n_out - 1).astype(int)
    a = 0
    for i in range(n_out - 2):
        lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
        nlo, nhi = edges[i + 1], edges[i + 2] if i + 2 < len(edges) else n
        nhi = max(nhi, nlo + 1)
        avg_x, avg_y = np.nanmean(x[nlo:nhi]), np.nanmean(y[nlo:nhi])
        area = np.abs((x[a] - avg_x) * (y[lo:hi] - y[a]) - (x[a] - x[lo:hi]) * (avg_y - y[a]))
        k = lo + int(np.nanargmax(area)) if np.isfinite(area).any() else lo
        idx.append(k)
        a = k
    idx.append(n - 1)
    return np.asarray(idx)


def minmax(y: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin (index of min, index of max): envelope-preserving decimation for 4 kHz EMG."""
    n = len(y)
    if n <= 2 * n_bins:
        i = np.arange(n)
        return i, i
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    lo = np.empty(n_bins, int)
    hi = np.empty(n_bins, int)
    for b in range(n_bins):
        seg = y[edges[b]:edges[b + 1]]
        if np.isfinite(seg).any():
            lo[b] = edges[b] + int(np.nanargmin(seg))
            hi[b] = edges[b] + int(np.nanargmax(seg))
        else:
            lo[b] = hi[b] = edges[b]
    return lo, hi


def rdp(points: np.ndarray, tol: float) -> np.ndarray:
    """Indices kept by Ramer-Douglas-Peucker on an (N, D) path (any D), iteratively.

    Keeps every point farther than ``tol`` (data units, e.g. mm) from the
    simplified path; NaN rows are kept as breaks.
    """
    P = np.asarray(points, float)
    n = len(P)
    if n <= 2:
        return np.arange(n)
    ok = np.isfinite(P).all(axis=1)
    keep = np.zeros(n, bool)
    keep[~ok] = True
    # Process each finite run separately.
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        keep[a] = keep[b - 1] = True
        stack = [(a, b - 1)]
        while stack:
            i, j = stack.pop()
            if j <= i + 1:
                continue
            seg = P[j] - P[i]
            L2 = float(seg @ seg)
            Q = P[i + 1:j] - P[i]
            if L2 == 0.0:
                dist = np.linalg.norm(Q, axis=1)
            else:
                t = np.clip(Q @ seg / L2, 0.0, 1.0)
                dist = np.linalg.norm(Q - np.outer(t, seg), axis=1)
            k = int(np.argmax(dist))
            if dist[k] > tol:
                m = i + 1 + k
                keep[m] = True
                stack += [(i, m), (m, j)]
    return np.flatnonzero(keep)


def thin_index(x: np.ndarray, *ys: np.ndarray, budget: int = 5000) -> np.ndarray:
    """Indices kept of a series with ``x`` and value arrays ``ys`` (all one length).

    Per finite run (a run never borrows points from its neighbours): the run's ends, the
    extremes of every value array, and min/max per bin (monotone x: an envelope-preserving
    decimation for long signals) or LTTB (paths). One NaN sample is kept between runs, so
    gaps stay gaps. Series within the budget are returned whole.
    """
    n = len(x)
    if n <= budget:
        return np.arange(n)
    x = np.asarray(x, float)
    vals = [np.asarray(y, float) for y in ys] or [x]
    ok = np.isfinite(x)
    for y in vals:
        ok &= np.isfinite(y)
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
    runs = list(zip(edges[::2], edges[1::2]))
    total = max(int(ok.sum()), 1)
    keep: list[np.ndarray] = []
    gaps = np.flatnonzero(~ok)
    if gaps.size:                                 # the first sample of every gap: a break
        first = gaps[np.r_[True, np.diff(gaps) > 1]]
        keep.append(first)
    for a, b in runs:
        m = b - a
        k = max(4, int(round(budget * m / total)))
        if m <= k:
            keep.append(np.arange(a, b))
            continue
        xr = x[a:b]
        keep.append(np.array([a, b - 1]))
        for y in vals:
            keep.append(a + np.array([int(np.argmin(y[a:b])), int(np.argmax(y[a:b]))]))
        if np.all(np.diff(xr) >= 0):
            lo, hi = minmax(vals[0][a:b], max(k // 2, 2))
            keep += [a + lo, a + hi]
        else:
            keep.append(a + lttb(xr, vals[0][a:b], k))
    return np.unique(np.concatenate(keep)) if keep else np.arange(0)
