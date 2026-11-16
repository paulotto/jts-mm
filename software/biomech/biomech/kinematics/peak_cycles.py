# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Peak-to-peak repetitions on a 1-D amplitude trace (evaluations experiments/5_4_2 and 5_4_3).

This is the cycle definition of those evaluations, kept for comparability
with its figures: one cycle runs from one excursion **peak to the next peak**.
It is *not* :func:`biomech.kinematics.cycles.detect` (an02 §3), which cuts
rest -> peak -> rest at the minima around each peak and is what the ICORR
pipeline uses. The two give different windows on the same trace; do not pool
their results.

Algorithm (unchanged from the original ``experiments/_lib`` helper):

1. ``scipy.signal.find_peaks`` with distance ``min_period_s`` and prominence
   ``prominence_frac`` x std(y);
2. a gap between neighbouring peaks longer than ``max(2.2 x median period,
   2 x min_period_s)`` (or 2.5 s with < 3 peaks) gets a second search at
   ``fill_frac`` x std, keeping the candidate nearest the gap's middle, so a
   quieter repetition is not swallowed into its neighbour;
3. consecutive peaks at least ``min_period_s`` apart form a cycle.

NaN handling (new): untracked samples are bridged by linear interpolation for
the peak search only; a peak that falls on a bridged sample is discarded, and
the std uses finite samples. The original helper passed NaN straight into
``find_peaks``, whose result is then undefined around the gap.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def _bridge(y: np.ndarray) -> np.ndarray:
    finite = np.isfinite(y)
    if finite.all():
        return y
    idx = np.arange(y.size)
    out = y.copy()
    out[~finite] = np.interp(idx[~finite], idx[finite], y[finite])
    return out


def peak_to_peak_cycles(
    y: np.ndarray,
    fs_hz: float,
    min_period_s: float = 0.4,
    *,
    t_s: np.ndarray | None = None,
    prominence_frac: float = 0.25,
    fill_frac: float = 0.12,
) -> list[tuple[int, int]]:
    """Index pairs ``(peak_k, peak_k+1)`` into ``y``; ``t_s`` (s) when sampling is uneven."""
    y = np.asarray(y, dtype=np.float64).ravel()
    finite = np.isfinite(y)
    if int(finite.sum()) < 8 or fs_hz <= 0:
        return []
    if t_s is None:
        tt = np.arange(y.size, dtype=np.float64) / float(fs_hz)
    else:
        tt = np.asarray(t_s, dtype=np.float64).ravel()
        if tt.size != y.size:
            tt = np.arange(y.size, dtype=np.float64) / float(fs_hz)
    yb = _bridge(y)
    distance = max(int(min_period_s * fs_hz), 2)
    y_std = float(np.std(y[finite]))
    prominence = max(y_std * float(prominence_frac), 1e-6)
    peaks, _ = find_peaks(yb, distance=distance, prominence=prominence)
    peaks = [int(p) for p in peaks if finite[p]]
    if len(peaks) >= 3:
        med = float(np.median(np.diff(tt[peaks])))
        max_gap = max(2.2 * med, 2.0 * min_period_s)
    else:
        max_gap = max(2.5, 2.0 * min_period_s)
    extra: list[int] = []
    fill_prom = max(y_std * float(fill_frac), 1e-6)
    for a, b in zip(peaks[:-1], peaks[1:]):
        if tt[b] - tt[a] <= max_gap:
            continue
        sl = slice(a + 1, b)
        if int(finite[sl].sum()) < 8:
            continue
        loc, props = find_peaks(yb[sl], distance=distance, prominence=fill_prom)
        if loc.size == 0:
            continue
        prominences = np.asarray(props.get("prominences", []), dtype=float)
        mid = 0.5 * (tt[a] + tt[b])
        cand: list[tuple[float, float, int]] = []
        for j, rel in enumerate(loc):
            k = int(a + 1 + rel)
            if not finite[k]:
                continue
            if tt[k] - tt[a] < min_period_s or tt[b] - tt[k] < min_period_s:
                continue
            pr = float(prominences[j]) if j < prominences.size else 0.0
            cand.append((abs(tt[k] - mid), -pr, k))
        if cand:
            extra.append(min(cand)[2])
    if extra:
        peaks = sorted(set(peaks).union(extra))
    if len(peaks) < 2:
        return []
    return [(int(a), int(b)) for a, b in zip(peaks[:-1], peaks[1:]) if b - a >= distance]


def nan_fraction(y: np.ndarray, a: int, b: int) -> float:
    """Share of non-finite samples in ``y[a:b+1]`` (to flag a cycle, not to drop it)."""
    seg = np.asarray(y, dtype=np.float64)[a:b + 1]
    return float(1.0 - np.isfinite(seg).mean()) if seg.size else 1.0


__all__ = ["nan_fraction", "peak_to_peak_cycles"]
