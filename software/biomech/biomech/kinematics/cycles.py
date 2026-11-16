# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Movement cycles from a 1-D excursion signal (an02 §3).

A cycle is start (rest/closed) -> peak (maximal excursion) -> end (rest). Peaks
need a prominence of ``prominence_frac`` of the signal's 5-95 % range and are
at least ``min_period_s`` apart; start/end are the minima between neighbouring
peaks (or the phase edges). Cycles are kept but flagged (``ok`` False, with
the reason in ``flags``) when their window contains more than ``max_nan_frac``
untracked samples (``untracked``), when they are shorter than ``min_period_s``
(``short``), when start or end is the first/last sample of the phase, i.e. the
phase edge cut the movement (``edge``), and, if ``return_frac`` is given, when
start and end differ by more than that fraction of the amplitude (``no_return``;
only meaningful for tasks that start and end closed, not for alternating
excursions such as left/right, whose cycles run from one side to the other).

The evaluation scripts (experiments/5_4_2 and 5_4_3) cut peak-to-peak instead
(:func:`biomech.kinematics.peak_cycles.peak_to_peak_cycles`); the two give
different windows -- never pool them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class Cycle:
    start: int
    peak: int
    end: int
    nan_frac: float
    flags: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.flags


def detect(x: np.ndarray, fs: float, *, min_period_s: float = 0.4,
           prominence_frac: float = 0.3, min_amplitude: float = 0.0,
           max_nan_frac: float = 0.1, return_frac: float | None = None) -> list[Cycle]:
    y = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(y)
    if finite.sum() < 8:
        return []
    yf = y.copy()
    idx = np.arange(y.size)
    yf[~finite] = np.interp(idx[~finite], idx[finite], y[finite])
    lo, hi = np.percentile(yf[finite], [5, 95])
    rng = max(hi - lo, 1e-9)
    peaks, _ = find_peaks(yf, prominence=prominence_frac * rng,
                          distance=max(int(min_period_s * fs), 1))
    out: list[Cycle] = []
    for i, pk in enumerate(peaks):
        a = peaks[i - 1] if i > 0 else 0
        b = peaks[i + 1] if i + 1 < len(peaks) else y.size - 1
        s = a + int(np.argmin(yf[a:pk + 1]))
        e = pk + int(np.argmin(yf[pk:b + 1]))
        if e <= pk or pk <= s:
            continue
        amp = yf[pk] - max(yf[s], yf[e])
        if amp < max(min_amplitude, prominence_frac * rng * 0.5):
            continue
        nan_frac = float(1.0 - finite[s:e + 1].mean())
        flags = []
        if nan_frac > max_nan_frac:
            flags.append("untracked")
        if (e - s) / fs < min_period_s:
            flags.append("short")
        if s == 0 or e == y.size - 1:
            flags.append("edge")
        full = yf[pk] - min(yf[s], yf[e])
        if return_frac is not None and abs(yf[s] - yf[e]) > return_frac * full:
            flags.append("no_return")
        out.append(Cycle(s, int(pk), e, nan_frac, tuple(flags)))
    return out


def time_normalize(y: np.ndarray, n: int = 101) -> np.ndarray:
    """Linear resampling of one cycle (any trailing dims) to ``n`` points; NaN-tolerant."""
    a = np.asarray(y, dtype=np.float64)
    if a.shape[0] < 2:
        return np.full((n,) + a.shape[1:], np.nan)
    x = np.linspace(0.0, 1.0, a.shape[0])
    xi = np.linspace(0.0, 1.0, n)
    flat = a.reshape(a.shape[0], -1)
    out = np.full((n, flat.shape[1]), np.nan)
    for j in range(flat.shape[1]):
        ok = np.isfinite(flat[:, j])
        if ok.sum() >= 2:
            out[:, j] = np.interp(xi, x[ok], flat[ok, j], left=np.nan, right=np.nan)
    return out.reshape((n,) + a.shape[1:])
