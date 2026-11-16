# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Master-clock time base: gap-aware segments, derivatives and resampling.

Every stream in a session is stamped on the master clock (``t_master_us``).
This module is the one place that cuts signals at gaps, differentiates them
and resamples them onto another time grid; kinematics, IMU, the data views,
the animator and ``ml`` use it instead of their own copies.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation, Slerp


def odd_window(window_s: float, fs: float) -> int:
    """Sample count of a centred moving window: the odd number nearest ``window_s * fs`` (>= 1).

    A centred moving average over an even number of samples is not symmetric
    (``uniform_filter1d`` / ``np.convolve(mode="same")`` put the extra sample on one
    side), so it shifts the signal by half a sample (E17/E26). An exact even product
    rounds up (300.0 -> 301). ``jtsmm.emg.onset.odd_window`` is the same rule for the
    jtsmm primitives (jtsmm does not import biomech).
    """
    x = float(window_s) * float(fs)
    if not np.isfinite(x) or x < 1.0:
        return 1
    return int(2 * np.floor(x / 2.0) + 1)


def _odd(n: int) -> int:
    """``n`` itself when odd, else ``n + 1`` (at least 1)."""
    n = max(int(n), 1)
    return n if n % 2 else n + 1


def segments(ok: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``(start, stop)`` index ranges of the runs where ``ok`` is True."""
    m = np.concatenate(([False], np.asarray(ok, bool), [False])).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def time_segments(t_us: np.ndarray, ok: np.ndarray | None = None,
                  max_gap_s: float | None = None) -> list[tuple[int, int]]:
    """Segments of valid samples that are also not split by a time gap > ``max_gap_s``."""
    t = np.asarray(t_us, dtype=np.float64)
    good = np.ones(len(t), bool) if ok is None else np.asarray(ok, bool).copy()
    out = []
    for a, b in segments(good):
        if max_gap_s is None or b - a < 2:
            out.append((a, b))
            continue
        cut = np.flatnonzero(np.diff(t[a:b]) > max_gap_s * 1e6) + 1
        edges = [0, *cut.tolist(), b - a]
        out += [(a + e0, a + e1) for e0, e1 in zip(edges[:-1], edges[1:]) if e1 > e0]
    return out


def gradient(t_us: np.ndarray, x: np.ndarray, *, smooth_n: int = 1,
             max_gap_s: float | None = None) -> np.ndarray:
    """d x / d t (per second), computed per finite segment and never across a gap.

    NaN samples stay NaN; no step to zero leaks into neighbouring samples (the
    old ``gradient(nan_to_num(x))`` turned every gap edge into a spike that the
    smoother then spread into valid samples). ``smooth_n`` > 1 applies a centred
    moving mean inside each segment; an even ``smooth_n`` is raised to the next odd
    count (and a segment shorter than the window uses its largest odd length), so the
    smoother never shifts the derivative by half a sample (E17). Callers derive the
    count with :func:`odd_window`.
    """
    t = np.asarray(t_us, dtype=np.float64) * 1e-6
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x) if x.ndim == 1 else np.isfinite(x).all(axis=tuple(range(1, x.ndim)))
    for a, b in time_segments(t * 1e6, ok, max_gap_s):
        if b - a < 2:
            continue
        g = np.gradient(x[a:b], t[a:b], axis=0)
        if smooth_n > 1:
            n = min(_odd(smooth_n), b - a if (b - a) % 2 else b - a - 1)
            if n > 1:
                g = uniform_filter1d(g, n, axis=0, mode="nearest")
        out[a:b] = g
    return out


def resample(t_src_us: np.ndarray, x: np.ndarray, t_dst_us: np.ndarray, *,
             method: Literal["linear", "hold", "nearest"] = "linear",
             max_gap_s: float | None = None) -> np.ndarray:
    """Resample ``x`` (N, ...) from ``t_src`` onto ``t_dst`` (M,).

    ``linear`` interpolates per finite segment, ``hold`` takes the latest sample at
    or before each target time (sample-and-hold, causal), ``nearest`` the closest.
    Targets outside the source span, or inside a gap longer than ``max_gap_s``,
    are NaN.
    """
    ts = np.asarray(t_src_us, dtype=np.float64)
    td = np.asarray(t_dst_us, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(len(x), -1)
    out = np.full((len(td), flat.shape[1]), np.nan)
    if len(ts) == 0:
        return out.reshape((len(td),) + x.shape[1:])
    ok = np.isfinite(flat).all(axis=1)
    if method in ("hold", "nearest"):
        tsv, fv = ts[ok], flat[ok]
        if len(tsv) == 0:
            return out.reshape((len(td),) + x.shape[1:])
        if method == "hold":
            i = np.searchsorted(tsv, td, side="right") - 1
            good = i >= 0
            ref = tsv[np.clip(i, 0, None)]
        else:
            j = np.clip(np.searchsorted(tsv, td), 1, max(len(tsv) - 1, 1))
            left = tsv[j - 1]
            right = tsv[np.minimum(j, len(tsv) - 1)]
            i = np.where(np.abs(td - left) <= np.abs(right - td), j - 1, np.minimum(j, len(tsv) - 1))
            good = np.ones(len(td), bool)
            ref = tsv[i]
        good &= (td >= tsv[0]) & (td <= tsv[-1] if method == "nearest" else True)
        if max_gap_s is not None:
            good &= np.abs(td - ref) <= max_gap_s * 1e6
        out[good] = fv[np.clip(i, 0, len(fv) - 1)[good]]
        return out.reshape((len(td),) + x.shape[1:])
    for a, b in time_segments(ts, ok, max_gap_s):
        if b - a < 2:
            continue
        sel = (td >= ts[a]) & (td <= ts[b - 1])
        if not sel.any():
            continue
        for c in range(flat.shape[1]):
            out[sel, c] = np.interp(td[sel], ts[a:b], flat[a:b, c])
    return out.reshape((len(td),) + x.shape[1:])


def resample_rotations(t_src_us: np.ndarray, R: np.ndarray, t_dst_us: np.ndarray,
                       max_gap_s: float | None = None) -> np.ndarray:
    """Slerp rotation matrices (N, 3, 3) onto ``t_dst``; NaN outside the span or in gaps."""
    ts = np.asarray(t_src_us, dtype=np.float64)
    td = np.asarray(t_dst_us, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    out = np.full((len(td), 3, 3), np.nan)
    ok = np.isfinite(R.reshape(len(R), -1)).all(axis=1)
    for a, b in time_segments(ts, ok, max_gap_s):
        if b - a < 2:
            continue
        sel = (td >= ts[a]) & (td <= ts[b - 1])
        if sel.any():
            out[sel] = Slerp(ts[a:b], Rotation.from_matrix(R[a:b]))(td[sel]).as_matrix()
    return out


def window(t_us: np.ndarray, t0_us: float, t1_us: float) -> slice:
    """Index slice of the samples with ``t0 <= t <= t1`` (``t`` sorted)."""
    t = np.asarray(t_us)
    return slice(int(np.searchsorted(t, t0_us, "left")), int(np.searchsorted(t, t1_us, "right")))
