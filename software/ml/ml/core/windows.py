# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Trailing windows, gap rejection and window labels.

A window ending at sample ``e`` covers samples ``e - win + 1 .. e`` of one phase
(windows never cross a phase edge because they are generated per phase). Its
timestamp is the end sample's ``t_master_us``; a window whose first and last
sample are more than ``gap_factor`` x W apart spans a recording gap and is
rejected, so a window always means the same physical duration.

Labels (``-1`` = ignore):

* ``end`` (default): the label in force at the window end, i.e. the last label
  sample at or before ``t_end`` (no older than ``max_age_us``). This is what an
  online decoder is asked at that instant.
* ``majority``: the label in force for the longest time inside the window,
  rejected when its share is below ``purity_min`` or when any part of the window
  is ``-1`` / unlabelled (legacy rule; labels hold from one label sample to the
  next).

``purity`` (the share of the window's duration whose label agrees with the
assigned label) is returned in both modes as a diagnostic.
"""

from __future__ import annotations

import numpy as np

from ml.config.classes import CLASS_COUNT, IGNORE

LABEL_MODES = ("end", "majority")


def window_ends(n: int, win: int, stride: int) -> np.ndarray:
    """End sample indices (inclusive) of all complete windows in ``n`` samples."""
    if win < 1 or stride < 1:
        raise ValueError("window and stride must be >= 1 sample")
    return np.arange(win - 1, n, stride, dtype=np.int64)


def gap_windows(t_us: np.ndarray, ends: np.ndarray, win: int, window_us: float,
                gap_factor: float = 1.1) -> np.ndarray:
    """True for windows whose span exceeds ``gap_factor`` x the nominal duration."""
    t = np.asarray(t_us, dtype=np.int64)
    return (t[ends] - t[ends - win + 1]) > gap_factor * window_us


def window_labels(t_end: np.ndarray, label_t: np.ndarray, label_y: np.ndarray,
                  window_us: float, *, mode: str = "end", purity_min: float = 0.6,
                  max_age_us: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(y, purity)`` for windows ending at ``t_end``; ``y = -1`` means ignore."""
    if mode not in LABEL_MODES:
        raise ValueError(f"label_at must be one of {LABEL_MODES}, got {mode!r}")
    te = np.asarray(t_end, dtype=np.int64)
    lt = np.asarray(label_t, dtype=np.int64)
    ly = np.asarray(label_y, dtype=np.int64)
    y = np.full(te.size, IGNORE, dtype=np.int64)
    purity = np.full(te.size, np.nan)
    if lt.size == 0 or te.size == 0:
        return y, purity
    # Time each label is in force inside (t_end - W, t_end]: label i holds on
    # [t_i, t_i+1) (the last one for one median period); prefix sums per class.
    period = float(np.median(np.diff(lt))) if lt.size > 1 else window_us
    seg_end = np.append(lt[1:], lt[-1] + period).astype(np.float64)
    cls = np.where((ly >= 0) & (ly < CLASS_COUNT), ly, CLASS_COUNT)   # last column = ignore
    dur = np.zeros((lt.size + 1, CLASS_COUNT + 1))
    dur[np.arange(1, lt.size + 1), cls] = seg_end - lt
    cum = np.cumsum(dur, axis=0)

    def held(T: np.ndarray) -> np.ndarray:            # (n, 8) time per class before T
        k = np.searchsorted(lt, T, side="right") - 1
        kk = np.maximum(k, 0)
        out = cum[kk].copy()                           # whole segments before segment k
        part = np.clip(T - lt[kk], 0, seg_end[kk] - lt[kk]) * (k >= 0)
        out[np.arange(T.size), cls[kk]] += part
        return out

    counts = held(te.astype(np.float64)) - held(te - float(window_us))
    n_in = counts.sum(axis=1)
    n_in = np.where(n_in > 1e-9 * window_us, n_in, 0.0)

    if mode == "end":
        li = np.searchsorted(lt, te, side="right") - 1
        ok = li >= 0
        if max_age_us is not None:
            ok &= (te - lt[np.maximum(li, 0)]) <= max_age_us
        y = np.where(ok, ly[np.maximum(li, 0)], IGNORE)
        y = np.where((y >= 0) & (y < CLASS_COUNT), y, IGNORE)
    else:
        best = counts[:, :CLASS_COUNT].argmax(axis=1)
        top = counts[np.arange(te.size), best]
        with np.errstate(invalid="ignore", divide="ignore"):
            share = top / n_in
        covered = n_in >= (1 - 1e-6) * window_us          # no time before the first label
        keep = covered & (counts[:, CLASS_COUNT] <= 1e-9 * window_us) & (share >= purity_min)
        y = np.where(keep, best, IGNORE)
    lab = np.clip(y, 0, CLASS_COUNT - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        purity = np.where(n_in > 0, counts[np.arange(te.size), lab] / n_in, np.nan)
    purity[y < 0] = np.nan
    return y.astype(np.int64), purity


def label_runs(y: np.ndarray, *keys: np.ndarray) -> np.ndarray:
    """Run id (from 0) of consecutive rows with equal label and equal ``keys``."""
    y = np.asarray(y)
    if y.size == 0:
        return np.zeros(0, dtype=np.int64)
    change = np.concatenate(([True], y[1:] != y[:-1]))
    for k in keys:
        k = np.asarray(k)
        change[1:] |= k[1:] != k[:-1]
    return np.cumsum(change).astype(np.int64) - 1
