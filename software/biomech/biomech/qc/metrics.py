# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Pure data-quality metrics (an01 §3). Arrays in, numbers out."""

from __future__ import annotations

import numpy as np

from jtsmm.emg.mains import mains_share


def stream_rate_hz(t_us: np.ndarray) -> float:
    """Median sample rate from strictly increasing stamps (duplicates ignored)."""
    t = np.unique(np.asarray(t_us, dtype=np.int64))
    if t.size < 2:
        return float("nan")
    return float(1e6 / np.median(np.diff(t)))


def nonmonotonic_count(t_us: np.ndarray) -> int:
    """Stamps that are not strictly greater than their predecessor, in write order."""
    t = np.asarray(t_us, dtype=np.int64)
    return int(np.count_nonzero(np.diff(t) <= 0)) if t.size > 1 else 0


def coverage(t_us: np.ndarray, t0_us: int, t1_us: int, gap_s: float) -> float:
    """Share of ``[t0, t1]`` not lying inside a sample gap longer than ``gap_s``.

    The stretch before the first and after the last sample counts as a gap, so a
    stream that stops mid-phase (a mandible IMU that stopped recording) shows up as low coverage.
    """
    span = float(t1_us - t0_us)
    if span <= 0:
        return float("nan")
    t = np.unique(np.asarray(t_us, dtype=np.int64))
    t = t[(t >= t0_us) & (t <= t1_us)]
    edges = np.concatenate(([t0_us], t, [t1_us])).astype(np.float64)
    gaps = np.diff(edges)
    lost = gaps[gaps > gap_s * 1e6].sum()
    return float(1.0 - lost / span)


def fraction(mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool)
    return float(m.mean()) if m.size else float("nan")


def channel_mains_share(uv: np.ndarray, fs_hz: float, line_hz: float) -> np.ndarray:
    a = np.asarray(uv, dtype=np.float64)
    return np.array([mains_share(a[:, i], fs_hz, line_hz) for i in range(a.shape[1])])


def snr_db(ref_uv: float, rest_uv: float) -> float:
    if not (np.isfinite(ref_uv) and np.isfinite(rest_uv)) or ref_uv <= 0 or rest_uv <= 0:
        return float("nan")
    return float(20.0 * np.log10(ref_uv / rest_uv))
