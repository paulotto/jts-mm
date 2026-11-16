# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Velocities and accelerations of the jaw, one implementation for every consumer.

The optical poses are low-passed at 6 Hz (an02 §1) and sampled at the trigger
rate (40 Hz in the pilot). Derivatives are Savitzky-Golay estimates (cubic, a
window of about 0.15 s, i.e. matched to that bandwidth) computed per finite
segment, so a tracking gap never becomes a spike. Two angular rates exist: the
3-D angular velocity ``omega`` from the relative rotations
``log(R_{k+1} R_{k-1}^T) / (t_{k+1} - t_{k-1})`` (spatial, maxillary frame), and
the opening-angle rate ``theta_vel`` (Savitzky-Golay of theta_y), which is what
the clinical peak opening/closing velocities use.

The kinematics stage stores the incisor and opening-angle derivatives per phase
(``<phase>__incisor_vel``/``_acc``, ``__theta_vel``/``_acc``, ``__omega``), so
the clinical peak velocities, the data views and the animator read the same
numbers.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from biomech import timebase

WINDOW_S = 0.15
POLYORDER = 3


def _window(fs: float, window_s: float, polyorder: int) -> int:
    n = max(int(round(window_s * fs)) | 1, polyorder + 2 + (polyorder % 2 == 0))
    return n if n % 2 == 1 else n + 1


def derivatives(t_us: np.ndarray, x: np.ndarray, *, window_s: float = WINDOW_S,
                polyorder: int = POLYORDER, max_gap_s: float | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """First and second time derivative (per second, per s^2) of ``x`` (N,) or (N, D).

    Per finite segment; segments shorter than the window fall back to
    ``numpy.gradient`` (first) and NaN (second derivative).
    """
    t = np.asarray(t_us, dtype=np.float64) * 1e-6
    a = np.asarray(x, dtype=np.float64)
    one = a.ndim == 1
    if one:
        a = a[:, None]
    vel = np.full(a.shape, np.nan)
    acc = np.full(a.shape, np.nan)
    if len(t) < 3:
        return (vel[:, 0], acc[:, 0]) if one else (vel, acc)
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt if dt > 0 else 1.0
    n = _window(fs, window_s, polyorder)
    ok = np.isfinite(a).all(axis=1)
    gap = max_gap_s if max_gap_s is not None else 2.5 * dt
    for s0, s1 in timebase.time_segments(t * 1e6, ok, gap):
        m = s1 - s0
        if m < 3:
            continue
        seg = a[s0:s1]
        if m < n:
            vel[s0:s1] = np.gradient(seg, t[s0:s1], axis=0)
            continue
        # mode="interp": the ends use the polynomial fitted to the first/last window.
        vel[s0:s1] = savgol_filter(seg, n, polyorder, deriv=1, delta=dt, axis=0, mode="interp")
        acc[s0:s1] = savgol_filter(seg, n, polyorder, deriv=2, delta=dt, axis=0, mode="interp")
    return (vel[:, 0], acc[:, 0]) if one else (vel, acc)


def angular_velocity(t_us: np.ndarray, R: np.ndarray, *, max_gap_s: float | None = None
                     ) -> np.ndarray:
    """(N, 3) spatial angular velocity (deg/s) of rotations R (N, 3, 3); NaN in gaps."""
    t = np.asarray(t_us, dtype=np.float64) * 1e-6
    R = np.asarray(R, dtype=np.float64)
    out = np.full((len(t), 3), np.nan)
    if len(t) < 3:
        return out
    ok = np.isfinite(R.reshape(len(R), -1)).all(axis=1)
    dt = float(np.median(np.diff(t)))
    gap = max_gap_s if max_gap_s is not None else 2.5 * dt
    for s0, s1 in timebase.time_segments(t * 1e6, ok, gap):
        if s1 - s0 < 3:
            continue
        Rs = R[s0:s1]
        ts = t[s0:s1]
        rv = Rotation.from_matrix(Rs[2:] @ np.swapaxes(Rs[:-2], 1, 2)).as_rotvec()
        w = rv / (ts[2:] - ts[:-2])[:, None]
        out[s0 + 1:s1 - 1] = np.degrees(w)
        out[s0] = out[s0 + 1]
        out[s1 - 1] = out[s1 - 2]
    return out


def speed(v: np.ndarray) -> np.ndarray:
    """Magnitude of an (N, D) vector series (NaN-propagating)."""
    return np.linalg.norm(np.asarray(v, float), axis=1)
