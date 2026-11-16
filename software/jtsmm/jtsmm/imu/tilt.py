# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Gravity-referenced tilt from raw accelerometers, and a complementary filter.

The pilot recorded the raw BNO085 accelerometer *including gravity*. When the
sensor is quasi-static, that vector is the gravity reaction in the body frame:
a drift-free tilt reference that does not depend on the fusion filter's
heading (the game rotation vector has no magnetometer, so the two IMUs do not
share a heading reference). Used for the jaw-opening angle (an04 §4):

* :func:`signed_angle_about` gives the rotation of a body about a known axis
  between a reference posture and now, from gravity seen in that body;
* :func:`complementary` fuses a fast, possibly drifting angle (from the
  quaternions) with this slow, drift-free one, trusting the latter only while
  the sensor is quasi-static.
"""

from __future__ import annotations

import numpy as np

G = 9.80665

__all__ = ["G", "unit", "quasi_static", "signed_angle_about", "complementary"]


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


def quasi_static(accel: np.ndarray, tol_g: float = 0.05) -> np.ndarray:
    """True where ``| |a| - g |`` is within ``tol_g`` (fraction of g)."""
    n = np.linalg.norm(np.asarray(accel, dtype=np.float64), axis=-1)
    return np.abs(n - G) <= tol_g * G


def signed_angle_about(v_ref: np.ndarray, v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Signed angle (rad) that turns ``v_ref`` into ``v`` about ``axis`` (right-hand rule).

    Both vectors are projected onto the plane normal to ``axis``. For gravity
    seen in a body, a body rotation by +θ about ``axis`` turns the observed
    gravity by -θ; callers negate accordingly.
    """
    ax = unit(axis)
    b = np.asarray(v, dtype=np.float64)
    a = np.broadcast_to(np.asarray(v_ref, dtype=np.float64), b.shape)
    a = a - (a @ ax)[..., None] * ax
    b = b - (b @ ax)[..., None] * ax
    return np.arctan2(np.cross(a, b) @ ax, np.sum(a * b, axis=-1))


def complementary(t_s: np.ndarray, fast: np.ndarray, slow: np.ndarray,
                  slow_valid: np.ndarray, tau_s: float = 1.0) -> np.ndarray:
    """First-order complementary filter: increments of ``fast``, level of ``slow``.

    ``theta[k] = a (theta[k-1] + d fast[k]) + (1 - a) slow[k]`` with
    ``a = tau / (tau + dt)`` while ``slow_valid``; otherwise only the increment is
    integrated. Causal; NaN in ``fast`` holds the previous value.
    """
    t = np.asarray(t_s, dtype=np.float64)
    f = np.asarray(fast, dtype=np.float64)
    s = np.asarray(slow, dtype=np.float64)
    ok = np.asarray(slow_valid, dtype=bool) & np.isfinite(s)
    out = np.full(f.shape, np.nan)
    start = np.flatnonzero(np.isfinite(f) & ok)
    if start.size == 0:
        return out
    k0 = int(start[0])
    out[k0] = s[k0]
    for k in range(k0 + 1, len(f)):
        df = f[k] - f[k - 1] if np.isfinite(f[k]) and np.isfinite(f[k - 1]) else 0.0
        pred = out[k - 1] + df
        if ok[k]:
            a = tau_s / (tau_s + max(t[k] - t[k - 1], 1e-6))
            out[k] = a * pred + (1.0 - a) * s[k]
        else:
            out[k] = pred
    return out
