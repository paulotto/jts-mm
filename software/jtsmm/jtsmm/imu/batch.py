# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Vectorized quaternion helpers for whole IMU streams (numpy only).

Quaternions are (w, x, y, z), as stored in ``/imu_*/quat`` (BNO085 game rotation
vector, ``world←body``). The per-sample helpers in :mod:`jtsmm.imu.kinematics`
serve the live path; these operate on (N, 4) arrays for offline analysis and a
future controller's batch updates.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sanitize_stream",
    "quat_normalize",
    "quat_conj",
    "quat_mul",
    "quat_continuous",
    "quat_prepare",
    "quat_to_rotvec",
    "rotvec_to_quat",
    "quat_to_matrix",
    "matrix_to_quat",
    "quat_resample",
    "angular_velocity",
]


def sanitize_stream(t_us: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Sort by timestamp and drop repeated stamps (first occurrence kept).

    The BNO085 streams in the pilot carry 2–1158 non-monotonic stamps each
    (interrupt-edge stamping vs. report order); filters and derivatives assume
    strictly increasing time.
    """
    t = np.asarray(t_us, dtype=np.int64)
    order = np.argsort(t, kind="stable")
    ts = t[order]
    keep = np.concatenate(([True], np.diff(ts) > 0))
    idx = order[keep]
    return (t[idx],) + tuple(np.asarray(a)[idx] for a in arrays)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return q / n


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, broadcasting over leading dimensions."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_continuous(q: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive quaternions lie in the same hemisphere.

    Same result as the sequential rule "negate q[i] when dot(q[i], q_out[i-1]) < 0"
    (kept as :func:`_quat_continuous_loop`), vectorized: with d[i] = dot(q[i], q[i-1]) of
    the *input*, the sign of sample i is the parity of the negative d since the last d that
    is not < 0 and not > 0 (zero or NaN restart at +1). The sign of each d decides alone;
    where |d| is within rounding of zero, d is recomputed with ``np.dot`` exactly as the
    loop does, so the result is bit-identical.
    """
    q = np.asarray(q, dtype=np.float64).copy()
    if len(q) < 2:
        return q
    a, b = q[1:], q[:-1]
    prod = a * b
    d = prod.sum(axis=-1)
    # Any summation order of the products differs by at most a few ulp of sum|a_k b_k|:
    # outside this band the sign (and zero-ness) of d cannot depend on the order.
    near = ~(np.abs(d) > 1e-9 * np.abs(prod).sum(axis=-1)) & ~np.isnan(d)
    for i in np.flatnonzero(near):
        d[i] = np.dot(a[i], b[i])
    neg = d < 0
    restart = ~(neg | (d > 0))                     # zero or NaN: no flip, parity restarts
    cnt = np.cumsum(neg)
    # Negative count up to and including the last restart at or before each sample.
    last = np.maximum.accumulate(np.where(restart, np.arange(1, len(d) + 1), 0))
    base = np.where(last > 0, cnt[np.maximum(last - 1, 0)], 0)
    flip = np.flatnonzero((cnt - base) % 2) + 1
    q[flip] = -q[flip]
    return q


def _quat_continuous_loop(q: np.ndarray) -> np.ndarray:
    """Reference implementation of :func:`quat_continuous` (tests)."""
    q = np.asarray(q, dtype=np.float64).copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Rotation vector (rad), shortest rotation (w >= 0 hemisphere)."""
    q = quat_normalize(q)
    q = np.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    ang = 2.0 * np.arctan2(s, q[..., :1])
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(s > 1e-12, ang / s, 2.0)
    return v * scale


def rotvec_to_quat(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    ang = np.linalg.norm(r, axis=-1, keepdims=True)
    half = 0.5 * ang
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(ang > 1e-12, np.sin(half) / ang, 0.5)
    return np.concatenate([np.cos(half), r * k], axis=-1)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(quat_normalize(q), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) to unit quaternions (w >= 0)."""
    R = np.asarray(R, dtype=np.float64)
    m = R.reshape(-1, 3, 3)
    out = np.empty((m.shape[0], 4))
    tr = np.trace(m, axis1=1, axis2=2)
    for i, (Ri, t) in enumerate(zip(m, tr)):
        if t > 0:
            s = 2.0 * np.sqrt(t + 1.0)
            out[i] = [0.25 * s, (Ri[2, 1] - Ri[1, 2]) / s, (Ri[0, 2] - Ri[2, 0]) / s,
                      (Ri[1, 0] - Ri[0, 1]) / s]
        else:
            k = int(np.argmax(np.diag(Ri)))
            j, m = (k + 1) % 3, (k + 2) % 3
            s = 2.0 * np.sqrt(max(1.0 + Ri[k, k] - Ri[j, j] - Ri[m, m], 1e-12))
            v = np.empty(3)
            v[k] = 0.25 * s
            v[j] = (Ri[j, k] + Ri[k, j]) / s
            v[m] = (Ri[m, k] + Ri[k, m]) / s
            out[i] = [(Ri[m, j] - Ri[j, m]) / s, *v]
    out = np.where(out[:, :1] < 0, -out, out)
    return out.reshape(R.shape[:-2] + (4,))


def quat_prepare(q: np.ndarray) -> np.ndarray:
    """Normalized, sign-continuous stream: what :func:`quat_resample` interpolates. Compute
    it once per stream and pass it with ``prepared=True`` to resample repeatedly."""
    return quat_continuous(quat_normalize(q))


def quat_resample(t_src_us: np.ndarray, q_src: np.ndarray, t_dst_us: np.ndarray,
                  max_gap_us: int = 50_000, *, prepared: bool = False) -> np.ndarray:
    """SLERP onto ``t_dst``; NaN where the bracketing samples are > max_gap apart.

    ``t_src`` must be strictly increasing (see :func:`sanitize_stream`). ``prepared``:
    ``q_src`` is the output of :func:`quat_prepare` (same result, not recomputed).
    """
    t = np.asarray(t_src_us, dtype=np.int64)
    td = np.asarray(t_dst_us, dtype=np.int64)
    q = np.asarray(q_src, dtype=np.float64) if prepared else quat_prepare(q_src)
    out = np.full((td.size, 4), np.nan)
    if t.size < 2:
        return out
    j = np.searchsorted(t, td, side="right") - 1
    ok = (j >= 0) & (j < t.size - 1)
    j = np.clip(j, 0, t.size - 2)
    gap = t[j + 1] - t[j]
    ok &= gap <= max_gap_us
    exact = td == t[np.clip(np.searchsorted(t, td), 0, t.size - 1)]
    frac = np.clip((td - t[j]) / np.maximum(gap, 1), 0.0, 1.0)
    q0, q1 = q[j], q[j + 1]
    rel = quat_mul(quat_conj(q0), q1)
    step = quat_mul(q0, rotvec_to_quat(quat_to_rotvec(rel) * frac[:, None]))
    out[ok] = step[ok]
    hit = exact & ~ok
    if np.any(hit):
        k = np.clip(np.searchsorted(t, td[hit]), 0, t.size - 1)
        out[hit] = q[k]
    return out


def angular_velocity(t_us: np.ndarray, q: np.ndarray, *, frame: str = "body") -> np.ndarray:
    """Angular velocity (rad/s) by central differences of the orientation.

    ``frame="body"`` gives ω in the rotating frame (rotvec(conj(q[i-1]) q[i+1]) / dt);
    ``"world"`` in the reference frame. End samples repeat their neighbour; NaN
    quaternions propagate. There is no raw gyroscope in the pilot recordings,
    so this is the only angular-rate source; its bandwidth is that of the
    BNO085 fusion filter.
    """
    t = np.asarray(t_us, dtype=np.float64) * 1e-6
    q = quat_continuous(quat_normalize(q))
    out = np.full((len(q), 3), np.nan)
    if len(q) < 3:
        return out
    a, b = q[:-2], q[2:]
    d = quat_mul(quat_conj(a), b) if frame == "body" else quat_mul(b, quat_conj(a))
    dt = (t[2:] - t[:-2])[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        out[1:-1] = quat_to_rotvec(d) / dt
    out[0], out[-1] = out[1], out[-2]
    return out
