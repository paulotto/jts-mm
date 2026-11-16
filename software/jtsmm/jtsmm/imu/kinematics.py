# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Derived jaw kinematics from the two IMUs (viewer estimate, not ground truth).

The camera system provides geometric 6DOF ground truth; the IMUs give
orientation directly (fusion quaternion) but **not** absolute position. This
module derives:

1. **Relative orientation** ``q_rel = conj(q_cra) ⊗ q_mta`` (mandible relative
   to cranium), Hamilton product, w-first quaternions.
2. **Relative linear acceleration** by rotating each slot's linear acceleration
   into the cranium frame and subtracting.
3. **Short-term relative translation** by double-integrating ``a_rel`` with a
   Zero-Velocity Update (ZUPT) that resets velocity when the body is nearly
   still, plus a hard periodic reset. This drifts and is only a *visual
   estimate*; never treat it as millimetre-accurate displacement.

Everything here is a pure function or a small stateful integrator so it can be
unit-tested and shared by the hub and (mirrored minimally) the browser.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "quat_conjugate",
    "quat_multiply",
    "quat_rotate",
    "quat_slerp",
    "quat_at_time",
    "quat_from_rotation_matrix",
    "apply_mounting_quat",
    "relative_quaternion",
    "relative_linear_accel",
    "ZuptIntegrator",
]

Quat = tuple[float, float, float, float]
Vec3 = tuple[float, float, float]


def quat_conjugate(q: Quat) -> Quat:
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_multiply(a: Quat, b: Quat) -> Quat:
    """Hamilton product a ⊗ b (w-first)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _normalize(q: Quat) -> Quat:
    n = np.sqrt(sum(c * c for c in q))
    if n == 0:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(c / n for c in q)  # type: ignore[return-value]


def quat_rotate(q: Quat, v: Vec3) -> Vec3:
    """Rotate vector v by quaternion q (active rotation, w-first)."""
    qw, qx, qy, qz = q
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    rx = vx + qw * tx + (qy * tz - qz * ty)
    ry = vy + qw * ty + (qz * tx - qx * tz)
    rz = vz + qw * tz + (qx * ty - qy * tx)
    return (rx, ry, rz)


def quat_slerp(q0: Quat, q1: Quat, t: float) -> Quat:
    """Spherical linear interpolation; ``t`` in [0, 1]."""
    q0 = _normalize(q0)
    q1 = _normalize(q1)
    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0.0:
        q1 = tuple(-c for c in q1)
        dot = -dot
    if dot > 0.9995:
        out = tuple((1.0 - t) * a + t * b for a, b in zip(q0, q1))
        return _normalize(out)  # type: ignore[return-value]
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    if sin_theta < 1e-9:
        return q0
    w0 = np.sin((1.0 - t) * theta) / sin_theta
    w1 = np.sin(t * theta) / sin_theta
    return _normalize(tuple(w0 * a + w1 * b for a, b in zip(q0, q1)))  # type: ignore[return-value]


def quat_at_time(samples: list[tuple[int, Quat]], t_us: int) -> Quat | None:
    """Interpolate orientation at ``t_us`` from timestamped quaternion history."""
    if not samples:
        return None
    if t_us <= samples[0][0]:
        return samples[0][1]
    if t_us >= samples[-1][0]:
        return samples[-1][1]
    for i in range(len(samples) - 1):
        t0, q0 = samples[i]
        t1, q1 = samples[i + 1]
        if t0 <= t_us <= t1:
            if t1 == t0:
                return q0
            u = (t_us - t0) / (t1 - t0)
            return quat_slerp(q0, q1, u)
    return samples[-1][1]


def quat_from_rotation_matrix(R: np.ndarray) -> Quat:
    """Unit quaternion (w-first) for a proper rotation matrix."""
    m = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = float(np.trace(m))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return _normalize((float(w), float(x), float(y), float(z)))


def apply_mounting_quat(q_chip: Quat, R_mount: np.ndarray) -> Quat:
    """Orientation of the array body from the chip's fusion quaternion.

    ``q_chip`` is ``world←chip`` (the fusion output rotates chip vectors into
    the IMU world frame) and ``R_mount`` is ``body←chip``, so
    ``world←body = (world←chip)(chip←body)``:

        q_body = q_chip ⊗ conj(q(R_mount))

    Until 2026-09 this returned ``q(R_mount) ⊗ q_chip``, which is not a frame
    composition at all: it only matched for R_mount = identity.
    """
    q_m = quat_from_rotation_matrix(R_mount)
    return _normalize(quat_multiply(_normalize(q_chip), quat_conjugate(q_m)))


def relative_quaternion(q_mta: Quat, q_cra: Quat) -> Quat:
    """q_rel = conj(q_cra) ⊗ q_mta — mandible orientation in the cranium frame."""
    return _normalize(quat_multiply(quat_conjugate(_normalize(q_cra)), _normalize(q_mta)))


def relative_linear_accel(
    a_mta: Vec3, q_mta: Quat, a_cra: Vec3, q_cra: Quat
) -> Vec3:
    """Relative linear acceleration in the (IMU) world frame.

    Each slot reports linear acceleration in its CHIP frame; rotate both to the
    world frame with their raw ``world←chip`` fusion quaternions (NOT the
    mounting-corrected body quaternions: ``a_world = R_wc a_chip``) and
    subtract. Callers must pass quaternions **interpolated to the linear-accel
    timestamp** when the streams are not sample-aligned (typical for
    kinematics_default). The two game rotation vectors do not share a heading,
    so this difference is only meaningful up to that yaw offset.
    """
    aw_mta = quat_rotate(q_mta, a_mta)
    aw_cra = quat_rotate(q_cra, a_cra)
    return (aw_mta[0] - aw_cra[0], aw_mta[1] - aw_cra[1], aw_mta[2] - aw_cra[2])


@dataclass
class ZuptIntegrator:
    """Trapezoidal double-integration of acceleration with ZUPT + periodic reset.

    This produces a **drift-prone estimate** of relative displacement for the
    viewer only. Velocity is forced to zero when the acceleration magnitude
    stays below ``zupt_threshold`` for ``zupt_hold_s`` (limb briefly still), and
    both velocity and position hard-reset every ``auto_reset_s`` (and on
    :meth:`reset`) so error cannot accumulate without bound.
    """

    zupt_threshold: float = 0.3       # m/s^2; |a| below this counts as "still"
    zupt_hold_s: float = 0.1          # sustained still time before zeroing velocity
    auto_reset_s: float = 2.0         # periodic hard reset of the estimate
    _v: np.ndarray = None             # type: ignore[assignment]
    _p: np.ndarray = None             # type: ignore[assignment]
    _last_t: float | None = None
    _still_s: float = 0.0
    _since_reset_s: float = 0.0
    did_reset: bool = False

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._v = np.zeros(3, dtype=np.float64)
        self._p = np.zeros(3, dtype=np.float64)
        self._last_t = None
        self._still_s = 0.0
        self._since_reset_s = 0.0
        self.did_reset = True

    def update(self, t_s: float, accel: Vec3) -> tuple[Vec3, bool]:
        """Advance with a new acceleration sample; return (position, valid)."""
        a = np.asarray(accel, dtype=np.float64)
        self.did_reset = False
        if self._last_t is None:
            self._last_t = t_s
            return (0.0, 0.0, 0.0), True
        dt = t_s - self._last_t
        self._last_t = t_s
        if dt <= 0 or dt > 0.5:
            # Non-monotonic or a large gap: do not integrate across it.
            return tuple(self._p), True  # type: ignore[return-value]

        self._since_reset_s += dt
        if self._since_reset_s >= self.auto_reset_s:
            self.reset()
            return (0.0, 0.0, 0.0), True

        mag = float(np.linalg.norm(a))
        if mag < self.zupt_threshold:
            self._still_s += dt
            if self._still_s >= self.zupt_hold_s:
                self._v[:] = 0.0  # ZUPT: assume truly stationary
        else:
            self._still_s = 0.0
            self._v += a * dt
            self._p += self._v * dt
        return tuple(self._p), True  # type: ignore[return-value]

    @property
    def position(self) -> Vec3:
        return tuple(self._p)  # type: ignore[return-value]
