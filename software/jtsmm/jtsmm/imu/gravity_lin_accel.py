# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Gravity removal from raw accelerometer using an orientation quaternion.

Used for offline jaw-kinematics when recording gyro + raw accel and fusing with
Madgwick (or similar) on the host instead of using the BNO085 linear-acceleration
report.
"""

from __future__ import annotations

import numpy as np

G = 9.80665  # m/s^2


def rotate_vector_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate 3-vector ``v`` by unit quaternion ``q`` = (w, x, y, z)."""
    v = np.asarray(v, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q
    # v' = q * (0,v) * q_conj
    t = 2.0 * np.cross([x, y, z], v)
    return v + w * t + np.cross([x, y, z], t)


def gravity_body(q: np.ndarray, g: float = G) -> np.ndarray:
    """What a still accelerometer reads (the gravity reaction) in the body frame.

    ``q`` = (w, x, y, z) is ``world←body``, the fusion convention used across
    this codebase (it rotates body vectors into the world frame). The reaction
    is +g along world Z, so in the body frame it is ``R(q)^T [0, 0, g]``, i.e. a
    rotation by ``conj(q)``. Until 2026-09 this rotated by ``q`` itself: right
    only at identity, and a 90° tilt left 2g (19.6 m/s²) of "linear
    acceleration" instead of zero.
    """
    q = np.asarray(q, dtype=np.float64)
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
    return rotate_vector_by_quat(np.array([0.0, 0.0, g], dtype=np.float64), q_conj)


def linear_accel_from_raw(accel: np.ndarray, q: np.ndarray, g: float = G) -> np.ndarray:
    """Linear acceleration = raw accel minus gravity rotated into the body frame."""
    accel = np.asarray(accel, dtype=np.float64)
    return accel - gravity_body(q, g=g)
