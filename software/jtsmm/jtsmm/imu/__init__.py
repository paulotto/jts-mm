# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""IMU host-side helpers: unit conversion and derived kinematics.

This package converts the compact BNO085 wire records (tagged int16 motion
vectors and float32 quaternions/linear-accel, see s05 §1) into SI units and
derives relative jaw kinematics for the viewer. It is intentionally free of any
capture/transport dependency so it can be unit-tested against golden vectors.
"""

from __future__ import annotations

from .kinematics import (
    ZuptIntegrator,
    apply_mounting_quat,
    quat_conjugate,
    quat_from_rotation_matrix,
    quat_multiply,
    quat_rotate,
    relative_linear_accel,
    relative_quaternion,
)
from .units import (
    ACCEL_SCALE,
    GYRO_SCALE,
    MAG_SCALE,
    accel_to_si,
    gyro_to_si,
    mag_to_si,
    raw_to_si,
)
from .gravity_lin_accel import gravity_body, linear_accel_from_raw, rotate_vector_by_quat

__all__ = [
    "ACCEL_SCALE",
    "GYRO_SCALE",
    "MAG_SCALE",
    "accel_to_si",
    "gyro_to_si",
    "mag_to_si",
    "raw_to_si",
    "ZuptIntegrator",
    "apply_mounting_quat",
    "quat_from_rotation_matrix",
    "quat_conjugate",
    "quat_multiply",
    "quat_rotate",
    "relative_linear_accel",
    "relative_quaternion",
    "gravity_body",
    "linear_accel_from_raw",
    "rotate_vector_by_quat",
]
