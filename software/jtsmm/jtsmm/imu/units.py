# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""BNO085 raw int16 -> SI unit conversion (SH-2 fixed-point Q-points).

The tagged RAW wire record carries a motion vector as three int16 counts in the
sensor's SH-2 Q-point fixed-point format (records.h / bno085.c). This module
turns those counts into SI units, matching the datasheet
(BNO085 datasheet, CEVA):

============  ==========================  ======  =========  ===============
Report        Datasheet name              SH-2 id  Q-point    SI units
============  ==========================  ======  =========  ===============
accelerometer Acceleration Vector          0x01    Q8         m/s^2
gyroscope     Angular Velocity Vector      0x02    Q9         rad/s
magnetometer  Magnetic Field Strength      0x03    Q4         microtesla (uT)
============  ==========================  ======  =========  ===============

A value in Q(n) fixed point equals ``raw / 2**n``. Linear acceleration (0x04)
and the fusion quaternions are already converted to float on the wire by the
firmware, so they are not handled here.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

__all__ = [
    "ACCEL_Q",
    "GYRO_Q",
    "MAG_Q",
    "ACCEL_SCALE",
    "GYRO_SCALE",
    "MAG_SCALE",
    "accel_to_si",
    "gyro_to_si",
    "mag_to_si",
    "raw_to_si",
    "SENSOR_SCALE",
    "SENSOR_UNITS",
]

# SH-2 Q-points for the raw motion vectors (SH-2 reference manual §6.5).
ACCEL_Q = 8
GYRO_Q = 9
MAG_Q = 4

ACCEL_SCALE = 1.0 / (1 << ACCEL_Q)  # counts -> m/s^2
GYRO_SCALE = 1.0 / (1 << GYRO_Q)    # counts -> rad/s
MAG_SCALE = 1.0 / (1 << MAG_Q)      # counts -> uT

# Keyed by RAW sensor_id (records.IMU_SENSOR_*).
SENSOR_SCALE: dict[int, float] = {0x01: ACCEL_SCALE, 0x02: GYRO_SCALE, 0x03: MAG_SCALE}
SENSOR_UNITS: dict[int, str] = {0x01: "m/s^2", 0x02: "rad/s", 0x03: "uT"}

Vec3 = Tuple[float, float, float]


def _scale3(xyz: Sequence[float], scale: float) -> Vec3:
    return (float(xyz[0]) * scale, float(xyz[1]) * scale, float(xyz[2]) * scale)


def accel_to_si(counts: Sequence[float]) -> Vec3:
    """Accelerometer counts (Q8) -> m/s^2."""
    return _scale3(counts, ACCEL_SCALE)


def gyro_to_si(counts: Sequence[float]) -> Vec3:
    """Gyroscope counts (Q9) -> rad/s."""
    return _scale3(counts, GYRO_SCALE)


def mag_to_si(counts: Sequence[float]) -> Vec3:
    """Magnetometer counts (Q4) -> microtesla."""
    return _scale3(counts, MAG_SCALE)


def raw_to_si(sensor_id: int, counts: Sequence[float]) -> Vec3:
    """Convert a tagged RAW record's counts to SI using its sensor_id."""
    try:
        scale = SENSOR_SCALE[sensor_id]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unknown RAW sensor_id 0x{sensor_id:02X}") from exc
    return _scale3(counts, scale)


def raw_array_to_si(sensor_id: int, counts: Iterable[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Vectorized (N,3) counts -> SI for the writer/reader fast paths."""
    arr = np.asarray(counts, dtype=np.float64)
    return (arr * SENSOR_SCALE[sensor_id]).astype(np.float32)
